//! Text-only Qwen2.5-VL conditioning, including the final norm and template cut.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{embedding, rms_norm, Embedding, Module, RmsNorm, VarBuilder};
use tokenizers::Tokenizer;

use crate::layered_files::{LayeredLinear, TensorFile};
use crate::Result;

/// The language backbone; the vision tower is unnecessary for Layered's text prompts.
#[derive(Clone, Debug)]
pub struct Qwen25Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl Default for Qwen25Config {
    fn default() -> Self {
        Self {
            hidden_size: 3584,
            intermediate_size: 18944,
            num_layers: 28,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 152064,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

struct Block {
    q: LayeredLinear,
    k: LayeredLinear,
    v: LayeredLinear,
    o: LayeredLinear,
    gate: LayeredLinear,
    up: LayeredLinear,
    down: LayeredLinear,
    input_norm: RmsNorm,
    post_norm: RmsNorm,
}

impl Block {
    fn load(c: &Qwen25Config, vb: VarBuilder) -> Result<Self> {
        let linear = |i, o, bias, name| LayeredLinear::load(i, o, bias, true, None, vb.pp(name));
        let h = c.hidden_size;
        Ok(Self {
            q: linear(h, c.num_heads * c.head_dim, true, "self_attn.q_proj")?,
            k: linear(h, c.num_kv_heads * c.head_dim, true, "self_attn.k_proj")?,
            v: linear(h, c.num_kv_heads * c.head_dim, true, "self_attn.v_proj")?,
            o: linear(c.num_heads * c.head_dim, h, false, "self_attn.o_proj")?,
            gate: linear(h, c.intermediate_size, false, "mlp.gate_proj")?,
            up: linear(h, c.intermediate_size, false, "mlp.up_proj")?,
            down: linear(c.intermediate_size, h, false, "mlp.down_proj")?,
            input_norm: rms_norm(h, c.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_norm: rms_norm(h, c.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Qwen25Config, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let normalized = self.input_norm.forward(x)?;
        let q = self.q.forward(&normalized)?.reshape((b, s, c.num_heads, c.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = self.k.forward(&normalized)?.reshape((b, s, c.num_kv_heads, c.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(&normalized)?.reshape((b, s, c.num_kv_heads, c.head_dim))?.transpose(1, 2)?.contiguous()?;
        // Text positions are identical on all mRoPE axes: use split-half 1D RoPE.
        let q = candle_nn::rotary_emb::rope(&q.to_dtype(DType::F32)?, cos, sin)?.to_dtype(x.dtype())?;
        let k = candle_nn::rotary_emb::rope(&k.to_dtype(DType::F32)?, cos, sin)?.to_dtype(x.dtype())?;
        let groups = c.num_heads / c.num_kv_heads;
        let k = repeat_kv(&k, groups)?.to_dtype(DType::F32)?;
        let v = repeat_kv(&v, groups)?;
        let scores = (q.to_dtype(DType::F32)?.matmul(&k.transpose(2, 3)?)? / (c.head_dim as f64).sqrt())?.broadcast_add(mask)?;
        // CPU SDPA accumulates both products in f32, even for bf16 inputs.
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = probs.matmul(&v.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?.transpose(1, 2)?.reshape((b, s, c.num_heads * c.head_dim))?;
        let x = (x + self.o.forward(&attended)?)?;
        let norm = self.post_norm.forward(&x)?;
        let gate = candle_nn::ops::silu(&self.gate.forward(&norm)?)?;
        Ok((&x + self.down.forward(&(gate * self.up.forward(&norm)?)?)?)?)
    }
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?.expand((b, h, groups, s, d))?.reshape((b, h * groups, s, d))?.contiguous()?)
}

enum Blocks {
    Resident(Vec<Block>),
    Stream(TensorFile),
}

/// Zero-based block taps are before the final RMSNorm; conditioning is after it.
pub struct Qwen25Output {
    pub conditioning: Tensor,
    pub taps: Vec<(usize, Tensor)>,
}

pub struct Qwen25TextEncoder {
    config: Qwen25Config,
    embedding: Embedding,
    norm: RmsNorm,
    blocks: Blocks,
    device: Device,
    dtype: DType,
}

impl Qwen25TextEncoder {
    /// Load a language-model-root builder (`model.*` in the ComfyUI checkpoint).
    pub fn load(c: Qwen25Config, vb: VarBuilder) -> Result<Self> {
        Self::validate_config(&c)?;
        let blocks = (0..c.num_layers).map(|i| Block::load(&c, vb.pp(format!("layers.{i}")))).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding: embedding(c.vocab_size, c.hidden_size, vb.pp("embed_tokens"))?,
            norm: rms_norm(c.hidden_size, c.rms_norm_eps, vb.pp("norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            config: c,
            blocks: Blocks::Resident(blocks),
        })
    }

    /// Stream blocks for CPU parity; the embedding stays bf16 until its rows are gathered.
    pub fn from_file(path: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let config = Qwen25Config::default();
        let file = TensorFile::open(path)?;
        let embedding = Embedding::new(file.tensor("model.embed_tokens.weight", device)?, config.hidden_size);
        let norm = RmsNorm::new(file.tensor("model.norm.weight", device)?.to_dtype(dtype)?, config.rms_norm_eps);
        Ok(Self { config, embedding, norm, blocks: Blocks::Stream(file), device: device.clone(), dtype })
    }

    fn validate_config(c: &Qwen25Config) -> Result<()> {
        anyhow::ensure!(
            c.num_kv_heads > 0 && c.num_heads.is_multiple_of(c.num_kv_heads) && c.head_dim > 0 && c.head_dim.is_multiple_of(2),
            "invalid Qwen2.5 attention dimensions"
        );
        anyhow::ensure!(c.hidden_size == c.num_heads * c.head_dim && c.rope_theta.is_finite() && c.rope_theta > 0.0, "invalid Qwen2.5 configuration");
        Ok(())
    }

    /// Run IDs that already contain the chat template; return `[1, T, 3584]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self.forward_taps(ids, &[])?.conditioning)
    }

    pub fn forward_taps(&self, ids: &[u32], taps: &[usize]) -> Result<Qwen25Output> {
        anyhow::ensure!(!ids.is_empty() && ids.iter().all(|i| (*i as usize) < self.config.vocab_size), "empty prompt or token outside vocabulary");
        anyhow::ensure!(taps.iter().all(|i| *i < self.config.num_layers), "tap outside decoder");
        let s = ids.len();
        let mut x = self.embedding.forward(&Tensor::from_slice(ids, (1, s), &self.device)?)?.to_dtype(self.dtype)?;
        let pairs = self.config.head_dim / 2;
        let angles: Vec<f32> =
            (0..s).flat_map(|p| (0..pairs).map(move |j| p as f32 / self.config.rope_theta.powf((2 * j) as f32 / self.config.head_dim as f32))).collect();
        let cos = Tensor::from_vec(angles.iter().map(|a| a.cos()).collect::<Vec<_>>(), (s, pairs), &self.device)?;
        let sin = Tensor::from_vec(angles.iter().map(|a| a.sin()).collect::<Vec<_>>(), (s, pairs), &self.device)?;
        let mask: Vec<f32> = (0..s).flat_map(|i| (0..s).map(move |j| if j > i { f32::MIN / 4.0 } else { 0.0 })).collect();
        let mask = Tensor::from_vec(mask, (s, s), &self.device)?;
        let mut states = Vec::new();
        for i in 0..self.config.num_layers {
            x = match &self.blocks {
                Blocks::Resident(blocks) => {
                    blocks.get(i).ok_or_else(|| anyhow::anyhow!("missing decoder block {i}"))?.forward(&x, &self.config, &cos, &sin, &mask)?
                }
                Blocks::Stream(file) => Block::load(&self.config, file.builder(&format!("model.layers.{i}."), self.dtype, &self.device)?)?.forward(
                    &x,
                    &self.config,
                    &cos,
                    &sin,
                    &mask,
                )?,
            };
            if taps.contains(&i) {
                states.push((i, x.clone()));
            }
        }
        let x = self.norm.forward(&x)?;
        let cut = prefix_cut(ids);
        Ok(Qwen25Output { conditioning: x.narrow(1, cut, s - cut)?.contiguous()?, taps: states })
    }
}

pub const SYSTEM_PROMPT: &str =
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";

/// ComfyUI preserves empty text and accepts an already-templated conversation.
pub fn prompt_template(prompt: &str) -> String {
    if prompt.starts_with("<|im_start|>") || prompt.starts_with("<|start_header_id|>") {
        prompt.to_owned()
    } else {
        format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    }
}

/// Drop through the second im_start and its user/newline only when ComfyUI does.
pub fn prefix_cut(ids: &[u32]) -> usize {
    let mut cut = ids.iter().enumerate().filter(|(_, id)| **id == 151644).take(2).last().map(|(i, _)| i);
    if let Some(i) = cut {
        if ids.len() > i + 3 && ids.get(i + 1) == Some(&872) && ids.get(i + 2) == Some(&198) {
            cut = Some(i + 3);
        }
    }
    cut.unwrap_or_else(|| ids.len().saturating_sub(1))
}

/// Load ComfyUI's vocab/merges folder (assembled by [`crate::comfy_files::qwen_tokenizer`]) or
/// Qwen3-VL's compatible tokenizer.json; IDs agree.
pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let path = path.as_ref();
    if path.is_file() {
        return Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"));
    }
    crate::comfy_files::qwen_tokenizer(path)
}

pub fn tokenize(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    Ok(tokenizer.encode(prompt_template(prompt), false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?.get_ids().to_vec())
}
