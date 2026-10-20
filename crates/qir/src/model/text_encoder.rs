//! `Qwen3VLForConditionalGeneration` text encoder (text-only path), ported to
//! candle. **Phase 3.** For text-to-image the vision tower and DeepStack are
//! unused, so the backbone is a standard Qwen3 decoder: 36 layers, GQA (32 Q /
//! 8 KV heads, head_dim 128), per-head q/k RMSNorm, SwiGLU MLP, RMSNorm, RoPE
//! (theta 5e6). mRoPE collapses to 1D RoPE when every position id is the token
//! index (text-only), so plain RoPE is exact here.
//!
//! The pipeline extracts `hidden_states[-1]` — the last decoder layer's output
//! **before** the final RMSNorm (it hooks the norm to return its input) — for
//! all positions, then drops the leading system-prompt tokens. So this module
//! runs the layers and returns the pre-final-norm hidden states; it does NOT
//! apply the final norm or the lm_head.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::{embedding, linear_b, rms_norm, Activation, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::model::vision::VisionFeatures;
use crate::Result;

/// A no-bias linear, full-precision or Q8_0-quantized (GGUF, weight-only). Q8_0
/// halves the weight VRAM near-losslessly so the 8B text encoder fits alongside
/// the DiT and VAE, resident in 24 GB.
enum QLinear {
    Full(Linear),
    Quant(QMatMul),
}

impl QLinear {
    fn load(in_c: usize, out_c: usize, quant: bool, vb: VarBuilder) -> Result<Self> {
        if quant {
            let w = vb.get((out_c, in_c), "weight")?;
            let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;
            Ok(QLinear::Quant(QMatMul::from_qtensor(qt)?))
        } else {
            Ok(QLinear::Full(linear_b(in_c, out_c, false, vb)?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QLinear::Full(l) => Ok(l.forward(x)?),
            QLinear::Quant(q) => Ok(q.forward(x)?),
        }
    }
}

/// Qwen3-VL text config (from `text_encoder/config.json` `text_config`).
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            max_position_embeddings: 8192,
        }
    }
}

/// Qwen3-VL's interleaved multimodal RoPE: 64 frequency pairs per head; pair `j` turns by the
/// temporal position, except pairs `j ≡ 1 (mod 3)` by the height position and `j ≡ 2 (mod 3)`
/// by the width position for `j < 60` (`mrope_section [24, 20, 20]`). Text has all three
/// positions equal, so a prompt without pictures is plain 1D RoPE.
pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];

struct RotaryEmbedding {
    /// `(64,)` f32.
    inv_freq: Vec<f32>,
    device: Device,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, dev: &Device, _dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv_freq: Vec<f32> = (0..dim).step_by(2).map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32)).collect();
        Ok(Self { inv_freq, device: dev.clone() })
    }

    /// `(cos, sin)` of `(S, 64)` f32 for 3-axis positions `(3, S)` with the interleaved
    /// sections (`precompute_freqs_cis` with `interleaved_mrope`).
    fn tables(&self, positions: &[Vec<f32>; 3]) -> Result<(Tensor, Tensor)> {
        let s = positions[0].len();
        let n = self.inv_freq.len();
        let mut cos = vec![0f32; s * n];
        let mut sin = vec![0f32; s * n];
        for (j, f) in self.inv_freq.iter().enumerate() {
            let axis = if j < MROPE_SECTION[1] * 3 && j % 3 == 1 {
                1
            } else if j < MROPE_SECTION[2] * 3 && j % 3 == 2 {
                2
            } else {
                0
            };
            for i in 0..s {
                let a = positions[axis][i] * f;
                cos[i * n + j] = a.cos();
                sin[i * n + j] = a.sin();
            }
        }
        Ok((Tensor::from_vec(cos, (s, n), &self.device)?, Tensor::from_vec(sin, (s, n), &self.device)?))
    }

    /// Plain positions `0..S` on every axis.
    fn text_tables(&self, seq: usize) -> Result<(Tensor, Tensor)> {
        let p: Vec<f32> = (0..seq).map(|i| i as f32).collect();
        self.tables(&[p.clone(), p.clone(), p])
    }

    /// Apply RoPE to `(B, H, S, D)` q and k with f32 tables (the rotation runs in f32, as
    /// ComfyUI's kernel does; q and k keep their dtype).
    fn apply(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
        let dt = q.dtype();
        let q = candle_nn::rotary_emb::rope(&q.to_dtype(DType::F32)?.contiguous()?, cos, sin)?.to_dtype(dt)?;
        let k = candle_nn::rotary_emb::rope(&k.to_dtype(DType::F32)?.contiguous()?, cos, sin)?.to_dtype(dt)?;
        Ok((q, k))
    }
}

fn repeat_kv(x: Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x);
    }
    let (b, kvh, s, d) = x.dims4()?;
    let x = x.unsqueeze(2)?.broadcast_as((b, kvh, groups, s, d))?.reshape((b, kvh * groups, s, d))?;
    Ok(x)
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    groups: usize,
    scale: f64,
}

impl Attention {
    fn load(cfg: &TextConfig, quant: bool, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        Ok(Self {
            q_proj: QLinear::load(h, nh * hd, quant, vb.pp("q_proj"))?,
            k_proj: QLinear::load(h, nkv * hd, quant, vb.pp("k_proj"))?,
            v_proj: QLinear::load(h, nkv * hd, quant, vb.pp("v_proj"))?,
            o_proj: QLinear::load(nh * hd, h, quant, vb.pp("o_proj"))?,
            q_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            groups: nh / nkv,
            scale: 1.0 / (hd as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self.k_proj.forward(xs)?.reshape((b, s, self.num_kv_heads, self.head_dim))?;
        let v = self.v_proj.forward(xs)?.reshape((b, s, self.num_kv_heads, self.head_dim))?;
        // per-head RMSNorm on q,k (applied over head_dim), then to (B,H,S,D)
        let q = q.apply(&self.q_norm)?.transpose(1, 2)?;
        let k = k.apply(&self.k_norm)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let (q, k) = RotaryEmbedding::apply(&q, &k, &rope.0, &rope.1)?;
        let k = repeat_kv(k, self.groups)?.contiguous()?;
        let v = repeat_kv(v, self.groups)?.contiguous()?;
        // Scores and softmax in f32, as torch's SDPA keeps them: bf16 scores lose ~0.3 of a
        // logit at the magnitudes a prompt reaches, which showed as a 7 % embedding error
        // against ComfyUI (genai-native dev log 2026-10-20). A prompt is a few dozen tokens,
        // so the f32 pass costs nothing.
        let q32 = q.contiguous()?.to_dtype(DType::F32)?;
        let k32 = k.to_dtype(DType::F32)?;
        let attn = (q32.matmul(&k32.transpose(2, 3)?)? * self.scale)?;
        let attn = attn.broadcast_add(&mask.to_dtype(DType::F32)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.to_dtype(v.dtype())?;
        let out = attn.matmul(&v)?; // (B,H,S,D)
        let out = out.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn load(cfg: &TextConfig, quant: bool, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: QLinear::load(h, i, quant, vb.pp("gate_proj"))?,
            up: QLinear::load(h, i, quant, vb.pp("up_proj"))?,
            down: QLinear::load(i, h, quant, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?.apply(&Activation::Silu)?;
        let rhs = self.up.forward(xs)?;
        self.down.forward(&(lhs * rhs)?)
    }
}

struct DecoderLayer {
    attn: Attention,
    mlp: Mlp,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
}

impl DecoderLayer {
    fn load(cfg: &TextConfig, quant: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(cfg, quant, vb.pp("self_attn"))?,
            mlp: Mlp::load(cfg, quant, vb.pp("mlp"))?,
            input_ln: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_ln: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_ln.forward(xs)?;
        let xs = self.attn.forward(&xs, mask, rope)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let h = self.mlp.forward(&xs.apply(&self.post_attn_ln)?)?;
        Ok((residual + h)?)
    }
}

/// Qwen3-VL text encoder producing pre-final-norm hidden states.
pub struct QwenTextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    rope: RotaryEmbedding,
    device: Device,
}

impl QwenTextEncoder {
    /// Load the text encoder. When `quant` is set, the decoder linears are Q8_0
    /// (weight-only) — ~half the VRAM, near-lossless — so all three models stay
    /// resident in 24 GB. The token embedding stays full precision.
    pub fn load(cfg: &TextConfig, quant: bool, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rope = RotaryEmbedding::new(cfg, vb.device(), vb.dtype())?;
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DecoderLayer::load(cfg, quant, vb_l.pp(i))?);
        }
        Ok(Self { embed_tokens, layers, rope, device: vb.device().clone() })
    }

    /// Load from a pre-quantized GGUF (written by `prequantize-text`): linears
    /// come back as Q8_0 `QMatMul` directly, norms/embedding are dequantized —
    /// so NO bf16 weight tensor is ever created on the GPU (candle's CUDA pool
    /// stays small, unlike quantize-on-load). Fixes the resident-mode VRAM peak.
    pub fn load_gguf(cfg: &TextConfig, qvb: candle_transformers::quantized_var_builder::VarBuilder) -> Result<Self> {
        let dev = qvb.device().clone();
        let dt = DType::BF16;
        let vb_m = qvb.pp("model").pp("language_model");
        let embed_w = vb_m.pp("embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?.dequantize(&dev)?.to_dtype(dt)?;
        let embed_tokens = Embedding::new(embed_w, cfg.hidden_size);
        let rope = RotaryEmbedding::new(cfg, &dev, dt)?;
        let rms = |dim: usize, vb: &candle_transformers::quantized_var_builder::VarBuilder| -> Result<RmsNorm> {
            let w = vb.get(dim, "weight")?.dequantize(&dev)?.to_dtype(dt)?;
            Ok(RmsNorm::new(w, cfg.rms_norm_eps))
        };
        let qlin = |ic: usize, oc: usize, vb: &candle_transformers::quantized_var_builder::VarBuilder| -> Result<QLinear> {
            Ok(QLinear::Quant(candle_core::quantized::QMatMul::from_arc(vb.get((oc, ic), "weight")?)?))
        };
        let (nh, nkv, hd, h, ii) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size, cfg.intermediate_size);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_layers {
            let vb = vb_l.pp(i);
            let sa = vb.pp("self_attn");
            let attn = Attention {
                q_proj: qlin(h, nh * hd, &sa.pp("q_proj"))?,
                k_proj: qlin(h, nkv * hd, &sa.pp("k_proj"))?,
                v_proj: qlin(h, nkv * hd, &sa.pp("v_proj"))?,
                o_proj: qlin(nh * hd, h, &sa.pp("o_proj"))?,
                q_norm: rms(hd, &sa.pp("q_norm"))?,
                k_norm: rms(hd, &sa.pp("k_norm"))?,
                num_heads: nh,
                num_kv_heads: nkv,
                head_dim: hd,
                groups: nh / nkv,
                scale: 1.0 / (hd as f64).sqrt(),
            };
            let mlp = Mlp {
                gate: qlin(h, ii, &vb.pp("mlp").pp("gate_proj"))?,
                up: qlin(h, ii, &vb.pp("mlp").pp("up_proj"))?,
                down: qlin(ii, h, &vb.pp("mlp").pp("down_proj"))?,
            };
            layers.push(DecoderLayer { attn, mlp, input_ln: rms(h, &vb.pp("input_layernorm"))?, post_attn_ln: rms(h, &vb.pp("post_attention_layernorm"))? });
        }
        Ok(Self { embed_tokens, layers, rope, device: dev })
    }

    /// Build a causal additive mask `(1, 1, S, S)` (0 on/below diagonal, -inf above).
    fn causal_mask(&self, s: usize, dtype: DType) -> Result<Tensor> {
        let mut data = vec![0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                data[i * s + j] = f32::NEG_INFINITY;
            }
        }
        Ok(Tensor::from_vec(data, (1, 1, s, s), &self.device)?.to_dtype(dtype)?)
    }

    /// Run the decoder on `input_ids` `(B, S)` and return pre-final-norm hidden
    /// states `(B, S, hidden)`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, s) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mask = self.causal_mask(s, xs.dtype())?;
        let rope = self.rope.text_tables(s)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask, &rope)?;
        }
        Ok(xs)
    }

    /// The decoder on a prompt with pictures, as ComfyUI's `Qwen3VL` runs it: each `<|image_pad|>`
    /// token (one per picture, in order) is replaced by the picture's merged vision rows, the
    /// positions follow the multimodal RoPE (`qwen2vl_mrope_position_ids`), and the three
    /// DeepStack features are added to the picture's rows after decoder layers 0, 1 and 2.
    /// Returns the pre-final-norm hidden states `(1, S', hidden)` and each picture's
    /// `(start, size)` span in them.
    pub fn forward_with_images(&self, ids: &[u32], images: &[&VisionFeatures]) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let placeholders: Vec<usize> = ids.iter().enumerate().filter(|(_, t)| **t == IMAGE_PAD).map(|(i, _)| i).collect();
        anyhow::ensure!(placeholders.len() == images.len(), "{} <|image_pad|> tokens for {} pictures", placeholders.len(), images.len());
        let text = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device)?;
        let text = self.embed_tokens.forward(&text)?.squeeze(0)?; // (S0, hidden)
        let dt = text.dtype();
        // Splice: text rows between placeholders, the picture rows at each.
        let mut parts: Vec<Tensor> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut cursor = 0usize;
        let mut len = 0usize;
        for (p, img) in placeholders.iter().zip(images) {
            if *p > cursor {
                parts.push(text.narrow(0, cursor, p - cursor)?);
                len += p - cursor;
            }
            let n = img.tokens();
            spans.push((len, n));
            parts.push(img.merged.to_dtype(dt)?.to_device(&self.device)?);
            len += n;
            cursor = p + 1;
        }
        if cursor < ids.len() {
            parts.push(text.narrow(0, cursor, ids.len() - cursor)?);
            len += ids.len() - cursor;
        }
        let mut xs = Tensor::cat(&parts, 0)?.unsqueeze(0)?; // (1, S, hidden)
        let s = len;
        let grids: Vec<(usize, usize)> = images.iter().map(|i| (i.rows, i.cols)).collect();
        let rope = self.rope.tables(&mrope_positions(&spans, &grids, s))?;
        let mask = self.causal_mask(s, dt)?;
        for (li, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, &mask, &rope)?;
            if li < 3 {
                for ((start, n), img) in spans.iter().zip(images) {
                    let Some(ds) = img.deepstack.get(li) else { continue };
                    let rows = xs.narrow(1, *start, *n)?;
                    let added = (rows + ds.to_dtype(dt)?.to_device(&self.device)?.unsqueeze(0)?)?;
                    xs = xs.slice_assign(&[0..1, *start..*start + *n, 0..xs.dim(2)?], &added)?;
                }
            }
        }
        Ok((xs, spans))
    }
}

/// `<|image_pad|>`, the one token a picture occupies in the template.
pub const IMAGE_PAD: u32 = 151_655;

/// `qwen2vl_mrope_position_ids` for `spans` (start, size) of pictures with merged grids
/// `(rows, cols)` in a sequence of `seq` rows: text positions run sequentially on all three
/// axes; a picture's rows share its temporal position and spread over its grid; the text after
/// it resumes at `start + max(rows, cols)` (plus what earlier pictures shifted).
pub fn mrope_positions(spans: &[(usize, usize)], grids: &[(usize, usize)], seq: usize) -> [Vec<f32>; 3] {
    let mut pos = [vec![0f32; seq], vec![0f32; seq], vec![0f32; seq]];
    for axis in pos.iter_mut() {
        for (i, v) in axis.iter_mut().enumerate() {
            *v = i as f32;
        }
    }
    let mut offset: i64 = 0;
    for (&(start, n), &(rows, cols)) in spans.iter().zip(grids) {
        let end = start + n;
        let len_max = rows.max(cols);
        let start_next = (len_max + start) as i64;
        for (k, i) in (end..seq).enumerate() {
            let v = (start_next + offset + k as i64) as f32;
            pos[0][i] = v;
            pos[1][i] = v;
            pos[2][i] = v;
        }
        let base = start as i64 + offset;
        for k in 0..n {
            pos[0][start + k] = base as f32;
            pos[1][start + k] = (base + (k / cols.max(1)) as i64) as f32;
            pos[2][start + k] = (base + (k % cols.max(1)) as i64) as f32;
        }
        offset += len_max as i64 - n as i64;
    }
    pos
}

/// The DiT's conditioning from a multimodal hidden state: drop the `drop` leading rows (the
/// system turn) and every picture span, and record where each picture's reference latent goes
/// (`image_slots`: the kept rows before it).
pub fn condition(hidden: &Tensor, spans: &[(usize, usize)], drop: usize) -> Result<(Tensor, Vec<usize>)> {
    let (_b, s, _) = hidden.dims3()?;
    let mut keep: Vec<u32> = Vec::with_capacity(s);
    let mut slots = Vec::with_capacity(spans.len());
    let mut i = drop;
    for &(start, n) in spans {
        anyhow::ensure!(start >= i, "picture span at {start} inside the dropped rows");
        keep.extend((i..start).map(|v| v as u32));
        slots.push(keep.len());
        i = start + n;
    }
    keep.extend((i..s).map(|v| v as u32));
    let n = keep.len();
    let idx = Tensor::from_vec(keep, n, hidden.device())?;
    let cond = hidden.index_select(&idx, 1)?.contiguous()?;
    Ok((cond, slots))
}

/// The prompt-encoding recipe from `QwenImage21Pipeline` (text-to-image).
pub mod prompt {
    /// System prompt prepended to every text-to-image prompt.
    pub const SYS_PROMPT: &str = "Comprehend and analyze the provided prompt.";

    /// Build the exact t2i chat-template string for a user prompt.
    pub fn t2i_template(prompt: &str) -> String {
        let p = if prompt.is_empty() { " " } else { prompt };
        format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
    }

    /// One picture's place in the template.
    pub const VISION_BLOCK: &str = "<|vision_start|><|image_pad|><|vision_end|>";

    /// The template with `images` pictures ahead of the prompt, as `QwenImage21Tokenizer`
    /// builds it: `<image1>` + the vision block for each, separated by spaces, then the text.
    pub fn edit_template(prompt: &str, images: usize) -> String {
        if images == 0 {
            return t2i_template(prompt);
        }
        let p = if prompt.is_empty() { " " } else { prompt };
        let refs: Vec<String> = (1..=images).map(|i| format!("<image{i}>{VISION_BLOCK}")).collect();
        format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{}{p}<|im_end|>\n<|im_start|>assistant\n", refs.join(" "))
    }
}

/// Cosine similarity between two flattened tensors (for validation).
pub fn cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?.to_dtype(DType::F32)?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (na * nb + 1e-8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_has_roles() {
        let t = prompt::t2i_template("a cat");
        assert!(t.contains("<|im_start|>system\nComprehend and analyze"));
        assert!(t.contains("<|im_start|>user\na cat<|im_end|>"));
        assert!(t.ends_with("<|im_start|>assistant\n"));
        let e = prompt::edit_template("make it winter", 2);
        assert!(e.contains("<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>make it winter<|im_end|>"), "{e}");
        assert_eq!(prompt::edit_template("x", 0), prompt::t2i_template("x"));
    }

    #[test]
    fn mrope_positions_follow_the_node() {
        // text 0,1 | picture of 2x2 merged tokens at rows 2..6 | text after
        let p = mrope_positions(&[(2, 4)], &[(2, 2)], 8);
        assert_eq!(p[0], [0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0, 5.0]);
        assert_eq!(p[1], [0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0]);
        assert_eq!(p[2], [0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0]);
        // a wide picture: text after resumes at start + max(rows, cols)
        let p = mrope_positions(&[(1, 6)], &[(2, 3)], 9);
        assert_eq!(p[0][7..], [4.0, 5.0]);
        assert_eq!(p[2][1..7], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        // two pictures: the second shifts by (max − n) of the first (negative here)
        let p = mrope_positions(&[(0, 4), (5, 4)], &[(2, 2), (2, 2)], 10);
        assert_eq!(p[0][4], 2.0, "text between: start_next = 0 + 2");
        assert_eq!(p[0][5], 3.0, "second picture at 5 + offset(2 − 4)");
        assert_eq!(p[0][9], 5.0, "after the second: 5 + 2 + (−2) + (−2) ... = start_next 7 + offset −4");
    }

    #[test]
    fn condition_drops_the_system_turn_and_the_pictures() -> Result<()> {
        let dev = Device::Cpu;
        let h = Tensor::arange(0f32, 10.0, &dev)?.reshape((1, 10, 1))?;
        let (c, slots) = condition(&h, &[(4, 3)], 2)?;
        assert_eq!(c.flatten_all()?.to_vec1::<f32>()?, [2.0, 3.0, 7.0, 8.0, 9.0]);
        assert_eq!(slots, [2]);
        Ok(())
    }

    #[test]
    fn empty_prompt_becomes_space() {
        assert!(prompt::t2i_template("").contains("user\n <|im_end|>"));
    }

    #[test]
    fn repeat_kv_expands_heads() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::zeros((1, 8, 5, 128), DType::F32, &dev)?;
        assert_eq!(repeat_kv(x, 4)?.dims(), &[1, 32, 5, 128]);
        Ok(())
    }

    #[test]
    fn cosine_identical_is_one() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &dev)?;
        assert!((cosine(&a, &a)? - 1.0).abs() < 1e-5);
        Ok(())
    }
}
