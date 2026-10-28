//! Qwen2.5-VL text and edit conditioning, including multimodal positions and the template cut.

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
        self.forward_inner(x, c, cos, sin, mask, false)
    }

    fn forward_edit(&self, x: &Tensor, c: &Qwen25Config, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, c, cos, sin, mask, true)
    }

    fn forward_inner(&self, x: &Tensor, c: &Qwen25Config, cos: &Tensor, sin: &Tensor, mask: &Tensor, edit: bool) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let normalized = if edit { edit_norm(&self.input_norm, x)? } else { self.input_norm.forward(x)? };
        let q = self.q.forward(&normalized)?.reshape((b, s, c.num_heads, c.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = self.k.forward(&normalized)?.reshape((b, s, c.num_kv_heads, c.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(&normalized)?.reshape((b, s, c.num_kv_heads, c.head_dim))?.transpose(1, 2)?.contiguous()?;
        // Text retains its original f32 attention; CPU edit may accumulate in f64.
        let acc = if edit && x.device().is_cpu() && x.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
        let q = candle_nn::rotary_emb::rope(&q.to_dtype(acc)?, cos, sin)?.to_dtype(x.dtype())?;
        let k = candle_nn::rotary_emb::rope(&k.to_dtype(acc)?, cos, sin)?.to_dtype(x.dtype())?;
        let groups = c.num_heads / c.num_kv_heads;
        let k = repeat_kv(&k, groups)?.to_dtype(acc)?;
        let v = repeat_kv(&v, groups)?;
        let scores = (q.to_dtype(acc)?.matmul(&k.transpose(2, 3)?)? / (c.head_dim as f64).sqrt())?.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = probs.matmul(&v.to_dtype(acc)?)?.to_dtype(x.dtype())?.transpose(1, 2)?.reshape((b, s, c.num_heads * c.head_dim))?;
        let x = (x + self.o.forward(&attended)?)?;
        let norm = if edit { edit_norm(&self.post_norm, &x)? } else { self.post_norm.forward(&x)? };
        let gate = candle_nn::ops::silu(&self.gate.forward(&norm)?)?;
        Ok((&x + self.down.forward(&(gate * self.up.forward(&norm)?)?)?)?)
    }
}

// Avoid rounding each CPU edit residual back to f32; Layered keeps its original kernel.
fn edit_norm(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    if x.dtype() == DType::F64 {
        let mean = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let scale = (mean + norm.eps())?.sqrt()?.recip()?;
        Ok(x.broadcast_mul(&scale)?.broadcast_mul(&norm.weight().to_dtype(x.dtype())?)?)
    } else {
        Ok(norm.forward(x)?)
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
    /// Encode EditPlus using caller-supplied merged vision rows; the tower stays independent.
    /// Grids are `[t,h,w]` in 14px patches before merging; outputs are `[1,T,3584]` and `[1,T]`.
    pub fn encode_edit(&self, tokenizer: &Tokenizer, prompt: &str, pictures: &[(Tensor, [usize; 3])]) -> Result<(Tensor, Tensor)> {
        let input = self.prepare_edit(tokenizer, prompt, pictures)?;
        let output = self.forward_edit_taps(&input, &[])?;
        let s = input.ids.len();
        Ok((output.conditioning, input.mask.narrow(1, input.cut, s - input.cut)?.contiguous()?))
    }

    /// Expose the splice and positions for parity checks without running the decoder.
    pub fn prepare_edit(&self, tokenizer: &Tokenizer, prompt: &str, pictures: &[(Tensor, [usize; 3])]) -> Result<Qwen25EditInput> {
        anyhow::ensure!(self.config.hidden_size == 3584 && self.config.head_dim == 128, "edit requires the Qwen25 7B backbone");
        let template = edit_prompt_template(prompt, pictures.len())?;
        let ids = tokenizer.encode(template, false).map_err(|e| anyhow::anyhow!("tokenize edit: {e}"))?.get_ids().to_vec();
        let grids = pictures.iter().map(|(_, grid)| *grid).collect::<Vec<_>>();
        let (expanded, spans, positions) = edit_layout(&ids, &grids)?;
        let mut chunks = Vec::new();
        let mut cursor = 0;
        for ((rows, grid), &(start, size)) in pictures.iter().zip(&spans) {
            anyhow::ensure!(rows.dims() == [size, 3584], "vision rows do not match merged grid {grid:?}");
            if start > cursor {
                chunks.push(self.embedding.forward(&Tensor::from_slice(&expanded[cursor..start], (1, start - cursor), &self.device)?)?.to_dtype(self.dtype)?);
            }
            chunks.push(rows.to_device(&self.device)?.to_dtype(self.dtype)?.unsqueeze(0)?);
            cursor = start + size;
        }
        if cursor < expanded.len() {
            chunks.push(self.embedding.forward(&Tensor::from_slice(&expanded[cursor..], (1, expanded.len() - cursor), &self.device)?)?.to_dtype(self.dtype)?);
        }
        let embeddings = Tensor::cat(&chunks, 1)?;
        let mask = Tensor::from_vec(edit_attention_mask(&expanded, &spans), (1, expanded.len()), &self.device)?;
        Ok(Qwen25EditInput { cut: prefix_cut(&ids), ids: expanded, spans, positions, embeddings, mask })
    }

    /// Run contiguous-section Qwen25 M-RoPE, preserving full-sequence pre-norm taps.
    /// CPU f32 requests use wider intermediates to bound drift; outputs keep the encoder dtype.
    pub fn forward_edit_taps(&self, input: &Qwen25EditInput, taps: &[usize]) -> Result<Qwen25Output> {
        let s = input.ids.len();
        anyhow::ensure!(s > 0 && input.cut < s && input.embeddings.dims() == [1, s, self.config.hidden_size], "invalid edit embeddings or cut");
        anyhow::ensure!(input.mask.dims() == [1, s] && input.positions.iter().all(|p| p.len() == s), "invalid edit mask or positions");
        anyhow::ensure!(self.config.head_dim == 128 && taps.iter().all(|i| *i < self.config.num_layers), "invalid edit decoder configuration or taps");
        let compute = if self.device.is_cpu() && self.dtype == DType::F32 { DType::F64 } else { self.dtype };
        let mut cos_values = Vec::with_capacity(s * 64);
        let mut sin_values = Vec::with_capacity(s * 64);
        for p in 0..s {
            // `j` is the frequency index of the formula below as well as the table's.
            #[allow(clippy::needless_range_loop)]
            for j in 0..64 {
                let axis = if j < 16 {
                    0
                } else if j < 40 {
                    1
                } else {
                    2
                };
                // ComfyUI rounds the reciprocal before multiplying positions in f32.
                let inv_freq = if self.config.rope_theta == 1_000_000.0 {
                    f32::from_bits(EDIT_INV_FREQ[j])
                } else {
                    1.0 / self.config.rope_theta.powf((2 * j) as f32 / 128.0)
                };
                let angle = input.positions[axis][p] as f32 * inv_freq;
                let (cos, sin) = ((angle as f64).cos() as f32, (angle as f64).sin() as f32);
                cos_values.push(cos);
                sin_values.push(sin);
            }
        }
        let cos = Tensor::from_vec(cos_values, (s, 64), &self.device)?.to_dtype(if compute == DType::F64 { DType::F64 } else { DType::F32 })?;
        let sin = Tensor::from_vec(sin_values, (s, 64), &self.device)?.to_dtype(if compute == DType::F64 { DType::F64 } else { DType::F32 })?;
        let attended = input.mask.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mask = (0..s)
            .flat_map(|i| {
                attended.iter().enumerate().map(move |(j, a)| (if j > i { f32::MIN / 4.0 } else { 0.0 }) + (if *a == 0.0 { f32::MIN / 4.0 } else { 0.0 }))
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (s, s), &self.device)?.to_dtype(if compute == DType::F64 { DType::F64 } else { DType::F32 })?;
        let mut x = input.embeddings.to_dtype(compute)?;
        let mut states = Vec::new();
        for i in 0..self.config.num_layers {
            x = match &self.blocks {
                Blocks::Resident(blocks) => {
                    blocks.get(i).ok_or_else(|| anyhow::anyhow!("missing decoder block {i}"))?.forward_edit(&x, &self.config, &cos, &sin, &mask)?
                }
                Blocks::Stream(file) => Block::load(&self.config, file.builder(&format!("model.layers.{i}."), self.dtype, &self.device)?)?.forward_edit(
                    &x,
                    &self.config,
                    &cos,
                    &sin,
                    &mask,
                )?,
            };
            if taps.contains(&i) {
                states.push((i, x.to_dtype(self.dtype)?));
            }
        }
        Ok(Qwen25Output { conditioning: edit_norm(&self.norm, &x)?.narrow(1, input.cut, s - input.cut)?.contiguous()?.to_dtype(self.dtype)?, taps: states })
    }

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

    /// CUDA retains blocks for both prompts; CPU streams them. Embeddings stay bf16 until gathered.
    pub fn from_file(path: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let config = Qwen25Config::default();
        let file = TensorFile::open(path)?;
        let embedding = Embedding::new(file.tensor("model.embed_tokens.weight", device)?, config.hidden_size);
        let norm = RmsNorm::new(file.tensor("model.norm.weight", device)?.to_dtype(dtype)?, config.rms_norm_eps);
        let blocks = if device.is_cuda() {
            Blocks::Resident(
                (0..config.num_layers)
                    .map(|i| Block::load(&config, file.builder(&format!("model.layers.{i}."), dtype, device)?))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            Blocks::Stream(file)
        };
        Ok(Self { config, embedding, norm, blocks, device: device.clone(), dtype })
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

pub const EDIT_SYSTEM_PROMPT: &str = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";

// Torch f32 reciprocals of theta^(2j/128), theta=1e6. Preserve ComfyUI's rounding
// independently of the host powf library (one frequency differs on Windows).
const EDIT_INV_FREQ: [u32; 64] = [
    1065353216, 1062095789, 1059470816, 1057355503, 1054337179, 1051589887, 1049376004, 1046607925, 1043732615, 1041415569, 1038909398, 1035900102, 1033475085,
    1031243027, 1028093503, 1025555485, 1023510243, 1020314024, 1017657739, 1015517192, 1012562930, 1009782866, 1007542573, 1004841542, 1001931932, 999587246,
    997151246, 994106053, 991652109, 989493488, 986306396, 983738103, 981668463, 978534184, 975846212, 973680130, 970790693, 967977467, 965710450, 963077266,
    960132947, 957760292, 955395298, 952313781, 949830564, 947746260, 944521150, 941922219, 939827890, 936756289, 934036253, 931844333, 929020495, 926173710,
    923879651, 921315122, 918335681, 915934722, 913641584, 910523307, 908010468, 905985516, 902737785, 900107852,
];

/// Expanded IDs retain 151655 over vision spans for inspection; embeddings replace those rows.
pub struct Qwen25EditInput {
    pub ids: Vec<u32>,
    pub spans: Vec<(usize, usize)>,
    pub positions: [Vec<i64>; 3],
    pub embeddings: Tensor,
    pub mask: Tensor,
    pub cut: usize,
}

/// EditPlus labels consecutive picture slots and concatenates them directly with the prompt.
pub fn edit_prompt_template(prompt: &str, pictures: usize) -> Result<String> {
    anyhow::ensure!(pictures <= 3, "EditPlus accepts at most three pictures");
    let labels = (1..=pictures).map(|i| format!("Picture {i}: <|vision_start|><|image_pad|><|vision_end|>")).collect::<String>();
    let text = format!("{labels}{prompt}");
    if text.starts_with("<|im_start|>") || text.starts_with("<|start_header_id|>") {
        return Ok(text);
    }
    Ok(format!("<|im_start|>system\n{EDIT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"))
}

/// [`edit_layout`]'s result: the expanded ids, each picture's (start, length) span, and the three
/// M-RoPE position axes.
pub type EditLayout = (Vec<u32>, Vec<(usize, usize)>, [Vec<i64>; 3]);

/// Match Qwen25's image indexing, including its height repetition and cumulative offsets.
pub fn edit_layout(ids: &[u32], grids: &[[usize; 3]]) -> Result<EditLayout> {
    anyhow::ensure!(grids.len() <= 3 && !ids.is_empty(), "invalid edit token layout");
    let mut expanded = Vec::new();
    let mut spans = Vec::new();
    for &id in ids {
        if id == 151655 && spans.len() < grids.len() {
            let [t, h, w] = grids[spans.len()];
            anyhow::ensure!(t > 0 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0, "invalid pre-merge vision grid");
            let size = t.checked_mul(h / 2).and_then(|n| n.checked_mul(w / 2)).ok_or_else(|| anyhow::anyhow!("vision grid overflow"))?;
            spans.push((expanded.len(), size));
            expanded.resize(expanded.len().checked_add(size).ok_or_else(|| anyhow::anyhow!("vision span overflow"))?, id);
        } else {
            expanded.push(id);
        }
    }
    anyhow::ensure!(spans.len() == grids.len(), "missing image placeholder");
    let mask = edit_attention_mask(&expanded, &spans);
    let mut positions: [Vec<i64>; 3] = std::array::from_fn(|_| (0..expanded.len()).map(|p| p as i64).collect());
    let mut offset = 0i64;
    for (&(start, size), &[t, h, w]) in spans.iter().zip(grids) {
        let base = start as i64 + offset;
        let len_max = t.max(h).max(w) / 2;
        for axis in &mut positions {
            let mut attended = 0;
            for (j, value) in axis.iter_mut().enumerate().skip(start + size) {
                if mask[j] != 0.0 {
                    *value = base + len_max as i64 + attended;
                    attended += 1;
                } else {
                    // ComfyUI initializes unassigned masked positions to one.
                    *value = 1;
                }
            }
        }
        let height_repeat = size.div_ceil(h / 2);
        for j in 0..size {
            positions[0][start + j] = base;
            positions[1][start + j] = base + (j / height_repeat) as i64;
            positions[2][start + j] = base + (j % (w / 2)) as i64;
        }
        offset += len_max as i64 - size as i64;
    }
    Ok((expanded, spans, positions))
}

fn edit_attention_mask(ids: &[u32], spans: &[(usize, usize)]) -> Vec<f32> {
    let mut eos = false;
    ids.iter()
        .enumerate()
        .map(|(i, id)| {
            if spans.iter().any(|&(start, size)| i >= start && i < start + size) {
                1.0
            } else {
                if *id == 151643 {
                    eos = true;
                }
                if eos {
                    0.0
                } else {
                    1.0
                }
            }
        })
        .collect()
}

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

#[cfg(test)]
mod tests {
    use super::EDIT_INV_FREQ;

    #[test]
    fn comfy_edit_inverse_frequencies() -> anyhow::Result<()> {
        let meta: serde_json::Value = serde_json::from_str(include_str!("../../tests/fixtures/qwen25_edit.json"))?;
        let expected: Vec<u32> = serde_json::from_value(meta["mrope_inv_freq_bits"].clone())?;
        assert_eq!(EDIT_INV_FREQ.as_slice(), expected.as_slice());
        Ok(())
    }
}
