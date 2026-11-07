//! FLUX 3 Image Klein's text encoder on MLX: the language tower of Qwen3-VL-4B-Instruct (32 of
//! its 36 layers, all the reference reads), from mflux's MLX-quantized files. The prompt goes
//! through Qwen's chat template. The hidden states after layers 4, 8, …, 32 are stacked into
//! `(L, 20480)` tokens. L is the prompt's length padded on the right to a multiple of 80, as
//! the reference pads it.
//!
//! Padding: the reference masks the pad tokens as keys, and gives every pad position 1. So
//! every pad query sees the same input (the pad token at position 1) and attends to the same
//! keys (all the prompt's own tokens). Every pad row therefore comes out the same at every
//! layer. This runs the prompt causally, plus one pad row that attends to all of it, and
//! repeats that row.

use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype, fast, ops};
use tokenizers::Tokenizer;

use crate::nn::{GROUP, Linear, Params, cols, dim, get, rows, shape};

pub const HIDDEN: usize = 2560;
const LAYERS: usize = 32;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const THETA: f64 = 5_000_000.0;
const EPS: f32 = 1e-6;
/// The hidden states the DiT reads (after this many layers).
const OUTPUT_LAYERS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
const PAD_MULTIPLE: usize = 80;
const MAX_LENGTH: usize = 4096;
/// Qwen's `<|endoftext|>`, its pad token.
const PAD_TOKEN: &str = "<|endoftext|>";

/// Qwen3-VL's chat template for one user message, ready for the answer.
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// The prompt's length padded to the reference's multiple of 80, at most 4096.
pub fn padded_len(real: usize) -> usize {
    real.div_ceil(PAD_MULTIPLE).saturating_mul(PAD_MULTIPLE).clamp(PAD_MULTIPLE, MAX_LENGTH)
}

struct Layer {
    in_norm: Array,
    post_norm: Array,
    q_norm: Array,
    k_norm: Array,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Layer {
    fn load(p: &Params, i: usize) -> Result<Self> {
        let n = format!("layers.{i}");
        let w = |s: &str| get(p, &format!("{n}.{s}.weight"));
        let l = |s: &str| Linear::load(p, &format!("{n}.{s}"));
        Ok(Self {
            in_norm: w("input_layernorm")?,
            post_norm: w("post_attention_layernorm")?,
            q_norm: w("self_attn.q_norm")?,
            k_norm: w("self_attn.k_norm")?,
            q: l("self_attn.q_proj")?,
            k: l("self_attn.k_proj")?,
            v: l("self_attn.v_proj")?,
            o: l("self_attn.o_proj")?,
            gate: l("mlp.gate_proj")?,
            up: l("mlp.up_proj")?,
            down: l("mlp.down_proj")?,
        })
    }
}

fn rms(x: &Array, w: &Array) -> Result<Array> {
    Ok(fast::rms_norm(x.as_dtype(Dtype::Float32)?, Some(&w.as_dtype(Dtype::Float32)?), EPS)?)
}

/// Rotate-half RoPE tables `(S, 1, D)` f32 for `positions`.
fn rope(positions: &[usize]) -> Result<(Array, Array)> {
    let half = HEAD_DIM / 2;
    let (mut cos, mut sin) = (Vec::with_capacity(positions.len() * HEAD_DIM), Vec::with_capacity(positions.len() * HEAD_DIM));
    for &p in positions {
        let row: Vec<f64> = (0..half).map(|i| p as f64 / THETA.powf((2 * i) as f64 / HEAD_DIM as f64)).collect();
        for _ in 0..2 {
            cos.extend(row.iter().map(|a| a.cos() as f32));
            sin.extend(row.iter().map(|a| a.sin() as f32));
        }
    }
    let s = shape(&[positions.len(), 1, HEAD_DIM])?;
    Ok((Array::from_slice(&cos, &s), Array::from_slice(&sin, &s)))
}

/// `x·cos + rotate_half(x)·sin` for `x (S, H, D)` f32.
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let half = HEAD_DIM / 2;
    let turned = ops::concatenate(&[cols(x, half, HEAD_DIM)?.negative()?, cols(x, 0, half)?], 2)?;
    Ok(x.multiply(cos)?.add(turned.multiply(sin)?)?)
}

/// `(S, H, D)` to `(1, H, S, D)`.
fn heads_first(t: &Array) -> Result<Array> {
    Ok(t.transpose_axes(&[1, 0, 2])?.expand_dims(0)?)
}

pub struct KleinText {
    dtype: Dtype,
    tok: Tokenizer,
    pad_id: u32,
    embed: (Array, Array, Array, i32),
    layers: Vec<Layer>,
}

impl KleinText {
    /// mflux's `text_encoder` folder: every `*.safetensors` in it, and its `tokenizer.json`.
    pub fn load(dir: &Path, dtype: Dtype) -> Result<Self> {
        let p = super::load_dir(dir, dtype)?;
        let tok_path = dir.join("tokenizer.json");
        let tok = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("load {}: {e}", tok_path.display()))?;
        let pad_id = tok.token_to_id(PAD_TOKEN).context("the tokenizer has no <|endoftext|>")?;
        let w = get(&p, "embed_tokens.weight")?;
        let s = get(&p, "embed_tokens.scales")?;
        let z = get(&p, "embed_tokens.biases")?;
        let k = s.shape().last().copied().unwrap_or(0) * GROUP;
        let words = w.shape().last().copied().unwrap_or(0);
        anyhow::ensure!(k > 0 && (words * 32) % k == 0, "embed_tokens: quantized shapes {:?} and {:?}", w.shape(), s.shape());
        let layers = (0..LAYERS).map(|i| Layer::load(&p, i)).collect::<Result<_>>()?;
        Ok(Self { dtype, tok, pad_id, embed: (w, s, z, words * 32 / k), layers })
    }

    /// The embeddings of `ids` `(S, HIDDEN)` f32: their rows of the quantized table, dequantized.
    fn embed(&self, ids: &[u32]) -> Result<Array> {
        let (w, s, z, bits) = &self.embed;
        let idx = Array::from_slice(&ids.iter().map(|&i| i as i32).collect::<Vec<_>>(), &[dim(ids.len())?]);
        let take = |a: &Array| -> Result<Array> { Ok(a.take_axis(&idx, 0)?) };
        Ok(ops::dequantize(&take(w)?, &take(s)?, &take(z)?, GROUP, *bits)?.as_dtype(Dtype::Float32)?)
    }

    /// The prompt's tokens `(L, 20480)` in the run dtype, `L` padded to a multiple of 80.
    pub fn encode(&self, prompt: &str) -> Result<Array> {
        let enc = self.tok.encode(template(prompt), false).map_err(|e| anyhow::anyhow!("encode the prompt: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(MAX_LENGTH);
        let real = ids.len();
        anyhow::ensure!(real > 0, "an empty prompt encoding");
        let total = padded_len(real);
        let pads = total - real;
        let dt = self.dtype;

        let mut x = self.embed(&ids)?;
        let mut pad = if pads > 0 { Some(self.embed(&[self.pad_id])?) } else { None };
        let positions: Vec<usize> = (0..real).collect();
        let (cos, sin) = rope(&positions)?;
        let (pcos, psin) = rope(&[1])?;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut outputs: Vec<Array> = Vec::with_capacity(OUTPUT_LAYERS.len());
        for (i, l) in self.layers.iter().enumerate() {
            // The prompt's rows and the pad row through the same projections.
            let both = match &pad {
                Some(p) => ops::concatenate(&[&x, p], 0)?,
                None => x.clone(),
            };
            let n = both.shape()[0];
            let y = rms(&both, &l.in_norm)?.as_dtype(dt)?;
            let q = rms(&l.q.forward(&y)?.reshape(&[n, dim(HEADS)?, dim(HEAD_DIM)?])?, &l.q_norm)?;
            let k = rms(&l.k.forward(&y)?.reshape(&[n, dim(KV_HEADS)?, dim(HEAD_DIM)?])?, &l.k_norm)?;
            let v = l.v.forward(&y)?.reshape(&[n, dim(KV_HEADS)?, dim(HEAD_DIM)?])?;
            let qr = apply_rope(&rows(&q, 0, real)?, &cos, &sin)?.as_dtype(dt)?;
            let kr = apply_rope(&rows(&k, 0, real)?, &cos, &sin)?.as_dtype(dt)?;
            let vr = rows(&v, 0, real)?;
            // Causal over the prompt; the KV heads shared by their query groups.
            let att = fast::scaled_dot_product_attention(
                heads_first(&qr)?,
                heads_first(&kr)?,
                heads_first(&vr)?,
                scale,
                fast::ScaledDotProductAttentionMask::Causal,
                None,
            )?;
            let mut att = att.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[dim(real)?, dim(HEADS * HEAD_DIM)?])?;
            if pad.is_some() {
                // The pad row (position 1) sees every prompt token and no pad.
                let qp = apply_rope(&rows(&q, real, real + 1)?, &pcos, &psin)?.as_dtype(dt)?;
                let ap = fast::scaled_dot_product_attention(heads_first(&qp)?, heads_first(&kr)?, heads_first(&vr)?, scale, None, None)?;
                let ap = ap.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[1, dim(HEADS * HEAD_DIM)?])?;
                att = ops::concatenate(&[&att, &ap], 0)?;
            }
            let h = both.add(l.o.forward(&att)?.as_dtype(Dtype::Float32)?)?;
            let y = rms(&h, &l.post_norm)?.as_dtype(dt)?;
            let f = mlx_rs::nn::silu(l.gate.forward(&y)?)?.multiply(l.up.forward(&y)?)?;
            let h = h.add(l.down.forward(&f)?.as_dtype(Dtype::Float32)?)?;
            h.eval()?;
            x = rows(&h, 0, real)?;
            if pad.is_some() {
                pad = Some(rows(&h, real, real + 1)?);
            }
            if OUTPUT_LAYERS.contains(&(i + 1)) {
                let mut full = x.clone();
                if let Some(p) = &pad {
                    full = ops::concatenate(&[&full, &ops::broadcast_to(p, &[dim(pads)?, dim(HIDDEN)?])?], 0)?;
                }
                let full = full.as_dtype(dt)?;
                full.eval()?;
                outputs.push(full);
            }
        }
        // Each token's eight hidden states side by side, the earliest first.
        let out = ops::concatenate(&outputs, 1)?;
        out.eval()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_follows_the_reference() {
        assert_eq!(padded_len(1), 80);
        assert_eq!(padded_len(80), 80);
        assert_eq!(padded_len(81), 160);
        assert_eq!(padded_len(5000), 4096);
        assert_eq!(template("a cat"), "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n");
    }
}
