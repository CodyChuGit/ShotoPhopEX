//! MOSS-Image's text encoder on MLX: Qwen3.5-9B's language model as
//! `qwen_image_rs::moss::text` runs it (its last hidden state after the final norm). The
//! embedding stays qir's INT8 table on the CPU (only the prompt's rows are gathered); the
//! residual stream, norms, attention and the delta rule run in f32, the linears in the run
//! dtype.

use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype, fast, ops};
use qwen_image_rs::comfy_int8::Prequantized;
use qwen_image_rs::moss::text::TextConfig;
use qwen_image_rs::moss::weights::Weights;

use crate::convert::{Converter, Precision};
use crate::nn::{Linear, Params, cols, dim, from_candle, get, shape};

const P: &str = "model.language_model";

/// The encoder's weights: the converted layers and qir's INT8 embedding table.
pub struct TextWeights {
    pub params: Params,
    pub embed: Prequantized,
}

/// Convert the encoder file: every layer's large linears quantized (when `precision` asks: 8-bit
/// is the encoder's, 4-bit costs it 2 % cosine over 8 layers, spread over every kind of linear),
/// the DeltaNet's small `in_proj_b`/`in_proj_a` dense, the norms and the DeltaNet's state
/// parameters in f32. `cfg.layers` limits the layers read.
pub fn convert(path: &Path, cfg: &TextConfig, dtype: Dtype, precision: Precision) -> Result<TextWeights> {
    let w = Weights::open(path, &candle_core::Device::Cpu, candle_core::DType::F32, false)?;
    let embed = w.prequantized(&format!("{P}.embed_tokens")).context("the encoder's INT8 embedding table")?;
    anyhow::ensure!(embed.in_features == cfg.hidden, "embedding width {} (expected {})", embed.in_features, cfg.hidden);
    let mut c = Converter::new(&w, dtype, precision)?;
    c.tensor_f32(&format!("{P}.norm.weight"))?;
    for i in 0..cfg.layers {
        convert_layer(&mut c, cfg, i)?;
    }
    Ok(TextWeights { params: c.out, embed })
}

/// [`convert`] one layer at a time into `dir`: `base.safetensors` (the final norm) and
/// `layer-{i}.safetensors`, each written as soon as it is converted (so at most one layer is in
/// memory); the encoder then streams them ([`Qwen35Text::streamed`]).
pub fn convert_streamed(path: &Path, cfg: &TextConfig, dtype: Dtype, precision: Precision, dir: &Path) -> Result<()> {
    let w = Weights::open(path, &candle_core::Device::Cpu, candle_core::DType::F32, false)?;
    let mut c = Converter::new(&w, dtype, precision)?;
    for i in 0..cfg.layers {
        convert_layer(&mut c, cfg, i)?;
        crate::convert::save(&std::mem::take(&mut c.out), &dir.join(format!("layer-{i}.safetensors")))?;
        mlx_rs::memory::clear_cache()?;
    }
    c.tensor_f32(&format!("{P}.norm.weight"))?;
    // Last: a folder with its base holds every layer.
    crate::convert::save(&c.out, &dir.join("base.safetensors"))
}

/// The encoder's INT8 embedding table, as the file has it.
pub fn embedding(path: &Path, cfg: &TextConfig) -> Result<Prequantized> {
    let w = Weights::open(path, &candle_core::Device::Cpu, candle_core::DType::F32, false)?;
    let embed = w.prequantized(&format!("{P}.embed_tokens")).context("the encoder's INT8 embedding table")?;
    anyhow::ensure!(embed.in_features == cfg.hidden, "embedding width {} (expected {})", embed.in_features, cfg.hidden);
    Ok(embed)
}

fn convert_layer(c: &mut Converter, cfg: &TextConfig, i: usize) -> Result<()> {
    {
        let l = format!("{P}.layers.{i}");
        for n in ["input_layernorm", "post_attention_layernorm"] {
            c.tensor_f32(&format!("{l}.{n}.weight"))?;
        }
        for n in ["gate_proj", "up_proj", "down_proj"] {
            c.linear(&format!("{l}.mlp.{n}"), true, None)?;
        }
        if is_full_attention(cfg, i) {
            let a = format!("{l}.self_attn");
            for n in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                c.linear(&format!("{a}.{n}"), true, None)?;
            }
            for n in ["q_norm", "k_norm"] {
                c.tensor_f32(&format!("{a}.{n}.weight"))?;
            }
        } else {
            let a = format!("{l}.linear_attn");
            for (n, q) in [("in_proj_qkv", true), ("in_proj_z", true), ("in_proj_b", false), ("in_proj_a", false), ("out_proj", true)] {
                c.linear(&format!("{a}.{n}"), q, None)?;
            }
            for n in ["conv1d.weight", "A_log", "dt_bias", "norm.weight"] {
                c.tensor_f32(&format!("{a}.{n}"))?;
            }
        }
    }
    Ok(())
}

fn is_full_attention(cfg: &TextConfig, layer: usize) -> bool {
    (layer + 1).is_multiple_of(cfg.full_attention_interval)
}

/// RMSNorm with the zero-centred `(1 + w)` scale, in f32.
struct Norm1p {
    weight: Array,
}

impl Norm1p {
    fn load(p: &Params, name: &str) -> Result<Self> {
        Ok(Self { weight: get(p, name)?.add(Array::from_f32(1.0))? })
    }

    fn forward(&self, x: &Array, eps: f32) -> Result<Array> {
        Ok(fast::rms_norm(x.as_dtype(Dtype::Float32)?, Some(&self.weight), eps)?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

struct GatedAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Norm1p,
    k_norm: Norm1p,
}

struct DeltaNet {
    qkv: Linear,
    z: Linear,
    b: Linear,
    a: Linear,
    out: Linear,
    /// `(conv_dim, kernel)` f32.
    conv: Array,
    /// `-exp(A_log)` `(value_heads,)` f32.
    decay: Array,
    dt_bias: Array,
    norm: Array,
}

enum Mixer {
    Full(GatedAttention),
    Linear(DeltaNet),
}

struct Layer {
    input_norm: Norm1p,
    post_norm: Norm1p,
    mixer: Mixer,
    mlp: Mlp,
}

pub struct Qwen35Text {
    cfg: TextConfig,
    dtype: Dtype,
    embed: Prequantized,
    layers: Vec<Layer>,
    norm: Norm1p,
    /// [`Self::streamed`]'s folder of layers.
    stream: Option<std::path::PathBuf>,
}

impl Layer {
    fn load(p: &Params, cfg: &TextConfig, i: usize) -> Result<Self> {
        let l = format!("{P}.layers.{i}");
        let lin = |n: &str| Linear::load(p, &format!("{l}.{n}"));
        let mixer = if is_full_attention(cfg, i) {
            Mixer::Full(GatedAttention {
                q: lin("self_attn.q_proj")?,
                k: lin("self_attn.k_proj")?,
                v: lin("self_attn.v_proj")?,
                o: lin("self_attn.o_proj")?,
                q_norm: Norm1p::load(p, &format!("{l}.self_attn.q_norm.weight"))?,
                k_norm: Norm1p::load(p, &format!("{l}.self_attn.k_norm.weight"))?,
            })
        } else {
            let a = format!("{l}.linear_attn");
            let conv = get(p, &format!("{a}.conv1d.weight"))?;
            let conv = conv.reshape(&[conv.shape()[0], dim(cfg.conv_kernel)?])?;
            Mixer::Linear(DeltaNet {
                qkv: lin("linear_attn.in_proj_qkv")?,
                z: lin("linear_attn.in_proj_z")?,
                b: lin("linear_attn.in_proj_b")?,
                a: lin("linear_attn.in_proj_a")?,
                out: lin("linear_attn.out_proj")?,
                conv,
                decay: get(p, &format!("{a}.A_log"))?.exp()?.negative()?,
                dt_bias: get(p, &format!("{a}.dt_bias"))?,
                norm: get(p, &format!("{a}.norm.weight"))?,
            })
        };
        Ok(Layer {
            input_norm: Norm1p::load(p, &format!("{l}.input_layernorm.weight"))?,
            post_norm: Norm1p::load(p, &format!("{l}.post_attention_layernorm.weight"))?,
            mixer,
            mlp: Mlp { gate: lin("mlp.gate_proj")?, up: lin("mlp.up_proj")?, down: lin("mlp.down_proj")? },
        })
    }
}

impl Qwen35Text {
    pub fn load(cfg: &TextConfig, w: TextWeights, dtype: Dtype) -> Result<Self> {
        let p = &w.params;
        let layers = (0..cfg.layers).map(|i| Layer::load(p, cfg, i)).collect::<Result<_>>()?;
        let norm = Norm1p::load(p, &format!("{P}.norm.weight"))?;
        Ok(Self { cfg: cfg.clone(), dtype, embed: w.embed, layers, norm, stream: None })
    }

    /// The encoder streamed from [`convert_streamed`]'s `dir`: each layer read just before it runs
    /// and dropped after, so it holds one layer (about 180 MB) rather than all of them.
    pub fn streamed(cfg: &TextConfig, embed: Prequantized, dir: &Path, dtype: Dtype) -> Result<Self> {
        let base = crate::convert::load(&dir.join("base.safetensors"))?;
        let norm = Norm1p::load(&base, &format!("{P}.norm.weight"))?;
        Ok(Self { cfg: cfg.clone(), dtype, embed, layers: Vec::new(), norm, stream: Some(dir.to_path_buf()) })
    }

    /// The last hidden state `(tokens, hidden)` f32.
    pub fn forward(&self, ids: &[u32]) -> Result<Array> {
        anyhow::ensure!(!ids.is_empty(), "no tokens to encode");
        let eps = self.cfg.eps as f32;
        let dt = self.dtype;
        let mut x = from_candle(&self.embed.gather_cpu(ids)?)?;
        let (cos, sin) = self.rope(ids.len())?;
        for i in 0..self.cfg.layers {
            let streamed;
            let layer = match &self.stream {
                Some(dir) => {
                    streamed = Layer::load(&crate::convert::load(&dir.join(format!("layer-{i}.safetensors")))?, &self.cfg, i)?;
                    &streamed
                }
                None => self.layers.get(i).context("a layer went missing")?,
            };
            let h = layer.input_norm.forward(&x, eps)?.as_dtype(dt)?;
            let h = match &layer.mixer {
                Mixer::Full(a) => self.full_attention(a, &h, &cos, &sin)?,
                Mixer::Linear(d) => self.delta_net(d, &h)?,
            };
            x = x.add(h.as_dtype(Dtype::Float32)?)?;
            let y = layer.post_norm.forward(&x, eps)?.as_dtype(dt)?;
            let m = &layer.mlp;
            let h = m.down.forward(&mlx_rs::nn::silu(m.gate.forward(&y)?)?.multiply(m.up.forward(&y)?)?)?;
            x = x.add(h.as_dtype(Dtype::Float32)?)?;
            // A layer at a time: the delta rule's graph stays small.
            x.eval()?;
        }
        self.norm.forward(&x, eps)
    }

    /// cos/sin `(tokens, 1, rotary_dim)` f32 for positions `0..n` (each half repeated).
    fn rope(&self, n: usize) -> Result<(Array, Array)> {
        let r = self.cfg.rotary_dim;
        let half = r / 2;
        let inv: Vec<f32> = (0..half).map(|j| (1.0 / self.cfg.rope_theta.powf((2 * j) as f64 / r as f64)) as f32).collect();
        let mut angles = Vec::with_capacity(n * r);
        for p in 0..n {
            for _ in 0..2 {
                angles.extend(inv.iter().map(|f| p as f32 * f));
            }
        }
        let a = Array::from_slice(&angles, &shape(&[n, 1, r])?);
        Ok((a.cos()?, a.sin()?))
    }

    /// `x[..., :r]` rotated (rotate-half), the rest passed through; `x (S, H, D)` f32.
    fn partial_rope(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let r = self.cfg.rotary_dim;
        let d = self.cfg.head_dim;
        let half = r / 2;
        let rot = cols(x, 0, r)?;
        let turned = ops::concatenate(&[cols(&rot, half, r)?.negative()?, cols(&rot, 0, half)?], 2)?;
        let y = rot.multiply(cos)?.add(turned.multiply(sin)?)?;
        Ok(ops::concatenate(&[y, cols(x, r, d)?], 2)?)
    }

    fn full_attention(&self, a: &GatedAttention, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let c = &self.cfg;
        let eps = c.eps as f32;
        let s = x.shape()[0];
        let (h, kvh, hd) = (dim(c.heads)?, dim(c.kv_heads)?, dim(c.head_dim)?);
        let qg = a.q.forward(x)?.reshape(&[s, h, 2 * hd])?;
        let q = a.q_norm.forward(&cols(&qg, 0, c.head_dim)?, eps)?;
        let gate = cols(&qg, c.head_dim, 2 * c.head_dim)?.reshape(&[s, h * hd])?;
        let k = a.k_norm.forward(&a.k.forward(x)?.reshape(&[s, kvh, hd])?, eps)?;
        let v = a.v.forward(x)?.reshape(&[s, kvh, hd])?.as_dtype(Dtype::Float32)?;
        let q = self.partial_rope(&q, cos, sin)?;
        let k = self.partial_rope(&k, cos, sin)?;
        // (1, H, S, D) f32, the KV heads shared by their query groups; causal.
        let heads_first = |t: &Array| -> Result<Array> { Ok(t.transpose_axes(&[1, 0, 2])?.expand_dims(0)?) };
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let o =
            fast::scaled_dot_product_attention(heads_first(&q)?, heads_first(&k)?, heads_first(&v)?, scale, fast::ScaledDotProductAttentionMask::Causal, None)?;
        let o = o.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[s, h * hd])?.as_dtype(self.dtype)?;
        a.o.forward(&o.multiply(ops::sigmoid(&gate)?)?)
    }

    fn delta_net(&self, d: &DeltaNet, x: &Array) -> Result<Array> {
        let c = &self.cfg;
        let dt = self.dtype;
        let s = x.shape()[0];
        let (kd, vd) = (c.linear_key_heads * c.linear_key_dim, c.linear_value_heads * c.linear_value_dim);
        // The causal depthwise conv (left zero padding), then SiLU.
        let proj = d.qkv.forward(x)?.as_dtype(Dtype::Float32)?; // (S, conv_dim)
        let k_len = c.conv_kernel;
        let width = proj.shape()[1];
        let padded = ops::concatenate(&[Array::zeros::<f32>(&[dim(k_len - 1)?, width])?, proj], 0)?;
        let mut conv = Array::zeros::<f32>(&[s, width])?;
        for j in 0..k_len {
            let tap = padded.try_index(dim(j)?..dim(j)? + s)?;
            conv = conv.add(tap.multiply(cols(&d.conv, j, j + 1)?.t())?)?;
        }
        let mixed = mlx_rs::nn::silu(conv.as_dtype(dt)?)?.as_dtype(Dtype::Float32)?;
        let (kh, vh, kdim, vdim) = (dim(c.linear_key_heads)?, dim(c.linear_value_heads)?, dim(c.linear_key_dim)?, dim(c.linear_value_dim)?);
        let q = cols(&mixed, 0, kd)?.reshape(&[s, kh, kdim])?;
        let k = cols(&mixed, kd, 2 * kd)?.reshape(&[s, kh, kdim])?;
        let v = cols(&mixed, 2 * kd, 2 * kd + vd)?.reshape(&[s, vh, vdim])?;
        let beta = ops::sigmoid(d.b.forward(x)?)?.as_dtype(Dtype::Float32)?; // (S, VH)
        let a = d.a.forward(x)?.as_dtype(Dtype::Float32)?.add(&d.dt_bias)?;
        let g = mlx_rs::nn::softplus(&a)?.multiply(&d.decay)?; // (S, VH), <= 0
        // Each key head serves `rep` consecutive value heads.
        let rep = vh / kh;
        let widen = |t: Array| -> Result<Array> { Ok(ops::broadcast_to(t.expand_dims(2)?, &[s, kh, rep, kdim])?.reshape(&[s, vh, kdim])?) };
        let (q, k) = (widen(l2norm(&q)?)?, widen(l2norm(&k)?)?);
        let out = gated_delta_rule(&q, &k, &v, &g, &beta)?; // (S, VH, VD) f32
        // The gated RMSNorm per head (plain weight) times SiLU of `z`, with qir's roundings.
        let out = out.as_dtype(dt)?.as_dtype(Dtype::Float32)?;
        let eps = Array::from_f32(c.eps as f32);
        let rms = out.square()?.mean_axis(-1, true)?.add(&eps)?.sqrt()?;
        let normed = out.divide(&rms)?.as_dtype(dt)?.as_dtype(Dtype::Float32)?.multiply(&d.norm)?;
        let z = d.z.forward(x)?.as_dtype(Dtype::Float32)?.reshape(&[s, vh, vdim])?;
        let gated = normed.multiply(mlx_rs::nn::silu(&z)?)?.as_dtype(dt)?.reshape(&[s, vh * vdim])?;
        d.out.forward(&gated)
    }
}

/// `x / sqrt(Σx² + 1e-6)` over the last axis.
fn l2norm(x: &Array) -> Result<Array> {
    Ok(x.divide(x.square()?.sum_axis(-1, true)?.add(Array::from_f32(1e-6))?.sqrt()?)?)
}

/// The gated delta rule from a zero state, token by token: `q`, `k` `(S, H, Dk)`
/// (l2-normalised), `v` `(S, H, Dv)`, `g` (log decay) and `beta` `(S, H)`, all f32. Per head
/// `M ← e^g·M + k (β(v − kᵀ e^g M))ᵀ`, each output `qᵀ M / √Dk`. Returns `(S, H, Dv)`.
// ponytail: the plain recurrence (S small ops per layer, fine for a prompt); qir's chunked
// form if long prompts ever make the encode slow.
pub fn gated_delta_rule(q: &Array, k: &Array, v: &Array, g: &Array, beta: &Array) -> Result<Array> {
    let qs = q.shape();
    let (s, h, dk) = (qs[0], qs[1], qs[2]);
    let dv = v.shape()[2];
    let scale = Array::from_f32(1.0 / (dk as f32).sqrt());
    let mut m = Array::zeros::<f32>(&[h, dk, dv])?;
    let mut outs = Vec::with_capacity(usize::try_from(s)?);
    for t in 0..s {
        let (q_t, k_t, v_t) = (q.try_index(t)?, k.try_index(t)?, v.try_index(t)?); // (H, ·)
        let e = g.try_index(t)?.exp()?.reshape(&[h, 1, 1])?;
        m = m.multiply(&e)?;
        let kv_mem = k_t.expand_dims(1)?.matmul(&m)?.squeeze_axes(&[1])?; // (H, Dv)
        let delta = v_t.subtract(&kv_mem)?.multiply(beta.try_index(t)?.expand_dims(1)?)?;
        m = m.add(k_t.expand_dims(2)?.multiply(delta.expand_dims(1)?)?)?;
        outs.push(q_t.multiply(&scale)?.expand_dims(1)?.matmul(&m)?.squeeze_axes(&[1])?);
    }
    Ok(ops::stack(&outs, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{compare, to_vec};

    /// The recurrence against qir's chunked form over a length that leaves a partial chunk.
    #[test]
    fn delta_rule_matches_qirs_chunked_form() -> Result<()> {
        use candle_core::{Device, Tensor};
        let (s, h, dk, dv) = (70usize, 2usize, 8usize, 4usize);
        let wave = |n: usize, f: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * f).sin() * 0.7 + (i as f32 * 0.013).cos() * 0.3).collect() };
        let dev = Device::Cpu;
        let unit = |t: Tensor| -> Result<Tensor> { Ok(t.broadcast_div(&(t.sqr()?.sum_keepdim(2)? + 1e-6)?.sqrt()?)?) };
        let q = unit(Tensor::from_vec(wave(s * h * dk, 0.37), (s, h, dk), &dev)?)?;
        let k = unit(Tensor::from_vec(wave(s * h * dk, 0.91), (s, h, dk), &dev)?)?;
        let v = Tensor::from_vec(wave(s * h * dv, 0.53), (s, h, dv), &dev)?;
        let g = (Tensor::from_vec(wave(s * h, 1.7), (s, h), &dev)?.abs()? * -0.2)?;
        let beta = Tensor::from_vec(wave(s * h, 2.3).iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(), (s, h), &dev)?;
        let want = qwen_image_rs::moss::text::chunk_gated_delta_rule(&q, &k, &v, &g, &beta)?.flatten_all()?.to_vec1::<f32>()?;
        let got = to_vec(&gated_delta_rule(&from_candle(&q)?, &from_candle(&k)?, &from_candle(&v)?, &from_candle(&g)?, &from_candle(&beta)?)?)?;
        let (cos, max) = compare(&got, &want)?;
        assert!(max < 1e-4 && cos > 0.99999, "cos {cos}, max {max}");
        Ok(())
    }
}
