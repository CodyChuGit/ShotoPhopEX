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

    /// Every layer of [`convert_streamed`]'s `dir` held (about 5.2 GB at 6-bit), for
    /// [`Self::generate`], which runs each layer once a token.
    pub fn resident(cfg: &TextConfig, embed: Prequantized, dir: &Path, dtype: Dtype) -> Result<Self> {
        let base = crate::convert::load(&dir.join("base.safetensors"))?;
        let norm = Norm1p::load(&base, &format!("{P}.norm.weight"))?;
        let layers =
            (0..cfg.layers).map(|i| Layer::load(&crate::convert::load(&dir.join(format!("layer-{i}.safetensors")))?, cfg, i)).collect::<Result<_>>()?;
        Ok(Self { cfg: cfg.clone(), dtype, embed, layers, norm, stream: None })
    }

    /// The last hidden state `(tokens, hidden)` f32.
    pub fn forward(&self, ids: &[u32]) -> Result<Array> {
        self.run(ids, 0, None)
    }

    /// Greedy decoding after `ids` (a chat already templated): at most `max_new` tokens, ending
    /// at the first of `stop` (not returned). `head` is the language-model head ([`lm_head`]).
    /// The layers must be held ([`Self::resident`]).
    pub fn generate(&self, head: &Linear, ids: &[u32], max_new: usize, stop: &[u32], cancel: &dyn Fn() -> bool) -> Result<Vec<u32>> {
        anyhow::ensure!(self.stream.is_none(), "generation needs the layers held (Qwen35Text::resident)");
        let mut states: Vec<State> = (0..self.cfg.layers).map(|_| State::default()).collect();
        let mut out = Vec::new();
        let (mut next, mut pos) = (ids.to_vec(), 0);
        let t_all = std::time::Instant::now();
        let mut prefill_ms = 0.0;
        while out.len() < max_new {
            if cancel() {
                return Err(qwen_image_rs::pipeline::Cancelled.into());
            }
            let h = self.run(&next, pos, Some(&mut states))?;
            pos += next.len();
            let last = h.try_index(h.shape()[0] - 1..)?.as_dtype(self.dtype)?;
            let token = ops::indexing::argmax_axis(&head.forward(&last)?, -1, None)?.as_dtype(Dtype::Uint32)?;
            token.eval()?;
            if pos == ids.len() {
                prefill_ms = t_all.elapsed().as_secs_f64() * 1000.0;
            }
            let t = token.as_slice::<u32>().first().copied().context("no token")?;
            if stop.contains(&t) {
                break;
            }
            out.push(t);
            next = vec![t];
        }
        if crate::profile::enabled() {
            let decode = t_all.elapsed().as_secs_f64() * 1000.0 - prefill_ms;
            eprintln!(
                "profile: chat prefill {} tokens {prefill_ms:.0} ms; decode {} tokens {decode:.0} ms ({:.0} ms/token)",
                ids.len(),
                out.len(),
                decode / out.len().max(1) as f64
            );
        }
        Ok(out)
    }

    /// The layers over `ids` at positions `start..`, each layer's state carried in `states`
    /// when given (decoding), the final norm applied.
    fn run(&self, ids: &[u32], start: usize, mut states: Option<&mut [State]>) -> Result<Array> {
        anyhow::ensure!(!ids.is_empty(), "no tokens to encode");
        let eps = self.cfg.eps as f32;
        let dt = self.dtype;
        let mut x = from_candle(&self.embed.gather_cpu(ids)?)?;
        let (cos, sin) = self.rope(start, ids.len())?;
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
            let state = states.as_deref_mut().and_then(|s| s.get_mut(i));
            let h = match &layer.mixer {
                Mixer::Full(a) => self.full_attention(a, &h, &cos, &sin, state)?,
                Mixer::Linear(d) => self.delta_net(d, &h, state)?,
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

    /// cos/sin `(tokens, 1, rotary_dim)` f32 for positions `start..start + n` (each half
    /// repeated).
    fn rope(&self, start: usize, n: usize) -> Result<(Array, Array)> {
        let r = self.cfg.rotary_dim;
        let half = r / 2;
        let inv: Vec<f32> = (0..half).map(|j| (1.0 / self.cfg.rope_theta.powf((2 * j) as f64 / r as f64)) as f32).collect();
        let mut angles = Vec::with_capacity(n * r);
        for p in start..start + n {
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

    fn full_attention(&self, a: &GatedAttention, x: &Array, cos: &Array, sin: &Array, state: Option<&mut State>) -> Result<Array> {
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
        let (mut k, mut v) = (heads_first(&k)?, heads_first(&v)?);
        let mut past = false;
        if let Some(st) = state {
            if let Some((pk, pv)) = &st.kv {
                k = ops::concatenate(&[pk.clone(), k], 2)?;
                v = ops::concatenate(&[pv.clone(), v], 2)?;
                past = true;
            }
            st.kv = Some((k.clone(), v.clone()));
        }
        // A prompt is causal from position 0; a token after it sees everything before.
        anyhow::ensure!(!past || s == 1, "a decode step is one token");
        let mask = if past { None } else { Some(fast::ScaledDotProductAttentionMask::Causal) };
        let o = match mask {
            Some(m) => fast::scaled_dot_product_attention(heads_first(&q)?, &k, &v, scale, m, None)?,
            None => fast::scaled_dot_product_attention(heads_first(&q)?, &k, &v, scale, None, None)?,
        };
        let o = o.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[s, h * hd])?.as_dtype(self.dtype)?;
        a.o.forward(&o.multiply(ops::sigmoid(&gate)?)?)
    }

    fn delta_net(&self, d: &DeltaNet, x: &Array, mut state: Option<&mut State>) -> Result<Array> {
        let c = &self.cfg;
        let dt = self.dtype;
        let s = x.shape()[0];
        let (kd, vd) = (c.linear_key_heads * c.linear_key_dim, c.linear_value_heads * c.linear_value_dim);
        // The causal depthwise conv (left zero padding), then SiLU.
        let proj = d.qkv.forward(x)?.as_dtype(Dtype::Float32)?; // (S, conv_dim)
        let k_len = c.conv_kernel;
        let width = proj.shape()[1];
        let tail = match state.as_deref().and_then(|st| st.conv.clone()) {
            Some(t) => t,
            None => Array::zeros::<f32>(&[dim(k_len - 1)?, width])?,
        };
        let padded = ops::concatenate(&[tail, proj], 0)?;
        if let Some(st) = state.as_deref_mut() {
            st.conv = Some(padded.try_index(s..)?);
        }
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
        let m0 = state.as_deref().and_then(|st| st.m.clone());
        let (out, m) = gated_delta_rule_from(m0, &q, &k, &v, &g, &beta)?; // (S, VH, VD) f32
        if let Some(st) = state {
            st.m = Some(m);
        }
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
    Ok(gated_delta_rule_from(None, q, k, v, g, beta)?.0)
}

/// [`gated_delta_rule`] from the state `m0` `(H, Dk, Dv)` (zero when `None`); also returns the
/// state after the last token.
fn gated_delta_rule_from(m0: Option<Array>, q: &Array, k: &Array, v: &Array, g: &Array, beta: &Array) -> Result<(Array, Array)> {
    let qs = q.shape();
    let (s, h, dk) = (qs[0], qs[1], qs[2]);
    let dv = v.shape()[2];
    let scale = Array::from_f32(1.0 / (dk as f32).sqrt());
    let mut m = match m0 {
        Some(m) => m,
        None => Array::zeros::<f32>(&[h, dk, dv])?,
    };
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
    Ok((ops::stack(&outs, 0)?, m))
}

/// One layer's decoding state: the attention's keys and values `(1, KVH, T, D)` so far, or the
/// DeltaNet's last `kernel - 1` projected rows and its recurrent state.
#[derive(Default)]
struct State {
    kv: Option<(Array, Array)>,
    conv: Option<Array>,
    m: Option<Array>,
}

/// The language-model head (`lm_head`, 248320 × 4096) converted in `precision`, cached at
/// `path`: what turns the last hidden state into the next token.
pub fn lm_head(text: &Path, dtype: Dtype, precision: Precision, path: &Path) -> Result<Linear> {
    if !path.is_file() {
        let w = Weights::open(text, &candle_core::Device::Cpu, candle_core::DType::F32, false)?;
        let mut c = Converter::new(&w, dtype, precision)?;
        c.linear("lm_head", true, None).context("the encoder file's lm_head")?;
        crate::convert::save(&c.out, path)?;
    }
    Linear::load(&crate::convert::load(path)?, "lm_head")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{compare, to_vec};

    /// Decoding carries the state: two halves from the state between them equal one pass.
    #[test]
    fn delta_rule_state_carries_across_calls() -> Result<()> {
        let (s, h, dk, dv) = (12i32, 2i32, 4i32, 3i32);
        let wave = |n: i32, f: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * f).sin() * 0.5).collect() };
        let q = l2norm(&Array::from_slice(&wave(s * h * dk, 0.37), &[s, h, dk]))?;
        let k = l2norm(&Array::from_slice(&wave(s * h * dk, 0.91), &[s, h, dk]))?;
        let v = Array::from_slice(&wave(s * h * dv, 0.53), &[s, h, dv]);
        let g = Array::from_slice(&wave(s * h, 1.7), &[s, h]).abs()?.multiply(Array::from_f32(-0.2))?;
        let beta = ops::sigmoid(&Array::from_slice(&wave(s * h, 2.3), &[s, h]))?;
        let half = |t: &Array, a: i32, b: i32| t.try_index(a..b);
        let (whole, _) = gated_delta_rule_from(None, &q, &k, &v, &g, &beta)?;
        let (first, m) = gated_delta_rule_from(None, &half(&q, 0, 7)?, &half(&k, 0, 7)?, &half(&v, 0, 7)?, &half(&g, 0, 7)?, &half(&beta, 0, 7)?)?;
        let (second, _) = gated_delta_rule_from(Some(m), &half(&q, 7, s)?, &half(&k, 7, s)?, &half(&v, 7, s)?, &half(&g, 7, s)?, &half(&beta, 7, s)?)?;
        let (cos, max) = compare(&to_vec(&ops::concatenate(&[first, second], 0)?)?, &to_vec(&whole)?)?;
        assert!(max < 1e-5 && cos > 0.99999, "cos {cos}, max {max}");
        Ok(())
    }

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
