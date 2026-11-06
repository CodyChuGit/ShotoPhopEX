//! MOSS-Image's text encoder: the language model of Qwen3.5-9B, its last hidden state (after the
//! final norm) for the prompt in its chat template with thinking on. A hybrid decoder: 32 layers,
//! every fourth one full attention (gated output, partial RoPE), the other 24 Gated DeltaNet
//! linear attention (a causal depthwise conv, then the gated delta rule over the sequence). A
//! clean-room port of the architecture as Hugging Face transformers' `Qwen3_5TextModel` and
//! ComfyUI's `qwen35.py` describe it, checked against the former on the same weights.

use candle_core::{DType, Device, Tensor, D};

use super::weights::{QLinear, Weights};
use crate::comfy_int8::Prequantized;
use crate::Result;

/// The prompt as the pipeline hands it to the tokenizer (`apply_chat_template` with one user
/// turn, the generation prompt and `enable_thinking=True`).
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
}

/// Qwen3.5-9B's language model.
#[derive(Clone, Debug)]
pub struct TextConfig {
    pub hidden: usize,
    pub layers: usize,
    pub full_attention_interval: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_key_dim: usize,
    pub linear_value_dim: usize,
    pub conv_kernel: usize,
    pub eps: f64,
}

impl TextConfig {
    pub fn qwen35_9b() -> Self {
        Self {
            hidden: 4096,
            layers: 32,
            full_attention_interval: 4,
            heads: 16,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 10_000_000.0,
            linear_key_heads: 16,
            linear_value_heads: 32,
            linear_key_dim: 128,
            linear_value_dim: 128,
            conv_kernel: 4,
            eps: 1e-6,
        }
    }

    fn is_full_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }
}

/// RMSNorm with the zero-centred `(1 + weight)` scale, computed in f32.
struct Norm1p {
    weight: Tensor, // f32, 1 + w
    eps: f64,
}

impl Norm1p {
    fn load(w: &Weights, name: &str, eps: f64) -> Result<Self> {
        Ok(Self { weight: (w.get_as(name, DType::F32)? + 1.0)?, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x32 = x.to_dtype(DType::F32)?;
        let rms = (x32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        Ok(x32.broadcast_div(&rms)?.broadcast_mul(&self.weight)?.to_dtype(x.dtype())?)
    }
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        self.down.forward(&h)
    }
}

/// Full attention with a sigmoid gate on its output (`q_proj` gives each head its query and
/// its gate side by side), per-head q/k norms and RoPE on the first `rotary_dim` channels.
struct GatedAttention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    q_norm: Norm1p,
    k_norm: Norm1p,
}

/// The Gated DeltaNet mixer.
struct DeltaNet {
    qkv: QLinear,
    z: QLinear,
    b: QLinear,
    a: QLinear,
    out: QLinear,
    /// `(conv_dim, kernel)` f32.
    conv: Tensor,
    /// `-exp(A_log)`, `(value_heads,)` f32.
    decay: Tensor,
    dt_bias: Tensor,
    /// The gated norm's weight, `(value_dim,)` f32 (a plain scale).
    norm: Tensor,
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
    embed: Prequantized,
    layers: Vec<Layer>,
    norm: Norm1p,
    dev: Device,
    dtype: DType,
}

impl Qwen35Text {
    /// The language model from ComfyUI's `qwen3.5_9b_*` file (keys `model.language_model.*`;
    /// the vision tower, `lm_head` and the MTP head are not read).
    pub fn load(cfg: &TextConfig, w: &Weights) -> Result<Self> {
        let p = "model.language_model";
        anyhow::ensure!(w.is_quantized(&format!("{p}.embed_tokens")), "the encoder's embedding table is not an INT8 ConvRot triple");
        let embed = w.prequantized(&format!("{p}.embed_tokens"))?;
        anyhow::ensure!(embed.in_features == cfg.hidden, "embedding width {} (expected {})", embed.in_features, cfg.hidden);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = format!("{p}.layers.{i}");
            let mixer = if cfg.is_full_attention(i) {
                let a = format!("{l}.self_attn");
                Mixer::Full(GatedAttention {
                    q: w.linear(&format!("{a}.q_proj"))?,
                    k: w.linear(&format!("{a}.k_proj"))?,
                    v: w.linear(&format!("{a}.v_proj"))?,
                    o: w.linear(&format!("{a}.o_proj"))?,
                    q_norm: Norm1p::load(w, &format!("{a}.q_norm.weight"), cfg.eps)?,
                    k_norm: Norm1p::load(w, &format!("{a}.k_norm.weight"), cfg.eps)?,
                })
            } else {
                let a = format!("{l}.linear_attn");
                let conv = w.get_as(&format!("{a}.conv1d.weight"), DType::F32)?;
                let conv = conv.reshape((conv.dim(0)?, cfg.conv_kernel))?;
                Mixer::Linear(DeltaNet {
                    qkv: w.linear(&format!("{a}.in_proj_qkv"))?,
                    z: w.linear(&format!("{a}.in_proj_z"))?,
                    b: w.linear(&format!("{a}.in_proj_b"))?,
                    a: w.linear(&format!("{a}.in_proj_a"))?,
                    out: w.linear(&format!("{a}.out_proj"))?,
                    conv,
                    decay: w.get_as(&format!("{a}.A_log"), DType::F32)?.exp()?.neg()?,
                    dt_bias: w.get_as(&format!("{a}.dt_bias"), DType::F32)?,
                    norm: w.get_as(&format!("{a}.norm.weight"), DType::F32)?,
                })
            };
            layers.push(Layer {
                input_norm: Norm1p::load(w, &format!("{l}.input_layernorm.weight"), cfg.eps)?,
                post_norm: Norm1p::load(w, &format!("{l}.post_attention_layernorm.weight"), cfg.eps)?,
                mixer,
                mlp: Mlp {
                    gate: w.linear(&format!("{l}.mlp.gate_proj"))?,
                    up: w.linear(&format!("{l}.mlp.up_proj"))?,
                    down: w.linear(&format!("{l}.mlp.down_proj"))?,
                },
            });
        }
        Ok(Self { cfg: cfg.clone(), embed, layers, norm: Norm1p::load(w, &format!("{p}.norm.weight"), cfg.eps)?, dev: w.device().clone(), dtype: w.dtype() })
    }

    /// The last hidden state `(tokens, hidden)` in the run dtype.
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(!ids.is_empty(), "no tokens to encode");
        let mut x = self.embed.gather_cpu(ids)?.to_dtype(self.dtype)?.to_device(&self.dev)?;
        let (cos, sin) = self.rope(ids.len())?;
        for layer in &self.layers {
            let h = layer.input_norm.forward(&x)?;
            let h = match &layer.mixer {
                Mixer::Full(a) => self.full_attention(a, &h, &cos, &sin)?,
                Mixer::Linear(d) => self.delta_net(d, &h)?,
            };
            x = (x + h)?;
            let h = layer.mlp.forward(&layer.post_norm.forward(&x)?)?;
            x = (x + h)?;
        }
        self.norm.forward(&x)
    }

    /// cos/sin `(tokens, rotary_dim)` f32 for positions `0..n` (text-only: Qwen3.5's three
    /// interleaved M-RoPE sections see one position, the plain rotation).
    fn rope(&self, n: usize) -> Result<(Tensor, Tensor)> {
        let half = self.cfg.rotary_dim / 2;
        let inv: Vec<f32> = (0..half).map(|j| (1.0 / self.cfg.rope_theta.powf((2 * j) as f64 / self.cfg.rotary_dim as f64)) as f32).collect();
        let mut angles = Vec::with_capacity(n * self.cfg.rotary_dim);
        for p in 0..n {
            for _ in 0..2 {
                angles.extend(inv.iter().map(|f| p as f32 * f));
            }
        }
        let a = Tensor::from_vec(angles, (n, self.cfg.rotary_dim), &self.dev)?;
        Ok((a.cos()?, a.sin()?))
    }

    fn full_attention(&self, a: &GatedAttention, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let s = x.dim(0)?;
        let qg = a.q.forward(x)?.reshape((s, c.heads, 2 * c.head_dim))?;
        let q = a.q_norm.forward(&qg.narrow(2, 0, c.head_dim)?.contiguous()?)?;
        let gate = qg.narrow(2, c.head_dim, c.head_dim)?.contiguous()?.reshape((s, c.heads * c.head_dim))?;
        let k = a.k_norm.forward(&a.k.forward(x)?.reshape((s, c.kv_heads, c.head_dim))?)?;
        let v = a.v.forward(x)?.reshape((s, c.kv_heads, c.head_dim))?;
        let q = partial_rope(&q, cos, sin, c.rotary_dim)?;
        let k = partial_rope(&k, cos, sin, c.rotary_dim)?;
        // (H, S, D) in f32, the KV heads repeated for their query groups; causal.
        let rep = c.heads / c.kv_heads;
        let q = q.transpose(0, 1)?.to_dtype(DType::F32)?.contiguous()?;
        let expand = |t: Tensor| -> Result<Tensor> {
            let t = t.transpose(0, 1)?.to_dtype(DType::F32)?; // (KVH, S, D)
            Ok(t.unsqueeze(1)?.expand((c.kv_heads, rep, s, c.head_dim))?.reshape((c.heads, s, c.head_dim))?.contiguous()?)
        };
        let (k, v) = (expand(k)?, expand(v)?);
        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * (1.0 / (c.head_dim as f64).sqrt()))?;
        let mask: Vec<f32> = (0..s).flat_map(|i| (0..s).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })).collect();
        let scores = scores.broadcast_add(&Tensor::from_vec(mask, (1, s, s), &self.dev)?)?;
        let out = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?; // (H, S, D)
        let out = out.transpose(0, 1)?.reshape((s, c.heads * c.head_dim))?.to_dtype(self.dtype)?;
        let out = (out * candle_nn::ops::sigmoid(&gate)?)?;
        a.o.forward(&out)
    }

    fn delta_net(&self, d: &DeltaNet, x: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let s = x.dim(0)?;
        let (kd, vd) = (c.linear_key_heads * c.linear_key_dim, c.linear_value_heads * c.linear_value_dim);
        // The causal depthwise conv (kernel `K`, left zero padding), then SiLU.
        let proj = d.qkv.forward(x)?.to_dtype(DType::F32)?; // (S, conv_dim)
        let k_len = c.conv_kernel;
        let padded = Tensor::cat(&[Tensor::zeros((k_len - 1, proj.dim(1)?), DType::F32, &self.dev)?, proj], 0)?;
        let mut conv = padded.narrow(0, 0, s)?.broadcast_mul(&d.conv.narrow(1, 0, 1)?.t()?)?;
        for j in 1..k_len {
            conv = (conv + padded.narrow(0, j, s)?.broadcast_mul(&d.conv.narrow(1, j, 1)?.t()?)?)?;
        }
        let mixed = candle_nn::ops::silu(&conv.to_dtype(self.dtype)?)?.to_dtype(DType::F32)?;
        let q = mixed.narrow(1, 0, kd)?.reshape((s, c.linear_key_heads, c.linear_key_dim))?;
        let k = mixed.narrow(1, kd, kd)?.reshape((s, c.linear_key_heads, c.linear_key_dim))?;
        let v = mixed.narrow(1, 2 * kd, vd)?.reshape((s, c.linear_value_heads, c.linear_value_dim))?;
        let beta = candle_nn::ops::sigmoid(&d.b.forward(x)?)?.to_dtype(DType::F32)?; // (S, VH)
        let a = d.a.forward(x)?.to_dtype(DType::F32)?.broadcast_add(&d.dt_bias)?;
        let g = softplus(&a)?.broadcast_mul(&d.decay)?; // (S, VH), <= 0
                                                        // Each key head serves `rep` value heads (repeat_interleave on the head axis).
        let rep = c.linear_value_heads / c.linear_key_heads;
        let widen = |t: Tensor| -> Result<Tensor> {
            Ok(t.unsqueeze(2)?.expand((s, c.linear_key_heads, rep, c.linear_key_dim))?.reshape((s, c.linear_value_heads, c.linear_key_dim))?)
        };
        let (q, k) = (widen(l2norm(&q)?)?, widen(l2norm(&k)?)?);
        let out = chunk_gated_delta_rule(&q, &k, &v, &g, &beta)?; // (S, VH, VD) f32
                                                                  // The gated RMSNorm per head: plain weight, times SiLU of `z`.
        let out = out.to_dtype(self.dtype)?.to_dtype(DType::F32)?;
        let rms = (out.sqr()?.mean_keepdim(D::Minus1)? + c.eps)?.sqrt()?;
        let normed = out.broadcast_div(&rms)?.to_dtype(self.dtype)?.to_dtype(DType::F32)?.broadcast_mul(&d.norm)?;
        let z = d.z.forward(x)?.to_dtype(DType::F32)?.reshape((s, c.linear_value_heads, c.linear_value_dim))?;
        let gated = (normed * candle_nn::ops::silu(&z)?)?.to_dtype(self.dtype)?.reshape((s, vd))?;
        d.out.forward(&gated)
    }
}

/// `x[..., :r]` rotated by RoPE (`x·cos + rotate_half(x)·sin`), the rest passed through;
/// `x (S, H, D)`, `cos`/`sin` `(S, r)` f32.
fn partial_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, r: usize) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let dt = x.dtype();
    let rot = x.narrow(2, 0, r)?.to_dtype(DType::F32)?;
    let half = r / 2;
    let rotated = Tensor::cat(&[rot.narrow(2, half, half)?.neg()?, rot.narrow(2, 0, half)?], 2)?;
    let (cos, sin) = (cos.unsqueeze(1)?, sin.unsqueeze(1)?);
    let y = (rot.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?.to_dtype(dt)?;
    Ok(Tensor::cat(&[y, x.narrow(2, r, d - r)?], 2)?)
}

/// `log(1 + exp(x))`, linear above 20 (torch's threshold).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let soft = (x.clamp(f32::MIN, 20f32)?.exp()? + 1.0)?.log()?;
    let big = x.ge(20f64)?;
    Ok(big.where_cond(x, &soft)?)
}

/// `x / sqrt(Σx² + 1e-6)` over the last axis (the kernel's `use_qk_l2norm_in_kernel`).
fn l2norm(x: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?)
}

/// Chunk size of the delta rule's parallel form.
const CHUNK: usize = 64;

/// The gated delta rule over the sequence from a zero state, in its chunked form: `q`, `k`
/// `(S, H, Dk)` (l2-normalised), `v` `(S, H, Dv)`, `g` (log decay) and `beta` `(S, H)`, all
/// f32. Per head the state `M (Dk, Dv)` follows `M ← e^g·M + k (β(v − kᵀ e^g M))ᵀ` and each output
/// is `qᵀ M / √Dk`. Returns `(S, H, Dv)`.
pub fn chunk_gated_delta_rule(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let (s, h, dk) = q.dims3()?;
    let dv = v.dim(2)?;
    let dev = q.device();
    let n = s.div_ceil(CHUNK);
    let pad = n * CHUNK - s;
    // (H, S, ·), padded with zeros to whole chunks, then (H, N, C, ·).
    let heads_first = |t: &Tensor| -> Result<Tensor> {
        let t = t.transpose(0, 1)?.contiguous()?;
        let t = if pad > 0 {
            let mut shape = t.dims().to_vec();
            shape[1] = pad;
            Tensor::cat(&[t.clone(), Tensor::zeros(shape, DType::F32, dev)?], 1)?
        } else {
            t
        };
        Ok(t)
    };
    let scale = 1.0 / (dk as f64).sqrt();
    let q = (heads_first(q)? * scale)?.reshape((h, n, CHUNK, dk))?;
    let k = heads_first(k)?.reshape((h, n, CHUNK, dk))?;
    let v = heads_first(v)?.reshape((h, n, CHUNK, dv))?;
    let beta = heads_first(beta)?.reshape((h, n, CHUNK, 1))?;
    let g = heads_first(g)?.reshape((h, n, CHUNK))?.cumsum(2)?; // (H, N, C)
    let k_beta = k.broadcast_mul(&beta)?;
    let v_beta = v.broadcast_mul(&beta)?;
    // decay[i, j] = exp(g_i − g_j) for j ≤ i, 0 above the diagonal.
    let lower = Tensor::tril2(CHUNK, DType::F32, dev)?; // 1 on and below the diagonal
    let diff = g.unsqueeze(3)?.broadcast_sub(&g.unsqueeze(2)?)?;
    let decay = (diff.broadcast_mul(&lower)?.exp()?.broadcast_mul(&lower))?; // (H, N, C, C)
    let strict = (&lower - Tensor::eye(CHUNK, DType::F32, dev)?)?;
    // A = −(β k kᵀ ⊙ decay) strictly below the diagonal; T = (I − A)⁻¹ by forward substitution.
    let a = k_beta.matmul(&k.transpose(2, 3)?.contiguous()?)?.mul(&decay)?.broadcast_mul(&strict)?.neg()?;
    let t = unit_lower_inverse(&a)?;
    let value = t.matmul(&v_beta)?; // (H, N, C, Dv)
    let k_cumdecay = t.matmul(&k_beta.broadcast_mul(&g.exp()?.unsqueeze(3)?)?)?; // (H, N, C, Dk)
    let mut state = Tensor::zeros((h, dk, dv), DType::F32, dev)?;
    let mut outs = Vec::with_capacity(n);
    for i in 0..n {
        let (q_i, k_i, v_i) = (q.narrow(1, i, 1)?.squeeze(1)?, k.narrow(1, i, 1)?.squeeze(1)?, value.narrow(1, i, 1)?.squeeze(1)?);
        let g_i = g.narrow(1, i, 1)?.squeeze(1)?; // (H, C)
        let attn = q_i.matmul(&k_i.transpose(1, 2)?.contiguous()?)?.mul(&decay.narrow(1, i, 1)?.squeeze(1)?)?.broadcast_mul(&lower)?;
        let v_new = (v_i - k_cumdecay.narrow(1, i, 1)?.squeeze(1)?.matmul(&state)?)?;
        let inter = q_i.broadcast_mul(&g_i.exp()?.unsqueeze(2)?)?.matmul(&state)?;
        outs.push((inter + attn.matmul(&v_new)?)?);
        let g_last = g_i.narrow(1, CHUNK - 1, 1)?; // (H, 1)
        let k_w = k_i.broadcast_mul(&g_last.broadcast_sub(&g_i)?.exp()?.unsqueeze(2)?)?;
        state = (state.broadcast_mul(&g_last.exp()?.unsqueeze(2)?)? + k_w.transpose(1, 2)?.contiguous()?.matmul(&v_new)?)?;
    }
    let out = Tensor::cat(&outs, 1)?.narrow(1, 0, s)?; // (H, S, Dv)
    Ok(out.transpose(0, 1)?.contiguous()?)
}

/// `(I − A)⁻¹` for each strictly lower triangular `A (…, C, C)`: the unit lower triangular `T`
/// with `T[i, j] = δᵢⱼ + Σₖ A[i, k] T[k, j]`, row by row on the CPU in f64.
fn unit_lower_inverse(a: &Tensor) -> Result<Tensor> {
    let dims = a.dims().to_vec();
    let c = dims[dims.len() - 1];
    let src = a.flatten_all()?.to_dtype(DType::F64)?.to_device(&Device::Cpu)?.to_vec1::<f64>()?;
    let mut out = vec![0f64; src.len()];
    for (m, t) in src.chunks(c * c).zip(out.chunks_mut(c * c)) {
        for i in 0..c {
            t[i * c + i] = 1.0;
            for kk in 0..i {
                let aik = m[i * c + kk];
                if aik != 0.0 {
                    for j in 0..=kk {
                        t[i * c + j] += aik * t[kk * c + j];
                    }
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, dims, &Device::Cpu)?.to_dtype(DType::F32)?.to_device(a.device())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chunked rule against its recurrence, over a length that leaves a partial chunk.
    #[test]
    fn chunked_delta_rule_matches_the_recurrence() -> Result<()> {
        let dev = Device::Cpu;
        let (s, h, dk, dv) = (70, 2, 8, 4);
        let wave = |n: usize, f: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * f).sin() * 0.7 + (i as f32 * 0.013).cos() * 0.3).collect() };
        let q = l2norm(&Tensor::from_vec(wave(s * h * dk, 0.37), (s, h, dk), &dev)?)?;
        let k = l2norm(&Tensor::from_vec(wave(s * h * dk, 0.91), (s, h, dk), &dev)?)?;
        let v = Tensor::from_vec(wave(s * h * dv, 0.53), (s, h, dv), &dev)?;
        let g = (Tensor::from_vec(wave(s * h, 1.7), (s, h), &dev)?.abs()? * -0.2)?;
        let beta = candle_nn::ops::sigmoid(&Tensor::from_vec(wave(s * h, 2.3), (s, h), &dev)?)?;
        let fast = chunk_gated_delta_rule(&q, &k, &v, &g, &beta)?;

        let (qv, kv, vv) = (q.to_vec3::<f32>()?, k.to_vec3::<f32>()?, v.to_vec3::<f32>()?);
        let (gv, bv) = (g.to_vec2::<f32>()?, beta.to_vec2::<f32>()?);
        let scale = 1.0 / (dk as f32).sqrt();
        let mut worst = 0f32;
        let fast = fast.to_vec3::<f32>()?;
        for hh in 0..h {
            let mut m = vec![vec![0f32; dv]; dk];
            for t in 0..s {
                let e = gv[t][hh].exp();
                m.iter_mut().flatten().for_each(|x| *x *= e);
                let kv_mem: Vec<f32> = (0..dv).map(|j| (0..dk).map(|i| kv[t][hh][i] * m[i][j]).sum()).collect();
                let delta: Vec<f32> = (0..dv).map(|j| (vv[t][hh][j] - kv_mem[j]) * bv[t][hh]).collect();
                for i in 0..dk {
                    for j in 0..dv {
                        m[i][j] += kv[t][hh][i] * delta[j];
                    }
                }
                for j in 0..dv {
                    let o: f32 = (0..dk).map(|i| qv[t][hh][i] * scale * m[i][j]).sum();
                    worst = worst.max((o - fast[t][hh][j]).abs());
                }
            }
        }
        assert!(worst < 1e-4, "chunked vs recurrent: max abs {worst}");
        Ok(())
    }
}
