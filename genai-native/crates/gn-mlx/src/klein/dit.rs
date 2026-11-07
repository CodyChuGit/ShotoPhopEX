//! FLUX 3 Image Klein's DiT on MLX, as Black Forest Labs' reference (`flux3/dit.py`) and
//! mflux's port (`flux3_transformer.py`) run it. Five blocks for each kind of token (the picture
//! being made, the reference pictures and the text), each kind attending only to itself, then 28
//! single-stream blocks over all of them. Every block uses one shared modulation per kind of
//! token, set by the time (the picture's own, 0 for the references and the text) plus, in the
//! distilled flow map, the time the step jumps to.
//!
//! The weights are mflux's files (`mflux-community/flux-3-image-klein-mflux-q4` and `-q8`):
//! diffusers' names, every linear MLX-quantized (group 64). The residual stream stays in f32,
//! and each linear sees its input in the run dtype. In f16, V is taken 4 times smaller and the
//! feed-forward's up half 8 times smaller, and their outputs are scaled back in f32. This is
//! mflux's float16 headroom: in the bf16 reference those outputs reach 3e4 and 3e5.

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype, fast, ops};

use super::lora_names;
use crate::convert::Lora;
use crate::nn::{Linear, Params, cols, dim, get, rows, shape};

pub const HIDDEN: usize = 3072;
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const DEPTH: usize = 5;
pub const DEPTH_SINGLE: usize = 28;
pub const CHANNELS: usize = 128;
pub const TEXT_DIM: usize = 20480;
/// RoPE's four axes `(t, row, column, layer)`, 32 dimensions each.
const AXES: [usize; 4] = [32, 32, 32, 32];
const THETA: f64 = 10_000.0;
/// mflux's float16 headroom (see the module's documentation).
const V_HEADROOM: f32 = 4.0;
const FF_HEADROOM: f32 = 8.0;

/// LoRA pairs on one linear, `(A (r, in), B·scale·k (out, r), 1/k)` f16 ([`Lora::pair`]):
/// `y += (x Aᵀ) Bᵀ / k`, the sum in f32.
#[derive(Default)]
struct Adapters(Vec<(Array, Array, f32)>);

impl Adapters {
    fn apply(&self, x: &Array, y: Array) -> Result<Array> {
        if self.0.is_empty() {
            return Ok(y);
        }
        let x16 = x.as_dtype(Dtype::Float16)?;
        let mut d: Option<Array> = None;
        for (a, b, inv) in &self.0 {
            let t = x16.matmul(a.t())?.matmul(b.t())?.as_dtype(Dtype::Float32)?.multiply(Array::from_f32(*inv))?;
            d = Some(match d {
                Some(s) => s.add(&t)?,
                None => t,
            });
        }
        let Some(d) = d else { return Ok(y) };
        Ok(y.as_dtype(Dtype::Float32)?.add(&d)?.as_dtype(y.dtype())?)
    }
}

/// A linear with its LoRAs (unmerged).
struct Lin {
    base: Linear,
    adapters: Adapters,
}

impl Lin {
    fn load(p: &Params, m: &str) -> Result<Self> {
        Ok(Self { base: Linear::load(p, m)?, adapters: Adapters::default() })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.adapters.apply(x, self.base.forward(x)?)
    }

    /// Put `loras`' pairs for `module` (mflux's name) on this linear; how many matched.
    fn set_loras(&mut self, module: &str, loras: &[Lora]) -> Result<usize> {
        let mut a = Adapters::default();
        for l in loras {
            for name in lora_names(module) {
                if let Some(pair) = l.pair(&name)? {
                    a.0.push(pair);
                    break;
                }
            }
        }
        let n = a.0.len();
        self.adapters = a;
        Ok(n)
    }
}

/// RoPE for `(S, H, D)`: `cos` and `sin` `(S, 1, D/2, 1)` f32; adjacent pairs rotate together.
pub struct Rope {
    cos: Array,
    sin: Array,
}

impl Rope {
    /// From the tokens' `(t, row, column, layer)` positions; the angles in f64 on the host (as
    /// the reference does), the tables in f32.
    pub fn new(ids: &[[f32; 4]]) -> Result<Self> {
        let half = AXES.iter().sum::<usize>() / 2;
        let (mut cos, mut sin) = (Vec::with_capacity(ids.len() * half), Vec::with_capacity(ids.len() * half));
        for id in ids {
            for (axis, &d) in AXES.iter().enumerate() {
                for i in (0..d).step_by(2) {
                    let omega = 1.0 / THETA.powf(i as f64 / d as f64);
                    let a = f64::from(id[axis]) * omega;
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        let s = shape(&[ids.len(), 1, half, 1])?;
        Ok(Self { cos: Array::from_slice(&cos, &s), sin: Array::from_slice(&sin, &s) })
    }

    /// `x (S, H, D)` f32, rotated in f32.
    fn apply(&self, x: &Array) -> Result<Array> {
        let s = x.shape();
        let (n, h, d) = (s[0], s[1], s[2]);
        let pairs = x.reshape(&[n, h, d / 2, 2])?;
        let (x0, x1) = (cols(&pairs, 0, 1)?, cols(&pairs, 1, 2)?);
        let r0 = x0.multiply(&self.cos)?.subtract(x1.multiply(&self.sin)?)?;
        let r1 = x0.multiply(&self.sin)?.add(x1.multiply(&self.cos)?)?;
        Ok(ops::concatenate(&[r0, r1], 3)?.reshape(&[n, h, d])?)
    }
}

/// LayerNorm without its own scale (eps 1e-6) in f32, then `(1 + scale)·x + shift`.
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array> {
    let y = fast::layer_norm(x.as_dtype(Dtype::Float32)?, None, None, 1e-6)?;
    Ok(y.add(y.multiply(scale)?)?.add(shift)?)
}

/// Bidirectional attention over `(S, H, D)`: `(S, H·D)`.
fn attention(q: &Array, k: &Array, v: &Array) -> Result<Array> {
    let s = q.shape();
    let (n, h, d) = (s[0], s[1], s[2]);
    let heads_first = |t: &Array| -> Result<Array> { Ok(t.transpose_axes(&[1, 0, 2])?.expand_dims(0)?) };
    let scale = 1.0 / (d as f32).sqrt();
    let o = fast::scaled_dot_product_attention(heads_first(q)?, heads_first(k)?, heads_first(v)?, scale, None, None)?;
    Ok(o.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[n, h * d])?)
}

/// RMSNorm (eps 1e-6) of `x (S, H, D)` in f32 times the weight.
fn rms(x: &Array, w: &Array) -> Result<Array> {
    Ok(fast::rms_norm(x.as_dtype(Dtype::Float32)?, Some(&w.as_dtype(Dtype::Float32)?), 1e-6)?)
}

struct Block {
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    ff_in: Lin,
    ff_out: Lin,
    norm_q: Array,
    norm_k: Array,
}

/// A block's shift, scale and gate `(rows, HIDDEN)` f32: one row for every token, or one row
/// broadcast to all of them.
#[derive(Clone)]
struct Modulation {
    shift: Array,
    scale: Array,
    gate: Array,
}

impl Block {
    fn load(p: &Params, n: &str) -> Result<Self> {
        let l = |s: &str| Lin::load(p, &format!("{n}.{s}"));
        Ok(Self {
            q: l("attn.to_q")?,
            k: l("attn.to_k")?,
            v: l("attn.to_v")?,
            out: l("attn.to_out")?,
            ff_in: l("ff.linear_in")?,
            ff_out: l("ff.linear_out")?,
            norm_q: get(p, &format!("{n}.attn.norm_q.weight"))?,
            norm_k: get(p, &format!("{n}.attn.norm_k.weight"))?,
        })
    }

    fn set_loras(&mut self, n: &str, loras: &[Lora]) -> Result<usize> {
        let mut m = 0;
        for (lin, s) in [
            (&mut self.q, "attn.to_q"),
            (&mut self.k, "attn.to_k"),
            (&mut self.v, "attn.to_v"),
            (&mut self.out, "attn.to_out"),
            (&mut self.ff_in, "ff.linear_in"),
            (&mut self.ff_out, "ff.linear_out"),
        ] {
            m += lin.set_loras(&format!("{n}.{s}"), loras)?;
        }
        Ok(m)
    }

    /// `x (S, HIDDEN)` f32 to `x + gate·(attention + feed-forward)`, both from the same
    /// modulated input.
    fn forward(&self, x: &Array, rope: &Rope, m: &Modulation, dt: Dtype) -> Result<Array> {
        let n = x.shape()[0];
        let h = dim(HEADS)?;
        let hd = dim(HEAD_DIM)?;
        let y = modulate(x, &m.shift, &m.scale)?.as_dtype(dt)?;
        let split = |t: Array| -> Result<Array> { Ok(t.reshape(&[n, h, hd])?) };
        let q = rope.apply(&rms(&split(self.q.forward(&y)?)?, &self.norm_q)?)?.as_dtype(dt)?;
        let k = rope.apply(&rms(&split(self.k.forward(&y)?)?, &self.norm_k)?)?.as_dtype(dt)?;
        // The headroom divides the input, not the output: in f16 the output itself overflows
        // (3e4..3e5 in the bf16 reference). These linears have no bias, so it is exact.
        let v = split(self.v.forward(&y.multiply(Array::from_f32(1.0 / V_HEADROOM).as_dtype(dt)?)?)?)?;
        let att = attention(&q, &k, &v)?;
        let a = self.out.forward(&att)?.as_dtype(Dtype::Float32)?.multiply(Array::from_f32(V_HEADROOM))?;
        let gu = self.ff_in.forward(&y.multiply(Array::from_f32(1.0 / FF_HEADROOM).as_dtype(dt)?)?)?;
        let f = gu.shape()[1] as usize / 2;
        // The gate back to its scale in f32; the up half stays FF_HEADROOM times smaller.
        let gate = cols(&gu, 0, f)?.as_dtype(Dtype::Float32)?.multiply(Array::from_f32(FF_HEADROOM))?;
        let up = cols(&gu, f, 2 * f)?.as_dtype(Dtype::Float32)?;
        let hid = mlx_rs::nn::silu(gate)?.multiply(&up)?.as_dtype(dt)?;
        let ff = self.ff_out.forward(&hid)?.as_dtype(Dtype::Float32)?.multiply(Array::from_f32(FF_HEADROOM))?;
        Ok(x.add(a.add(&ff)?.multiply(&m.gate)?)?)
    }
}

/// The time embedding's MLP (`linear_2(silu(linear_1(sinusoid)))`).
struct TimeEmbedder {
    l1: Linear,
    l2: Linear,
}

/// bf16's rounding of `v` (to nearest, ties to even), as the reference carries its times.
pub fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

impl TimeEmbedder {
    fn load(p: &Params, n: &str) -> Result<Self> {
        Ok(Self { l1: Linear::load(p, &format!("{n}.linear_1"))?, l2: Linear::load(p, &format!("{n}.linear_2"))? })
    }

    /// `(1, HIDDEN)` for the time `t` in 0..1.
    fn forward(&self, t: f32, dt: Dtype) -> Result<Array> {
        let half = 128;
        let t = bf16_round(bf16_round(t) * 1000.0);
        let mut emb = vec![0f32; 2 * half];
        for i in 0..half {
            let freq = (-(10_000f32.ln()) * i as f32 / half as f32).exp();
            let a = t * freq;
            emb[i] = bf16_round(a.cos());
            emb[i + half] = bf16_round(a.sin());
        }
        let emb = Array::from_slice(&emb, &[1, dim(2 * half)?]).as_dtype(dt)?;
        self.l2.forward(&mlx_rs::nn::silu(self.l1.forward(&emb)?)?)
    }
}

/// One modulation per kind of token: `linear(silu(vec))` split into shift, scale and gate.
fn modulation(lin: &Linear, vec: &Array) -> Result<Modulation> {
    let m = lin.forward(&mlx_rs::nn::silu(vec)?)?.as_dtype(Dtype::Float32)?;
    Ok(Modulation { shift: cols(&m, 0, HIDDEN)?, scale: cols(&m, HIDDEN, 2 * HIDDEN)?, gate: cols(&m, 2 * HIDDEN, 3 * HIDDEN)? })
}

/// The tokens of one pass: the picture being made `(n, CHANNELS)`, the references' (empty for
/// text to image) and the text's `(L, TEXT_DIM)`, with their positions.
pub struct Tokens<'a> {
    pub image: &'a Array,
    pub image_ids: &'a [[f32; 4]],
    pub cond: Option<(&'a Array, &'a [[f32; 4]])>,
    pub text: &'a Array,
    pub text_ids: &'a [[f32; 4]],
}

/// The positions and embedded tokens that stay the same at every step: the references' and
/// the text's, through their own five blocks.
pub struct Prepared {
    cond: Option<Array>,
    text: Array,
    rope_image: Rope,
    rope_cond: Option<Rope>,
    rope_text: Rope,
    rope_single: Rope,
}

pub struct KleinDit {
    dtype: Dtype,
    flow_map: bool,
    time: TimeEmbedder,
    target_time: Option<TimeEmbedder>,
    early: [Linear; 3],
    single_mod: [Linear; 3],
    x_image: Linear,
    x_cond: Linear,
    context: Linear,
    image_blocks: Vec<Block>,
    cond_blocks: Vec<Block>,
    text_blocks: Vec<Block>,
    single: Vec<Block>,
    norm_out: Linear,
    proj_out: Linear,
}

impl KleinDit {
    pub fn load(p: &Params, dtype: Dtype) -> Result<Self> {
        let mods = |kind: &str| -> Result<[Linear; 3]> {
            Ok([
                Linear::load(p, &format!("{kind}.image.linear"))?,
                Linear::load(p, &format!("{kind}.image_cond.linear"))?,
                Linear::load(p, &format!("{kind}.txt.linear"))?,
            ])
        };
        let blocks = |prefix: &str, n: usize| -> Result<Vec<Block>> { (0..n).map(|i| Block::load(p, &format!("{prefix}.{i}"))).collect() };
        let flow_map = p.contains_key("target_timestep_embedder.linear_1.weight");
        Ok(Self {
            dtype,
            flow_map,
            time: TimeEmbedder::load(p, "timestep_embedder")?,
            target_time: if flow_map { Some(TimeEmbedder::load(p, "target_timestep_embedder")?) } else { None },
            early: mods("early_stream_modulations")?,
            single_mod: mods("single_stream_modulations")?,
            x_image: Linear::load(p, "x_embedders.image")?,
            x_cond: Linear::load(p, "x_embedders.image_cond")?,
            context: Linear::load(p, "context_embedder")?,
            image_blocks: blocks("mode_blocks.image", DEPTH)?,
            cond_blocks: blocks("mode_blocks.image_cond", DEPTH)?,
            text_blocks: blocks("txt_mode_blocks", DEPTH)?,
            single: blocks("single_transformer_blocks", DEPTH_SINGLE)?,
            norm_out: Linear::load(p, "norm_out.linear")?,
            proj_out: Linear::load(p, "proj_out")?,
        })
    }

    /// Whether this is the distilled flow map (16 steps, no CFG), not Base.
    pub fn is_flow_map(&self) -> bool {
        self.flow_map
    }

    /// Put `loras` on every block's linears, unmerged (an empty list takes them off). How many
    /// modules matched.
    pub fn set_loras(&mut self, loras: &[Lora]) -> Result<usize> {
        let mut m = 0;
        for (prefix, blocks) in [
            ("mode_blocks.image", &mut self.image_blocks),
            ("mode_blocks.image_cond", &mut self.cond_blocks),
            ("txt_mode_blocks", &mut self.text_blocks),
            ("single_transformer_blocks", &mut self.single),
        ] {
            for (i, b) in blocks.iter_mut().enumerate() {
                m += b.set_loras(&format!("{prefix}.{i}"), loras)?;
            }
        }
        Ok(m)
    }

    /// The vectors for each kind of token: the image's time `t`, 0 for the rest, plus the
    /// target time for a flow map.
    fn vectors(&self, t: f32, target: Option<f32>) -> Result<(Array, Array)> {
        let dt = self.dtype;
        let mut image = self.time.forward(t, dt)?.as_dtype(Dtype::Float32)?;
        let mut context = self.time.forward(0.0, dt)?.as_dtype(Dtype::Float32)?;
        match (&self.target_time, target) {
            (Some(tt), Some(r)) => {
                let r = tt.forward(r, dt)?.as_dtype(Dtype::Float32)?;
                image = image.add(&r)?;
                context = context.add(&r)?;
            }
            (None, None) => {}
            _ => anyhow::bail!("a target time is needed by, and only by, the distilled flow map"),
        }
        Ok((image.as_dtype(dt)?, context.as_dtype(dt)?))
    }

    /// What stays the same at every step: the references' and the text's embedded tokens and
    /// the positions. Their blocks run again every step, because a flow map's target time
    /// changes their modulation.
    pub fn prepare(&self, tok: &Tokens) -> Result<Prepared> {
        let dt = self.dtype;
        // The encoder's hidden states reach past f16 (they came out inf in f16): scaled into
        // range by a power of two on the way in (the projection has no bias, so it is exact) and
        // back in f32.
        let peak = tok.text.as_dtype(Dtype::Float32)?.abs()?.max(None)?;
        peak.eval()?;
        let peak = peak.item_exact::<f32>();
        let k = if peak.is_finite() && peak > 4096.0 { 2f32.powi(-((peak / 4096.0).log2().ceil() as i32)) } else { 1.0 };
        let text = self.context.forward(&tok.text.multiply(Array::from_f32(k))?.as_dtype(dt)?)?.as_dtype(Dtype::Float32)?.multiply(Array::from_f32(1.0 / k))?;
        let (cond, rope_cond) = match tok.cond {
            Some((c, ids)) => (Some(self.x_cond.forward(&c.as_dtype(dt)?)?.as_dtype(Dtype::Float32)?), Some(Rope::new(ids)?)),
            None => (None, None),
        };
        let mut all: Vec<[f32; 4]> = tok.image_ids.to_vec();
        if let Some((_, ids)) = tok.cond {
            all.extend_from_slice(ids);
        }
        all.extend_from_slice(tok.text_ids);
        mlx_rs::transforms::eval([&text])?;
        Ok(Prepared { cond, text, rope_image: Rope::new(tok.image_ids)?, rope_cond, rope_text: Rope::new(tok.text_ids)?, rope_single: Rope::new(&all)? })
    }

    /// The velocity `(n, CHANNELS)` f32 at time `t`; `target` the time a flow map jumps to.
    pub fn forward(&self, tok: &Tokens, prep: &Prepared, t: f32, target: Option<f32>) -> Result<Array> {
        let dt = self.dtype;
        let n = tok.image.shape()[0] as usize;
        let (vi, vc) = self.vectors(t, target)?;
        let early_i = modulation(&self.early[0], &vi)?;
        let early_c = modulation(&self.early[1], &vc)?;
        let early_t = modulation(&self.early[2], &vc)?;

        let mut x = self.x_image.forward(&tok.image.as_dtype(dt)?)?.as_dtype(Dtype::Float32)?;
        let mut cond = prep.cond.clone();
        let mut text = prep.text.clone();
        for d in 0..DEPTH {
            x = self.image_blocks[d].forward(&x, &prep.rope_image, &early_i, dt)?;
            if let (Some(c), Some(r)) = (cond.as_ref(), prep.rope_cond.as_ref()) {
                cond = Some(self.cond_blocks[d].forward(c, r, &early_c, dt)?);
            }
            text = self.text_blocks[d].forward(&text, &prep.rope_text, &early_t, dt)?;
        }

        // One stream over [image, references, text], each kind with its own modulation rows.
        let si = modulation(&self.single_mod[0], &vi)?;
        let sc = modulation(&self.single_mod[1], &vc)?;
        let st = modulation(&self.single_mod[2], &vc)?;
        let mut parts = vec![x];
        let mut counts = vec![(n, &si)];
        if let Some(c) = cond {
            counts.push((c.shape()[0] as usize, &sc));
            parts.push(c);
        }
        counts.push((text.shape()[0] as usize, &st));
        parts.push(text);
        let rows_of = |pick: fn(&Modulation) -> &Array| -> Result<Array> {
            let r: Vec<Array> =
                counts.iter().map(|(k, m)| -> Result<Array> { Ok(ops::broadcast_to(pick(m), &[dim(*k)?, dim(HIDDEN)?])?) }).collect::<Result<_>>()?;
            Ok(ops::concatenate(&r, 0)?)
        };
        let m = Modulation { shift: rows_of(|m| &m.shift)?, scale: rows_of(|m| &m.scale)?, gate: rows_of(|m| &m.gate)? };
        let mut h = ops::concatenate(&parts, 0)?;
        for b in &self.single {
            h = b.forward(&h, &prep.rope_single, &m, dt)?;
        }

        // diffusers' AdaLayerNormContinuous: scale first, then shift.
        let mo = self.norm_out.forward(&mlx_rs::nn::silu(&vi)?)?.as_dtype(Dtype::Float32)?;
        let (scale, shift) = (cols(&mo, 0, HIDDEN)?, cols(&mo, HIDDEN, 2 * HIDDEN)?);
        let y = modulate(&rows(&h, 0, n)?, &shift, &scale)?.as_dtype(dt)?;
        Ok(self.proj_out.forward(&y)?.as_dtype(Dtype::Float32)?)
    }
}

/// `(t, row, column, layer)` of an `h × w` latent grid: rows and columns at half scale (the
/// reference's image RoPE scale), `layer` 0 for the picture being made and `-(1 + i)` for
/// reference `i`.
pub fn grid_ids(h: usize, w: usize, layer: f32) -> Vec<[f32; 4]> {
    (0..h).flat_map(|y| (0..w).map(move |x| [0.0, y as f32 * 0.5, x as f32 * 0.5, layer])).collect()
}

/// `(0, 0, 0, i)` for the text's `len` tokens.
pub fn text_ids(len: usize) -> Vec<[f32; 4]> {
    (0..len).map(|i| [0.0, 0.0, 0.0, i as f32]).collect()
}

/// A latent `(1, CHANNELS, h, w)` as tokens `(h·w, CHANNELS)`, row by row.
pub fn to_tokens(z: &Array) -> Result<Array> {
    let s = z.shape();
    anyhow::ensure!(s.len() == 4 && s[0] == 1, "a latent of shape {s:?}");
    Ok(z.reshape(&[s[1], s[2] * s[3]])?.t())
}

/// Tokens `(h·w, CHANNELS)` back to a latent `(1, CHANNELS, h, w)`.
pub fn from_tokens(x: &Array, h: usize, w: usize) -> Result<Array> {
    let c = x.shape().get(1).copied().context("tokens of rank 1")?;
    Ok(x.t().reshape(&[1, c, dim(h)?, dim(w)?])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(1000.0), 1000.0);
        // 1 + 2^-8 is halfway between bf16's 1 and 1 + 2^-7: ties to even.
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
    }

    #[test]
    fn grids_and_text_positions() {
        let g = grid_ids(2, 3, -1.0);
        assert_eq!(g.len(), 6);
        assert_eq!(g[4], [0.0, 0.5, 0.5, -1.0]);
        assert_eq!(text_ids(3)[2], [0.0, 0.0, 0.0, 2.0]);
    }
}
