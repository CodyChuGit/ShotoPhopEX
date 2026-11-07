//! MOSS-Image's DiT on MLX: `qwen_image_rs::moss::dit` (its candle chains) step for step. The
//! residual stream stays in f32; each linear sees its input in the run dtype.

use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype, fast, ops};
use qwen_image_rs::moss::dit::MossConfig;
use qwen_image_rs::moss::weights::Weights;

use crate::convert::{Converter, Lora, Precision};
use crate::nn::{Linear, Params, cols, dim, get, rows, scalar, shape};

/// Convert the DiT file to [`Params`]: the blocks' and the text refiner's linears quantized
/// (when `precision` asks), the embedders, modulation and head dense; `loras` merged into the
/// blocks (their deltas summed). `cfg.layers` limits the blocks read (the checks' cut-down
/// model).
pub fn convert(path: &Path, cfg: &MossConfig, dtype: Dtype, precision: Precision, loras: &[Lora]) -> Result<Params> {
    let w = Weights::open(path, &candle_core::Device::Cpu, candle_core::DType::F32, false)?;
    let mut c = Converter::new(&w, dtype, precision)?;
    for m in [
        "video_embedder",
        "text_embedder.proj",
        "time_embedding.0",
        "time_embedding.2",
        "shared_block_modulation",
        "video_head_modulation",
        "final_linear_video",
    ] {
        c.linear(m, false, None)?;
    }
    c.tensor("final_norm_video.weight")?;
    for i in 0..2 {
        let p = format!("text_embedder.blocks.{i}");
        for n in ["norm1", "norm2", "q_norm", "k_norm"] {
            c.tensor(&format!("{p}.{n}.weight"))?;
        }
        for n in ["qkv", "proj", "ffn_up", "ffn_gate", "ffn_down"] {
            c.linear(&format!("{p}.{n}"), true, None)?;
        }
    }
    let delta = |m: &str| -> Result<Option<Array>> {
        let mut sum: Option<Array> = None;
        for l in loras {
            if let Some(d) = l.delta(m)? {
                sum = Some(match sum {
                    Some(s) => s.add(&d)?,
                    None => d,
                });
            }
        }
        Ok(sum)
    };
    for i in 0..cfg.layers {
        let p = format!("layers.{i}");
        for n in ["norm1", "norm2", "q_norm", "k_norm"] {
            c.tensor(&format!("{p}.{n}.weight"))?;
        }
        // q, k and v as one linear (ComfyUI's `proj.qkv`, or the official split stacked).
        let qkv = format!("{p}.proj.qkv");
        let wqkv = if c.contains(&format!("{qkv}.weight")) {
            c.dense(&qkv)?
        } else {
            let parts = ["q", "k", "v"].iter().map(|n| c.dense(&format!("{p}.proj.{n}"))).collect::<Result<Vec<_>>>()?;
            ops::concatenate(&parts, 0)?
        };
        let mut d_qkv = delta(&qkv)?;
        let split: Vec<Option<Array>> = ["q", "k", "v"].iter().map(|n| delta(&format!("{p}.proj.{n}"))).collect::<Result<_>>()?;
        if split.iter().any(Option::is_some) {
            let side = dim(cfg.dim)?;
            let parts =
                split.into_iter().map(|d| -> Result<Array> { Ok(d.map_or_else(|| Array::zeros::<f32>(&[side, side]), Ok)?) }).collect::<Result<Vec<_>>>()?;
            let stacked = ops::concatenate(&parts, 0)?;
            d_qkv = Some(match d_qkv {
                Some(d) => d.add(&stacked)?,
                None => stacked,
            });
        }
        c.put_linear(&qkv, wqkv, true, d_qkv)?;
        for n in ["o", "ffn_up_gate", "ffn_down"] {
            let m = format!("{p}.proj.{n}");
            let d = delta(&m)?;
            c.linear(&m, true, d)?;
        }
    }
    Ok(c.out)
}

/// RMSNorm (eps 1e-6) in f32 times the weight; out in f32.
fn rms(x: &Array, w: &Array) -> Result<Array> {
    Ok(fast::rms_norm(x.as_dtype(Dtype::Float32)?, Some(&w.as_dtype(Dtype::Float32)?), 1e-6)?)
}

/// SiTU: `4·tanh(gate/4)·σ(gate) · 25·tanh(hidden/25)`, in `hidden`'s dtype.
fn situ(hidden: &Array, gate: &Array) -> Result<Array> {
    let dt = hidden.dtype();
    let a = ops::tanh(gate.multiply(scalar(0.25, dt)?)?)?.multiply(scalar(4.0, dt)?)?.multiply(ops::sigmoid(gate)?)?;
    let b = ops::tanh(hidden.multiply(scalar(1.0 / 25.0, dt)?)?)?.multiply(scalar(25.0, dt)?)?;
    Ok(a.multiply(&b)?)
}

/// RoPE tables for `(S, H, D)`: `cos` and `sin` `(S, 1, D)` (each half repeated, the
/// rotate-half pairing), in the run dtype.
pub struct Rope {
    cos: Array,
    sin: Array,
}

impl Rope {
    /// From qir's `[sin | cos]` rows `(S, D)` f32.
    fn new(table: &Array, d: usize, dtype: Dtype) -> Result<Self> {
        let half = d / 2;
        let twice = |a: Array| -> Result<Array> { Ok(ops::concatenate(&[&a, &a], 1)?.expand_dims(1)?.as_dtype(dtype)?) };
        Ok(Self { sin: twice(cols(table, 0, half)?)?, cos: twice(cols(table, half, d)?)? })
    }

    fn rows(&self, a: usize, b: usize) -> Result<Self> {
        Ok(Self { cos: rows(&self.cos, a, b)?, sin: rows(&self.sin, a, b)? })
    }

    /// `x·cos + rotate_half(x)·sin` for `x (S, H, D)`.
    fn apply(&self, x: &Array) -> Result<Array> {
        let d = *x.shape().last().context("a 0-d array")? as usize;
        let half = d / 2;
        let turned = ops::concatenate(&[cols(x, half, d)?.negative()?, cols(x, 0, half)?], 2)?;
        Ok(x.multiply(&self.cos)?.add(turned.multiply(&self.sin)?)?)
    }
}

/// Bidirectional attention over `q`, `k`, `v` `(S, H, D)`: `(S, H·D)`.
fn attention(q: &Array, k: &Array, v: &Array) -> Result<Array> {
    let s = q.shape();
    let (n, h, d) = (s[0], s[1], s[2]);
    let heads_first = |t: &Array| -> Result<Array> { Ok(t.transpose_axes(&[1, 0, 2])?.expand_dims(0)?) };
    let scale = 1.0 / (d as f32).sqrt();
    let o = fast::scaled_dot_product_attention(heads_first(q)?, heads_first(k)?, heads_first(v)?, scale, None, None)?;
    Ok(o.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[n, h * d])?)
}

/// `(S, dim)` to `q`, `k`, `v` `(S, H, D)`.
fn split_qkv(qkv: &Array, dim_: usize, heads: usize) -> Result<[Array; 3]> {
    let s = qkv.shape()[0];
    let part = |i: usize| -> Result<Array> { Ok(cols(qkv, i * dim_, (i + 1) * dim_)?.reshape(&[s, dim(heads)?, dim(dim_ / heads)?])?) };
    Ok([part(0)?, part(1)?, part(2)?])
}

struct TextBlock {
    norm1: Array,
    norm2: Array,
    q_norm: Array,
    k_norm: Array,
    qkv: Linear,
    proj: Linear,
    ffn_up: Linear,
    ffn_gate: Linear,
    ffn_down: Linear,
}

impl TextBlock {
    fn load(p: &Params, n: &str) -> Result<Self> {
        let w = |s: &str| get(p, &format!("{n}.{s}.weight"));
        let l = |s: &str| Linear::load(p, &format!("{n}.{s}"));
        Ok(Self {
            norm1: w("norm1")?,
            norm2: w("norm2")?,
            q_norm: w("q_norm")?,
            k_norm: w("k_norm")?,
            qkv: l("qkv")?,
            proj: l("proj")?,
            ffn_up: l("ffn_up")?,
            ffn_gate: l("ffn_gate")?,
            ffn_down: l("ffn_down")?,
        })
    }

    /// `x (S, dim)` f32.
    fn forward(&self, x: &Array, rope: &Rope, heads: usize, dt: Dtype) -> Result<Array> {
        let d = x.shape()[1] as usize;
        let qkv = self.qkv.forward(&rms(x, &self.norm1)?.as_dtype(dt)?)?;
        let [q, k, v] = split_qkv(&qkv, d, heads)?;
        let q = rope.apply(&rms(&q, &self.q_norm)?.as_dtype(dt)?)?;
        let k = rope.apply(&rms(&k, &self.k_norm)?.as_dtype(dt)?)?;
        let x = x.add(self.proj.forward(&attention(&q, &k, &v)?)?)?;
        let y = rms(&x, &self.norm2)?.as_dtype(dt)?;
        let h = situ(&self.ffn_up.forward(&y)?, &self.ffn_gate.forward(&y)?)?;
        Ok(x.add(self.ffn_down.forward(&h)?)?)
    }
}

struct Block {
    norm1: Array,
    norm2: Array,
    q_norm: Array,
    k_norm: Array,
    qkv: Linear,
    o: Linear,
    up_gate: Linear,
    down: Linear,
    /// LoRAs applied unmerged (`qkv`, `o`, `up_gate`, `down`): a few-step LoRA's delta is
    /// below the rounding of the f16 or quantized weight it would be merged into.
    adapters: [Adapters; 4],
}

/// LoRA pairs on one linear, `(A (r, in), B·scale·k (out, r), 1/k)` f16 ([`Lora::pair`]):
/// `y += (x Aᵀ) Bᵀ / k`, the sum in f32.
#[derive(Default)]
struct Adapters(Vec<(Array, Array, f32)>);

impl Adapters {
    fn apply(&self, x: &Array, y: Array) -> Result<Array> {
        if self.0.is_empty() {
            return Ok(y);
        }
        // In f16 (Metal accumulates in f32; B's scale keeps the product well inside f16), added
        // to the output in f32.
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

impl Block {
    fn load(p: &Params, n: &str) -> Result<Self> {
        let w = |s: &str| get(p, &format!("{n}.{s}.weight"));
        let l = |s: &str| Linear::load(p, &format!("{n}.proj.{s}"));
        Ok(Self {
            norm1: w("norm1")?,
            norm2: w("norm2")?,
            q_norm: w("q_norm")?,
            k_norm: w("k_norm")?,
            qkv: l("qkv")?,
            o: l("o")?,
            up_gate: l("ffn_up_gate")?,
            down: l("ffn_down")?,
            adapters: Default::default(),
        })
    }

    /// `x (S, dim)` f32; `m` the six modulation rows `(1, dim)` f32; `refs` MOSS Edit's
    /// cached keys and values, attended after the sequence's own. Also returns this block's
    /// keys (normed, rotated) and values `(S, H, D)`, for a reference pass to keep.
    fn forward(&self, x: &Array, rope: &Rope, m: &[Array; 6], heads: usize, dt: Dtype, refs: Option<&RefLayer>) -> Result<(Array, RefLayer)> {
        let d = x.shape()[1] as usize;
        let [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f] = m;
        let modulate = |y: Array, scale: &Array, shift: &Array| -> Result<Array> { Ok(y.add(y.multiply(scale)?)?.add(shift)?.as_dtype(dt)?) };
        let y = modulate(rms(x, &self.norm1)?, scale_a, shift_a)?;
        let [q, k, v] = split_qkv(&self.adapters[0].apply(&y, self.qkv.forward(&y)?)?, d, heads)?;
        let q = rope.apply(&rms(&q, &self.q_norm)?.as_dtype(dt)?)?;
        let k = rope.apply(&rms(&k, &self.k_norm)?.as_dtype(dt)?)?;
        let att = match refs {
            Some(r) => attention(&q, &ops::concatenate(&[&k, &r.k], 0)?, &ops::concatenate(&[&v, &r.v], 0)?)?,
            None => attention(&q, &k, &v)?,
        };
        let a = self.adapters[1].apply(&att, self.o.forward(&att)?)?;
        let x = x.add(a.multiply(gate_a)?)?;
        let y = modulate(rms(&x, &self.norm2)?, scale_f, shift_f)?;
        let hg = self.adapters[2].apply(&y, self.up_gate.forward(&y)?)?;
        let f = hg.shape()[1] as usize / 2;
        let h = situ(&cols(&hg, 0, f)?, &cols(&hg, f, 2 * f)?)?;
        Ok((x.add(self.adapters[3].apply(&h, self.down.forward(&h)?)?.multiply(gate_f)?)?, RefLayer { k, v }))
    }
}

/// One block's keys and values of MOSS Edit's references `(N, H, D)`.
#[derive(Clone)]
struct RefLayer {
    k: Array,
    v: Array,
}

/// MOSS Edit's references through every block, made once per edit: the reference branch runs
/// at t = 0 and attends only to itself, so it is the same at every step and for both prompts.
pub struct RefCache {
    layers: Vec<RefLayer>,
}

pub struct MossDit {
    cfg: MossConfig,
    dtype: Dtype,
    video_embedder: Linear,
    text_proj: Linear,
    text_blocks: Vec<TextBlock>,
    time0: Linear,
    time2: Linear,
    block_mod: Linear,
    head_mod: Linear,
    layers: Vec<Block>,
    final_norm: Array,
    final_linear: Linear,
}

impl MossDit {
    pub fn load(cfg: &MossConfig, p: &Params, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            cfg: cfg.clone(),
            dtype,
            video_embedder: Linear::load(p, "video_embedder")?,
            text_proj: Linear::load(p, "text_embedder.proj")?,
            text_blocks: (0..2).map(|i| TextBlock::load(p, &format!("text_embedder.blocks.{i}"))).collect::<Result<_>>()?,
            time0: Linear::load(p, "time_embedding.0")?,
            time2: Linear::load(p, "time_embedding.2")?,
            block_mod: Linear::load(p, "shared_block_modulation")?,
            head_mod: Linear::load(p, "video_head_modulation")?,
            layers: (0..cfg.layers).map(|i| Block::load(p, &format!("layers.{i}"))).collect::<Result<_>>()?,
            final_norm: get(p, "final_norm_video.weight")?,
            final_linear: Linear::load(p, "final_linear_video")?,
        })
    }

    pub fn config(&self) -> &MossConfig {
        &self.cfg
    }

    /// Put `loras` on the blocks, unmerged (an empty list takes them off). A file's separate
    /// `proj.q`, `proj.k`, `proj.v` adapters each go on the stacked `qkv`, their `B`s padded to
    /// their rows. Returns how many modules matched.
    pub fn set_loras(&mut self, loras: &[Lora]) -> Result<usize> {
        let mut matched = 0;
        for (i, block) in self.layers.iter_mut().enumerate() {
            let p = format!("layers.{i}");
            let mut adapters: [Adapters; 4] = Default::default();
            for l in loras {
                if let Some(pair) = l.pair(&format!("{p}.proj.qkv"))? {
                    adapters[0].0.push(pair);
                    matched += 1;
                }
                // q, k and v each as its own pair on the stacked qkv: B padded with zero rows to
                // its third of the output (each keeps its own scale).
                for (j, n) in ["q", "k", "v"].iter().enumerate() {
                    let Some((a, b, inv)) = l.pair(&format!("{p}.proj.{n}"))? else { continue };
                    matched += 1;
                    let r = b.shape()[1];
                    let zeros = |rows: usize| -> Result<Array> { Ok(Array::zeros::<f32>(&[dim(rows)?, r])?.as_dtype(Dtype::Float16)?) };
                    let mut parts = Vec::with_capacity(3);
                    if j > 0 {
                        parts.push(zeros(j * self.cfg.dim)?);
                    }
                    parts.push(b);
                    if j < 2 {
                        parts.push(zeros((2 - j) * self.cfg.dim)?);
                    }
                    let b = ops::concatenate(&parts, 0)?;
                    b.eval()?;
                    adapters[0].0.push((a, b, inv));
                }
                for (slot, n) in [(1, "o"), (2, "ffn_up_gate"), (3, "ffn_down")] {
                    if let Some(pair) = l.pair(&format!("{p}.proj.{n}"))? {
                        adapters[slot].0.push(pair);
                        matched += 1;
                    }
                }
            }
            block.adapters = adapters;
        }
        Ok(matched)
    }

    /// qir's `[sin | cos]` rows `(h·w + text_len, head_dim)` f32: the image tokens (frame 0,
    /// text index `text_len`, height and width centred) then the text tokens.
    fn rope_table(&self, h: usize, w: usize, text_len: usize) -> Result<Array> {
        let nb = self.cfg.head_dim() / 8;
        let bands: Vec<f32> = (0..nb).map(|i| 1.0 / self.cfg.rope_base.powf(i as f32 / nb as f32)).collect();
        let mut out = Vec::with_capacity((h * w + text_len) * 8 * nb);
        let mut push = |pos: [f32; 4]| {
            let phase: Vec<f32> = pos.iter().flat_map(|p| bands.iter().map(move |b| p * b)).collect();
            out.extend(phase.iter().map(|p| p.sin()));
            out.extend(phase.iter().map(|p| p.cos()));
        };
        let (ch, cw) = ((h as f32 - 1.0) / 2.0, (w as f32 - 1.0) / 2.0);
        for y in 0..h {
            for x in 0..w {
                push([0.0, text_len as f32, y as f32 - ch, x as f32 - cw]);
            }
        }
        for t in 0..text_len {
            push([0.0, t as f32, 0.0, 0.0]);
        }
        Ok(Array::from_slice(&out, &shape(&[h * w + text_len, 8 * nb])?))
    }

    /// MOSS Edit's reference rows `[sin | cos]` `(h·w, head_dim)` f32 for reference `index`:
    /// frame `index + 1`, text index 0, height and width centred on its own grid.
    fn reference_rope(&self, h: usize, w: usize, index: usize) -> Result<Array> {
        let nb = self.cfg.head_dim() / 8;
        let bands: Vec<f32> = (0..nb).map(|i| 1.0 / self.cfg.rope_base.powf(i as f32 / nb as f32)).collect();
        let mut out = Vec::with_capacity(h * w * 8 * nb);
        let (ch, cw) = ((h as f32 - 1.0) / 2.0, (w as f32 - 1.0) / 2.0);
        for y in 0..h {
            for x in 0..w {
                let pos = [(index + 1) as f32, 0.0, y as f32 - ch, x as f32 - cw];
                let phase: Vec<f32> = pos.iter().flat_map(|p| bands.iter().map(move |b| p * b)).collect();
                out.extend(phase.iter().map(|p| p.sin()));
                out.extend(phase.iter().map(|p| p.cos()));
            }
        }
        Ok(Array::from_slice(&out, &shape(&[h * w, 8 * nb])?))
    }

    /// Run MOSS Edit's references `(1, channels, h, w)` (the FLUX.2 encoder's latents, the
    /// picture being edited first) through every block once, at t = 0, attending only to each
    /// other; the blocks' keys and values for the sampling steps.
    pub fn prepare_references(&self, refs: &[Array]) -> Result<RefCache> {
        anyhow::ensure!(!refs.is_empty(), "an edit needs a reference");
        let dt = self.dtype;
        let mut tokens = Vec::with_capacity(refs.len());
        let mut ropes = Vec::with_capacity(refs.len());
        for (i, r) in refs.iter().enumerate() {
            let s = r.shape();
            anyhow::ensure!(
                s.len() == 4 && s[1] as usize == self.cfg.channels,
                "a reference of shape {s:?} (the model takes (1, {}, h, w))",
                self.cfg.channels
            );
            let (c, h, w) = (s[1], s[2] as usize, s[3] as usize);
            tokens.push(self.video_embedder.forward(&r.reshape(&[c, dim(h * w)?])?.t().as_dtype(dt)?)?.as_dtype(Dtype::Float32)?);
            ropes.push(self.reference_rope(h, w, i)?);
        }
        let mut x = ops::concatenate(&tokens, 0)?;
        let rope = Rope::new(&ops::concatenate(&ropes, 0)?, self.cfg.head_dim(), dt)?;
        let (m, _) = self.modulation(0.0)?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (next, kv) = layer.forward(&x, &rope, &m, self.cfg.heads, dt, None)?;
            mlx_rs::transforms::eval([&next, &kv.k, &kv.v])?;
            layers.push(kv);
            x = next;
        }
        Ok(RefCache { layers })
    }

    /// The six block rows and the two head rows `(1, dim)` f32 for `timestep`.
    fn modulation(&self, timestep: f32) -> Result<([Array; 6], [Array; 2])> {
        let half = self.cfg.freq_dim / 2;
        let t = f64::from(timestep);
        let mut emb = vec![0f32; self.cfg.freq_dim];
        for i in 0..half {
            let phase = t * 10_000f64.powf(-(i as f64) / half as f64);
            emb[i] = phase.cos() as f32;
            emb[i + half] = phase.sin() as f32;
        }
        let emb = Array::from_slice(&emb, &[1, dim(self.cfg.freq_dim)?]).as_dtype(self.dtype)?;
        let time = self.time2.forward(&mlx_rs::nn::silu(self.time0.forward(&emb)?)?)?;
        let m = self.block_mod.forward(&time)?.as_dtype(Dtype::Float32)?;
        let head = self.head_mod.forward(&time)?.as_dtype(Dtype::Float32)?;
        let d = self.cfg.dim;
        let row = |t: &Array, i: usize| cols(t, i * d, (i + 1) * d);
        Ok(([row(&m, 0)?, row(&m, 1)?, row(&m, 2)?, row(&m, 3)?, row(&m, 4)?, row(&m, 5)?], [row(&head, 0)?, row(&head, 1)?]))
    }

    /// The velocity `(1, channels, h, w)` f32 for the latent `x (1, channels, h, w)` at
    /// `timestep` (σ·1000) and the encoder's hidden state `ctx (L, text_dim)`; the blocks in
    /// `skip` are left out; `refs` MOSS Edit's references, attended by every block.
    pub fn forward(&self, x: &Array, timestep: f32, ctx: &Array, skip: &[usize], refs: Option<&RefCache>) -> Result<Array> {
        if let Some(r) = refs {
            anyhow::ensure!(r.layers.len() == self.layers.len(), "a reference cache of {} blocks for {}", r.layers.len(), self.layers.len());
        }
        let s = x.shape();
        anyhow::ensure!(s.len() == 4 && s[1] as usize == self.cfg.channels, "a latent of shape {s:?} (the model takes (1, {}, h, w))", self.cfg.channels);
        let (c, h, w) = (s[1], s[2] as usize, s[3] as usize);
        let n = h * w;
        let l = ctx.shape()[0] as usize;
        let dt = self.dtype;
        let rope = Rope::new(&self.rope_table(h, w, l)?, self.cfg.head_dim(), dt)?;
        let video = self.video_embedder.forward(&x.reshape(&[c, dim(n)?])?.t().as_dtype(dt)?)?;
        let mut text = self.text_proj.forward(&ctx.as_dtype(dt)?)?.as_dtype(Dtype::Float32)?;
        let text_rope = rope.rows(n, n + l)?;
        for b in &self.text_blocks {
            text = b.forward(&text, &text_rope, self.cfg.heads, dt)?;
        }
        let (m, head) = self.modulation(timestep)?;
        let mut tokens = ops::concatenate(&[video.as_dtype(Dtype::Float32)?, text], 0)?;
        for (i, layer) in self.layers.iter().enumerate() {
            if !skip.contains(&i) {
                tokens = layer.forward(&tokens, &rope, &m, self.cfg.heads, dt, refs.map(|r| &r.layers[i]))?.0;
            }
        }
        let y = rms(&rows(&tokens, 0, n)?, &self.final_norm)?;
        let y = y.add(y.multiply(&head[1])?)?.add(&head[0])?.as_dtype(dt)?;
        let y = self.final_linear.forward(&y)?.as_dtype(Dtype::Float32)?; // (n, channels)
        Ok(y.t().reshape(&[1, c, dim(h)?, dim(w)?])?)
    }
}
