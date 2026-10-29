//! MOSS-Image's DiT: a 5B single-stream transformer over the 128-channel latent (one token per
//! latent pixel) and the projected text, with one shared AdaLN modulation for every block, a
//! two-block text refiner, 4-axis RoPE (frame, text index, height, width; the image centred)
//! and the SiTU feed-forward. A clean-room port of the architecture as the official
//! `pipeline_moss_image.py` (Apache-2.0) and stable-diffusion.cpp's `moss_image.hpp` describe it,
//! checked against the former on the same weights.
//!
//! ComfyUI's file keeps the official key names except that each block's q, k and v are one
//! `proj.qkv` linear.

use candle_core::{DType, Device, Tensor, D};

use super::weights::{QLinear, Weights};
use crate::Result;

/// MOSS-Image's shape (Base and SFT share it).
#[derive(Clone, Debug)]
pub struct MossConfig {
    pub dim: usize,
    pub ffn: usize,
    pub heads: usize,
    pub layers: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub channels: usize,
    pub rope_base: f32,
}

impl Default for MossConfig {
    fn default() -> Self {
        Self { dim: 3072, ffn: 8192, heads: 24, layers: 42, text_dim: 4096, freq_dim: 256, channels: 128, rope_base: 10_000.0 }
    }
}

impl MossConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}

/// RMSNorm (eps 1e-6) in f32, rounded to the input's dtype, then times the plain weight.
struct Rms {
    weight: Tensor,
}

impl Rms {
    fn load(w: &Weights, name: &str) -> Result<Self> {
        Ok(Self { weight: w.get(name)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x32 = x.to_dtype(DType::F32)?;
        let y = x32.broadcast_mul(&(x32.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?)?;
        Ok(y.to_dtype(x.dtype())?.broadcast_mul(&self.weight)?)
    }
}

/// SiTU: `4·tanh(gate/4)·σ(gate) · 25·tanh(hidden/25)`.
fn situ(hidden: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let a = ((gate / 4.0)?.tanh()? * 4.0)?.mul(&candle_nn::ops::sigmoid(gate)?)?;
    let b = ((hidden / 25.0)?.tanh()? * 25.0)?;
    Ok(a.mul(&b)?)
}

/// RoPE for `x (S, H, D)` with `rope (S, D)` holding `[sin | cos]` of `D/2` phases (the
/// rotate-half pairing: channel `i` with `i + D/2`).
fn rotate(x: &Tensor, rope: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let sin = rope.narrow(1, 0, half)?;
    let cos = rope.narrow(1, half, half)?;
    let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(1)?;
    let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(1)?;
    let turned = Tensor::cat(&[x.narrow(2, half, half)?.neg()?, x.narrow(2, 0, half)?], 2)?;
    Ok((x.broadcast_mul(&cos)? + turned.broadcast_mul(&sin)?)?)
}

/// Bidirectional attention over `q`, `k`, `v` `(S, H, D)` (RoPE applied); `(S, H·D)`.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (s, h, d) = q.dims3()?;
    let scale = 1.0 / (d as f64).sqrt();
    #[cfg(feature = "flash")]
    if q.device().is_cuda() && q.dtype() == DType::BF16 && d == crate::flash::HEAD_DIM {
        let out =
            crate::flash::attention(&q.unsqueeze(0)?.transpose(1, 2)?, &k.unsqueeze(0)?.transpose(1, 2)?, &v.unsqueeze(0)?.transpose(1, 2)?, scale as f32)?;
        return Ok(out.reshape((s, h * d))?);
    }
    const ROWS: usize = 1024;
    let kt = k.transpose(0, 1)?.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?; // (H, D, S)
    let vh = v.transpose(0, 1)?.to_dtype(DType::F32)?.contiguous()?; // (H, S, D)
    let qh = q.transpose(0, 1)?.to_dtype(DType::F32)?.contiguous()?;
    let mut outs = Vec::with_capacity(s.div_ceil(ROWS));
    let mut r0 = 0;
    while r0 < s {
        let n = ROWS.min(s - r0);
        let scores = (qh.narrow(1, r0, n)?.matmul(&kt)? * scale)?;
        outs.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&vh)?);
        r0 += n;
    }
    let out = Tensor::cat(&outs, 1)?.transpose(0, 1)?.to_dtype(q.dtype())?;
    Ok(out.reshape((s, h * d))?)
}

/// A text-refiner block: plain pre-norm attention and SiTU, biases on every linear.
struct TextBlock {
    norm1: Rms,
    norm2: Rms,
    q_norm: Rms,
    k_norm: Rms,
    qkv: QLinear,
    proj: QLinear,
    ffn_up: QLinear,
    ffn_gate: QLinear,
    ffn_down: QLinear,
}

impl TextBlock {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            norm1: Rms::load(w, &format!("{p}.norm1.weight"))?,
            norm2: Rms::load(w, &format!("{p}.norm2.weight"))?,
            q_norm: Rms::load(w, &format!("{p}.q_norm.weight"))?,
            k_norm: Rms::load(w, &format!("{p}.k_norm.weight"))?,
            qkv: w.linear(&format!("{p}.qkv"))?,
            proj: w.linear(&format!("{p}.proj"))?,
            ffn_up: w.linear(&format!("{p}.ffn_up"))?,
            ffn_gate: w.linear(&format!("{p}.ffn_gate"))?,
            ffn_down: w.linear(&format!("{p}.ffn_down"))?,
        })
    }

    fn forward(&self, x: &Tensor, rope: &Tensor, heads: usize) -> Result<Tensor> {
        let (s, dim) = x.dims2()?;
        let hd = dim / heads;
        let qkv = self.qkv.forward(&self.norm1.forward(x)?)?;
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(1, i * dim, dim)?.reshape((s, heads, hd))?) };
        let q = rotate(&self.q_norm.forward(&part(0)?)?, rope)?;
        let k = rotate(&self.k_norm.forward(&part(1)?)?, rope)?;
        let x = (x + self.proj.forward(&attention(&q, &k, &part(2)?)?)?)?;
        let y = self.norm2.forward(&x)?;
        let h = situ(&self.ffn_up.forward(&y)?, &self.ffn_gate.forward(&y)?)?;
        Ok((&x + self.ffn_down.forward(&h)?)?)
    }
}

/// A main block: AdaLN from the shared modulation `(shift, scale, gate)` for attention and for
/// the SiTU feed-forward.
struct Block {
    norm1: Rms,
    norm2: Rms,
    q_norm: Rms,
    k_norm: Rms,
    qkv: QLinear,
    o: QLinear,
    up_gate: QLinear,
    down: QLinear,
}

impl Block {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        let qkv = if w.contains(&format!("{p}.proj.qkv.weight")) {
            w.linear(&format!("{p}.proj.qkv"))?
        } else {
            // The official split q/k/v (a diffusers export), stacked here.
            let parts = ["q", "k", "v"].iter().map(|n| w.get(&format!("{p}.proj.{n}.weight"))).collect::<Result<Vec<_>>>()?;
            QLinear::Dense(candle_nn::Linear::new(Tensor::cat(&parts, 0)?, None))
        };
        Ok(Self {
            norm1: Rms::load(w, &format!("{p}.norm1.weight"))?,
            norm2: Rms::load(w, &format!("{p}.norm2.weight"))?,
            q_norm: Rms::load(w, &format!("{p}.q_norm.weight"))?,
            k_norm: Rms::load(w, &format!("{p}.k_norm.weight"))?,
            qkv,
            o: w.linear(&format!("{p}.proj.o"))?,
            up_gate: w.linear(&format!("{p}.proj.ffn_up_gate"))?,
            down: w.linear(&format!("{p}.proj.ffn_down"))?,
        })
    }

    /// `m`: the six modulation rows `(6, dim)` in the run dtype.
    fn forward(&self, x: &Tensor, rope: &Tensor, m: &[Tensor; 6], heads: usize) -> Result<Tensor> {
        let (s, dim) = x.dims2()?;
        let hd = dim / heads;
        let [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f] = m;
        let y = self.norm1.forward(x)?;
        let y = (&y + y.broadcast_mul(scale_a)?)?.broadcast_add(shift_a)?;
        let qkv = self.qkv.forward(&y)?;
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(1, i * dim, dim)?.reshape((s, heads, hd))?) };
        let q = rotate(&self.q_norm.forward(&part(0)?)?, rope)?;
        let k = rotate(&self.k_norm.forward(&part(1)?)?, rope)?;
        let a = self.o.forward(&attention(&q, &k, &part(2)?)?)?;
        let x = (x + a.broadcast_mul(gate_a)?)?;
        let y = self.norm2.forward(&x)?;
        let y = (&y + y.broadcast_mul(scale_f)?)?.broadcast_add(shift_f)?;
        let hg = self.up_gate.forward(&y)?;
        let f = hg.dim(1)? / 2;
        let h = situ(&hg.narrow(1, 0, f)?, &hg.narrow(1, f, f)?)?;
        Ok((&x + self.down.forward(&h)?.broadcast_mul(gate_f)?)?)
    }
}

pub struct MossDit {
    cfg: MossConfig,
    video_embedder: QLinear,
    text_proj: QLinear,
    text_blocks: Vec<TextBlock>,
    time0: QLinear,
    time2: QLinear,
    block_mod: QLinear,
    head_mod: QLinear,
    layers: Vec<Block>,
    final_norm: Rms,
    final_linear: QLinear,
    dev: Device,
    dtype: DType,
}

impl MossDit {
    pub fn load(cfg: &MossConfig, w: &Weights) -> Result<Self> {
        let layers = (0..cfg.layers).map(|i| Block::load(w, &format!("layers.{i}"))).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            video_embedder: w.linear("video_embedder")?,
            text_proj: w.linear("text_embedder.proj")?,
            text_blocks: (0..2).map(|i| TextBlock::load(w, &format!("text_embedder.blocks.{i}"))).collect::<Result<Vec<_>>>()?,
            time0: w.linear("time_embedding.0")?,
            time2: w.linear("time_embedding.2")?,
            block_mod: w.linear("shared_block_modulation")?,
            head_mod: w.linear("video_head_modulation")?,
            layers,
            final_norm: Rms::load(w, "final_norm_video.weight")?,
            final_linear: w.linear("final_linear_video")?,
            dev: w.device().clone(),
            dtype: w.dtype(),
        })
    }

    pub fn config(&self) -> &MossConfig {
        &self.cfg
    }

    /// The rows `[sin | cos]` `(tokens, head_dim)` f32: the image tokens (row-major, at frame 0,
    /// text index `text_len`, height and width centred on the grid) then the text tokens (at
    /// text index 0..), each of the four axes turning `head_dim/8` bands.
    pub fn rope(&self, h: usize, w: usize, text_len: usize) -> Result<Tensor> {
        let nb = self.cfg.head_dim() / 8;
        let bands: Vec<f32> = (0..nb).map(|i| 1.0 / self.cfg.rope_base.powf(i as f32 / nb as f32)).collect();
        let mut rows = Vec::with_capacity((h * w + text_len) * 8 * nb);
        let mut push = |pos: [f32; 4]| {
            let phase: Vec<f32> = pos.iter().flat_map(|p| bands.iter().map(move |b| p * b)).collect();
            rows.extend(phase.iter().map(|p| p.sin()));
            rows.extend(phase.iter().map(|p| p.cos()));
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
        Ok(Tensor::from_vec(rows, (h * w + text_len, 8 * nb), &self.dev)?)
    }

    /// The modulation for a timestep: the six block rows and the two head rows `(shift, scale)`.
    fn modulation(&self, timestep: f32) -> Result<([Tensor; 6], [Tensor; 2])> {
        let half = self.cfg.freq_dim / 2;
        let t = f64::from(timestep);
        let mut emb = vec![0f32; self.cfg.freq_dim];
        for i in 0..half {
            let phase = t * 10_000f64.powf(-(i as f64) / half as f64);
            emb[i] = phase.cos() as f32;
            emb[i + half] = phase.sin() as f32;
        }
        let emb = Tensor::from_vec(emb, (1, self.cfg.freq_dim), &self.dev)?.to_dtype(self.dtype)?;
        let time = self.time2.forward(&candle_nn::ops::silu(&self.time0.forward(&emb)?)?)?;
        let m = self.block_mod.forward(&time)?;
        let head = self.head_mod.forward(&time)?;
        let d = self.cfg.dim;
        let row = |t: &Tensor, i: usize| t.narrow(1, i * d, d);
        Ok(([row(&m, 0)?, row(&m, 1)?, row(&m, 2)?, row(&m, 3)?, row(&m, 4)?, row(&m, 5)?], [row(&head, 0)?, row(&head, 1)?]))
    }

    /// The projected, refined text `(L, dim)` for `ctx (L, text_dim)` and the text rows of `rope`.
    fn text(&self, ctx: &Tensor, rope: &Tensor) -> Result<Tensor> {
        let mut t = self.text_proj.forward(&ctx.to_dtype(self.dtype)?)?;
        for b in &self.text_blocks {
            t = b.forward(&t, rope, self.cfg.heads)?;
        }
        Ok(t)
    }

    /// The velocity `(1, channels, h, w)` f32 for the latent `x (1, channels, h, w)` at
    /// `timestep` (σ·1000) and the encoder's hidden state `ctx (L, text_dim)`; the blocks in
    /// `skip` are left out (the guidance's negative branch skips block 22).
    pub fn forward(&self, x: &Tensor, timestep: f32, ctx: &Tensor, skip: &[usize]) -> Result<Tensor> {
        let (_, c, h, w) = x.dims4()?;
        anyhow::ensure!(c == self.cfg.channels, "a latent of {c} channels (the model takes {})", self.cfg.channels);
        let n = h * w;
        let l = ctx.dim(0)?;
        let rope = self.rope(h, w, l)?.to_dtype(self.dtype)?;
        let video = self.video_embedder.forward(&x.reshape((c, n))?.t()?.contiguous()?.to_dtype(self.dtype)?)?;
        let text = self.text(ctx, &rope.narrow(0, n, l)?)?;
        let (m, head) = self.modulation(timestep)?;
        let mut tokens = Tensor::cat(&[video, text], 0)?;
        for (i, layer) in self.layers.iter().enumerate() {
            if !skip.contains(&i) {
                tokens = layer.forward(&tokens, &rope, &m, self.cfg.heads)?;
            }
        }
        let y = self.final_norm.forward(&tokens.narrow(0, 0, n)?)?;
        let y = (&y + y.broadcast_mul(&head[1])?)?.broadcast_add(&head[0])?;
        let y = self.final_linear.forward(&y)?; // (n, channels)
        Ok(y.to_dtype(DType::F32)?.t()?.contiguous()?.reshape((1, c, h, w))?)
    }
}

/// MOSS-Image's sampling schedule: `steps` σ from 1 on the shift-5 curve, each snapped to the
/// nearest σ of the 1000-step training grid (ties to the lower), repeats dropped; the final 0
/// appended. The DiT's timestep is σ·1000.
pub fn sigmas(steps: usize, shift: f64) -> Vec<f32> {
    const TOTAL: usize = 1000;
    let grid = |count: usize, s: f64| -> Vec<f32> {
        let lower = (s / (TOTAL as f64 + s - 1.0)) as f32;
        // torch.linspace in f32
        (0..count)
            .map(|i| {
                let v = if count == 1 { 1.0 } else { linspace_f32(1.0, lower, count, i) };
                let s = s as f32;
                s * v / (1.0 + (s - 1.0) * v)
            })
            .collect()
    };
    let wanted = grid(steps.max(1), shift);
    let mut training = grid(TOTAL, 1.0);
    training.reverse(); // ascending
    let mut out: Vec<f32> = Vec::with_capacity(steps + 1);
    for s in wanted {
        let upper = training.partition_point(|&t| t < s).min(TOTAL - 1);
        let lower = upper.saturating_sub(1);
        let pick = if (s - training[lower]).abs() <= (training[upper] - s).abs() { lower } else { upper };
        let v = training[pick];
        if out.last() != Some(&v) {
            out.push(v);
        }
    }
    out.push(0.0);
    out
}

/// `torch.linspace(start, end, steps)[i]` in f32 (its two-sided evaluation).
fn linspace_f32(start: f32, end: f32, steps: usize, i: usize) -> f32 {
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    if i < half {
        start + step * i as f32
    } else {
        end - step * (steps - 1 - i) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_the_reference_at_20_steps() {
        // The official scheduler's sigmas at 20 steps, shift 5 (ref_moss.py run on torch).
        let want =
            [1.0, 0.989, 0.977, 0.964, 0.95, 0.934, 0.916, 0.896, 0.874, 0.849, 0.82, 0.786, 0.747, 0.701, 0.645, 0.577, 0.492, 0.381, 0.233, 0.024, 0.0];
        let got = sigmas(20, 5.0);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
    }

    #[test]
    fn situ_is_bounded() -> Result<()> {
        let x = Tensor::new(&[-1e4f32, -3.0, 0.0, 3.0, 1e4], &Device::Cpu)?;
        let y = situ(&x, &x)?.to_vec1::<f32>()?;
        assert!(y.iter().all(|v| v.is_finite() && v.abs() <= 100.0), "{y:?}");
        assert_eq!(y[2], 0.0);
        Ok(())
    }
}
