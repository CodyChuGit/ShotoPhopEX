//! P-H3-Image's VAE (ComfyUI `PH3ImageVAE`): the H3 3D-CNN encoder run on one frame (each
//! causal conv reduces to its last temporal tap), a posterior sample at seed 42, and a ViT
//! decoder at a fixed RoPE time coordinate. Both run in 256-pixel tiles blended as ComfyUI
//! blends them, so the pictures match its tiled output.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::Module;

use super::dit::{attention, rope, swiglu_down};
use crate::moss::weights::{QLinear, Weights};
use crate::Result;

/// The latent's spatial factor.
pub const SCALE: usize = 16;
pub const TILE: usize = 256;
pub const TILE_OVERLAP_MIN: usize = 64;
pub const ENCODE_SEED: u64 = 42;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// The decoded frame's RoPE time coordinate (`PH3_IMAGE_FRAME_POSITION`).
const FRAME_POSITION: f64 = 2.0 * (0.5 / 7.0) - 1.0;
const HEAD_DIM: usize = 64;
const REGISTERS: usize = 4;

// ---------------------------------------------------------------- encoder (f32)

/// A causal 3D conv on one frame: its last temporal tap as a 2D conv.
struct Conv {
    w: Tensor,
    b: Tensor,
    stride: usize,
}

impl Conv {
    fn load(w: &Weights, p: &str, stride: usize) -> Result<Self> {
        let k = w.get_as(&format!("{p}.weight"), DType::F32)?;
        let kt = k.dim(2)?;
        Ok(Self { w: k.narrow(2, kt - 1, 1)?.squeeze(2)?.contiguous()?, b: w.get_as(&format!("{p}.bias"), DType::F32)?, stride })
    }

    /// `x (1, C, H, W)`, already padded.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv2d(&self.w, 0, self.stride, 1, 1)?;
        Ok(y.broadcast_add(&self.b.reshape((1, (), 1, 1))?)?)
    }
}

/// Reflect padding of `(1, C, H, W)` by one pixel on the chosen sides.
fn reflect1(x: &Tensor, left: bool, right: bool, top: bool, bottom: bool) -> Result<Tensor> {
    let mut x = x.clone();
    let w = x.dim(3)?;
    let mut parts = Vec::new();
    if left {
        parts.push(x.narrow(3, 1, 1)?);
    }
    parts.push(x.clone());
    if right {
        parts.push(x.narrow(3, w - 2, 1)?);
    }
    if parts.len() > 1 {
        x = Tensor::cat(&parts, 3)?;
    }
    let h = x.dim(2)?;
    let mut parts = Vec::new();
    if top {
        parts.push(x.narrow(2, 1, 1)?);
    }
    parts.push(x.clone());
    if bottom {
        parts.push(x.narrow(2, h - 2, 1)?);
    }
    Ok(if parts.len() > 1 { Tensor::cat(&parts, 2)? } else { x })
}

fn group_norm(w: &Weights, p: &str) -> Result<candle_nn::GroupNorm> {
    let weight = w.get_as(&format!("{p}.weight"), DType::F32)?;
    let c = weight.dim(0)?;
    Ok(candle_nn::GroupNorm::new(weight, w.get_as(&format!("{p}.bias"), DType::F32)?, c, 32, 1e-6)?)
}

struct Res {
    norm1: candle_nn::GroupNorm,
    conv1: Conv,
    norm2: candle_nn::GroupNorm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl Res {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let pad = |t: &Tensor| reflect1(t, true, true, true, true);
        let h = self.conv1.forward(&pad(&candle_nn::ops::silu(&self.norm1.forward(x)?)?)?)?;
        let h = self.conv2.forward(&pad(&candle_nn::ops::silu(&self.norm2.forward(&h)?)?)?)?;
        let sc = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok((h + sc)?)
    }
}

pub struct Encoder {
    conv_in: Conv,
    levels: Vec<(Vec<Res>, Option<Conv>)>,
    norm_out: candle_nn::GroupNorm,
    conv_out: Conv,
    quant_conv: Conv,
}

impl Encoder {
    fn load(w: &Weights) -> Result<Self> {
        let mut levels = Vec::new();
        for i in 0.. {
            let p = format!("encoder.down.{i}");
            if !w.contains(&format!("{p}.block.0.conv1.weight")) {
                break;
            }
            let mut blocks = Vec::new();
            for j in 0.. {
                let b = format!("{p}.block.{j}");
                if !w.contains(&format!("{b}.conv1.weight")) {
                    break;
                }
                blocks.push(Res {
                    norm1: group_norm(w, &format!("{b}.norm1"))?,
                    conv1: Conv::load(w, &format!("{b}.conv1"), 1)?,
                    norm2: group_norm(w, &format!("{b}.norm2"))?,
                    conv2: Conv::load(w, &format!("{b}.conv2"), 1)?,
                    shortcut: if w.contains(&format!("{b}.nin_shortcut.weight")) { Some(Conv::load(w, &format!("{b}.nin_shortcut"), 1)?) } else { None },
                });
            }
            let down = if w.contains(&format!("{p}.downsample.conv.weight")) { Some(Conv::load(w, &format!("{p}.downsample.conv"), 2)?) } else { None };
            levels.push((blocks, down));
        }
        anyhow::ensure!(!levels.is_empty(), "a VAE file without the encoder");
        Ok(Self {
            conv_in: Conv::load(w, "encoder.conv_in", 1)?,
            levels,
            norm_out: group_norm(w, "encoder.norm_out")?,
            conv_out: Conv::load(w, "encoder.conv_out", 1)?,
            quant_conv: Conv::load(w, "quant_conv", 1)?,
        })
    }

    /// Normalized pixels `(1, 3, H, W)` -> moments `(1, 48, H/16, W/16)`.
    fn moments(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(&reflect1(x, true, true, true, true)?)?;
        for (blocks, down) in &self.levels {
            for b in blocks {
                h = b.forward(&h)?;
            }
            if let Some(d) = down {
                h = d.forward(&reflect1(&h, false, true, false, true)?)?;
            }
        }
        let h = self.conv_out.forward(&reflect1(&candle_nn::ops::silu(&self.norm_out.forward(&h)?)?, true, true, true, true)?)?;
        self.quant_conv.forward(&h)
    }
}

// ---------------------------------------------------------------- tiles

/// ComfyUI `split_tiles`: tile starts, lengths and the overlaps between neighbours.
pub fn split_tiles(len: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if TILE >= len {
        return (vec![0], vec![len], Vec::new());
    }
    let mut n = len.div_ceil(TILE);
    let remaining = loop {
        let used = TILE * n;
        let ov = TILE_OVERLAP_MIN * (n - 1);
        if used >= ov + len {
            break used - ov - len;
        }
        n += 1;
    };
    let mut overlaps = vec![TILE_OVERLAP_MIN; n - 1];
    for i in 0..remaining / SCALE {
        overlaps[i % (n - 1)] += SCALE;
    }
    let mut starts = vec![0];
    for o in &overlaps {
        let last = *starts.last().unwrap_or(&0);
        starts.push(last + TILE - o);
    }
    (starts, vec![TILE; n], overlaps)
}

/// ComfyUI `blend`: `b`'s first `extent` rows along `dim` fade in over `a`'s last ones.
fn blend(a: &Tensor, b: &Tensor, extent: usize, dim: usize) -> Result<Tensor> {
    let e = extent.min(a.dim(dim)?).min(b.dim(dim)?);
    if e == 0 {
        return Ok(b.clone());
    }
    let mut shape = vec![1usize; b.rank()];
    shape[dim] = e;
    let wb: Vec<f32> = (0..e).map(|k| k as f32 / e as f32).collect();
    let wa: Vec<f32> = wb.iter().map(|v| 1.0 - v).collect();
    let wa = Tensor::from_vec(wa, shape.clone(), b.device())?.to_dtype(b.dtype())?;
    let wb = Tensor::from_vec(wb, shape, b.device())?.to_dtype(b.dtype())?;
    let an = a.dim(dim)?;
    let blended = (a.narrow(dim, an - e, e)?.broadcast_mul(&wa)? + b.narrow(dim, 0, e)?.broadcast_mul(&wb)?)?;
    let bn = b.dim(dim)?;
    if e < bn {
        Ok(Tensor::cat(&[blended, b.narrow(dim, e, bn - e)?], dim)?)
    } else {
        Ok(blended)
    }
}

/// `t` without its last `n` along `dim`.
fn trim_end(t: &Tensor, dim: usize, n: usize) -> Result<Tensor> {
    Ok(t.narrow(dim, 0, t.dim(dim)? - n)?)
}

fn tail(t: &Tensor, dim: usize, n: usize) -> Result<Tensor> {
    let len = t.dim(dim)?;
    Ok(t.narrow(dim, len - n, n)?)
}

// ---------------------------------------------------------------- decoder

struct DecBlock {
    norm1: Tensor,
    qkv: QLinear,
    out: QLinear,
    scale1: Tensor,
    norm2: Tensor,
    w1: QLinear,
    w2: QLinear,
    scale2: Tensor,
}

fn rms(x: &Tensor, w: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let y = x32.broadcast_mul(&(x32.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?)?.to_dtype(x.dtype())?;
    Ok(match w {
        Some(w) => y.broadcast_mul(w)?,
        None => y,
    })
}

pub struct Decoder {
    post_quant: Tensor, // (24, 24) f32
    post_quant_b: Tensor,
    x_embedder: QLinear,
    registers: Tensor, // (4, dim)
    blocks: Vec<DecBlock>,
    norm_out: candle_nn::LayerNorm,
    proj_out: QLinear,
    heads: usize,
    inv_freq: Vec<f32>,
    dtype: DType,
}

impl Decoder {
    fn load(w: &Weights) -> Result<Self> {
        let pq = w.get_as("post_quant_conv.weight", DType::F32)?;
        let c = pq.dim(0)?;
        let mut blocks = Vec::new();
        for i in 0.. {
            let p = format!("decoder.transformer_blocks.{i}");
            if !w.contains(&format!("{p}.scale1")) {
                break;
            }
            blocks.push(DecBlock {
                norm1: w.get(&format!("{p}.norm1.weight"))?,
                qkv: w.linear(&format!("{p}.attn.to_qkv"))?,
                out: w.linear(&format!("{p}.attn.to_out"))?,
                scale1: w.get(&format!("{p}.scale1"))?,
                norm2: w.get(&format!("{p}.norm2.weight"))?,
                w1: w.linear(&format!("{p}.ff.w1"))?,
                w2: w.linear(&format!("{p}.ff.w2"))?,
                scale2: w.get(&format!("{p}.scale2"))?,
            });
        }
        anyhow::ensure!(!blocks.is_empty(), "a VAE file without the decoder");
        let registers = w.get("decoder.register_tokens")?;
        let dim = registers.dim(2)?;
        let registers = registers.reshape((registers.dim(1)?, dim))?;
        // RotaryEmbeddingND(48, 100, 3): 8 frequencies per axis
        let rope_dim = HEAD_DIM * 3 / 4;
        let step = 2.0 * 3.0 / rope_dim as f64;
        let inv_freq: Vec<f32> = (0..rope_dim / 6).map(|i| (1.0 / 100f64.powf(i as f64 * step)) as f32).collect();
        Ok(Self {
            post_quant: pq.reshape((c, c))?,
            post_quant_b: w.get_as("post_quant_conv.bias", DType::F32)?,
            x_embedder: w.linear("decoder.x_embedder")?,
            registers,
            blocks,
            norm_out: candle_nn::LayerNorm::new(w.get("decoder.norm_out.weight")?, w.get("decoder.norm_out.bias")?, 1e-5),
            proj_out: w.linear("decoder.proj_out")?,
            heads: dim / HEAD_DIM,
            inv_freq,
            dtype: w.dtype(),
        })
    }

    /// RoPE tables `(S, 24)` for an `h x w` tile and the five suffix rows: angles
    /// `2π · id · inv_freq` with the ids in the run dtype, as ComfyUI builds them.
    fn rope_tables(&self, h: usize, w: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let round = |v: f64| -> f32 {
            match self.dtype {
                DType::BF16 => half::bf16::from_f64(v).to_f32(),
                DType::F16 => half::f16::from_f64(v).to_f32(),
                _ => v as f32,
            }
        };
        let coord = |k: usize, n: usize| round(2.0 * ((k as f64 + 0.5) / n as f64) - 1.0);
        let t = round(FRAME_POSITION);
        let n = self.inv_freq.len();
        let rows = h * w + REGISTERS + 1;
        let mut cos = Vec::with_capacity(rows * 3 * n);
        let mut sin = Vec::with_capacity(rows * 3 * n);
        let mut push = |ids: [f32; 3]| {
            for id in ids {
                for f in &self.inv_freq {
                    let a = 2.0 * std::f32::consts::PI * id * f;
                    cos.push(round(f64::from(a.cos())));
                    sin.push(round(f64::from(a.sin())));
                }
            }
        };
        for i in 0..h {
            for j in 0..w {
                push([t, coord(i, h), coord(j, w)]);
            }
        }
        for _ in 0..REGISTERS + 1 {
            push([0.0, 0.0, 0.0]);
        }
        Ok((Tensor::from_vec(cos, (rows, 3 * n), dev)?, Tensor::from_vec(sin, (rows, 3 * n), dev)?))
    }

    /// Denormalized latent `(24, h, w)` f32 -> raw decoder output `(3, 16h, 16w)` f32.
    fn tile(&self, z: &Tensor) -> Result<Tensor> {
        let (c, h, w) = z.dims3()?;
        let n = h * w;
        let x = z.reshape((c, n))?.t()?.matmul(&self.post_quant.t()?)?.broadcast_add(&self.post_quant_b)?; // (n, 24)
        let x = self.x_embedder.forward(&x.to_dtype(self.dtype)?)?.to_dtype(self.dtype)?;
        let dim = x.dim(1)?;
        let zero = Tensor::zeros((1, dim), self.dtype, x.device())?;
        let mut hs = Tensor::cat(&[x, self.registers.to_dtype(self.dtype)?, zero], 0)?;
        let rows = hs.dim(0)?;
        let (cos, sin) = self.rope_tables(h, w, hs.device())?;
        let (cos, sin) = (cos.to_dtype(DType::F32)?, sin.to_dtype(DType::F32)?);
        for b in &self.blocks {
            let y = rms(&hs, Some(&b.norm1), 1e-5)?;
            let qkv = b.qkv.forward(&y)?.to_dtype(self.dtype)?.reshape((rows, self.heads, 3 * HEAD_DIM))?;
            let q = rope(&rms(&qkv.narrow(2, 0, HEAD_DIM)?, None, 1e-5)?, &cos, &sin)?;
            let k = rope(&rms(&qkv.narrow(2, HEAD_DIM, HEAD_DIM)?, None, 1e-5)?, &cos, &sin)?;
            let v = qkv.narrow(2, 2 * HEAD_DIM, HEAD_DIM)?.contiguous()?;
            let o = attention(&q.contiguous()?, &k.contiguous()?, &v)?;
            let o = b.out.forward(&o)?.to_dtype(self.dtype)?;
            hs = (&hs + o.broadcast_mul(&b.scale1)?)?;
            let y = rms(&hs, Some(&b.norm2), 1e-5)?;
            let gu = b.w1.forward(&y)?.to_dtype(self.dtype)?;
            let f = swiglu_down(&b.w2, &gu)?.to_dtype(self.dtype)?;
            hs = (&hs + f.broadcast_mul(&b.scale2)?)?;
        }
        let out = self.proj_out.forward(&self.norm_out.forward(&hs.narrow(0, 0, n)?)?)?.to_dtype(DType::F32)?;
        let oc = out.dim(1)? / (SCALE * SCALE);
        let img = out.reshape((h, w, oc, SCALE, SCALE))?.permute((2, 0, 3, 1, 4))?; // c, h, py, w, px
        Ok(img.contiguous()?.reshape((oc, h * SCALE, w * SCALE))?)
    }
}

// ---------------------------------------------------------------- the VAE

pub struct H3Vae {
    encoder: Option<Encoder>,
    decoder: Decoder,
    mean: Vec<f32>,
    std: Vec<f32>,
    dev: Device,
}

impl H3Vae {
    pub fn load(w: &Weights) -> Result<Self> {
        let encoder = if w.contains("encoder.conv_in.weight") { Some(Encoder::load(w)?) } else { None };
        let vec = |n: &str| -> Result<Vec<f32>> { Ok(w.get_as(n, DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?) };
        Ok(Self { encoder, decoder: Decoder::load(w)?, mean: vec("latents_mean")?, std: vec("latents_std")?, dev: w.device().clone() })
    }

    pub fn can_encode(&self) -> bool {
        self.encoder.is_some()
    }

    fn channel(&self, v: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_slice(v, (v.len(), 1, 1), &self.dev)?)
    }

    /// Normalized latent `(24, 1, h, w)` -> pixels `(3, 16h, 16w)` f32 in `[0, 1]`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = z.to_device(&self.dev)?.to_dtype(DType::F32)?.squeeze(1)?;
        let z = z.broadcast_mul(&self.channel(&self.std)?)?.broadcast_add(&self.channel(&self.mean)?)?;
        let (_, lh, lw) = z.dims3()?;
        let (height, width) = (lh * SCALE, lw * SCALE);
        let (y_idx, y_len, y_ov) = split_tiles(height);
        let (x_idx, x_len, x_ov) = split_tiles(width);
        let mut out_rows: Vec<Tensor> = Vec::new();
        let mut strip: Option<Tensor> = None;
        for i in 0..y_idx.len() {
            let zrow = z.narrow(1, y_idx[i] / SCALE, y_len[i] / SCALE)?;
            let mut new_strip: Vec<Tensor> = Vec::new();
            let mut row: Vec<Tensor> = Vec::new();
            let mut left_tail: Option<Tensor> = None;
            for j in 0..x_idx.len() {
                let mut tile = self.decoder.tile(&zrow.narrow(2, x_idx[j] / SCALE, x_len[j] / SCALE)?)?;
                if i > 0 {
                    let s = strip.as_ref().ok_or_else(|| anyhow::anyhow!("tile strip"))?;
                    tile = blend(&s.narrow(2, x_idx[j], x_len[j])?, &tile, y_ov[i - 1], 1)?;
                }
                if j > 0 {
                    let lt = left_tail.as_ref().ok_or_else(|| anyhow::anyhow!("tile tail"))?;
                    tile = blend(lt, &tile, x_ov[j - 1], 2)?;
                }
                if j + 1 < x_idx.len() {
                    left_tail = Some(tail(&tile, 2, x_ov[j])?);
                    tile = trim_end(&tile, 2, x_ov[j])?;
                }
                if i + 1 < y_idx.len() {
                    new_strip.push(tail(&tile, 1, y_ov[i])?);
                    tile = trim_end(&tile, 1, y_ov[i])?;
                }
                row.push(tile);
            }
            out_rows.push(Tensor::cat(&row, 2)?);
            strip = if new_strip.is_empty() { None } else { Some(Tensor::cat(&new_strip, 2)?) };
        }
        let raw = Tensor::cat(&out_rows, 1)?;
        Ok(raw.broadcast_mul(&self.channel(&IMAGENET_STD)?)?.broadcast_add(&self.channel(&IMAGENET_MEAN)?)?.clamp(0f32, 1f32)?)
    }

    /// Pixels `(3, H, W)` in `[-1, 1]` (sides multiples of 16) -> the normalized posterior
    /// sample `(24, 1, H/16, W/16)` f32 ComfyUI conditions on (seed 42).
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let enc = self.encoder.as_ref().ok_or_else(|| anyhow::anyhow!("this VAE file has no encoder; an edit needs it"))?;
        let x = x.to_device(&self.dev)?.to_dtype(DType::F32)?;
        let x = ((x + 1.0)? * 0.5)?.broadcast_sub(&self.channel(&IMAGENET_MEAN)?)?.broadcast_div(&self.channel(&IMAGENET_STD)?)?;
        let (_, height, width) = x.dims3()?;
        let (y_idx, y_len, y_ov) = split_tiles(height);
        let (x_idx, x_len, x_ov) = split_tiles(width);
        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        for (yi, yl) in y_idx.iter().zip(&y_len) {
            let mut row = Vec::new();
            for (xi, xl) in x_idx.iter().zip(&x_len) {
                let t = x.narrow(1, *yi, *yl)?.narrow(2, *xi, *xl)?.unsqueeze(0)?;
                row.push(enc.moments(&t)?.squeeze(0)?);
            }
            rows.push(row);
        }
        let ly: Vec<usize> = y_ov.iter().map(|o| o / SCALE).collect();
        let lx: Vec<usize> = x_ov.iter().map(|o| o / SCALE).collect();
        let mut out_rows = Vec::new();
        for i in 0..rows.len() {
            let mut out = Vec::new();
            for j in 0..rows[i].len() {
                let mut tile = rows[i][j].clone();
                if i > 0 {
                    tile = blend(&rows[i - 1][j], &tile, ly[i - 1], 1)?;
                }
                if j > 0 {
                    tile = blend(&rows[i][j - 1], &tile, lx[j - 1], 2)?;
                }
                if i + 1 < rows.len() {
                    tile = trim_end(&tile, 1, ly[i])?;
                }
                if j + 1 < rows[i].len() {
                    tile = trim_end(&tile, 2, lx[j])?;
                }
                out.push(tile);
            }
            out_rows.push(Tensor::cat(&out, 2)?);
        }
        let moments = Tensor::cat(&out_rows, 1)?;
        let c = moments.dim(0)? / 2;
        let mean = moments.narrow(0, 0, c)?;
        let logvar = moments.narrow(0, c, c)?.clamp(-30f32, 20f32)?;
        let noise = crate::comfy::torch_randn(ENCODE_SEED, mean.elem_count()).map_err(anyhow::Error::msg)?;
        let noise = Tensor::from_vec(noise, mean.dims(), &self.dev)?;
        let z = (mean + (logvar * 0.5)?.exp()?.mul(&noise)?)?;
        let z = z.broadcast_sub(&self.channel(&self.mean)?)?.broadcast_div(&self.channel(&self.std)?)?;
        Ok(z.unsqueeze(1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_match_comfy() {
        assert_eq!(split_tiles(200), (vec![0], vec![200], vec![]));
        // 384: two tiles, the 64 px slack spread over the one overlap
        assert_eq!(split_tiles(384), (vec![0, 128], vec![256, 256], vec![128]));
        let (s, l, o) = split_tiles(2912);
        assert_eq!(l.len(), s.len());
        assert_eq!(s.last().copied().unwrap() + 256, 2912);
        assert!(o.iter().all(|v| v % 16 == 0 && *v >= 64));
    }
}
