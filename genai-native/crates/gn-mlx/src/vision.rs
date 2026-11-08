//! Qwen3.5-9B's vision tower (`model.visual` in MOSS's text-encoder file) on MLX, as
//! `qwen_image_rs::model::vision` runs Qwen3-VL's (the same tower without DeepStack): 16×16
//! patches with the frame doubled as the temporal pair, the learned 48×48 position table
//! resampled to the grid, 27 blocks with 2D RoPE, and the 2×2 merger to the language model's
//! 4096. Its linears are ConvRot INT8 like the language model's, but rotated in groups of 64 and
//! 16 (qir's kernels take 256): dequantized here once and cached in f16.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype, fast, ops};
use qwen_image_rs::model::vision::VisionConfig;

use crate::nn::{Linear, Params, cols, dim, get, shape};

const P: &str = "model.visual";

/// Qwen3.5-9B's tower: Qwen3-VL-8B's sizes without DeepStack.
pub fn config() -> VisionConfig {
    VisionConfig { deepstack_visual_indexes: Vec::new(), ..VisionConfig::default() }
}

/// ComfyUI's ConvRot rotation of `g` inputs `(g, g)` f32, `g` a power of 4: the regular
/// Hadamard `R` (the Kronecker power of `H4 / 2`, −1 on its diagonal) with its rows reversed,
/// `H[i][j] = R[g − 1 − i][j]` (qir's 256 case: `comfy_int8::convert`). A stored weight `W'` is
/// `W` with its inputs rotated, and `W = W' H` group-wise.
pub fn comfy_rotation(g: usize) -> Result<Array> {
    anyhow::ensure!(g >= 4 && g.is_power_of_two() && g.trailing_zeros() % 2 == 0, "ConvRot group {g} is not a power of 4");
    let digits = g.trailing_zeros() / 2;
    let r = |mut i: usize, mut j: usize| -> f32 {
        let mut v = 1.0f32;
        for _ in 0..digits {
            v *= if i % 4 == j % 4 { -0.5 } else { 0.5 };
            i /= 4;
            j /= 4;
        }
        v
    };
    let data: Vec<f32> = (0..g).flat_map(|i| (0..g).map(move |j| r(g - 1 - i, j))).collect();
    Ok(Array::from_slice(&data, &shape(&[g, g])?))
}

/// The ConvRot descriptor's group size (`{"format": "int8_tensorwise", "convrot": true,
/// "convrot_groupsize": 64}`).
fn group_of(t: &HashMap<String, Array>, module: &str) -> Result<usize> {
    let raw = t.get(&format!("{module}.comfy_quant")).with_context(|| format!("{module}: no comfy_quant descriptor"))?;
    raw.eval()?;
    let bytes: Vec<u8> = raw.as_slice::<u8>().to_vec();
    let d: serde_json::Value = serde_json::from_slice(&bytes).with_context(|| format!("{module}: comfy_quant descriptor"))?;
    anyhow::ensure!(d["format"] == "int8_tensorwise" && d["convrot"] == true, "{module}: {d} (only int8_tensorwise ConvRot is read)");
    d["convrot_groupsize"].as_u64().map(|g| g as usize).context("no convrot_groupsize")
}

/// `module`'s dense weight `(out, in)` f32 from its INT8 rows, row scales and rotation.
pub fn dequantize(t: &HashMap<String, Array>, module: &str) -> Result<Array> {
    let g = group_of(t, module)?;
    let w = t.get(&format!("{module}.weight")).with_context(|| format!("{module}.weight"))?;
    let s = t.get(&format!("{module}.weight_scale")).with_context(|| format!("{module}.weight_scale"))?;
    let sh = w.shape();
    let (n, k) = (sh[0], sh[1]);
    anyhow::ensure!(k as usize % g == 0, "{module}: {k} inputs, groups of {g}");
    let w = w.as_dtype(Dtype::Float32)?.multiply(s.as_dtype(Dtype::Float32)?.reshape(&[n, 1])?)?;
    let gi = dim(g)?;
    Ok(w.reshape(&[n * (k / gi), gi])?.matmul(&comfy_rotation(g)?)?.reshape(&[n, k])?)
}

/// Convert the tower from `text` (the encoder file) into `out`: the INT8 linears dequantized,
/// everything in f16 (norms and biases included; the tower runs in f16 with f32 norms).
pub fn convert(text: &Path, out: &Path) -> Result<()> {
    let cfg = config();
    let t = Array::load_safetensors(text).with_context(|| format!("read {}", text.display()))?;
    let f16 = |a: &Array| -> Result<Array> { Ok(a.as_dtype(Dtype::Float16)?) };
    let mut p: Params = HashMap::new();
    let mut put = |name: &str, a: Array| -> Result<()> {
        a.eval()?;
        p.insert(name.to_string(), a);
        Ok(())
    };
    for n in ["patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight", "merger.norm.weight", "merger.norm.bias"] {
        let name = format!("{P}.{n}");
        put(&name, f16(t.get(&name).with_context(|| name.clone())?)?)?;
    }
    for m in ["merger.linear_fc1", "merger.linear_fc2"] {
        for s in ["weight", "bias"] {
            let name = format!("{P}.{m}.{s}");
            put(&name, f16(t.get(&name).with_context(|| name.clone())?)?)?;
        }
    }
    for i in 0..cfg.depth {
        let b = format!("{P}.blocks.{i}");
        for n in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn.qkv.bias", "attn.proj.bias", "mlp.linear_fc1.bias", "mlp.linear_fc2.bias"] {
            let name = format!("{b}.{n}");
            put(&name, f16(t.get(&name).with_context(|| name.clone())?)?)?;
        }
        for m in ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"] {
            let module = format!("{b}.{m}");
            put(&format!("{module}.weight"), f16(&dequantize(&t, &module)?)?)?;
        }
    }
    drop(t);
    crate::convert::save(&p, out)
}

struct Block {
    norm1: (Array, Array),
    norm2: (Array, Array),
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

/// A converted tower ([`convert`]), held while it runs.
pub struct VisionTower {
    cfg: VisionConfig,
    patch_w: Array,
    patch_b: Array,
    pos: Array,
    blocks: Vec<Block>,
    merger_norm: (Array, Array),
    fc1: Linear,
    fc2: Linear,
}

/// The tower's output for one picture: the rows that replace `<|image_pad|>` and the merged grid.
pub struct VisionFeatures {
    /// `(rows · cols, 4096)` f32.
    pub merged: Array,
    pub rows: usize,
    pub cols: usize,
}

fn layer_norm(x: &Array, wb: &(Array, Array)) -> Result<Array> {
    Ok(fast::layer_norm(x.as_dtype(Dtype::Float32)?, Some(&wb.0.as_dtype(Dtype::Float32)?), Some(&wb.1.as_dtype(Dtype::Float32)?), 1e-6)?)
}

impl VisionTower {
    pub fn load(path: &Path) -> Result<Self> {
        let cfg = config();
        let p = crate::convert::load(path)?;
        let pd = dim(cfg.patch_dim())?;
        let h = dim(cfg.hidden_size)?;
        let norm = |n: &str| -> Result<(Array, Array)> { Ok((get(&p, &format!("{n}.weight"))?, get(&p, &format!("{n}.bias"))?)) };
        let blocks = (0..cfg.depth)
            .map(|i| -> Result<Block> {
                let b = format!("{P}.blocks.{i}");
                Ok(Block {
                    norm1: norm(&format!("{b}.norm1"))?,
                    norm2: norm(&format!("{b}.norm2"))?,
                    qkv: Linear::load(&p, &format!("{b}.attn.qkv"))?,
                    proj: Linear::load(&p, &format!("{b}.attn.proj"))?,
                    fc1: Linear::load(&p, &format!("{b}.mlp.linear_fc1"))?,
                    fc2: Linear::load(&p, &format!("{b}.mlp.linear_fc2"))?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            patch_w: get(&p, &format!("{P}.patch_embed.proj.weight"))?.reshape(&[h, pd])?,
            patch_b: get(&p, &format!("{P}.patch_embed.proj.bias"))?,
            pos: get(&p, &format!("{P}.pos_embed.weight"))?,
            merger_norm: norm(&format!("{P}.merger.norm"))?,
            fc1: Linear::load(&p, &format!("{P}.merger.linear_fc1"))?,
            fc2: Linear::load(&p, &format!("{P}.merger.linear_fc2"))?,
            blocks,
            cfg,
        })
    }

    /// The tower on straight RGBA8 `rgba` (`w × h`, resized here to the tower's grid with
    /// [`qwen_image_rs::model::vision::vit_size`] as the Windows chat does).
    pub fn forward(&self, rgba: &[u8], w: usize, h: usize) -> Result<VisionFeatures> {
        use qwen_image_rs::model::vision as qv;
        let cfg = &self.cfg;
        let (vw, vh) = qv::vit_size(w, h, 3136, 12_845_056);
        // Alpha over white, as ComfyUI's node composites, in [0, 1], (3, H, W).
        let mut planes = vec![0f32; 3 * w * h];
        for (i, px) in rgba.chunks_exact(4).enumerate().take(w * h) {
            let a = f32::from(px[3]) / 255.0;
            for c in 0..3 {
                planes[c * w * h + i] = (f32::from(px[c]) / 255.0) * a + (1.0 - a);
            }
        }
        let rgb = candle_core::Tensor::from_vec(planes, (3, h, w), &candle_core::Device::Cpu)?;
        let rgb = if (vw, vh) == (w, h) { rgb } else { qv::resize_bilinear(&rgb, vh, vw)? };
        let (patches, gh, gw) = qv::patchify(&rgb, cfg)?;
        let n = gh * gw;
        let x = crate::nn::from_candle(&patches)?.as_dtype(Dtype::Float16)?;
        let x = x.matmul(self.patch_w.t())?.add(&self.patch_b)?.as_dtype(Dtype::Float32)?;
        let x = x.add(self.position_rows(gh, gw)?)?;
        let (cos, sin) = rotary(cfg, gh, gw)?;
        let (heads, hd) = (cfg.num_heads, cfg.head_dim());
        let mut x = x;
        for b in &self.blocks {
            let y = layer_norm(&x, &b.norm1)?.as_dtype(Dtype::Float16)?;
            let qkv = b.qkv.forward(&y)?.reshape(&[dim(n)?, 3, dim(heads)?, dim(hd)?])?;
            let part = |i: i32| -> Result<Array> { Ok(qkv.take_axis(&Array::from_int(i), 1)?.as_dtype(Dtype::Float32)?) };
            let (q, k, v) = (rope_half(&part(0)?, &cos, &sin)?, rope_half(&part(1)?, &cos, &sin)?, part(2)?);
            let first = |t: &Array| -> Result<Array> { Ok(t.as_dtype(Dtype::Float16)?.transpose_axes(&[1, 0, 2])?.expand_dims(0)?) };
            let o = fast::scaled_dot_product_attention(first(&q)?, first(&k)?, first(&v)?, 1.0 / (hd as f32).sqrt(), None, None)?;
            let o = o.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[dim(n)?, dim(heads * hd)?])?;
            x = x.add(b.proj.forward(&o)?.as_dtype(Dtype::Float32)?)?;
            let y = layer_norm(&x, &b.norm2)?.as_dtype(Dtype::Float16)?;
            // gelu_pytorch_tanh, as the config names it.
            let m = b.fc2.forward(&mlx_rs::nn::gelu_approximate(b.fc1.forward(&y)?)?)?;
            x = x.add(m.as_dtype(Dtype::Float32)?)?;
            x.eval()?;
        }
        let unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merged = layer_norm(&x, &self.merger_norm)?.reshape(&[dim(n / unit)?, dim(cfg.hidden_size * unit)?])?.as_dtype(Dtype::Float16)?;
        // The merger's exact GELU.
        let merged = self.fc2.forward(&mlx_rs::nn::gelu(self.fc1.forward(&merged)?)?)?.as_dtype(Dtype::Float32)?;
        merged.eval()?;
        let m = cfg.spatial_merge_size;
        Ok(VisionFeatures { merged, rows: gh / m, cols: gw / m })
    }

    /// The 48×48 table bilinearly resampled to the `gh × gw` grid, in merge order (qir's
    /// `pos_embed_grid`, in f32).
    fn position_rows(&self, gh: usize, gw: usize) -> Result<Array> {
        let cfg = &self.cfg;
        let side = (cfg.num_position_embeddings as f64).sqrt() as usize;
        let lin = |n: usize| -> Vec<f32> { if n == 1 { vec![0.0] } else { (0..n).map(|i| (side - 1) as f32 * i as f32 / (n - 1) as f32).collect() } };
        let (hs, ws) = (lin(gh), lin(gw));
        let m = cfg.spatial_merge_size;
        let mut idx: [Vec<u32>; 4] = Default::default();
        let mut wts: [Vec<f32>; 4] = Default::default();
        for br in 0..gh / m {
            for bc in 0..gw / m {
                for ir in 0..m {
                    for ic in 0..m {
                        let (hi, wi) = (hs[br * m + ir], ws[bc * m + ic]);
                        let (h0, w0) = (hi.floor() as usize, wi.floor() as usize);
                        let (dh, dw) = (hi - h0 as f32, wi - w0 as f32);
                        let (h1, w1) = ((h0 + 1).min(side - 1), (w0 + 1).min(side - 1));
                        for (j, (r, c, wgt)) in
                            [(h0, w0, (1.0 - dh) * (1.0 - dw)), (h0, w1, (1.0 - dh) * dw), (h1, w0, dh * (1.0 - dw)), (h1, w1, dh * dw)].into_iter().enumerate()
                        {
                            idx[j].push((r * side + c) as u32);
                            wts[j].push(wgt);
                        }
                    }
                }
            }
        }
        let n = dim(gh * gw)?;
        let table = self.pos.as_dtype(Dtype::Float32)?;
        let mut out: Option<Array> = None;
        for j in 0..4 {
            let rows = table.take_axis(&Array::from_slice(&idx[j], &[n]), 0)?;
            let term = rows.multiply(Array::from_slice(&wts[j], &[n, 1]))?;
            out = Some(match out {
                Some(o) => o.add(&term)?,
                None => term,
            });
        }
        out.context("no position rows")
    }
}

/// The 2D RoPE for the grid in merge order: `(N, 1, head_dim)` cos and sin, the first 18 pairs
/// by row and the next 18 by column, the halves repeated (qir's `rotary` for candle's rope).
fn rotary(cfg: &VisionConfig, gh: usize, gw: usize) -> Result<(Array, Array)> {
    let half = cfg.head_dim() / 2;
    let nfreq = half / 2;
    let inv: Vec<f32> = (0..nfreq).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / half as f32)).collect();
    let m = cfg.spatial_merge_size;
    let mut angles = Vec::with_capacity(gh * gw * 2 * half);
    for br in 0..gh / m {
        for bc in 0..gw / m {
            for ir in 0..m {
                for ic in 0..m {
                    let (row, col) = ((br * m + ir) as f32, (bc * m + ic) as f32);
                    let a: Vec<f32> = inv.iter().map(|f| row * f).chain(inv.iter().map(|f| col * f)).collect();
                    angles.extend_from_slice(&a);
                    angles.extend_from_slice(&a);
                }
            }
        }
    }
    let a = Array::from_slice(&angles, &shape(&[gh * gw, 1, 2 * half])?);
    Ok((a.cos()?, a.sin()?))
}

/// Rotate-half RoPE over the whole head `x (N, H, D)` f32.
fn rope_half(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let d = *x.shape().last().context("0-d")? as usize;
    let h = d / 2;
    let turned = ops::concatenate(&[cols(x, h, d)?.negative()?, cols(x, 0, h)?], 2)?;
    Ok(x.multiply(cos)?.add(turned.multiply(sin)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rotation is orthogonal at every group size, and at 256 it is qir's with its rows
    /// reversed (what `comfy_int8::convert` undoes by reversing each group's columns).
    #[test]
    fn comfy_rotation_matches_qirs_at_256_and_is_orthogonal() -> Result<()> {
        for g in [16usize, 64, 256] {
            let h = comfy_rotation(g)?;
            let eye = h.matmul(h.t())?;
            let v = crate::nn::to_vec(&eye)?;
            for i in 0..g {
                for j in 0..g {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((v[i * g + j] - want).abs() < 1e-5, "g {g}: H Hᵀ[{i}][{j}] = {}", v[i * g + j]);
                }
            }
        }
        let qir = qwen_image_rs::model::rotation::regular_hadamard_256(&candle_core::Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let ours = crate::nn::to_vec(&comfy_rotation(256)?)?;
        for i in 0..256 {
            for j in 0..256 {
                assert_eq!(ours[i * 256 + j], qir[(255 - i) * 256 + j]);
            }
        }
        Ok(())
    }
}
