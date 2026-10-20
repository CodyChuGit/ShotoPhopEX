//! Qwen3-VL's vision tower (the Qwen3.5-style ViT with DeepStack), as ComfyUI's
//! `Qwen3VLVisionModel` runs it for the Qwen-Image-2.1 edit route: a reference picture goes
//! through it and its merged features replace the `<|image_pad|>` token in the prompt, while
//! the three DeepStack features are added to the first three decoder layers' hidden states at
//! those positions ([`crate::model::text_encoder`]).
//!
//! Everything runs in f32, as ComfyUI casts the pixels to f32 before the tower (the weights
//! follow the input). One picture at a time (one `grid`), 16×16 patches with the frame
//! duplicated as the temporal pair, 2×2 spatial merge.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use crate::Result;

/// Qwen3-VL-8B's vision config (`text_encoder/config.json` `vision_config`).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub depth: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
    pub out_hidden_size: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            num_heads: 16,
            intermediate_size: 4304,
            depth: 27,
            patch_size: 16,
            temporal_patch_size: 2,
            in_channels: 3,
            spatial_merge_size: 2,
            num_position_embeddings: 2304,
            deepstack_visual_indexes: vec![8, 16, 24],
            out_hidden_size: 4096,
        }
    }
}

impl VisionConfig {
    /// Values per flattened patch: `C · T · P · P`.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// `process_qwen2vl_images` for one picture: the side lengths it resizes to (multiples of
/// `patch · merge` = 32, rounded to nearest, between `min_pixels` and `max_pixels`).
pub fn vit_size(width: usize, height: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let factor = 32.0;
    let round = |v: f64| ((v / factor).round() * factor) as usize;
    let (mut w_bar, mut h_bar) = (round(width as f64), round(height as f64));
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / factor).floor() * factor).max(factor) as usize;
        w_bar = ((width as f64 / beta / factor).floor() * factor).max(factor) as usize;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor).ceil() * factor) as usize;
        w_bar = ((width as f64 * beta / factor).ceil() * factor) as usize;
    }
    (w_bar, h_bar)
}

/// The ViT's input for one picture already at a `vit_size`: `rgb` is `(3, H, W)` in `[0, 1]`
/// (ComfyUI's image tensor, alpha composited over white by the node), normalised with
/// mean/std 0.5, the frame duplicated as the temporal pair, cut into 16×16 patches ordered
/// `[block row][block col][row in block][col in block]` for the 2×2 merge, each patch
/// flattened `[c][t][py][px]`. Returns `(patches (N, 1536), grid_h, grid_w)`.
pub fn patchify(rgb: &Tensor, cfg: &VisionConfig) -> Result<(Tensor, usize, usize)> {
    let (c, h, w) = rgb.dims3()?;
    let (p, m, tp) = (cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size);
    anyhow::ensure!(
        c == cfg.in_channels && h.is_multiple_of(p * m) && w.is_multiple_of(p * m),
        "patchify: ({c}, {h}, {w}) is not 3 channels at multiples of {}",
        p * m
    );
    let (gh, gw) = (h / p, w / p);
    let x = ((rgb.to_dtype(DType::F32)? - 0.5)? / 0.5)?;
    // Gather on the host: (c, h, w) -> rows [br][bc][ir][ic], columns [c][t][py][px] (the
    // frame repeated for t). Eight axes is more than candle reshapes in one go.
    let src = x.flatten_all()?.to_vec1::<f32>()?;
    let pd = cfg.patch_dim();
    let mut out = vec![0f32; gh * gw * pd];
    let mut k = 0;
    for br in 0..gh / m {
        for bc in 0..gw / m {
            for ir in 0..m {
                for ic in 0..m {
                    let (r0, c0) = ((br * m + ir) * p, (bc * m + ic) * p);
                    let row = &mut out[k * pd..(k + 1) * pd];
                    for ch in 0..c {
                        for t in 0..tp {
                            for py in 0..p {
                                for px in 0..p {
                                    row[((ch * tp + t) * p + py) * p + px] = src[(ch * h + r0 + py) * w + c0 + px];
                                }
                            }
                        }
                    }
                    k += 1;
                }
            }
        }
    }
    Ok((Tensor::from_vec(out, (gh * gw, pd), rgb.device())?, gh, gw))
}

/// `F.interpolate(mode="bilinear", align_corners=False)` of a `(3, H, W)` picture to
/// `dh × dw`, on the host (a reference only passes here when it is over the tower's pixel cap).
pub fn resize_bilinear(rgb: &Tensor, dh: usize, dw: usize) -> Result<Tensor> {
    let (c, sh, sw) = rgb.dims3()?;
    let src = rgb.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let taps = |n_src: usize, n_dst: usize| -> Vec<(usize, usize, f32)> {
        let scale = n_src as f32 / n_dst as f32;
        (0..n_dst)
            .map(|i| {
                let x = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
                let x0 = (x.floor() as usize).min(n_src - 1);
                let x1 = (x0 + 1).min(n_src - 1);
                (x0, x1, x - x0 as f32)
            })
            .collect()
    };
    let (ty, tx) = (taps(sh, dh), taps(sw, dw));
    let mut out = vec![0f32; c * dh * dw];
    for ch in 0..c {
        for (y, &(y0, y1, fy)) in ty.iter().enumerate() {
            for (x, &(x0, x1, fx)) in tx.iter().enumerate() {
                let p = |yy: usize, xx: usize| src[(ch * sh + yy) * sw + xx];
                let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                out[(ch * dh + y) * dw + x] = top * (1.0 - fy) + bot * fy;
            }
        }
    }
    Ok(Tensor::from_vec(out, (c, dh, dw), rgb.device())?)
}

/// A 2D RoPE table for the patch grid: token `k` (in merge order) at `(row, col)` gets the
/// frequencies `[row · f, col · f]` over the 18 + 18 pairs. Returns `(cos, sin)` of
/// `(N, head_dim / 2)`, the halves candle's `rope` applies to the split-half layout.
fn rotary(cfg: &VisionConfig, gh: usize, gw: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = cfg.head_dim() / 2; // 36
    let dim = half; // the rotary module's `dim`
    let nfreq = dim / 2; // 18
    let inv: Vec<f32> = (0..nfreq).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim as f32)).collect();
    let m = cfg.spatial_merge_size;
    let n = gh * gw;
    let mut cos = vec![0f32; n * half];
    let mut sin = vec![0f32; n * half];
    let mut k = 0;
    for br in 0..gh / m {
        for bc in 0..gw / m {
            for ir in 0..m {
                for ic in 0..m {
                    let (row, col) = ((br * m + ir) as f32, (bc * m + ic) as f32);
                    for (j, f) in inv.iter().enumerate() {
                        let (a, b) = (row * f, col * f);
                        cos[k * half + j] = a.cos();
                        sin[k * half + j] = a.sin();
                        cos[k * half + nfreq + j] = b.cos();
                        sin[k * half + nfreq + j] = b.sin();
                    }
                    k += 1;
                }
            }
        }
    }
    Ok((Tensor::from_vec(cos, (n, half), dev)?, Tensor::from_vec(sin, (n, half), dev)?))
}

/// The learned 48×48 position table bilinearly resampled to the `gh × gw` patch grid, in the
/// merge token order (`fast_pos_embed_interpolate`).
fn pos_embed(table: &Tensor, cfg: &VisionConfig, gh: usize, gw: usize) -> Result<Tensor> {
    let side = (cfg.num_position_embeddings as f64).sqrt() as usize; // 48
    let lin = |n: usize| -> Vec<f32> {
        if n == 1 {
            vec![0.0]
        } else {
            (0..n).map(|i| (side - 1) as f32 * i as f32 / (n - 1) as f32).collect()
        }
    };
    let (hs, ws) = (lin(gh), lin(gw));
    let m = cfg.spatial_merge_size;
    let mut idx = [Vec::with_capacity(gh * gw), Vec::new(), Vec::new(), Vec::new()];
    let mut wts = [Vec::with_capacity(gh * gw), Vec::new(), Vec::new(), Vec::new()];
    for br in 0..gh / m {
        for bc in 0..gw / m {
            for ir in 0..m {
                for ic in 0..m {
                    let (hi, wi) = (hs[br * m + ir], ws[bc * m + ic]);
                    let (hf, wf) = (hi.floor(), wi.floor());
                    let (dh, dw) = (hi - hf, wi - wf);
                    let (h0, w0) = (hf as usize, wf as usize);
                    let (h1, w1) = ((h0 + 1).min(side - 1), (w0 + 1).min(side - 1));
                    let cells = [(h0, w0, (1.0 - dh) * (1.0 - dw)), (h0, w1, (1.0 - dh) * dw), (h1, w0, dh * (1.0 - dw)), (h1, w1, dh * dw)];
                    for (j, (r, c, wgt)) in cells.iter().enumerate() {
                        idx[j].push((r * side + c) as u32);
                        wts[j].push(*wgt);
                    }
                }
            }
        }
    }
    let n = gh * gw;
    // ComfyUI interpolates in the table's own dtype, bf16: the four weights, the products and
    // the sums are bf16 values before the f32 add to the patches. Done the same way here.
    let table = table.to_dtype(DType::BF16)?;
    let mut out: Option<Tensor> = None;
    for j in 0..4 {
        let rows = table.index_select(&Tensor::from_vec(idx[j].clone(), n, table.device())?, 0)?;
        let w = Tensor::from_vec(wts[j].clone(), (n, 1), table.device())?.to_dtype(DType::BF16)?;
        let term = rows.broadcast_mul(&w)?;
        out = Some(match out {
            Some(o) => (o + term)?,
            None => term,
        });
    }
    Ok(out.ok_or_else(|| anyhow::anyhow!("no position embedding"))?.to_dtype(DType::F32)?)
}

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    fn load(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            norm1: layer_norm(h, 1e-6, vb.pp("norm1"))?,
            norm2: layer_norm(h, 1e-6, vb.pp("norm2"))?,
            qkv: linear(h, 3 * h, vb.pp("attn").pp("qkv"))?,
            proj: linear(h, h, vb.pp("attn").pp("proj"))?,
            fc1: linear(h, i, vb.pp("mlp").pp("linear_fc1"))?,
            fc2: linear(i, h, vb.pp("mlp").pp("linear_fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor, cfg: &VisionConfig, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (n, h) = x.dims2()?;
        let (heads, hd) = (cfg.num_heads, cfg.head_dim());
        let qkv = self.qkv.forward(&self.norm1.forward(x)?)?.reshape((n, 3, heads, hd))?;
        // (1, heads, N, hd) for candle's rope and the matmuls
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(1, i, 1)?.squeeze(1)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?) };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.squeeze(0)?.transpose(0, 1)?.reshape((n, h))?;
        let x = (x + self.proj.forward(&out)?)?;
        let m = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu()?)?;
        Ok((x + m)?)
    }
}

/// The main merger (norm per token, 2×2 merge, two linears with an exact GELU) and the
/// DeepStack mergers (merge first, then norm over the merged width).
struct Merger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    norm_after_merge: bool,
}

impl Merger {
    fn load(cfg: &VisionConfig, norm_after_merge: bool, vb: VarBuilder) -> Result<Self> {
        let md = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        Ok(Self {
            norm: layer_norm(if norm_after_merge { md } else { cfg.hidden_size }, 1e-6, vb.pp("norm"))?,
            fc1: linear(md, md, vb.pp("linear_fc1"))?,
            fc2: linear(md, cfg.out_hidden_size, vb.pp("linear_fc2"))?,
            norm_after_merge,
        })
    }

    fn forward(&self, x: &Tensor, cfg: &VisionConfig) -> Result<Tensor> {
        let (n, h) = x.dims2()?;
        let unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let x =
            if self.norm_after_merge { self.norm.forward(&x.reshape((n / unit, h * unit))?)? } else { self.norm.forward(x)?.reshape((n / unit, h * unit))? };
        Ok(self.fc2.forward(&self.fc1.forward(&x)?.gelu_erf()?)?)
    }
}

/// What the tower gives the text encoder for one picture.
pub struct VisionFeatures {
    /// `(N / 4, 4096)`: one row per merged token, the rows that replace `<|image_pad|>`.
    pub merged: Tensor,
    /// The DeepStack features for decoder layers 0, 1, 2, same shape.
    pub deepstack: Vec<Tensor>,
    /// The merged grid: `grid_h / 2` rows of `grid_w / 2` tokens.
    pub rows: usize,
    pub cols: usize,
}

impl VisionFeatures {
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VisionTower {
    cfg: VisionConfig,
    patch_weight: Tensor, // (hidden, patch_dim)
    patch_bias: Tensor,
    pos_table: Tensor, // (2304, hidden)
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
    device: Device,
}

impl VisionTower {
    /// Load from the encoder's VarBuilder rooted at `model.visual` (f32).
    pub fn load(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let (h, pd) = (cfg.hidden_size, cfg.patch_dim());
        let proj = vb.pp("patch_embed").pp("proj");
        let patch_weight = proj.get((h, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size), "weight")?.reshape((h, pd))?;
        let patch_bias = proj.get(h, "bias")?;
        let pos_table = vb.pp("pos_embed").get((cfg.num_position_embeddings, h), "weight")?;
        let bvb = vb.pp("blocks");
        let blocks = (0..cfg.depth).map(|i| Block::load(cfg, bvb.pp(i.to_string()))).collect::<Result<Vec<_>>>()?;
        let dvb = vb.pp("deepstack_merger_list");
        let deepstack = (0..cfg.deepstack_visual_indexes.len()).map(|i| Merger::load(cfg, true, dvb.pp(i.to_string()))).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            patch_weight,
            patch_bias,
            pos_table,
            blocks,
            merger: Merger::load(cfg, false, vb.pp("merger"))?,
            deepstack,
            device: vb.device().clone(),
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Run the tower on `patches (N, 1536)` of a `gh × gw` patch grid.
    pub fn forward(&self, patches: &Tensor, gh: usize, gw: usize) -> Result<VisionFeatures> {
        let cfg = &self.cfg;
        let n = gh * gw;
        anyhow::ensure!(patches.dims2()? == (n, cfg.patch_dim()), "patches {:?} for a {gh}x{gw} grid", patches.dims());
        let x = patches.to_dtype(self.patch_weight.dtype())?.to_device(&self.device)?;
        let x = x.matmul(&self.patch_weight.t()?)?.broadcast_add(&self.patch_bias)?;
        let pos = pos_embed(&self.pos_table, cfg, gh, gw)?.to_dtype(x.dtype())?;
        let x = (x + pos)?;
        let (cos, sin) = rotary(cfg, gh, gw, &self.device)?;
        let mut x = x;
        let mut deepstack = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x, cfg, &cos, &sin)?;
            if let Some(j) = cfg.deepstack_visual_indexes.iter().position(|&d| d == i) {
                deepstack.push(self.deepstack[j].forward(&x, cfg)?);
            }
        }
        let merged = self.merger.forward(&x, cfg)?;
        let m = cfg.spatial_merge_size;
        Ok(VisionFeatures { merged, deepstack, rows: gh / m, cols: gw / m })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vit_sizes_follow_the_node() {
        assert_eq!(vit_size(1024, 1024, 3136, 12_845_056), (1024, 1024));
        assert_eq!(vit_size(1000, 700, 3136, 12_845_056), (992, 704));
        assert_eq!(vit_size(20, 20, 3136, 12_845_056), (64, 64));
    }

    #[test]
    fn patch_order_is_block_major() -> Result<()> {
        let cfg = VisionConfig { patch_size: 2, spatial_merge_size: 2, temporal_patch_size: 1, in_channels: 1, ..Default::default() };
        // an 8x4 picture of one channel: value = row*10 + col
        let data: Vec<f32> = (0..8 * 4).map(|i| ((i / 4) * 10 + i % 4) as f32 / 100.0).collect();
        let rgb = Tensor::from_vec(data, (1, 8, 4), &Device::Cpu)?;
        let (p, gh, gw) = patchify(&rgb, &cfg)?;
        assert_eq!((gh, gw), (4, 2));
        let v = p.to_vec2::<f32>()?;
        // token 0 = block (0,0), intra (0,0) = patch at rows 0-1, cols 0-1 → first value (0,0)
        assert!((v[0][0] - (0.0 / 100.0 - 0.5) / 0.5).abs() < 1e-6);
        // token 1 = block (0,0), intra (0,1) = patch at rows 0-1, cols 2-3 → first value (0,2)
        assert!((v[1][0] - (2.0 / 100.0 - 0.5) / 0.5).abs() < 1e-6);
        // token 2 = block (0,0), intra (1,0) = rows 2-3, cols 0-1 → (2,0) = 20
        assert!((v[2][0] - (20.0 / 100.0 - 0.5) / 0.5).abs() < 1e-6);
        // token 4 = block (1,0): rows 4-5 → (4,0) = 40
        assert!((v[4][0] - (40.0 / 100.0 - 0.5) / 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn position_table_resamples_bilinearly() -> Result<()> {
        let cfg = VisionConfig { num_position_embeddings: 4, hidden_size: 1, ..Default::default() };
        // a 2x2 table with values 0,1,2,3 (row-major)
        let table = Tensor::from_vec(vec![0f32, 1.0, 2.0, 3.0], (4, 1), &Device::Cpu)?;
        // a 4x2 grid (merge 2): rows at 0, 1/3, 2/3, 1 of the way; cols at 0 and 1
        let e = pos_embed(&table, &cfg, 4, 2)?.to_vec2::<f32>()?;
        // token order: block (0,0): (r0,c0)=0, (r0,c1)=1, (r1,c0)=2/3, (r1,c1)=1+2/3 (bf16 arithmetic: 1e-2)
        assert!((e[0][0] - 0.0).abs() < 1e-2 && (e[1][0] - 1.0).abs() < 1e-2);
        assert!((e[2][0] - 2.0 / 3.0).abs() < 1e-2 && (e[3][0] - 5.0 / 3.0).abs() < 1e-2);
        Ok(())
    }
}
