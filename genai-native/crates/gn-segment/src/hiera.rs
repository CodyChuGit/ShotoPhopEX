//! Hiera (Ryali et al., "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles",
//! 2023): a plain hierarchical ViT with mask-unit attention in its early stages and max-pooled
//! queries between stages. Follows timm's `hiera.py` (Apache-2.0) and its `features_only`
//! getter: the four stage outputs, rolled back to spatial order, as NCHW maps.
//!
//! timm "unrolls" the token grid once per pooling stride so that the tokens of one mask unit
//! (and the 2x2 neighbourhood a pooling step takes the maximum of) sit at a fixed stride in the
//! token list. With strides `s` applied `k` times, a token at row `h` of a `gh` grid is
//! `h = hk * s^k + sy_k * s^(k-1) + ... + sy_1`, and the unrolled order is
//! `[sy_1, sx_1, ..., sy_k, sx_k, hk, wk]` (`sy_1` the finest, slowest-varying). Pooling takes
//! the maximum over the leading `(sy_1, sx_1)`; the features are rerolled with the strides that
//! remain at that stage.
//!
//! Precision: the residual stream, the layer norms and the attention softmax are f32; the
//! linear layers, attention products and convolution run in the compute dtype (f16/bf16 on a
//! GPU), as PyTorch's autocast does.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Module, Tensor};

use crate::store::Store;

/// The shapes of a Hiera image model (timm's arguments).
#[derive(Clone, Debug, PartialEq)]
pub struct HieraConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub stages: Vec<usize>,
    /// How many stage transitions pool the queries.
    pub q_pool: usize,
    /// The pooling stride per axis.
    pub q_stride: usize,
    /// The mask unit's side, in tokens.
    pub mask_unit: usize,
    /// Which stages use mask-unit (windowed) attention; the first block of a stage uses the
    /// previous stage's choice, as in timm.
    pub mask_unit_attn: Vec<bool>,
    pub dim_mul: f64,
    pub head_mul: f64,
    pub patch_kernel: usize,
    pub patch_stride: usize,
    pub patch_padding: usize,
    pub mlp_ratio: f64,
    pub in_chans: usize,
    /// The square input side the model is built for.
    pub img_size: usize,
    pub layer_norm_eps: f64,
}

impl HieraConfig {
    /// timm's `hiera_base_plus_224` (embed 112, 2 heads, stages 2/3/16/3) at another input
    /// size and channel count.
    pub fn base_plus(img_size: usize, in_chans: usize) -> HieraConfig {
        HieraConfig {
            embed_dim: 112,
            num_heads: 2,
            stages: vec![2, 3, 16, 3],
            q_pool: 3,
            q_stride: 2,
            mask_unit: 8,
            mask_unit_attn: vec![true, true, false, false],
            dim_mul: 2.0,
            head_mul: 2.0,
            patch_kernel: 7,
            patch_stride: 4,
            patch_padding: 3,
            mlp_ratio: 4.0,
            in_chans,
            img_size,
            layer_norm_eps: 1e-6,
        }
    }

    /// The channels of each stage's output.
    pub fn stage_dims(&self) -> Vec<usize> {
        let mut d = self.embed_dim as f64;
        (0..self.stages.len())
            .map(|i| {
                if i > 0 {
                    d = (d * self.dim_mul).floor();
                }
                d as usize
            })
            .collect()
    }

    /// The token grid side after the patch embedding.
    pub fn tokens_side(&self) -> usize {
        self.img_size / self.patch_stride
    }

    /// The index of each stage's last block.
    pub fn stage_ends(&self) -> Vec<usize> {
        let mut acc = 0;
        self.stages
            .iter()
            .map(|s| {
                acc += s;
                acc - 1
            })
            .collect()
    }
}

/// A linear layer in the compute dtype on (N, C) rows.
pub(crate) struct Lin {
    inner: candle_nn::Linear,
}

impl Lin {
    pub(crate) fn load(st: &Store, name: &str, cin: usize, cout: usize, bias: bool, dtype: DType) -> Result<Lin> {
        let w = st.get(&format!("{name}.weight"), &[cout, cin], dtype)?;
        let b = if bias { Some(st.get(&format!("{name}.bias"), &[cout], dtype)?) } else { None };
        Ok(Lin { inner: candle_nn::Linear::new(w, b) })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.inner.forward(&x.to_dtype(self.inner.weight().dtype())?)?)
    }
}

/// LayerNorm over the last dimension, in f32.
struct Norm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl Norm {
    fn load(st: &Store, name: &str, dim: usize, eps: f64) -> Result<Norm> {
        Ok(Norm { weight: st.get(&format!("{name}.weight"), &[dim], DType::F32)?, bias: st.get(&format!("{name}.bias"), &[dim], DType::F32)?, eps: eps as f32 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::layer_norm(&x.to_dtype(DType::F32)?.contiguous()?, &self.weight, &self.bias, self.eps)?)
    }
}

/// Elements one attention score block may hold (a chunk of queries against all keys).
const SCORE_BUDGET: usize = 1 << 26;

/// Softmax attention of (b, m, d) queries over (b, n, d) keys and values, in chunks of queries,
/// the softmax in f32.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let (b, m, _) = q.dims3()?;
    let n = k.dim(1)?;
    let dtype = q.dtype();
    let kt = k.t()?.contiguous()?;
    let v = v.contiguous()?;
    // Chunk along the batch when it is large, else along the queries.
    let per_batch = m * n;
    if per_batch <= SCORE_BUDGET {
        let bs = (SCORE_BUDGET / per_batch).max(1);
        let mut outs = Vec::new();
        let mut b0 = 0;
        while b0 < b {
            let nb = bs.min(b - b0);
            let s = q.narrow(0, b0, nb)?.affine(scale, 0.0)?.matmul(&kt.narrow(0, b0, nb)?)?;
            let p = candle_nn::ops::softmax_last_dim(&s.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
            outs.push(p.matmul(&v.narrow(0, b0, nb)?)?);
            b0 += nb;
        }
        return Ok(Tensor::cat(&outs, 0)?);
    }
    let rows = (SCORE_BUDGET / (b * n)).max(1);
    let mut outs = Vec::new();
    let mut m0 = 0;
    while m0 < m {
        let nm = rows.min(m - m0);
        let s = q.narrow(1, m0, nm)?.affine(scale, 0.0)?.matmul(&kt)?;
        let p = candle_nn::ops::softmax_last_dim(&s.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        outs.push(p.matmul(&v)?);
        m0 += nm;
    }
    Ok(Tensor::cat(&outs, 1)?)
}

struct Block {
    norm1: Norm,
    /// The expansion projection of a stage's first block (`dim != dim_out`).
    proj: Option<Lin>,
    qkv: Lin,
    attn_proj: Lin,
    norm2: Norm,
    fc1: Lin,
    fc2: Lin,
    dim_out: usize,
    heads: usize,
    /// The flattened query pooling stride (4 for 2x2), 1 when none.
    q_stride: usize,
    /// The flattened mask-unit size after pooling.
    window: usize,
    mask_unit_attn: bool,
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = self.qkv.inner.weight().dtype();
        let xn = self.norm1.forward(x)?.to_dtype(dtype)?;
        let x = match &self.proj {
            Some(p) => {
                let y = p.forward(&xn)?;
                let n = y.dim(0)?;
                y.reshape((self.q_stride, n / self.q_stride, self.dim_out))?.max(0)?.to_dtype(DType::F32)?
            }
            None => x.clone(),
        };
        let a = self.attention(&xn)?;
        let x = (x + a.to_dtype(DType::F32)?)?;
        let m = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?.to_dtype(dtype)?)?.gelu_erf()?)?;
        Ok((x + m.to_dtype(DType::F32)?)?)
    }

    /// timm's `MaskUnitAttention` on (N, dim) unrolled tokens.
    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let (heads, hd) = (self.heads, self.dim_out / self.heads);
        let scale = (hd as f64).powf(-0.5);
        let qkv = self.qkv.forward(x)?;
        let out = if self.mask_unit_attn {
            let nw = n / (self.q_stride * self.window);
            ensure!(nw > 0 && n % nw == 0, "Hiera: {n} tokens do not split into mask units");
            let t = n / nw;
            // (T, nw, 3, H, hd) -> (3, H, nw, T, hd)
            let qkv = qkv.reshape((t, nw, 3, heads, hd))?.permute((2, 3, 1, 0, 4))?.contiguous()?;
            let (mut q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
            if self.q_stride > 1 {
                q = q.reshape((heads, nw, self.q_stride, t / self.q_stride, hd))?.max(2)?;
            }
            let tq = q.dim(2)?;
            let o = attend(&q.reshape((heads * nw, tq, hd))?, &k.reshape((heads * nw, t, hd))?, &v.reshape((heads * nw, t, hd))?, scale)?;
            // (H, nw, Tq, hd) -> (Tq, nw, H, hd)
            o.reshape((heads, nw, tq, hd))?.permute((2, 1, 0, 3))?.contiguous()?.reshape((tq * nw, self.dim_out))?
        } else {
            let qkv = qkv.reshape((n, 3, heads, hd))?.permute((1, 2, 0, 3))?.contiguous()?;
            let (mut q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
            if self.q_stride > 1 {
                q = q.reshape((heads, self.q_stride, n / self.q_stride, hd))?.max(1)?;
            }
            let nq = q.dim(1)?;
            let o = attend(&q.contiguous()?, &k, &v, scale)?;
            o.permute((1, 0, 2))?.contiguous()?.reshape((nq, self.dim_out))?
        };
        self.attn_proj.forward(&out)
    }
}

/// The permutation from row-major `(hk, sy_k..sy_1, wk, sx_k..sx_1, C)` to the unrolled
/// `(sy_1, sx_1, ..., sy_k, sx_k, hk, wk, C)`.
fn unroll_perm(k: usize) -> Vec<usize> {
    let mut p = Vec::with_capacity(2 * k + 3);
    for j in 1..=k {
        p.push(1 + k - j);
        p.push(k + 2 + k - j);
    }
    p.extend([0, k + 1, 2 * k + 2]);
    p
}

fn rowmajor_dims(gh: usize, gw: usize, s: usize, k: usize, c: usize) -> Vec<usize> {
    let f = s.pow(k as u32);
    let mut d = vec![gh / f];
    d.extend(std::iter::repeat_n(s, k));
    d.push(gw / f);
    d.extend(std::iter::repeat_n(s, k));
    d.push(c);
    d
}

/// Row-major (gh*gw, C) tokens to timm's unrolled order for `k` strides of `s`.
pub(crate) fn unroll(x: &Tensor, gh: usize, gw: usize, s: usize, k: usize) -> Result<Tensor> {
    let (n, c) = x.dims2()?;
    if k == 0 {
        return Ok(x.clone());
    }
    let f = s.pow(k as u32);
    ensure!(gh.is_multiple_of(f) && gw.is_multiple_of(f), "Hiera: a {gh}x{gw} token grid does not unroll by {f}");
    Ok(x.reshape(rowmajor_dims(gh, gw, s, k, c))?.permute(unroll_perm(k))?.contiguous()?.reshape((n, c))?)
}

/// The inverse of [`unroll`]: unrolled tokens back to a (gh, gw, C) grid.
pub(crate) fn reroll(x: &Tensor, gh: usize, gw: usize, s: usize, k: usize) -> Result<Tensor> {
    let (_, c) = x.dims2()?;
    if k == 0 {
        return Ok(x.reshape((gh, gw, c))?);
    }
    let r = rowmajor_dims(gh, gw, s, k, c);
    let perm = unroll_perm(k);
    let udims: Vec<usize> = perm.iter().map(|&i| r[i]).collect();
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    Ok(x.reshape(udims)?.permute(inv)?.contiguous()?.reshape((gh, gw, c))?)
}

/// Hiera, resident, returning its four stage maps.
pub struct Hiera {
    config: HieraConfig,
    patch_w: Tensor,
    patch_b: Tensor,
    /// (N, C) f32, already at the configured grid.
    pos: Tensor,
    blocks: Vec<Block>,
    dtype: DType,
}

impl Hiera {
    /// Load from `st` under `prefix` (e.g. `encoder.model.`), computing in `dtype`.
    pub(crate) fn load(st: &Store, prefix: &str, mut config: HieraConfig, dtype: DType) -> Result<Hiera> {
        let c = &mut config;
        let pk = c.patch_kernel;
        let pw_name = format!("{prefix}patch_embed.proj.weight");
        if let Some(shape) = st.shape(&pw_name) {
            ensure!(shape.len() == 4, "{pw_name} is {shape:?}");
            c.embed_dim = shape[0];
            c.in_chans = shape[1];
        }
        let c = &config;
        let patch_w = st.get(&pw_name, &[c.embed_dim, c.in_chans, pk, pk], dtype)?;
        let patch_b = st.get(&format!("{prefix}patch_embed.proj.bias"), &[c.embed_dim], dtype)?;
        let g = c.tokens_side();
        ensure!(c.img_size.is_multiple_of(c.patch_stride), "Hiera: input {} is not a multiple of the patch stride", c.img_size);
        let pos_name = format!("{prefix}pos_embed");
        let pos = match st.shape(&pos_name) {
            Some(shape) => pos_embed(&st.get(&pos_name, &shape, DType::F32)?, c.embed_dim, g)?,
            None => {
                ensure!(st.is_random(), "missing tensor `{pos_name}`");
                st.get(&pos_name, &[g * g, c.embed_dim], DType::F32)?
            }
        };
        let ends = c.stage_ends();
        let q_pool_blocks: Vec<usize> = ends.iter().take(c.q_pool).map(|e| e + 1).collect();
        let flat_q = c.q_stride * c.q_stride;
        let mut window = c.mask_unit * c.mask_unit;
        let (mut dim, mut heads, mut stage) = (c.embed_dim, c.num_heads, 0usize);
        let depth: usize = c.stages.iter().sum();
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let mut dim_out = dim;
            let mask_unit_attn = *c.mask_unit_attn.get(stage).context("Hiera: mask_unit_attn shorter than the stages")?;
            if i > 0 && ends.contains(&(i - 1)) {
                dim_out = (dim as f64 * c.dim_mul) as usize;
                heads = (heads as f64 * c.head_mul) as usize;
                stage += 1;
                if q_pool_blocks.contains(&i) {
                    window /= flat_q;
                }
            }
            let p = format!("{prefix}blocks.{i}");
            let hidden = (dim_out as f64 * c.mlp_ratio) as usize;
            let eps = c.layer_norm_eps;
            blocks.push(Block {
                norm1: Norm::load(st, &format!("{p}.norm1"), dim, eps)?,
                proj: if dim != dim_out { Some(Lin::load(st, &format!("{p}.proj"), dim, dim_out, true, dtype)?) } else { None },
                qkv: Lin::load(st, &format!("{p}.attn.qkv"), dim, 3 * dim_out, true, dtype)?,
                attn_proj: Lin::load(st, &format!("{p}.attn.proj"), dim_out, dim_out, true, dtype)?,
                norm2: Norm::load(st, &format!("{p}.norm2"), dim_out, eps)?,
                fc1: Lin::load(st, &format!("{p}.mlp.fc1"), dim_out, hidden, true, dtype)?,
                fc2: Lin::load(st, &format!("{p}.mlp.fc2"), hidden, dim_out, true, dtype)?,
                dim_out,
                heads,
                q_stride: if q_pool_blocks.contains(&i) { flat_q } else { 1 },
                window,
                mask_unit_attn,
            });
            dim = dim_out;
        }
        Ok(Hiera { config, patch_w, patch_b, pos, blocks, dtype })
    }

    pub fn config(&self) -> &HieraConfig {
        &self.config
    }

    /// (1, in_chans, S, S) input (S the configured size) to the four stage maps, NCHW f32,
    /// strides 4, 8, 16, 32 for the default patch and pooling.
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let c = &self.config;
        let (b, _, h, w) = x.dims4()?;
        ensure!(b == 1 && h == c.img_size && w == c.img_size, "Hiera is built for 1x{}x{} input, got {:?}", c.img_size, c.img_size, x.dims());
        let e = x.to_dtype(self.dtype)?.conv2d(&self.patch_w, c.patch_padding, c.patch_stride, 1, 1)?.broadcast_add(&self.patch_b.reshape((1, (), 1, 1))?)?;
        let (_, d, gh, gw) = e.dims4()?;
        let t = e.to_dtype(DType::F32)?.reshape((d, gh * gw))?.t()?.contiguous()?;
        let t = (t + &self.pos)?;
        let n_unroll = c.stages.len() - 1;
        let mut t = unroll(&t, gh, gw, c.q_stride, n_unroll)?;
        let ends = c.stage_ends();
        let mut feats = Vec::with_capacity(ends.len());
        let mut pooled = 0usize;
        for (i, blk) in self.blocks.iter().enumerate() {
            if blk.q_stride > 1 {
                pooled += 1;
            }
            t = blk.forward(&t)?;
            if ends.contains(&i) {
                let f = c.q_stride.pow(pooled as u32);
                let (sh, sw) = (gh / f, gw / f);
                let grid = reroll(&t, sh, sw, c.q_stride, n_unroll - pooled)?;
                feats.push(grid.permute((2, 0, 1))?.contiguous()?.unsqueeze(0)?);
            }
        }
        Ok(feats)
    }
}

/// The stored position embedding as (g*g, C) f32: as is when its count matches, else resized
/// as a square grid with bicubic interpolation (PyTorch's, A = -0.75, align_corners = false).
/// Accepts (1, N, C), (N, C) and NCHW (1, C, S, S).
pub(crate) fn pos_embed(t: &Tensor, dim: usize, g: usize) -> Result<Tensor> {
    let grid = match t.rank() {
        4 => {
            let (_, ch, sh, sw) = t.dims4()?;
            ensure!(ch == dim, "pos_embed has {ch} channels, the model {dim}");
            t.squeeze(0)?.permute((1, 2, 0))?.contiguous()?.reshape((sh, sw, dim))?
        }
        2 | 3 => {
            let flat = t.reshape(((), dim))?;
            let n = flat.dim(0)?;
            let s = (n as f64).sqrt().round() as usize;
            ensure!(s * s == n, "pos_embed holds {n} positions, not a square grid");
            flat.reshape((s, s, dim))?
        }
        r => anyhow::bail!("pos_embed has rank {r}"),
    };
    let (sh, sw, _) = grid.dims3()?;
    let grid = if (sh, sw) == (g, g) { grid } else { crate::vitmatte::resize_bicubic(&grid, g, g)? };
    Ok(grid.reshape((g * g, dim))?.to_dtype(DType::F32)?)
}

#[cfg(test)]
pub(crate) fn tiny_config(img_size: usize) -> HieraConfig {
    HieraConfig { embed_dim: 8, num_heads: 1, stages: vec![1, 2, 2, 1], in_chans: 4, ..HieraConfig::base_plus(img_size, 4) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// timm's `Unroll` written as its loop of views and permutes, on index labels.
    fn timm_unroll(gh: usize, gw: usize, k: usize) -> Vec<usize> {
        // x[b][h][w] = h * gw + w; B grows by 4 per step.
        let mut cur: Vec<Vec<usize>> = vec![(0..gh * gw).collect()]; // batch of row-major grids
        let (mut h, mut w) = (gh, gw);
        for _ in 0..k {
            let (h2, w2) = (h / 2, w / 2);
            let mut next = Vec::new();
            for g in &cur {
                for sy in 0..2 {
                    for sx in 0..2 {
                        let mut sub = Vec::with_capacity(h2 * w2);
                        for y in 0..h2 {
                            for x in 0..w2 {
                                sub.push(g[(y * 2 + sy) * w + x * 2 + sx]);
                            }
                        }
                        next.push(sub);
                    }
                }
            }
            cur = next;
            (h, w) = (h2, w2);
        }
        cur.concat()
    }

    #[test]
    fn unroll_matches_timms_loop_and_reroll_inverts_it() -> Result<()> {
        for (gh, gw, k) in [(8, 8, 3), (16, 8, 3), (8, 4, 2), (4, 4, 1)] {
            let x = Tensor::arange(0u32, (gh * gw) as u32, &Device::Cpu)?.to_dtype(DType::F32)?.reshape((gh * gw, 1))?;
            let u = unroll(&x, gh, gw, 2, k)?;
            let got: Vec<usize> = u.flatten_all()?.to_vec1::<f32>()?.iter().map(|v| *v as usize).collect();
            assert_eq!(got, timm_unroll(gh, gw, k), "{gh}x{gw} k={k}");
            let back = reroll(&u, gh, gw, 2, k)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(back, x.flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(())
    }

    #[test]
    fn base_plus_has_timms_shapes() {
        let c = HieraConfig::base_plus(2048, 4);
        assert_eq!(c.stage_dims(), vec![112, 224, 448, 896]);
        assert_eq!(c.stage_ends(), vec![1, 4, 20, 23]);
        assert_eq!(c.tokens_side(), 512);
    }

    #[test]
    fn stage_maps_have_the_feature_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let st = Store::random(&dev);
        let c = tiny_config(64);
        let m = Hiera::load(&st, "encoder.model.", c, DType::F32)?;
        let x = Tensor::rand(0f32, 1.0, (1, 4, 64, 64), &dev)?;
        let f = m.forward(&x)?;
        let dims: Vec<Vec<usize>> = f.iter().map(|t| t.dims().to_vec()).collect();
        assert_eq!(dims, vec![vec![1, 8, 16, 16], vec![1, 16, 8, 8], vec![1, 32, 4, 4], vec![1, 64, 2, 2]]);
        Ok(())
    }

    #[test]
    fn mask_unit_attention_stays_inside_its_unit() -> Result<()> {
        // A change in one 8x8 mask unit of the input moves only that unit's stage-1 features
        // (stage 1 is all mask-unit attention, the patch conv reaches 3 pixels past a patch).
        let dev = Device::Cpu;
        let st = Store::random(&dev);
        let m = Hiera::load(&st, "", tiny_config(128), DType::F32)?;
        let x = Tensor::rand(0f32, 1.0, (1, 4, 128, 128), &dev)?;
        let a = m.forward(&x)?.remove(0);
        // Bump one pixel deep inside token unit (0, 0) (pixels 0..32).
        let bump =
            Tensor::zeros((1, 4, 128, 128), DType::F32, &dev)?.slice_assign(&[0..1, 0..4, 10..11, 10..11], &Tensor::ones((1, 4, 1, 1), DType::F32, &dev)?)?;
        let b = m.forward(&(x + bump)?)?.remove(0);
        let d = (a - b)?.abs()?.sum(1)?.squeeze(0)?.to_vec2::<f32>()?; // (32, 32) tokens
        for (y, row) in d.iter().enumerate() {
            for (x, v) in row.iter().enumerate() {
                if y >= 8 || x >= 8 {
                    assert!(*v == 0.0, "token ({y}, {x}) outside the unit changed by {v}");
                }
            }
        }
        assert!(d[2][2] > 0.0);
        Ok(())
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn pos_embed_resizes_a_square_grid() -> Result<()> {
        // (1, 4, 1): a 2x2 grid [[1, 2], [3, 4]] to 4x4; PyTorch's bicubic gives row 0
        // 0.68359375, 1.015625, 1.5625, 1.89453125 (hand-computed with A = -0.75).
        let t = Tensor::new(&[[[1f32], [2.0], [3.0], [4.0]]], &Device::Cpu)?;
        let r = pos_embed(&t, 1, 4)?.flatten_all()?.to_vec1::<f32>()?;
        let want = [0.68359375, 1.015625, 1.5625, 1.89453125];
        for (a, b) in r[..4].iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{r:?}");
        }
        // The grid is a linear ramp, x + 2y: rows step by 2 * (1.2265625 - 0.89453125).
        assert!((r[4] - (0.68359375 + 2.0 * (1.2265625 - 0.89453125))).abs() < 1e-6, "{r:?}");
        // Same count: unchanged.
        let same = pos_embed(&t, 1, 2)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(same, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    /// Parity with timm, skipped unless `HIERA_REF_DIR` holds `weights.safetensors` (a
    /// `features_only` Hiera B+ state dict under any prefix ending in `patch_embed.proj.weight`'s
    /// parent), `input.npy` (1, C, S, S) and `feat0..3.npy` (timm's NCHW stage maps).
    #[test]
    fn parity_with_timm_features() -> Result<()> {
        let Ok(dir) = std::env::var("HIERA_REF_DIR") else {
            eprintln!("hiera parity: skipped (HIERA_REF_DIR unset)");
            return Ok(());
        };
        let dir = std::path::PathBuf::from(dir);
        let dev = Device::Cpu;
        let st = Store::open(&dir.join("weights.safetensors"), &dev)?;
        let names = st.names();
        let key = names.iter().find(|n| n.ends_with("patch_embed.proj.weight")).context("no patch embedding")?;
        let prefix = &key[..key.len() - "patch_embed.proj.weight".len()];
        let x = Tensor::read_npy(dir.join("input.npy"))?.to_dtype(DType::F32)?;
        let (_, c, s, _) = x.dims4()?;
        let m = Hiera::load(&st, prefix, HieraConfig::base_plus(s, c), DType::F32)?;
        let feats = m.forward(&x)?;
        for (i, f) in feats.iter().enumerate() {
            let want = Tensor::read_npy(dir.join(format!("feat{i}.npy")))?.to_dtype(DType::F32)?;
            assert_eq!(f.dims(), want.dims(), "feat{i}");
            let d = (f - &want)?.abs()?.flatten_all()?;
            let max = d.max(0)?.to_scalar::<f32>()?;
            let mean = d.mean(0)?.to_scalar::<f32>()?;
            let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            eprintln!("hiera parity feat{i} {:?}: mean abs {mean:.2e}, max abs {max:.2e} (max |ref| {scale:.2})", f.dims());
            assert!(max <= 1e-3 * scale.max(1.0), "feat{i}: max abs {max}");
        }
        Ok(())
    }
}
