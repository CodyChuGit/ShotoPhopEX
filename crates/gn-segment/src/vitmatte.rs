//! ViTMatte (Yao et al., "ViTMatte: Boosting Image Matting with Pretrained Plain Vision
//! Transformers", 2023; MIT): a trimap-guided matting model. A plain ViT (ViTDet blocks:
//! windowed and global attention with decomposed relative positions, residual conv blocks)
//! reads the picture with the trimap as a fourth channel; a light conv stream recovers detail
//! and fuses it back up to full resolution, where a head gives the alpha. Hair strands come out
//! of the trimap's unknown band.
//!
//! Follows Hugging Face transformers' `VitMatteForImageMatting` / `VitDetBackbone` and its
//! checkpoint layout (`hustvl/vitmatte-*`: `model.safetensors` + `config.json`).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, VarBuilder};

/// The shapes the checkpoint's `config.json` gives (the small model's when absent).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub patch_size: usize,
    /// Positions the absolute embeddings were trained at (a square grid of this / patch).
    pub pretrain_image_size: usize,
    /// The grid the global layers' relative positions were trained at (this / patch).
    pub image_size: usize,
    pub window_size: usize,
    pub window_blocks: Vec<usize>,
    pub residual_blocks: Vec<usize>,
    pub layer_norm_eps: f64,
    pub batch_norm_eps: f64,
    pub convstream: Vec<usize>,
    pub fusion: Vec<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            hidden_size: 384,
            num_layers: 12,
            num_heads: 6,
            mlp_ratio: 4,
            patch_size: 16,
            pretrain_image_size: 224,
            image_size: 512,
            window_size: 14,
            window_blocks: vec![0, 1, 3, 4, 6, 7, 9, 10],
            residual_blocks: vec![2, 5, 8, 11],
            layer_norm_eps: 1e-6,
            batch_norm_eps: 1e-5,
            convstream: vec![48, 96, 192],
            fusion: vec![256, 128, 64, 32],
        }
    }
}

impl Config {
    /// Read a transformers `config.json` (`VitMatteConfig` with a `VitDetConfig` backbone).
    pub fn from_json(v: &serde_json::Value) -> Config {
        let d = Config::default();
        let b = &v["backbone_config"];
        let num = |o: &serde_json::Value, k: &str, def: usize| o[k].as_u64().map_or(def, |x| x as usize);
        let list = |o: &serde_json::Value, k: &str, def: &[usize]| {
            o[k].as_array().map_or_else(|| def.to_vec(), |a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
        };
        let size = |o: &serde_json::Value, k: &str, def: usize| match &o[k] {
            serde_json::Value::Array(a) => a.first().and_then(|x| x.as_u64()).map_or(def, |x| x as usize),
            x => x.as_u64().map_or(def, |x| x as usize),
        };
        Config {
            hidden_size: num(b, "hidden_size", d.hidden_size),
            num_layers: num(b, "num_hidden_layers", d.num_layers),
            num_heads: num(b, "num_attention_heads", d.num_heads),
            mlp_ratio: num(b, "mlp_ratio", d.mlp_ratio),
            patch_size: size(b, "patch_size", d.patch_size),
            pretrain_image_size: size(b, "pretrain_image_size", d.pretrain_image_size),
            image_size: size(b, "image_size", d.image_size),
            window_size: num(b, "window_size", d.window_size),
            window_blocks: list(b, "window_block_indices", &d.window_blocks),
            residual_blocks: list(b, "residual_block_indices", &d.residual_blocks),
            layer_norm_eps: b["layer_norm_eps"].as_f64().unwrap_or(d.layer_norm_eps),
            batch_norm_eps: v["batch_norm_eps"].as_f64().unwrap_or(d.batch_norm_eps),
            convstream: list(v, "convstream_hidden_sizes", &d.convstream),
            fusion: list(v, "fusion_hidden_sizes", &d.fusion),
        }
    }
}

/// A conv with its BatchNorm folded in (inference statistics), then optionally ReLU.
struct ConvBn {
    conv: Conv2d,
    relu: bool,
}

impl ConvBn {
    #[allow(clippy::too_many_arguments)]
    fn load(vb: &VarBuilder, conv: &str, bn: Option<&str>, cin: usize, cout: usize, k: usize, stride: usize, eps: f64, relu: bool) -> Result<ConvBn> {
        let cfg = Conv2dConfig { padding: k / 2, stride, ..Default::default() };
        let w = vb.get((cout, cin, k, k), &format!("{conv}.weight"))?;
        let b = vb.get(cout, &format!("{conv}.bias")).ok();
        let (w, b) = match bn {
            Some(bn) => {
                let g = vb.get(cout, &format!("{bn}.weight"))?;
                let beta = vb.get(cout, &format!("{bn}.bias"))?;
                let mean = vb.get(cout, &format!("{bn}.running_mean"))?;
                let var = vb.get(cout, &format!("{bn}.running_var"))?;
                let scale = g.div(&(var + eps)?.sqrt()?)?;
                let w = w.broadcast_mul(&scale.reshape((cout, 1, 1, 1))?)?;
                let b0 = b.unwrap_or(Tensor::zeros(cout, w.dtype(), w.device())?);
                let b = (b0.sub(&mean)?.mul(&scale)? + beta)?;
                (w, Some(b))
            }
            None => (w, b),
        };
        Ok(ConvBn { conv: Conv2d::new(w, b, cfg), relu })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        Ok(if self.relu { y.relu()? } else { y })
    }
}

/// LayerNorm over the channels of an NCHW map (ViTDet's residual blocks).
struct ChannelNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl ChannelNorm {
    fn load(vb: &VarBuilder, name: &str, c: usize) -> Result<ChannelNorm> {
        Ok(ChannelNorm {
            weight: vb.get(c, &format!("{name}.weight"))?.reshape((1, c, 1, 1))?,
            bias: vb.get(c, &format!("{name}.bias"))?.reshape((1, c, 1, 1))?,
            eps: 1e-6,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(1)?;
        let y = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        Ok(y.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)?)
    }
}

struct Residual {
    conv1: Conv2d,
    norm1: ChannelNorm,
    conv2: Conv2d,
    norm2: ChannelNorm,
    conv3: Conv2d,
    norm3: ChannelNorm,
}

impl Residual {
    fn load(vb: &VarBuilder, p: &str, dim: usize) -> Result<Residual> {
        let half = dim / 2;
        let conv = |name: &str, cin: usize, cout: usize, k: usize| -> Result<Conv2d> {
            let w = vb.get((cout, cin, k, k), &format!("{p}.{name}.weight"))?;
            Ok(Conv2d::new(w, None, Conv2dConfig { padding: k / 2, ..Default::default() }))
        };
        Ok(Residual {
            conv1: conv("conv1", dim, half, 1)?,
            norm1: ChannelNorm::load(vb, &format!("{p}.norm1"), half)?,
            conv2: conv("conv2", half, half, 3)?,
            norm2: ChannelNorm::load(vb, &format!("{p}.norm2"), half)?,
            conv3: conv("conv3", half, dim, 1)?,
            norm3: ChannelNorm::load(vb, &format!("{p}.norm3"), dim)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.norm1.forward(&self.conv1.forward(x)?)?.gelu_erf()?;
        let y = self.norm2.forward(&self.conv2.forward(&y)?)?.gelu_erf()?;
        let y = self.norm3.forward(&self.conv3.forward(&y)?)?;
        Ok((x + y)?)
    }
}

struct Block {
    norm1: LayerNorm,
    qkv: candle_nn::Linear,
    proj: candle_nn::Linear,
    rel_h: Tensor,
    rel_w: Tensor,
    norm2: LayerNorm,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    window: usize,
    residual: Option<Residual>,
    heads: usize,
}

/// Query rows per attention chunk (a global layer at 1536 px has 9216 tokens).
const QUERY_CHUNK: usize = 1024;

impl Block {
    fn load(vb: &VarBuilder, i: usize, c: &Config) -> Result<Block> {
        let p = format!("backbone.encoder.layer.{i}");
        let dim = c.hidden_size;
        let head = dim / c.num_heads;
        let window = if c.window_blocks.contains(&i) { c.window_size } else { 0 };
        let grid = if window > 0 { window } else { c.image_size / c.patch_size };
        let ln = |name: &str| -> Result<LayerNorm> {
            Ok(LayerNorm::new(vb.get(dim, &format!("{p}.{name}.weight"))?, vb.get(dim, &format!("{p}.{name}.bias"))?, c.layer_norm_eps))
        };
        let lin = |name: &str, i: usize, o: usize| -> Result<candle_nn::Linear> {
            Ok(candle_nn::Linear::new(vb.get((o, i), &format!("{p}.{name}.weight"))?, Some(vb.get(o, &format!("{p}.{name}.bias"))?)))
        };
        let hidden = dim * c.mlp_ratio;
        Ok(Block {
            norm1: ln("norm1")?,
            qkv: lin("attention.qkv", dim, dim * 3)?,
            proj: lin("attention.proj", dim, dim)?,
            rel_h: vb.get((2 * grid - 1, head), &format!("{p}.attention.rel_pos_h"))?,
            rel_w: vb.get((2 * grid - 1, head), &format!("{p}.attention.rel_pos_w"))?,
            norm2: ln("norm2")?,
            fc1: lin("mlp.fc1", dim, hidden)?,
            fc2: lin("mlp.fc2", hidden, dim)?,
            window,
            residual: if c.residual_blocks.contains(&i) { Some(Residual::load(vb, &format!("{p}.residual"), dim)?) } else { None },
            heads: c.num_heads,
        })
    }

    /// `x`: (1, H, W, C) tokens.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, h, w, c) = x.dims4()?;
        let shortcut = x.clone();
        let y = self.norm1.forward(x)?;
        let y = if self.window > 0 {
            let ws = self.window;
            let (ph, pw) = ((ws - h % ws) % ws, (ws - w % ws) % ws);
            let y = y.pad_with_zeros(1, 0, ph)?.pad_with_zeros(2, 0, pw)?;
            let (hp, wp) = (h + ph, w + pw);
            let (nh, nw) = (hp / ws, wp / ws);
            let win = y.reshape((1, nh, ws, nw, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((nh * nw, ws, ws, c))?;
            let out = self.attention(&win)?;
            out.reshape((1, nh, nw, ws, ws, c))?
                .permute((0, 1, 3, 2, 4, 5))?
                .contiguous()?
                .reshape((1, hp, wp, c))?
                .narrow(1, 0, h)?
                .narrow(2, 0, w)?
                .contiguous()?
        } else {
            self.attention(&y)?
        };
        let x = (shortcut + y)?;
        let m = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        let x = (x + m)?;
        match &self.residual {
            Some(r) => Ok(r.forward(&x.permute((0, 3, 1, 2))?.contiguous()?)?.permute((0, 2, 3, 1))?.contiguous()?),
            None => Ok(x),
        }
    }

    /// Attention over each (B, H, W, C) map, with decomposed relative positions.
    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        let n = h * w;
        let hd = c / self.heads;
        let scale = (hd as f64).powf(-0.5);
        let qkv = self.qkv.forward(x)?.reshape((b, n, 3, self.heads, hd))?.permute((2, 0, 3, 1, 4))?.contiguous()?.reshape((3, b * self.heads, n, hd))?;
        let (q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
        let rh = rel_pos(h, h, &self.rel_h)?; // (h, h, hd)
        let rw = rel_pos(w, w, &self.rel_w)?; // (w, w, hd)
        let kt = k.t()?.contiguous()?;
        let mut outs = Vec::new();
        let rows = (QUERY_CHUNK / w).max(1); // query rows (of the map) per chunk
        let mut y0 = 0;
        while y0 < h {
            let ny = rows.min(h - y0);
            let qc = q.narrow(1, y0 * w, ny * w)?.contiguous()?; // (bh, ny*w, hd)
            let mut attn = (qc.affine(scale, 0.0)?.matmul(&kt))?; // (bh, ny*w, n)
            // rel_h[b, y, x, ky] = q[b, y, x, :] · Rh[y, ky, :]
            let qr = qc.reshape((b * self.heads, ny, w, hd))?;
            let rhc = rh.narrow(0, y0, ny)?.contiguous()?; // (ny, h, hd)
            let relh = qr.permute((1, 0, 2, 3))?.contiguous()?.reshape((ny, b * self.heads * w, hd))?.matmul(&rhc.t()?.contiguous()?)?; // (ny, bh*w, h)
            let relh = relh.reshape((ny, b * self.heads, w, h))?.permute((1, 0, 2, 3))?.contiguous()?; // (bh, ny, w, h)
            let relw = qr.permute((2, 0, 1, 3))?.contiguous()?.reshape((w, b * self.heads * ny, hd))?.matmul(&rw.t()?.contiguous()?)?; // (w, bh*ny, w)
            let relw = relw.reshape((w, b * self.heads, ny, w))?.permute((1, 2, 0, 3))?.contiguous()?; // (bh, ny, w, kw)
            attn = attn.reshape((b * self.heads, ny, w, h, w))?.broadcast_add(&relh.unsqueeze(4)?)?.broadcast_add(&relw.unsqueeze(3)?)?.reshape((
                b * self.heads,
                ny * w,
                n,
            ))?;
            let p = candle_nn::ops::softmax_last_dim(&attn)?;
            outs.push(p.matmul(&v)?);
            y0 += ny;
        }
        let o = Tensor::cat(&outs, 1)?; // (bh, n, hd)
        let o = o.reshape((b, self.heads, h, w, hd))?.permute((0, 2, 3, 1, 4))?.contiguous()?.reshape((b, h, w, c))?;
        Ok(self.proj.forward(&o)?)
    }
}

/// `get_rel_pos`: the relative position table for q and k sizes, (q, k, C), resized linearly
/// (align_corners = false) when the trained table is another length.
fn rel_pos(q: usize, k: usize, table: &Tensor) -> Result<Tensor> {
    let max = 2 * q.max(k) - 1;
    let (len, cdim) = table.dims2()?;
    let t = if len == max { table.clone() } else { resize_linear(table, max)? };
    let qs = (k as f64 / q as f64).max(1.0);
    let ks = (q as f64 / k as f64).max(1.0);
    let mut idx = Vec::with_capacity(q * k);
    for i in 0..q {
        for j in 0..k {
            idx.push(((i as f64 * qs - j as f64 * ks) + (k as f64 - 1.0) * ks) as u32);
        }
    }
    let idx = Tensor::from_vec(idx, q * k, t.device())?;
    Ok(t.index_select(&idx, 0)?.reshape((q, k, cdim))?)
}

/// 1-D linear resize of the rows of (L, C) to `n` rows, as `F.interpolate(mode="linear")`.
fn resize_linear(t: &Tensor, n: usize) -> Result<Tensor> {
    let (len, c) = t.dims2()?;
    let src = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    let scale = len as f64 / n as f64;
    let mut out = Vec::with_capacity(n * c);
    for i in 0..n {
        let x = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let x0 = (x.floor() as usize).min(len - 1);
        let x1 = (x0 + 1).min(len - 1);
        let f = (x - x0 as f64) as f32;
        out.extend(src[x0].iter().zip(&src[x1]).map(|(a, b)| a * (1.0 - f) + b * f));
    }
    Ok(Tensor::from_vec(out, (n, c), t.device())?.to_dtype(t.dtype())?)
}

/// Bicubic resize (A = -0.75, align_corners = false, borders clamped) of a (S, S, C) grid to (h, w, C), as
/// `F.interpolate(mode="bicubic")`.
fn resize_bicubic(t: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (sh, sw, c) = t.dims3()?;
    let src = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let cubic = |x: f64| -> [f64; 4] {
        let a = -0.75;
        let w1 = |x: f64| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        let w2 = |x: f64| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
        [w2(x + 1.0), w1(x), w1(1.0 - x), w2(2.0 - x)]
    };
    let taps = |out: usize, inp: usize| -> Vec<(isize, [f64; 4])> {
        let scale = inp as f64 / out as f64;
        (0..out)
            .map(|o| {
                let x = (o as f64 + 0.5) * scale - 0.5;
                let x0 = x.floor();
                (x0 as isize, cubic(x - x0))
            })
            .collect()
    };
    let (ty, tx) = (taps(h, sh), taps(w, sw));
    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    let mut out = vec![0f32; h * w * c];
    for (oy, (y0, wy)) in ty.iter().enumerate() {
        for (ox, (x0, wx)) in tx.iter().enumerate() {
            for (dy, fy) in wy.iter().enumerate() {
                let sy = clamp(y0 - 1 + dy as isize, sh);
                for (dx, fx) in wx.iter().enumerate() {
                    let sx = clamp(x0 - 1 + dx as isize, sw);
                    let f = (fy * fx) as f32;
                    let s = &src[(sy * sw + sx) * c..(sy * sw + sx + 1) * c];
                    let d = &mut out[(oy * w + ox) * c..(oy * w + ox + 1) * c];
                    for (d, s) in d.iter_mut().zip(s) {
                        *d += f * s;
                    }
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, (h, w, c), t.device())?.to_dtype(t.dtype())?)
}

/// ViTMatte, resident.
pub struct VitMatte {
    config: Config,
    patch: Conv2d,
    pos: Tensor,
    blocks: Vec<Block>,
    convstream: Vec<ConvBn>,
    fusion: Vec<ConvBn>,
    head1: ConvBn,
    head2: ConvBn,
    device: Device,
}

impl VitMatte {
    /// Load `model.safetensors` (and `config.json` beside it, if any) from `dir`.
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<VitMatte> {
        let dir = dir.as_ref();
        let config = match std::fs::read_to_string(dir.join("config.json")) {
            Ok(s) => Config::from_json(&serde_json::from_str(&s).context("ViTMatte config.json")?),
            Err(_) => Config::default(),
        };
        let weights = dir.join("model.safetensors");
        ensure!(weights.exists(), "no ViTMatte weights at {}", weights.display());
        let tensors = candle_core::safetensors::load(&weights, device)?.into_iter().map(|(k, t)| Ok((k, t.to_dtype(DType::F32)?))).collect::<Result<_>>()?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Self::from_vb(&vb, config, device)
    }

    fn from_vb(vb: &VarBuilder, config: Config, device: &Device) -> Result<VitMatte> {
        let c = &config;
        let dim = c.hidden_size;
        let p = c.patch_size;
        let patch = Conv2d::new(
            vb.get((dim, 4, p, p), "backbone.embeddings.projection.weight")?,
            Some(vb.get(dim, "backbone.embeddings.projection.bias")?),
            Conv2dConfig { stride: p, ..Default::default() },
        );
        let g = c.pretrain_image_size / p;
        let pos = vb.get((1, g * g + 1, dim), "backbone.embeddings.position_embeddings")?.narrow(1, 1, g * g)?.reshape((g, g, dim))?;
        let blocks = (0..c.num_layers).map(|i| Block::load(vb, i, c)).collect::<Result<Vec<_>>>()?;
        let eps = c.batch_norm_eps;
        let mut chans = vec![4];
        chans.extend(&c.convstream);
        let convstream = (0..c.convstream.len())
            .map(|i| {
                let q = format!("decoder.convstream.convs.{i}");
                ConvBn::load(vb, &format!("{q}.conv"), Some(&format!("{q}.batch_norm")), chans[i], chans[i + 1], 3, 2, eps, true)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut fch = vec![dim];
        fch.extend(&c.fusion);
        let fusion = (0..c.fusion.len())
            .map(|i| {
                let q = format!("decoder.fusion_blocks.{i}.conv");
                let cin = fch[i] + chans[chans.len() - 1 - i];
                ConvBn::load(vb, &format!("{q}.conv"), Some(&format!("{q}.batch_norm")), cin, fch[i + 1], 3, 1, eps, true)
            })
            .collect::<Result<Vec<_>>>()?;
        let last = *c.fusion.last().context("no fusion blocks")?;
        let head1 = ConvBn::load(vb, "decoder.matting_head.matting_convs.0", Some("decoder.matting_head.matting_convs.1"), last, 16, 3, 1, 1e-5, true)?;
        let head2 = ConvBn::load(vb, "decoder.matting_head.matting_convs.3", None, 16, 1, 1, 1, eps, false)?;
        Ok(VitMatte { config, patch, pos, blocks, convstream, fusion, head1, head2, device: device.clone() })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The alpha for an RGBA8 picture (alpha ignored) and a trimap (0 background, 128 unknown,
    /// 255 foreground; any value is read as its fraction), both `width × height`. Returns the
    /// alpha, 0–1, row-major.
    pub fn matte(&self, rgba: &[u8], trimap: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        ensure!(rgba.len() == width * height * 4 && trimap.len() == width * height, "picture and trimap must be {width}×{height}");
        let n = width * height;
        let mut planes = vec![0f32; 4 * n];
        for i in 0..n {
            for ch in 0..3 {
                planes[ch * n + i] = (f32::from(rgba[i * 4 + ch]) / 255.0 - 0.5) / 0.5;
            }
            planes[3 * n + i] = f32::from(trimap[i]) / 255.0;
        }
        let x = Tensor::from_vec(planes, (1, 4, height, width), &self.device)?;
        // Padded (zeros, bottom and right) to a multiple of 32, as the processor does.
        let (ph, pw) = ((32 - height % 32) % 32, (32 - width % 32) % 32);
        let x = x.pad_with_zeros(2, 0, ph)?.pad_with_zeros(3, 0, pw)?;
        let a = self.forward(&x)?;
        let a = a.narrow(2, 0, height)?.narrow(3, 0, width)?.contiguous()?;
        Ok(a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
    }

    /// (1, 4, H, W) normalised input, H and W multiples of 32 → (1, 1, H, W) alpha.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let p = self.config.patch_size;
        let (gh, gw) = (h / p, w / p);
        let e = self.patch.forward(x)?.permute((0, 2, 3, 1))?; // (1, gh, gw, C)
        let pos = if self.pos.dim(0)? == gh && self.pos.dim(1)? == gw { self.pos.clone() } else { resize_bicubic(&self.pos, gh, gw)? };
        let mut t = e.broadcast_add(&pos.unsqueeze(0)?)?.contiguous()?;
        for b in &self.blocks {
            t = b.forward(&t)?;
        }
        let mut f = t.permute((0, 3, 1, 2))?.contiguous()?; // (1, C, gh, gw)
        let mut details = vec![x.clone()];
        for c in &self.convstream {
            let next = c.forward(details.last().context("conv stream")?)?;
            details.push(next);
        }
        for (i, fb) in self.fusion.iter().enumerate() {
            let d = &details[details.len() - 1 - i];
            let (_, _, dh, dw) = d.dims4()?;
            let up = f.upsample_bilinear2d(dh, dw, false)?;
            f = fb.forward(&Tensor::cat(&[d, &up], 1)?)?;
        }
        let y = self.head2.forward(&self.head1.forward(&f)?)?;
        Ok(candle_nn::ops::sigmoid(&y)?)
    }
}

/// A trimap from a coverage mask: unknown (128) within `band` pixels of its edge, foreground
/// (255) deeper inside, background (0) further out.
pub fn trimap_from_mask(mask: &[f32], width: usize, height: usize, band: usize) -> Vec<u8> {
    let n = width * height;
    let inside: Vec<bool> = mask.iter().take(n).map(|v| *v >= 0.5).collect();
    let fg = erode_or_dilate(&inside, width, height, band, true);
    let reach = erode_or_dilate(&inside, width, height, band, false);
    (0..n)
        .map(|i| {
            if fg[i] {
                255
            } else if reach[i] {
                128
            } else {
                0
            }
        })
        .collect()
}

/// A square erosion (`erode`) or dilation of `m` by `r`, in two separable running-count passes.
fn erode_or_dilate(m: &[bool], w: usize, h: usize, r: usize, erode: bool) -> Vec<bool> {
    // Erosion is the complement of dilating the complement.
    let src: Vec<bool> = if erode { m.iter().map(|b| !*b).collect() } else { m.to_vec() };
    let pass = |src: &[bool], horizontal: bool| -> Vec<bool> {
        let (lines, len) = if horizontal { (h, w) } else { (w, h) };
        let at = |line: usize, k: usize| if horizontal { line * w + k } else { k * w + line };
        let mut out = vec![false; src.len()];
        for line in 0..lines {
            let mut count = 0usize;
            for k in 0..len.min(r + 1) {
                count += usize::from(src[at(line, k)]);
            }
            for k in 0..len {
                out[at(line, k)] = count > 0;
                if k + r + 1 < len {
                    count += usize::from(src[at(line, k + r + 1)]);
                }
                if k >= r {
                    count -= usize::from(src[at(line, k - r)]);
                }
            }
        }
        out
    };
    let d = pass(&pass(&src, true), false);
    if erode { d.iter().map(|b| !*b).collect() } else { d }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimap_bands_the_edge() -> Result<()> {
        let (w, h) = (20, 10);
        let mask: Vec<f32> = (0..w * h).map(|i| if (i % w) < 10 { 1.0 } else { 0.0 }).collect();
        let t = trimap_from_mask(&mask, w, h, 2);
        assert_eq!(t[5 * w], 255);
        assert_eq!(t[5 * w + 8], 128);
        assert_eq!(t[5 * w + 11], 128);
        assert_eq!(t[5 * w + 19], 0);
        assert_eq!(t[5 * w + 7], 255);
        assert_eq!(t[5 * w + 12], 0);
        Ok(())
    }

    #[test]
    fn rel_pos_indexes_like_the_reference() -> Result<()> {
        // q = k = 3: index (i - j) + 2.
        let table = Tensor::arange(0f32, 5.0, &Device::Cpu)?.reshape((5, 1))?;
        let r = rel_pos(3, 3, &table)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(r, vec![2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0]);
        Ok(())
    }

    #[test]
    fn linear_and_bicubic_resizes_keep_constants() -> Result<()> {
        let t = Tensor::ones((7, 2), DType::F32, &Device::Cpu)?;
        assert!(resize_linear(&t, 13)?.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| (v - 1.0).abs() < 1e-6));
        let g = Tensor::ones((4, 4, 3), DType::F32, &Device::Cpu)?;
        assert!(resize_bicubic(&g, 6, 5)?.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| (v - 1.0).abs() < 1e-5));
        Ok(())
    }

    #[test]
    fn a_random_small_model_runs_end_to_end() -> Result<()> {
        let dev = Device::Cpu;
        let c = Config {
            hidden_size: 32,
            num_layers: 2,
            num_heads: 2,
            mlp_ratio: 2,
            patch_size: 16,
            pretrain_image_size: 64,
            image_size: 64,
            window_size: 2,
            window_blocks: vec![0],
            residual_blocks: vec![1],
            layer_norm_eps: 1e-6,
            batch_norm_eps: 1e-5,
            convstream: vec![8, 8, 8],
            fusion: vec![8, 8, 8, 8],
        };
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // Create every tensor the loader asks for (random init), then load through it.
        let m = VitMatte::from_vb(&vb, c, &dev)?;
        let (w, h) = (40, 24);
        let rgba = vec![128u8; w * h * 4];
        let tri = vec![128u8; w * h];
        let a = m.matte(&rgba, &tri, w, h)?;
        assert_eq!(a.len(), w * h);
        assert!(a.iter().all(|v| (0.0..=1.0).contains(v)));
        Ok(())
    }
}
