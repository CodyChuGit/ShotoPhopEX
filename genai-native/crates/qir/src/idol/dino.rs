//! DINOv3 ViT-H/16+ (ComfyUI `comfy/image_encoders/dino3.py`, HF tensor names), as Idol's
//! conditioning uses it: ImageNet-normalised pixels in, the whole token sequence (cls, 4
//! registers, then the patches) out, after the final affine norm and one more non-affine
//! layer norm (`TripoSplatConditioning`).

use candle_core::{DType, Device, Tensor};

use super::nn::{attention, Lin, Norm};
use crate::moss::weights::Weights;
use crate::Result;

const PATCH: usize = 16;
const REGISTERS: usize = 4;
const ROPE_THETA: f32 = 100.0;
const EPS: f64 = 1e-5;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

struct Layer {
    norm1: Norm,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ls1: Tensor,
    norm2: Norm,
    gate: Lin,
    up: Lin,
    down: Lin,
    ls2: Tensor,
}

impl Layer {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        let lin = |n: &str| Lin::load(w, &format!("{p}.{n}"));
        Ok(Self {
            norm1: Norm::load(w, &format!("{p}.norm1"), EPS)?,
            q: lin("attention.q_proj")?,
            k: lin("attention.k_proj")?,
            v: lin("attention.v_proj")?,
            o: lin("attention.o_proj")?,
            ls1: w.get(&format!("{p}.layer_scale1.lambda1"))?,
            norm2: Norm::load(w, &format!("{p}.norm2"), EPS)?,
            gate: lin("mlp.gate_proj")?,
            up: lin("mlp.up_proj")?,
            down: lin("mlp.down_proj")?,
            ls2: w.get(&format!("{p}.layer_scale2.lambda1"))?,
        })
    }

    fn forward(&self, x: &Tensor, heads: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let d = c / heads;
        let h = self.norm1.forward(x)?;
        let split = |t: Tensor| t.reshape((b, l, heads, d));
        let q = rope_patches(&split(self.q.forward(&h)?)?, cos, sin)?;
        let k = rope_patches(&split(self.k.forward(&h)?)?, cos, sin)?;
        let v = split(self.v.forward(&h)?)?;
        let a = self.o.forward(&attention(&q, &k, &v)?)?;
        let x = (x + a.broadcast_mul(&self.ls1)?)?;
        let h = self.norm2.forward(&x)?;
        let m = self.down.forward(&(candle_nn::ops::silu(&self.gate.forward(&h)?)? * self.up.forward(&h)?)?)?;
        Ok((x + m.broadcast_mul(&self.ls2)?)?)
    }
}

/// `rotate_half` RoPE on the patch tokens of `x (B, L, H, D)` (the cls and register tokens at
/// the front are left alone); `cos`, `sin (P, D)` in f32.
fn rope_patches(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, l, _, d) = x.dims4()?;
    let p = cos.dim(0)?;
    let dt = x.dtype();
    let prefix = x.narrow(1, 0, l - p)?;
    let xp = x.narrow(1, l - p, p)?.to_dtype(DType::F32)?;
    let x1 = xp.narrow(3, 0, d / 2)?;
    let x2 = xp.narrow(3, d / 2, d / 2)?;
    let rot = Tensor::cat(&[x2.neg()?, x1], 3)?;
    // (P, D) against (B, P, H, D): broadcast over B and H.
    let c = cos.unsqueeze(1)?.unsqueeze(0)?;
    let s = sin.unsqueeze(1)?.unsqueeze(0)?;
    let out = (xp.broadcast_mul(&c)? + rot.broadcast_mul(&s)?)?.to_dtype(dt)?;
    Ok(Tensor::cat(&[prefix, out], 1)?)
}

/// The 2D RoPE tables for a `gh × gw` patch grid: `(P, D)` cos and sin.
fn rope_tables(gh: usize, gw: usize, head_dim: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    // inv_freq = 1 / theta^(arange(0, 1, 4/D)), in f32 like torch.
    let n = head_dim / 4;
    let inv: Vec<f32> = (0..n).map(|i| 1.0 / ROPE_THETA.powf(i as f32 * (4.0 / head_dim as f32))).collect();
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut cos = Vec::with_capacity(gh * gw * head_dim);
    let mut sin = Vec::with_capacity(gh * gw * head_dim);
    for y in 0..gh {
        for x in 0..gw {
            let cy = 2.0 * ((y as f32 + 0.5) / gh as f32) - 1.0;
            let cx = 2.0 * ((x as f32 + 0.5) / gw as f32) - 1.0;
            let half: Vec<f32> = inv.iter().map(|&f| two_pi * cy * f).chain(inv.iter().map(|&f| two_pi * cx * f)).collect();
            for _ in 0..2 {
                cos.extend(half.iter().map(|a| a.cos()));
                sin.extend(half.iter().map(|a| a.sin()));
            }
        }
    }
    let p = gh * gw;
    Ok((Tensor::from_vec(cos, (p, head_dim), dev)?, Tensor::from_vec(sin, (p, head_dim), dev)?))
}

pub struct Dino {
    patch_w: Tensor,
    patch_b: Tensor,
    cls: Tensor,
    registers: Tensor,
    layers: Vec<Layer>,
    norm: Norm,
    heads: usize,
}

impl Dino {
    pub fn load(w: &Weights) -> Result<Self> {
        let mut layers = Vec::new();
        while w.contains(&format!("layer.{}.norm1.weight", layers.len())) {
            layers.push(Layer::load(w, &format!("layer.{}", layers.len()))?);
        }
        let hidden = w.shape("embeddings.cls_token")?[2];
        Ok(Self {
            patch_w: w.get("embeddings.patch_embeddings.weight")?,
            patch_b: w.get("embeddings.patch_embeddings.bias")?,
            cls: w.get("embeddings.cls_token")?,
            registers: w.get("embeddings.register_tokens")?,
            layers,
            norm: Norm::load(w, "norm", EPS)?,
            // ViT-H/16+ and ViT-L both use 64-wide heads.
            heads: hidden / 64,
        })
    }

    /// Idol's `feature1` for `rgb (B, 3, H, W)` in `[0, 1]`: `(B, 5 + H/16·W/16, C)` in f32.
    pub fn features(&self, rgb: &Tensor) -> Result<Tensor> {
        let dev = rgb.device();
        let (b, _, ih, iw) = rgb.dims4()?;
        let mean = Tensor::from_slice(&MEAN, (1, 3, 1, 1), dev)?;
        let std = Tensor::from_slice(&STD, (1, 3, 1, 1), dev)?;
        let px = rgb.to_dtype(DType::F32)?.broadcast_sub(&mean)?.broadcast_div(&std)?;
        let dt = self.patch_w.dtype();
        let pe = px.to_dtype(dt)?.conv2d(&self.patch_w, 0, PATCH, 1, 1)?; // (B, C, gh, gw)
        let pe = pe.broadcast_add(&self.patch_b.reshape((1, (), 1, 1))?)?;
        let (_, c, gh, gw) = pe.dims4()?;
        let patches = pe.flatten_from(2)?.transpose(1, 2)?; // (B, P, C)
        let cls = self.cls.expand((b, 1, c))?;
        let reg = self.registers.expand((b, REGISTERS, c))?;
        let mut x = Tensor::cat(&[cls, reg, patches], 1)?.contiguous()?;
        let (cos, sin) = rope_tables(gh, gw, c / self.heads, dev)?;
        for layer in &self.layers {
            x = layer.forward(&x, self.heads, &cos, &sin)?;
        }
        let x = self.norm.forward(&x)?.to_dtype(DType::F32)?;
        debug_assert_eq!((ih / PATCH, iw / PATCH), (gh, gw));
        Norm::plain(EPS).forward(&x)
    }
}
