//! Idol's flow model (ComfyUI `comfy/ldm/triposplat/model.py`, `LatentSeqMMFlowModel`, the
//! TripoSplat architecture): jointly denoises a `(B, 8192, 16)` shape code and a `(B, 1, 5)`
//! camera token, conditioned on DINOv3's tokens of the picture plus its FLUX.2 latent. Each
//! latent token sits at a fixed Sobol anchor in the unit cube; attention uses "RePo" RoPE,
//! whose 3D positions are predicted per token and per head from the token itself.

use candle_core::{DType, Device, Tensor, D};

use super::nn::{attention, point_embedding, pow2_freqs, sinusoid, HeadRms, Lin, Mlp, Norm};
use crate::moss::weights::Weights;
use crate::Result;

pub const TOKENS: usize = 8192;
pub const LATENT_CHANNELS: usize = 16;
pub const CAM_CHANNELS: usize = 5;
/// The Sobol anchors' seed (`SobolEngine(dimension=3, scramble=True, seed=123)`).
const SOBOL_SEED: u32 = 123;

/// Per-token, per-head 3D rotary positions (`RePo3DRotaryEmbedding`).
struct RePo {
    norm: Norm,
    gate: Lin,
    content: Lin,
    fin: Lin,
    /// `freqs_0..2` concatenated, each multiplied into its axis's offset.
    freqs: [Tensor; 3],
    heads: usize,
}

impl RePo {
    fn load(w: &Weights, p: &str, heads: usize) -> Result<Self> {
        let f = |i: usize| w.get_as(&format!("{p}.freqs_{i}"), DType::F32);
        Ok(Self {
            norm: Norm::load(w, &format!("{p}.norm"), 1e-5)?,
            gate: Lin::load(w, &format!("{p}.gate_map"))?,
            content: Lin::load(w, &format!("{p}.content_map"))?,
            fin: Lin::load(w, &format!("{p}.final_map"))?,
            freqs: [f(0)?, f(1)?, f(2)?],
            heads,
        })
    }

    /// `(cos, sin)`, each `(B, L, H, D/2)` in f32.
    fn forward(&self, h: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.norm.forward(h)?;
        let feat = (candle_nn::ops::silu(&self.gate.forward(&x)?)? * self.content.forward(&x)?)?;
        let out = self.fin.forward(&feat)?.to_dtype(DType::F32)?;
        let (b, l, _) = out.dims3()?;
        let delta = out.reshape((b, l, self.heads, 3))?;
        let mut angs = Vec::with_capacity(3);
        for (axis, f) in self.freqs.iter().enumerate() {
            let d = delta.narrow(3, axis, 1)?; // (B, L, H, 1)
            angs.push((d.broadcast_mul(&f.reshape((1, 1, 1, f.dim(0)?))?)? * std::f64::consts::PI)?);
        }
        let ang = Tensor::cat(&angs, 3)?;
        Ok((ang.cos()?, ang.sin()?))
    }
}

/// Flux-style interleaved RoPE: pairs `(2i, 2i+1)` of `x (B, L, H, D)`.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, l, h, d) = x.dims4()?;
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?.reshape((b, l, h, d / 2, 2))?;
    let x0 = xf.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = xf.narrow(4, 1, 1)?.squeeze(4)?;
    let o0 = (cos.mul(&x0)? - sin.mul(&x1)?)?;
    let o1 = (sin.mul(&x0)? + cos.mul(&x1)?)?;
    Ok(Tensor::stack(&[o0, o1], 4)?.reshape((b, l, h, d))?.to_dtype(dt)?)
}

struct Attn {
    qkv: Lin,
    out: Lin,
    q_norm: HeadRms,
    k_norm: HeadRms,
    heads: usize,
}

impl Attn {
    fn load(w: &Weights, p: &str, heads: usize) -> Result<Self> {
        Ok(Self {
            qkv: Lin::load(w, &format!("{p}.qkv"))?,
            out: Lin::load(w, &format!("{p}.out"))?,
            q_norm: HeadRms::load(w, &format!("{p}.q_norm"))?,
            k_norm: HeadRms::load(w, &format!("{p}.k_norm"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, rot: &(Tensor, Tensor)) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let hd = c / self.heads;
        let qkv = self.qkv.forward(x)?.reshape((b, l, 3, self.heads, hd))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        // RoPE first, then the per-head norms (the model's order).
        let q = self.q_norm.forward(&rope(&q, &rot.0, &rot.1)?)?;
        let k = self.k_norm.forward(&rope(&k, &rot.0, &rot.1)?)?;
        self.out.forward(&attention(&q, &k, &v.contiguous()?)?)
    }
}

/// `UnifiedTransformerBlock`: AdaLN-modulated (the shared modulation plus the block's
/// `shift_table`) or plain pre-norm.
struct Block {
    norm1: Norm,
    norm2: Norm,
    attn: Attn,
    mlp: Mlp,
    shift_table: Option<Tensor>,
}

impl Block {
    fn load(w: &Weights, p: &str, heads: usize, modulated: bool) -> Result<Self> {
        let (norm1, norm2) = if modulated {
            (Norm::plain(1e-6), Norm::plain(1e-6))
        } else {
            (Norm::load(w, &format!("{p}.norm1"), 1e-6)?, Norm::load(w, &format!("{p}.norm2"), 1e-6)?)
        };
        let shift_table = if modulated { Some(w.get(&format!("{p}.shift_table"))?) } else { None };
        Ok(Self { norm1, norm2, attn: Attn::load(w, &format!("{p}.attn"), heads)?, mlp: Mlp::load(w, &format!("{p}.mlp"))?, shift_table })
    }

    /// `x (B, L, C)`; `m (B, 6C)` the shared modulation for modulated blocks.
    fn forward(&self, x: &Tensor, m: Option<&Tensor>, rot: &(Tensor, Tensor)) -> Result<Tensor> {
        match (&self.shift_table, m) {
            (Some(table), Some(m)) => {
                let m = m.broadcast_add(&table.to_dtype(m.dtype())?)?;
                let c = x.dim(2)?;
                let part = |i: usize| -> Result<Tensor> { Ok(m.narrow(1, i * c, c)?.unsqueeze(1)?) };
                let (sh1, sc1, g1, sh2, sc2, g2) = (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);
                let h = self.norm1.forward(x)?.broadcast_mul(&(sc1 + 1.0)?)?.broadcast_add(&sh1)?;
                let x = (x + self.attn.forward(&h, rot)?.broadcast_mul(&g1)?)?;
                let h = self.norm2.forward(&x)?.broadcast_mul(&(sc2 + 1.0)?)?.broadcast_add(&sh2)?;
                Ok((&x + self.mlp.forward(&h)?.broadcast_mul(&g2)?)?)
            }
            _ => {
                let x = (x + self.attn.forward(&self.norm1.forward(x)?, rot)?)?;
                Ok((&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?)
            }
        }
    }
}

pub struct IdolDit {
    t_fc1: Lin,
    t_fc2: Lin,
    ada: Lin,
    input: Lin,
    cond: Lin,
    cond2: Lin,
    /// The Sobol anchors' embedding `(1, 8192, C)`.
    pos_emb: Tensor,
    noise_repo: Vec<RePo>,
    context_repo: Vec<RePo>,
    repo: Vec<RePo>,
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    cam_refiner: Mlp,
    blocks: Vec<Block>,
    shift_table: Tensor,
    out: Lin,
    cam_out: Lin,
    dtype: DType,
}

/// The anchors' embedding: `PcdAbsolutePositionEmbedder(channels)` on torch's Sobol points.
pub fn anchor_embedding(channels: usize, dev: &Device) -> Result<Tensor> {
    let pts = super::sobol::scrambled_3d(TOKENS, SOBOL_SEED);
    let x = Tensor::from_vec(pts, (TOKENS, 3), dev)?;
    Ok(point_embedding(&x, &pow2_freqs(channels, 3, 16), channels)?.unsqueeze(0)?)
}

impl IdolDit {
    pub fn load(w: &Weights) -> Result<Self> {
        let count = |p: &str| (0..).take_while(|i| w.contains(&format!("{p}.{i}.attn.qkv.weight"))).count();
        let channels = w.shape("input_layer.weight")?[0];
        let heads = w.shape("blocks.0.attn.q_norm.gamma")?[0];
        let repos = |p: &str, n: usize| -> Result<Vec<RePo>> { (0..n).map(|i| RePo::load(w, &format!("{p}.{i}"), heads)).collect() };
        let blocks = |p: &str, modulated: bool| -> Result<Vec<Block>> { (0..count(p)).map(|i| Block::load(w, &format!("{p}.{i}"), heads, modulated)).collect() };
        let nb = count("blocks");
        let nr = count("noise_refiner");
        Ok(Self {
            t_fc1: Lin::load(w, "t_embedder.mlp.0")?,
            t_fc2: Lin::load(w, "t_embedder.mlp.2")?,
            ada: Lin::load(w, "adaLN_modulation.1")?,
            input: Lin::load(w, "input_layer")?,
            cond: Lin::load(w, "cond_embedder")?,
            cond2: Lin::load(w, "cond_embedder2")?,
            pos_emb: anchor_embedding(channels, w.device())?.to_dtype(w.dtype())?,
            noise_repo: repos("noise_repo_layers", nr)?,
            context_repo: repos("context_repo_layers", count("context_refiner"))?,
            repo: repos("repo_layers", nb)?,
            noise_refiner: blocks("noise_refiner", true)?,
            context_refiner: blocks("context_refiner", false)?,
            cam_refiner: Mlp::load(w, "cam_refiner")?,
            blocks: blocks("blocks", true)?,
            shift_table: w.get_as("shift_table", DType::F32)?,
            out: Lin::load(w, "out_layer")?,
            cam_out: Lin::load(w, "cam_out_layer")?,
            dtype: w.dtype(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The conditioning sequence: DINOv3's tokens `feat1 (B, Lc, 1280)` plus the FLUX.2
    /// latent `feat2 (B, 128, h, w)` flattened and front-padded to `Lc`, both embedded.
    pub fn context(&self, feat1: &Tensor, feat2: &Tensor) -> Result<Tensor> {
        let mut h = self.cond.forward(&feat1.to_dtype(self.dtype)?)?;
        let (b, c2, fh, fw) = feat2.dims4()?;
        let f2 = feat2.reshape((b, c2, fh * fw))?.transpose(1, 2)?.contiguous()?;
        let pad = feat1.dim(1)? - fh * fw;
        let f2 = f2.pad_with_zeros(1, pad, 0)?;
        h = (h + self.cond2.forward(&f2.to_dtype(self.dtype)?)?)?;
        Ok(h)
    }

    /// The velocity for `z (B, 8192, 16)` and `cam (B, 1, 5)` at `timestep` (sigma · 1000),
    /// given [`Self::context`]'s embedded `ctx`. The context refiner's output does not depend on
    /// the timestep, so the caller may pass it pre-refined (see [`Self::refine_context`]).
    pub fn forward(&self, z: &Tensor, cam: &Tensor, timestep: f32, ctx_refined: &Tensor) -> Result<(Tensor, Tensor)> {
        let b = z.dim(0)?;
        let dev = z.device();
        let t = sinusoid(&vec![timestep; b], 256, 10000.0, 1.0, dev)?.to_dtype(self.dtype)?;
        let t_emb = self.t_fc2.forward(&candle_nn::ops::silu(&self.t_fc1.forward(&t)?)?)?;
        let t_mod = self.ada.forward(&candle_nn::ops::silu(&t_emb)?)?;

        let mut hx = self.input.forward(&z.to_dtype(self.dtype)?)?.broadcast_add(&self.pos_emb)?;
        for (blk, repo) in self.noise_refiner.iter().zip(&self.noise_repo) {
            let rot = repo.forward(&hx)?;
            hx = blk.forward(&hx, Some(&t_mod), &rot)?;
        }
        let hcam = self.cam_refiner.forward(&cam.to_dtype(self.dtype)?)?;
        let lz = hx.dim(1)?;
        let lc = cam.dim(1)?;
        let mut h = Tensor::cat(&[&hx, ctx_refined, &hcam], 1)?;
        for (blk, repo) in self.blocks.iter().zip(&self.repo) {
            let rot = repo.forward(&h)?;
            h = blk.forward(&h, Some(&t_mod), &rot)?;
        }
        let total = h.dim(1)?;
        let plain = Norm::plain(1e-5);
        let hx = plain.forward(&h.narrow(1, 0, lz)?)?;
        let hc = plain.forward(&h.narrow(1, total - lc, lc)?)?;
        // shift, scale = (shift_table + t_emb[:, None]).chunk(2, dim=1)
        let st = self.shift_table.to_dtype(self.dtype)?.broadcast_add(&t_emb.unsqueeze(1)?)?; // (B, 2, C)
        let shift = st.narrow(1, 0, 1)?;
        let scale = (st.narrow(1, 1, 1)? + 1.0)?;
        let hx = hx.broadcast_mul(&scale)?.broadcast_add(&shift)?;
        let hc = hc.broadcast_mul(&scale)?.broadcast_add(&shift)?;
        Ok((self.out.forward(&hx)?.to_dtype(DType::F32)?, self.cam_out.forward(&hc)?.to_dtype(DType::F32)?))
    }

    /// The context refiner over [`Self::context`] (plain blocks, no timestep).
    pub fn refine_context(&self, ctx: &Tensor) -> Result<Tensor> {
        let mut h = ctx.clone();
        for (blk, repo) in self.context_refiner.iter().zip(&self.context_repo) {
            let rot = repo.forward(&h)?;
            h = blk.forward(&h, None, &rot)?;
        }
        Ok(h)
    }
}

/// `max |a - b| / max |b|` (for the parity checks and the pipeline's own sanity checks).
pub fn max_rel(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?;
    let scale = b.abs()?.max(D::Minus1)?.to_scalar::<f32>()?.max(1e-6);
    Ok((a - b)?.abs()?.max(D::Minus1)?.to_scalar::<f32>()? / scale)
}
