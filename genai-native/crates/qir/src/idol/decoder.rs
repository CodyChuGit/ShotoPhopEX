//! Idol's splat decoder (ComfyUI `comfy/ldm/triposplat/vae.py` and `gaussian.py`): an octree
//! decoder samples points inside the unit cube from the shape code, level by level, then the
//! elastic gaussian decoder turns each point into 32 gaussians. The result is in the viewer's
//! frame (Y up), activated: positions, linear scales, unit wxyz rotations, opacities in
//! `[0, 1]` and the SH DC colour.

use candle_core::{DType, Device, Tensor};

use super::nn::{attention, log2_freqs, point_embedding, HeadRms, Lin, Mlp, Norm};
use crate::comfy::Mt19937;
use crate::moss::weights::Weights;
use crate::Result;

const MAX_LEVEL: usize = 8;
/// Gaussians per decoded point (`rep_config.num_gaussians`).
pub const GAUSSIANS_PER_POINT: usize = 32;
const PERTURB_SIZE: f32 = 1.5;
const FILTER_KERNEL: f32 = 0.0009;
const SCALING_BIAS: f32 = 0.004;
const OPACITY_BIAS: f32 = 0.1;
const ROT_LR: f32 = 0.1;

/// Self or cross attention with per-head RMS on q and k (`MultiHeadAttention`).
struct Mha {
    qkv: Option<Lin>,
    q: Option<Lin>,
    kv: Option<Lin>,
    out: Lin,
    q_norm: HeadRms,
    k_norm: HeadRms,
    heads: usize,
}

impl Mha {
    fn load(w: &Weights, p: &str, cross: bool) -> Result<Self> {
        let lin = |n: &str| Lin::load(w, &format!("{p}.{n}"));
        Ok(Self {
            qkv: if cross { None } else { Some(lin("to_qkv")?) },
            q: if cross { Some(lin("to_q")?) } else { None },
            kv: if cross { Some(lin("to_kv")?) } else { None },
            out: lin("to_out")?,
            q_norm: HeadRms::load(w, &format!("{p}.q_rms_norm"))?,
            k_norm: HeadRms::load(w, &format!("{p}.k_rms_norm"))?,
            heads: w.shape(&format!("{p}.q_rms_norm.gamma"))?[0],
        })
    }

    fn forward(&self, x: &Tensor, ctx: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let (heads, d) = (self.heads, c / self.heads);
        let (q, k, v) = match (&self.qkv, ctx) {
            (Some(qkv), _) => {
                let t = qkv.forward(x)?.reshape((b, l, 3, heads, d))?;
                (t.narrow(2, 0, 1)?.squeeze(2)?, t.narrow(2, 1, 1)?.squeeze(2)?, t.narrow(2, 2, 1)?.squeeze(2)?)
            }
            (None, Some(ctx)) => {
                let lk = ctx.dim(1)?;
                let q = self.q.as_ref().expect("cross").forward(x)?.reshape((b, l, heads, d))?;
                let t = self.kv.as_ref().expect("cross").forward(ctx)?.reshape((b, lk, 2, heads, d))?;
                (q, t.narrow(2, 0, 1)?.squeeze(2)?, t.narrow(2, 1, 1)?.squeeze(2)?)
            }
            (None, None) => unreachable!("cross attention needs a context"),
        };
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        self.out.forward(&attention(&q, &k, &v.contiguous()?)?)
    }
}

/// `in_proj(x) + pos(x)`, then `input_layer`: the decoders' point embedding of `x (1, L, 3)`.
fn embed_points(in_proj: &Lin, input_layer: &Lin, freqs: &[f32], channels: usize, x: &Tensor, dt: DType) -> Result<Tensor> {
    let (b, l, _) = x.dims3()?;
    let pos = point_embedding(&x.reshape((b * l, 3))?, freqs, channels)?.reshape((b, l, ()))?;
    let h = (in_proj.forward(&x.to_dtype(dt)?)? + pos.to_dtype(dt)?)?;
    input_layer.forward(&h)
}

/// `F.layer_norm` (no weights, eps 1e-5) in f32, back in `dt`.
fn final_norm(h: &Tensor) -> Result<Tensor> {
    Ok(Norm::plain(1e-5).forward(&h.to_dtype(DType::F32)?)?.to_dtype(h.dtype())?)
}

/// `LevelEmbedder.level_embedding`: `[cos, sin]` of `((res·f)·2)·π`, `f = exp(-ln(1024)·i/128)`,
/// in f32 in torch's order.
fn level_embedding(res: usize, dev: &Device) -> Result<Tensor> {
    let half = 128;
    let neg_log = -(1024f64.ln()) as f32;
    let args: Vec<f32> = (0..half).map(|i| (res as f32 * (neg_log * i as f32 / half as f32).exp()) * 2.0 * std::f32::consts::PI).collect();
    let v: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
    Ok(Tensor::from_vec(v, (1, 2 * half), dev)?)
}

struct OctreeBlock {
    cross: Mha,
    mlp: Mlp,
}

/// `OctreeProbabilityFixedlenDecoder`: per parent node, 8 child-occupancy logits.
struct Octree {
    in_proj: Lin,
    input_layer: Lin,
    l_fc1: Lin,
    l_fc2: Lin,
    ada: Lin,
    blocks: Vec<OctreeBlock>,
    out_proj: Lin,
    freqs: Vec<f32>,
    channels: usize,
    dt: DType,
}

impl Octree {
    fn load(w: &Weights) -> Result<Self> {
        let mut blocks = Vec::new();
        while w.contains(&format!("octree.blocks.{}.cross_attn.to_q.weight", blocks.len())) {
            let p = format!("octree.blocks.{}", blocks.len());
            blocks.push(OctreeBlock { cross: Mha::load(w, &format!("{p}.cross_attn"), true)?, mlp: Mlp::load(w, &format!("{p}.mlp"))? });
        }
        Ok(Self {
            in_proj: Lin::load(w, "octree.in_proj")?,
            input_layer: Lin::load(w, "octree.input_layer")?,
            l_fc1: Lin::load(w, "octree.l_embedder.mlp.0")?,
            l_fc2: Lin::load(w, "octree.l_embedder.mlp.2")?,
            ada: Lin::load(w, "octree.adaLN_modulation.1")?,
            blocks,
            out_proj: Lin::load(w, "octree.out_proj")?,
            freqs: log2_freqs(w.out_features("octree.in_proj")?, 3, 10),
            channels: w.out_features("octree.in_proj")?,
            dt: w.dtype(),
        })
    }

    /// Logits `(L, 8)` in f32 for parents at `x (1, L, 3)` (normalised centres) at resolution `res`.
    fn logits(&self, x: &Tensor, res: usize, cond: &Tensor) -> Result<Vec<f32>> {
        let dev = x.device();
        let mut h = embed_points(&self.in_proj, &self.input_layer, &self.freqs, self.channels, x, self.dt)?;
        let emb = level_embedding(res, dev)?.to_dtype(self.dt)?;
        let l = self.l_fc2.forward(&candle_nn::ops::silu(&self.l_fc1.forward(&emb)?)?)?;
        let m = self.ada.forward(&candle_nn::ops::silu(&l)?)?.to_dtype(DType::F32)?; // (1, 6C)
        let c = h.dim(2)?;
        let chunk = |i: usize| -> Result<Tensor> { Ok(m.narrow(1, i * c, c)?.unsqueeze(1)?) };
        let (sh1, sc1, g1, sh2, sc2, g2) = (chunk(0)?, chunk(1)?, chunk(2)?, chunk(3)?, chunk(4)?, chunk(5)?);
        let norm = Norm::plain(1e-6);
        let modulate = |x: &Tensor, sh: &Tensor, sc: &Tensor| -> Result<Tensor> {
            let n = norm.forward(&x.to_dtype(DType::F32)?)?;
            Ok(n.broadcast_mul(&(sc + 1.0)?)?.broadcast_add(sh)?.to_dtype(self.dt)?)
        };
        for b in &self.blocks {
            let a = b.cross.forward(&modulate(&h, &sh1, &sc1)?, Some(cond))?;
            h = (h.to_dtype(DType::F32)? + a.to_dtype(DType::F32)?.broadcast_mul(&g1)?)?.to_dtype(self.dt)?;
            let f = b.mlp.forward(&modulate(&h, &sh2, &sc2)?)?;
            h = (h.to_dtype(DType::F32)? + f.to_dtype(DType::F32)?.broadcast_mul(&g2)?)?.to_dtype(self.dt)?;
        }
        let logits = self.out_proj.forward(&final_norm(&h)?)?;
        Ok(logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }
}

/// torch's `torch.rand` on a seeded CPU generator: float32 = low 24 bits of one draw / 2^24.
pub struct TorchRand(Mt19937);

impl TorchRand {
    pub fn new(seed: u64) -> Self {
        Self(Mt19937::new(seed))
    }

    pub fn next(&mut self) -> f32 {
        self.0.uniform_f32()
    }
}

fn softmax8(l: &[f32]) -> [f32; 8] {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0f32; 8];
    let mut s = 0f32;
    for i in 0..8 {
        e[i] = (l[i] - m).exp();
        s += e[i];
    }
    e.map(|v| v / s)
}

/// `sample_probs` for one row: `count` systematic draws over the 8 children of one node.
fn systematic(probs: &[f32; 8], count: usize, u0: f32) -> [usize; 8] {
    let sum: f32 = probs.iter().sum();
    let mut cdf = [0f32; 8];
    let mut acc = 0f32;
    for i in 0..8 {
        acc += if sum == 0.0 { 1.0 / 8.0 } else { probs[i] / sum.max(1.0) };
        cdf[i] = acc.min(1.0 - 1e-12);
    }
    let cnt = count.max(1) as f32;
    let mut out = [0usize; 8];
    for j in 0..count {
        let u = ((u0 + j as f32) / cnt).min(1.0 - 1e-12);
        let idx = cdf.iter().position(|&c| c >= u).unwrap_or(7);
        out[idx] += 1;
    }
    out
}

struct GsBlock {
    self_attn: Mha,
    norm2: Norm,
    cross: Mha,
    mlp: Mlp,
}

/// `ElasticGaussianFixedlenDecoder`: 480 numbers per point (32 gaussians' parameters).
struct Gs {
    in_proj: Lin,
    input_layer: Lin,
    blocks: Vec<GsBlock>,
    out_proj: Lin,
    freqs: Vec<f32>,
    perturb: Vec<f32>,
    base_offset_scale: f32,
    channels: usize,
    dt: DType,
}

impl Gs {
    fn load(w: &Weights) -> Result<Self> {
        let mut blocks = Vec::new();
        while w.contains(&format!("gs.blocks.{}.self_attn.to_qkv.weight", blocks.len())) {
            let p = format!("gs.blocks.{}", blocks.len());
            blocks.push(GsBlock {
                self_attn: Mha::load(w, &format!("{p}.self_attn"), false)?,
                norm2: Norm::load(w, &format!("{p}.norm2"), 1e-6)?,
                cross: Mha::load(w, &format!("{p}.cross_attn"), true)?,
                mlp: Mlp::load(w, &format!("{p}.mlp"))?,
            });
        }
        let perturb = w.get_as("gs.points_offset_perturbation", DType::F32)?.flatten_all()?.to_vec1()?;
        let base = w.get_as("gs.base_offset_scale", DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0];
        Ok(Self {
            in_proj: Lin::load(w, "gs.in_proj")?,
            input_layer: Lin::load(w, "gs.input_layer")?,
            blocks,
            out_proj: Lin::load(w, "gs.out_proj")?,
            freqs: log2_freqs(w.out_features("gs.in_proj")?, 3, 10),
            channels: w.out_features("gs.in_proj")?,
            perturb,
            base_offset_scale: base,
            dt: w.dtype(),
        })
    }

    fn features(&self, x: &Tensor, cond: &Tensor) -> Result<Vec<f32>> {
        let mut h = embed_points(&self.in_proj, &self.input_layer, &self.freqs, self.channels, x, self.dt)?;
        let plain = Norm::plain(1e-6);
        for b in &self.blocks {
            h = (&h + b.self_attn.forward(&plain.forward(&h)?, None)?)?;
            h = (&h + b.cross.forward(&b.norm2.forward(&h)?, Some(cond))?)?;
            h = (&h + b.mlp.forward(&plain.forward(&h)?)?)?;
        }
        let out = self.out_proj.forward(&final_norm(&h)?)?;
        Ok(out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }
}

/// Activated gaussians in the viewer's frame, row-major.
#[derive(Debug, Clone, Default)]
pub struct Splat {
    pub positions: Vec<f32>,
    pub scales: Vec<f32>,
    /// Unit quaternions, `w x y z`.
    pub rotations: Vec<f32>,
    pub opacities: Vec<f32>,
    /// The SH DC coefficient per channel (`rgb = 0.5 + C0·dc`).
    pub sh_dc: Vec<f32>,
}

impl Splat {
    pub fn len(&self) -> usize {
        self.opacities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.opacities.is_empty()
    }
}

pub struct IdolDecoder {
    octree: Octree,
    gs: Gs,
}

fn softplus(x: f32) -> f32 {
    // torch's softplus (beta 1, threshold 20).
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `q_T ⊗ q` for the frame change `T` (a +90° turn about x), then unit length with `w ≥ 0`
/// (what `_matrix_to_quat(T · R(q))` gives).
fn turn_quat(q: [f32; 4]) -> [f32; 4] {
    let n = (q.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let [w, x, y, z] = q.map(|v| v / n);
    let h = std::f32::consts::FRAC_1_SQRT_2;
    // (h, h, 0, 0) ⊗ (w, x, y, z)
    let mut r = [h * (w - x), h * (x + w), h * (y - z), h * (z + y)];
    if r[0] < 0.0 {
        r = r.map(|v| -v);
    }
    let n = (r.iter().map(|v| v * v).sum::<f32>()).sqrt();
    r.map(|v| v / n)
}

impl IdolDecoder {
    pub fn load(w: &Weights) -> Result<Self> {
        Ok(Self { octree: Octree::load(w)?, gs: Gs::load(w)? })
    }

    /// The octree decoder's child logits `(L·8)` for parent centres `x (1, L, 3)` at `res`.
    pub fn octree_logits(&self, x: &Tensor, res: usize, cond: &Tensor) -> Result<Vec<f32>> {
        self.octree.logits(&x.to_dtype(DType::F32)?, res, &cond.to_dtype(self.octree.dt)?)
    }

    /// The octree's `n` points in `[0, 1]³` for the shape code `cond (1, 8192, 16)`, drawing
    /// from `rng` exactly as torch's sampler does; row-major `(n, 3)`.
    pub fn sample_points(&self, cond: &Tensor, n: usize, level: usize, rng: &mut TorchRand) -> Result<Vec<f32>> {
        let dev = cond.device();
        let cond = cond.to_dtype(self.octree.dt)?;
        let mut coords: Vec<[i64; 3]> = vec![[0, 0, 0]];
        let mut counts: Vec<usize> = vec![n];
        for lv in 1..=level {
            let res_p = (1usize << (lv - 1)) as f32;
            let parents: Vec<f32> = coords.iter().flat_map(|c| c.map(|v| (v as f32 + 0.5) / res_p)).collect();
            let x = Tensor::from_vec(parents, (1, coords.len(), 3), dev)?;
            let logits = self.octree.logits(&x, 1 << lv, &cond)?;
            // One uniform per row, all drawn before any row is resolved (`torch.rand(R, 1)`).
            let us: Vec<f32> = (0..coords.len()).map(|_| rng.next()).collect();
            let mut next_c = Vec::new();
            let mut next_n = Vec::new();
            for (r, c) in coords.iter().enumerate() {
                let p = softmax8(&logits[r * 8..r * 8 + 8]);
                let s = systematic(&p, counts[r], us[r]);
                for (ci, &k) in s.iter().enumerate() {
                    if k > 0 {
                        let off = [(ci & 1) as i64, ((ci >> 1) & 1) as i64, ((ci >> 2) & 1) as i64];
                        next_c.push([c[0] * 2 + off[0], c[1] * 2 + off[1], c[2] * 2 + off[2]]);
                        next_n.push(k);
                    }
                }
            }
            coords = next_c;
            counts = next_n;
        }
        let res = (1usize << level) as f32;
        let mut ints = Vec::with_capacity(n);
        for (c, &k) in coords.iter().zip(&counts) {
            ints.extend(std::iter::repeat_n(*c, k));
        }
        let mut out = Vec::with_capacity(n * 3);
        for c in &ints {
            for &v in c {
                out.push((v as f32 + rng.next()) / res);
            }
        }
        Ok(out)
    }

    /// Decode the shape code `latent (1, 8192, 16)` into `num_gaussians` (a multiple of 32)
    /// gaussians, with the octree sampler seeded by `seed` (`VAEDecodeTripoSplat`).
    pub fn decode(&self, latent: &Tensor, num_gaussians: usize, seed: u64) -> Result<Splat> {
        let mut rng = TorchRand::new(seed);
        let n = (num_gaussians / GAUSSIANS_PER_POINT).max(1);
        let points = self.sample_points(latent, n, MAX_LEVEL, &mut rng)?;
        self.gaussians(latent, &points)
    }

    /// The gaussians for given octree `points (n, 3)`.
    pub fn gaussians(&self, latent: &Tensor, points: &[f32]) -> Result<Splat> {
        let n = points.len() / 3;
        let x = Tensor::from_slice(points, (1, n, 3), latent.device())?;
        let h = self.gs.features(&x, &latent.to_dtype(self.gs.dt)?)?;
        let k = GAUSSIANS_PER_POINT;
        let width = h.len() / n;
        debug_assert_eq!(width, k * 15);
        let (r_xyz, r_dc, r_sc, r_rot, r_op, r_os) = (0, 3 * k, 6 * k, 9 * k, 13 * k, 14 * k);
        let scale_bias = SCALING_BIAS + (-(-SCALING_BIAS).exp_m1()).ln();
        let opacity_bias = (OPACITY_BIAS / (1.0 - OPACITY_BIAS)).ln();
        let mut s = Splat::default();
        for p in 0..n {
            let row = &h[p * width..(p + 1) * width];
            for g in 0..k {
                let os = softplus(row[r_os + g] + self.gs.base_offset_scale);
                let mut xyz = [0f32; 3];
                for c in 0..3 {
                    let o = row[r_xyz + g * 3 + c] + self.gs.perturb[g * 3 + c];
                    xyz[c] = o.tanh() * 0.5 * PERTURB_SIZE * os + points[p * 3 + c] - 0.5;
                }
                // xyz @ T.T: (x, -z, y).
                s.positions.extend([xyz[0], -xyz[2], xyz[1]]);
                for c in 0..3 {
                    let v = softplus(row[r_sc + g * 3 + c] + scale_bias);
                    s.scales.push((v * v + FILTER_KERNEL * FILTER_KERNEL).sqrt());
                    s.sh_dc.push(row[r_dc + g * 3 + c]);
                }
                let r = &row[r_rot + g * 4..r_rot + g * 4 + 4];
                s.rotations.extend(turn_quat([r[0] * ROT_LR + 1.0, r[1] * ROT_LR, r[2] * ROT_LR, r[3] * ROT_LR]));
                let o = row[r_op + g] + opacity_bias;
                s.opacities.push(1.0 / (1.0 + (-o).exp()));
            }
        }
        Ok(s)
    }
}

/// A binary 3DGS `.ply` of `s` (SplatToFile3D's layout: log scales, logit opacities).
pub fn ply_bytes(s: &Splat) -> Vec<u8> {
    let n = s.len();
    let attrs = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
    let mut out = format!("ply\nformat binary_little_endian 1.0\nelement vertex {n}\n").into_bytes();
    for a in attrs {
        out.extend(format!("property float {a}\n").bytes());
    }
    out.extend(b"end_header\n");
    out.reserve(n * attrs.len() * 4);
    for i in 0..n {
        let op = s.opacities[i].clamp(1e-6, 1.0 - 1e-6);
        let vals = [
            s.positions[i * 3],
            s.positions[i * 3 + 1],
            s.positions[i * 3 + 2],
            0.0,
            0.0,
            0.0,
            s.sh_dc[i * 3],
            s.sh_dc[i * 3 + 1],
            s.sh_dc[i * 3 + 2],
            (op / (1.0 - op)).ln(),
            s.scales[i * 3].max(1e-8).ln(),
            s.scales[i * 3 + 1].max(1e-8).ln(),
            s.scales[i * 3 + 2].max(1e-8).ln(),
            s.rotations[i * 4],
            s.rotations[i * 4 + 1],
            s.rotations[i * 4 + 2],
            s.rotations[i * 4 + 3],
        ];
        for v in vals {
            out.extend(v.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systematic_draws_sum_to_the_count() {
        let p = [0.5, 0.25, 0.125, 0.125, 0.0, 0.0, 0.0, 0.0];
        let s = systematic(&p, 8, 0.3);
        assert_eq!(s.iter().sum::<usize>(), 8);
        assert_eq!(s, [4, 2, 1, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn the_frame_turn_matches_the_matrix() {
        // Identity rotation turns into T itself: a +90° turn about x.
        let q = turn_quat([1.0, 0.0, 0.0, 0.0]);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        for (a, b) in q.iter().zip([h, h, 0.0, 0.0]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn ply_has_the_3dgs_header() {
        let s = Splat { positions: vec![0.0; 3], scales: vec![1.0; 3], rotations: vec![1.0, 0.0, 0.0, 0.0], opacities: vec![0.5], sh_dc: vec![0.0; 3] };
        let b = ply_bytes(&s);
        let text = String::from_utf8_lossy(&b);
        assert!(text.starts_with("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"));
        let end = text.find("end_header\n").unwrap() + "end_header\n".len();
        assert_eq!(b.len() - end, 17 * 4);
    }

    #[test]
    fn torch_rand_uses_24_bits() {
        let mut r = TorchRand::new(0);
        let v = r.next();
        assert!((0.0..1.0).contains(&v));
    }
}
