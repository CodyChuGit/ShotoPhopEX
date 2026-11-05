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
use crate::lora::{LoraPair, LoraSlot, LoraStack};
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
    /// `qkv`'s output rows as the file's rows (`None`: the file's order), for a LoRA's `B`.
    qkv_rows: Option<Vec<usize>>,
    /// LoRA adapters (Kromoss' turbo, MOSS Edit), set per request: q, k and v as one adapter on
    /// the stacked `qkv`.
    l_qkv: LoraSlot,
    l_o: LoraSlot,
    l_up_gate: LoraSlot,
    l_down: LoraSlot,
    /// The one-pass kernels' weights (CUDA bf16 with `fusednorm`): f32 norms, and q/k norms in
    /// the interleaved channel order `qkv`'s q and k rows were loaded in (see [`interleaved`]).
    #[cfg(feature = "fusednorm")]
    fused: Option<FusedNorms>,
}

#[cfg(feature = "fusednorm")]
struct FusedNorms {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

/// The order that turns each head's rotate-half RoPE pairs `(i, i + D/2)` into adjacent pairs
/// `(2i, 2i + 1)`: position `2i` takes channel `i`, `2i + 1` channel `i + D/2`. Applied to both q
/// and k it leaves every `q·k` unchanged, and lets the interleaved fused RoPE kernel serve.
fn interleaved(d: usize) -> Vec<usize> {
    (0..d / 2).flat_map(|i| [i, i + d / 2]).collect()
}

/// SageAttention2 for the blocks (its P·V accumulation from `QIR_SAGE`), unless
/// `QIR_MOSS_ATTN=exact` asks for exact bf16 attention. At 1024², 50 steps: 17.7 s with the exact
/// kernel, 10.9 s with SA2 (ComfyUI 10.8 s), DiT cosine to the official model 0.999 either way.
#[cfg(feature = "sage2")]
fn sage_attention() -> Result<Option<crate::sage2::PvAccum>> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("QIR_MOSS_ATTN").map_or(true, |v| v.trim() != "exact")) {
        return Ok(None);
    }
    Ok(match crate::sage2::attention_impl()? {
        crate::sage2::AttentionImpl::Sage2(accum) => Some(accum),
        _ => None,
    })
}

/// Whether the blocks take the one-pass kernels: a CUDA bf16 run with `fusednorm`, unless
/// `QIR_MOSS_FUSED=0` asks for the candle chains (to compare).
fn use_fused(w: &Weights) -> bool {
    cfg!(feature = "fusednorm") && w.device().is_cuda() && w.dtype() == DType::BF16 && std::env::var("QIR_MOSS_FUSED").map_or(true, |v| v.trim() != "0")
}

impl Block {
    fn load(w: &Weights, p: &str, heads: usize) -> Result<Self> {
        let fused = use_fused(w);
        let dim = w.get(&format!("{p}.norm1.weight"))?.dim(0)?;
        let hd = dim / heads;
        // Fused: q's and k's rows in each head's interleaved order, v's as they are.
        let order: Option<Vec<usize>> = fused.then(|| {
            let head = interleaved(hd);
            (0..3 * dim).map(|r| if r < 2 * dim { r / hd * hd + head[r % hd] } else { r }).collect()
        });
        let qkv = if w.contains(&format!("{p}.proj.qkv.weight")) {
            w.linear_rows(&format!("{p}.proj.qkv"), order.as_deref())?
        } else {
            // The official split q/k/v (a diffusers export), stacked here.
            let parts = ["q", "k", "v"].iter().map(|n| w.get(&format!("{p}.proj.{n}.weight"))).collect::<Result<Vec<_>>>()?;
            let full = Tensor::cat(&parts, 0)?;
            let full = match &order {
                Some(o) => full.index_select(&Tensor::from_vec(o.iter().map(|&i| i as u32).collect::<Vec<_>>(), o.len(), w.device())?, 0)?,
                None => full,
            };
            QLinear::Dense(candle_nn::Linear::new(full, None))
        };
        #[cfg(feature = "fusednorm")]
        let fused = if fused {
            let head = Tensor::from_vec(interleaved(hd).iter().map(|&i| i as u32).collect::<Vec<_>>(), hd, w.device())?;
            let f32w = |n: &str| w.get_as(&format!("{p}.{n}.weight"), DType::F32);
            Some(FusedNorms {
                norm1: f32w("norm1")?,
                norm2: f32w("norm2")?,
                q_norm: f32w("q_norm")?.index_select(&head, 0)?,
                k_norm: f32w("k_norm")?.index_select(&head, 0)?,
            })
        } else {
            None
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
            qkv_rows: order,
            l_qkv: LoraSlot::default(),
            l_o: LoraSlot::default(),
            l_up_gate: LoraSlot::default(),
            l_down: LoraSlot::default(),
            #[cfg(feature = "fusednorm")]
            fused,
        })
    }

    /// The stack's adapters on this block (`p`: `layers.{i}`). The official layout's separate
    /// `proj.q`, `proj.k`, `proj.v` adapters of one file become one adapter on `qkv`: their `A`s
    /// stacked, their `B`s block-diagonal, its rows in `qkv`'s order; an adapter trained on
    /// ComfyUI's fused `proj.qkv` is taken as it is (rows reordered the same way).
    fn set_loras(&self, stack: &LoraStack, p: &str, dim: usize) -> Result<()> {
        let mut qkv = Vec::new();
        for (file, strength) in &stack.layers {
            if *strength == 0.0 {
                continue;
            }
            let mut parts = Vec::new();
            if let Some(e) = file.entries.get(&format!("{p}.proj.qkv")) {
                parts.push((e, 0usize, 3 * dim));
            }
            for (j, n) in ["q", "k", "v"].iter().enumerate() {
                if let Some(e) = file.entries.get(&format!("{p}.proj.{n}")) {
                    parts.push((e, j * dim, dim));
                }
            }
            if parts.is_empty() {
                continue;
            }
            let dev = parts[0].0.a.device().clone();
            let dtype = parts[0].0.a.dtype();
            let ranks: usize = parts.iter().map(|(e, _, _)| e.rank()).sum();
            // Each part's scale folded into its B, so one pair carries them all.
            let mut a_rows = Vec::with_capacity(parts.len());
            let mut b = Tensor::zeros((3 * dim, ranks), dtype, &dev)?;
            let mut col = 0;
            for (e, row0, rows) in parts {
                anyhow::ensure!(e.b.dim(0)? == rows, "{p}: a LoRA with {} output rows ({rows} expected)", e.b.dim(0)?);
                let r = e.rank();
                let scale = f64::from(strength * e.alpha / r.max(1) as f32);
                a_rows.push(e.a.clone());
                let block = (e.b.to_dtype(DType::F32)? * scale)?.to_dtype(dtype)?;
                let padded = block.pad_with_zeros(0, row0, 3 * dim - row0 - rows)?.pad_with_zeros(1, col, ranks - col - r)?;
                b = (b + padded)?;
                col += r;
            }
            let a = Tensor::cat(&a_rows, 0)?;
            let b = match &self.qkv_rows {
                Some(o) => b.index_select(&Tensor::from_vec(o.iter().map(|&i| i as u32).collect::<Vec<_>>(), o.len(), &dev)?, 0)?,
                None => b,
            };
            qkv.push(LoraPair::new(&a, &b, 1.0)?);
        }
        self.l_qkv.set(qkv);
        self.l_o.set(stack.pairs(&format!("{p}.proj.o"))?);
        self.l_up_gate.set(stack.pairs(&format!("{p}.proj.ffn_up_gate"))?);
        self.l_down.set(stack.pairs(&format!("{p}.proj.ffn_down"))?);
        Ok(())
    }

    /// The block in one-pass kernels: `x (1, S, dim)` holding one or more sequences back to back
    /// (`seqs`: each one's first row, length and `(2, len, D/2)` cos/sin per pair; the row-wise
    /// work runs once over all of them, the attention per sequence), `mods (1, 1, 6·dim)`
    /// (shift, scale, gate for attention, then for the feed-forward).
    #[cfg(feature = "fusednorm")]
    fn forward_fused(&self, f: &FusedNorms, x: &Tensor, seqs: &[(usize, usize, &Tensor)], mods: &Tensor, heads: usize, refs: Option<&RefLayer>) -> Result<Tensor> {
        use crate::krea2_fused::{head_rope, rms_mod};
        use crate::profile::scope;
        let (_, s, dim) = x.dims3()?;
        let hd = dim / heads;
        let dev = x.device();
        let (sf, df) = (s as f64, dim as f64);
        let ffn = self.up_gate_width();
        let y = scope("moss norm", 0.0, dev, || rms_mod(x, &f.norm1, Some((mods, 1, 0)), 1e-6))?;
        let qkv = scope("moss qkv", 2.0 * sf * df * 3.0 * df, dev, || self.l_qkv.add_to(self.qkv.forward(&y)?, &y))?; // (1, S, 3·dim)
        let mut outs = Vec::with_capacity(seqs.len());
        for &(r0, n, cs) in seqs {
            let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(1, r0, n)?.narrow(2, i * dim, dim)?.reshape((1, n, heads, hd))?) };
            let nf = n as f64;
            if let Some(r) = refs {
                // MOSS Edit: the sequence's queries over its own keys and the references' (cached,
                // already normed and rotated; the references never attend back).
                let k = head_rope(&part(1)?, &f.k_norm, Some(cs), 1e-6)?;
                let k = Tensor::cat(&[&k, &r.k], 1)?;
                let v = Tensor::cat(&[&part(2)?, &r.v], 1)?;
                let m = k.dim(1)? as f64;
                #[cfg(feature = "sage2")]
                if let Some(accum) = sage_attention()? {
                    use crate::sage2::{attention as sage, quant_v, rope_quant, Role};
                    let (cos, sin) = (cs.get(0)?.to_dtype(DType::BF16)?, cs.get(1)?.to_dtype(DType::BF16)?);
                    outs.push(scope("moss attention", 4.0 * nf * m * df, dev, || {
                        let q = rope_quant(&head_rope(&part(0)?, &f.q_norm, None, 1e-6)?, &cos, &sin, Role::Query)?;
                        // K is rotated already: quantized under the identity rotation.
                        let half = hd / 2;
                        let (one, zero) = (Tensor::ones((k.dim(1)?, half), DType::BF16, dev)?, Tensor::zeros((k.dim(1)?, half), DType::BF16, dev)?);
                        let k = rope_quant(&k, &one, &zero, Role::Key)?;
                        let v = quant_v(&v.to_dtype(DType::F16)?, accum)?;
                        Ok(sage(&q, &k, &v, 1.0 / (hd as f32).sqrt(), false)?.reshape((1, n, dim))?)
                    })?);
                    continue;
                }
                let q = scope("moss head rope", 0.0, dev, || Ok(head_rope(&part(0)?, &f.q_norm, Some(cs), 1e-6)?.squeeze(0)?))?;
                outs.push(scope("moss attention", 4.0 * nf * m * df, dev, || Ok(attention(&q, &k.squeeze(0)?, &v.squeeze(0)?)?.unsqueeze(0)?))?);
                continue;
            }
            #[cfg(feature = "sage2")]
            if let Some(accum) = sage_attention()? {
                // SageAttention2: q and k normed here, rotated and INT8-quantized in one kernel,
                // V in FP8.
                use crate::sage2::{attention as sage, quant_v, rope_quant, Role};
                let (cos, sin) = (cs.get(0)?.to_dtype(DType::BF16)?, cs.get(1)?.to_dtype(DType::BF16)?);
                outs.push(scope("moss attention", 4.0 * nf * nf * df, dev, || {
                    let q = rope_quant(&head_rope(&part(0)?, &f.q_norm, None, 1e-6)?, &cos, &sin, Role::Query)?;
                    let k = rope_quant(&head_rope(&part(1)?, &f.k_norm, None, 1e-6)?, &cos, &sin, Role::Key)?;
                    let v = quant_v(&part(2)?.to_dtype(DType::F16)?, accum)?;
                    Ok(sage(&q, &k, &v, 1.0 / (hd as f32).sqrt(), false)?.reshape((1, n, dim))?)
                })?);
                continue;
            }
            let q = scope("moss head rope", 0.0, dev, || Ok(head_rope(&part(0)?, &f.q_norm, Some(cs), 1e-6)?.squeeze(0)?))?;
            let k = scope("moss head rope", 0.0, dev, || Ok(head_rope(&part(1)?, &f.k_norm, Some(cs), 1e-6)?.squeeze(0)?))?;
            outs.push(scope("moss attention", 4.0 * nf * nf * df, dev, || Ok(attention(&q, &k, &part(2)?.squeeze(0)?)?.unsqueeze(0)?))?);
        }
        let a = if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 1)? };
        self.finish_fused(f, x, &a, mods, dev, sf, df, ffn)
    }

    /// A reference's pass through the block (MOSS Edit): `x (1, N, dim)` the references' rows,
    /// `cs` their `(2, N, D/2)` cos/sin, `mods` the t = 0 modulation; the rows out and this
    /// block's keys (normed, rotated) and values `(1, N, H, D)` for the target to attend to.
    #[cfg(feature = "fusednorm")]
    fn reference_fused(&self, f: &FusedNorms, x: &Tensor, cs: &Tensor, mods: &Tensor, heads: usize) -> Result<(Tensor, RefLayer)> {
        use crate::krea2_fused::{head_rope, rms_mod};
        let (_, s, dim) = x.dims3()?;
        let hd = dim / heads;
        let dev = x.device();
        let y = rms_mod(x, &f.norm1, Some((mods, 1, 0)), 1e-6)?;
        let qkv = self.l_qkv.add_to(self.qkv.forward(&y)?, &y)?;
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(2, i * dim, dim)?.reshape((1, s, heads, hd))?) };
        let q = head_rope(&part(0)?, &f.q_norm, Some(cs), 1e-6)?;
        let k = head_rope(&part(1)?, &f.k_norm, Some(cs), 1e-6)?;
        let v = part(2)?.contiguous()?;
        let a = attention(&q.squeeze(0)?, &k.squeeze(0)?, &v.squeeze(0)?)?.unsqueeze(0)?;
        let (sf, df) = (s as f64, dim as f64);
        let x = self.finish_fused(f, x, &a, mods, dev, sf, df, self.up_gate_width())?;
        Ok((x, RefLayer { k, v }))
    }

    /// [`Self::reference_fused`] on the candle chains: `x (N, dim)`, `rope (N, D)`; keys and
    /// values `(N, H, D)`.
    fn reference(&self, x: &Tensor, rope: &Tensor, m: &[Tensor; 6], heads: usize) -> Result<(Tensor, RefLayer)> {
        let (s, dim) = x.dims2()?;
        let hd = dim / heads;
        let [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f] = m;
        let y = self.norm1.forward(x)?;
        let y = (&y + y.broadcast_mul(scale_a)?)?.broadcast_add(shift_a)?;
        let qkv = self.l_qkv.add_to(self.qkv.forward(&y)?, &y)?;
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(1, i * dim, dim)?.reshape((s, heads, hd))?) };
        let q = rotate(&self.q_norm.forward(&part(0)?)?, rope)?;
        let k = rotate(&self.k_norm.forward(&part(1)?)?, rope)?;
        let v = part(2)?.contiguous()?;
        let att = attention(&q, &k, &v)?;
        let a = self.l_o.add_to(self.o.forward(&att)?, &att)?;
        let x = (x + a.broadcast_mul(gate_a)?)?;
        let y = self.norm2.forward(&x)?;
        let y = (&y + y.broadcast_mul(scale_f)?)?.broadcast_add(shift_f)?;
        let hg = self.l_up_gate.add_to(self.up_gate.forward(&y)?, &y)?;
        let f = hg.dim(1)? / 2;
        let h = situ(&hg.narrow(1, 0, f)?, &hg.narrow(1, f, f)?)?;
        let x = (&x + self.l_down.add_to(self.down.forward(&h)?, &h)?.broadcast_mul(gate_f)?)?;
        Ok((x, RefLayer { k, v }))
    }

    /// The fused block after attention: `a (1, S, dim)`.
    #[cfg(feature = "fusednorm")]
    #[allow(clippy::too_many_arguments)]
    fn finish_fused(&self, f: &FusedNorms, x: &Tensor, a: &Tensor, mods: &Tensor, dev: &Device, sf: f64, df: f64, ffn: f64) -> Result<Tensor> {
        use crate::krea2_fused::{gate_add, rms_mod};
        use crate::profile::scope;
        let s = x.dim(1)?;
        let o = scope("moss o", 2.0 * sf * df * df, dev, || self.l_o.add_to(self.o.forward(a)?, a))?;
        let x = scope("moss gate add", 0.0, dev, || gate_add(x, &o, mods, 2))?;
        let y = scope("moss norm", 0.0, dev, || rms_mod(&x, &f.norm2, Some((mods, 4, 3)), 1e-6))?;
        let hg = scope("moss up_gate", 2.0 * sf * df * 2.0 * ffn, dev, || self.l_up_gate.add_to(self.up_gate.forward(&y)?, &y))?;
        let hg = hg.reshape((s, hg.dim(2)?))?;
        // SiTU inside the down projection's activation quantize (ConvRot; 1 % faster than its
        // own pass), unless `QIR_MOSS_SITU_FUSED=0`.
        static FUSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fuse = *FUSE.get_or_init(|| std::env::var("QIR_MOSS_SITU_FUSED").map_or(true, |v| v.trim() != "0"));
        let fused = if fuse && self.l_down.is_empty() { scope("moss situ+down", 2.0 * sf * df * ffn, dev, || self.down.forward_situ(&hg))? } else { None };
        let d = match fused {
            Some(d) => d.unsqueeze(0)?,
            None => {
                let h = scope("moss situ", 0.0, dev, || super::fused::situ(&hg))?;
                scope("moss down", 2.0 * sf * df * ffn, dev, || Ok(self.l_down.add_to(self.down.forward(&h)?, &h)?.unsqueeze(0)?))?
            }
        };
        scope("moss gate add", 0.0, dev, || gate_add(&x, &d, mods, 5))
    }

    /// The feed-forward width (for the profile's operation counts).
    #[cfg(feature = "fusednorm")]
    fn up_gate_width(&self) -> f64 {
        8192.0
    }

    /// `m`: the six modulation rows `(6, dim)` in the run dtype.
    fn forward(&self, x: &Tensor, rope: &Tensor, m: &[Tensor; 6], heads: usize, refs: Option<&RefLayer>) -> Result<Tensor> {
        let (s, dim) = x.dims2()?;
        let hd = dim / heads;
        let [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f] = m;
        let y = self.norm1.forward(x)?;
        let y = (&y + y.broadcast_mul(scale_a)?)?.broadcast_add(shift_a)?;
        let qkv = self.l_qkv.add_to(self.qkv.forward(&y)?, &y)?;
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(1, i * dim, dim)?.reshape((s, heads, hd))?) };
        let q = rotate(&self.q_norm.forward(&part(0)?)?, rope)?;
        let k = rotate(&self.k_norm.forward(&part(1)?)?, rope)?;
        let att = match refs {
            Some(r) => attention(&q, &Tensor::cat(&[&k, &r.k], 0)?, &Tensor::cat(&[&part(2)?, &r.v], 0)?)?,
            None => attention(&q, &k, &part(2)?)?,
        };
        let a = self.l_o.add_to(self.o.forward(&att)?, &att)?;
        let x = (x + a.broadcast_mul(gate_a)?)?;
        let y = self.norm2.forward(&x)?;
        let y = (&y + y.broadcast_mul(scale_f)?)?.broadcast_add(shift_f)?;
        let hg = self.l_up_gate.add_to(self.up_gate.forward(&y)?, &y)?;
        let f = hg.dim(1)? / 2;
        let h = situ(&hg.narrow(1, 0, f)?, &hg.narrow(1, f, f)?)?;
        Ok((&x + self.l_down.add_to(self.down.forward(&h)?, &h)?.broadcast_mul(gate_f)?)?)
    }
}

/// One block's keys and values of MOSS Edit's references.
#[derive(Clone)]
struct RefLayer {
    k: Tensor,
    v: Tensor,
}

/// MOSS Edit's references through every block, made once per edit (the reference branch runs
/// at t = 0 and attends only to itself, so it is the same at every step and for both prompts).
pub struct RefCache {
    layers: Vec<RefLayer>,
}

impl RefCache {
    /// The tokens it holds.
    pub fn tokens(&self) -> usize {
        self.layers.first().and_then(|l| l.v.dims().iter().rev().nth(2).copied()).unwrap_or(0)
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
    /// The last two grids' RoPE tables (run-dtype rows, `(cos, sin)` per pair): every step of a
    /// picture asks for the same ones, for its prompt and its negative.
    tables: std::sync::Mutex<Vec<Tables>>,
    /// The adapters on the blocks now, and the modules of theirs the blocks lack.
    bound: std::sync::Mutex<Option<(LoraStack, Vec<String>)>>,
}

#[derive(Clone)]
struct Tables {
    /// `(h, w, text_len)` they are for.
    key: (usize, usize, usize),
    rope: Tensor,
    #[cfg_attr(not(feature = "fusednorm"), allow(dead_code))]
    cs: Tensor,
}

impl MossDit {
    pub fn load(cfg: &MossConfig, w: &Weights) -> Result<Self> {
        let layers = (0..cfg.layers).map(|i| Block::load(w, &format!("layers.{i}"), cfg.heads)).collect::<Result<Vec<_>>>()?;
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
            tables: std::sync::Mutex::new(Vec::new()),
            bound: std::sync::Mutex::new(None),
        })
    }

    /// [`Self::rope`] in the run dtype and its `(2, tokens, head_dim/2)` cos/sin form, made once
    /// per grid.
    fn tables(&self, h: usize, w: usize, l: usize) -> Result<Tables> {
        let mut slot = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = slot.iter().find(|t| t.key == (h, w, l)) {
            return Ok(t.clone());
        }
        let rope32 = self.rope(h, w, l)?;
        let half = self.cfg.head_dim() / 2;
        let cs = Tensor::stack(&[rope32.narrow(1, half, half)?, rope32.narrow(1, 0, half)?], 0)?.contiguous()?;
        let t = Tables { key: (h, w, l), rope: rope32.to_dtype(self.dtype)?, cs };
        // The prompt's and the negative's (their text lengths differ).
        if slot.len() >= 2 {
            slot.remove(0);
        }
        slot.push(t.clone());
        Ok(t)
    }

    pub fn config(&self) -> &MossConfig {
        &self.cfg
    }

    /// Put `stack`'s adapters on the blocks (an empty stack takes them off); the modules it names
    /// that the blocks do not have come back (a file for another model).
    pub fn set_loras(&self, stack: &LoraStack) -> Result<Vec<String>> {
        let mut bound = self.bound.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((previous, unmatched)) = bound.as_ref() {
            if previous.same_adapters(stack) {
                return Ok(unmatched.clone());
            }
        }
        *bound = None;
        for (i, b) in self.layers.iter().enumerate() {
            b.set_loras(stack, &format!("layers.{i}"), self.cfg.dim)?;
        }
        let n = self.layers.len();
        let known = |m: &str| -> bool {
            let Some(rest) = m.strip_prefix("layers.") else { return false };
            let Some((idx, tail)) = rest.split_once('.') else { return false };
            idx.parse::<usize>().is_ok_and(|i| i < n)
                && matches!(tail, "proj.q" | "proj.k" | "proj.v" | "proj.qkv" | "proj.o" | "proj.ffn_up_gate" | "proj.ffn_down")
        };
        let unmatched: Vec<String> = stack.unmatched(&known).into_iter().map(str::to_owned).collect();
        *bound = Some((stack.clone(), unmatched.clone()));
        Ok(unmatched)
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

    /// MOSS Edit's reference rows `[sin | cos]` `(h·w, head_dim)` f32 for reference `index`
    /// (0-based): frame `index + 1`, text index 0, height and width centred on its own grid.
    pub fn reference_rope(&self, h: usize, w: usize, index: usize) -> Result<Tensor> {
        let nb = self.cfg.head_dim() / 8;
        let bands: Vec<f32> = (0..nb).map(|i| 1.0 / self.cfg.rope_base.powf(i as f32 / nb as f32)).collect();
        let mut rows = Vec::with_capacity(h * w * 8 * nb);
        let (ch, cw) = ((h as f32 - 1.0) / 2.0, (w as f32 - 1.0) / 2.0);
        for y in 0..h {
            for x in 0..w {
                let pos = [(index + 1) as f32, 0.0, y as f32 - ch, x as f32 - cw];
                let phase: Vec<f32> = pos.iter().flat_map(|p| bands.iter().map(move |b| p * b)).collect();
                rows.extend(phase.iter().map(|p| p.sin()));
                rows.extend(phase.iter().map(|p| p.cos()));
            }
        }
        Ok(Tensor::from_vec(rows, (h * w, 8 * nb), &self.dev)?)
    }

    /// Run MOSS Edit's references `(1, channels, h, w)` (the FLUX.2 encoder's latents, the
    /// picture being edited first) through every block once, at t = 0, attending only to each
    /// other; the blocks' keys and values for the sampling steps.
    pub fn prepare_references(&self, refs: &[Tensor]) -> Result<RefCache> {
        anyhow::ensure!(!refs.is_empty(), "an edit needs a reference");
        let mut tokens = Vec::with_capacity(refs.len());
        let mut ropes = Vec::with_capacity(refs.len());
        for (i, r) in refs.iter().enumerate() {
            let (_, c, h, w) = r.dims4()?;
            anyhow::ensure!(c == self.cfg.channels, "a reference of {c} channels (the model takes {})", self.cfg.channels);
            tokens.push(self.video_embedder.forward(&r.reshape((c, h * w))?.t()?.contiguous()?.to_dtype(self.dtype)?)?);
            ropes.push(self.reference_rope(h, w, i)?);
        }
        let x = Tensor::cat(&tokens, 0)?;
        let rope32 = Tensor::cat(&ropes, 0)?;
        let (m, _) = self.modulation(0.0)?;
        let mut layers = Vec::with_capacity(self.layers.len());
        #[cfg(feature = "fusednorm")]
        if self.layers.first().is_some_and(|b| b.fused.is_some()) {
            let half = self.cfg.head_dim() / 2;
            let cs = Tensor::stack(&[rope32.narrow(1, half, half)?, rope32.narrow(1, 0, half)?], 0)?.contiguous()?;
            let mods = Tensor::cat(&m, 1)?.unsqueeze(0)?.contiguous()?;
            let mut x = x.unsqueeze(0)?;
            for layer in &self.layers {
                let Some(f) = &layer.fused else { anyhow::bail!("a block without its fused weights") };
                let (next, kv) = layer.reference_fused(f, &x, &cs, &mods, self.cfg.heads)?;
                layers.push(kv);
                x = next;
            }
            return Ok(RefCache { layers });
        }
        let rope = rope32.to_dtype(self.dtype)?;
        let mut x = x;
        for layer in &self.layers {
            let (next, kv) = layer.reference(&x, &rope, &m, self.cfg.heads)?;
            layers.push(kv);
            x = next;
        }
        Ok(RefCache { layers })
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
        self.forward_refs(x, timestep, ctx, skip, None)
    }

    /// [`Self::forward`] attending to MOSS Edit's references too.
    pub fn forward_refs(&self, x: &Tensor, timestep: f32, ctx: &Tensor, skip: &[usize], refs: Option<&RefCache>) -> Result<Tensor> {
        if let Some(r) = refs {
            anyhow::ensure!(r.layers.len() == self.layers.len(), "a reference cache of {} blocks for {}", r.layers.len(), self.layers.len());
        }
        let layer_refs = |i: usize| refs.map(|r| &r.layers[i]);
        let (_, c, h, w) = x.dims4()?;
        anyhow::ensure!(c == self.cfg.channels, "a latent of {c} channels (the model takes {})", self.cfg.channels);
        let n = h * w;
        let l = ctx.dim(0)?;
        let tables = self.tables(h, w, l)?;
        let rope = tables.rope.clone();
        let video = self.video_embedder.forward(&x.reshape((c, n))?.t()?.contiguous()?.to_dtype(self.dtype)?)?;
        let text = self.text(ctx, &rope.narrow(0, n, l)?)?;
        let (m, head) = self.modulation(timestep)?;
        let mut tokens = Tensor::cat(&[video, text], 0)?;
        #[cfg(feature = "fusednorm")]
        if self.layers.first().is_some_and(|b| b.fused.is_some()) {
            // The six rows as one (1, 1, 6·dim) tensor, the RoPE as (cos, sin) per pair.
            let mods = Tensor::cat(&m, 1)?.unsqueeze(0)?.contiguous()?;
            let s = tokens.dim(0)?;
            let mut x = tokens.unsqueeze(0)?;
            for (i, layer) in self.layers.iter().enumerate() {
                if let (false, Some(f)) = (skip.contains(&i), &layer.fused) {
                    x = layer.forward_fused(f, &x, &[(0, s, &tables.cs)], &mods, self.cfg.heads, layer_refs(i))?;
                }
            }
            tokens = x.squeeze(0)?;
        }
        #[cfg(feature = "fusednorm")]
        let unfused = self.layers.first().is_none_or(|b| b.fused.is_none());
        #[cfg(not(feature = "fusednorm"))]
        let unfused = true;
        if unfused {
            for (i, layer) in self.layers.iter().enumerate() {
                if !skip.contains(&i) {
                    tokens = layer.forward(&tokens, &rope, &m, self.cfg.heads, layer_refs(i))?;
                }
            }
        }
        self.head(&tokens.narrow(0, 0, n)?, &head, c, h, w)
    }

    /// The output head on the image rows `(n, dim)`: the velocity `(1, c, h, w)` f32.
    fn head(&self, rows: &Tensor, head: &[Tensor; 2], c: usize, h: usize, w: usize) -> Result<Tensor> {
        let y = self.final_norm.forward(rows)?;
        let y = (&y + y.broadcast_mul(&head[1])?)?.broadcast_add(&head[0])?;
        let y = self.final_linear.forward(&y)?; // (n, channels)
        Ok(y.to_dtype(DType::F32)?.t()?.contiguous()?.reshape((1, c, h, w))?)
    }

    /// Both velocities of a guided step at once: [`Self::forward`] for `cond` with no block
    /// skipped and for `uncond` with `uncond_skip` left out. With the fused blocks the two
    /// sequences share every row-wise pass (norms, the four GEMMs at twice the rows, SiTU, the
    /// residuals), attention stays per sequence; the image tokens' embedding is made once.
    pub fn forward_pair(&self, x: &Tensor, timestep: f32, cond: &Tensor, uncond: &Tensor, uncond_skip: &[usize]) -> Result<(Tensor, Tensor)> {
        self.forward_pair_refs(x, timestep, cond, uncond, uncond_skip, None)
    }

    /// [`Self::forward_pair`] attending to MOSS Edit's references too (both prompts see them).
    pub fn forward_pair_refs(
        &self,
        x: &Tensor,
        timestep: f32,
        cond: &Tensor,
        uncond: &Tensor,
        uncond_skip: &[usize],
        refs: Option<&RefCache>,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg_attr(not(feature = "fusednorm"), allow(unused_variables))]
        let layer_refs = |i: usize| refs.map(|r| &r.layers[i]);
        #[cfg(feature = "fusednorm")]
        if self.layers.first().is_some_and(|b| b.fused.is_some()) {
            let (_, c, h, w) = x.dims4()?;
            anyhow::ensure!(c == self.cfg.channels, "a latent of {c} channels (the model takes {})", self.cfg.channels);
            let n = h * w;
            let (lc, lu) = (cond.dim(0)?, uncond.dim(0)?);
            let (tc, tu) = (self.tables(h, w, lc)?, self.tables(h, w, lu)?);
            let video = self.video_embedder.forward(&x.reshape((c, n))?.t()?.contiguous()?.to_dtype(self.dtype)?)?;
            let text_c = self.text(cond, &tc.rope.narrow(0, n, lc)?)?;
            let text_u = self.text(uncond, &tu.rope.narrow(0, n, lu)?)?;
            let (m, head) = self.modulation(timestep)?;
            let mods = Tensor::cat(&m, 1)?.unsqueeze(0)?.contiguous()?;
            let (sc, su) = (n + lc, n + lu);
            let mut xs = Tensor::cat(&[&video, &text_c, &video, &text_u], 0)?.unsqueeze(0)?;
            for (i, layer) in self.layers.iter().enumerate() {
                let Some(f) = &layer.fused else { continue };
                xs = if uncond_skip.contains(&i) {
                    let xc = layer.forward_fused(f, &xs.narrow(1, 0, sc)?, &[(0, sc, &tc.cs)], &mods, self.cfg.heads, layer_refs(i))?;
                    Tensor::cat(&[&xc, &xs.narrow(1, sc, su)?], 1)?
                } else {
                    layer.forward_fused(f, &xs, &[(0, sc, &tc.cs), (sc, su, &tu.cs)], &mods, self.cfg.heads, layer_refs(i))?
                };
            }
            let xs = xs.squeeze(0)?;
            let rows = Tensor::cat(&[&xs.narrow(0, 0, n)?, &xs.narrow(0, sc, n)?], 0)?;
            let y = self.final_norm.forward(&rows)?;
            let y = (&y + y.broadcast_mul(&head[1])?)?.broadcast_add(&head[0])?;
            let y = self.final_linear.forward(&y)?.to_dtype(DType::F32)?; // (2n, channels)
            let v = |r0: usize| -> Result<Tensor> { Ok(y.narrow(0, r0, n)?.t()?.contiguous()?.reshape((1, c, h, w))?) };
            return Ok((v(0)?, v(n)?));
        }
        Ok((self.forward_refs(x, timestep, cond, &[], refs)?, self.forward_refs(x, timestep, uncond, uncond_skip, refs)?))
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
