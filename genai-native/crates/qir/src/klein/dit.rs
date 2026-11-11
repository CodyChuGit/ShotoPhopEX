//! FLUX 3 Image Klein's DiT on candle (CUDA, or the CPU for tests), from ComfyUI's single file
//! (`comfy/ldm/flux3/model.py`'s names): five blocks for each kind of token (the picture being
//! made, the reference pictures and the text), each kind attending only to itself, then 28
//! single-stream blocks over all of them. Each block is one LayerNorm + modulation feeding a
//! parallel attention and SwiGLU MLP under one gate. Every block of a kind shares one
//! modulation, set by the time (the picture's own, 0 for the references and the text) plus, in
//! the distilled flow map, the time the step jumps to.
//!
//! The same model as `gn-mlx`'s Klein (which reads mflux's diffusers-named files); this one reads
//! the reference's names. The residual stream is kept in f32; each linear sees its input in the
//! run dtype (bf16 on CUDA). Linears are ConvRot INT8 where the file has them (`Weights`), dense
//! otherwise.

use candle_core::{DType, Device, Tensor, D};

use crate::lora::{LoraPair, LoraSlot, LoraStack};
use crate::moss::weights::{QLinear, Weights};
use crate::Result;

pub const HIDDEN: usize = 3072;
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const DEPTH: usize = 5;
pub const DEPTH_SINGLE: usize = 28;
pub const CHANNELS: usize = 128;
pub const TEXT_DIM: usize = 20480;
/// RoPE's four axes `(t, row, column, layer)`, 32 dimensions each.
const AXES: [usize; 4] = [32, 32, 32, 32];
const THETA: f64 = 10_000.0;
const FREQ_DIM: usize = 256;

/// The prefix the file's tensors carry, if any (`model.diffusion_model.` in a full checkpoint).
fn prefix_of(w: &Weights) -> &'static str {
    ["", "model.diffusion_model.", "diffusion_model."].into_iter().find(|p| w.contains(&format!("{p}txt_in.weight"))).unwrap_or("")
}

/// A plain weight that may be stored as `weight` or `scale` (RMSNorm's two spellings).
fn norm_weight(w: &Weights, module: &str) -> Result<Tensor> {
    for s in ["weight", "scale"] {
        let n = format!("{module}.{s}");
        if w.contains(&n) {
            return w.get_as(&n, DType::F32);
        }
    }
    anyhow::bail!("{module}: no weight")
}

/// A linear with its LoRA slot.
struct Lin {
    base: QLinear,
    lora: LoraSlot,
}

impl Lin {
    fn load(w: &Weights, module: &str) -> Result<Self> {
        Ok(Self { base: w.linear(module)?, lora: LoraSlot::default() })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.lora.add_to(self.base.forward(x)?, x)
    }
}

/// The names a LoRA file may give one of the reference's modules (`content_mode_blocks.image.0.q_proj`):
/// the reference's own and diffusers' (`mode_blocks.image.0.attn.to_q`), which mflux and some
/// trainers write.
pub fn lora_names(module: &str) -> Vec<String> {
    let mut out = vec![module.to_string()];
    let parts = [
        ("q_proj", "attn.to_q"),
        ("k_proj", "attn.to_k"),
        ("v_proj", "attn.to_v"),
        ("attn_out", "attn.to_out"),
        ("mlp_in", "ff.linear_in"),
        ("mlp_out", "ff.linear_out"),
    ];
    let blocks = [
        ("content_mode_blocks.image_cond.", "mode_blocks.image_cond."),
        ("content_mode_blocks.image.", "mode_blocks.image."),
        ("txt_mode_blocks.", "txt_mode_blocks."),
        ("single_blocks.", "single_transformer_blocks."),
    ];
    for (ours, theirs) in blocks {
        let Some(rest) = module.strip_prefix(ours) else { continue };
        let Some((index, part)) = rest.split_once('.') else { break };
        if let Some((_, d)) = parts.iter().find(|(a, _)| *a == part) {
            out.push(format!("{theirs}{index}.{d}"));
            if *d == "attn.to_out" {
                out.push(format!("{theirs}{index}.attn.to_out.0"));
            }
        }
        break;
    }
    out
}

/// The stack's pairs for `module` under any of its names.
fn pairs_for(stack: &LoraStack, module: &str) -> Result<Vec<LoraPair>> {
    let mut out = Vec::new();
    for n in lora_names(module) {
        out.extend(stack.pairs(&n)?);
    }
    Ok(out)
}

/// q, k and v: one fused linear (`qkv`) or three.
enum Qkv {
    Fused(Lin),
    Split([Lin; 3]),
}

struct Block {
    qkv: Qkv,
    attn_out: Lin,
    mlp_in: Lin,
    mlp_out: Lin,
    /// `(HEAD_DIM,)` f32.
    q_norm: Tensor,
    k_norm: Tensor,
}

/// A block's shift, scale and gate `(rows, HIDDEN)` f32: one row broadcast to every token of its
/// span, or one row per token.
#[derive(Clone)]
struct Modulation {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

/// LayerNorm without affine (eps 1e-6) of `x (S, HIDDEN)` f32, then `(1 + scale)·x + shift`.
fn modulate(x: &Tensor, m: &Modulation) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let c = x.broadcast_sub(&mean)?;
    let var = c.sqr()?.mean_keepdim(D::Minus1)?;
    let y = c.broadcast_mul(&(var + 1e-6)?.sqrt()?.recip()?)?;
    Ok((&y + y.broadcast_mul(&m.scale)?)?.broadcast_add(&m.shift)?)
}

/// RMSNorm (eps 1e-6) of `x (S, H, D)` in f32 times `w (D,)`.
fn rms(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let y = x.broadcast_mul(&(x.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?)?;
    Ok(y.broadcast_mul(w)?)
}

/// Bidirectional attention over `q`, `k`, `v` `(1, H, S, D)` in the run dtype: `(S, H·D)`.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (_, h, s, d) = q.dims4()?;
    let scale = 1.0 / (d as f64).sqrt();
    #[cfg(feature = "flash")]
    if q.device().is_cuda() && q.dtype() == DType::BF16 && d == crate::flash::HEAD_DIM {
        // The kernel answers `(1, S, H, D)`, already in the order the rows are read back in.
        let out = crate::flash::attention(q, k, v, scale as f32)?;
        return Ok(out.squeeze(0)?.reshape((s, h * d))?);
    }
    const ROWS: usize = 1024;
    let qh = q.squeeze(0)?.to_dtype(DType::F32)?.contiguous()?;
    let kt = k.squeeze(0)?.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
    let vh = v.squeeze(0)?.to_dtype(DType::F32)?.contiguous()?;
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

/// RoPE tables `(cos, sin)` `(S, HEAD_DIM/2)` f32 for `(t, row, column, layer)` positions; the
/// reference's adjacent pairs ([`candle_nn::rotary_emb::rope_i`]).
pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    pub fn new(ids: &[[f32; 4]], dev: &Device) -> Result<Self> {
        let half = AXES.iter().sum::<usize>() / 2;
        let (mut cos, mut sin) = (Vec::with_capacity(ids.len() * half), Vec::with_capacity(ids.len() * half));
        for id in ids {
            for (axis, &d) in AXES.iter().enumerate() {
                for i in (0..d).step_by(2) {
                    let omega = 1.0 / THETA.powf(i as f64 / d as f64);
                    let a = f64::from(id[axis]) * omega;
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        Ok(Self { cos: Tensor::from_vec(cos, (ids.len(), half), dev)?, sin: Tensor::from_vec(sin, (ids.len(), half), dev)? })
    }

    /// `x (1, H, S, D)` f32, rotated in f32.
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::rotary_emb::rope_i(&x.contiguous()?, &self.cos, &self.sin)?)
    }

    /// Rows `start..start+len`.
    fn rows(&self, start: usize, len: usize) -> Result<Self> {
        Ok(Self { cos: self.cos.narrow(0, start, len)?, sin: self.sin.narrow(0, start, len)? })
    }
}

impl Block {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        let l = |s: &str| Lin::load(w, &format!("{p}.{s}"));
        let qkv = if w.contains(&format!("{p}.qkv.weight")) { Qkv::Fused(l("qkv")?) } else { Qkv::Split([l("q_proj")?, l("k_proj")?, l("v_proj")?]) };
        Ok(Self {
            qkv,
            attn_out: l("attn_out")?,
            mlp_in: l("mlp_in")?,
            mlp_out: l("mlp_out")?,
            q_norm: norm_weight(w, &format!("{p}.norm.query_norm"))?,
            k_norm: norm_weight(w, &format!("{p}.norm.key_norm"))?,
        })
    }

    /// Put `stack`'s adapters on this block (`p`: its name, e.g. `single_blocks.3`). Separate
    /// q/k/v adapters on a fused `qkv` become one: their `A`s stacked, their `B`s block-diagonal.
    fn set_loras(&self, stack: &LoraStack, p: &str) -> Result<()> {
        match &self.qkv {
            Qkv::Split(l) => {
                for (lin, n) in l.iter().zip(["q_proj", "k_proj", "v_proj"]) {
                    lin.lora.set(pairs_for(stack, &format!("{p}.{n}"))?);
                }
            }
            Qkv::Fused(lin) => {
                let mut fused = pairs_for(stack, &format!("{p}.qkv"))?;
                for (file, strength) in &stack.layers {
                    if *strength == 0.0 {
                        continue;
                    }
                    // This file's q, k and v adapters (under any of their names) as one pair on
                    // qkv: the `A`s stacked, the `B`s block-diagonal, each part's scale in its B.
                    let entry = |n: &str| lora_names(&format!("{p}.{n}")).into_iter().find_map(|m| file.entries.get(&m));
                    let parts: Vec<(usize, &crate::lora::Entry)> =
                        ["q_proj", "k_proj", "v_proj"].iter().enumerate().filter_map(|(j, n)| entry(n).map(|e| (j, e))).collect();
                    let Some((_, first)) = parts.first() else { continue };
                    let (dev, dtype) = (first.a.device().clone(), first.a.dtype());
                    let ranks: usize = parts.iter().map(|(_, e)| e.rank()).sum();
                    let mut a_rows = Vec::with_capacity(parts.len());
                    let mut b = Tensor::zeros((3 * HIDDEN, ranks), dtype, &dev)?;
                    let mut col = 0;
                    for (j, e) in parts {
                        anyhow::ensure!(e.b.dim(0)? == HIDDEN, "{p}: a LoRA with {} output rows ({HIDDEN} expected)", e.b.dim(0)?);
                        let r = e.rank();
                        let scale = f64::from(strength * e.alpha / r.max(1) as f32);
                        a_rows.push(e.a.clone());
                        let blk = (e.b.to_dtype(DType::F32)? * scale)?.to_dtype(dtype)?;
                        b = (b + blk.pad_with_zeros(0, j * HIDDEN, (2 - j) * HIDDEN)?.pad_with_zeros(1, col, ranks - col - r)?)?;
                        col += r;
                    }
                    fused.push(LoraPair::new(&Tensor::cat(&a_rows, 0)?, &b, 1.0)?);
                }
                lin.lora.set(fused);
            }
        }
        self.attn_out.lora.set(pairs_for(stack, &format!("{p}.attn_out"))?);
        self.mlp_in.lora.set(pairs_for(stack, &format!("{p}.mlp_in"))?);
        self.mlp_out.lora.set(pairs_for(stack, &format!("{p}.mlp_out"))?);
        Ok(())
    }

    /// `x (S, HIDDEN)` f32 to `x + gate·(attention + MLP)`, both from the same modulated input.
    /// `spans`: each span's first row, length and modulation (rows broadcast over the span).
    fn forward(&self, x: &Tensor, rope: &Rope, spans: &[(usize, usize, &Modulation)], dt: DType) -> Result<Tensor> {
        let s = x.dim(0)?;
        let y = if spans.len() == 1 {
            modulate(x, spans[0].2)?
        } else {
            let parts = spans.iter().map(|(start, len, m)| modulate(&x.narrow(0, *start, *len)?, m)).collect::<Result<Vec<_>>>()?;
            Tensor::cat(&parts, 0)?
        };
        let y = y.to_dtype(dt)?;
        let (q, k, v) = match &self.qkv {
            Qkv::Fused(l) => {
                let qkv = l.forward(&y)?;
                (qkv.narrow(1, 0, HIDDEN)?, qkv.narrow(1, HIDDEN, HIDDEN)?, qkv.narrow(1, 2 * HIDDEN, HIDDEN)?)
            }
            Qkv::Split([q, k, v]) => (q.forward(&y)?, k.forward(&y)?, v.forward(&y)?),
        };
        let heads = |t: Tensor| -> Result<Tensor> { Ok(t.reshape((s, HEADS, HEAD_DIM))?) };
        let first = |t: Tensor| -> Result<Tensor> { Ok(t.transpose(0, 1)?.unsqueeze(0)?.contiguous()?) };
        let q = rope.apply(&first(rms(&heads(q)?, &self.q_norm)?)?)?.to_dtype(dt)?;
        let k = rope.apply(&first(rms(&heads(k)?, &self.k_norm)?)?)?.to_dtype(dt)?;
        let v = first(heads(v)?)?;
        let att = attention(&q, &k, &v)?;
        let a = self.attn_out.forward(&att)?.to_dtype(DType::F32)?;
        let gu = self.mlp_in.forward(&y)?;
        let f = gu.dim(1)? / 2;
        let gate = gu.narrow(1, 0, f)?.to_dtype(DType::F32)?;
        let up = gu.narrow(1, f, f)?.to_dtype(DType::F32)?;
        let hid = candle_nn::ops::silu(&gate)?.mul(&up)?.to_dtype(dt)?;
        let out = (a + self.mlp_out.forward(&hid)?.to_dtype(DType::F32)?)?;
        if spans.len() == 1 {
            return Ok((x + out.broadcast_mul(&spans[0].2.gate)?)?);
        }
        let parts = spans
            .iter()
            .map(|(start, len, m)| -> Result<Tensor> { Ok((x.narrow(0, *start, *len)? + out.narrow(0, *start, *len)?.broadcast_mul(&m.gate)?)?) })
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&parts, 0)?)
    }
}

/// bf16's rounding of `v` (to nearest, ties to even), as the reference carries its times.
pub fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// The time embedding's MLP (`out_layer(silu(in_layer(sinusoid)))`).
struct TimeEmbedder {
    l1: QLinear,
    l2: QLinear,
}

impl TimeEmbedder {
    fn load(w: &Weights, n: &str) -> Result<Self> {
        Ok(Self { l1: w.linear(&format!("{n}.in_layer"))?, l2: w.linear(&format!("{n}.out_layer"))? })
    }

    /// `(1, HIDDEN)` f32 for the time `t` in 0..1 (the reference embeds it in bf16).
    fn forward(&self, t: f32, dt: DType, dev: &Device) -> Result<Tensor> {
        let half = FREQ_DIM / 2;
        let t = bf16_round(bf16_round(t) * 1000.0);
        let mut emb = vec![0f32; FREQ_DIM];
        for i in 0..half {
            let freq = (-(10_000f32.ln()) * i as f32 / half as f32).exp();
            let a = t * freq;
            emb[i] = bf16_round(a.cos());
            emb[i + half] = bf16_round(a.sin());
        }
        let emb = Tensor::from_vec(emb, (1, FREQ_DIM), dev)?.to_dtype(dt)?;
        Ok(self.l2.forward(&candle_nn::ops::silu(&self.l1.forward(&emb)?)?)?.to_dtype(DType::F32)?)
    }
}

/// `lin(silu(vec))` split into shift, scale and gate.
fn modulation(lin: &QLinear, vec: &Tensor, dt: DType) -> Result<Modulation> {
    let m = lin.forward(&candle_nn::ops::silu(&vec.to_dtype(dt)?)?)?.to_dtype(DType::F32)?;
    Ok(Modulation { shift: m.narrow(1, 0, HIDDEN)?, scale: m.narrow(1, HIDDEN, HIDDEN)?, gate: m.narrow(1, 2 * HIDDEN, HIDDEN)? })
}

/// The tokens and positions that stay the same at every step of one picture.
pub struct Prepared {
    /// The references' embedded tokens `(R, HIDDEN)` f32, if any.
    cond: Option<Tensor>,
    /// The text's `(L, HIDDEN)` f32.
    text: Tensor,
    image_len: usize,
    rope_image: Rope,
    rope_cond: Option<Rope>,
    rope_text: Rope,
    rope_single: Rope,
}

pub struct KleinDit {
    dtype: DType,
    dev: Device,
    flow_map: bool,
    time: TimeEmbedder,
    target_time: Option<TimeEmbedder>,
    /// image, image_cond, txt.
    early: [QLinear; 3],
    single_mod: [QLinear; 3],
    x_image: QLinear,
    x_cond: QLinear,
    context: QLinear,
    image_blocks: Vec<Block>,
    cond_blocks: Vec<Block>,
    text_blocks: Vec<Block>,
    single: Vec<Block>,
    final_mod: QLinear,
    final_linear: QLinear,
    /// The adapters on the blocks now and the modules of theirs the blocks lack.
    bound: std::sync::Mutex<Option<(LoraStack, Vec<String>)>>,
}

impl KleinDit {
    pub fn load(w: &Weights) -> Result<Self> {
        let pre = prefix_of(w);
        let n = |s: &str| format!("{pre}{s}");
        let lin = |s: &str| w.linear(&n(s));
        let mods = |kind: &str| -> Result<[QLinear; 3]> {
            Ok([lin(&format!("{kind}.image.lin"))?, lin(&format!("{kind}.image_cond.lin"))?, lin(&format!("{kind}.txt.lin"))?])
        };
        let blocks = |prefix: &str, count: usize| -> Result<Vec<Block>> { (0..count).map(|i| Block::load(w, &n(&format!("{prefix}.{i}")))).collect() };
        let flow_map = w.contains(&n("ref_time_in.in_layer.weight"));
        Ok(Self {
            dtype: w.dtype(),
            dev: w.device().clone(),
            flow_map,
            time: TimeEmbedder::load(w, &n("time_in"))?,
            target_time: if flow_map { Some(TimeEmbedder::load(w, &n("ref_time_in"))?) } else { None },
            early: mods("early_stream_modulations")?,
            single_mod: mods("single_stream_modulations")?,
            x_image: lin("emb_in.image")?,
            x_cond: lin("emb_in.image_cond")?,
            context: lin("txt_in")?,
            image_blocks: blocks("content_mode_blocks.image", DEPTH)?,
            cond_blocks: blocks("content_mode_blocks.image_cond", DEPTH)?,
            text_blocks: blocks("txt_mode_blocks", DEPTH)?,
            single: blocks("single_blocks", DEPTH_SINGLE)?,
            final_mod: lin("final_layer.image.adaLN_modulation.1")?,
            final_linear: lin("final_layer.image.linear")?,
            bound: std::sync::Mutex::new(None),
        })
    }

    /// Whether this is the distilled flow map (16 steps, no CFG), not Base.
    pub fn is_flow_map(&self) -> bool {
        self.flow_map
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Put `stack`'s adapters on every block (an empty stack takes them off); the modules it
    /// names that Klein does not have come back (a file for another model).
    pub fn set_loras(&self, stack: &LoraStack) -> Result<Vec<String>> {
        let mut bound = self.bound.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((previous, unmatched)) = bound.as_ref() {
            if previous.same_adapters(stack) {
                return Ok(unmatched.clone());
            }
        }
        *bound = None;
        for (prefix, blocks) in [
            ("content_mode_blocks.image", &self.image_blocks),
            ("content_mode_blocks.image_cond", &self.cond_blocks),
            ("txt_mode_blocks", &self.text_blocks),
            ("single_blocks", &self.single),
        ] {
            for (i, b) in blocks.iter().enumerate() {
                b.set_loras(stack, &format!("{prefix}.{i}"))?;
            }
        }
        let known = |m: &str| -> bool { known_module(m) };
        let unmatched: Vec<String> = stack.unmatched(&known).into_iter().map(str::to_owned).collect();
        *bound = Some((stack.clone(), unmatched.clone()));
        Ok(unmatched)
    }

    /// The vectors `(1, HIDDEN)` f32 of the picture (time `t`) and of the rest (time 0), plus
    /// the target time's for a flow map.
    fn vectors(&self, t: f32, target: Option<f32>) -> Result<(Tensor, Tensor)> {
        let (dt, dev) = (self.dtype, &self.dev);
        let mut image = self.time.forward(t, dt, dev)?;
        let mut context = self.time.forward(0.0, dt, dev)?;
        match (&self.target_time, target) {
            (Some(tt), Some(r)) => {
                let r = tt.forward(r, dt, dev)?;
                image = (image + &r)?;
                context = (context + r)?;
            }
            (None, None) => {}
            _ => anyhow::bail!("a target time is needed by, and only by, the distilled flow map"),
        }
        Ok((image, context))
    }

    /// The text's and the references' embedded tokens and every position, once per picture.
    /// `text (L, TEXT_DIM)`, `cond`: the references' tokens `(R, CHANNELS)` with their positions.
    pub fn prepare(&self, image_ids: &[[f32; 4]], cond: Option<(&Tensor, &[[f32; 4]])>, text: &Tensor) -> Result<Prepared> {
        let dt = self.dtype;
        let t = self.context.forward(&text.to_dtype(dt)?)?.to_dtype(DType::F32)?;
        let tl = t.dim(0)?;
        let text_ids: Vec<[f32; 4]> = (0..tl).map(|i| [0.0, 0.0, 0.0, i as f32]).collect();
        let (c, rope_cond) = match cond {
            Some((c, ids)) => (Some(self.x_cond.forward(&c.to_dtype(dt)?)?.to_dtype(DType::F32)?), Some(Rope::new(ids, &self.dev)?)),
            None => (None, None),
        };
        let mut all: Vec<[f32; 4]> = image_ids.to_vec();
        if let Some((_, ids)) = cond {
            all.extend_from_slice(ids);
        }
        all.extend_from_slice(&text_ids);
        let rope_single = Rope::new(&all, &self.dev)?;
        Ok(Prepared {
            cond: c,
            text: t,
            image_len: image_ids.len(),
            rope_image: rope_single.rows(0, image_ids.len())?,
            rope_cond,
            rope_text: rope_single.rows(all.len() - tl, tl)?,
            rope_single,
        })
    }

    /// The velocity `(n, CHANNELS)` f32 at time `t` for the picture's tokens `x (n, CHANNELS)`;
    /// `target` the time a flow map jumps to.
    pub fn forward(&self, x: &Tensor, prep: &Prepared, t: f32, target: Option<f32>) -> Result<Tensor> {
        let dt = self.dtype;
        let n = prep.image_len;
        let (vi, vc) = self.vectors(t, target)?;
        let early_i = modulation(&self.early[0], &vi, dt)?;
        let early_c = modulation(&self.early[1], &vc, dt)?;
        let early_t = modulation(&self.early[2], &vc, dt)?;

        let mut img = self.x_image.forward(&x.to_dtype(dt)?)?.to_dtype(DType::F32)?;
        let mut cond = prep.cond.clone();
        let mut text = prep.text.clone();
        let tl = text.dim(0)?;
        for d in 0..DEPTH {
            img = self.image_blocks[d].forward(&img, &prep.rope_image, &[(0, n, &early_i)], dt)?;
            if let (Some(c), Some(r)) = (cond.as_ref(), prep.rope_cond.as_ref()) {
                let rl = c.dim(0)?;
                cond = Some(self.cond_blocks[d].forward(c, r, &[(0, rl, &early_c)], dt)?);
            }
            text = self.text_blocks[d].forward(&text, &prep.rope_text, &[(0, tl, &early_t)], dt)?;
        }

        // One stream over [image, references, text], each kind with its own modulation.
        let si = modulation(&self.single_mod[0], &vi, dt)?;
        let sc = modulation(&self.single_mod[1], &vc, dt)?;
        let st = modulation(&self.single_mod[2], &vc, dt)?;
        let mut parts = vec![img];
        let mut spans: Vec<(usize, usize, &Modulation)> = vec![(0, n, &si)];
        let mut start = n;
        if let Some(c) = cond {
            let rl = c.dim(0)?;
            spans.push((start, rl, &sc));
            start += rl;
            parts.push(c);
        }
        spans.push((start, tl, &st));
        parts.push(text);
        let mut h = Tensor::cat(&parts, 0)?;
        for b in &self.single {
            h = b.forward(&h, &prep.rope_single, &spans, dt)?;
        }

        // The reference's LastLayer: shift first, then scale.
        let mo = self.final_mod.forward(&candle_nn::ops::silu(&vi.to_dtype(dt)?)?)?.to_dtype(DType::F32)?;
        let m = Modulation { shift: mo.narrow(1, 0, HIDDEN)?, scale: mo.narrow(1, HIDDEN, HIDDEN)?, gate: mo.narrow(1, 0, HIDDEN)? };
        let y = modulate(&h.narrow(0, 0, n)?, &m)?.to_dtype(dt)?;
        Ok(self.final_linear.forward(&y)?.to_dtype(DType::F32)?)
    }
}

/// Whether a LoRA module name is one of Klein's block linears (the reference's or diffusers'
/// names).
fn known_module(m: &str) -> bool {
    let groups: [(&str, usize); 8] = [
        ("content_mode_blocks.image.", DEPTH),
        ("content_mode_blocks.image_cond.", DEPTH),
        ("txt_mode_blocks.", DEPTH),
        ("single_blocks.", DEPTH_SINGLE),
        ("mode_blocks.image.", DEPTH),
        ("mode_blocks.image_cond.", DEPTH),
        ("single_transformer_blocks.", DEPTH_SINGLE),
        ("txt_mode_blocks.", DEPTH),
    ];
    for (p, count) in groups {
        let Some(rest) = m.strip_prefix(p) else { continue };
        let Some((idx, tail)) = rest.split_once('.') else { continue };
        if idx.parse::<usize>().is_ok_and(|i| i < count)
            && matches!(
                tail,
                "qkv"
                    | "q_proj"
                    | "k_proj"
                    | "v_proj"
                    | "attn_out"
                    | "mlp_in"
                    | "mlp_out"
                    | "attn.to_q"
                    | "attn.to_k"
                    | "attn.to_v"
                    | "attn.to_out"
                    | "attn.to_out.0"
                    | "ff.linear_in"
                    | "ff.linear_out"
            )
        {
            return true;
        }
    }
    false
}

/// `(t, row, column, layer)` of an `h × w` latent grid: rows and columns at half scale (the
/// reference's image RoPE scale), `layer` 0 for the picture being made and `-(1 + i)` for
/// reference `i`.
pub fn grid_ids(h: usize, w: usize, layer: f32) -> Vec<[f32; 4]> {
    (0..h).flat_map(|y| (0..w).map(move |x| [0.0, y as f32 * 0.5, x as f32 * 0.5, layer])).collect()
}

/// A latent `(1, CHANNELS, h, w)` as tokens `(h·w, CHANNELS)`, row by row.
pub fn to_tokens(z: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = z.dims4()?;
    anyhow::ensure!(b == 1, "a batch of {b} latents");
    Ok(z.reshape((c, h * w))?.t()?.contiguous()?)
}

/// Tokens `(h·w, CHANNELS)` back to a latent `(1, CHANNELS, h, w)`.
pub fn from_tokens(x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let c = x.dim(1)?;
    Ok(x.t()?.contiguous()?.reshape((1, c, h, w))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
    }

    #[test]
    fn lora_names_cover_diffusers() {
        let n = lora_names("content_mode_blocks.image.0.attn_out");
        assert!(n.contains(&"mode_blocks.image.0.attn.to_out.0".to_string()));
        assert!(lora_names("content_mode_blocks.image_cond.4.q_proj").contains(&"mode_blocks.image_cond.4.attn.to_q".to_string()));
        assert!(lora_names("single_blocks.27.mlp_in").contains(&"single_transformer_blocks.27.ff.linear_in".to_string()));
        assert!(known_module("single_blocks.27.mlp_in") && known_module("mode_blocks.image.0.attn.to_out.0"));
        assert!(!known_module("single_blocks.28.mlp_in") && !known_module("layers.0.proj.o"));
    }

    #[test]
    fn rope_pairs_are_adjacent() -> Result<()> {
        // Position (0, 0, 0, 1) turns only the layer axis: its first pair by 1 radian.
        let r = Rope::new(&[[0.0, 0.0, 0.0, 1.0]], &Device::Cpu)?;
        let mut x = vec![0f32; HEAD_DIM];
        x[96] = 1.0; // the layer axis' first pair, its first half
        let x = Tensor::from_vec(x, (1, 1, 1, HEAD_DIM), &Device::Cpu)?;
        let y: Vec<f32> = r.apply(&x)?.flatten_all()?.to_vec1()?;
        assert!((y[96] - 1f32.cos()).abs() < 1e-5 && (y[97] - 1f32.sin()).abs() < 1e-5);
        assert!(y.iter().enumerate().all(|(i, v)| i == 96 || i == 97 || v.abs() < 1e-6));
        Ok(())
    }
}
