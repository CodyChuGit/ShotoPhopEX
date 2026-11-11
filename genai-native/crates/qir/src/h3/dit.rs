//! The MiniMax H3 DiT (ComfyUI `comfy/ldm/minimax/model.py`, `MiniMaxH3Model`): one packed
//! sequence `[text | reference pictures | audio | target frame]` through a dense single-stream
//! transformer with per-segment AdaLN rows (timestep x modality), 3-axis split-half RoPE on the
//! first 96 of 128 head dims, and SwiGLU MLPs.
//!
//! P-H3-Image (PrunaAI) is this AV model used for a still: one video latent frame and two audio
//! latents. MiniMax-H3-Image (no audio stream, a PDD head bank) loads through the same code.
//!
//! The weights come from Comfy-Org/PrunaAI's repacks as they are ([`Weights`]): the block
//! linears as ConvRot INT8 on CUDA (dense elsewhere), everything else bf16/f32.

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};

use crate::moss::weights::{QLinear, Weights};
use crate::Result;

/// Reference rows run at this timestep (`VISUAL_COND_TIMESTEP`).
pub const VISUAL_COND_TIMESTEP: f32 = 0.999;
/// AdaLN modality tags (`SEG_TAG`): rows are `timestep_index * 3 + tag`.
pub const TAG_VISUAL: usize = 0;
pub const TAG_TEXT: usize = 1;
pub const TAG_AUDIO: usize = 2;
/// The video latent's channels and the DiT's 2x2 patch.
pub const LATENT_CHANNELS: usize = 24;
pub const AUDIO_CHANNELS: usize = 32;

/// The model's shape, read from the checkpoint (as ComfyUI's `model_detection` does).
#[derive(Clone, Debug, PartialEq)]
pub struct H3Config {
    pub hidden: usize,
    pub blocks: usize,
    pub refiner_blocks: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub text_dim: usize,
    /// `adaln_t_table` rows (curve-form checkpoints), `None` for a time embedder.
    pub curve_grid: Option<usize>,
    pub t_dim: usize,
    pub rope_len: usize,
    /// Output heads per stream (a PDD bank when > 1).
    pub pdd_steps: usize,
    pub audio: bool,
    pub eps: f64,
}

impl H3Config {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let count = |prefix: &str| -> usize { (0..).take_while(|i| w.contains(&format!("{prefix}{i}.norm1.weight"))).count() };
        let shape = |name: &str| -> Result<Vec<usize>> { Ok(w.shape(name)?) };
        let hidden = shape("video_patch_proj.weight")?[0];
        let head_dim = shape("blocks.0.attn.q_norm.weight")?[0];
        let qkv = w.out_features("blocks.0.attn.qkv_proj")?;
        let ffn = w.out_features("blocks.0.mlp.fc1")? / 2;
        let text_dim = shape("condition_proj.weight")?[1];
        let (curve_grid, t_dim) = if w.contains("adaln_t_table") {
            let s = shape("adaln_t_table")?;
            (Some(s[0]), s[1])
        } else {
            (None, shape("time_embedder.proj_out.weight")?[0])
        };
        let video_dim = LATENT_CHANNELS * 4;
        let out_rows = shape("final_layer.video_out.weight")?[0];
        anyhow::ensure!(out_rows % video_dim == 0, "final_layer.video_out has {out_rows} rows (not a multiple of {video_dim})");
        Ok(Self {
            hidden,
            blocks: count("blocks."),
            refiner_blocks: count("token_refiner.blocks."),
            heads: qkv / (3 * head_dim),
            head_dim,
            ffn,
            text_dim,
            curve_grid,
            t_dim,
            rope_len: shape("rope.inv_freq")?[0],
            pdd_steps: out_rows / video_dim,
            audio: w.contains("audio_patch_proj.weight"),
            eps: 1e-5,
        })
    }

    /// The RoPE'd prefix of each head (`3 axes x rope_len` pairs).
    pub fn rot_dim(&self) -> usize {
        6 * self.rope_len
    }
}

/// RMSNorm over the last dim in f32, rounded to the input dtype, times the weight.
struct Rms {
    weight: Tensor,
    eps: f64,
}

impl Rms {
    fn load(w: &Weights, name: &str, eps: f64) -> Result<Self> {
        Ok(Self { weight: w.get(name)?, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x32 = x.to_dtype(DType::F32)?;
        let y = x32.broadcast_mul(&(x32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?.recip()?)?;
        Ok(y.to_dtype(x.dtype())?.broadcast_mul(&self.weight.to_dtype(x.dtype())?)?)
    }
}

/// Split-half RoPE on the first `cos.dim(1) * 2` channels of `x (S, H, D)`: pair `i` is
/// channels `i` and `i + half`; the rest pass through (comfy-kitchen's `rot_dim`).
pub(crate) fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = cos.dim(1)?;
    let rot = 2 * half;
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let a = xf.narrow(2, 0, half)?;
    let b = xf.narrow(2, half, half)?;
    let c = cos.unsqueeze(1)?;
    let s = sin.unsqueeze(1)?;
    let ra = (a.broadcast_mul(&c)? - b.broadcast_mul(&s)?)?;
    let rb = (a.broadcast_mul(&s)? + b.broadcast_mul(&c)?)?;
    let mut parts = vec![ra, rb];
    if d > rot {
        parts.push(xf.narrow(2, rot, d - rot)?);
    }
    Ok(Tensor::cat(&parts, 2)?.to_dtype(dt)?)
}

/// Bidirectional attention over `q`, `k`, `v` `(S, H, D)`; `(S, H·D)`.
pub(crate) fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
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

/// `fc2(silu(gate) * up)` for the merged `fc1` output `(S, 2F)` (gate first, ComfyUI's
/// `swiglu`), fused into the INT8 quantization on CUDA.
pub(crate) fn swiglu_down(fc2: &QLinear, gu: &Tensor) -> Result<Tensor> {
    let f = gu.dim(1)? / 2;
    let gate = gu.narrow(1, 0, f)?;
    let up = gu.narrow(1, f, f)?;
    #[cfg(feature = "convrot")]
    if let QLinear::Convrot(l, None) = fc2 {
        return l.forward_swiglu(&gate, &up, crate::convrot::EpilogueOut::Bf16);
    }
    let h = (candle_nn::ops::silu(&gate)? * up)?;
    fc2.forward(&h)
}

struct Attn {
    qkv: QLinear,
    out: QLinear,
    q_norm: Rms,
    k_norm: Rms,
}

impl Attn {
    fn load(w: &Weights, p: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            qkv: w.linear(&format!("{p}.qkv_proj"))?,
            out: w.linear(&format!("{p}.out_proj"))?,
            q_norm: Rms::load(w, &format!("{p}.q_norm.weight"), eps)?,
            k_norm: Rms::load(w, &format!("{p}.k_norm.weight"), eps)?,
        })
    }

    fn forward(&self, x: &Tensor, heads: usize, head_dim: usize, rope_cs: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let s = x.dim(0)?;
        let inner = heads * head_dim;
        let qkv = self.qkv.forward(x)?;
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(1, i * inner, inner)?.reshape((s, heads, head_dim))?) };
        let mut q = self.q_norm.forward(&part(0)?)?;
        let mut k = self.k_norm.forward(&part(1)?)?;
        if let Some((c, sn)) = rope_cs {
            q = rope(&q, c, sn)?;
            k = rope(&k, c, sn)?;
        }
        let a = attention(&q.contiguous()?, &k.contiguous()?, &part(2)?.contiguous()?)?;
        self.out.forward(&a)
    }
}

struct Mlp {
    fc1: QLinear,
    fc2: QLinear,
}

impl Mlp {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self { fc1: w.linear(&format!("{p}.fc1"))?, fc2: w.linear(&format!("{p}.fc2"))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        swiglu_down(&self.fc2, &self.fc1.forward(x)?)
    }
}

/// A text-refiner block: plain pre-norm attention and MLP (no AdaLN, no RoPE).
struct RefinerBlock {
    norm1: Rms,
    norm2: Rms,
    attn: Attn,
    mlp: Mlp,
}

/// One AdaLN projection `t_emb (k) -> expand x hidden x modalities`, kept in f32.
struct Adaln {
    weight: Tensor, // (expand*hidden*modalities, k) f32
    bias: Tensor,
    silu: bool,
    expand: usize,
    modalities: usize,
}

impl Adaln {
    fn load(w: &Weights, p: &str, silu: bool, expand: usize, modalities: usize) -> Result<Self> {
        Ok(Self {
            weight: w.get_as(&format!("{p}.linear.weight"), DType::F32)?,
            bias: w.get_as(&format!("{p}.linear.bias"), DType::F32)?,
            silu,
            expand,
            modalities,
        })
    }

    /// `t_emb (M, k)` f32 -> `expand` tensors `(M*modalities, hidden)` in `dtype`.
    fn forward(&self, t_emb: &Tensor, dtype: DType) -> Result<Vec<Tensor>> {
        let x = if self.silu { candle_nn::ops::silu(t_emb)? } else { t_emb.clone() };
        let y = x.matmul(&self.weight.t()?)?.broadcast_add(&self.bias)?;
        let m = y.dim(0)?;
        let width = y.dim(1)? / self.modalities;
        let y = y.reshape((m * self.modalities, width))?;
        let hidden = width / self.expand;
        (0..self.expand).map(|j| Ok(y.narrow(1, j * hidden, hidden)?.to_dtype(dtype)?)).collect()
    }
}

struct Block {
    norm1: Rms,
    norm2: Rms,
    attn: Attn,
    mlp: Mlp,
    adaln: Adaln,
}

/// A contiguous run of rows sharing one AdaLN row (`timestep_index * 3 + tag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModSeg {
    pub start: usize,
    pub len: usize,
    pub row: usize,
}

/// `x[seg] * (1 + scale[row]) + shift[row]` per segment.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor, segs: &[ModSeg]) -> Result<Tensor> {
    let parts: Vec<Tensor> = segs
        .iter()
        .map(|g| -> Result<Tensor> {
            let xs = x.narrow(0, g.start, g.len)?;
            let sc = scale.i(g.row)?.unsqueeze(0)?;
            let sh = shift.i(g.row)?.unsqueeze(0)?;
            Ok((xs.broadcast_mul(&(sc + 1.0)?)?.broadcast_add(&sh))?)
        })
        .collect::<Result<_>>()?;
    Ok(Tensor::cat(&parts, 0)?)
}

/// `x + gate[row] * y` per segment.
fn gated_add(x: &Tensor, gate: &Tensor, y: &Tensor, segs: &[ModSeg]) -> Result<Tensor> {
    let parts: Vec<Tensor> = segs
        .iter()
        .map(|g| -> Result<Tensor> {
            let ys = y.narrow(0, g.start, g.len)?;
            let gt = gate.i(g.row)?.unsqueeze(0)?;
            Ok((x.narrow(0, g.start, g.len)? + ys.broadcast_mul(&gt)?)?)
        })
        .collect::<Result<_>>()?;
    Ok(Tensor::cat(&parts, 0)?)
}

/// Everything the blocks need for one forward, made once per sampling run where it can be.
pub struct Prepared {
    /// Refined text rows `(T, hidden)`.
    pub text: Tensor,
    /// RoPE `cos`/`sin` `(S, 3 * rope_len)` f32.
    pub cos: Tensor,
    pub sin: Tensor,
    pub layout: crate::h3::layout::Layout,
}

pub struct H3Dit {
    cfg: H3Config,
    video_patch_proj: candle_nn::Linear, // f32
    audio_patch_proj: Option<candle_nn::Linear>,
    condition_proj: QLinear,
    t_table: Option<Tensor>, // (grid, k) f32
    time_embedder: Option<(candle_nn::Linear, candle_nn::Linear)>,
    inv_freq: Vec<f32>,
    refiner: Vec<RefinerBlock>,
    refiner_norm: Rms,
    blocks: Vec<Block>,
    final_norm: Rms,
    final_adaln: Adaln,
    video_out: candle_nn::Linear, // f32
    audio_out: Option<candle_nn::Linear>,
    /// The PDD bank's sigma grid (decreasing, `pdd_steps + 1` points).
    pdd_grid: Vec<f32>,
    dtype: DType,
    dev: Device,
}

fn f32_linear(w: &Weights, p: &str) -> Result<candle_nn::Linear> {
    Ok(candle_nn::Linear::new(w.get_as(&format!("{p}.weight"), DType::F32)?, Some(w.get_as(&format!("{p}.bias"), DType::F32)?)))
}

impl H3Dit {
    pub fn load(w: &Weights) -> Result<Self> {
        let cfg = H3Config::from_weights(w)?;
        let eps = cfg.eps;
        let curve = cfg.curve_grid.is_some();
        let refiner = (0..cfg.refiner_blocks)
            .map(|i| -> Result<RefinerBlock> {
                let p = format!("token_refiner.blocks.{i}");
                Ok(RefinerBlock {
                    norm1: Rms::load(w, &format!("{p}.norm1.weight"), eps)?,
                    norm2: Rms::load(w, &format!("{p}.norm2.weight"), eps)?,
                    attn: Attn::load(w, &format!("{p}.attn"), eps)?,
                    mlp: Mlp::load(w, &format!("{p}.mlp"))?,
                })
            })
            .collect::<Result<_>>()?;
        let mut blocks = Vec::with_capacity(cfg.blocks);
        for i in 0..cfg.blocks {
            let p = format!("blocks.{i}");
            blocks.push(Block {
                norm1: Rms::load(w, &format!("{p}.norm1.weight"), eps)?,
                norm2: Rms::load(w, &format!("{p}.norm2.weight"), eps)?,
                attn: Attn::load(w, &format!("{p}.attn"), eps)?,
                mlp: Mlp::load(w, &format!("{p}.mlp"))?,
                adaln: Adaln::load(w, &format!("{p}.adaln_proj"), !curve, 6, 3)?,
            });
        }
        let inv_freq = w.get_as("rope.inv_freq", DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let time_embedder = if curve {
            None
        } else {
            Some((f32_linear(w, "time_embedder.proj_in")?, f32_linear(w, "time_embedder.proj_out")?))
        };
        let pdd_grid = if cfg.pdd_steps > 1 { pdd_sigmas(w, cfg.pdd_steps)? } else { vec![1.0, 0.0] };
        Ok(Self {
            pdd_grid,
            video_patch_proj: f32_linear(w, "video_patch_proj")?,
            audio_patch_proj: if cfg.audio { Some(f32_linear(w, "audio_patch_proj")?) } else { None },
            condition_proj: w.linear("condition_proj")?,
            t_table: if curve { Some(w.get_as("adaln_t_table", DType::F32)?) } else { None },
            time_embedder,
            inv_freq,
            refiner,
            refiner_norm: Rms::load(w, "token_refiner.final_norm.weight", eps)?,
            blocks,
            final_norm: Rms::load(w, "final_layer.norm.weight", eps)?,
            final_adaln: Adaln::load(w, "final_layer.adaln_proj", !curve, 2, 1)?,
            video_out: f32_linear(w, "final_layer.video_out")?,
            audio_out: if cfg.audio { Some(f32_linear(w, "final_layer.audio_out")?) } else { None },
            dtype: w.dtype(),
            dev: w.device().clone(),
            cfg,
        })
    }

    pub fn config(&self) -> &H3Config {
        &self.cfg
    }

    /// `condition_proj` + the token refiner on the text encoder's states `(T, text_dim)`.
    pub fn refine_text(&self, states: &Tensor) -> Result<Tensor> {
        let mut x = self.condition_proj.forward(&states.to_dtype(self.dtype)?)?;
        for b in &self.refiner {
            x = (b.attn.forward(&b.norm1.forward(&x)?, self.cfg.heads, self.cfg.head_dim, None)? + &x)?;
            x = (b.mlp.forward(&b.norm2.forward(&x)?)? + &x)?;
        }
        self.refiner_norm.forward(&x)
    }

    /// RoPE tables for `layout`: per row, angles `pos_axis * inv_freq` for t, h, w.
    pub fn rope_tables(&self, layout: &crate::h3::layout::Layout) -> Result<(Tensor, Tensor)> {
        let n = self.inv_freq.len();
        let s = layout.positions.len();
        let mut cos = Vec::with_capacity(s * 3 * n);
        let mut sin = Vec::with_capacity(s * 3 * n);
        for p in &layout.positions {
            for axis in p.iter() {
                for f in &self.inv_freq {
                    // positions are f64 in ComfyUI, cast to f32 before the product
                    let a = (*axis as f32) * f;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
        Ok((Tensor::from_vec(cos, (s, 3 * n), &self.dev)?, Tensor::from_vec(sin, (s, 3 * n), &self.dev)?))
    }

    pub fn prepare(&self, text_states: &Tensor, layout: crate::h3::layout::Layout) -> Result<Prepared> {
        anyhow::ensure!(text_states.dim(0)? == layout.text_len, "text rows {} vs layout {}", text_states.dim(0)?, layout.text_len);
        let text = self.refine_text(text_states)?;
        let (cos, sin) = self.rope_tables(&layout)?;
        Ok(Prepared { text, cos, sin, layout })
    }

    /// `t_emb (M, k)` f32 for the timesteps `ts`.
    fn time_embedding(&self, ts: &[f32]) -> Result<Tensor> {
        if let Some(table) = &self.t_table {
            let grid = table.dim(0)?;
            let rows: Vec<Tensor> = ts
                .iter()
                .map(|&t| -> Result<Tensor> {
                    let pos = t.clamp(0.0, 1.0) * (grid - 1) as f32;
                    let i0 = (pos.floor() as usize).min(grid - 2);
                    let frac = pos - i0 as f32;
                    let a = table.i(i0)?;
                    let b = table.i(i0 + 1)?;
                    Ok((&a + ((b - &a)? * f64::from(frac))?)?)
                })
                .collect::<Result<_>>()?;
            return Ok(Tensor::stack(&rows, 0)?);
        }
        let (pin, pout) = self.time_embedder.as_ref().expect("a time embedder without a curve table");
        let half = pin.weight().dim(1)? / 2;
        let freqs: Vec<f32> = (0..half).map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32).collect();
        let mut emb = Vec::with_capacity(ts.len() * half * 2);
        for &t in ts {
            emb.extend(freqs.iter().map(|f| (t * f).cos()));
            emb.extend(freqs.iter().map(|f| (t * f).sin()));
        }
        let e = Tensor::from_vec(emb, (ts.len(), half * 2), &self.dev)?;
        Ok(pout.forward(&candle_nn::ops::silu(&pin.forward(&e)?)?)?)
    }

    /// One evaluation at `sigma` (the video stream's; audio follows at shift 1).
    ///
    /// `video (C, 1, H, W)` and `audio (32, 2, T)` are the noisy latents (f32), `refs` the
    /// reference latents `(C, 1, h, w)` and `cond_seed` the sampling seed for their noise
    /// augmentation (ComfyUI `_cond_video_rows`). Returns the velocities `(video, audio)` the way the
    /// model reports them (ComfyUI's `_forward`: the negated head output), f32.
    pub fn forward(&self, p: &Prepared, video: &Tensor, audio: Option<&Tensor>, refs: &[Tensor], cond_seed: u64, sigma: f32, sample_sigmas: &[f32]) -> Result<(Tensor, Option<Tensor>)> {
        let lay = &p.layout;
        let t = 1.0 - sigma;
        let t_ref = t.max(VISUAL_COND_TIMESTEP);
        let mut unique = vec![t];
        if !refs.is_empty() && (t_ref - t).abs() > 0.0 {
            unique.push(t_ref);
        }
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup();
        let ti = |v: f32| unique.iter().position(|u| *u == v).unwrap();
        let segs = lay.mod_segments(ti(t), ti(t_ref));

        // embed + pack
        let (_, _, lat_h, lat_w) = video.dims4()?;
        let video_rows = patchify(video)?; // (h/2*w/2, 96)
        let mut cond_rows = Vec::new();
        for r in refs {
            // every reference restarts the same RNG stream: 0.999 * r + 0.001 * randn(seed)
            let r = patchify(r)?;
            let noise = crate::comfy::torch_randn(cond_seed, r.elem_count()).map_err(anyhow::Error::msg)?;
            let noise = Tensor::from_vec(noise, r.dims(), r.device())?;
            cond_rows.push(((r * 0.999f64)? + (noise * (1.0f64 - 0.999f64))?)?);
        }
        let mut all = cond_rows.clone();
        all.push(video_rows);
        let vemb = self.video_patch_proj.forward(&Tensor::cat(&all, 0)?)?.to_dtype(self.dtype)?;
        let n_ref_rows: usize = cond_rows.iter().map(|r| r.dim(0)).collect::<candle_core::Result<Vec<_>>>()?.iter().sum();
        let mut pieces = vec![p.text.clone()];
        if n_ref_rows > 0 {
            pieces.push(vemb.narrow(0, 0, n_ref_rows)?);
        }
        if let Some(a) = audio {
            let proj = self.audio_patch_proj.as_ref().context_audio()?;
            pieces.push(proj.forward(&pack_audio(a)?)?.to_dtype(self.dtype)?);
        }
        pieces.push(vemb.narrow(0, n_ref_rows, vemb.dim(0)? - n_ref_rows)?);
        let mut h = Tensor::cat(&pieces, 0)?;
        anyhow::ensure!(h.dim(0)? == lay.seq_len(), "packed rows {} vs layout {}", h.dim(0)?, lay.seq_len());

        let t_emb = self.time_embedding(&unique)?;
        let rope_cs = Some((&p.cos, &p.sin));
        for b in &self.blocks {
            let m = b.adaln.forward(&t_emb, self.dtype)?;
            let (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f) = (&m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
            let y = modulate(&b.norm1.forward(&h)?, shift_a, scale_a, &segs)?;
            let a = b.attn.forward(&y, self.cfg.heads, self.cfg.head_dim, rope_cs)?;
            h = gated_add(&h, gate_a, &a, &segs)?;
            let y = modulate(&b.norm2.forward(&h)?, shift_f, scale_f, &segs)?;
            let f = b.mlp.forward(&y)?;
            h = gated_add(&h, gate_f, &f, &segs)?;
        }

        // final layer: per target stream, its own timestep row (modality 1 of 1)
        let m = self.final_adaln.forward(&t_emb, DType::F32)?;
        let (shift, scale) = (&m[0], &m[1]);
        let row = ti(t);
        let head_in = |a: usize, n: usize| -> Result<Tensor> {
            let x = self.final_norm.forward(&h.narrow(0, a, n)?)?.to_dtype(DType::F32)?;
            Ok(x.broadcast_mul(&(scale.i(row)?.unsqueeze(0)? + 1.0)?)?.broadcast_add(&shift.i(row)?.unsqueeze(0)?)?)
        };
        let (va, vn) = lay.video_span();
        let v_rows = self.head(&self.video_out, &head_in(va, vn)?, sigma, sample_sigmas)?;
        let v = unpatchify(&v_rows, lat_h, lat_w)?.neg()?;
        let a_out = match (audio, &self.audio_out) {
            (Some(_), Some(head)) => {
                let (aa, an) = lay.audio_span();
                Some(unpack_audio(&self.head(head, &head_in(aa, an)?, sigma, sample_sigmas)?)?.neg()?)
            }
            _ => None,
        };
        Ok((v, a_out))
    }

    /// A plain head, or the PDD bank's dt-weighted mix for the step from `sigma` to the next
    /// sample sigma (ComfyUI `_pdd_head`).
    fn head(&self, head: &candle_nn::Linear, x: &Tensor, sigma: f32, sample_sigmas: &[f32]) -> Result<Tensor> {
        let n = self.cfg.pdd_steps;
        if n == 1 {
            return Ok(head.forward(x)?);
        }
        let grid = &self.pdd_grid;
        anyhow::ensure!(grid.len() == n + 1, "PDD grid has {} points for {n} heads", grid.len());
        let i = sample_sigmas.iter().enumerate().min_by(|a, b| (a.1 - sigma).abs().partial_cmp(&(b.1 - sigma).abs()).unwrap()).map(|(i, _)| i).unwrap_or(0);
        let next = sample_sigmas[(i + 1).min(sample_sigmas.len() - 1)];
        let nearest = |s: f32| (0..=n).min_by(|a, b| (grid[*a] - s).abs().partial_cmp(&(grid[*b] - s).abs()).unwrap()).unwrap();
        let start = nearest(sigma).min(n - 1);
        let stop = nearest(next).max(start + 1);
        let dt: Vec<f32> = (start..stop).map(|k| grid[k] - grid[k + 1]).collect();
        let total: f32 = dt.iter().sum();
        let out = head.weight().dim(0)? / n;
        let inp = head.weight().dim(1)?;
        let rows = head.weight().reshape((n, out, inp))?;
        let brows = head.bias().context_bias()?.reshape((n, out))?;
        let first = start.max(1);
        let mut w = rows.i(0)?;
        let mut b = brows.i(0)?;
        for k in first..stop {
            let wk = f64::from(dt[k - start] / total);
            w = (w + (rows.i(k)? * wk)?)?;
            b = (b + (brows.i(k)? * wk)?)?;
        }
        Ok(x.matmul(&w.t()?)?.broadcast_add(&b)?)
    }
}

/// MiniMax-H3-Image's `config.transformer.pdd_sigmas` (the file's metadata).
fn pdd_sigmas(w: &Weights, steps: usize) -> Result<Vec<f32>> {
    let raw = w.file_metadata("config").ok_or_else(|| anyhow::anyhow!("a {steps}-head PDD checkpoint without config metadata"))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let s = v["transformer"]["pdd_sigmas"].as_array().ok_or_else(|| anyhow::anyhow!("config.transformer.pdd_sigmas missing"))?;
    let grid: Vec<f32> = s.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
    anyhow::ensure!(grid.len() == steps + 1, "pdd_sigmas has {} points for {steps} heads", grid.len());
    Ok(grid)
}

trait OptionExt<T> {
    fn context_audio(self) -> Result<T>;
    fn context_bias(self) -> Result<T>;
}

impl<T> OptionExt<T> for Option<T> {
    fn context_audio(self) -> Result<T> {
        self.ok_or_else(|| anyhow::anyhow!("an audio latent for a model without an audio stream"))
    }
    fn context_bias(self) -> Result<T> {
        self.ok_or_else(|| anyhow::anyhow!("an output head without a bias"))
    }
}

/// `(C, 1, H, W)` -> `(H/2 * W/2, C*2*2)` rows, channel-major then the 2x2 patch
/// (ComfyUI `patchify_video` with patch (1, 2, 2)).
pub fn patchify(x: &Tensor) -> Result<Tensor> {
    let (c, t, h, w) = x.dims4()?;
    anyhow::ensure!(t == 1 && h % 2 == 0 && w % 2 == 0, "patchify takes one frame with even sides, got {:?}", x.dims());
    let x = x.to_dtype(DType::F32)?.reshape((c, h / 2, 2, w / 2, 2))?; // c, h, p, w, q
    let x = x.permute((1, 3, 0, 2, 4))?; // h, w, c, p, q
    Ok(x.contiguous()?.reshape((h / 2 * (w / 2), c * 4))?)
}

/// The inverse of [`patchify`]: `(h*w, C*4)` -> `(C, 1, 2h, 2w)`.
pub fn unpatchify(rows: &Tensor, lat_h: usize, lat_w: usize) -> Result<Tensor> {
    let c = rows.dim(1)? / 4;
    let x = rows.reshape((lat_h / 2, lat_w / 2, c, 2, 2))?; // h, w, c, p, q
    let x = x.permute((2, 0, 3, 1, 4))?; // c, h, p, w, q
    Ok(x.contiguous()?.reshape((c, 1, lat_h, lat_w))?)
}

/// `(32, 2, T)` -> `(2T, 32)` channel-major rows (ComfyUI `pack_audio`).
pub fn pack_audio(a: &Tensor) -> Result<Tensor> {
    let (c, ch, t) = a.dims3()?;
    Ok(a.to_dtype(DType::F32)?.permute((1, 2, 0))?.contiguous()?.reshape((ch * t, c))?)
}

/// The inverse of [`pack_audio`].
pub fn unpack_audio(rows: &Tensor) -> Result<Tensor> {
    let (n, c) = rows.dims2()?;
    Ok(rows.reshape((2, n / 2, c))?.permute((2, 0, 1))?.contiguous()?)
}
