//! Krea 2 (plan Phase 4): what the model needs from the Qwen3-VL-4B encoder, as ComfyUI's
//! `krea2` text encoder hands it over, and (below) the single-stream DiT itself.
//!
//! The conditioning is a stack of twelve hidden states of the 4B language model
//! (`hidden_states[2, 5, 8, …, 35]`, transformers' numbering: `0` is the embedding), kept for
//! the tokens after the chat template's prefix (the system turn and `<|im_start|>user\n`),
//! shaped `(tokens, 12, 2560)`; the DiT's `txtfusion` adapter fuses the twelve per token.

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use crate::fp8::{Fp8Linear, Fp8Plan};
use crate::lora::{with_lora, LoraSlot, LoraStack};
use crate::model::text_encoder::QwenTextEncoder;
use crate::Result;

/// The hidden states Krea 2 taps (transformers' `hidden_states` indices).
pub const TAPS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];
/// The language model's width.
pub const TEXT_DIM: usize = 2560;
/// The chat template (ComfyUI's `KREA2_TEMPLATE`): the Qwen-Image system prompt.
pub const SYS_PROMPT: &str =
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";

const IM_START: u32 = 151_644;
const USER: u32 = 872;
const NEWLINE: u32 = 198;

/// The prompt in Krea 2's chat template.
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// The prompt with one picture in view, as the krea2_edit LoRAs were trained (the
/// comfyui-krea2edit `Krea2EditGroundedEncode` template): the same system turn, the picture's
/// vision block opening the user turn, then the instruction.
pub fn edit_template(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Stock EditPlus overrides the encoder's default template and numbers each picture.
pub fn style_template(prompt: &str, pictures: usize) -> String {
    let labels: String = (1..=pictures).map(|i| format!("Picture {i}: <|vision_start|><|image_pad|><|vision_end|>")).collect();
    format!("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{labels}{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// [`condition`] with pictures in view: their vision rows spliced in for `<|image_pad|>`, the
/// twelve taps kept from the end of the template's prefix (the pictures come after it, so the
/// prefix is the same rows as in `ids`), the picture rows kept as ComfyUI keeps them.
pub fn condition_with_images(te: &QwenTextEncoder, ids: &[u32], images: &[&crate::model::vision::VisionFeatures]) -> Result<Tensor> {
    let (taps, spans) = te.forward_with_images_taps(ids, images, &TAPS)?;
    let end = template_end(ids);
    anyhow::ensure!(spans.iter().all(|(start, _)| *start >= end), "a picture inside the template's prefix");
    let total = taps.first().map(|t| t.dim(1)).transpose()?.unwrap_or(0);
    let keep = total.saturating_sub(end);
    anyhow::ensure!(keep > 0, "the prompt has no tokens after the template");
    let rows: Vec<Tensor> = taps.iter().map(|t| t.narrow(1, end, keep)).collect::<candle_core::Result<_>>()?;
    Ok(Tensor::stack(&rows, 2)?.squeeze(0)?.contiguous()?)
}

/// How many leading tokens the encoder's output drops: up to the second `<|im_start|>`, and
/// through the `user\n` that follows it (ComfyUI's `Krea2TEModel.encode_token_weights`).
pub fn template_end(ids: &[u32]) -> usize {
    let mut end = 0;
    let mut seen = 0;
    for (i, t) in ids.iter().enumerate() {
        if *t == IM_START && seen < 2 {
            end = i;
            seen += 1;
        }
    }
    if ids.len() > end + 3 && ids.get(end + 1) == Some(&USER) && ids.get(end + 2) == Some(&NEWLINE) {
        end += 3;
    }
    end
}

/// The conditioning `(tokens', 12, 2560)` for `ids` (the templated prompt's tokens): the
/// twelve tapped states with the template's prefix dropped.
pub fn condition(te: &QwenTextEncoder, ids: &[u32], dev: &candle_core::Device) -> Result<Tensor> {
    let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), dev)?;
    let taps = te.forward_taps(&input, &TAPS)?;
    let end = template_end(ids);
    let keep = ids.len().saturating_sub(end);
    anyhow::ensure!(keep > 0, "the prompt has no tokens after the template");
    let rows: Vec<Tensor> = taps.iter().map(|t| t.narrow(1, end, keep)).collect::<candle_core::Result<_>>()?;
    // (1, keep, 2560) × 12 → (keep, 12, 2560)
    let stacked = Tensor::stack(&rows, 2)?; // (1, keep, 12, 2560)
    Ok(stacked.squeeze(0)?.contiguous()?)
}

// ---------------------------------------------------------------------------------------------
// The single-stream DiT (ComfyUI's `comfy/ldm/krea2/model.py` and stable-diffusion.cpp's
// `krea2.hpp` describe the same network; this is a clean-room port of the architecture).
// ---------------------------------------------------------------------------------------------

/// Krea 2's shape (Turbo and Raw share it).
#[derive(Clone, Debug)]
pub struct Krea2Config {
    pub features: usize,
    pub timestep_dim: usize,
    pub text_dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub mlp_multiplier: usize,
    pub layers: usize,
    pub patch: usize,
    pub channels: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    pub text_kv_heads: usize,
    pub theta: f32,
    pub eps: f64,
}

impl Default for Krea2Config {
    fn default() -> Self {
        Self {
            features: 6144,
            timestep_dim: 256,
            text_dim: TEXT_DIM,
            heads: 48,
            kv_heads: 12,
            mlp_multiplier: 4,
            layers: 28,
            patch: 2,
            channels: 16,
            text_layers: TAPS.len(),
            text_heads: 20,
            text_kv_heads: 20,
            theta: 1000.0,
            eps: 1e-5,
        }
    }
}

impl Krea2Config {
    pub fn head_dim(&self) -> usize {
        self.features / self.heads
    }

    /// The SwiGLU width: `2·features/3 · multiplier`, rounded up to 128.
    pub fn mlp_dim(features: usize, multiplier: usize) -> usize {
        ((2 * features / 3) * multiplier).div_ceil(128) * 128
    }

    /// The RoPE pairs per axis (frame, height, width): the head's pairs less twelve sixteenths
    /// for the frame, six sixteenths each for height and width.
    pub fn axes(&self) -> [usize; 3] {
        let d = self.head_dim();
        let unit = d / 16;
        [d - 12 * unit, 6 * unit, 6 * unit]
    }
}

/// The linears ConvRot takes: the attention and SwiGLU projections of the blocks and of the
/// text-fusion adapter (`<prefix>` is the tensor name without `.weight`). The tails (`first`,
/// `tmlp`, `tproj`, `txtmlp`, `last`, the projector) stay bf16.
pub fn is_convrot_target(prefix: &str) -> bool {
    let under = prefix.starts_with("blocks.") || prefix.starts_with("txtfusion.");
    let name = prefix.rsplit('.').next().unwrap_or("");
    let kind = prefix.rsplit('.').nth(1).unwrap_or("");
    under && ((kind == "attn" && matches!(name, "wq" | "wk" | "wv" | "gate" | "wo")) || (kind == "mlp" && matches!(name, "gate" | "up" | "down")))
}

/// The policy tag a Krea 2 ConvRot cache entry carries (a change invalidates the entry).
pub fn convrot_policy_tag() -> String {
    format!("qir-krea2-convrot-v{};group={};blocks+txtfusion attn/mlp", crate::model::dit::PREQUANT_FORMAT, crate::model::rotation::GROUP)
}

/// A linear of the DiT: bf16, or ConvRot W8A8 INT8 when the rotation is given and the shape
/// allows (the 2.1 port's kernels, `convrot` feature): the block and text-fusion linears take
/// it so the 24 GB bf16 model fits beside the encoder; the small tails stay bf16. A
/// prequantized file (`weight_i8` + `col_scale`, the cache's) loads as it is.
enum KLinear {
    Full(Linear),
    #[cfg(feature = "convrot")]
    Convrot(crate::convrot::ConvRotLinear),
    /// The file's own e4m3 weight (ComfyUI's `fp8_scaled` repacks), `crate::fp8`.
    Fp8(Fp8Linear),
}

/// How the block and text-fusion linears are held: as the file has them, rotated to ConvRot
/// INT8, or (an `fp8_scaled` file) in its e4m3 bytes.
#[derive(Clone, Copy, Default)]
pub struct Quant<'a> {
    #[cfg_attr(not(feature = "convrot"), allow(dead_code))]
    rot: Option<&'a Tensor>,
    fp8: Option<&'a Fp8Plan>,
}

impl KLinear {
    fn load(in_c: usize, out_c: usize, bias: bool, q: Quant<'_>, vb: VarBuilder) -> Result<Self> {
        if q.fp8.is_some() && !bias {
            if let Some(l) = crate::fp8::load_linear(in_c, out_c, q.fp8, &vb)? {
                return Ok(Self::Fp8(l));
            }
        }
        #[cfg(feature = "convrot")]
        if let Some(r) = q.rot {
            if !bias && in_c.is_multiple_of(crate::model::rotation::GROUP) && out_c.is_multiple_of(8) {
                if vb.contains_tensor("weight_i8") {
                    let w_i8 = vb.get_unchecked_dtype("weight_i8", DType::U8)?;
                    let col_scale = vb.get_unchecked_dtype("col_scale", DType::F32)?;
                    return Ok(Self::Convrot(crate::convrot::ConvRotLinear::from_prequantized(w_i8, col_scale, r)?));
                }
                let w = vb.get((out_c, in_c), "weight")?;
                return Ok(Self::Convrot(crate::convrot::ConvRotLinear::from_weight(&w, r)?));
            }
        }
        if !vb.contains_tensor("weight") && vb.contains_tensor("weight_i8") {
            anyhow::bail!("{}: the file holds only a ConvRot INT8 weight for this layer (run with --convrot)", vb.prefix());
        }
        let _ = q;
        Ok(Self::Full(if bias { linear(in_c, out_c, vb)? } else { linear_no_bias(in_c, out_c, vb)? }))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Full(l) => Ok(l.forward(x)?),
            #[cfg(feature = "convrot")]
            Self::Convrot(c) => c.forward(x),
            Self::Fp8(l) => l.forward(x),
        }
    }
}

/// RMSNorm with the `(1 + scale)` weight (the scale is stored zero-centred), in f32.
struct RmsNorm1p {
    weight: Tensor, // (C,) f32, 1 + scale
    eps: f64,
}

impl RmsNorm1p {
    fn load(c: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(c, "scale")?.to_dtype(DType::F32)?;
        Ok(Self { weight: (scale + 1.0)?, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "fusednorm")]
        if fused(x) {
            return crate::krea2_fused::rms_mod(x, &self.weight, None, self.eps);
        }
        let dt = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let rms = (x32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        Ok(x32.broadcast_div(&rms)?.broadcast_mul(&self.weight)?.to_dtype(dt)?)
    }

    /// [`Self::forward`] then AdaLN: `· (1 + mods[scale]) + mods[shift]`, `mods (B, 1, 6F)`.
    fn modulated(&self, x: &Tensor, mods: &Tensor, scale: usize, shift: usize) -> Result<Tensor> {
        #[cfg(feature = "fusednorm")]
        if fused(x) {
            return crate::krea2_fused::rms_mod(x, &self.weight, Some((mods, scale, shift)), self.eps);
        }
        let f = self.weight.dim(0)?;
        let (sc, sh) = (mods.narrow(2, scale * f, f)?, mods.narrow(2, shift * f, f)?);
        Ok(self.forward(x)?.broadcast_mul(&(sc + 1.0)?)?.broadcast_add(&sh)?)
    }
}

/// Whether `x` takes the fused kernels (`crate::krea2_fused`): bf16 on CUDA, unless
/// `QIR_KREA2_FUSED=0` asks for the candle chains (to compare).
#[cfg(feature = "fusednorm")]
fn fused(x: &Tensor) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    x.dtype() == DType::BF16 && x.device().is_cuda() && *ON.get_or_init(|| std::env::var("QIR_KREA2_FUSED").map_or(true, |v| v.trim() != "0"))
}

/// `x + v · mods[chunk]`, the gated residual.
fn gate_add(x: &Tensor, v: &Tensor, mods: &Tensor, chunk: usize) -> Result<Tensor> {
    #[cfg(feature = "fusednorm")]
    if fused(x) {
        return crate::krea2_fused::gate_add(x, v, mods, chunk);
    }
    let f = x.dim(D::Minus1)?;
    Ok((x + v.broadcast_mul(&mods.narrow(2, chunk * f, f)?)?)?)
}

fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

/// `down(silu(gate(x)) · up(x))`.
struct SwiGlu {
    gate: KLinear,
    up: KLinear,
    down: KLinear,
    /// LoRA side paths (`crate::lora`), empty unless a request sets them.
    lgate: LoraSlot,
    lup: LoraSlot,
    ldown: LoraSlot,
}

impl SwiGlu {
    fn load(features: usize, multiplier: usize, q: Quant<'_>, vb: VarBuilder) -> Result<Self> {
        let m = Krea2Config::mlp_dim(features, multiplier);
        Ok(Self {
            gate: KLinear::load(features, m, false, q, vb.pp("gate"))?,
            up: KLinear::load(features, m, false, q, vb.pp("up"))?,
            down: KLinear::load(m, features, false, q, vb.pp("down"))?,
            lgate: LoraSlot::default(),
            lup: LoraSlot::default(),
            ldown: LoraSlot::default(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = with_lora(&self.lgate, self.gate.forward(x)?, x)?;
        let u = with_lora(&self.lup, self.up.forward(x)?, x)?;
        #[cfg(feature = "fusednorm")]
        let hin = if fused(&g) { crate::krea2_fused::swiglu(&g, &u)? } else { (candle_nn::ops::silu(&g)? * u)? };
        #[cfg(not(feature = "fusednorm"))]
        let hin = (candle_nn::ops::silu(&g)? * u)?;
        with_lora(&self.ldown, self.down.forward(&hin)?, &hin)
    }

    fn set_loras(&self, stack: &LoraStack, prefix: &str) -> Result<()> {
        self.lgate.set(stack.pairs_exact(&format!("{prefix}.gate"))?);
        self.lup.set(stack.pairs_exact(&format!("{prefix}.up"))?);
        self.ldown.set(stack.pairs_exact(&format!("{prefix}.down"))?);
        Ok(())
    }
}

/// A constant added to the attention scores (natural-log units, before the softmax) of the rows
/// from `rows_from` on, for the keys in `keys`: an edit's `ref_boost`, `ln(b)` on the target's
/// attention to the reference's tokens (the comfyui-krea2edit nodes' `_ref_attn_bias`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreBias {
    pub rows_from: usize,
    pub keys: (usize, usize),
    pub log_boost: f32,
}

/// Plain attention of `q` `(B,H,S,D)` over `k`, `v` `(B,H,S,D)`, scores in f32 in query chunks.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, bias: Option<ScoreBias>) -> Result<Tensor> {
    const CHUNK: usize = 1024;
    let s = q.dim(2)?;
    let skv = k.dim(2)?;
    let keys = k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
    let mut outs = Vec::with_capacity(s.div_ceil(CHUNK));
    let mut q0 = 0;
    while q0 < s {
        let n = CHUNK.min(s - q0);
        let qc = q.narrow(2, q0, n)?.to_dtype(DType::F32)?;
        let mut scores = (qc.matmul(&keys)? * scale)?;
        if let Some(b) = bias.filter(|b| q0 + n > b.rows_from) {
            let mut add = vec![0f32; n * skv];
            for r in b.rows_from.saturating_sub(q0)..n {
                for c in b.keys.0..b.keys.1.min(skv) {
                    add[r * skv + c] = b.log_boost;
                }
            }
            scores = scores.broadcast_add(&Tensor::from_vec(add, (1, 1, n, skv), q.device())?)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
        outs.push(attn.matmul(v)?);
        q0 += n;
    }
    Ok(Tensor::cat(&outs, 2)?)
}

/// GQA with per-head q/k RMSNorm, optional RoPE and a sigmoid gate on the output.
struct Attention {
    wq: KLinear,
    wk: KLinear,
    wv: KLinear,
    gate: KLinear,
    wo: KLinear,
    qnorm: RmsNorm1p,
    knorm: RmsNorm1p,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// LoRA side paths on the five linears, empty unless a request sets them.
    lq: LoraSlot,
    lk: LoraSlot,
    lv: LoraSlot,
    lgate: LoraSlot,
    lo: LoraSlot,
}

impl Attention {
    fn load(dim: usize, heads: usize, kv_heads: usize, eps: f64, q: Quant<'_>, vb: VarBuilder) -> Result<Self> {
        let hd = dim / heads;
        Ok(Self {
            wq: KLinear::load(dim, heads * hd, false, q, vb.pp("wq"))?,
            wk: KLinear::load(dim, kv_heads * hd, false, q, vb.pp("wk"))?,
            wv: KLinear::load(dim, kv_heads * hd, false, q, vb.pp("wv"))?,
            gate: KLinear::load(dim, dim, false, q, vb.pp("gate"))?,
            wo: KLinear::load(dim, dim, false, q, vb.pp("wo"))?,
            qnorm: RmsNorm1p::load(hd, eps, vb.pp("qknorm").pp("qnorm"))?,
            knorm: RmsNorm1p::load(hd, eps, vb.pp("qknorm").pp("knorm"))?,
            heads,
            kv_heads,
            head_dim: hd,
            lq: LoraSlot::default(),
            lk: LoraSlot::default(),
            lv: LoraSlot::default(),
            lgate: LoraSlot::default(),
            lo: LoraSlot::default(),
        })
    }

    fn set_loras(&self, stack: &LoraStack, prefix: &str) -> Result<()> {
        self.lq.set(stack.pairs_exact(&format!("{prefix}.wq"))?);
        self.lk.set(stack.pairs_exact(&format!("{prefix}.wk"))?);
        self.lv.set(stack.pairs_exact(&format!("{prefix}.wv"))?);
        self.lgate.set(stack.pairs_exact(&format!("{prefix}.gate"))?);
        self.lo.set(stack.pairs_exact(&format!("{prefix}.wo"))?);
        Ok(())
    }

    /// The sigmoid gate on the attention output, with its LoRA delta.
    fn gated(&self, out: Tensor, x: &Tensor) -> Result<Tensor> {
        let g = with_lora(&self.lgate, self.gate.forward(x)?, x)?;
        #[cfg(feature = "fusednorm")]
        let t = if fused(&g) { crate::krea2_fused::sig_mul(&out, &g)? } else { (out * candle_nn::ops::sigmoid(&g)?)? };
        #[cfg(not(feature = "fusednorm"))]
        let t = (out * candle_nn::ops::sigmoid(&g)?)?;
        with_lora(&self.lo, self.wo.forward(&t)?, &t)
    }

    /// `x` `(B,S,dim)`; `rope` the `(S, D/2)` f32 cos/sin tables (interleaved pairs) or none;
    /// `bias` on the scores (exact attention only: SageAttention is not taken with one).
    fn forward(&self, x: &Tensor, rope: Option<&(Tensor, Tensor)>, bias: Option<ScoreBias>) -> Result<Tensor> {
        #[cfg(feature = "flash")]
        let kbias = bias.map(|b| crate::flash::KeyBias { rows_from: b.rows_from, keys: b.keys, log_boost: b.log_boost });
        let (b, s, _) = x.dims3()?;
        let (h, kvh, d) = (self.heads, self.kv_heads, self.head_dim);
        let choice = attention_choice();
        // The fused route: per-head norm and RoPE in one pass over the BSHD projections, the
        // flash kernel reading them where they are, the gate fused into its multiply.
        #[cfg(all(feature = "fusednorm", feature = "flash"))]
        if choice == Attn::Flash && d == crate::flash::HEAD_DIM && fused(x) {
            let cs = match rope {
                Some((cos, sin)) => Some(Tensor::stack(&[cos, sin], 0)?),
                None => None,
            };
            let q = crate::krea2_fused::head_rope(
                &with_lora(&self.lq, self.wq.forward(x)?, x)?.reshape((b, s, h, d))?,
                &self.qnorm.weight,
                cs.as_ref(),
                self.qnorm.eps,
            )?;
            let k = crate::krea2_fused::head_rope(
                &with_lora(&self.lk, self.wk.forward(x)?, x)?.reshape((b, s, kvh, d))?,
                &self.knorm.weight,
                cs.as_ref(),
                self.knorm.eps,
            )?;
            let v = with_lora(&self.lv, self.wv.forward(x)?, x)?.reshape((b, s, kvh, d))?;
            let flops = 4.0 * (b * h * s * s * d) as f64;
            let out = crate::profile::scope("attention", flops, x.device(), || {
                crate::flash::attention_biased(&q.transpose(1, 2)?, &k.transpose(1, 2)?, &v.transpose(1, 2)?, 1.0 / (d as f32).sqrt(), kbias)
            })?;
            return self.gated(out.reshape((b, s, h * d))?, x);
        }
        let qb = self.qnorm.forward(&with_lora(&self.lq, self.wq.forward(x)?, x)?.reshape((b, s, h, d))?)?; // (B,S,H,D)
        let kb = self.knorm.forward(&with_lora(&self.lk, self.wk.forward(x)?, x)?.reshape((b, s, kvh, d))?)?;
        let vb = with_lora(&self.lv, self.wv.forward(x)?, x)?.reshape((b, s, kvh, d))?;
        // With RoPE and the `sage2` kernels, when asked (`QIR_KREA2_ATTN=sage`): INT8-QK
        // SageAttention2 straight from the BSHD projections (the rotation fused into the
        // quantiser), grouped heads handled by the kernel. Otherwise exact attention: the
        // `flash` kernel when built (the default), else the plain path below.
        #[cfg(feature = "sage2")]
        if let (Some((cos, sin)), Attn::Sage, None) = (rope, choice, bias) {
            if let crate::sage2::AttentionImpl::Sage2(accum) = crate::sage2::attention_impl()? {
                use crate::sage2::{attention, quant_v, rope_quant, Role};
                let dt = x.dtype();
                let (c, sn) = (cos.to_dtype(dt)?, sin.to_dtype(dt)?);
                let q = rope_quant(&qb.contiguous()?, &c, &sn, Role::Query)?;
                let k = rope_quant(&kb.contiguous()?, &c, &sn, Role::Key)?;
                let v = quant_v(&vb.to_dtype(DType::F16)?.contiguous()?, accum)?;
                let out = attention(&q, &k, &v, 1.0 / (d as f32).sqrt(), false)?.to_dtype(dt)?; // (B,S,H,D)
                return self.gated(out.reshape((b, s, h * d))?, x);
            }
        }
        let q = qb.transpose(1, 2)?.contiguous()?; // (B,H,S,D)
        let k = kb.transpose(1, 2)?.contiguous()?;
        let v = vb.transpose(1, 2)?.contiguous()?;
        let (q, k) = match rope {
            Some((cos, sin)) => {
                let dt = q.dtype();
                let q = candle_nn::rotary_emb::rope_i(&q.to_dtype(DType::F32)?, cos, sin)?.to_dtype(dt)?;
                let k = candle_nn::rotary_emb::rope_i(&k.to_dtype(DType::F32)?, cos, sin)?.to_dtype(dt)?;
                (q, k)
            }
            None => (q, k),
        };
        #[cfg(feature = "flash")]
        if choice == Attn::Flash && d == crate::flash::HEAD_DIM && q.dtype() == DType::BF16 && q.device().is_cuda() {
            let flops = 4.0 * (b * h * s * s * d) as f64;
            let out = crate::profile::scope("attention", flops, q.device(), || crate::flash::attention_biased(&q, &k, &v, 1.0 / (d as f32).sqrt(), kbias))?; // (B,S,H,D)
            return self.gated(out.reshape((b, s, h * d))?, x);
        }
        let _ = choice; // builds without the kernels have only the plain path
        let groups = h / kvh;
        let (k, v) = if groups > 1 {
            let rep = |t: &Tensor| -> Result<Tensor> { Ok(t.unsqueeze(2)?.broadcast_as((b, kvh, groups, s, d))?.reshape((b, h, s, d))?.contiguous()?) };
            (rep(&k)?, rep(&v)?)
        } else {
            (k, v)
        };
        let out = attend(&q, &k, &v, 1.0 / (d as f64).sqrt(), bias)?; // (B,H,S,D)
        self.gated(out.transpose(1, 2)?.reshape((b, s, h * d))?, x)
    }
}

/// Krea 2's attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Attn {
    /// Exact, one pass (`crate::flash`); the plain path without the `flash` feature.
    Flash,
    /// SageAttention2 (`crate::sage2`, its `QIR_SAGE` accumulation): faster, but measured
    /// further from ComfyUI's pictures (17 of 28 against 21).
    Sage,
    /// The plain f32 path (the reference).
    Plain,
}

/// `QIR_KREA2_ATTN`: `sage`, `plain`, or anything else for exact attention; read once.
fn attention_choice() -> Attn {
    static CHOICE: std::sync::OnceLock<Attn> = std::sync::OnceLock::new();
    *CHOICE.get_or_init(|| {
        let a = match std::env::var("QIR_KREA2_ATTN").ok().as_deref().map(str::trim) {
            Some("sage") => Attn::Sage,
            Some("plain") => Attn::Plain,
            _ => Attn::Flash,
        };
        tracing::info!(attention = ?a, "Krea 2 attention");
        a
    })
}

/// A text-fusion block: pre-norm attention and post-norm SwiGLU, plain residuals.
struct TextFusionBlock {
    prenorm: RmsNorm1p,
    postnorm: RmsNorm1p,
    attn: Attention,
    mlp: SwiGlu,
}

impl TextFusionBlock {
    fn load(cfg: &Krea2Config, q: Quant<'_>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            prenorm: RmsNorm1p::load(cfg.text_dim, cfg.eps, vb.pp("prenorm"))?,
            postnorm: RmsNorm1p::load(cfg.text_dim, cfg.eps, vb.pp("postnorm"))?,
            attn: Attention::load(cfg.text_dim, cfg.text_heads, cfg.text_kv_heads, cfg.eps, q, vb.pp("attn"))?,
            mlp: SwiGlu::load(cfg.text_dim, cfg.mlp_multiplier, q, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.prenorm.forward(x)?, None, None)?)?;
        Ok((&x + self.mlp.forward(&self.postnorm.forward(&x)?)?)?)
    }
}

/// `txtfusion`: two blocks across each token's twelve layer states, a learned 12→1 mix, two
/// blocks along the sequence.
struct TextFusion {
    layerwise: [TextFusionBlock; 2],
    projector: Linear, // (1, 12), no bias
    refiner: [TextFusionBlock; 2],
}

impl TextFusion {
    fn load(cfg: &Krea2Config, q: Quant<'_>, vb: VarBuilder) -> Result<Self> {
        let lw = vb.pp("layerwise_blocks");
        let rf = vb.pp("refiner_blocks");
        Ok(Self {
            layerwise: [TextFusionBlock::load(cfg, q, lw.pp("0"))?, TextFusionBlock::load(cfg, q, lw.pp("1"))?],
            projector: linear_no_bias(cfg.text_layers, 1, vb.pp("projector"))?,
            refiner: [TextFusionBlock::load(cfg, q, rf.pp("0"))?, TextFusionBlock::load(cfg, q, rf.pp("1"))?],
        })
    }

    /// `ctx` `(B, seq, layers, dim)` → `(B, seq, dim)`.
    fn forward(&self, ctx: &Tensor) -> Result<Tensor> {
        let (b, seq, n, d) = ctx.dims4()?;
        let mut x = ctx.reshape((b * seq, n, d))?;
        for blk in &self.layerwise {
            x = blk.forward(&x)?;
        }
        // (B·seq, n, d) → (B·seq, d, n) → projector over n → (B, seq, d)
        let x = x.transpose(1, 2)?.contiguous()?;
        let mut x = self.projector.forward(&x)?.squeeze(D::Minus1)?.reshape((b, seq, d))?;
        for blk in &self.refiner {
            x = blk.forward(&x)?;
        }
        Ok(x)
    }
}

/// A DiT block: AdaLN-single modulation from the shared `tvec` plus the block's own bias.
struct Block {
    lin: Tensor, // (6·features,)
    prenorm: RmsNorm1p,
    postnorm: RmsNorm1p,
    attn: Attention,
    mlp: SwiGlu,
}

impl Block {
    fn load(cfg: &Krea2Config, q: Quant<'_>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin: vb.pp("mod").get(6 * cfg.features, "lin")?,
            prenorm: RmsNorm1p::load(cfg.features, cfg.eps, vb.pp("prenorm"))?,
            postnorm: RmsNorm1p::load(cfg.features, cfg.eps, vb.pp("postnorm"))?,
            attn: Attention::load(cfg.features, cfg.heads, cfg.kv_heads, cfg.eps, q, vb.pp("attn"))?,
            mlp: SwiGlu::load(cfg.features, cfg.mlp_multiplier, q, vb.pp("mlp"))?,
        })
    }

    /// `x` `(B,S,F)`, `tvec` `(B,1,6F)`; `bias` on the attention scores (an edit's reference).
    fn forward(&self, x: &Tensor, tvec: &Tensor, rope: &(Tensor, Tensor), bias: Option<ScoreBias>) -> Result<Tensor> {
        // mods: (pre scale, pre shift, pre gate, post scale, post shift, post gate) × F.
        let mods = tvec.broadcast_add(&self.lin)?; // (B,1,6F)
        let h = self.prenorm.modulated(x, &mods, 0, 1)?;
        let x = gate_add(x, &self.attn.forward(&h, Some(rope), bias)?, &mods, 2)?;
        let h = self.postnorm.modulated(&x, &mods, 3, 4)?;
        gate_add(&x, &self.mlp.forward(&h)?, &mods, 5)
    }

    /// Split modulation at the reference boundary while retaining joint attention.
    fn forward_zero(&self, x: &Tensor, tvec: &Tensor, zero: &Tensor, split: usize, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let mods = tvec.broadcast_add(&self.lin)?;
        let refs = zero.broadcast_add(&self.lin)?;
        let n = x.dim(1)? - split;
        let normalize = |norm: &RmsNorm1p, x: &Tensor, scale, shift| -> Result<Tensor> {
            Ok(Tensor::cat(&[norm.modulated(&x.narrow(1, 0, split)?, &mods, scale, shift)?, norm.modulated(&x.narrow(1, split, n)?, &refs, scale, shift)?], 1)?)
        };
        let residual = |x: &Tensor, h: &Tensor, gate| -> Result<Tensor> {
            Ok(Tensor::cat(
                &[
                    gate_add(&x.narrow(1, 0, split)?, &h.narrow(1, 0, split)?, &mods, gate)?,
                    gate_add(&x.narrow(1, split, n)?, &h.narrow(1, split, n)?, &refs, gate)?,
                ],
                1,
            )?)
        };
        let h = normalize(&self.prenorm, x, 0, 1)?;
        let x = residual(x, &self.attn.forward(&h, Some(rope), None)?, 2)?;
        let h = normalize(&self.postnorm, &x, 3, 4)?;
        residual(&x, &self.mlp.forward(&h)?, 5)
    }
}

/// An edit's reference for [`Krea2Dit::forward_edit`]: the picture's clean latent in the
/// sampler's space `(B, 16, h, w)` (sides no larger than the target's) and `ref_boost`, the
/// factor on the target's attention to its tokens (1 = none; the General Edit V2 LoRA's card
/// recommends 4).
pub struct EditRef {
    pub latent: Tensor,
    pub boost: f32,
}

/// The whole model.
pub struct Krea2Dit {
    cfg: Krea2Config,
    first: Linear,
    tmlp: (Linear, Linear),
    tproj: Linear,
    txtfusion: TextFusion,
    txtmlp: (RmsNorm1p, Linear, Linear),
    blocks: Vec<Block>,
    last_norm: RmsNorm1p,
    last_lin: Linear,
    last_mod: Tensor, // (2, F)
    device: Device,
    bound_loras: std::sync::Mutex<Option<(LoraStack, Vec<String>)>>,
}

impl Krea2Dit {
    /// Load the model; with `convrot` the block and text-fusion linears are rotated and
    /// quantised to INT8 on the way in (the `convrot` feature; ignored without it).
    pub fn load(cfg: &Krea2Config, convrot: bool, vb: VarBuilder) -> Result<Self> {
        Self::load_with(cfg, convrot, None, vb)
    }

    /// [`Self::load`] for a file whose block linears may be e4m3 (`fp8`: what
    /// [`Fp8Plan::from_files`] read from it); those layers keep the file's bytes and ConvRot is
    /// not applied to them.
    pub fn load_with(cfg: &Krea2Config, convrot: bool, fp8: Option<&Fp8Plan>, vb: VarBuilder) -> Result<Self> {
        let f = cfg.features;
        let p2 = cfg.patch * cfg.patch;
        #[cfg(feature = "convrot")]
        let rot: Option<Tensor> = if convrot && fp8.is_none() { Some(crate::model::rotation::regular_hadamard_256(vb.device())?) } else { None };
        #[cfg(not(feature = "convrot"))]
        let rot: Option<Tensor> = {
            let _ = convrot;
            None
        };
        let q = Quant { rot: rot.as_ref(), fp8 };
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            blocks.push(Block::load(cfg, q, vb.pp("blocks").pp(i))?);
        }
        Ok(Self {
            cfg: cfg.clone(),
            first: linear(cfg.channels * p2, f, vb.pp("first"))?,
            tmlp: (linear(cfg.timestep_dim, f, vb.pp("tmlp").pp("0"))?, linear(f, f, vb.pp("tmlp").pp("2"))?),
            tproj: linear(f, 6 * f, vb.pp("tproj").pp("1"))?,
            txtfusion: TextFusion::load(cfg, q, vb.pp("txtfusion"))?,
            txtmlp: (
                RmsNorm1p::load(cfg.text_dim, cfg.eps, vb.pp("txtmlp").pp("0"))?,
                linear(cfg.text_dim, f, vb.pp("txtmlp").pp("1"))?,
                linear(f, f, vb.pp("txtmlp").pp("3"))?,
            ),
            blocks,
            last_norm: RmsNorm1p::load(f, cfg.eps, vb.pp("last").pp("norm"))?,
            last_lin: linear(f, p2 * cfg.channels, vb.pp("last").pp("linear"))?,
            last_mod: vb.pp("last").pp("modulation").get((2, f), "lin")?,
            device: vb.device().clone(),
            bound_loras: std::sync::Mutex::new(None),
        })
    }

    pub fn config(&self) -> &Krea2Config {
        &self.cfg
    }

    /// Put `stack`'s adapters on every block and text-fusion linear they name (an empty stack
    /// clears them all); the AnyPaint and outpaint LoRAs address exactly these. Returns the
    /// modules the stack names that this model has not, for the caller to report.
    pub fn set_loras(&self, stack: &LoraStack) -> Result<Vec<String>> {
        let mut bound = self.bound_loras.lock().map_err(|_| anyhow::anyhow!("Krea adapter binding lock poisoned"))?;
        if let Some((previous, unmatched)) = bound.as_ref() {
            if previous.same_adapters(stack) {
                return Ok(unmatched.clone());
            }
        }
        *bound = None;
        for (i, b) in self.blocks.iter().enumerate() {
            b.attn.set_loras(stack, &format!("blocks.{i}.attn"))?;
            b.mlp.set_loras(stack, &format!("blocks.{i}.mlp"))?;
        }
        for (name, group) in [("layerwise_blocks", &self.txtfusion.layerwise), ("refiner_blocks", &self.txtfusion.refiner)] {
            for (j, blk) in group.iter().enumerate() {
                blk.attn.set_loras(stack, &format!("txtfusion.{name}.{j}.attn"))?;
                blk.mlp.set_loras(stack, &format!("txtfusion.{name}.{j}.mlp"))?;
            }
        }
        let n = self.blocks.len();
        let known = |m: &str| -> bool {
            let leaf = |tail: &str| matches!(tail, "attn.wq" | "attn.wk" | "attn.wv" | "attn.gate" | "attn.wo" | "mlp.gate" | "mlp.up" | "mlp.down");
            if let Some(rest) = m.strip_prefix("blocks.") {
                let Some((idx, tail)) = rest.split_once('.') else { return false };
                return idx.parse::<usize>().is_ok_and(|i| i < n) && leaf(tail);
            }
            for g in ["txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks."] {
                if let Some(rest) = m.strip_prefix(g) {
                    let Some((idx, tail)) = rest.split_once('.') else { return false };
                    return idx.parse::<usize>().is_ok_and(|i| i < 2) && leaf(tail);
                }
            }
            false
        };
        let unmatched: Vec<String> = stack.unmatched(&known).into_iter().map(String::from).collect();
        *bound = Some((stack.clone(), unmatched.clone()));
        Ok(unmatched)
    }

    /// Flux's sinusoidal embedding of `t` (×1000, 256 wide: 128 cosines then 128 sines), f32.
    fn timestep_embedding(&self, t: f32) -> Result<Tensor> {
        let half = self.cfg.timestep_dim / 2;
        let tt = t * 1000.0;
        let mut v = vec![0f32; 2 * half];
        for i in 0..half {
            let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            v[i] = (tt * freq).cos();
            v[half + i] = (tt * freq).sin();
        }
        Ok(Tensor::from_vec(v, (1, 1, 2 * half), &self.device)?)
    }

    /// Interleaved RoPE tables `(S, D/2)` in f32 for 3-axis positions.
    fn rope_tables(&self, ids: &[[f32; 3]]) -> Result<(Tensor, Tensor)> {
        let axes = self.cfg.axes();
        let half: usize = axes.iter().sum::<usize>() / 2;
        let s = ids.len();
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        for (r, id) in ids.iter().enumerate() {
            let mut off = 0;
            for (a, dim) in axes.iter().enumerate() {
                for j in 0..dim / 2 {
                    let omega = self.cfg.theta.powf(-(2.0 * j as f32) / *dim as f32);
                    let ang = id[a] * omega;
                    cos[r * half + off + j] = ang.cos();
                    sin[r * half + off + j] = ang.sin();
                }
                off += dim / 2;
            }
        }
        Ok((Tensor::from_vec(cos, (s, half), &self.device)?, Tensor::from_vec(sin, (s, half), &self.device)?))
    }

    /// `(B, C, H, W)` → `(B, h·w, C·p·p)` with `(c, ph, pw)` innermost, and the grid.
    fn patchify(&self, x: &Tensor) -> Result<(Tensor, usize, usize)> {
        let p = self.cfg.patch;
        let (b, c, hh, ww) = x.dims4()?;
        anyhow::ensure!(hh.is_multiple_of(p) && ww.is_multiple_of(p), "the latent's sides must be multiples of {p} ({hh}x{ww})");
        let (h, w) = (hh / p, ww / p);
        let t = x.reshape((b, c, h, p, w, p))?.permute((0, 2, 4, 1, 3, 5))?.contiguous()?.reshape((b, h * w, c * p * p))?;
        Ok((t, h, w))
    }

    fn unpatchify(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (p, c) = (self.cfg.patch, self.cfg.channels);
        let b = x.dim(0)?;
        Ok(x.reshape((b, h, w, c, p, p))?.permute((0, 3, 1, 4, 2, 5))?.contiguous()?.reshape((b, c, h * p, w * p))?)
    }

    /// The velocity for the latent `x` `(B, 16, H, W)` at `t` (the sigma, 1 → 0) with the
    /// conditioning `ctx` `(B, seq, 12, 2560)`.
    pub fn forward(&self, x: &Tensor, t: f32, ctx: &Tensor) -> Result<Tensor> {
        self.forward_edit(x, t, ctx, None)
    }

    /// Stock Krea multi-reference conditioning: target first, positive frame indices,
    /// independent spatial grids, and timestep-zero modulation for reference tokens.
    pub fn forward_style(&self, x: &Tensor, t: f32, ctx: &Tensor, references: &[Tensor]) -> Result<Tensor> {
        self.style_forward(x, t, ctx, references, None)
    }

    /// CPU parity runner: load one DiT block at a time to bound checkpoint residency.
    /// Load this model with zero blocks; `weights` supplies the original block prefixes.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_style_streamed(
        &self,
        x: &Tensor,
        t: f32,
        ctx: &Tensor,
        references: &[Tensor],
        layers: usize,
        weights: &VarBuilder<'_>,
        loras: &LoraStack,
    ) -> Result<Tensor> {
        anyhow::ensure!(x.device().is_cpu() && self.blocks.is_empty(), "streaming requires a CPU model loaded with zero blocks");
        self.style_forward(x, t, ctx, references, Some((layers, weights, loras)))
    }

    fn style_forward(&self, x: &Tensor, t: f32, ctx: &Tensor, references: &[Tensor], streamed: Option<(usize, &VarBuilder<'_>, &LoraStack)>) -> Result<Tensor> {
        anyhow::ensure!(!references.is_empty() && references.len() <= 2, "style needs one or two references");
        let dt = x.dtype();
        let (target, h, w) = self.patchify(x)?;
        let context = self.txtfusion.forward(&ctx.to_dtype(dt)?)?;
        let context = self.txtmlp.2.forward(&gelu_tanh(&self.txtmlp.1.forward(&self.txtmlp.0.forward(&context)?)?)?)?;
        let seq = context.dim(1)?;
        let mut ids = vec![[0.0; 3]; seq];
        let mut parts = vec![context, self.first.forward(&target)?];
        for r in 0..h {
            for c in 0..w {
                ids.push([0.0, r as f32, c as f32]);
            }
        }
        let split = ids.len();
        for (i, reference) in references.iter().enumerate() {
            // Comfy pads odd reference grids circularly before patchification.
            let mut reference = reference.to_dtype(dt)?;
            for axis in [2, 3] {
                let len = reference.dim(axis)?;
                let pad = (self.cfg.patch - len % self.cfg.patch) % self.cfg.patch;
                if pad > 0 {
                    reference = Tensor::cat(&[&reference, &reference.narrow(axis, 0, pad)?], axis)?;
                }
            }
            let (tokens, rh, rw) = self.patchify(&reference)?;
            anyhow::ensure!(tokens.dim(0)? == x.dim(0)?, "reference batch differs from target");
            for r in 0..rh {
                for c in 0..rw {
                    ids.push([(i + 1) as f32, r as f32, c as f32]);
                }
            }
            parts.push(self.first.forward(&tokens)?);
        }
        let embedding =
            |time| -> Result<Tensor> { Ok(self.tmlp.1.forward(&gelu_tanh(&self.tmlp.0.forward(&self.timestep_embedding(time)?.to_dtype(dt)?)?)?)?) };
        let temb = embedding(t)?;
        let tvec = self.tproj.forward(&gelu_tanh(&temb)?)?;
        let zero = self.tproj.forward(&gelu_tanh(&embedding(0.0)?)?)?;
        let rope = self.rope_tables(&ids)?;
        let mut xs = Tensor::cat(&parts, 1)?;
        if let Some((layers, weights, loras)) = streamed {
            for i in 0..layers {
                let prefix = format!("blocks.{i}");
                let block = Block::load(&self.cfg, Quant { rot: None, fp8: None }, weights.pp(&prefix))?;
                block.attn.set_loras(loras, &format!("{prefix}.attn"))?;
                block.mlp.set_loras(loras, &format!("{prefix}.mlp"))?;
                xs = block.forward_zero(&xs, &tvec, &zero, split, &rope)?;
                tracing::info!(block = i, "streamed Krea style block");
            }
        } else {
            for block in &self.blocks {
                xs = block.forward_zero(&xs, &tvec, &zero, split, &rope)?;
            }
        }
        let xs = xs.narrow(1, seq, h * w)?;
        let lm = self.last_mod.to_dtype(dt)?;
        let scale = temb.broadcast_add(&lm.i(0)?)?;
        let shift = temb.broadcast_add(&lm.i(1)?)?;
        let out = self.last_norm.forward(&xs)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        self.unpatchify(&self.last_lin.forward(&out)?, h, w)
    }

    /// [`Self::forward`] with an edit's reference ([`EditRef`]): the sequence is
    /// `[text | reference | target]`, the reference's clean tokens on RoPE frame 1 at the
    /// target's own grid positions ("aligned", how the krea2_edit LoRAs were trained), the same
    /// timestep modulation for every token, and only the target's tokens come out (the
    /// comfyui-krea2edit nodes' `krea2_edit_forward`, read 2026-10-27).
    pub fn forward_edit(&self, x: &Tensor, t: f32, ctx: &Tensor, edit: Option<&EditRef>) -> Result<Tensor> {
        let dt = x.dtype();
        let (img, h, w) = self.patchify(x)?;
        let img = self.first.forward(&img)?; // (B, N, F)
        let temb = self.tmlp.1.forward(&gelu_tanh(&self.tmlp.0.forward(&self.timestep_embedding(t)?.to_dtype(dt)?)?)?)?; // (1,1,F)
        let tvec = self.tproj.forward(&gelu_tanh(&temb)?)?; // (1,1,6F)
        let context = self.txtfusion.forward(&ctx.to_dtype(dt)?)?;
        let context = self.txtmlp.2.forward(&gelu_tanh(&self.txtmlp.1.forward(&self.txtmlp.0.forward(&context)?)?)?)?; // (B, seq, F)
        let seq = context.dim(1)?;
        let mut ids: Vec<[f32; 3]> = vec![[0.0; 3]; seq];
        let mut parts: Vec<Tensor> = vec![context];
        let mut bias = None;
        if let Some(e) = edit {
            let (rtok, rh, rw) = self.patchify(&e.latent.to_dtype(dt)?)?;
            anyhow::ensure!(rtok.dim(0)? == img.dim(0)?, "the reference's batch differs from the target's");
            // Centred inside the target's grid when smaller (`_imgids_offset`); aligned when equal.
            let (oh, ow) = ((h as f32 - rh as f32).max(0.0) / 2.0, (w as f32 - rw as f32).max(0.0) / 2.0);
            for r in 0..rh {
                for c in 0..rw {
                    ids.push([1.0, r as f32 + oh, c as f32 + ow]);
                }
            }
            if e.boost != 1.0 && e.boost > 0.0 {
                bias = Some(ScoreBias { rows_from: seq + rh * rw, keys: (seq, seq + rh * rw), log_boost: e.boost.ln() });
            }
            parts.push(self.first.forward(&rtok)?);
        }
        let prefix: usize = ids.len();
        for r in 0..h {
            for c in 0..w {
                ids.push([0.0, r as f32, c as f32]);
            }
        }
        parts.push(img);
        let rope = self.rope_tables(&ids)?;
        let mut xs = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;
        for blk in &self.blocks {
            xs = blk.forward(&xs, &tvec, &rope, bias)?;
        }
        // The last layer: scale and shift are `temb` plus the layer's own two rows.
        let xs = xs.narrow(1, prefix, h * w)?;
        let lm = self.last_mod.to_dtype(dt)?; // (2, F)
        let scale = temb.broadcast_add(&lm.i(0)?)?; // (1,1,F)
        let shift = temb.broadcast_add(&lm.i(1)?)?;
        let out = self.last_norm.forward(&xs)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        let out = self.last_lin.forward(&out)?; // (B, N, C·p·p)
        self.unpatchify(&out, h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shape_follows_the_file() {
        let c = Krea2Config::default();
        assert_eq!((c.head_dim(), c.axes()), (128, [32, 48, 48]));
        assert_eq!(Krea2Config::mlp_dim(6144, 4), 16384);
        assert_eq!(Krea2Config::mlp_dim(2560, 4), 6912);
    }

    #[test]
    fn the_convrot_policy_names_the_block_linears_only() {
        for p in ["blocks.0.attn.wq", "blocks.27.attn.gate", "blocks.3.mlp.down", "txtfusion.layerwise_blocks.1.attn.wo", "txtfusion.refiner_blocks.0.mlp.up"] {
            assert!(is_convrot_target(p), "{p}");
        }
        for p in ["first", "tmlp.0", "tproj.1", "txtmlp.1", "last.linear", "txtfusion.projector", "blocks.0.attn.qknorm.qnorm", "blocks.0.mod"] {
            assert!(!is_convrot_target(p), "{p}");
        }
        assert!(convrot_policy_tag().starts_with("qir-krea2-convrot-v"));
    }

    #[test]
    fn the_prefix_ends_after_the_user_turn_opens() {
        // system turn … <|im_start|> user \n prompt …
        let ids = [IM_START, 8948, NEWLINE, 1, 2, 151_645, NEWLINE, IM_START, USER, NEWLINE, 100, 101, 151_645];
        assert_eq!(template_end(&ids), 10);
        // Without the `user\n` pair the cut sits at the second <|im_start|> itself.
        let ids = [IM_START, 8948, NEWLINE, 1, IM_START, 5, 6, 7, 8];
        assert_eq!(template_end(&ids), 4);
        assert_eq!(template_end(&[]), 0);
    }

    #[test]
    fn the_template_is_comfyuis() {
        let t = template("a fox");
        assert!(
            t.starts_with("<|im_start|>system\nDescribe the image by detailing") && t.ends_with("<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n")
        );
    }
}
