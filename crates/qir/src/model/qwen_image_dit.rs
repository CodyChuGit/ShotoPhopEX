//! Layered Qwen-Image's 60 dual-stream blocks, without sampler or pipeline state.
//! Checkpoint builders are unprefixed ComfyUI model keys; CPU uses f32, CUDA bf16.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::Result;

/// Read individual tensors without mapping or copying the entire 20 GB checkpoint.
/// Builders retain only the header; callers control block residency by dropping blocks.
pub fn checkpoint_builder(path: impl AsRef<std::path::Path>, device: &Device) -> Result<VarBuilder<'static>> {
    use std::io::Read;
    let path = path.as_ref().to_path_buf();
    let mut file = std::fs::File::open(&path)?;
    let mut len = [0; 8];
    file.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    anyhow::ensure!(len < 256 << 20, "oversized safetensors header");
    let mut header = vec![0; len as usize];
    file.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    Ok(VarBuilder::from_backend(
        Box::new(Checkpoint { path, header, offset: len + 8 }),
        if device.is_cuda() { DType::BF16 } else { DType::F32 },
        device.clone(),
    ))
}

struct Checkpoint {
    path: std::path::PathBuf,
    header: serde_json::Value,
    offset: u64,
}

impl candle_nn::var_builder::SimpleBackend for Checkpoint {
    fn get(&self, shape: candle_core::Shape, name: &str, _: candle_nn::Init, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!("{name}: expected {shape:?}, got {:?}", tensor.shape());
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        use std::io::{Read, Seek, SeekFrom};
        let parse = || -> Result<Tensor> {
            let entry = self.header.get(name).ok_or_else(|| anyhow::anyhow!("missing checkpoint tensor {name}"))?;
            let shape = serde_json::from_value::<Vec<usize>>(entry["shape"].clone())?;
            let offsets = serde_json::from_value::<[u64; 2]>(entry["data_offsets"].clone())?;
            let stored = match entry["dtype"].as_str() {
                Some("BF16") => DType::BF16,
                Some("F32") => DType::F32,
                Some("F16") => DType::F16,
                Some("F8_E4M3") => DType::F8E4M3,
                Some("U8") => DType::U8,
                other => anyhow::bail!("{name}: unsupported stored dtype {other:?}"),
            };
            let size = offsets[1].checked_sub(offsets[0]).ok_or_else(|| anyhow::anyhow!("invalid tensor offsets"))?;
            anyhow::ensure!(size <= 1 << 30, "oversized tensor {name}");
            let mut file = std::fs::File::open(&self.path)?;
            file.seek(SeekFrom::Start(self.offset + offsets[0]))?;
            let mut bytes = vec![0; size as usize];
            file.read_exact(&mut bytes)?;
            Ok(Tensor::from_raw_buffer(&bytes, stored, &shape, device)?.to_dtype(dtype)?)
        };
        parse().map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.header.get(name).is_some()
    }
}

pub const WIDTH: usize = 3072;
pub const TEXT_WIDTH: usize = 3584;
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const BLOCKS: usize = 60;

/// Biased mixed-file linear. FP8 multiplication follows the checkpoint's metadata.
struct LayeredLinear {
    dense: Option<candle_nn::Linear>,
    fp8: Option<(Tensor, Tensor)>,
    bias: Tensor,
}

impl LayeredLinear {
    fn load(input: usize, output: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let fp8 = if vb.contains_tensor("weight_scale") {
            let bytes = vb.get_unchecked_dtype("comfy_quant", DType::U8)?.to_vec1::<u8>()?;
            let quant: serde_json::Value = serde_json::from_slice(&bytes)?;
            anyhow::ensure!(quant["full_precision_matrix_mult"] == true && quant["format"] == "float8_e4m3fn", "Layered requires full-precision E4M3 matmuls");
            let weight = vb.get_unchecked_dtype("weight", DType::F8E4M3)?;
            anyhow::ensure!(weight.dims2()? == (output, input), "incorrect Layered linear weight shape");
            let scale = vb.get_unchecked_dtype("weight_scale", DType::F32)?.reshape(())?;
            Some((weight, scale))
        } else {
            None
        };
        let dense = if fp8.is_none() { Some(candle_nn::linear_no_bias(input, output, vb.clone())?) } else { None };
        Ok(Self { dense, fp8, bias: vb.get(output, "bias")? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = match (&self.dense, &self.fp8) {
            (Some(dense), _) => dense.forward(x)?,
            (_, Some((weight, scale))) => {
                // Comfy's eager dequantization casts BOTH operands to the compute dtype.
                // The shared fp8 reader rounds scales to bf16 even for CPU f32 inputs.
                let weight = weight.to_dtype(x.dtype())?.broadcast_mul(&scale.to_dtype(x.dtype())?)?;
                let mut shape = x.dims().to_vec();
                let output = weight.dim(0)?;
                let input = weight.dim(1)?;
                if let Some(last) = shape.last_mut() {
                    *last = output;
                }
                x.reshape(((), input))?.matmul(&weight.t()?)?.reshape(shape)?
            }
            _ => anyhow::bail!("missing Layered linear weight"),
        };
        Ok(y.broadcast_add(&self.bias)?)
    }
}

/// Non-affine LayerNorm accumulates in f32, including on CUDA.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    let centered = f.broadcast_sub(&f.mean_keepdim(D::Minus1)?)?;
    Ok(centered.broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?.to_dtype(x.dtype())?)
}

fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    // PyTorch RMSNorm casts the normalized result before multiplying the weight.
    Ok(f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?.to_dtype(x.dtype())?.broadcast_mul(weight)?)
}

/// Match addcmul's single bf16 rounding at modulation and gated residual boundaries.
fn add_product(base: &Tensor, left: &Tensor, right: &Tensor) -> Result<Tensor> {
    Ok(left.to_dtype(DType::F32)?.broadcast_mul(&right.to_dtype(DType::F32)?)?.broadcast_add(&base.to_dtype(DType::F32)?)?.to_dtype(base.dtype())?)
}

/// Spatial 2x2 patches in (frame,row,column,channel,patch-row,patch-column) order.
/// Padding belongs to the caller; Layered's pipeline supplies even latent dimensions.
pub fn patchify(x: &Tensor) -> Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;
    anyhow::ensure!(c == 16 && h.is_multiple_of(2) && w.is_multiple_of(2), "Layered patches need 16 channels and even spatial dimensions");
    Ok(x.reshape(&[b, c, t, h / 2, 2, w / 2, 2])?.permute(&[0, 2, 3, 5, 1, 4, 6][..])?.reshape((b, t * (h / 2) * (w / 2), 64))?)
}

/// Inverse of [`patchify`], useful at the velocity boundary.
pub fn unpatchify(x: &Tensor, frames: usize, height: usize, width: usize) -> Result<Tensor> {
    anyhow::ensure!(height.is_multiple_of(2) && width.is_multiple_of(2), "even latent dimensions required");
    let b = x.dim(0)?;
    Ok(x.reshape(&[b, frames, height / 2, width / 2, 16, 2, 2])?.permute(&[0, 4, 1, 2, 5, 3, 6][..])?.reshape((b, 16, frames, height, width))?)
}

/// Centered token-grid positions. A single-frame reference uses index -1, -2, ...;
/// multiple generated frames use 0..frames irrespective of index, like ComfyUI.
pub fn image_positions(batch: usize, frames: usize, rows: usize, cols: usize, index: i32, device: &Device) -> Result<Tensor> {
    anyhow::ensure!(batch > 0 && frames > 0 && rows > 0 && cols > 0, "empty image grid");
    let mut ids = Vec::with_capacity(frames * rows * cols * 3);
    for t in 0..frames {
        for y in 0..rows {
            for x in 0..cols {
                ids.extend([if frames > 1 { t as f32 } else { index as f32 }, y as f32 - (rows / 2) as f32, x as f32 - (cols / 2) as f32]);
            }
        }
    }
    Ok(Tensor::from_vec(ids, (1, frames * rows * cols, 3), device)?.broadcast_as((batch, frames * rows * cols, 3))?.contiguous()?)
}

/// Text starts at the integer half of the larger target token-grid dimension.
pub fn text_positions(batch: usize, tokens: usize, rows: usize, cols: usize, device: &Device) -> Result<Tensor> {
    let start = (rows / 2).max(cols / 2);
    let ids: Vec<f32> = (0..tokens).flat_map(|i| [(start + i) as f32; 3]).collect();
    Ok(Tensor::from_vec(ids, (1, tokens, 3), device)?.broadcast_as((batch, tokens, 3))?.contiguous()?)
}

/// Three-axis adjacent-pair rotations, computed in f64 then stored in the model dtype.
pub struct LayeredRope {
    cos: Tensor,
    sin: Tensor,
}

impl LayeredRope {
    pub fn new(ids: &Tensor, dtype: DType) -> Result<Self> {
        let (b, s, axes) = ids.dims3()?;
        anyhow::ensure!(axes == 3, "RoPE needs three position axes");
        let positions = ids.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut cos = Vec::with_capacity(b * s * 64);
        let mut sin = Vec::with_capacity(b * s * 64);
        for pos in positions.as_chunks::<3>().0 {
            for (axis, dim) in [16, 56, 56].into_iter().enumerate() {
                for i in 0..dim / 2 {
                    let angle = f64::from(pos[axis]) * 10000f64.powf(-(2 * i) as f64 / dim as f64);
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (b, 1, s, 64), ids.device())?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, (b, 1, s, 64), ids.device())?.to_dtype(dtype)?,
        })
    }

    /// ComfyUI's rotation matrix layout (B,1,S,64,2,2), for fixture inspection.
    pub fn matrices(&self) -> Result<Tensor> {
        let row0 = Tensor::stack(&[&self.cos, &self.sin.neg()?], D::Minus1)?;
        let row1 = Tensor::stack(&[&self.sin, &self.cos], D::Minus1)?;
        Ok(Tensor::stack(&[row0, row1], D::Minus2)?)
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        anyhow::ensure!(d == HEAD_DIM, "Layered RoPE needs 128-wide heads");
        let pairs = x.reshape((b, h, s, 64, 2))?;
        let a = pairs.narrow(4, 0, 1)?.squeeze(4)?;
        let z = pairs.narrow(4, 1, 1)?.squeeze(4)?;
        let even = add_product(&a.broadcast_mul(&self.cos)?, &z, &self.sin.neg()?)?;
        let odd = add_product(&a.broadcast_mul(&self.sin)?, &z, &self.cos)?;
        Ok(Tensor::stack(&[even, odd], 4)?.reshape((b, h, s, d))?)
    }
}

/// ComfyUI Timesteps(256, flip_sin_to_cos=true, scale=1000).
pub fn timestep_projection(timestep: &Tensor) -> Result<Tensor> {
    let frequencies: Vec<f32> = (0..128).map(|i| (-(10000f32).ln() * i as f32 / 128.0).exp()).collect();
    let frequencies = Tensor::from_vec(frequencies, (1, 128), timestep.device())?;
    let phase = (timestep.to_dtype(DType::F32)?.reshape(((), 1))?.broadcast_mul(&frequencies)? * 1000.0)?;
    Ok(Tensor::cat(&[phase.cos()?, phase.sin()?], 1)?)
}

/// Input projections and learned timestep conditioning, independently loadable for parity.
pub struct LayeredInput {
    img: LayeredLinear,
    txt: LayeredLinear,
    txt_norm: Tensor,
    time1: LayeredLinear,
    time2: LayeredLinear,
    addition: Tensor,
}

impl LayeredInput {
    pub fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            img: LayeredLinear::load(64, WIDTH, vb.pp("img_in"))?,
            txt: LayeredLinear::load(TEXT_WIDTH, WIDTH, vb.pp("txt_in"))?,
            txt_norm: vb.pp("txt_norm").get(TEXT_WIDTH, "weight")?,
            time1: LayeredLinear::load(256, WIDTH, vb.pp("time_text_embed.timestep_embedder.linear_1"))?,
            time2: LayeredLinear::load(WIDTH, WIDTH, vb.pp("time_text_embed.timestep_embedder.linear_2"))?,
            addition: vb.pp("time_text_embed.addition_t_embedding").get((2, WIDTH), "weight")?,
        })
    }

    pub fn timestep(&self, timestep: &Tensor, additional_t_cond: Option<&Tensor>) -> Result<Tensor> {
        let time = timestep_projection(timestep)?.to_dtype(self.addition.dtype())?;
        let time = self.time2.forward(&candle_nn::ops::silu(&self.time1.forward(&time)?)?)?;
        let batch = time.dim(0)?;
        let ids = match additional_t_cond {
            Some(ids) => ids.to_dtype(DType::U32)?.reshape(batch)?,
            None => Tensor::zeros(batch, DType::U32, time.device())?,
        };
        Ok((time + self.addition.index_select(&ids, 0)?)?)
    }

    pub fn project(&self, img: &Tensor, txt: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((self.img.forward(img)?, self.txt.forward(&rms_norm(txt, &self.txt_norm)?)?))
    }
}

struct Stream {
    image: bool,
    modulation: LayeredLinear,
    q: LayeredLinear,
    k: LayeredLinear,
    v: LayeredLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    out: LayeredLinear,
    up: LayeredLinear,
    down: LayeredLinear,
}

impl Stream {
    fn load(text: bool, vb: VarBuilder<'_>) -> Result<Self> {
        let stream = if text { "txt" } else { "img" };
        let attn = vb.pp("attn");
        Ok(Self {
            image: !text,
            modulation: LayeredLinear::load(WIDTH, 6 * WIDTH, vb.pp(format!("{stream}_mod.1")))?,
            q: LayeredLinear::load(WIDTH, WIDTH, attn.pp(if text { "add_q_proj" } else { "to_q" }))?,
            k: LayeredLinear::load(WIDTH, WIDTH, attn.pp(if text { "add_k_proj" } else { "to_k" }))?,
            v: LayeredLinear::load(WIDTH, WIDTH, attn.pp(if text { "add_v_proj" } else { "to_v" }))?,
            q_norm: attn.pp(if text { "norm_added_q" } else { "norm_q" }).get(HEAD_DIM, "weight")?,
            k_norm: attn.pp(if text { "norm_added_k" } else { "norm_k" }).get(HEAD_DIM, "weight")?,
            out: LayeredLinear::load(WIDTH, WIDTH, attn.pp(if text { "to_add_out" } else { "to_out.0" }))?,
            up: LayeredLinear::load(WIDTH, 4 * WIDTH, vb.pp(format!("{stream}_mlp.net.0.proj")))?,
            down: LayeredLinear::load(4 * WIDTH, WIDTH, vb.pp(format!("{stream}_mlp.net.2")))?,
        })
    }

    fn modulate(x: &Tensor, params: &Tensor, offset: usize) -> Result<Tensor> {
        let shift = params.narrow(1, offset * WIDTH, WIDTH)?.unsqueeze(1)?;
        let scale = (params.narrow(1, (offset + 1) * WIDTH, WIDTH)? + 1.0)?.unsqueeze(1)?;
        add_product(&shift, &layer_norm(x)?, &scale)
    }

    fn qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let heads = |x: Tensor| -> Result<Tensor> { Ok(x.reshape((b, s, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?) };
        Ok((rms_norm(&heads(self.q.forward(x)?)?, &self.q_norm)?, rms_norm(&heads(self.k.forward(x)?)?, &self.k_norm)?, heads(self.v.forward(x)?)?))
    }

    fn finish(&self, x: &Tensor, attention: &Tensor, params: &Tensor) -> Result<Tensor> {
        let projected = self.out.forward(attention)?;
        let gate = params.narrow(1, 2 * WIDTH, WIDTH)?.unsqueeze(1)?;
        let x = if self.image { add_product(x, &gate, &projected)? } else { (x + projected.broadcast_mul(&gate)?)? };
        let mlp = self.down.forward(&self.up.forward(&Self::modulate(&x, params, 3)?)?.gelu()?)?;
        add_product(&x, &params.narrow(1, 5 * WIDTH, WIDTH)?.unsqueeze(1)?, &mlp)
    }
}

/// One independently loadable dual-stream block. Outputs are (image, text).
pub struct LayeredBlock {
    image: Stream,
    text: Stream,
}

impl LayeredBlock {
    pub fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self { image: Stream::load(false, vb.clone())?, text: Stream::load(true, vb)? })
    }

    pub fn forward(&self, image: &Tensor, text: &Tensor, temb: &Tensor, rope: &LayeredRope) -> Result<(Tensor, Tensor)> {
        let activated = candle_nn::ops::silu(temb)?;
        let im = self.image.modulation.forward(&activated)?;
        let tm = self.text.modulation.forward(&activated)?;
        let (iq, ik, iv) = self.image.qkv(&Stream::modulate(image, &im, 0)?)?;
        let (tq, tk, tv) = self.text.qkv(&Stream::modulate(text, &tm, 0)?)?;
        let q = rope.apply(&Tensor::cat(&[tq, iq], 2)?)?;
        let k = rope.apply(&Tensor::cat(&[tk, ik], 2)?)?;
        let v = Tensor::cat(&[tv, iv], 2)?;
        let (b, _, s, _) = q.dims4()?;
        let attention = joint_attention(&q, &k, &v)?.reshape((b, s, WIDTH))?;
        let nt = text.dim(1)?;
        Ok((self.image.finish(image, &attention.narrow(1, nt, image.dim(1)?)?, &im)?, self.text.finish(text, &attention.narrow(1, 0, nt)?, &tm)?))
    }
}

/// Exact bidirectional attention; cap temporary scores by chunking queries.
fn joint_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    #[cfg(feature = "flash")]
    if q.device().is_cuda() && q.dtype() == DType::BF16 {
        return crate::flash::attention(q, k, v, scale as f32);
    }
    let keys = k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
    let values = v.to_dtype(DType::F32)?.contiguous()?;
    let n = q.dim(2)?;
    let mut out = Vec::new();
    for start in (0..n).step_by(128) {
        let query = q.narrow(2, start, 128.min(n - start))?.to_dtype(DType::F32)?.contiguous()?;
        let scores = (query.matmul(&keys)? * scale)?;
        out.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&values)?);
    }
    Ok(Tensor::cat(&out, 2)?.to_dtype(v.dtype())?.transpose(1, 2)?.contiguous()?)
}

/// Adaptive output normalization (scale then shift) and 64-channel patch velocity.
pub struct LayeredOutput {
    modulation: LayeredLinear,
    projection: LayeredLinear,
}

impl LayeredOutput {
    pub fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            modulation: LayeredLinear::load(WIDTH, WIDTH * 2, vb.pp("norm_out.linear"))?,
            projection: LayeredLinear::load(WIDTH, 64, vb.pp("proj_out"))?,
        })
    }

    pub fn normalize(&self, image: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let modulation = self.modulation.forward(&candle_nn::ops::silu(temb)?)?;
        let scale = (modulation.narrow(1, 0, WIDTH)? + 1.0)?.unsqueeze(1)?;
        let shift = modulation.narrow(1, WIDTH, WIDTH)?.unsqueeze(1)?;
        add_product(&shift, &layer_norm(image)?, &scale)
    }

    pub fn forward(&self, image: &Tensor, temb: &Tensor) -> Result<Tensor> {
        self.projection.forward(&self.normalize(image, temb)?)
    }
}

/// Layered checkpoint backbone, distinct from Qwen-Image-2.1's single-stream DiT.
pub struct LayeredDit {
    pub input: LayeredInput,
    blocks: Vec<LayeredBlock>,
    pub output: LayeredOutput,
    streamed: Option<VarBuilder<'static>>,
}

impl LayeredDit {
    pub fn load(vb: VarBuilder<'_>) -> Result<Self> {
        let blocks = (0..BLOCKS).map(|i| LayeredBlock::load(vb.pp(format!("transformer_blocks.{i}")))).collect::<Result<Vec<_>>>()?;
        Ok(Self { input: LayeredInput::load(vb.clone())?, blocks, output: LayeredOutput::load(vb)?, streamed: None })
    }

    /// Every block resident, read from the checkpoint as `open_streamed` reads it (fp8 kept
    /// until each matmul): about 20 GB on the GPU, for a worker that splits more than once.
    pub fn open_resident(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        Self::load(checkpoint_builder(path, device)?)
    }

    /// Keep only one block resident at a time, including when running on CPU.
    pub fn open_streamed(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let vb = checkpoint_builder(path, device)?;
        Ok(Self { input: LayeredInput::load(vb.clone())?, blocks: Vec::new(), output: LayeredOutput::load(vb.clone())?, streamed: Some(vb) })
    }

    /// One CFG branch. Image tokens include trailing references; slice them off afterwards.
    /// Positions are (B,S,3); the optional additional condition defaults to entry zero.
    pub fn forward(
        &self,
        img_tokens: &Tensor,
        txt_tokens: &Tensor,
        timestep: &Tensor,
        img_positions: &Tensor,
        txt_positions: &Tensor,
        additional_t_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (mut img, mut txt) = self.input.project(img_tokens, txt_tokens)?;
        let temb = self.input.timestep(timestep, additional_t_cond)?;
        let rope = LayeredRope::new(&Tensor::cat(&[txt_positions, img_positions], 1)?, img.dtype())?;
        for block in &self.blocks {
            (img, txt) = block.forward(&img, &txt, &temb, &rope)?;
        }
        if let Some(vb) = &self.streamed {
            for i in 0..BLOCKS {
                let block = LayeredBlock::load(vb.pp(format!("transformer_blocks.{i}")))?;
                (img, txt) = block.forward(&img, &txt, &temb, &rope)?;
            }
        }
        self.output.forward(&img, &temb)
    }
}
