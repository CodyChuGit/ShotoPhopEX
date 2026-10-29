//! The tensors of MOSS-Image's ComfyUI files (Comfy-Org/MOSS-Image): plain bf16/f32 tensors
//! through a memory-mapped [`VarBuilder`], and the `int8_tensorwise` ConvRot linears through
//! [`Int8File`] as they are on disk (no requantization). A quantized linear runs as ConvRot
//! W8A8 on CUDA (the `convrot` feature, bf16), and is dequantized to a dense weight anywhere
//! else (the CPU tests, a build without the kernels).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use safetensors::tensor::{Dtype, Metadata};

use crate::comfy_int8::{self, Prequantized};
use crate::model::rotation::{regular_hadamard_256, rotate, GROUP};
use crate::Result;

/// The INT8 ConvRot modules of a ComfyUI file, read one at a time. Unlike
/// [`crate::comfy_int8::Int8File`] it indexes a file whose other parts use rotation groups qir
/// has no kernel for (Qwen3.5's vision tower: 64 and 16): those modules are listed but refuse
/// to load, and nothing here reads them.
struct Int8Index {
    file: File,
    metadata: Metadata,
    data_start: u64,
    /// Module → `[out, in]`, for the 256-group modules.
    modules: BTreeMap<String, (usize, usize)>,
    /// Module → its rotation group, for the rest.
    other: BTreeMap<String, usize>,
}

#[derive(serde::Deserialize)]
struct Descriptor {
    format: String,
    #[serde(default)]
    convrot: bool,
    #[serde(default)]
    convrot_groupsize: usize,
}

impl Int8Index {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes).context("safetensors header length")?;
        let len = u64::from_le_bytes(bytes);
        anyhow::ensure!(len <= 64 * 1024 * 1024 && len <= file_len.saturating_sub(8), "invalid safetensors header length {len}");
        let mut header = vec![0; usize::try_from(len)?];
        file.read_exact(&mut header)?;
        let metadata: Metadata = serde_json::from_slice(&header).context("invalid safetensors header")?;
        let mut this = Self { file, metadata, data_start: 8 + len, modules: BTreeMap::new(), other: BTreeMap::new() };
        let names: Vec<String> = this.metadata.tensors().keys().filter_map(|n| n.strip_suffix(".comfy_quant").map(str::to_owned)).collect();
        for name in names {
            let raw = this.read(&format!("{name}.comfy_quant"))?;
            anyhow::ensure!(raw.len() <= 4096, "{name}: comfy_quant descriptor too long");
            let d: Descriptor = serde_json::from_slice(&raw).with_context(|| format!("{name}: comfy_quant descriptor"))?;
            anyhow::ensure!(d.format == "int8_tensorwise" && d.convrot, "{name}: {} (only int8_tensorwise ConvRot is read)", d.format);
            if d.convrot_groupsize != GROUP {
                this.other.insert(name, d.convrot_groupsize);
                continue;
            }
            let w = this.metadata.info(&format!("{name}.weight")).with_context(|| format!("{name}.weight missing"))?;
            anyhow::ensure!(w.dtype == Dtype::I8 && w.shape.len() == 2, "{name}.weight must be a 2-D I8 matrix");
            this.modules.insert(name, (w.shape[0], w.shape[1]));
        }
        Ok(this)
    }

    fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        let (start, end) = self.metadata.info(name).with_context(|| format!("missing tensor {name}"))?.data_offsets;
        self.file.seek(SeekFrom::Start(self.data_start + start as u64))?;
        let mut bytes = vec![0; end - start];
        self.file.read_exact(&mut bytes).with_context(|| format!("read {name}"))?;
        Ok(bytes)
    }

    fn load(&mut self, module: &str) -> Result<Prequantized> {
        if let Some(g) = self.other.get(module) {
            anyhow::bail!("{module}: ConvRot group {g} (qir's kernels take {GROUP})");
        }
        let &(n, k) = self.modules.get(module).with_context(|| format!("{module}: not an INT8 module"))?;
        let descriptor = self.read(&format!("{module}.comfy_quant"))?;
        let weight = self.read(&format!("{module}.weight"))?;
        let scale = self.read(&format!("{module}.weight_scale"))?;
        let scale = scale.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        comfy_int8::convert(&descriptor, &[n, k], weight, &[n, 1], scale).with_context(|| format!("{module}: INT8 ConvRot"))
    }

    fn contains(&self, module: &str) -> bool {
        self.modules.contains_key(module) || self.other.contains_key(module)
    }
}

/// One safetensors file of the model, plain and quantized tensors alike.
pub struct Weights {
    vb: VarBuilder<'static>,
    int8: Option<Mutex<Int8Index>>,
    /// qir's ConvRot rotation, on the device (f32).
    #[cfg_attr(not(feature = "convrot"), allow(dead_code))]
    rot: Tensor,
    dev: Device,
    dtype: DType,
    #[cfg_attr(not(feature = "convrot"), allow(dead_code))]
    convrot: bool,
}

impl Weights {
    /// Open `path` for running in `dtype` on `dev`; `convrot` asks for the INT8 kernels where
    /// the build and the device have them.
    pub fn open(path: &Path, dev: &Device, dtype: DType, convrot: bool) -> Result<Self> {
        let files: Vec<PathBuf> = vec![path.to_path_buf()];
        // The plain tensors are read through the map; the I8 ones never through it.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
        let int8 = Int8Index::open(path).with_context(|| format!("{}: its INT8 ConvRot layers", path.display()))?;
        let int8 = if int8.modules.is_empty() && int8.other.is_empty() { None } else { Some(Mutex::new(int8)) };
        let convrot = convrot && cfg!(feature = "convrot") && dev.is_cuda() && dtype == DType::BF16;
        Ok(Self { vb, int8, rot: regular_hadamard_256(dev)?, dev: dev.clone(), dtype, convrot })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn contains(&self, name: &str) -> bool {
        self.vb.contains_tensor(name) || self.is_quantized(name.strip_suffix(".weight").unwrap_or(name))
    }

    /// Whether `module` (a name without `.weight`) is an INT8 ConvRot triple.
    pub fn is_quantized(&self, module: &str) -> bool {
        self.int8.as_ref().is_some_and(|f| f.lock().is_ok_and(|f| f.contains(module)))
    }

    /// A plain tensor in the run dtype.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        Ok(self.vb.get_unchecked_dtype(name, self.dtype)?)
    }

    /// A plain tensor in `dtype`.
    pub fn get_as(&self, name: &str, dtype: DType) -> Result<Tensor> {
        Ok(self.vb.get_unchecked_dtype(name, dtype)?)
    }

    /// An INT8 module as the file has it (its groups already in qir's rotation basis).
    pub fn prequantized(&self, module: &str) -> Result<Prequantized> {
        let file = self.int8.as_ref().with_context(|| format!("{module}: the file has no INT8 layers"))?;
        let mut file = file.lock().map_err(|_| anyhow::anyhow!("INT8 file lock poisoned"))?;
        file.load(module)
    }

    /// The linear `module` (`.weight`, optional `.bias`): ConvRot INT8 or dense.
    pub fn linear(&self, module: &str) -> Result<QLinear> {
        let bias = if self.vb.contains_tensor(&format!("{module}.bias")) { Some(self.get(&format!("{module}.bias"))?) } else { None };
        if self.is_quantized(module) {
            let p = self.prequantized(module)?;
            #[cfg(feature = "convrot")]
            if self.convrot {
                let (q, s) = p.tensors(&self.dev)?;
                return Ok(QLinear::Convrot(crate::convrot::ConvRotLinear::from_prequantized(q, s, &self.rot)?, bias));
            }
            let w = dequantize(&p)?.to_dtype(self.dtype)?.to_device(&self.dev)?;
            return Ok(QLinear::Dense(candle_nn::Linear::new(w, bias)));
        }
        Ok(QLinear::Dense(candle_nn::Linear::new(self.get(&format!("{module}.weight"))?, bias)))
    }
}

/// `W` of an INT8 module, `(out, in)` f32 on the CPU: the rows' bytes times their scales, rotated
/// back (qir's rotation is symmetric and orthogonal, so it is its own inverse).
pub fn dequantize(p: &Prequantized) -> Result<Tensor> {
    let values: Vec<f32> = p.weight_i8.chunks(p.in_features).zip(&p.col_scale).flat_map(|(row, s)| row.iter().map(move |&b| (b as i8) as f32 * s)).collect();
    let w = Tensor::from_vec(values, (p.out_features, p.in_features), &Device::Cpu)?;
    rotate(&w, &regular_hadamard_256(&Device::Cpu)?)
}

/// A linear of the model: ConvRot W8A8 (bf16 in and out) or a dense weight.
pub enum QLinear {
    Dense(candle_nn::Linear),
    #[cfg(feature = "convrot")]
    Convrot(crate::convrot::ConvRotLinear, Option<Tensor>),
}

impl QLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => Ok(l.forward(x)?),
            #[cfg(feature = "convrot")]
            Self::Convrot(l, bias) => {
                let y = l.forward(&x.to_dtype(DType::BF16)?)?;
                Ok(match bias {
                    Some(b) => y.broadcast_add(b)?,
                    None => y,
                })
            }
        }
    }
}
