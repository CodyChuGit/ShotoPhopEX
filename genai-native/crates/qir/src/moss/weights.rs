//! The tensors of MOSS-Image's ComfyUI files (Comfy-Org/MOSS-Image): plain bf16/f32 tensors
//! through a memory-mapped [`VarBuilder`], and the `int8_tensorwise` ConvRot linears through
//! [`Int8File`] as they are on disk (no requantization). A quantized linear runs as ConvRot
//! W8A8 on CUDA (the `convrot` feature, bf16), and is dequantized to a dense weight anywhere
//! else (the CPU tests, a build without the kernels).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use safetensors::tensor::{Dtype, Metadata};

use crate::comfy_int8::{self, Prequantized};
use crate::model::rotation::{regular_hadamard_256, rotate, GROUP};
use crate::Result;

/// The INT8 ConvRot modules of a ComfyUI file, read one at a time from a mapping of it. Unlike
/// [`crate::comfy_int8::Int8File`] it indexes a file whose other parts use rotation groups qir
/// has no kernel for (Qwen3.5's vision tower: 64 and 16): those modules are listed but refuse
/// to load, and nothing here reads them.
struct Int8Index {
    map: memmap2::Mmap,
    metadata: Metadata,
    data_start: usize,
    /// Module → `[out, in]`, for the 256-group modules.
    modules: BTreeMap<String, (usize, usize)>,
    /// Module → its rotation group, for the rest.
    other: BTreeMap<String, usize>,
    /// The `asym_w4a8_int8` modules among `modules`, with their group size: 4-bit codes that
    /// decode to the same INT8 grid (comfy_kitchen's `_dequant_int4_grouped_to_int8`).
    w4a8: BTreeMap<String, usize>,
}

#[derive(serde::Deserialize)]
struct Descriptor {
    format: String,
    #[serde(default)]
    convrot: bool,
    #[serde(default)]
    convrot_groupsize: usize,
    #[serde(default)]
    group_size: Option<usize>,
}

/// The descriptor of the INT8 grid a W4A8 module decodes to.
const INT8_DESCRIPTOR: &[u8] = br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#;

/// An e4m3fn byte as f32 (bias 7, no infinities, `0x7f`/`0xff` NaN).
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xf);
    let m = f32::from(b & 0x7);
    if e == 0xf && b & 0x7 == 0x7 {
        return f32::NAN;
    }
    if e == 0 {
        sign * m / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// W4A8 storage to its INT8 grid in ComfyUI's basis, `(n, k)` bytes: each code through the
/// codebook (or `code - 8` without one) times its group's scale, rounded half to even and
/// clamped to ±127, as comfy_kitchen decodes it for the GEMM.
pub fn w4a8_to_int8(packed: &[u8], n: usize, k: usize, group: usize, s_rel: &[f32], codebook: Option<&[f32; 16]>) -> Result<Vec<u8>> {
    anyhow::ensure!(k.is_multiple_of(2) && k.is_multiple_of(group) && group > 0, "W4A8: K={k} with group {group}");
    anyhow::ensure!(packed.len() == n * k / 2 && s_rel.len() == n * (k / group), "W4A8: storage does not match ({n}, {k})");
    let mut out = vec![0u8; n * k];
    for (r, row) in out.chunks_exact_mut(k).enumerate() {
        let src = &packed[r * k / 2..(r + 1) * k / 2];
        let scales = &s_rel[r * (k / group)..(r + 1) * (k / group)];
        for (c, v) in row.iter_mut().enumerate() {
            let byte = src[c / 2];
            let code = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
            let level = match codebook {
                Some(cb) => cb[usize::from(code)],
                None => f32::from(code) - 8.0,
            };
            let q = (level * scales[c / group]).round_ties_even().clamp(-127.0, 127.0);
            *v = (q as i8) as u8;
        }
    }
    Ok(out)
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
        // SAFETY: an installed checkpoint stays unchanged while it is loaded (as for candle's
        // mapped VarBuilder of the same file); every read is bounds-checked against the map.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let mut this = Self { map, metadata, data_start: usize::try_from(8 + len)?, modules: BTreeMap::new(), other: BTreeMap::new(), w4a8: BTreeMap::new() };
        let names: Vec<String> = this.metadata.tensors().keys().filter_map(|n| n.strip_suffix(".comfy_quant").map(str::to_owned)).collect();
        for name in names {
            let raw = this.read(&format!("{name}.comfy_quant"))?;
            anyhow::ensure!(raw.len() <= 4096, "{name}: comfy_quant descriptor too long");
            let d: Descriptor = serde_json::from_slice(raw).with_context(|| format!("{name}: comfy_quant descriptor"))?;
            let w4a8 = d.format == "asym_w4a8_int8";
            anyhow::ensure!((d.format == "int8_tensorwise" || w4a8) && d.convrot, "{name}: {} (only int8_tensorwise and asym_w4a8_int8 ConvRot are read)", d.format);
            if d.convrot_groupsize != GROUP {
                this.other.insert(name, d.convrot_groupsize);
                continue;
            }
            let w = this.metadata.info(&format!("{name}.weight")).with_context(|| format!("{name}.weight missing"))?;
            anyhow::ensure!(w.dtype == Dtype::I8 && w.shape.len() == 2, "{name}.weight must be a 2-D I8 matrix");
            let k = if w4a8 { w.shape[1] * 2 } else { w.shape[1] };
            if w4a8 {
                this.w4a8.insert(name.clone(), d.group_size.unwrap_or(16));
            }
            this.modules.insert(name, (w.shape[0], k));
        }
        Ok(this)
    }

    fn read(&self, name: &str) -> Result<&[u8]> {
        let (start, end) = self.metadata.info(name).with_context(|| format!("missing tensor {name}"))?.data_offsets;
        let range = self.data_start.checked_add(start).zip(self.data_start.checked_add(end)).with_context(|| format!("{name}: offsets overflow"))?;
        self.map.get(range.0..range.1).with_context(|| format!("{name}: payload outside the file"))
    }

    /// `module` with its output rows in `rows`' order ([`comfy_int8::convert_rows`]).
    fn load(&self, module: &str, rows: Option<&[usize]>) -> Result<Prequantized> {
        if let Some(g) = self.other.get(module) {
            anyhow::bail!("{module}: ConvRot group {g} (qir's kernels take {GROUP})");
        }
        let &(n, k) = self.modules.get(module).with_context(|| format!("{module}: not an INT8 module"))?;
        let f32s = |name: &str| -> Result<Vec<f32>> { Ok(self.read(name)?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()) };
        if let Some(&group) = self.w4a8.get(module) {
            let s_rel: Vec<f32> = match self.metadata.info(&format!("{module}.weight_s_rel")).map(|i| i.dtype) {
                Some(Dtype::F32) => f32s(&format!("{module}.weight_s_rel"))?,
                _ => self.read(&format!("{module}.weight_s_rel"))?.iter().map(|&b| e4m3(b)).collect(),
            };
            let codebook = if self.metadata.info(&format!("{module}.weight_codebook")).is_some() {
                let v = f32s(&format!("{module}.weight_codebook"))?;
                Some(<[f32; 16]>::try_from(v.as_slice()).map_err(|_| anyhow::anyhow!("{module}: the codebook must have 16 levels"))?)
            } else {
                None
            };
            let grid = w4a8_to_int8(self.read(&format!("{module}.weight"))?, n, k, group, &s_rel, codebook.as_ref()).with_context(|| format!("{module}: W4A8"))?;
            let scale = f32s(&format!("{module}.weight_s_channel"))?;
            return comfy_int8::convert_rows(INT8_DESCRIPTOR, &[n, k], &grid, &[n, 1], &scale, rows).with_context(|| format!("{module}: W4A8 ConvRot"));
        }
        let descriptor = self.read(&format!("{module}.comfy_quant"))?;
        let weight = self.read(&format!("{module}.weight"))?;
        let scale = f32s(&format!("{module}.weight_scale"))?;
        comfy_int8::convert_rows(descriptor, &[n, k], weight, &[n, 1], &scale, rows).with_context(|| format!("{module}: INT8 ConvRot"))
    }

    fn contains(&self, module: &str) -> bool {
        self.modules.contains_key(module) || self.other.contains_key(module)
    }
}

/// One safetensors file of the model, plain and quantized tensors alike.
pub struct Weights {
    vb: VarBuilder<'static>,
    /// The file's header (shapes and dtypes, no data).
    meta: Metadata,
    int8: Option<Int8Index>,
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
        let meta = int8.metadata.clone();
        let int8 = if int8.modules.is_empty() && int8.other.is_empty() { None } else { Some(int8) };
        let convrot = convrot && cfg!(feature = "convrot") && dev.is_cuda() && dtype == DType::BF16;
        Ok(Self { vb, meta, int8, rot: regular_hadamard_256(dev)?, dev: dev.clone(), dtype, convrot })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// A tensor's shape from the header (nothing is read).
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.meta.info(name).with_context(|| format!("missing tensor {name}"))?.shape.clone())
    }

    /// A linear's output features (`<module>.weight`'s rows, any storage).
    pub fn out_features(&self, module: &str) -> Result<usize> {
        Ok(self.shape(&format!("{module}.weight"))?[0])
    }

    /// The file's `__metadata__` value for `key`.
    pub fn file_metadata(&self, key: &str) -> Option<String> {
        self.meta.metadata().as_ref().and_then(|m| m.get(key).cloned())
    }

    /// The tensor names in the file.
    pub fn names(&self) -> Vec<String> {
        self.meta.tensors().keys().cloned().collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.vb.contains_tensor(name) || self.is_quantized(name.strip_suffix(".weight").unwrap_or(name))
    }

    /// Whether `module` (a name without `.weight`) is an INT8 ConvRot triple.
    pub fn is_quantized(&self, module: &str) -> bool {
        self.int8.as_ref().is_some_and(|f| f.contains(module))
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
        self.prequantized_rows(module, None)
    }

    /// [`Self::prequantized`] with its output rows in `rows`' order.
    fn prequantized_rows(&self, module: &str, rows: Option<&[usize]>) -> Result<Prequantized> {
        self.int8.as_ref().with_context(|| format!("{module}: the file has no INT8 layers"))?.load(module, rows)
    }

    /// The linear `module` (`.weight`, optional `.bias`): ConvRot INT8 or dense.
    pub fn linear(&self, module: &str) -> Result<QLinear> {
        self.linear_rows(module, None)
    }

    /// [`Self::linear`] with its output rows reordered: row `i` of the result is the file's row
    /// `rows[i]` (a reordering of the output channels, exact for INT8 and dense alike).
    pub fn linear_rows(&self, module: &str, rows: Option<&[usize]>) -> Result<QLinear> {
        let mut bias = if self.vb.contains_tensor(&format!("{module}.bias")) { Some(self.get(&format!("{module}.bias"))?) } else { None };
        let index = match rows {
            Some(r) => Some(Tensor::from_vec(r.iter().map(|&i| i as u32).collect::<Vec<_>>(), r.len(), &self.dev)?),
            None => None,
        };
        if let (Some(b), Some(ix)) = (&bias, &index) {
            bias = Some(b.index_select(ix, 0)?);
        }
        if self.is_quantized(module) {
            let p = self.prequantized_rows(module, rows)?;
            #[cfg(feature = "convrot")]
            if self.convrot {
                let (q, s) = p.tensors(&self.dev)?;
                return Ok(QLinear::Convrot(crate::convrot::ConvRotLinear::from_prequantized(q, s, &self.rot)?, bias));
            }
            let w = dequantize(&p)?.to_dtype(self.dtype)?.to_device(&self.dev)?;
            return Ok(QLinear::Dense(candle_nn::Linear::new(w, bias)));
        }
        let w = self.get(&format!("{module}.weight"))?;
        let w = match &index {
            Some(ix) => w.index_select(ix, 0)?,
            None => w,
        };
        Ok(QLinear::Dense(candle_nn::Linear::new(w, bias)))
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
    /// `self(situ(hidden, gate))` for the merged `hg (M, 2F)` (hidden first), the activation
    /// rotated and quantized straight from `hg` in one kernel when this is a ConvRot linear;
    /// `None` otherwise (the caller applies SiTU itself).
    pub fn forward_situ(&self, hg: &Tensor) -> Result<Option<Tensor>> {
        match self {
            #[cfg(feature = "convrot")]
            Self::Convrot(l, None) => {
                let f = hg.dim(1)? / 2;
                Ok(Some(l.forward_situ(&hg.narrow(1, f, f)?, &hg.narrow(1, 0, f)?, crate::convrot::EpilogueOut::Bf16)?))
            }
            _ => {
                let _ = hg;
                Ok(None)
            }
        }
    }

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
