//! A ComfyUI file to MLX arrays, one tensor at a time: qir reads it (its INT8 ConvRot triples
//! as the file has them, its plain tensors through the map), the GPU dequantizes (bytes times
//! row scales, then qir's group rotation undone), adds any LoRA delta in f32, rounds to f16 and
//! quantizes the large linears to 4-bit. Converted weights save to and load from an MLX
//! safetensors cache.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use mlx_rs::{Array, Dtype, ops};
use qwen_image_rs::lora::LoraFile;
use qwen_image_rs::model::rotation::{GROUP as ROT_GROUP, regular_hadamard_256};
use qwen_image_rs::moss::weights::Weights;

use crate::nn::{GROUP, Params, dim, from_candle};

/// How the large linears are held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Dense f16 (the reference for the 4-bit path; twice the memory of the file's INT8).
    F16,
    /// MLX 4-bit, group 64, f16 scales and biases.
    Q4,
    /// MLX 8-bit, group 64.
    Q8,
    /// MLX 6-bit and 5-bit, group 64.
    Q6,
    Q5,
}

impl Precision {
    pub fn tag(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q4 => "q4g64",
            Self::Q8 => "q8g64",
            Self::Q6 => "q6g64",
            Self::Q5 => "q5g64",
        }
    }

    /// The bits of a quantized linear (`None`: dense).
    pub fn bits(self) -> Option<i32> {
        match self {
            Self::F16 => None,
            Self::Q4 => Some(4),
            Self::Q8 => Some(8),
            Self::Q6 => Some(6),
            Self::Q5 => Some(5),
        }
    }
}

/// A LoRA merged into the weights at load: `W + strength·α/r·B A`.
pub struct Lora {
    file: LoraFile,
    strength: f32,
}

impl Lora {
    pub fn open(path: &Path, strength: f32) -> Result<Self> {
        Ok(Self { file: LoraFile::load(path, &Device::Cpu, DType::F32).with_context(|| format!("the LoRA {}", path.display()))?, strength })
    }

    /// The modules the file adapts.
    pub fn modules(&self) -> impl Iterator<Item = &str> {
        self.file.entries.keys().map(String::as_str)
    }

    /// `module`'s pair `(A (r, in), B·strength·α/r·k (out, r), 1/k)` with A and B in f16, if the
    /// file adapts it. `k` is the power of two that brings B's largest value near 1: a few-step
    /// LoRA's B is mostly below f16's smallest normal number (6e-5), and an edit LoRA's, scaled
    /// by a fixed large factor, overflowed f16 in the product.
    pub fn pair(&self, module: &str) -> Result<Option<(Array, Array, f32)>> {
        let Some(e) = self.file.entries.get(module) else { return Ok(None) };
        let r = e.a.dim(0)?;
        let b = from_candle(&e.b)?.multiply(Array::from_f32(self.strength * e.alpha / r.max(1) as f32))?;
        let peak = b.abs()?.max(None)?.as_dtype(Dtype::Float32)?;
        peak.eval()?;
        let peak = peak.item_exact::<f32>();
        let k = pow2_scale(peak);
        let a = from_candle(&e.a)?.as_dtype(Dtype::Float16)?;
        let b = b.multiply(Array::from_f32(k))?.as_dtype(Dtype::Float16)?;
        mlx_rs::transforms::eval([&a, &b])?;
        Ok(Some((a, b, 1.0 / k)))
    }

    /// `module`'s delta `(out, in)` f32, if the file adapts it.
    pub fn delta(&self, module: &str) -> Result<Option<Array>> {
        let Some(e) = self.file.entries.get(module) else { return Ok(None) };
        let r = e.a.dim(0)?;
        let scale = self.strength * e.alpha / r.max(1) as f32;
        Ok(Some(from_candle(&e.b)?.matmul(&from_candle(&e.a)?)?.multiply(Array::from_f32(scale))?))
    }
}

/// The power of two that brings `peak` to at most 1 (and above 1/2), within 1/64..65536; 1 for a
/// zero or non-finite peak.
fn pow2_scale(peak: f32) -> f32 {
    if !(peak.is_finite() && peak > 0.0) {
        return 1.0;
    }
    2f32.powi(-(peak.log2().ceil() as i32)).clamp(1.0 / 64.0, 65536.0)
}

/// Reads one file's tensors into [`Params`].
pub struct Converter<'a> {
    w: &'a Weights,
    /// qir's rotation `R (256, 256)` f32.
    rot: Array,
    pub out: Params,
    pub dtype: Dtype,
    pub precision: Precision,
}

impl<'a> Converter<'a> {
    /// `w` opened in f32 on the CPU (qir's reading; nothing is computed there).
    pub fn new(w: &'a Weights, dtype: Dtype, precision: Precision) -> Result<Self> {
        Ok(Self { w, rot: from_candle(&regular_hadamard_256(&Device::Cpu)?)?, out: Params::new(), dtype, precision })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.w.contains(name)
    }

    /// `module`'s weight `(out, in)` f32 on the GPU, dequantized if the file has it in INT8.
    pub fn dense(&self, module: &str) -> Result<Array> {
        if !self.w.is_quantized(module) {
            return from_candle(&self.w.get_as(&format!("{module}.weight"), DType::F32)?);
        }
        let p = self.w.prequantized(module)?;
        anyhow::ensure!(p.in_features.is_multiple_of(ROT_GROUP), "{module}: {} inputs (rotation groups of {ROT_GROUP})", p.in_features);
        anyhow::ensure!(p.weight_i8.len() == p.out_features * p.in_features && p.col_scale.len() == p.out_features, "{module}: invalid INT8 storage");
        let (n, k) = (dim(p.out_features)?, dim(p.in_features)?);
        let bytes: Vec<i8> = p.weight_i8.iter().map(|&b| b as i8).collect();
        let w = Array::from_slice(&bytes, &[n, k]).as_dtype(Dtype::Float32)?.multiply(Array::from_slice(&p.col_scale, &[n, 1]))?;
        // `W Rᵀ` group-wise: qir's R is symmetric and orthogonal, its own inverse.
        let g = dim(ROT_GROUP)?;
        Ok(w.reshape(&[n * (k / g), g])?.matmul(self.rot.t())?.reshape(&[n, k])?)
    }

    /// A plain tensor (a norm's weight, a bias) in the run dtype.
    pub fn tensor(&mut self, name: &str) -> Result<()> {
        let t = from_candle(&self.w.get_as(name, DType::F32)?)?.as_dtype(self.dtype)?;
        t.eval()?;
        self.out.insert(name.to_string(), t);
        Ok(())
    }

    /// A plain tensor in f32 under `name` (state the models keep in f32).
    pub fn tensor_f32(&mut self, name: &str) -> Result<()> {
        let t = from_candle(&self.w.get_as(name, DType::F32)?)?;
        t.eval()?;
        self.out.insert(name.to_string(), t);
        Ok(())
    }

    /// The linear `module` with its bias (if any): 4-bit when `quantize` and the precision
    /// asks for it, else dense in the run dtype. `delta` (a LoRA's, f32) is added first.
    pub fn linear(&mut self, module: &str, quantize: bool, delta: Option<Array>) -> Result<()> {
        let w = self.dense(module)?;
        self.put_linear(module, w, quantize, delta)
    }

    /// [`Self::linear`] from a weight already in hand.
    pub fn put_linear(&mut self, module: &str, w: Array, quantize: bool, delta: Option<Array>) -> Result<()> {
        let w = match delta {
            Some(d) => {
                anyhow::ensure!(d.shape() == w.shape(), "{module}: a LoRA delta of {:?} for a weight of {:?}", d.shape(), w.shape());
                w.add(&d)?
            }
            None => w,
        };
        let w = w.as_dtype(self.dtype)?;
        let k = *w.shape().last().unwrap_or(&0);
        if let (true, Some(bits), 0) = (quantize, self.precision.bits(), k % GROUP) {
            let (q, s, z) = ops::quantize(&w, GROUP, bits)?;
            mlx_rs::transforms::eval([&q, &s, &z])?;
            self.out.insert(format!("{module}.weight"), q);
            self.out.insert(format!("{module}.scales"), s);
            self.out.insert(format!("{module}.biases"), z);
        } else {
            w.eval()?;
            self.out.insert(format!("{module}.weight"), w);
        }
        let bias = format!("{module}.bias");
        if self.w.contains(&bias) {
            self.tensor(&bias)?;
        }
        Ok(())
    }
}

/// Save converted weights (written beside and renamed into place, so a crash leaves no half file).
pub fn save(p: &Params, path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let tmp = path.with_extension("partial.safetensors");
    Array::save_safetensors(p.iter().map(|(k, v)| (k.as_str(), v)), None, &tmp).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
    Ok(())
}

/// Load converted weights, every array read before returning.
pub fn load(path: &Path) -> Result<Params> {
    let p = Array::load_safetensors(path).with_context(|| format!("read {}", path.display()))?;
    mlx_rs::transforms::eval(p.values())?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::pow2_scale;

    /// A few-step LoRA's tiny B is lifted, an edit LoRA's B is not blown past f16 (it was, at a
    /// fixed 4096: black pictures).
    #[test]
    fn lora_b_scale_keeps_the_peak_near_one() {
        for peak in [1e-4f32, 3e-3, 0.02, 0.5, 1.0, 3.0] {
            let scaled = peak * pow2_scale(peak);
            assert!(scaled <= 1.0 && scaled > 0.5, "{peak} -> {scaled}");
        }
        assert_eq!(pow2_scale(0.0), 1.0);
        assert_eq!(pow2_scale(f32::NAN), 1.0);
        assert_eq!(pow2_scale(1e-12), 65536.0);
    }
}
