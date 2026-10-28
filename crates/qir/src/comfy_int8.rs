//! ComfyUI INT8 ConvRot import without dequantizing or requantizing weights.
//! Reverse each 256-column group to match qir's activation rotation basis.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{ensure, Context};
use candle_core::{Device, Tensor};
use safetensors::tensor::{Dtype, Metadata, TensorInfo};
use serde::Deserialize;

use crate::model::rotation::{regular_hadamard_256, rotate, GROUP};
use crate::Result;

const MAX_HEADER: u64 = 16 * 1024 * 1024;
const MAX_DESCRIPTOR: usize = 4096;

/// The supported descriptor; reject additional fields rather than guessing their semantics.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    format: String,
    convrot: bool,
    convrot_groupsize: usize,
}

impl Descriptor {
    /// Decode bounded JSON and require the exact rotation/quantization contract.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(!bytes.is_empty() && bytes.len() <= MAX_DESCRIPTOR, "comfy_quant: descriptor length must be 1..={MAX_DESCRIPTOR}");
        let d: Self = serde_json::from_slice(bytes).context("comfy_quant: invalid or unknown descriptor")?;
        ensure!(d.format == "int8_tensorwise", "comfy_quant: unsupported format {:?}", d.format);
        ensure!(d.convrot, "comfy_quant: convrot must be true");
        ensure!(d.convrot_groupsize == GROUP, "comfy_quant: unsupported convrot_groupsize {} (need {GROUP})", d.convrot_groupsize);
        Ok(d)
    }
}

fn shape(weight: &[usize], scale: &[usize]) -> Result<(usize, usize)> {
    let [n, k] = weight else { anyhow::bail!("INT8 weight must have shape [out,in], got {weight:?}") };
    ensure!(*n > 0 && *k > 0 && k.is_multiple_of(GROUP), "INT8 weight needs out > 0 and in a positive multiple of {GROUP}, got {weight:?}");
    ensure!(scale == [*n, 1], "weight_scale must have shape [{n},1], got {scale:?}");
    Ok((*n, *k))
}

/// qir's prequantized representation, with signed bytes carried unchanged in U8.
pub struct Prequantized {
    pub weight_i8: Vec<u8>,
    pub col_scale: Vec<f32>,
    pub out_features: usize,
    pub in_features: usize,
}

/// Adapt one module in place: only permute bytes, preserving quantization and row scales.
pub fn convert(descriptor: &[u8], weight_shape: &[usize], mut weight: Vec<u8>, scale_shape: &[usize], scale: Vec<f32>) -> Result<Prequantized> {
    Descriptor::decode(descriptor)?;
    let (n, k) = shape(weight_shape, scale_shape)?;
    ensure!(n.checked_mul(k) == Some(weight.len()), "INT8 weight byte count does not match {weight_shape:?}");
    ensure!(scale.len() == n, "weight_scale value count does not match {scale_shape:?}");
    ensure!(scale.iter().all(|s| s.is_finite() && *s > 0.0), "weight_scale values must be finite and positive");
    // H_comfy[i,j] = R_qir[255-i,j], hence W_qir = W_comfy P.
    // P is its own inverse; no sign changes or floating-point operations are needed.
    for group in weight.as_chunks_mut::<GROUP>().0 {
        group.reverse();
    }
    Ok(Prequantized { weight_i8: weight, col_scale: scale, out_features: n, in_features: k })
}

impl Prequantized {
    /// Create U8/F32 tensors for `ConvRotLinear::from_prequantized`, without another rotation.
    pub fn tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        Ok((
            Tensor::from_slice(&self.weight_i8, (self.out_features, self.in_features), device)?,
            Tensor::from_slice(&self.col_scale, self.out_features, device)?,
        ))
    }

    /// Gather and unrotate only requested embedding rows, avoiding a full floating table.
    /// Returns `[indices.len(), in_features]` F32 on CPU; cast after inverse rotation.
    pub fn gather_cpu(&self, indices: &[u32]) -> Result<Tensor> {
        shape(&[self.out_features, self.in_features], &[self.out_features, 1])?;
        ensure!(
            self.out_features.checked_mul(self.in_features) == Some(self.weight_i8.len()) && self.col_scale.len() == self.out_features,
            "invalid prequantized storage"
        );
        let count = indices.len().checked_mul(self.in_features).context("embedding gather size overflow")?;
        let mut values = Vec::with_capacity(count);
        for &index in indices {
            let row = index as usize;
            ensure!(row < self.out_features, "embedding index {row} outside vocabulary {}", self.out_features);
            let start = row * self.in_features;
            values.extend(self.weight_i8[start..start + self.in_features].iter().map(|&b| (b as i8) as f32 * self.col_scale[row]));
        }
        let x = Tensor::from_vec(values, (indices.len(), self.in_features), &Device::Cpu)?;
        if indices.is_empty() {
            return Ok(x);
        }
        // qir's R is symmetric and orthogonal: the inverse is the same group rotation.
        rotate(&x, &regular_hadamard_256(&Device::Cpu)?)
    }
}

/// Header index that reads only bounded descriptors until a module is explicitly loaded.
pub struct Int8File {
    file: File,
    metadata: Metadata,
    data_start: u64,
    modules: BTreeMap<String, (usize, usize)>,
}

impl Int8File {
    /// Validate every quantized triple using headers and small `comfy_quant` payloads only.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes).context("safetensors header length")?;
        let len = u64::from_le_bytes(bytes);
        ensure!(len <= MAX_HEADER && len <= file_len.saturating_sub(8), "invalid safetensors header length {len}");
        let mut header = vec![0; usize::try_from(len)?];
        file.read_exact(&mut header)?;
        // Metadata validates offsets, dtype sizes, shape products and contiguous ranges.
        let metadata: Metadata = serde_json::from_slice(&header).context("invalid safetensors header")?;
        let data_start = 8 + len;
        ensure!(metadata.data_len() as u64 == file_len - data_start, "safetensors data length disagrees with header");
        let mut this = Self { file, metadata, data_start, modules: BTreeMap::new() };
        let mut names: Vec<String> = this.metadata.tensors().keys().filter_map(|n| n.strip_suffix(".comfy_quant").map(str::to_owned)).collect();
        names.sort();
        for name in names {
            let result = (|| -> Result<(usize, usize)> {
                let descriptor = this.info(&format!("{name}.comfy_quant"))?;
                ensure!(
                    descriptor.dtype == Dtype::U8 && descriptor.shape.len() == 1 && descriptor.shape[0] <= MAX_DESCRIPTOR,
                    "comfy_quant must be bounded 1-D U8"
                );
                Descriptor::decode(&this.read(&format!("{name}.comfy_quant"))?)?;
                let weight = this.info(&format!("{name}.weight"))?;
                let scale = this.info(&format!("{name}.weight_scale"))?;
                ensure!(weight.dtype == Dtype::I8, "weight must be I8, got {:?}", weight.dtype);
                ensure!(scale.dtype == Dtype::F32, "weight_scale must be F32, got {:?}", scale.dtype);
                shape(&weight.shape, &scale.shape)
            })();
            this.modules.insert(name.clone(), result.with_context(|| format!("{name}: INT8 ConvRot"))?);
        }
        for (name, info) in this.metadata.tensors() {
            if info.dtype == Dtype::I8 || name.ends_with(".weight_scale") {
                let module = name.strip_suffix(".weight").or_else(|| name.strip_suffix(".weight_scale"));
                ensure!(module.is_some_and(|n| this.modules.contains_key(n)), "{name}: missing comfy_quant triple");
            }
        }
        Ok(this)
    }

    /// Validated module names and `[out,in]` dimensions, for mixed-precision dispatch.
    pub fn modules(&self) -> &BTreeMap<String, (usize, usize)> {
        &self.modules
    }

    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.metadata.info(name).with_context(|| format!("missing tensor {name}"))
    }

    fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        let (start, end) = self.info(name)?.data_offsets;
        self.file.seek(SeekFrom::Start(self.data_start + start as u64))?;
        let mut bytes = vec![0; end - start];
        self.file.read_exact(&mut bytes).with_context(|| format!("read {name}"))?;
        Ok(bytes)
    }

    /// Read one signed-byte matrix and scales; adapt without a full-model allocation.
    pub fn load(&mut self, module: &str) -> Result<Prequantized> {
        let &(n, k) = self.modules.get(module).with_context(|| format!("{module}: not a validated INT8 module"))?;
        let descriptor = self.read(&format!("{module}.comfy_quant"))?;
        let weight = self.read(&format!("{module}.weight"))?;
        let scale = self.read(&format!("{module}.weight_scale"))?;
        let scale = scale.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        convert(&descriptor, &[n, k], weight, &[n, 1], scale).with_context(|| format!("{module}: convert INT8 ConvRot"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::process::Command;

    const DESCRIPTOR: &[u8] = br#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#;
    const PYTHON: &str = r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe";
    const MODELS: &str = r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\models";

    fn comfy_matrix() -> Vec<f32> {
        // int8_utils._build_hadamard's H4, Kronecker-expanded exactly as in that code.
        let h4 = [[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]];
        let mut matrix = vec![0.; GROUP * GROUP];
        for i in 0..GROUP {
            for j in 0..GROUP {
                let (mut a, mut b, mut v) = (i, j, 1. / 16.);
                for _ in 0..4 {
                    v *= h4[a % 4][b % 4];
                    a /= 4;
                    b /= 4;
                }
                matrix[i * GROUP + j] = v;
            }
        }
        matrix
    }

    #[test]
    fn reversal_is_exact_for_both_code_bases() -> Result<()> {
        let comfy = comfy_matrix();
        let qir = regular_hadamard_256(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        for i in 0..GROUP {
            for j in 0..GROUP {
                assert_eq!(comfy[i * GROUP + j], qir[(GROUP - 1 - i) * GROUP + j]);
                assert_eq!(comfy[i * GROUP + j], comfy[j * GROUP + i]);
            }
        }
        // All entries of H R^T are the reversal permutation, not merely a few rows.
        for i in 0..GROUP {
            for j in 0..GROUP {
                let dot: f32 = (0..GROUP).map(|k| comfy[i * GROUP + k] * qir[j * GROUP + k]).sum();
                assert_eq!(dot, if i + j == GROUP - 1 { 1. } else { 0. });
            }
        }
        Ok(())
    }

    #[test]
    fn descriptors_shapes_and_signed_bytes() -> Result<()> {
        assert_eq!(DESCRIPTOR.len(), 72);
        Descriptor::decode(DESCRIPTOR)?;
        for bad in [
            "{}",
            r#"{"format":"fp8","convrot":true,"convrot_groupsize":256}"#,
            r#"{"format":"int8_tensorwise","convrot":false,"convrot_groupsize":256}"#,
            r#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":128}"#,
            r#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256,"new_basis":true}"#,
            r#"{"format":"int8_tensorwise","convrot":1,"convrot_groupsize":256}"#,
            r#"{"format":"int8_tensorwise","format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#,
        ] {
            assert!(Descriptor::decode(bad.as_bytes()).is_err(), "accepted {bad}");
        }
        assert!(Descriptor::decode(&vec![b' '; MAX_DESCRIPTOR + 1]).is_err());
        assert!(Descriptor::decode(&[255]).is_err());
        for (w, s) in [(vec![2, 255], vec![2, 1]), (vec![2, 256], vec![2]), (vec![2, 256], vec![1, 2]), (vec![0, 256], vec![0, 1]), (vec![2, 0], vec![2, 1])] {
            assert!(shape(&w, &s).is_err());
        }
        for scales in [vec![0., 1.], vec![f32::NAN, 1.], vec![f32::INFINITY, 1.], vec![-1., 1.], vec![1.]] {
            assert!(convert(DESCRIPTOR, &[2, 256], vec![0; 512], &[2, 1], scales).is_err());
        }
        assert!(convert(DESCRIPTOR, &[2, 256], vec![0; 511], &[2, 1], vec![1.; 2]).is_err());
        let bytes: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let converted = convert(DESCRIPTOR, &[2, 512], bytes.clone(), &[2, 1], vec![0.25, 2.])?;
        assert_eq!(converted.col_scale, [0.25, 2.]);
        for (before, after) in bytes.as_chunks::<GROUP>().0.iter().zip(converted.weight_i8.as_chunks::<GROUP>().0.iter()) {
            assert!(before.iter().rev().eq(after.iter()));
        }
        assert_eq!(converted.weight_i8[0] as i8, -1);
        assert_eq!(converted.weight_i8[127] as i8, -128);
        let (w, s) = converted.tensors(&Device::Cpu)?;
        assert_eq!(w.dtype(), DType::U8);
        assert_eq!(w.dims(), [2, 512]);
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.dims(), [2]);
        assert_eq!(converted.gather_cpu(&[])?.dims(), [0, 512]);
        assert!(converted.gather_cpu(&[2]).is_err());
        Ok(())
    }

    #[derive(Deserialize)]
    struct Fixture {
        basis: Vec<f32>,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        n: usize,
        k: usize,
        q: Vec<u8>,
        scale: Vec<f32>,
        dequant: Vec<f32>,
        dequant_bf16: Vec<f32>,
        indices: Vec<u32>,
        embedding: Vec<f32>,
        embedding_bf16: Vec<f32>,
    }

    // Invoke the installed implementation, not a second quantizer approximating its BF16 math.
    const FIXTURE_SCRIPT: &str = r#"
import json, torch
from comfy_kitchen.backends.eager.quantization import (
    quantize_int8_convrot_weight, dequantize_int8_convrot_weight,
    dequantize_int8_convrot_weight_dtype, dequantize_int8_embedding)
from comfy_kitchen.tensor.int8_utils import _build_hadamard
torch.set_num_threads(2)
torch.manual_seed(419)
def flat(x): return x.flatten().tolist()
cases = []
for k in (256, 768):
    w = torch.randn((7,k), device='cpu').to(torch.bfloat16)
    w[0].zero_()
    w[3,17] = -31.5
    q, s = quantize_int8_convrot_weight(w, 256)
    indices = torch.tensor([[6,0],[3,6]], dtype=torch.int64)
    cases.append(dict(n=7, k=k, q=flat(q.view(torch.uint8)), scale=flat(s),
        dequant=flat(dequantize_int8_convrot_weight(q,s,256)),
        dequant_bf16=flat(dequantize_int8_convrot_weight_dtype(q,s,256,2).float()),
        indices=flat(indices), embedding=flat(dequantize_int8_embedding(q,s,indices,256,0)),
        embedding_bf16=flat(dequantize_int8_embedding(q,s,indices,256,2).float())))
print(json.dumps(dict(basis=flat(_build_hadamard(256)), cases=cases)))
"#;

    fn parity(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len());
        let max = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let bound = want.iter().map(|x| x.abs()).fold(1f32, f32::max) * 2e-6;
        eprintln!("{label}: max F32 error {max:e}, bound {bound:e}");
        assert!(max <= bound, "{label}: {max} > {bound}");
    }

    fn bf16_parity(got: &[f32], want: &[f32]) {
        for (&a, &b) in got.iter().zip(want) {
            let a = half::bf16::from_f32(a);
            let b = half::bf16::from_f32(b);
            // Permuting the reduction can straddle a BF16 midpoint, or change the sign
            // of a near-zero cancellation residual. Bound both sources of rounding.
            assert!(a.to_bits().abs_diff(b.to_bits()) <= 1 || (a.to_f32() - b.to_f32()).abs() <= 2e-6, "BF16 {a} vs {b}");
        }
    }

    #[test]
    fn comfy_kitchen_cpu_weight_and_embedding_parity() -> Result<()> {
        let python = std::env::var_os("COMFY_PYTHON").map(std::path::PathBuf::from).unwrap_or_else(|| PYTHON.into());
        if !python.is_file() {
            eprintln!("SKIP: ComfyUI embedded Python absent; set COMFY_PYTHON to run fixture parity");
            return Ok(());
        }
        let output = Command::new(python)
            .args(["-I", "-c", FIXTURE_SCRIPT])
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("OMP_NUM_THREADS", "2")
            .env("MKL_NUM_THREADS", "2")
            .output()?;
        ensure!(output.status.success(), "ComfyUI fixture generation failed: {}", String::from_utf8_lossy(&output.stderr));
        let fixture: Fixture = serde_json::from_slice(&output.stdout).context("ComfyUI fixture JSON")?;
        assert_eq!(fixture.basis, comfy_matrix());
        for case in fixture.cases {
            let p = convert(DESCRIPTOR, &[case.n, case.k], case.q.clone(), &[case.n, 1], case.scale.clone())?;
            let all = p.gather_cpu(&(0..case.n as u32).collect::<Vec<_>>())?.flatten_all()?.to_vec1::<f32>()?;
            parity(&all, &case.dequant, &format!("weight 7x{}", case.k));
            bf16_parity(&all, &case.dequant_bf16);
            let embedding = p.gather_cpu(&case.indices)?.flatten_all()?.to_vec1::<f32>()?;
            parity(&embedding, &case.embedding, &format!("embedding 2x2x{}", case.k));
            bf16_parity(&embedding, &case.embedding_bf16);
            assert_eq!(&embedding[..case.k], &embedding[3 * case.k..]); // duplicate indices
            assert!(embedding[case.k..2 * case.k].iter().all(|&v| v == 0.));
            // An omitted adapter and an accidental second adapter must both be detected.
            let double = convert(DESCRIPTOR, &[case.n, case.k], p.weight_i8, &[case.n, 1], case.scale)?;
            assert_eq!(double.weight_i8, case.q);
            let wrong = double.gather_cpu(&case.indices)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(wrong.iter().zip(&case.embedding).any(|(a, b)| (a - b).abs() > 0.1));
        }
        Ok(())
    }

    #[test]
    fn installed_files_header_and_descriptors_only() -> Result<()> {
        for (relative, count) in [("diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 192), ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", 254)]
        {
            let path = Path::new(MODELS).join(relative);
            if !path.is_file() {
                eprintln!("SKIP: {} absent", path.display());
                continue;
            }
            let file = Int8File::open(&path)?;
            assert_eq!(file.modules().len(), count);
            assert_eq!(file.metadata.tensors().values().filter(|t| t.dtype == Dtype::I8).count(), count);
            let names: Vec<_> = file.modules().keys().cloned().collect();
            for name in names {
                assert_eq!(file.info(&format!("{name}.comfy_quant"))?.shape, [72]);
            }
            eprintln!("{}: {count} validated triples; read header and {} descriptor bytes only", path.display(), 72 * count);
        }
        Ok(())
    }

    #[test]
    fn file_loading_and_malformed_triples() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        struct Temporary(std::path::PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        // Tiny real safetensors files exercise the same header-only path as installed files.
        let write = |entries: Vec<(&str, &str, Vec<usize>, Vec<u8>)>| -> Result<Temporary> {
            let mut header = serde_json::Map::new();
            let mut data = Vec::new();
            for (name, dtype, dims, bytes) in entries {
                let start = data.len();
                data.extend(bytes);
                header.insert(name.into(), serde_json::json!({"dtype":dtype,"shape":dims,"data_offsets":[start,data.len()]}));
            }
            let mut header = serde_json::to_vec(&header)?;
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend(data);
            let path =
                Temporary(std::env::temp_dir().join(format!("qir-comfy-int8-{}-{}.safetensors", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed))));
            std::fs::write(&path.0, bytes)?;
            Ok(path)
        };
        let entries = || {
            vec![
                ("layer.weight", "I8", vec![2, 512], (0..1024).map(|i| i as u8).collect()),
                ("layer.weight_scale", "F32", vec![2, 1], [0.25f32, 2.].into_iter().flat_map(f32::to_le_bytes).collect()),
                ("layer.comfy_quant", "U8", vec![72], DESCRIPTOR.to_vec()),
            ]
        };
        let path = write(entries())?;
        let mut file = Int8File::open(&path.0)?;
        assert_eq!(file.modules().get("layer"), Some(&(2, 512)));
        assert!(file.load("missing").is_err());
        let p = file.load("layer")?;
        assert_eq!(p.col_scale, [0.25, 2.]);
        assert_eq!(p.weight_i8[0], 255);
        assert_eq!(p.weight_i8[512], 255);
        drop(file);
        for (remove, message) in [(0, "missing tensor layer.weight"), (1, "missing tensor layer.weight_scale"), (2, "missing comfy_quant triple")] {
            let mut e = entries();
            e.remove(remove);
            let bad = write(e)?;
            let error = Int8File::open(&bad.0).err().context("accepted missing triple member")?;
            assert!(format!("{error:#}").contains(message), "{error:#}");
        }
        for (index, dtype, dims, message) in [
            (0, "U8", vec![2, 512], "weight must be I8"),
            (1, "I32", vec![2, 1], "weight_scale must be F32"),
            (1, "F32", vec![2], "weight_scale must have shape"),
            (2, "U8", vec![1, 72], "comfy_quant must be bounded 1-D U8"),
        ] {
            let mut e = entries();
            e[index].1 = dtype;
            e[index].2 = dims;
            let bad = write(e)?;
            let error = Int8File::open(&bad.0).err().context("accepted malformed triple")?;
            assert!(format!("{error:#}").contains(message), "{error:#}");
        }
        let mut e = entries();
        e[2].3 = vec![b'?'; 72];
        let bad = write(e)?;
        assert!(Int8File::open(&bad.0).is_err());
        let mut bytes = std::fs::read(&path.0)?;
        bytes.pop();
        std::fs::write(&path.0, &bytes)?;
        assert!(Int8File::open(&path.0).is_err());
        std::fs::write(&path.0, (MAX_HEADER + 1).to_le_bytes())?;
        assert!(Int8File::open(&path.0).is_err());
        Ok(())
    }
}
