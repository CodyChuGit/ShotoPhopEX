//! LoRA adapters as a runtime side path. For a linear `y = W x` an adapter adds `s · B (A x)`
//! with `A` `(r, in)`, `B` `(out, r)` in the model's dtype and `s = strength · alpha / r`. The
//! base weights stay as loaded (the INT8 ConvRot ones included), so adapters come and go per
//! request through a [`LoraSlot`] on each linear, several stack by addition, and a strength is
//! a number rather than a re-quantisation.
//!
//! Files: safetensors in the `lora_A` / `lora_B` layout (PEFT, diffusers, ai-toolkit: AusBoss's,
//! Viggle's, the BFS and AnyPaint files) or `lora_down` / `lora_up` (Kohya), an optional `alpha`
//! per module (absent: `alpha = r`, so `s = strength`), under the `diffusion_model.`,
//! `transformer.` or no prefix. ComfyUI's Qwen-Image-2.1 fuses `img_mlp.gate_layer` and
//! `img_mlp.proj` into one `img_mlp.gate_up`; a LoRA trained on that layout addresses the fused
//! rows, and is split here as ComfyUI's own loader does: the first half of `B`'s rows is the
//! gate, the second the proj (`comfy/lora.py`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use candle_core::{DType, Device, Tensor};

use crate::Result;

/// One adapter on one linear, ready to apply: `a_t` `(in, r)`, `b_t` `(r, out)`, the scale.
#[derive(Clone, Debug)]
pub struct LoraPair {
    a_t: Tensor,
    b_t: Tensor,
    scale: f32,
}

impl LoraPair {
    /// From `a` `(r, in)` and `b` `(out, r)`.
    pub fn new(a: &Tensor, b: &Tensor, scale: f32) -> Result<Self> {
        let (r, _in) = a.dims2()?;
        let (_out, rb) = b.dims2()?;
        anyhow::ensure!(r == rb, "a LoRA's A ({r}) and B ({rb}) disagree on the rank");
        Ok(Self { a_t: a.t()?.contiguous()?, b_t: b.t()?.contiguous()?, scale })
    }

    pub fn rank(&self) -> usize {
        self.a_t.dim(1).unwrap_or(0)
    }

    pub fn in_features(&self) -> usize {
        self.a_t.dim(0).unwrap_or(0)
    }

    pub fn out_features(&self) -> usize {
        self.b_t.dim(1).unwrap_or(0)
    }

    /// `x` `(…, in)` → `s · (x Aᵀ) Bᵀ` `(…, out)`, in `x`'s dtype.
    pub fn delta(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().ok_or_else(|| anyhow::anyhow!("a LoRA delta needs a 1-D or wider input"))?;
        anyhow::ensure!(k == self.in_features(), "a LoRA for {} inputs applied to {k}", self.in_features());
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, k))?;
        let (a_t, b_t) = if x2.dtype() == self.a_t.dtype() { (self.a_t.clone(), self.b_t.clone()) } else { (self.a_t.to_dtype(x2.dtype())?, self.b_t.to_dtype(x2.dtype())?) };
        let y = (x2.matmul(&a_t)?.matmul(&b_t)? * f64::from(self.scale))?;
        let mut out = dims[..dims.len() - 1].to_vec();
        out.push(self.out_features());
        Ok(y.reshape(out)?)
    }
}

/// The adapters on one linear, settable between forwards.
#[derive(Default, Debug)]
pub struct LoraSlot(RwLock<Vec<LoraPair>>);

impl LoraSlot {
    pub fn set(&self, pairs: Vec<LoraPair>) {
        *self.0.write().unwrap_or_else(|e| e.into_inner()) = pairs;
    }

    pub fn clear(&self) {
        self.set(Vec::new());
    }

    pub fn is_empty(&self) -> bool {
        self.0.read().map(|v| v.is_empty()).unwrap_or(true)
    }

    /// The summed delta of the adapters for `x`, or `None` without any.
    pub fn delta(&self, x: &Tensor) -> Result<Option<Tensor>> {
        let pairs = self.0.read().unwrap_or_else(|e| e.into_inner());
        let mut acc: Option<Tensor> = None;
        for p in pairs.iter() {
            let d = p.delta(x)?;
            acc = Some(match acc {
                Some(a) => (a + d)?,
                None => d,
            });
        }
        Ok(acc)
    }

    /// `base + delta(x)` (`base` as it is without adapters). `base` may be a view: the sum is a
    /// fresh tensor; a dtype of `base` other than `x`'s is kept.
    pub fn add_to(&self, base: Tensor, x: &Tensor) -> Result<Tensor> {
        match self.delta(x)? {
            Some(d) => {
                let d = if d.dtype() == base.dtype() { d } else { d.to_dtype(base.dtype())? };
                Ok((base + d)?)
            }
            None => Ok(base),
        }
    }
}

/// `base` (any shape with `x`'s leading dims) plus the slot's delta for `x`, reshaped onto
/// `base`; `base` itself when the slot is empty (a view stays a view).
pub fn with_lora(slot: &LoraSlot, base: Tensor, x: &Tensor) -> Result<Tensor> {
    match slot.delta(x)? {
        None => Ok(base),
        Some(d) => {
            let d = d.reshape(base.dims())?;
            let d = if d.dtype() == base.dtype() { d } else { d.to_dtype(base.dtype())? };
            Ok((base.contiguous()? + d)?)
        }
    }
}

/// One module's adapter as the file holds it.
#[derive(Clone, Debug)]
pub struct Entry {
    /// `(r, in)`.
    pub a: Tensor,
    /// `(out, r)`.
    pub b: Tensor,
    /// The file's `alpha`, or `r`.
    pub alpha: f32,
}

impl Entry {
    pub fn rank(&self) -> usize {
        self.a.dim(0).unwrap_or(0)
    }
}

/// A LoRA file on a device: its modules by the base model's names (`transformer_blocks.3.attn.to_q`,
/// `blocks.0.attn.wq`, `txtfusion.layerwise_blocks.1.mlp.down`, …).
#[derive(Debug)]
pub struct LoraFile {
    pub path: PathBuf,
    pub entries: BTreeMap<String, Entry>,
}

const PREFIXES: [&str; 3] = ["diffusion_model.", "transformer.", "lora_unet_"];
const A_SUFFIXES: [&str; 2] = [".lora_A.weight", ".lora_down.weight"];
const B_SUFFIXES: [&str; 2] = [".lora_B.weight", ".lora_up.weight"];

fn strip_prefix(name: &str) -> &str {
    for p in PREFIXES {
        if let Some(rest) = name.strip_prefix(p) {
            return rest;
        }
    }
    name
}

/// The module a tensor belongs to and which part it is.
fn classify(name: &str) -> Option<(&str, Part)> {
    for s in A_SUFFIXES {
        if let Some(m) = name.strip_suffix(s) {
            return Some((m, Part::A));
        }
    }
    for s in B_SUFFIXES {
        if let Some(m) = name.strip_suffix(s) {
            return Some((m, Part::B));
        }
    }
    name.strip_suffix(".alpha").map(|m| (m, Part::Alpha))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    A,
    B,
    Alpha,
}

impl LoraFile {
    /// Read `path` onto `dev` in `dtype`; tensors that are not a LoRA's are ignored.
    pub fn load(path: &Path, dev: &Device, dtype: DType) -> Result<Self> {
        let raw = candle_core::safetensors::load(path, dev).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        let mut parts: BTreeMap<String, (Option<Tensor>, Option<Tensor>, Option<f32>)> = BTreeMap::new();
        for (name, t) in raw {
            let Some((module, part)) = classify(strip_prefix(&name)) else { continue };
            let e = parts.entry(module.to_string()).or_default();
            match part {
                Part::A => e.0 = Some(t.to_dtype(dtype)?),
                Part::B => e.1 = Some(t.to_dtype(dtype)?),
                Part::Alpha => e.2 = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.first().copied(),
            }
        }
        let mut entries = BTreeMap::new();
        for (module, (a, b, alpha)) in parts {
            let (Some(a), Some(b)) = (a, b) else {
                anyhow::bail!("{}: `{module}` has only half of its LoRA", path.display());
            };
            let (r, _) = a.dims2().map_err(|e| anyhow::anyhow!("{}: `{module}`: A is not a matrix: {e}", path.display()))?;
            let (out, rb) = b.dims2().map_err(|e| anyhow::anyhow!("{}: `{module}`: B is not a matrix: {e}", path.display()))?;
            anyhow::ensure!(r == rb, "{}: `{module}`: A has rank {r}, B {rb}", path.display());
            let alpha = alpha.unwrap_or(r as f32);
            // ComfyUI's fused gate|up of Qwen-Image-2.1: the gate rows first, then the proj rows.
            if let Some(stem) = module.strip_suffix(".img_mlp.gate_up") {
                anyhow::ensure!(out.is_multiple_of(2), "{}: `{module}`: {out} rows cannot be two halves", path.display());
                let half = out / 2;
                entries.insert(format!("{stem}.img_mlp.gate_layer"), Entry { a: a.clone(), b: b.narrow(0, 0, half)?.contiguous()?, alpha });
                entries.insert(format!("{stem}.img_mlp.proj"), Entry { a, b: b.narrow(0, half, half)?.contiguous()?, alpha });
            } else {
                entries.insert(module, Entry { a, b, alpha });
            }
        }
        anyhow::ensure!(!entries.is_empty(), "{}: no LoRA tensors (lora_A/lora_B or lora_down/lora_up)", path.display());
        Ok(Self { path: path.to_path_buf(), entries })
    }

    /// The module names, for a loader to match against a model's.
    pub fn modules(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
}

/// The adapters of a request: files with their strengths, in order.
#[derive(Clone, Default)]
pub struct LoraStack {
    pub layers: Vec<(Arc<LoraFile>, f32)>,
}

impl std::fmt::Debug for LoraStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut l = f.debug_list();
        for (file, s) in &self.layers {
            l.entry(&format_args!("{} @ {s}", file.path.display()));
        }
        l.finish()
    }
}

impl LoraStack {
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The pairs for the linear named `module`, one per file that has it, scaled by
    /// `strength · alpha / r`.
    pub fn pairs(&self, module: &str) -> Result<Vec<LoraPair>> {
        let mut out = Vec::new();
        for (file, strength) in &self.layers {
            if let Some(e) = file.entries.get(module) {
                let r = e.rank().max(1) as f32;
                out.push(LoraPair::new(&e.a, &e.b, strength * e.alpha / r)?);
            }
        }
        Ok(out)
    }

    /// The modules the stack names that `known` does not: a file for another model, or a layout
    /// this loader does not map yet.
    pub fn unmatched<'a>(&'a self, known: &dyn Fn(&str) -> bool) -> Vec<&'a str> {
        let mut v: Vec<&str> = self.layers.iter().flat_map(|(f, _)| f.modules()).filter(|m| !known(m)).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(dims: (usize, usize), seed: u64) -> Tensor {
        let (a, b) = dims;
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let v: Vec<f32> = (0..a * b)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect();
        Tensor::from_vec(v, dims, &Device::Cpu).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn the_side_path_equals_the_merged_weight() {
        let (inn, out, r) = (32, 24, 4);
        let w = rand((out, inn), 1);
        let a = rand((r, inn), 2);
        let b = rand((out, r), 3);
        let x = rand((5, inn), 4);
        let scale = 0.75f32;
        let merged = (&w + (b.matmul(&a).unwrap() * f64::from(scale)).unwrap()).unwrap();
        let want = x.matmul(&merged.t().unwrap()).unwrap();
        let slot = LoraSlot::default();
        slot.set(vec![LoraPair::new(&a, &b, scale).unwrap()]);
        let got = slot.add_to(x.matmul(&w.t().unwrap()).unwrap(), &x).unwrap();
        let diff = (got - want).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-4, "{diff}");
        // A 3-D input keeps its leading dims; an empty slot passes the base through.
        let x3 = rand((2 * 3, inn), 5).reshape((2, 3, inn)).unwrap();
        let d = slot.delta(&x3).unwrap().unwrap();
        assert_eq!(d.dims(), &[2, 3, out]);
        slot.clear();
        assert!(slot.delta(&x3).unwrap().is_none());
    }

    #[test]
    fn a_file_in_either_layout_loads_and_the_fused_gate_up_splits() {
        use candle_core::safetensors::save;
        let dir = std::env::temp_dir().join(format!("qir-lora-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.safetensors");
        let a = rand((4, 16), 1);
        let b = rand((8, 4), 2);
        let gate_up_b = rand((12, 4), 3);
        let alpha = Tensor::new(2.0f32, &Device::Cpu).unwrap();
        let mut m: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        m.insert("diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight".into(), a.clone());
        m.insert("diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight".into(), b.clone());
        m.insert("diffusion_model.transformer_blocks.0.attn.to_q.alpha".into(), alpha);
        m.insert("transformer.transformer_blocks.0.img_mlp.gate_up.lora_A.weight".into(), a.clone());
        m.insert("transformer.transformer_blocks.0.img_mlp.gate_up.lora_B.weight".into(), gate_up_b.clone());
        m.insert("blocks.1.attn.wq.lora_down.weight".into(), a.clone());
        m.insert("blocks.1.attn.wq.lora_up.weight".into(), b.clone());
        m.insert("not_a_lora.weight".into(), a.clone());
        save(&m, &p).unwrap();
        let f = LoraFile::load(&p, &Device::Cpu, DType::F32).unwrap();
        let names: Vec<&str> = f.modules().collect();
        assert_eq!(
            names,
            vec!["blocks.1.attn.wq", "transformer_blocks.0.attn.to_q", "transformer_blocks.0.img_mlp.gate_layer", "transformer_blocks.0.img_mlp.proj"]
        );
        assert_eq!(f.entries["transformer_blocks.0.attn.to_q"].alpha, 2.0);
        assert_eq!(f.entries["blocks.1.attn.wq"].alpha, 4.0, "no alpha: the rank");
        let gate = &f.entries["transformer_blocks.0.img_mlp.gate_layer"].b;
        let proj = &f.entries["transformer_blocks.0.img_mlp.proj"].b;
        assert_eq!((gate.dims(), proj.dims()), (&[6, 4][..], &[6, 4][..]));
        let first = gate_up_b.narrow(0, 0, 6).unwrap();
        assert_eq!(gate.to_vec2::<f32>().unwrap(), first.to_vec2::<f32>().unwrap(), "the gate is the first half");
        // The stack scales by strength · alpha / r.
        let stack = LoraStack { layers: vec![(Arc::new(f), 0.5)] };
        let pairs = stack.pairs("transformer_blocks.0.attn.to_q").unwrap();
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].scale - 0.25).abs() < 1e-6, "{}", pairs[0].scale);
        assert!(stack.pairs("nowhere").unwrap().is_empty());
        assert_eq!(stack.unmatched(&|m| m.starts_with("transformer_blocks")), vec!["blocks.1.attn.wq"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
