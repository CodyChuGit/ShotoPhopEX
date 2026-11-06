//! Compare adapter GEMMs with the original contiguous transposes.
use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use clap::Parser;
use qwen_image_rs::lora::{LoraFile, LoraPair, LoraSlot};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    lora: PathBuf,
    #[arg(long, default_value_t = 3844)]
    rows: usize,
    #[arg(long, default_value_t = 30)]
    iterations: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = qwen_image_rs::device::best_device()?;
    let file = LoraFile::load(&args.lora, &dev, DType::BF16)?;
    let (name, entry) = file.entries.iter().max_by_key(|(_, e)| e.a.elem_count() + e.b.elem_count()).context("no adapters")?;
    let k = entry.a.dim(1)?;
    let x = Tensor::randn(0f32, 1f32, (args.rows, k), &dev)?.to_dtype(DType::BF16)?;
    let at = entry.a.t()?.contiguous()?;
    let bt = entry.b.t()?.contiguous()?;
    let pair = LoraPair::new(&entry.a, &entry.b, 0.5)?;
    let base = Tensor::randn(0f32, 1f32, (args.rows, entry.b.dim(0)?), &dev)?.to_dtype(DType::BF16)?;
    let slot = LoraSlot::default();
    slot.set(vec![pair.clone()]);
    let old = || -> Result<Tensor> { Ok((x.matmul(&at)?.matmul(&bt)? * 0.5)?) };
    let new = || pair.delta(&x);
    let old_add = || -> Result<Tensor> { Ok((&base + old()?)?) };
    let new_add = || slot.add_to(base.clone(), &x);
    for (label, f) in [("original", &old as &dyn Fn() -> Result<Tensor>), ("candidate", &new), ("original_add", &old_add), ("candidate_add", &new_add)] {
        for _ in 0..3 {
            let _ = f()?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        for _ in 0..args.iterations {
            let _ = f()?;
        }
        dev.synchronize()?;
        println!("{label}: module={name} rows={} ms={:.6}", args.rows, t.elapsed().as_secs_f64() * 1000. / args.iterations as f64);
    }
    let d = (old()?.to_dtype(DType::F32)? - new()?.to_dtype(DType::F32)?)?.abs()?;
    println!("parity mean={} max={}", d.mean_all()?.to_scalar::<f32>()?, d.max_all()?.to_scalar::<f32>()?);
    let d = (old_add()?.to_dtype(DType::F32)? - new_add()?.to_dtype(DType::F32)?)?.abs()?;
    println!("add parity mean={} max={}", d.mean_all()?.to_scalar::<f32>()?, d.max_all()?.to_scalar::<f32>()?);
    Ok(())
}
