//! Compare load-time weight kernels against installed Kitchen fixtures.
use anyhow::Result;
#[cfg(feature = "convrot")]
fn main() -> Result<()> {
    use anyhow::Context;
    use candle_core::{DType, Tensor};
    let dev = qwen_image_rs::device::best_device()?;
    let path = std::env::args().nth(1).context("fixture path")?;
    let f = candle_core::safetensors::load(path, &dev)?;
    for i in 0..3 {
        let t = |key: &str| -> Result<&Tensor> { f.get(&format!("{i}.{key}")).context("fixture tensor missing") };
        let seed = t("seed")?.to_device(&candle_core::Device::Cpu)?.to_vec1::<i64>()?[0] as u64;
        let w = qwen_image_rs::lora_materialize::dequantize(t("base_q")?, t("base_s")?)?;
        let error = (&w.to_dtype(DType::F32)? - t("base_w")?.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("case={i} dequant_max={error}");
        anyhow::ensure!(error == 0., "dequantization differs from Kitchen");
        if let (Some(a), Some(b)) = (f.get(&format!("{i}.a")), f.get(&format!("{i}.b"))) {
            let rotation = qwen_image_rs::model::rotation::regular_hadamard_256(&dev)?;
            let linear = qwen_image_rs::convrot::ConvRotLinear::from_prequantized(t("base_q")?.clone(), t("base_s")?.clone(), &rotation)?;
            let module = ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.img_mlp.out", "transformer_blocks.0.img_mlp.gate_up"][i];
            linear.set_loras(&[qwen_image_rs::lora::LoraPair::new(a, b, 0.5)?], module)?;
            let (q, s) = linear.export()?;
            let got = q.flatten_all()?.to_vec1::<u8>()?;
            let want = t("q")?.flatten_all()?.to_vec1::<u8>()?;
            let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            let scale_error = (&s - t("s")?)?.abs()?.max_all()?.to_scalar::<f32>()?;
            println!("case={i} integrated_q_mismatches={mismatches}/{} scale_max={scale_error}", got.len());
            anyhow::ensure!(mismatches == 0 && scale_error == 0., "integrated weight patch differs from Kitchen");
        }
        for rounded in [false, true] {
            let (q, s) = qwen_image_rs::lora_materialize::requantize(t("patched")?, seed, rounded)?;
            let got = q.flatten_all()?.to_vec1::<u8>()?;
            let want = t("q")?.flatten_all()?.to_vec1::<u8>()?;
            let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            let scale_error = (&s - t("s")?)?.abs()?.max_all()?.to_scalar::<f32>()?;
            println!("case={i} rounded={rounded} q_mismatches={mismatches}/{} scale_max={scale_error}", got.len());
            if !rounded {
                anyhow::ensure!(mismatches == 0 && scale_error == 0., "requantization differs from Kitchen");
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "convrot"))]
fn main() -> Result<()> {
    anyhow::bail!("lora_merge_check requires the convrot feature")
}
