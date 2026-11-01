//! Q8_0 quantization on every CPU core. candle's `QTensor::quantize` of a CUDA tensor copies it
//! to the host as f32 and quantizes it on one thread: the chat head (151,936 × 4,096, the text
//! encoder's `lm_head`) took 10.2 s to load that way, on every first Remove after another model
//! family had the card (measured in PhotoshopEX 2026-11-01). Here the weight is read on the CPU,
//! split by rows across the cores, each chunk quantized with candle's own `BlockQ8_0::from_float`
//! (so the blocks are the bytes candle makes), and uploaded once.

use candle_core::quantized::k_quants::BlockQ8_0;
use candle_core::quantized::{GgmlDType, GgmlType, QStorage, QTensor};
use candle_core::{DType, Device, Tensor};

use crate::Result;

/// `w` (a 2-D tensor on the CPU, any float dtype) as a Q8_0 [`QTensor`] on `dev`, identical to
/// `QTensor::quantize(w, GgmlDType::Q8_0)`.
pub fn quantize_q8_0(w: &Tensor, dev: &Device) -> Result<QTensor> {
    let (rows, cols) = w.dims2()?;
    anyhow::ensure!(cols.is_multiple_of(BlockQ8_0::BLCK_SIZE), "Q8_0 needs rows of whole {}-value blocks, got {cols}", BlockQ8_0::BLCK_SIZE);
    let w = w.to_device(&Device::Cpu)?;
    let per_row = cols / BlockQ8_0::BLCK_SIZE;
    let mut blocks = vec![BlockQ8_0::zeros(); rows * per_row];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 32);
    let rows_per = rows.div_ceil(threads).max(1);
    if w.dtype() == DType::BF16 {
        // The checkpoint's bf16 as it is; each worker widens its own rows (exactly, as the
        // f32 cast does) one at a time.
        let src = w.flatten_all()?.to_vec1::<half::bf16>()?;
        std::thread::scope(|s| {
            for (from, to) in src.chunks(rows_per * cols).zip(blocks.chunks_mut(rows_per * per_row)) {
                s.spawn(move || {
                    let mut row = vec![0f32; cols];
                    for (r, b) in from.chunks_exact(cols).zip(to.chunks_exact_mut(per_row)) {
                        for (o, v) in row.iter_mut().zip(r) {
                            *o = v.to_f32();
                        }
                        BlockQ8_0::from_float(&row, b);
                    }
                });
            }
        });
    } else {
        let src = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        std::thread::scope(|s| {
            for (from, to) in src.chunks(rows_per * cols).zip(blocks.chunks_mut(rows_per * per_row)) {
                s.spawn(move || BlockQ8_0::from_float(from, to));
            }
        });
    }
    let cpu = QTensor::new(QStorage::Cpu(Box::new(blocks)), (rows, cols))?;
    if dev.is_cpu() {
        return Ok(cpu);
    }
    Ok(QTensor::new(QStorage::from_data(cpu.data()?, dev, GgmlDType::Q8_0)?, (rows, cols))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blocks_are_candles() {
        // Rows of several blocks, more rows than threads, values over a wide range.
        let (rows, cols) = (97, 160);
        let v: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 13) as f32)).collect();
        let w = Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap().to_dtype(DType::BF16).unwrap();
        let ours = quantize_q8_0(&w, &Device::Cpu).unwrap();
        let theirs = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
        assert_eq!(ours.shape(), theirs.shape());
        assert_eq!(ours.data().unwrap(), theirs.data().unwrap());
    }

    #[test]
    fn f32_weights_take_the_same_blocks() {
        let (rows, cols) = (5, 64);
        let v: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 150.0) / 7.0).collect();
        let w = Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap();
        let ours = quantize_q8_0(&w, &Device::Cpu).unwrap();
        assert_eq!(ours.data().unwrap(), QTensor::quantize(&w, GgmlDType::Q8_0).unwrap().data().unwrap());
    }

    #[test]
    fn a_ragged_row_is_refused() {
        let w = Tensor::zeros((4, 33), DType::F32, &Device::Cpu).unwrap();
        assert!(quantize_q8_0(&w, &Device::Cpu).is_err());
    }
}
