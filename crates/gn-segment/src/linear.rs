//! Bias-inclusive half GEMMs avoid an extra rounding between matrix products and bias.
use candle_core::{DType, Module, Result, Tensor};
pub(crate) struct Linear {
    inner: candle_nn::Linear,
    packed: Option<Tensor>,
}
impl Linear {
    pub(crate) fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let packed = if weight.device().is_cuda() && weight.dtype() == DType::F16 {
            if let Some(bias) = &bias {
                let output = weight.dim(0)?;
                Some(Tensor::cat(&[&weight, &bias.unsqueeze(1)?, &Tensor::zeros((output, 7), weight.dtype(), weight.device())?], 1)?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self { inner: candle_nn::Linear::new(weight, bias), packed })
    }
    /// Project NCHW channels without im2col or output transposes for 1x1 convolutions.
    pub(crate) fn forward_channels(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        if b != 1 {
            candle_core::bail!("SAM pointwise projection requires batch one");
        }
        let flat = x.reshape((c, h * w))?;
        let y = if let Some(weight) = &self.packed {
            let ones = Tensor::ones((1, h * w), x.dtype(), x.device())?;
            let zeros = Tensor::zeros((7, h * w), x.dtype(), x.device())?;
            weight.matmul(&Tensor::cat(&[flat, ones, zeros], 0)?)?
        } else {
            let y = self.inner.weight().matmul(&flat)?;
            if let Some(bias) = self.inner.bias() { y.broadcast_add(&bias.unsqueeze(1)?)? } else { y }
        };
        y.reshape((1, self.inner.weight().dim(0)?, h, w))
    }
    pub(crate) fn weight(&self) -> &Tensor {
        self.inner.weight()
    }
}
impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let Some(weight) = &self.packed else {
            return self.inner.forward(x);
        };
        let (output, input) = self.inner.weight().dims2()?;
        let rows = x.elem_count() / input;
        let ones = Tensor::ones((rows, 1), x.dtype(), x.device())?;
        let zeros = Tensor::zeros((rows, 7), x.dtype(), x.device())?;
        let x2 = Tensor::cat(&[&x.reshape((rows, input))?, &ones, &zeros], 1)?;
        let mut shape = x.dims().to_vec();
        let last = shape.last_mut().ok_or_else(|| candle_core::Error::Msg("linear requires channels".into()))?;
        *last = output;
        x2.matmul(&weight.t()?)?.reshape(shape)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn bias_inclusive_gemm_and_pointwise_keep_cancellation() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let x = Tensor::from_vec((0..3 * 16).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect::<Vec<_>>(), (3, 16), &device)?;
        let w = Tensor::from_vec((0..8 * 16).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect::<Vec<_>>(), (8, 16), &device)?;
        let bias = Tensor::from_slice(&[0.1_f32, -0.1, 0.02, -0.02, 0.3, -0.3, 0.01, -0.01], 8, &device)?.to_dtype(DType::F16)?;
        let half = Linear::new(w.to_dtype(DType::F16)?, Some(bias.clone()))?;
        let expected =
            x.matmul(&w.t()?)?.broadcast_add(&bias.to_dtype(DType::F32)?)?.to_dtype(DType::F16)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = half.forward(&x.to_dtype(DType::F16)?)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        let nchw = x.t()?.contiguous()?.reshape((1, 16, 1, 3))?.to_dtype(DType::F16)?;
        let channels = half.forward_channels(&nchw)?.reshape((8, 3))?.t()?.contiguous()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(channels, expected);
        Ok(())
    }
}
