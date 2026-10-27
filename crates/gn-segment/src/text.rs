//! SAM 3.1 CLIP token features, kept in f32 through the language resizer for parity.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{Embedding, LayerNorm, Linear, Module};

use crate::weights::split_qkv;

const CONTEXT: usize = 32;
const WIDTH: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;

/// The fixed-context CLIP encoder and language projection, without pooled CLIP output.
/// All parameters and intermediate tensors use f32 on the caller's Candle device.
pub struct TextEncoder {
    tokens: Embedding,
    positions: Tensor,
    blocks: Vec<Block>,
    final_norm: LayerNorm,
    resizer: Linear,
}

impl TextEncoder {
    /// Load raw SAM checkpoint names and cast the selected parameters to f32.
    /// Buffered loading avoids unsafe memory mapping and does not upload unrelated weights.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let loader = Loader { tensors: BufferedSafetensors::new(bytes)?, device };
        let lang = "detector.backbone.language_backbone";
        let encoder = format!("{lang}.encoder");
        let tokens = Embedding::new(loader.get(&format!("{encoder}.token_embedding.weight"), &[49408, WIDTH])?, WIDTH);
        let positions = loader.get(&format!("{encoder}.positional_embedding"), &[CONTEXT, WIDTH])?;
        let blocks = (0..24).map(|i| Block::load(&loader, &format!("{encoder}.transformer.resblocks.{i}"))).collect::<Result<_>>()?;
        Ok(Self {
            tokens,
            positions,
            blocks,
            final_norm: loader.norm(&format!("{encoder}.ln_final"))?,
            resizer: loader.linear(&format!("{lang}.resizer"), WIDTH, 256)?,
        })
    }

    /// Encode one already-tokenized prompt as `[1,32,256]` and its unchanged validity mask.
    /// Mask entries mark valid keys; padded queries still produce features, as in ComfyUI.
    pub fn encode(&self, ids: &[u32; CONTEXT], mask: &[bool; CONTEXT]) -> Result<(Tensor, [bool; CONTEXT])> {
        ensure!(ids.iter().all(|&id| id < 49408), "CLIP token id exceeds vocabulary");
        let device = self.positions.device();
        let ids = Tensor::from_slice(ids, (1, CONTEXT), device)?;
        let mut x = self.tokens.forward(&ids)?.broadcast_add(&self.positions)?;
        // ComfyUI adds separate finite padding and causal biases; their overlap is -inf.
        let bias: Vec<f32> =
            (0..CONTEXT).flat_map(|q| (0..CONTEXT).map(move |k| if mask[k] { 0.0 } else { -f32::MAX } + if k > q { -f32::MAX } else { 0.0 })).collect();
        let bias = Tensor::from_vec(bias, (1, 1, CONTEXT, CONTEXT), device)?;
        for block in &self.blocks {
            x = block.forward(&x, &bias)?;
        }
        Ok((self.resizer.forward(&self.final_norm.forward(&x)?)?, *mask))
    }
}

struct Block {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    fn load(loader: &Loader<'_>, p: &str) -> Result<Self> {
        let (qw, kw, vw) = split_qkv(&loader.get(&format!("{p}.attn.in_proj_weight"), &[3 * WIDTH, WIDTH])?)?;
        let (qb, kb, vb) = split_qkv(&loader.get(&format!("{p}.attn.in_proj_bias"), &[3 * WIDTH])?)?;
        Ok(Self {
            norm1: loader.norm(&format!("{p}.ln_1"))?,
            q: Linear::new(qw.contiguous()?, Some(qb)),
            k: Linear::new(kw.contiguous()?, Some(kb)),
            v: Linear::new(vw.contiguous()?, Some(vb)),
            out: loader.linear(&format!("{p}.attn.out_proj"), WIDTH, WIDTH)?,
            norm2: loader.norm(&format!("{p}.ln_2"))?,
            fc1: loader.linear(&format!("{p}.mlp.c_fc"), WIDTH, 4096)?,
            fc2: loader.linear(&format!("{p}.mlp.c_proj"), 4096, WIDTH)?,
        })
    }

    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let normalized = self.norm1.forward(x)?;
        let heads = |projection: &Linear| -> candle_core::Result<Tensor> {
            projection.forward(&normalized)?.reshape((1, CONTEXT, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (heads(&self.q)?, heads(&self.k)?, heads(&self.v)?);
        let scores = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (HEAD_DIM as f64).sqrt()))?.broadcast_add(bias)?;
        let attention = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let attention = attention.transpose(1, 2)?.contiguous()?.reshape((1, CONTEXT, WIDTH))?;
        let x = (x + self.out.forward(&attention)?)?;
        let hidden = self.fc1.forward(&self.norm2.forward(&x)?)?;
        let activated = (&hidden * candle_nn::ops::sigmoid(&(&hidden * 1.702)?)?)?;
        Ok((&x + self.fc2.forward(&activated)?)?)
    }
}

struct Loader<'a> {
    tensors: BufferedSafetensors,
    device: &'a Device,
}

impl Loader<'_> {
    fn get(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self.tensors.load(key, self.device).with_context(|| format!("load {key}"))?;
        ensure!(tensor.dims() == shape, "{key}: expected {shape:?}, got {:?}", tensor.dims());
        Ok(tensor.to_dtype(DType::F32)?)
    }

    fn linear(&self, p: &str, input: usize, output: usize) -> Result<Linear> {
        Ok(Linear::new(self.get(&format!("{p}.weight"), &[output, input])?, Some(self.get(&format!("{p}.bias"), &[output])?)))
    }

    fn norm(&self, p: &str) -> Result<LayerNorm> {
        Ok(LayerNorm::new(self.get(&format!("{p}.weight"), &[WIDTH])?, self.get(&format!("{p}.bias"), &[WIDTH])?, 1e-5))
    }
}
