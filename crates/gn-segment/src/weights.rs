//! Raw SAM 3.1 checkpoint names and shapes, before ComfyUI's loader rewrites them.

use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

use anyhow::{Context, Result, ensure};
use candle_core::Tensor;

/// A required tensor whose stored dimensions differ from the port's architecture.
#[derive(Debug, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub key: String,
    pub expected: Vec<usize>,
    pub actual: Vec<usize>,
}

/// Header-only validation results. Unused video tensors are allowed in the checkpoint.
#[derive(Debug, Default)]
pub struct Report {
    pub checked: usize,
    pub missing: Vec<String>,
    pub mismatches: Vec<ShapeMismatch>,
}

/// Validate required names and shapes without reading or mapping any tensor payload.
/// The header limit matches safetensors' 100 MB limit and bounds allocation on invalid input.
pub fn validate(path: impl AsRef<Path>) -> Result<Report> {
    let path = path.as_ref();
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut length = [0_u8; 8];
    file.read_exact(&mut length).context("read safetensors header length")?;
    let length = u64::from_le_bytes(length);
    ensure!(length <= 100_000_000, "safetensors header exceeds 100 MB");
    let mut bytes = vec![0; usize::try_from(length)?];
    file.read_exact(&mut bytes).context("read safetensors JSON header")?;
    let header: serde_json::Value = serde_json::from_slice(&bytes)?;
    let header = header.as_object().context("safetensors header must be an object")?;
    let mut report = Report::default();
    for (key, expected) in expected_shapes() {
        report.checked += 1;
        let Some(entry) = header.get(&key) else {
            report.missing.push(key);
            continue;
        };
        let actual: Vec<usize> =
            serde_json::from_value(entry.get("shape").context("tensor has no shape")?.clone()).with_context(|| format!("invalid shape for {key}"))?;
        if actual != expected {
            report.mismatches.push(ShapeMismatch { key, expected, actual });
        }
    }
    Ok(report)
}

/// Split either fused projection weights or biases along dimension zero in Q, K, V order.
/// Narrowing preserves the remaining dimensions, dtype and device without a cast.
pub fn split_qkv(fused: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    ensure!(matches!(fused.rank(), 1 | 2), "QKV must be a bias vector or weight matrix");
    let rows = fused.dim(0)?;
    ensure!(rows > 0 && rows % 3 == 0, "QKV dimension zero must be positive and divisible by three");
    let width = rows / 3;
    Ok((fused.narrow(0, 0, width)?, fused.narrow(0, width, width)?, fused.narrow(0, 2 * width, width)?))
}

/// Expand the architecture table into every raw checkpoint key used by the selected modules.
/// Complete geometry and segmentation modules are checked, including their dormant point,
/// third pixel-decoder and semantic heads. Propagation and video memory weights are omitted.
pub fn expected_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut t = Table::default();
    let vision = "detector.backbone.vision_backbone";
    let trunk = format!("{vision}.trunk");
    t.add(&format!("{trunk}.patch_embed.proj.weight"), &[1024, 3, 14, 14]);
    t.add(&format!("{trunk}.pos_embed"), &[1, 577, 1024]);
    t.norm(&format!("{trunk}.ln_pre"), 1024);
    for i in 0..32 {
        let p = format!("{trunk}.blocks.{i}");
        t.norm(&format!("{p}.norm1"), 1024);
        t.norm(&format!("{p}.norm2"), 1024);
        t.linear(&format!("{p}.attn.qkv"), 1024, 3072);
        t.linear(&format!("{p}.attn.proj"), 1024, 1024);
        t.linear(&format!("{p}.mlp.fc1"), 1024, 4736);
        t.linear(&format!("{p}.mlp.fc2"), 4736, 1024);
    }
    for fpn in ["convs", "interactive_convs"] {
        for (i, channels) in [256, 512, 1024].into_iter().enumerate() {
            let p = format!("{vision}.{fpn}.{i}");
            t.conv(&format!("{p}.conv_1x1"), &[256, channels, 1, 1], 256);
            t.conv(&format!("{p}.conv_3x3"), &[256, 256, 3, 3], 256);
            if i == 0 {
                t.conv(&format!("{p}.dconv_2x2_0"), &[1024, 512, 2, 2], 512);
                t.conv(&format!("{p}.dconv_2x2_1"), &[512, 256, 2, 2], 256);
            } else if i == 1 {
                t.conv(&format!("{p}.dconv_2x2"), &[1024, 512, 2, 2], 512);
            }
        }
    }
    let lang = "detector.backbone.language_backbone";
    t.add(&format!("{lang}.encoder.token_embedding.weight"), &[49408, 1024]);
    t.add(&format!("{lang}.encoder.positional_embedding"), &[32, 1024]);
    t.norm(&format!("{lang}.encoder.ln_final"), 1024);
    t.linear(&format!("{lang}.resizer"), 1024, 256);
    for i in 0..24 {
        let p = format!("{lang}.encoder.transformer.resblocks.{i}");
        t.fused_attention(&format!("{p}.attn"), 1024);
        t.norm(&format!("{p}.ln_1"), 1024);
        t.norm(&format!("{p}.ln_2"), 1024);
        t.linear(&format!("{p}.mlp.c_fc"), 1024, 4096);
        t.linear(&format!("{p}.mlp.c_proj"), 4096, 1024);
    }
    let geo = "detector.geometry_encoder";
    for (name, input) in [
        ("boxes_direct_project", 4),
        ("boxes_pos_enc_project", 258),
        ("points_direct_project", 2),
        ("points_pool_project", 256),
        ("points_pos_enc_project", 256),
        ("final_proj", 256),
    ] {
        t.linear(&format!("{geo}.{name}"), input, 256);
    }
    t.conv(&format!("{geo}.boxes_pool_project"), &[256, 256, 7, 7], 256);
    t.add(&format!("{geo}.label_embed.weight"), &[2, 256]);
    t.add(&format!("{geo}.cls_embed.weight"), &[1, 256]);
    for name in ["norm", "img_pre_norm", "encode_norm"] {
        t.norm(&format!("{geo}.{name}"), 256);
    }
    for (prefix, count) in [(format!("{geo}.encode"), 3), ("detector.transformer.encoder.layers".into(), 6)] {
        for i in 0..count {
            t.detector_layer(&format!("{prefix}.{i}"), &["self_attn", "cross_attn_image"], &["norm1", "norm2", "norm3"]);
        }
    }
    let dec = "detector.transformer.decoder";
    for i in 0..6 {
        t.detector_layer(&format!("{dec}.layers.{i}"), &["self_attn", "cross_attn", "ca_text"], &["norm1", "norm2", "norm3", "catext_norm"]);
    }
    t.add(&format!("{dec}.query_embed.weight"), &[200, 256]);
    t.add(&format!("{dec}.reference_points.weight"), &[200, 4]);
    t.add(&format!("{dec}.presence_token.weight"), &[1, 256]);
    t.norm(&format!("{dec}.norm"), 256);
    t.norm(&format!("{dec}.presence_token_out_norm"), 256);
    for (name, dims) in [
        ("ref_point_head", vec![512, 256, 256]),
        ("bbox_embed", vec![256, 256, 256, 4]),
        ("boxRPB_embed_x", vec![2, 256, 8]),
        ("boxRPB_embed_y", vec![2, 256, 8]),
        ("presence_token_head", vec![256, 256, 256, 1]),
    ] {
        t.mlp(&format!("{dec}.{name}.layers"), &dims);
    }
    let score = "detector.dot_prod_scoring";
    t.linear(&format!("{score}.hs_proj"), 256, 256);
    t.linear(&format!("{score}.prompt_proj"), 256, 256);
    t.mlp(&format!("{score}.prompt_mlp.layers"), &[256, 2048, 256]);
    t.norm(&format!("{score}.prompt_mlp.out_norm"), 256);
    let seg = "detector.segmentation_head";
    t.fused_attention(&format!("{seg}.cross_attend_prompt"), 256);
    t.norm(&format!("{seg}.cross_attn_norm"), 256);
    for i in 0..3 {
        t.conv(&format!("{seg}.pixel_decoder.conv_layers.{i}"), &[256, 256, 3, 3], 256);
        t.norm(&format!("{seg}.pixel_decoder.norms.{i}"), 256);
    }
    t.conv(&format!("{seg}.instance_seg_head"), &[256, 256, 1, 1], 256);
    t.conv(&format!("{seg}.semantic_seg_head"), &[1, 256, 1, 1], 1);
    t.mlp(&format!("{seg}.mask_predictor.mask_embed.layers"), &[256, 256, 256, 256]);
    let prompt = "tracker.model.interactive_sam_prompt_encoder";
    t.add(&format!("{prompt}.pe_layer.positional_encoding_gaussian_matrix"), &[2, 128]);
    for i in 0..4 {
        t.add(&format!("{prompt}.point_embeddings.{i}.weight"), &[1, 256]);
    }
    for name in ["not_a_point_embed", "no_mask_embed"] {
        t.add(&format!("{prompt}.{name}.weight"), &[1, 256]);
    }
    t.conv(&format!("{prompt}.mask_downscaling.0"), &[4, 1, 2, 2], 4);
    t.norm(&format!("{prompt}.mask_downscaling.1"), 4);
    t.conv(&format!("{prompt}.mask_downscaling.3"), &[16, 4, 2, 2], 16);
    t.norm(&format!("{prompt}.mask_downscaling.4"), 16);
    t.conv(&format!("{prompt}.mask_downscaling.6"), &[256, 16, 1, 1], 256);
    let mask = "tracker.model.interactive_sam_mask_decoder";
    for (name, count) in [("obj_score_token", 1), ("iou_token", 1), ("mask_tokens", 4)] {
        t.add(&format!("{mask}.{name}.weight"), &[count, 256]);
    }
    t.conv(&format!("{mask}.conv_s0"), &[32, 256, 1, 1], 32);
    t.conv(&format!("{mask}.conv_s1"), &[64, 256, 1, 1], 64);
    t.conv(&format!("{mask}.output_upscaling.0"), &[256, 64, 2, 2], 64);
    t.norm(&format!("{mask}.output_upscaling.1"), 64);
    t.conv(&format!("{mask}.output_upscaling.3"), &[64, 32, 2, 2], 32);
    for i in 0..4 {
        t.mlp(&format!("{mask}.output_hypernetworks_mlps.{i}.layers"), &[256, 256, 256, 32]);
    }
    t.mlp(&format!("{mask}.iou_prediction_head.layers"), &[256, 256, 256, 4]);
    t.mlp(&format!("{mask}.pred_obj_score_head.layers"), &[256, 256, 256, 1]);
    for i in 0..2 {
        let p = format!("{mask}.transformer.layers.{i}");
        for (name, width) in [("self_attn", 256), ("cross_attn_token_to_image", 128), ("cross_attn_image_to_token", 128)] {
            t.separate_attention(&format!("{p}.{name}"), width);
        }
        for j in 1..=4 {
            t.norm(&format!("{p}.norm{j}"), 256);
        }
        t.linear(&format!("{p}.mlp.lin1"), 256, 2048);
        t.linear(&format!("{p}.mlp.lin2"), 2048, 256);
    }
    t.separate_attention(&format!("{mask}.transformer.final_attn_token_to_image"), 128);
    t.norm(&format!("{mask}.transformer.norm_final_attn"), 256);
    t.add("tracker.model.interactivity_no_mem_embed", &[1, 1, 256]);
    t.0
}

#[derive(Default)]
struct Table(BTreeMap<String, Vec<usize>>);

impl Table {
    fn add(&mut self, key: &str, shape: &[usize]) {
        self.0.insert(key.into(), shape.to_vec());
    }
    fn norm(&mut self, p: &str, width: usize) {
        self.add(&format!("{p}.weight"), &[width]);
        self.add(&format!("{p}.bias"), &[width]);
    }
    fn linear(&mut self, p: &str, input: usize, output: usize) {
        self.conv(p, &[output, input], output);
    }
    fn conv(&mut self, p: &str, weight: &[usize], bias: usize) {
        self.add(&format!("{p}.weight"), weight);
        self.add(&format!("{p}.bias"), &[bias]);
    }
    fn mlp(&mut self, p: &str, dims: &[usize]) {
        for (i, pair) in dims.windows(2).enumerate() {
            self.linear(&format!("{p}.{i}"), pair[0], pair[1]);
        }
    }
    fn fused_attention(&mut self, p: &str, width: usize) {
        self.add(&format!("{p}.in_proj_weight"), &[3 * width, width]);
        self.add(&format!("{p}.in_proj_bias"), &[3 * width]);
        self.linear(&format!("{p}.out_proj"), width, width);
    }
    fn separate_attention(&mut self, p: &str, width: usize) {
        for name in ["q_proj", "k_proj", "v_proj"] {
            self.linear(&format!("{p}.{name}"), 256, width);
        }
        self.linear(&format!("{p}.out_proj"), width, 256);
    }
    fn detector_layer(&mut self, p: &str, attention: &[&str], norms: &[&str]) {
        for name in attention {
            self.fused_attention(&format!("{p}.{name}"), 256);
        }
        for name in norms {
            self.norm(&format!("{p}.{name}"), 256);
        }
        self.linear(&format!("{p}.linear1"), 256, 2048);
        self.linear(&format!("{p}.linear2"), 2048, 256);
    }
}

/// One buffered checkpoint read shared by every model part, without memory mapping.
pub struct Checkpoint {
    tensors: candle_core::safetensors::BufferedSafetensors,
}
impl Checkpoint {
    /// Read once; selected tensors are copied to the requested device as parts load.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { tensors: candle_core::safetensors::BufferedSafetensors::new(std::fs::read(path)?)? })
    }
    pub(crate) fn get(&self, key: &str, shape: &[usize], device: &candle_core::Device) -> Result<Tensor> {
        let t = self.tensors.load(key, device)?;
        anyhow::ensure!(t.dims() == shape, "{key}: expected {shape:?}, got {:?}", t.dims());
        anyhow::ensure!(t.dtype() == candle_core::DType::F16, "{key}: expected checkpoint F16");
        Ok(t)
    }
}
