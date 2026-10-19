//! SAM 3 as the Hugging Face `transformers` export lays it out: four graphs in one directory.
//! (The ComfyUI-faithful SAM 3.1 layout, with the tracker's point and box path, is `sam31`.)
//!
//! ```text
//! <dir>/vision-encoder.onnx     images [1,3,1008,1008] → fpn_feat_0..2, fpn_pos_2
//! <dir>/text-encoder.onnx       input_ids, attention_mask [1,32] → text_features [1,L,256], text_mask
//! <dir>/geometry-encoder.onnx   input_boxes [1,N,4] cxcywh, labels [1,N], fpn_feat_2, fpn_pos_2 → geometry_features, geometry_mask
//! <dir>/decoder.onnx            fpn_feat_0..2, fpn_pos_2, prompt_features, prompt_mask → pred_masks, pred_boxes, pred_logits, presence_logits
//! <dir>/tokenizer.json          the CLIP byte-pair vocabulary
//! ```
//!
//! The vision encoder runs once per picture; prompts re-run only the small graphs. Scoring and
//! mask post-processing follow ComfyUI's `SAM3_Detect` node: `sigmoid(pred_logits) > threshold`,
//! the best `max_instances` kept, masks stretched to the picture with bilinear interpolation
//! and cut at logit 0. This layout has no point path: points become small boxes through the
//! geometry encoder, which is an approximation.

use std::path::Path;
use std::time::Instant;

use ort::session::Session;
use ort::value::{Tensor, TensorRef};

pub use crate::graph::{Error as Sam3Error, Instance};
use crate::graph::{Feat, Result, dim, sigmoid, take_bool, take_f32};
use crate::resample::{bilinear, rgba_to_planes};
use crate::runtime::Runtime;
use crate::tokenizer::ClipTokenizer;

/// The side of the square window every picture is squashed into.
pub const INPUT: usize = 1008;
/// ImageNet mean and standard deviation, the normalisation the Hugging Face export was traced
/// with. (ComfyUI's port feeds the trunk `0..1` directly; an export of that port sets `None`.)
pub const IMAGENET: ([f32; 3], [f32; 3]) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);

pub const FILES: [&str; 5] = ["vision-encoder.onnx", "text-encoder.onnx", "geometry-encoder.onnx", "decoder.onnx", "tokenizer.json"];

/// Prompt features with their validity mask.
type PromptPart = (Feat, Vec<bool>);

pub struct Sam3 {
    vision: Session,
    text: Session,
    geometry: Session,
    decoder: Session,
    tokenizer: ClipTokenizer,
    /// Mean and standard deviation applied to the `0..1` planes before the vision encoder.
    pub normalize: Option<([f32; 3], [f32; 3])>,
}

/// A picture encoded once: the three FPN levels and the coarsest position encoding.
pub struct ImageEmbedding {
    pub width: usize,
    pub height: usize,
    fpn0: Feat,
    fpn1: Feat,
    fpn2: Feat,
    pos2: Feat,
    /// Wall time of the vision encoder.
    pub ms: u64,
}

/// What to segment.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    /// A concept, as the user typed it. Empty with geometry means the Hugging Face convention
    /// "visual".
    pub text: String,
    /// Positive boxes, in picture pixels (x0, y0, x1, y1).
    pub boxes: Vec<[f32; 4]>,
    /// Positive points, in picture pixels (small positive boxes through the detector).
    pub points: Vec<[f32; 2]>,
    /// Negative points, in picture pixels (small negative boxes).
    pub negatives: Vec<[f32; 2]>,
}

/// How to score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scoring {
    /// Instances whose `sigmoid(pred_logit)` is below this are dropped (ComfyUI: 0.5).
    pub threshold: f32,
    /// At most this many instances, best first (ComfyUI: 1 per prompt).
    pub max_instances: usize,
    /// Multiply by `sigmoid(presence_logit)` (Hugging Face's rule; ComfyUI does not).
    pub presence: bool,
}

impl Default for Scoring {
    fn default() -> Self {
        Scoring { threshold: 0.5, max_instances: 1, presence: false }
    }
}

/// Timings of one prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timing {
    pub text_ms: u64,
    pub geometry_ms: u64,
    pub decoder_ms: u64,
    pub post_ms: u64,
}

impl Sam3 {
    /// Open the four graphs and the tokenizer from `dir`.
    pub fn open(rt: &Runtime, dir: &Path) -> Result<Self> {
        for f in FILES {
            let p = dir.join(f);
            if !p.is_file() {
                return Err(Sam3Error::Missing(p));
            }
        }
        Ok(Sam3 {
            vision: rt.session(&dir.join(FILES[0]))?,
            text: rt.session(&dir.join(FILES[1]))?,
            geometry: rt.session(&dir.join(FILES[2]))?,
            decoder: rt.session(&dir.join(FILES[3]))?,
            tokenizer: ClipTokenizer::open(&dir.join(FILES[4])).map_err(Sam3Error::Tokenizer)?,
            normalize: Some(IMAGENET),
        })
    }

    /// Squash an RGBA8 picture into the window and run the vision encoder.
    pub fn embed(&mut self, rgba: &[u8], width: usize, height: usize) -> Result<ImageEmbedding> {
        let t0 = Instant::now();
        let mut planes = rgba_to_planes(rgba, width, height, INPUT, INPUT);
        if let Some((mean, std)) = self.normalize {
            let n = INPUT * INPUT;
            for c in 0..3 {
                for v in &mut planes[c * n..(c + 1) * n] {
                    *v = (*v - mean[c]) / std[c];
                }
            }
        }
        let input = Tensor::from_array(([1usize, 3, INPUT, INPUT], planes))?;
        let outputs = self.vision.run(ort::inputs!["images" => input])?;
        let fpn0 = take_f32(&outputs, "fpn_feat_0")?;
        let fpn1 = take_f32(&outputs, "fpn_feat_1")?;
        let fpn2 = take_f32(&outputs, "fpn_feat_2")?;
        let pos2 = take_f32(&outputs, "fpn_pos_2")?;
        Ok(ImageEmbedding { width, height, fpn0, fpn1, fpn2, pos2, ms: t0.elapsed().as_millis() as u64 })
    }

    /// Encode a text prompt: features `[1, L, 256]` and mask `[1, L]`.
    fn encode_text(&mut self, text: &str) -> Result<PromptPart> {
        let enc = self.tokenizer.encode(text).map_err(Sam3Error::Tokenizer)?;
        let ids = Tensor::from_array(([1usize, enc.ids.len()], enc.ids))?;
        let mask = Tensor::from_array(([1usize, enc.mask.len()], enc.mask))?;
        let outputs = self.text.run(ort::inputs!["input_ids" => ids, "attention_mask" => mask])?;
        let features = take_f32(&outputs, "text_features")?;
        let (_, m) = take_bool(&outputs, "text_mask")?;
        Ok((features, m))
    }

    /// Encode boxes (`cx, cy, w, h` normalised, with labels) against the coarsest FPN level.
    fn encode_geometry(&mut self, emb: &ImageEmbedding, boxes: &[[f32; 4]], labels: &[i64]) -> Result<PromptPart> {
        let n = boxes.len();
        let flat: Vec<f32> = boxes.iter().flatten().copied().collect();
        let b = Tensor::from_array(([1usize, n, 4], flat))?;
        let l = Tensor::from_array(([1usize, n], labels.to_vec()))?;
        let f2 = TensorRef::from_array_view((emb.fpn2.0.as_slice(), emb.fpn2.1.as_slice()))?;
        let p2 = TensorRef::from_array_view((emb.pos2.0.as_slice(), emb.pos2.1.as_slice()))?;
        let outputs = self.geometry.run(ort::inputs!["input_boxes" => b, "input_boxes_labels" => l, "fpn_feat_2" => f2, "fpn_pos_2" => p2])?;
        let features = take_f32(&outputs, "geometry_features")?;
        let (_, m) = take_bool(&outputs, "geometry_mask")?;
        Ok((features, m))
    }

    /// Segment one prompt against an embedded picture.
    pub fn segment(&mut self, emb: &ImageEmbedding, prompt: &Prompt, scoring: Scoring) -> Result<(Vec<Instance>, Timing)> {
        let mut timing = Timing::default();
        let (w, h) = (emb.width as f32, emb.height as f32);

        // Geometry: boxes as given, points as small boxes (1.5 % of the longer side).
        let side = w.max(h) * 0.015;
        let mut boxes: Vec<[f32; 4]> = Vec::new();
        let mut labels: Vec<i64> = Vec::new();
        let norm = |x0: f32, y0: f32, x1: f32, y1: f32| -> [f32; 4] {
            let (x0, x1) = (x0.clamp(0.0, w), x1.clamp(0.0, w));
            let (y0, y1) = (y0.clamp(0.0, h), y1.clamp(0.0, h));
            [(x0 + x1) * 0.5 / w, (y0 + y1) * 0.5 / h, (x1 - x0).max(1.0) / w, (y1 - y0).max(1.0) / h]
        };
        for b in &prompt.boxes {
            boxes.push(norm(b[0], b[1], b[2], b[3]));
            labels.push(1);
        }
        for p in &prompt.points {
            boxes.push(norm(p[0] - side, p[1] - side, p[0] + side, p[1] + side));
            labels.push(1);
        }
        for p in &prompt.negatives {
            boxes.push(norm(p[0] - side, p[1] - side, p[0] + side, p[1] + side));
            labels.push(0);
        }

        let text = if prompt.text.trim().is_empty() { "visual" } else { prompt.text.as_str() };
        let t = Instant::now();
        let ((tshape, tfeat), tmask) = self.encode_text(text)?;
        timing.text_ms = t.elapsed().as_millis() as u64;
        let tl = dim(&tshape, 1, "text_features")?;
        let mut feats = tfeat;
        let mut mask: Vec<bool> = tmask.into_iter().take(tl).collect();
        mask.resize(tl, true);

        if !boxes.is_empty() {
            let t = Instant::now();
            let ((gshape, gfeat), gmask) = self.encode_geometry(emb, &boxes, &labels)?;
            timing.geometry_ms = t.elapsed().as_millis() as u64;
            let gl = dim(&gshape, 1, "geometry_features")?;
            feats.extend_from_slice(&gfeat);
            let mut gm: Vec<bool> = gmask.into_iter().take(gl).collect();
            gm.resize(gl, true);
            mask.extend(gm);
        }
        let plen = mask.len();

        let t = Instant::now();
        let pf = Tensor::from_array(([1usize, plen, 256], feats))?;
        let pm = Tensor::from_array(([1usize, plen], mask))?;
        let f0 = TensorRef::from_array_view((emb.fpn0.0.as_slice(), emb.fpn0.1.as_slice()))?;
        let f1 = TensorRef::from_array_view((emb.fpn1.0.as_slice(), emb.fpn1.1.as_slice()))?;
        let f2 = TensorRef::from_array_view((emb.fpn2.0.as_slice(), emb.fpn2.1.as_slice()))?;
        let p2 = TensorRef::from_array_view((emb.pos2.0.as_slice(), emb.pos2.1.as_slice()))?;
        let outputs = self.decoder.run(ort::inputs![
            "fpn_feat_0" => f0, "fpn_feat_1" => f1, "fpn_feat_2" => f2, "fpn_pos_2" => p2,
            "prompt_features" => pf, "prompt_mask" => pm,
        ])?;
        let (mshape, masks) = take_f32(&outputs, "pred_masks")?;
        let (_, bboxes) = take_f32(&outputs, "pred_boxes")?;
        let (_, logits) = take_f32(&outputs, "pred_logits")?;
        let presence = take_f32(&outputs, "presence_logits").map(|(_, p)| p.first().copied().unwrap_or(0.0)).unwrap_or(0.0);
        drop(outputs);
        timing.decoder_ms = t.elapsed().as_millis() as u64;

        let t = Instant::now();
        let q = dim(&mshape, 1, "pred_masks")?;
        let mh = dim(&mshape, 2, "pred_masks")?;
        let mw = dim(&mshape, 3, "pred_masks")?;
        let presence_p = if scoring.presence { sigmoid(presence) } else { 1.0 };
        let mut ranked: Vec<(usize, f32)> = logits.iter().take(q).enumerate().map(|(i, l)| (i, sigmoid(*l) * presence_p)).filter(|(_, s)| *s > scoring.threshold).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(scoring.max_instances);

        let mut out = Vec::with_capacity(ranked.len());
        for (i, score) in ranked {
            let low = masks.get(i * mh * mw..(i + 1) * mh * mw).ok_or_else(|| Sam3Error::Shape("pred_masks shorter than its shape".into()))?;
            let full = bilinear(low, mw, mh, emb.width, emb.height);
            let mask: Vec<f32> = full.iter().map(|v| if *v > 0.0 { 1.0 } else { 0.0 }).collect();
            let b = bboxes.get(i * 4..i * 4 + 4).unwrap_or(&[0.0; 4]);
            out.push(Instance::from_mask(mask, emb.width, emb.height, score, [b[0] * w, b[1] * h, b[2] * w, b[3] * h]));
        }
        timing.post_ms = t.elapsed().as_millis() as u64;
        Ok((out, timing))
    }
}
