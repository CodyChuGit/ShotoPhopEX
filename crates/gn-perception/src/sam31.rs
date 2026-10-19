//! SAM 3.1 exactly as ComfyUI runs it: five graphs exported from ComfyUI's own modules by
//! `tools/export_sam31.py`, and this module reproducing the `SAM3_Detect` node on top of them
//! (prompt assembly, scoring, mask selection, the no-object rule, every interpolation step, the
//! crop-and-refine passes). PhotoshopEX's references were made with that node, so this is the
//! path that has to match them.
//!
//! ```text
//! <dir>/vision.onnx     images [1,3,1008,1008] (0..1, no normalisation)
//!                       → fpn_feat_0..2, fpn_pos_2, geo_cls, trk_feat_0..2
//! <dir>/text.onnx       input_ids, attention_mask [1,32] → text_features [1,32,256], text_mask
//! <dir>/geometry.onnx   boxes [1,M,4] cxcywh, fpn_feat_2 → geo_features [1,M,256]
//! <dir>/decoder.onnx    fpn_feat_0..2, fpn_pos_2, prompt_features [1,P,256], prompt_mask [1,P]
//!                       → pred_boxes [1,200,4] xyxy, pred_logits [1,200], pred_masks [1,200,288,288], presence [1]
//! <dir>/heads.onnx      trk_feat_0..2, point_coords [1,N,2], point_labels [1,N], mask_prompt [1,1,1008,1008], has_mask [1]
//!                       → masks [1,4,288,288], iou [1,4], object_score [1,1]
//! <dir>/tokenizer.json
//! ```
//!
//! The node's two routes:
//!
//! - **Text** (`select.byText`): detector → `sigmoid(score) > threshold`, best `max_detections`
//!   (1), each mask stretched to the picture, then refined through the tracker heads on a crop
//!   around its box (`refine_iterations` passes), and OR-ed with the coarse mask.
//! - **Points and boxes** (`select.byPoint`): tracker heads on the whole picture (one point →
//!   the best of three multimask heads, otherwise the single-mask head), then
//!   `refine_iterations - 1` passes that feed the mask back in.

use std::path::Path;
use std::time::Instant;

use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::Tensor;

pub use crate::graph::{Error, Instance};
use crate::graph::{Feat, Result, dim, sigmoid, take_bool, take_f32};
use crate::resample::bilinear;
use crate::runtime::{Device, Runtime};
use crate::tokenizer::ClipTokenizer;

/// The side of the square window every picture (and every refinement crop) is squashed into.
pub const INPUT: usize = 1008;
/// The side of the mask heads' output.
pub const LOW: usize = 288;
/// What the tracker writes over a mask whose object-score head says "no object".
pub const NO_OBJ_SCORE: f32 = -1024.0;
pub const FILES: [&str; 6] = ["vision.onnx", "text.onnx", "geometry.onnx", "decoder.onnx", "heads.onnx", "tokenizer.json"];

pub struct Sam31 {
    vision: Session,
    text: Session,
    geometry: Session,
    decoder: Session,
    heads: Session,
    tokenizer: ClipTokenizer,
    /// Where the vision graph's feature maps live between graphs: the device's memory, so the
    /// decoder and the heads read them in place instead of round-tripping 220 MB through the
    /// host per picture and 110 MB per prompt.
    device_mem: MemoryInfo<'static>,
    cpu_mem: MemoryInfo<'static>,
}

/// Everything the vision graph returns for one 1008×1008 frame. The feature maps stay on the
/// device; only the CLS token (256 numbers) comes to the host, where the prompt is assembled.
pub struct Frame {
    fpn0: Tensor<f32>,
    fpn1: Tensor<f32>,
    fpn2: Tensor<f32>,
    pos2: Tensor<f32>,
    geo_cls: Feat,
    trk: [Tensor<f32>; 3],
}

/// A picture encoded once.
pub struct Embedding {
    pub width: usize,
    pub height: usize,
    /// The picture itself, kept for the refinement crops.
    rgba: Vec<u8>,
    frame: Frame,
    /// Wall time of the vision graph, all in.
    pub ms: u64,
    /// Where that time went: squashing the picture, the session's run, copying the outputs out.
    pub split_ms: [u64; 3],
}

/// The node's knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// `sigmoid(score)` above this keeps a detection (ComfyUI: 0.5).
    pub threshold: f32,
    /// Detections kept per text prompt, best first (ComfyUI: 1 unless the prompt says `:N`).
    pub max_detections: usize,
    /// Tracker refinement passes (ComfyUI: 2; 0 keeps the detector's masks).
    pub refine_iterations: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { threshold: 0.5, max_detections: 1, refine_iterations: 2 }
    }
}

/// Timings of one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timing {
    pub text_ms: u64,
    pub decoder_ms: u64,
    /// Vision passes on refinement crops.
    pub crop_ms: u64,
    pub heads_ms: u64,
    pub post_ms: u64,
}

/// Row-major `w × h` planes of one picture region, squashed to the window: CHW, `0..1`.
fn planes_of(rgba: &[u8], width: usize, rect: [usize; 4]) -> Vec<f32> {
    let [x0, y0, rw, rh] = rect;
    let mut out = Vec::with_capacity(3 * INPUT * INPUT);
    for c in 0..3 {
        let mut plane = Vec::with_capacity(rw * rh);
        for y in y0..y0 + rh {
            for x in x0..x0 + rw {
                plane.push(rgba.get((y * width + x) * 4 + c).map_or(0.0, |v| f32::from(*v) / 255.0));
            }
        }
        out.extend(bilinear(&plane, rw, rh, INPUT, INPUT));
    }
    out
}

/// A memory description for one device.
fn memory_of(device: Device) -> ort::Result<MemoryInfo<'static>> {
    match device {
        Device::Cuda { ordinal } => MemoryInfo::new(AllocationDevice::CUDA, ordinal, AllocatorType::Device, MemoryType::Default),
        Device::Cpu => MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default),
    }
}

/// What the mask heads answered for one call.
struct HeadsOut {
    /// `[4, 288, 288]` logits.
    masks: Vec<f32>,
    iou: [f32; 4],
    object_score: f32,
}

impl Sam31 {
    pub fn open(rt: &Runtime, dir: &Path) -> Result<Self> {
        for f in FILES {
            let p = dir.join(f);
            if !p.is_file() {
                return Err(Error::Missing(p));
            }
        }
        Ok(Sam31 {
            vision: rt.session(&dir.join(FILES[0]))?,
            text: rt.session(&dir.join(FILES[1]))?,
            geometry: rt.session(&dir.join(FILES[2]))?,
            decoder: rt.session(&dir.join(FILES[3]))?,
            heads: rt.session(&dir.join(FILES[4]))?,
            tokenizer: ClipTokenizer::open(&dir.join(FILES[5])).map_err(Error::Tokenizer)?,
            device_mem: memory_of(rt.config().device)?,
            cpu_mem: memory_of(Device::Cpu)?,
        })
    }

    /// Does `dir` hold this layout?
    pub fn is_layout(dir: &Path) -> bool {
        FILES.iter().all(|f| dir.join(f).is_file())
    }

    /// The vision graph on one window; the frame and (run, copy-out) milliseconds. The outputs
    /// are bound to the device's memory and kept there as the frame.
    fn run_vision(&mut self, planes: Vec<f32>) -> Result<(Frame, [u64; 2])> {
        const ON_DEVICE: [&str; 7] = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2", "trk_feat_0", "trk_feat_1", "trk_feat_2"];
        let t = Instant::now();
        let input = Tensor::from_array(([1usize, 3, INPUT, INPUT], planes))?;
        let mut binding = self.vision.create_binding()?;
        binding.bind_input("images", &input)?;
        for name in ON_DEVICE {
            binding.bind_output_to_device(name, &self.device_mem)?;
        }
        binding.bind_output_to_device("geo_cls", &self.cpu_mem)?;
        let mut outputs = self.vision.run_binding(&binding)?;
        let run_ms = t.elapsed().as_millis() as u64;
        let t = Instant::now();
        let geo_cls = take_f32(&outputs, "geo_cls")?;
        let mut take = |name: &str| -> Result<Tensor<f32>> {
            let v = outputs.remove(name).ok_or_else(|| Error::Shape(format!("no output `{name}`")))?;
            v.downcast::<ort::value::TensorValueType<f32>>().map_err(|e| Error::Shape(format!("`{name}`: {e}")))
        };
        let frame = Frame {
            fpn0: take("fpn_feat_0")?,
            fpn1: take("fpn_feat_1")?,
            fpn2: take("fpn_feat_2")?,
            pos2: take("fpn_pos_2")?,
            geo_cls,
            trk: [take("trk_feat_0")?, take("trk_feat_1")?, take("trk_feat_2")?],
        };
        Ok((frame, [run_ms, t.elapsed().as_millis() as u64]))
    }

    /// Encode a whole RGBA8 picture.
    pub fn embed(&mut self, rgba: &[u8], width: usize, height: usize) -> Result<Embedding> {
        let t0 = Instant::now();
        let planes = planes_of(rgba, width, [0, 0, width, height]);
        let planes_ms = t0.elapsed().as_millis() as u64;
        let (frame, [run_ms, copy_ms]) = self.run_vision(planes)?;
        Ok(Embedding { width, height, rgba: rgba.to_vec(), frame, ms: t0.elapsed().as_millis() as u64, split_ms: [planes_ms, run_ms, copy_ms] })
    }

    /// One call of the tracker heads. `coords` are pixels of the 1008 window; `mask` is a
    /// 1008×1008 logit plane when given.
    fn run_heads(&mut self, trk: &[Tensor<f32>; 3], coords: &[[f32; 2]], labels: &[i32], mask: Option<&[f32]>) -> Result<HeadsOut> {
        let n = coords.len();
        let pc = Tensor::from_array(([1usize, n, 2], coords.iter().flatten().copied().collect::<Vec<f32>>()))?;
        let pl = Tensor::from_array(([1usize, n], labels.to_vec()))?;
        let (mask_plane, has) = match mask {
            Some(m) if m.len() == INPUT * INPUT => (m.to_vec(), 1.0f32),
            _ => (vec![0.0f32; INPUT * INPUT], 0.0f32),
        };
        let mp = Tensor::from_array(([1usize, 1, INPUT, INPUT], mask_plane))?;
        let hm = Tensor::from_array(([1usize], vec![has]))?;
        let outputs = self.heads.run(ort::inputs![
            "trk_feat_0" => &trk[0], "trk_feat_1" => &trk[1], "trk_feat_2" => &trk[2],
            "point_coords" => pc, "point_labels" => pl, "mask_prompt" => mp, "has_mask" => hm,
        ])?;
        let (mshape, masks) = take_f32(&outputs, "masks")?;
        let (_, iou) = take_f32(&outputs, "iou")?;
        let (_, score) = take_f32(&outputs, "object_score")?;
        if dim(&mshape, 1, "masks")? != 4 || dim(&mshape, 2, "masks")? != LOW || dim(&mshape, 3, "masks")? != LOW {
            return Err(Error::Shape(format!("heads masks {mshape:?}, expected [1, 4, {LOW}, {LOW}]")));
        }
        let mut iou4 = [0.0f32; 4];
        for (d, s) in iou4.iter_mut().zip(iou.iter()) {
            *d = *s;
        }
        Ok(HeadsOut { masks, iou: iou4, object_score: score.first().copied().unwrap_or(0.0) })
    }

    /// `SAM3Model.forward_segment`: one pass of the tracker heads on a frame, with points, a box
    /// or a mask, answering 1008×1008 logits.
    fn forward_segment(&mut self, trk: &[Tensor<f32>; 3], points: &[([f32; 2], i32)], bbox: Option<[f32; 4]>, mask: Option<&[f32]>) -> Result<Vec<f32>> {
        let mut coords: Vec<[f32; 2]> = Vec::new();
        let mut labels: Vec<i32> = Vec::new();
        if points.is_empty() {
            // forward_sam_heads: one "not a point" stands in for absent point prompts.
            coords.push([0.0, 0.0]);
            labels.push(-1);
        } else {
            for (p, l) in points {
                coords.push(*p);
                labels.push(*l);
            }
        }
        match bbox {
            Some([x0, y0, x1, y1]) => {
                coords.push([x0, y0]);
                labels.push(2);
                coords.push([x1, y1]);
                labels.push(3);
            }
            // The prompt encoder pads the point list with one more "not a point" when no box comes.
            None => {
                coords.push([0.0, 0.0]);
                labels.push(-1);
            }
        }
        let multimask = points.len() == 1;
        let out = self.run_heads(trk, &coords, &labels, mask)?;
        let token = if multimask {
            // The best of the three multimask heads by predicted IoU.
            let mut best = 1;
            for t in 2..4 {
                if out.iou[t] > out.iou[best] {
                    best = t;
                }
            }
            best
        } else {
            0
        };
        let low = &out.masks[token * LOW * LOW..(token + 1) * LOW * LOW];
        let high = if out.object_score > 0.0 { bilinear(low, LOW, LOW, INPUT, INPUT) } else { vec![NO_OBJ_SCORE; INPUT * INPUT] };
        Ok(high)
    }

    /// The node's point route: positives and negatives in picture pixels.
    pub fn segment_points(&mut self, emb: &Embedding, positives: &[[f32; 2]], negatives: &[[f32; 2]], opts: Options) -> Result<(Instance, Timing)> {
        let mut timing = Timing::default();
        let (w, h) = (emb.width as f32, emb.height as f32);
        let scale = |p: &[f32; 2]| [p[0] / w * INPUT as f32, p[1] / h * INPUT as f32];
        let mut pts: Vec<([f32; 2], i32)> = positives.iter().map(|p| (scale(p), 1)).collect();
        pts.extend(negatives.iter().map(|p| (scale(p), 0)));
        let t = Instant::now();
        let mut logit = self.forward_segment(&emb.frame.trk, &pts, None, None)?;
        for _ in 1..opts.refine_iterations.max(1) {
            logit = self.forward_segment(&emb.frame.trk, &[], None, Some(&logit))?;
        }
        timing.heads_ms = t.elapsed().as_millis() as u64;
        let t = Instant::now();
        let mask = threshold(&bilinear(&logit, INPUT, INPUT, emb.width, emb.height));
        timing.post_ms = t.elapsed().as_millis() as u64;
        Ok((Instance::from_mask(mask, emb.width, emb.height, 1.0, [0.0; 4]), timing))
    }

    /// The node's box route: one box in picture pixels (x0, y0, x1, y1).
    pub fn segment_box(&mut self, emb: &Embedding, bbox: [f32; 4], opts: Options) -> Result<(Instance, Timing)> {
        let mut timing = Timing::default();
        let (w, h) = (emb.width as f32, emb.height as f32);
        // The node receives the box as x, y, width, height, turns it into normalised cxcywh,
        // then back into 1008-space corners.
        let cx = (bbox[0] + bbox[2]) * 0.5 / w;
        let cy = (bbox[1] + bbox[3]) * 0.5 / h;
        let bw = (bbox[2] - bbox[0]) / w;
        let bh = (bbox[3] - bbox[1]) / h;
        let s = INPUT as f32;
        let corners = [(cx - bw / 2.0) * s, (cy - bh / 2.0) * s, (cx + bw / 2.0) * s, (cy + bh / 2.0) * s];
        let t = Instant::now();
        let mut logit = self.forward_segment(&emb.frame.trk, &[], Some(corners), None)?;
        for _ in 1..opts.refine_iterations.max(1) {
            logit = self.forward_segment(&emb.frame.trk, &[], None, Some(&logit))?;
        }
        timing.heads_ms = t.elapsed().as_millis() as u64;
        let t = Instant::now();
        let mask = threshold(&bilinear(&logit, INPUT, INPUT, emb.width, emb.height));
        timing.post_ms = t.elapsed().as_millis() as u64;
        Ok((Instance::from_mask(mask, emb.width, emb.height, 1.0, [0.0; 4]), timing))
    }

    /// Encode boxes (picture pixels, x0 y0 x1 y1) as detector prompts: `[1, M, 256]`.
    fn encode_boxes(&mut self, emb: &Embedding, boxes: &[[f32; 4]]) -> Result<Feat> {
        let (w, h) = (emb.width as f32, emb.height as f32);
        // The node's cxcywh, normalised to the picture.
        let flat: Vec<f32> = boxes.iter().flat_map(|b| [(b[0] + b[2]) * 0.5 / w, (b[1] + b[3]) * 0.5 / h, (b[2] - b[0]) / w, (b[3] - b[1]) / h]).collect();
        let bt = Tensor::from_array(([1usize, boxes.len(), 4], flat))?;
        let outputs = self.geometry.run(ort::inputs!["boxes" => bt, "fpn_feat_2" => &emb.frame.fpn2])?;
        take_f32(&outputs, "geo_features")
    }

    /// The node's text route; `boxes` (picture pixels) narrow the search as the node's `bboxes`
    /// input does when text is present.
    pub fn detect_text(&mut self, emb: &Embedding, text: &str, boxes: &[[f32; 4]], opts: Options) -> Result<(Vec<Instance>, Timing)> {
        let mut timing = Timing::default();
        let (w, h) = (emb.width, emb.height);

        let t = Instant::now();
        let enc = self.tokenizer.encode(text).map_err(Error::Tokenizer)?;
        let ids = Tensor::from_array(([1usize, enc.ids.len()], enc.ids))?;
        let am = Tensor::from_array(([1usize, enc.mask.len()], enc.mask))?;
        let outputs = self.text.run(ort::inputs!["input_ids" => ids, "attention_mask" => am])?;
        let (tshape, tfeat) = take_f32(&outputs, "text_features")?;
        let (_, tmask) = take_bool(&outputs, "text_mask")?;
        drop(outputs);
        timing.text_ms = t.elapsed().as_millis() as u64;
        let tl = dim(&tshape, 1, "text_features")?;
        let cl = dim(&emb.frame.geo_cls.0, 1, "geo_cls")?;
        // prompt = text ⊕ boxes ⊕ geo_cls (the detector's own image token); everything beyond the
        // text is valid.
        let mut feats = tfeat;
        let mut mask: Vec<bool> = tmask.into_iter().take(tl).collect();
        mask.resize(tl, true);
        let mut plen = tl + cl;
        if !boxes.is_empty() {
            let (gshape, gfeat) = self.encode_boxes(emb, boxes)?;
            let gl = dim(&gshape, 1, "geo_features")?;
            feats.extend_from_slice(&gfeat);
            mask.extend(std::iter::repeat_n(true, gl));
            plen += gl;
        }
        feats.extend_from_slice(&emb.frame.geo_cls.1);
        mask.extend(std::iter::repeat_n(true, cl));

        let t = Instant::now();
        let pf = Tensor::from_array(([1usize, plen, 256], feats))?;
        let pm = Tensor::from_array(([1usize, plen], mask))?;
        let f = &emb.frame;
        let outputs = self.decoder.run(ort::inputs![
            "fpn_feat_0" => &f.fpn0, "fpn_feat_1" => &f.fpn1, "fpn_feat_2" => &f.fpn2, "fpn_pos_2" => &f.pos2,
            "prompt_features" => pf, "prompt_mask" => pm,
        ])?;
        let (mshape, masks) = take_f32(&outputs, "pred_masks")?;
        let (_, boxes) = take_f32(&outputs, "pred_boxes")?;
        let (_, logits) = take_f32(&outputs, "pred_logits")?;
        drop(outputs);
        timing.decoder_ms = t.elapsed().as_millis() as u64;
        let q = dim(&mshape, 1, "pred_masks")?;
        let mh = dim(&mshape, 2, "pred_masks")?;
        let mw = dim(&mshape, 3, "pred_masks")?;

        // probs > threshold, best first, at most max_detections.
        let mut ranked: Vec<(usize, f32)> = logits.iter().take(q).enumerate().map(|(i, l)| (i, sigmoid(*l))).filter(|(_, p)| *p > opts.threshold).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(opts.max_detections);

        let mut out = Vec::with_capacity(ranked.len());
        for (i, score) in ranked {
            let t = Instant::now();
            let low = masks.get(i * mh * mw..(i + 1) * mh * mw).ok_or_else(|| Error::Shape("pred_masks shorter than its shape".into()))?;
            // detector.forward(orig_size=…): masks to the picture size, boxes to pixels.
            let coarse = bilinear(low, mw, mh, w, h);
            let b = boxes.get(i * 4..i * 4 + 4).unwrap_or(&[0.0; 4]);
            let bbox = [b[0] * w as f32, b[1] * h as f32, b[2] * w as f32, b[3] * h as f32];
            timing.post_ms += t.elapsed().as_millis() as u64;
            let mask = self.refine(emb, &coarse, bbox, opts.refine_iterations, &mut timing)?;
            out.push(Instance::from_mask(mask, w, h, score, bbox));
        }
        Ok((out, timing))
    }

    /// `_refine_mask`: refine a coarse detector mask through the tracker heads on a crop around
    /// its box, then OR it with the coarse mask.
    fn refine(&mut self, emb: &Embedding, coarse: &[f32], bbox: [f32; 4], iterations: usize, timing: &mut Timing) -> Result<Vec<f32>> {
        let (w, h) = (emb.width, emb.height);
        let coarse_bin = threshold(coarse);
        if iterations == 0 {
            return Ok(coarse_bin);
        }
        // The node does this arithmetic in Python floats; f64 here keeps the same truncations.
        let (x1, y1, x2, y2) = (f64::from(bbox[0]), f64::from(bbox[1]), f64::from(bbox[2]), f64::from(bbox[3]));
        let (bw, bh) = (x2 - x1, y2 - y1);
        let pad = 0.1;
        let cx1 = ((x1 - bw * pad).trunc() as i64).max(0) as usize;
        let cy1 = ((y1 - bh * pad).trunc() as i64).max(0) as usize;
        let cx2 = (((x2 + bw * pad).trunc() as i64).max(0) as usize).min(w);
        let cy2 = (((y2 + bh * pad).trunc() as i64).max(0) as usize).min(h);
        if cx2 <= cx1 || cy2 <= cy1 {
            return Ok(coarse_bin);
        }
        let (crop_w, crop_h) = (cx2 - cx1, cy2 - cy1);
        // The coarse mask is already at picture size, so its crop window is the same rectangle,
        // through the node's int(c / W * mask_w) arithmetic.
        let mx1 = ((cx1 as f64 / w as f64) * w as f64).trunc() as usize;
        let my1 = ((cy1 as f64 / h as f64) * h as f64).trunc() as usize;
        let mx2 = ((cx2 as f64 / w as f64) * w as f64).trunc() as usize;
        let my2 = ((cy2 as f64 / h as f64) * h as f64).trunc() as usize;
        if mx2 <= mx1 || my2 <= my1 {
            return Ok(coarse_bin);
        }

        let t = Instant::now();
        let (crop_frame, _) = self.run_vision(planes_of(&emb.rgba, w, [cx1, cy1, crop_w, crop_h]))?;
        timing.crop_ms += t.elapsed().as_millis() as u64;

        let (mw, mh) = (mx2 - mx1, my2 - my1);
        let mut region = Vec::with_capacity(mw * mh);
        for y in my1..my2 {
            region.extend_from_slice(&coarse[y * w + mx1..y * w + mx2]);
        }
        let t = Instant::now();
        let mut logit = bilinear(&region, mw, mh, INPUT, INPUT);
        for _ in 0..iterations {
            logit = self.forward_segment(&crop_frame.trk, &[], None, Some(&logit))?;
        }
        timing.heads_ms += t.elapsed().as_millis() as u64;

        let t = Instant::now();
        let refined = bilinear(&logit, INPUT, INPUT, crop_w, crop_h);
        let mut full = coarse_bin;
        for y in 0..crop_h {
            for x in 0..crop_w {
                if refined[y * crop_w + x] > 0.0 {
                    full[(cy1 + y) * w + cx1 + x] = 1.0;
                }
            }
        }
        timing.post_ms += t.elapsed().as_millis() as u64;
        Ok(full)
    }
}

/// Logits → coverage at logit 0.
fn threshold(logits: &[f32]) -> Vec<f32> {
    logits.iter().map(|v| if *v > 0.0 { 1.0 } else { 0.0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_crop_and_scale() {
        // A 2×2 picture: red, green / blue, white. Crop the right column.
        let rgba = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255];
        let p = planes_of(&rgba, 2, [1, 0, 1, 2]);
        assert_eq!(p.len(), 3 * INPUT * INPUT);
        // Top of the red plane is the green pixel's red (0), bottom is white's red (1).
        assert_eq!(p[0], 0.0);
        assert_eq!(p[INPUT * INPUT - 1], 1.0);
    }

    #[test]
    fn threshold_cuts_at_zero() {
        assert_eq!(threshold(&[-0.1, 0.0, 0.1, NO_OBJ_SCORE]), vec![0.0, 0.0, 1.0, 0.0]);
    }
}
