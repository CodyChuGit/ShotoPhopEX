//! Composed SAM 3.1 flows with the current worker's pixel and selection rules.
use crate::{
    detector::Detector,
    geometry::Geometry,
    mask_decoder::InteractiveHeads,
    resample::bilinear,
    text::{ClipTokenizer, TextEncoder},
    vision::{Vision, VisionOut},
    weights::Checkpoint,
};
use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use std::{path::Path, time::Instant};
pub const INPUT: usize = 1008;
pub const LOW: usize = 288;
pub const NO_OBJ_SCORE: f32 = -1024.0;
/// One found object, at picture resolution.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub score: f32,
    /// x0, y0, x1, y1 in picture pixels (zeros when the prompt carried no box prediction).
    pub bbox: [f32; 4],
    /// Row-major `width × height` coverage, 0 or 1.
    pub mask: Vec<f32>,
    /// Pixel bounds of the mask as x0, y0, width, height, and the pixel count (both 0 when empty).
    pub bounds: [u32; 4],
    pub pixels: u64,
}

impl Instance {
    /// Wrap a coverage plane, measuring it.
    pub fn from_mask(mask: Vec<f32>, width: usize, height: usize, score: f32, bbox: [f32; 4]) -> Instance {
        let (bounds, pixels) = coverage_bounds(&mask, width, height);
        Instance { score, bbox, mask, bounds, pixels }
    }
}

/// Pixel bounds (x0, y0, w, h) and count of a coverage plane.
pub fn coverage_bounds(mask: &[f32], width: usize, height: usize) -> ([u32; 4], u64) {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut n = 0u64;
    for y in 0..height {
        for x in 0..width {
            if mask.get(y * width + x).is_some_and(|v| *v > 0.5) {
                n += 1;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if n == 0 {
        return ([0; 4], 0);
    }
    ([x0 as u32, y0 as u32, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32], n)
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

/// Cached features and original pixels; callers retain this across prompts for the same image.
pub struct Frame {
    pub width: usize,
    pub height: usize,
    rgba: Vec<u8>,
    device_pixels: Option<Tensor>,
    vision: VisionOut,
    geo_cls: Tensor,
    pub ms: u64,
}
/// All model parts share a single checkpoint read at load time.
pub struct Sam31 {
    vision: Vision,
    text: TextEncoder,
    geometry: Geometry,
    detector: Detector,
    heads: InteractiveHeads,
    tokenizer: Option<ClipTokenizer>,
    device: Device,
}
struct HeadsOut {
    masks: Vec<f32>,
    iou: [f32; 4],
    object_score: f32,
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
impl Sam31 {
    /// Load model parts once. Set a tokenizer JSON path before selecting text.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::load_with_dtype(path, device, candle_core::DType::F32)
    }
    /// Select CUDA compute precision explicitly; F16 remains experimental until its parity gate passes.
    pub fn load_with_dtype(path: impl AsRef<Path>, device: &Device, dtype: candle_core::DType) -> Result<Self> {
        let ck = Checkpoint::open(path)?.with_compute_dtype(dtype)?;
        Ok(Self {
            vision: Vision::from_checkpoint(&ck, device)?,
            text: TextEncoder::from_checkpoint(&ck, device)?,
            geometry: Geometry::from_checkpoint(&ck, device)?,
            detector: Detector::from_checkpoint(&ck, device)?,
            heads: InteractiveHeads::from_checkpoint(&ck, device)?,
            tokenizer: None,
            device: device.clone(),
        })
    }
    /// Use a CLIP tokenizer.json (or ComfyUI's `sd1_tokenizer` folder); no downloads are performed.
    pub fn set_tokenizer(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.tokenizer = Some(ClipTokenizer::open(path)?);
        Ok(())
    }
    /// Stretch RGB8 or RGBA8 to the model window, ignoring alpha, and retain cacheable features.
    pub fn encode(&self, pixels: &[u8], width: usize, height: usize) -> Result<Frame> {
        let t = Instant::now();
        let count = width.checked_mul(height).filter(|n| *n > 0).ok_or_else(|| anyhow::anyhow!("invalid image dimensions"))?;
        let rgba = if count.checked_mul(4) == Some(pixels.len()) {
            pixels.to_vec()
        } else {
            ensure!(count.checked_mul(3) == Some(pixels.len()), "expected RGB8 or RGBA8");
            pixels.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect()
        };
        let device_pixels = if self.device.is_cuda() {
            Some(crate::profile::measure(&self.device, "host transfers", || Ok(Tensor::from_slice(&rgba, (1, height, width, 4), &self.device)?))?)
        } else {
            None
        };
        let input = if let Some(pixels) = &device_pixels {
            crate::profile::measure(&self.device, "image preprocessing", || image_input(pixels))?
        } else {
            Tensor::from_vec(planes_of(&rgba, width, [0, 0, width, height]), (1, 3, INPUT, INPUT), &self.device)?
        };
        let vision = self.vision.forward(&input)?;
        let geo_cls = crate::profile::measure(&self.device, "geometry", || self.geometry.cls(&vision.detector_fpn[2], &vision.image_pos))?;
        Ok(Frame { width, height, rgba, device_pixels, vision, geo_cls, ms: t.elapsed().as_millis() as u64 })
    }
    fn run_tracker(&self, planes: Vec<f32>) -> Result<[Tensor; 3]> {
        self.vision.forward_tracker(&Tensor::from_vec(planes, (1, 3, INPUT, INPUT), &self.device)?)
    }
    fn run_heads(&self, trk: &[Tensor; 3], coords: &[[f32; 2]], labels: &[i32], mask: Option<&[f32]>) -> Result<HeadsOut> {
        let mask = mask.filter(|m| m.len() == INPUT * INPUT).map(|m| Tensor::from_slice(m, (1, 1, INPUT, INPUT), &self.device)).transpose()?;
        let out = crate::profile::measure(&self.device, "heads", || self.heads.forward([&trk[0], &trk[1], &trk[2]], coords, labels, mask.as_ref()))?;
        let iou: [f32; 4] =
            out.iou.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?.try_into().map_err(|_| anyhow::anyhow!("expected four IoUs"))?;
        Ok(HeadsOut {
            masks: out.masks.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            iou,
            object_score: out.object_score.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0],
        })
    }
    /// `SAM3Model.forward_segment`: one pass of the tracker heads on a frame, with points, a box
    /// or a mask, answering 1008×1008 logits.
    fn forward_segment(&self, trk: &[Tensor; 3], points: &[([f32; 2], i32)], bbox: Option<[f32; 4]>, mask: Option<&[f32]>) -> Result<Vec<f32>> {
        let (coords, labels) = crate::prompt::pack_prompts(points, bbox);
        let multimask = points.len() == 1;
        let out = self.run_heads(trk, &coords, &labels, mask)?;
        Ok(select_head(out, multimask))
    }

    /// Keep selected logits and iterative feedback on CUDA until the final binary mask.
    fn forward_segment_device(&self, trk: &[Tensor; 3], points: &[([f32; 2], i32)], bbox: Option<[f32; 4]>, mask: Option<&Tensor>) -> Result<Tensor> {
        let (coords, labels) = crate::prompt::pack_prompts(points, bbox);
        let out = crate::profile::measure(&self.device, "heads", || self.heads.forward([&trk[0], &trk[1], &trk[2]], &coords, &labels, mask))?;
        self.choose_head_device(&out, points.len() == 1)
    }
    fn choose_head_device(&self, out: &crate::mask_decoder::InteractiveOutput, multimask: bool) -> Result<Tensor> {
        crate::profile::measure(&self.device, "mask post-processing", || {
            let low = if multimask {
                let best = out.iou.narrow(1, 1, 3)?.argmax(1)?;
                out.masks.narrow(1, 1, 3)?.index_select(&best, 1)?
            } else {
                out.masks.narrow(1, 0, 1)?
            };
            let logits = low.to_dtype(candle_core::DType::F32)?.upsample_bilinear2d(INPUT, INPUT, false)?;
            let valid = out.object_score.gt(0.0)?.reshape((1, 1, 1, 1))?.broadcast_as(logits.shape())?;
            Ok(valid.where_cond(&logits, &Tensor::full(NO_OBJ_SCORE, logits.shape(), &self.device)?)?)
        })
    }
    fn device_mask(&self, logits: &Tensor, width: usize, height: usize) -> Result<Vec<f32>> {
        let binary = crate::profile::measure(&self.device, "mask post-processing", || Ok(logits.upsample_bilinear2d(height, width, false)?.gt(0.0)?))?;
        crate::profile::measure(&self.device, "host transfers", || Ok(binary.flatten_all()?.to_vec1::<u8>()?.into_iter().map(f32::from).collect()))
    }
    /// The node's point route: positives and negatives in picture pixels.
    pub fn segment_points(&self, emb: &Frame, positives: &[[f32; 2]], negatives: &[[f32; 2]], opts: Options) -> Result<(Instance, Timing)> {
        let mut timing = Timing::default();
        let (w, h) = (emb.width as f32, emb.height as f32);
        let scale = |p: &[f32; 2]| [p[0] / w * INPUT as f32, p[1] / h * INPUT as f32];
        let mut pts: Vec<([f32; 2], i32)> = positives.iter().map(|p| (scale(p), 1)).collect();
        pts.extend(negatives.iter().map(|p| (scale(p), 0)));
        let t = Instant::now();
        if self.device.is_cuda() {
            let mut logits = self.forward_segment_device(&emb.vision.interactive_fpn, &pts, None, None)?;
            for _ in 1..opts.refine_iterations.max(1) {
                logits = self.forward_segment_device(&emb.vision.interactive_fpn, &[], None, Some(&logits))?;
            }
            timing.heads_ms = t.elapsed().as_millis() as u64;
            let t = Instant::now();
            let mask = self.device_mask(&logits, emb.width, emb.height)?;
            timing.post_ms = t.elapsed().as_millis() as u64;
            return Ok((Instance::from_mask(mask, emb.width, emb.height, 1.0, [0.0; 4]), timing));
        }
        let mut logit = self.forward_segment(&emb.vision.interactive_fpn, &pts, None, None)?;
        for _ in 1..opts.refine_iterations.max(1) {
            logit = self.forward_segment(&emb.vision.interactive_fpn, &[], None, Some(&logit))?;
        }
        timing.heads_ms = t.elapsed().as_millis() as u64;
        let t = Instant::now();
        let mask = threshold(&bilinear(&logit, INPUT, INPUT, emb.width, emb.height));
        timing.post_ms = t.elapsed().as_millis() as u64;
        Ok((Instance::from_mask(mask, emb.width, emb.height, 1.0, [0.0; 4]), timing))
    }

    /// The node's box route: one box in picture pixels (x0, y0, x1, y1).
    pub fn segment_box(&self, emb: &Frame, bbox: [f32; 4], opts: Options) -> Result<(Instance, Timing)> {
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
        if self.device.is_cuda() {
            let mut logits = self.forward_segment_device(&emb.vision.interactive_fpn, &[], Some(corners), None)?;
            for _ in 1..opts.refine_iterations.max(1) {
                logits = self.forward_segment_device(&emb.vision.interactive_fpn, &[], None, Some(&logits))?;
            }
            timing.heads_ms = t.elapsed().as_millis() as u64;
            let t = Instant::now();
            let mask = self.device_mask(&logits, emb.width, emb.height)?;
            timing.post_ms = t.elapsed().as_millis() as u64;
            return Ok((Instance::from_mask(mask, emb.width, emb.height, 1.0, [0.0; 4]), timing));
        }
        let mut logit = self.forward_segment(&emb.vision.interactive_fpn, &[], Some(corners), None)?;
        for _ in 1..opts.refine_iterations.max(1) {
            logit = self.forward_segment(&emb.vision.interactive_fpn, &[], None, Some(&logit))?;
        }
        timing.heads_ms = t.elapsed().as_millis() as u64;
        let t = Instant::now();
        let mask = threshold(&bilinear(&logit, INPUT, INPUT, emb.width, emb.height));
        timing.post_ms = t.elapsed().as_millis() as u64;
        Ok((Instance::from_mask(mask, emb.width, emb.height, 1.0, [0.0; 4]), timing))
    }

    /// Detector text selection with optional positive box tokens in original-image pixels.
    pub fn detect_text(&self, emb: &Frame, text: &str, boxes: &[[f32; 4]], opts: Options) -> Result<(Vec<Instance>, Timing)> {
        let mut timing = Timing::default();
        let (w, h) = (emb.width, emb.height);
        let t = Instant::now();
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!("set_tokenizer requires a local tokenizer.json"))?;
        let (ids, valid) = tokenizer.encode(text)?;
        let (features, valid) = crate::profile::measure(&self.device, "text", || self.text.encode(&ids, &valid))?;
        let mut parts = vec![features.to_dtype(emb.geo_cls.dtype())?];
        let mut mask: Vec<u8> = valid.into_iter().map(u8::from).collect();
        if !boxes.is_empty() {
            let flat: Vec<f32> = boxes
                .iter()
                .flat_map(|b| [(b[0] + b[2]) * 0.5 / w as f32, (b[1] + b[3]) * 0.5 / h as f32, (b[2] - b[0]) / w as f32, (b[3] - b[1]) / h as f32])
                .collect();
            parts.push(self.geometry.boxes(&emb.vision.detector_fpn[2], &emb.vision.image_pos, &Tensor::from_vec(flat, (1, boxes.len(), 4), &self.device)?)?);
            mask.extend(std::iter::repeat_n(1, boxes.len()));
        }
        parts.push(emb.geo_cls.clone());
        mask.push(1);
        let pf = Tensor::cat(&parts, 1)?;
        let plen = mask.len();
        let pm = Tensor::from_vec(mask, (1, plen), &self.device)?;
        timing.text_ms = t.elapsed().as_millis() as u64;
        let t = Instant::now();
        let f = &emb.vision;
        let out = self.detector.forward([&f.detector_fpn[0], &f.detector_fpn[1], &f.detector_fpn[2]], &f.image_pos, &pf, &pm)?;
        let (_, q, mh, mw) = out.masks.dims4()?;

        let boxes =
            crate::profile::measure(&self.device, "host transfers", || Ok(out.boxes.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?))?;
        let logits =
            crate::profile::measure(&self.device, "host transfers", || Ok(out.logits.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?))?;
        timing.decoder_ms = t.elapsed().as_millis() as u64;
        // probs > threshold, best first, at most max_detections.
        let mut ranked: Vec<(usize, f32)> = logits.iter().take(q).enumerate().map(|(i, l)| (i, sigmoid(*l))).filter(|(_, p)| *p > opts.threshold).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(opts.max_detections);

        let mask_tensor = out.masks;
        let mut out = Vec::with_capacity(ranked.len());
        for (i, score) in ranked {
            let t = Instant::now();
            let b = boxes.get(i * 4..i * 4 + 4).unwrap_or(&[0.0; 4]);
            let bbox = [b[0] * w as f32, b[1] * h as f32, b[2] * w as f32, b[3] * h as f32];
            if self.device.is_cuda() {
                let mask = self.refine_device(emb, &mask_tensor.narrow(1, i, 1)?, bbox, opts.refine_iterations, &mut timing)?;
                out.push(Instance::from_mask(mask, w, h, score, bbox));
                continue;
            }
            let low = crate::profile::measure(&self.device, "host transfers", || {
                Ok(mask_tensor.narrow(1, i, 1)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
            })?;
            // detector.forward(orig_size=…): masks to the picture size, boxes to pixels.
            let coarse = crate::profile::measure(&self.device, "mask post-processing", || Ok(bilinear(&low, mw, mh, w, h)))?;
            let b = boxes.get(i * 4..i * 4 + 4).unwrap_or(&[0.0; 4]);
            let bbox = [b[0] * w as f32, b[1] * h as f32, b[2] * w as f32, b[3] * h as f32];
            timing.post_ms += t.elapsed().as_millis() as u64;
            let mask = self.refine(emb, &coarse, bbox, opts.refine_iterations, &mut timing)?;
            out.push(Instance::from_mask(mask, w, h, score, bbox));
        }
        Ok((out, timing))
    }

    /// Keep detector logits, crop feedback and the final union on CUDA; download one mask.
    fn refine_device(&self, emb: &Frame, low: &Tensor, bbox: [f32; 4], iterations: usize, timing: &mut Timing) -> Result<Vec<f32>> {
        let (w, h) = (emb.width, emb.height);
        let coarse =
            crate::profile::measure(&self.device, "mask post-processing", || Ok(low.to_dtype(candle_core::DType::F32)?.upsample_bilinear2d(h, w, false)?))?;
        let binary = coarse.gt(0.0)?;
        let finish = |mask: &Tensor| -> Result<Vec<f32>> {
            crate::profile::measure(&self.device, "host transfers", || Ok(mask.flatten_all()?.to_vec1::<u8>()?.into_iter().map(f32::from).collect()))
        };
        if iterations == 0 {
            return finish(&binary);
        }
        let (x1, y1, x2, y2) = (f64::from(bbox[0]), f64::from(bbox[1]), f64::from(bbox[2]), f64::from(bbox[3]));
        let (bw, bh) = (x2 - x1, y2 - y1);
        let cx1 = ((x1 - bw * 0.1).trunc() as i64).max(0) as usize;
        let cy1 = ((y1 - bh * 0.1).trunc() as i64).max(0) as usize;
        let cx2 = (((x2 + bw * 0.1).trunc() as i64).max(0) as usize).min(w);
        let cy2 = (((y2 + bh * 0.1).trunc() as i64).max(0) as usize).min(h);
        if cx2 <= cx1 || cy2 <= cy1 {
            return finish(&binary);
        }
        let mx1 = ((cx1 as f64 / w as f64) * w as f64).trunc() as usize;
        let my1 = ((cy1 as f64 / h as f64) * h as f64).trunc() as usize;
        let mx2 = ((cx2 as f64 / w as f64) * w as f64).trunc() as usize;
        let my2 = ((cy2 as f64 / h as f64) * h as f64).trunc() as usize;
        if mx2 <= mx1 || my2 <= my1 {
            return finish(&binary);
        }
        let pixels = emb.device_pixels.as_ref().ok_or_else(|| anyhow::anyhow!("CUDA frame has no device pixels"))?;
        let t = Instant::now();
        let input =
            crate::profile::measure(&self.device, "crop preprocessing", || image_input(&pixels.narrow(1, cy1, cy2 - cy1)?.narrow(2, cx1, cx2 - cx1)?))?;
        let trk = self.vision.forward_tracker(&input)?;
        timing.crop_ms += t.elapsed().as_millis() as u64;
        let t = Instant::now();
        let mut feedback = crate::profile::measure(&self.device, "mask post-processing", || {
            Ok(coarse.narrow(2, my1, my2 - my1)?.narrow(3, mx1, mx2 - mx1)?.contiguous()?.upsample_bilinear2d(INPUT, INPUT, false)?)
        })?;
        for _ in 0..iterations {
            feedback = self.forward_segment_device(&trk, &[], None, Some(&feedback))?;
        }
        timing.heads_ms += t.elapsed().as_millis() as u64;
        let full = crate::profile::measure(&self.device, "mask post-processing", || {
            let refined =
                feedback.upsample_bilinear2d(cy2 - cy1, cx2 - cx1, false)?.gt(0.0)?.pad_with_zeros(2, cy1, h - cy2)?.pad_with_zeros(3, cx1, w - cx2)?;
            Ok(binary.broadcast_add(&refined)?.gt(0.0)?)
        })?;
        finish(&full)
    }
    /// `_refine_mask`: refine a coarse detector mask through the tracker heads on a crop around
    /// its box, then OR it with the coarse mask.
    fn refine(&self, emb: &Frame, coarse: &[f32], bbox: [f32; 4], iterations: usize, timing: &mut Timing) -> Result<Vec<f32>> {
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
        let crop_trk = self.run_tracker(planes_of(&emb.rgba, w, [cx1, cy1, crop_w, crop_h]))?;
        timing.crop_ms += t.elapsed().as_millis() as u64;

        let (mw, mh) = (mx2 - mx1, my2 - my1);
        let mut region = Vec::with_capacity(mw * mh);
        for y in my1..my2 {
            region.extend_from_slice(&coarse[y * w + mx1..y * w + mx2]);
        }
        let t = Instant::now();
        let mut logit = bilinear(&region, mw, mh, INPUT, INPUT);
        for _ in 0..iterations {
            logit = self.forward_segment(&crop_trk, &[], None, Some(&logit))?;
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
    /// Select text detections in descending score order.
    pub fn select_text(&self, frame: &Frame, prompt: &str, opts: Options) -> Result<Vec<Instance>> {
        Ok(self.detect_text(frame, prompt, &[], opts)?.0)
    }
    /// Select detector instances conditioned on an empty text prompt and positive box tokens.
    pub fn select_box(&self, frame: &Frame, boxes: &[[f32; 4]], opts: Options) -> Result<Vec<Instance>> {
        Ok(self.detect_text(frame, "", boxes, opts)?.0)
    }
    /// Run one tracker pass; points and xyxy box use picture pixels, feedback uses 1008-square logits.
    /// Returns the selected binary mask at picture resolution and the four raw predicted IoUs.
    pub fn interactive(
        &self,
        frame: &Frame,
        points: &[[f32; 2]],
        labels: &[i32],
        bbox: Option<[f32; 4]>,
        mask_input: Option<&[f32]>,
    ) -> Result<(Vec<f32>, [f32; 4])> {
        ensure!(points.len() == labels.len(), "points and labels must have equal lengths");
        let scale = |p: [f32; 2]| [p[0] / frame.width as f32 * INPUT as f32, p[1] / frame.height as f32 * INPUT as f32];
        let pts: Vec<_> = points.iter().zip(labels).map(|(p, l)| (scale(*p), *l)).collect();
        let bbox = bbox.map(|b| {
            let a = scale([b[0], b[1]]);
            let z = scale([b[2], b[3]]);
            [a[0], a[1], z[0], z[1]]
        });
        let (coords, labels) = crate::prompt::pack_prompts(&pts, bbox);
        if self.device.is_cuda() {
            let mask = mask_input.filter(|m| m.len() == INPUT * INPUT).map(|m| Tensor::from_slice(m, (1, 1, INPUT, INPUT), &self.device)).transpose()?;
            let trk = &frame.vision.interactive_fpn;
            let out = crate::profile::measure(&self.device, "heads", || self.heads.forward([&trk[0], &trk[1], &trk[2]], &coords, &labels, mask.as_ref()))?;
            let logits = self.choose_head_device(&out, pts.len() == 1)?;
            let iou: [f32; 4] =
                crate::profile::measure(&self.device, "host transfers", || Ok(out.iou.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?))?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("expected four IoUs"))?;
            return Ok((self.device_mask(&logits, frame.width, frame.height)?, iou));
        }
        let out = self.run_heads(&frame.vision.interactive_fpn, &coords, &labels, mask_input)?;
        let iou = out.iou;
        let logits = select_head(out, pts.len() == 1);
        Ok((threshold(&bilinear(&logits, INPUT, INPUT, frame.width, frame.height)), iou))
    }
}
/// Logits → coverage at logit 0.
fn threshold(logits: &[f32]) -> Vec<f32> {
    logits.iter().map(|v| if *v > 0.0 { 1.0 } else { 0.0 }).collect()
}

fn select_head(out: HeadsOut, multimask: bool) -> Vec<f32> {
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
    if out.object_score > 0.0 { bilinear(low, LOW, LOW, INPUT, INPUT) } else { vec![NO_OBJ_SCORE; INPUT * INPUT] }
}

/// Resize RGB on the device using the worker's half-pixel bilinear convention.
fn image_input(rgba: &Tensor) -> Result<Tensor> {
    Ok(rgba
        .narrow(3, 0, 3)?
        .permute((0, 3, 1, 2))?
        .contiguous()?
        .to_dtype(candle_core::DType::F32)?
        .affine(1.0 / 255.0, 0.0)?
        .upsample_bilinear2d(INPUT, INPUT, false)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_choice_ties_and_no_object_match_worker() {
        let output = |score| HeadsOut { masks: (0..4).flat_map(|i| vec![i as f32; LOW * LOW]).collect(), iou: [0.9, 0.7, 0.8, 0.8], object_score: score };
        assert!(select_head(output(1.0), true).iter().all(|v| *v == 2.0));
        assert!(select_head(output(1.0), false).iter().all(|v| *v == 0.0));
        assert!(select_head(output(0.0), true).iter().all(|v| *v == NO_OBJ_SCORE));
    }

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
