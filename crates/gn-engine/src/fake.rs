//! An engine that paints: a solid colour from the seed (an edit tints the input, a segment
//! returns a centred disc as coverage), `steps` progress reports with a configurable delay,
//! cancellation honoured between steps. For protocol tests and for wiring the editor before
//! a real model exists.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::protocol::{Blob, DoneInfo, ErrorKind, Health, RunRequest};
use crate::server::{Engine, EngineError};

pub struct FakeEngine {
    /// Milliseconds per step.
    pub step_ms: u64,
    pub resident: Vec<String>,
    /// Models this fake "has"; anything else is `ModelMissing`.
    pub installed: Vec<String>,
}

impl Default for FakeEngine {
    fn default() -> Self {
        Self { step_ms: 5, resident: Vec::new(), installed: vec!["fake/image".into(), "fake/edit".into(), "fake/segment".into()] }
    }
}

fn colour(seed: u64) -> [u8; 4] {
    let x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    [(x >> 8) as u8, (x >> 24) as u8, (x >> 40) as u8, 255]
}

impl Engine for FakeEngine {
    fn name(&self) -> String {
        "fake".into()
    }

    fn health(&mut self) -> Health {
        Health { ok: true, engine: "fake".into(), device: "none".into(), vram_total: Some(1 << 30), vram_free: Some(1 << 29), resident: self.resident.clone() }
    }

    fn run(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        cancel: &AtomicBool,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        if !self.installed.contains(&req.model) {
            return Err(EngineError::new(ErrorKind::ModelMissing, format!("{} is not installed", req.model)));
        }
        if !self.resident.contains(&req.model) {
            progress(0.0, "Loading");
            self.resident.push(req.model.clone());
        }
        let steps = req.steps.clamp(1, 1000);
        for i in 0..steps {
            if cancel.load(Ordering::SeqCst) {
                return Err(EngineError::cancelled());
            }
            std::thread::sleep(Duration::from_millis(self.step_ms));
            progress((i + 1) as f32 / steps as f32, "Sampling");
        }
        let (w, h) = match &req.image {
            Some(b) => (b.width, b.height),
            None => (req.width.max(1), req.height.max(1)),
        };
        let out = match req.task.as_str() {
            "segment" => {
                // Coverage: a disc of 30 % of the shorter side, centred.
                let (cx, cy, r) = (w as f32 / 2.0, h as f32 / 2.0, w.min(h) as f32 * 0.3);
                let mut bytes = Vec::with_capacity(w as usize * h as usize);
                for y in 0..h {
                    for x in 0..w {
                        let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                        bytes.push(if dx * dx + dy * dy <= r * r { 255 } else { 0 });
                    }
                }
                (Blob { width: w, height: h, channels: 1, index: 0 }, bytes)
            }
            "matte" => {
                // The input's colours with a centred disc as the alpha (the matte models answer
                // RGBA whose alpha is the cut-out).
                let (cx, cy, r) = (w as f32 / 2.0, h as f32 / 2.0, w.min(h) as f32 * 0.3);
                let src = req.image.as_ref().and_then(|b| blobs.get(b.index as usize)).cloned().unwrap_or_default();
                let mut bytes = Vec::with_capacity(w as usize * h as usize * 4);
                for y in 0..h {
                    for x in 0..w {
                        let i = (y as usize * w as usize + x as usize) * 4;
                        let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                        let a = if dx * dx + dy * dy <= r * r { 255 } else { 0 };
                        bytes.extend_from_slice(&[*src.get(i).unwrap_or(&0), *src.get(i + 1).unwrap_or(&0), *src.get(i + 2).unwrap_or(&0), a]);
                    }
                }
                (Blob { width: w, height: h, channels: 4, index: 0 }, bytes)
            }
            "edit" | "fill" | "expand" => {
                // The input tinted by the seed's colour (a visible, deterministic change).
                let c = colour(req.seed);
                let src = req.image.as_ref().and_then(|b| blobs.get(b.index as usize)).cloned().unwrap_or_default();
                let bytes: Vec<u8> = src
                    .chunks(4)
                    .flat_map(|p| {
                        [
                            p.first().map_or(c[0], |v| (u16::from(*v) + u16::from(c[0])) as u8 / 2),
                            p.get(1).map_or(c[1], |v| (u16::from(*v) + u16::from(c[1])) as u8 / 2),
                            p.get(2).map_or(c[2], |v| (u16::from(*v) + u16::from(c[2])) as u8 / 2),
                            255,
                        ]
                    })
                    .collect();
                (Blob { width: w, height: h, channels: 4, index: 0 }, bytes)
            }
            _ => crate::server::solid(w, h, colour(req.seed)),
        };
        Ok((DoneInfo { images: vec![out.0], texts: Vec::new(), seed: req.seed, elapsed_ms: 0 }, vec![out.1]))
    }

    fn free(&mut self) {
        self.resident.clear();
    }

    fn index(&mut self, dir: &str) -> Value {
        json!({"dir": dir, "installed": self.installed})
    }
}
