//! Several engines behind one worker: each request goes to the first engine whose index lists
//! its template (`installed`), so the perception engine and the diffusion engine share one
//! process, one channel and one health report.

use std::sync::atomic::AtomicBool;

use serde_json::{Value, json};

use crate::protocol::{DoneInfo, ErrorKind, Health, RunRequest};
use crate::server::{Engine, EngineError};

pub struct Multi {
    engines: Vec<Box<dyn Engine>>,
}

/// The template ids an engine's index lists under `installed`.
fn installed(v: &Value) -> Vec<String> {
    v.get("installed").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default()
}

impl Multi {
    pub fn new(engines: Vec<Box<dyn Engine>>) -> Self {
        Self { engines }
    }

    pub fn len(&self) -> usize {
        self.engines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    /// The engine that lists `model`.
    fn pick(&mut self, model: &str) -> Option<usize> {
        self.engines.iter_mut().position(|e| installed(&e.index("")).iter().any(|m| m == model))
    }
}

impl Engine for Multi {
    fn name(&self) -> String {
        self.engines.iter().map(|e| e.name()).collect::<Vec<_>>().join("+")
    }

    /// Healthy when one engine is; the device of the first that names one; everything resident.
    fn health(&mut self) -> Health {
        let mut out = Health { engine: self.name(), ..Health::default() };
        for e in &mut self.engines {
            let h = e.health();
            out.ok |= h.ok;
            if out.device.is_empty() {
                out.device = h.device;
            }
            out.vram_total = out.vram_total.or(h.vram_total);
            out.vram_free = out.vram_free.or(h.vram_free);
            out.resident.extend(h.resident);
        }
        out
    }

    fn run(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        cancel: &AtomicBool,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        let Some(i) = self.pick(&req.model) else {
            return Err(EngineError::new(ErrorKind::ModelMissing, format!("no engine here serves {}", req.model)));
        };
        // The light engines that are not serving this request give the device back first.
        for (j, other) in self.engines.iter_mut().enumerate() {
            if j != i && other.yields() {
                other.free();
            }
        }
        let Some(e) = self.engines.get_mut(i) else { return Err(EngineError::new(ErrorKind::Internal, "engine vanished")) };
        e.run(req, blobs, progress, cancel)
    }

    fn free(&mut self) {
        for e in &mut self.engines {
            e.free();
        }
    }

    /// Every engine's `installed` list, merged, with each engine's own index under `engines`.
    fn index(&mut self, dir: &str) -> Value {
        let mut all = Vec::new();
        let mut parts = serde_json::Map::new();
        for e in &mut self.engines {
            let v = e.index(dir);
            for m in installed(&v) {
                if !all.contains(&m) {
                    all.push(m);
                }
            }
            parts.insert(e.name(), v);
        }
        json!({"dir": dir, "installed": all, "engines": parts})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::FakeEngine;

    #[test]
    fn requests_go_to_the_engine_that_lists_the_template() {
        let a = FakeEngine { installed: vec!["sam3.1/segment".into()], ..Default::default() };
        let b = FakeEngine { installed: vec!["qwen-2.1/image".into()], ..Default::default() };
        let mut m = Multi::new(vec![Box::new(a), Box::new(b)]);
        assert_eq!(m.name(), "fake+fake");
        assert_eq!(installed(&m.index("")), vec!["sam3.1/segment".to_string(), "qwen-2.1/image".to_string()]);
        let mut req = RunRequest {
            model: "qwen-2.1/image".into(),
            task: "image".into(),
            prompt: "a fox".into(),
            negative: String::new(),
            seed: 1,
            steps: 1,
            guidance: 1.0,
            width: 4,
            height: 2,
            image: None,
            mask: None,
            references: Vec::new(),
            params: Default::default(),
        };
        let (info, pics) = m.run(&req, &[], &mut |_, _| {}, &AtomicBool::new(false)).unwrap();
        assert_eq!((info.images.len(), pics[0].len()), (1, 4 * 2 * 4));
        assert_eq!(m.health().resident, vec!["qwen-2.1/image".to_string()]);
        req.model = "nobody/has-it".into();
        assert!(matches!(m.run(&req, &[], &mut |_, _| {}, &AtomicBool::new(false)), Err(EngineError { kind: ErrorKind::ModelMissing, .. })));
        m.free();
        assert!(m.health().resident.is_empty());
    }

    #[test]
    fn a_light_engine_gives_the_device_back_when_another_runs() {
        let light = FakeEngine { installed: vec!["sam3.1/segment".into()], light: true, ..Default::default() };
        let heavy = FakeEngine { installed: vec!["qwen-2.1/image".into()], ..Default::default() };
        let mut m = Multi::new(vec![Box::new(light), Box::new(heavy)]);
        let req = |model: &str| RunRequest {
            model: model.into(),
            task: "image".into(),
            prompt: "a fox".into(),
            negative: String::new(),
            seed: 1,
            steps: 1,
            guidance: 1.0,
            width: 4,
            height: 2,
            image: None,
            mask: None,
            references: Vec::new(),
            params: Default::default(),
        };
        let go = |m: &mut Multi, model: &str| m.run(&req(model), &[], &mut |_, _| {}, &AtomicBool::new(false)).map(|_| ());
        go(&mut m, "sam3.1/segment").unwrap();
        go(&mut m, "qwen-2.1/image").unwrap();
        assert_eq!(m.health().resident, vec!["qwen-2.1/image".to_string()], "the light one left");
        // The heavy one stays when the light one runs.
        go(&mut m, "sam3.1/segment").unwrap();
        assert_eq!(m.health().resident, vec!["sam3.1/segment".to_string(), "qwen-2.1/image".to_string()]);
    }
}
