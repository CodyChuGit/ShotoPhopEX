//! Fixtures: the cases every engine runs. JSON files in `fixtures/`, one per suite
//! (`t2i.json`, `edit.json`, `fill.json`, `sam.json`). Geometry is given as fractions of the
//! picture so a case survives a picture being replaced by one of another size.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// A suite: its name and its cases.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suite {
    pub suite: String,
    pub cases: Vec<Case>,
}

/// One case.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Case {
    /// Unique within the suite; the file stem of its outputs.
    pub id: String,
    #[serde(flatten)]
    pub kind: Kind,
}

/// What a case asks for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Kind {
    /// Text to image.
    T2i { prompt: String, seed: u64, width: u32, height: u32 },
    /// The whole picture changed by instruction, with optional reference pictures.
    Edit {
        image: String,
        prompt: String,
        seed: u64,
        #[serde(default)]
        references: Vec<String>,
    },
    /// A rectangle of the picture regenerated from a prompt; `rect` = `[x, y, w, h]` as
    /// fractions of the picture.
    Fill { image: String, prompt: String, seed: u64, rect: [f32; 4] },
    /// A segmentation prompt on a picture.
    Sam {
        image: String,
        prompt: Prompt,
        #[serde(default)]
        negatives: Vec<[f32; 2]>,
    },
}

/// A segmentation prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Prompt {
    Text(String),
    /// `[x, y]` as fractions of the picture.
    Point([f32; 2]),
    /// `[x, y, w, h]` as fractions of the picture.
    Box([f32; 4]),
}

impl Suite {
    pub fn load(path: &Path) -> crate::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let s: Self = serde_json::from_str(&text).map_err(|e| format!("{} is not a suite: {e}", path.display()))?;
        s.check()?;
        Ok(s)
    }

    /// Ids unique, fractions in range, seeds present.
    pub fn check(&self) -> crate::Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        for c in &self.cases {
            if !seen.insert(c.id.as_str()) {
                return Err(format!("suite {}: case id `{}` twice", self.suite, c.id));
            }
            let unit = |v: f32| (0.0..=1.0).contains(&v);
            let ok = match &c.kind {
                Kind::T2i { width, height, prompt, .. } => *width >= 64 && *height >= 64 && !prompt.trim().is_empty(),
                Kind::Edit { prompt, image, .. } => !prompt.trim().is_empty() && !image.is_empty(),
                Kind::Fill { rect, prompt, .. } => rect.iter().all(|v| unit(*v)) && rect[2] > 0.0 && rect[3] > 0.0 && !prompt.trim().is_empty(),
                Kind::Sam { prompt, negatives, .. } => {
                    negatives.iter().all(|p| unit(p[0]) && unit(p[1]))
                        && match prompt {
                            Prompt::Text(t) => !t.trim().is_empty(),
                            Prompt::Point(p) => unit(p[0]) && unit(p[1]),
                            Prompt::Box(b) => b.iter().all(|v| unit(*v)) && b[2] > 0.0 && b[3] > 0.0,
                        }
                }
            };
            if !ok {
                return Err(format!("suite {}: case `{}` is malformed", self.suite, c.id));
            }
        }
        Ok(())
    }
}

impl Case {
    /// The picture a case needs, if any.
    pub fn image(&self) -> Option<&str> {
        match &self.kind {
            Kind::T2i { .. } => None,
            Kind::Edit { image, .. } | Kind::Fill { image, .. } | Kind::Sam { image, .. } => Some(image),
        }
    }

    /// The seed, when the case has one.
    pub fn seed(&self) -> Option<u64> {
        match &self.kind {
            Kind::T2i { seed, .. } | Kind::Edit { seed, .. } | Kind::Fill { seed, .. } => Some(*seed),
            Kind::Sam { .. } => None,
        }
    }
}

/// `[x, y, w, h]` fractions → whole pixels inside a `w × h` picture (at least 1 × 1).
pub fn rect_px(rect: [f32; 4], w: u32, h: u32) -> [u32; 4] {
    let (fw, fh) = (w as f32, h as f32);
    let x = (rect[0] * fw).round().clamp(0.0, fw - 1.0) as u32;
    let y = (rect[1] * fh).round().clamp(0.0, fh - 1.0) as u32;
    let rw = (rect[2] * fw).round().clamp(1.0, fw - x as f32) as u32;
    let rh = (rect[3] * fh).round().clamp(1.0, fh - y as f32) as u32;
    [x, y, rw, rh]
}

/// `[x, y]` fractions → a pixel inside a `w × h` picture.
pub fn point_px(p: [f32; 2], w: u32, h: u32) -> [u32; 2] {
    [(p[0] * w as f32).round().clamp(0.0, w.saturating_sub(1) as f32) as u32, (p[1] * h as f32).round().clamp(0.0, h.saturating_sub(1) as f32) as u32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_suites_load_and_check() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let mut total = 0;
        for name in ["t2i", "edit", "fill", "sam"] {
            let s = Suite::load(&dir.join(format!("{name}.json"))).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(s.suite, name);
            assert!(!s.cases.is_empty(), "{name}");
            total += s.cases.len();
        }
        assert!(total >= 40, "{total} cases");
    }

    #[test]
    fn geometry_resolves_inside_the_picture() {
        assert_eq!(rect_px([0.25, 0.5, 0.5, 0.5], 1000, 800), [250, 400, 500, 400]);
        assert_eq!(rect_px([0.9, 0.9, 0.5, 0.5], 100, 100), [90, 90, 10, 10], "clipped to the picture");
        assert_eq!(point_px([1.0, 0.0], 1024, 768), [1023, 0]);
        let bad: std::result::Result<Suite, _> =
            serde_json::from_str(r#"{"suite":"x","cases":[{"id":"a","kind":"fill","image":"i.png","prompt":"p","seed":1,"rect":[0,0,2,1]}]}"#);
        assert!(bad.map(|s| s.check()).unwrap_or(Err(String::new())).is_err(), "a rect wider than the picture is refused");
    }
}
