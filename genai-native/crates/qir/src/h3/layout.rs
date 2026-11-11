//! The packed sequence of an H3 still (ComfyUI `PackedLayout` for `latent_t = 1`):
//! `[text | reference pictures | audio | target frame]`, with each row's 3-axis position
//! `(t, h, w)` and the AdaLN segments.

use super::dit::{ModSeg, TAG_AUDIO, TAG_TEXT, TAG_VISUAL};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Text,
    RefImage,
    Audio,
    Video,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub text_len: usize,
    /// Per text token: [`TAG_TEXT`], or [`TAG_VISUAL`] inside a picture's vision block.
    pub text_tags: Vec<usize>,
    /// `(kind, start, len)` in packing order.
    pub segments: Vec<(Kind, usize, usize)>,
    /// `(t, h, w)` per row, f64 as ComfyUI builds them.
    pub positions: Vec<[f64; 3]>,
}

/// `linspace((1 - ratio)/2, (1 + ratio)/2, dim/2, endpoint=False) * 32` with
/// `ratio = dim / sqrt_area` (`_axis_from_sqrt_area`, patch 2).
fn axis(dim: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / 2;
    (0..n).map(|k| (k as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0).collect()
}

/// One latent frame's 2x2-patch rows `(h, w)` (row-major) and its w axis (`_frame_grid`).
fn frame_grid(h: usize, w: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let area = ((h * w) as f64).sqrt();
    let hh = axis(h, area);
    let ww = axis(w, area);
    let mut rows = Vec::with_capacity(hh.len() * ww.len());
    for a in &hh {
        for b in &ww {
            rows.push([*a, *b]);
        }
    }
    (rows, ww)
}

impl Layout {
    /// `refs`: each reference latent's `(h, w)`; `audio_t`: audio latent frames (2 for a
    /// P-H3 still, `None` for a model without audio); `lat_h`, `lat_w`: the target latent,
    /// rounded up to even.
    pub fn new(text_tags: Vec<usize>, refs: &[(usize, usize)], audio_t: Option<usize>, lat_h: usize, lat_w: usize) -> Self {
        let text_len = text_tags.len();
        let mut segments = Vec::new();
        let mut positions: Vec<[f64; 3]> = (0..text_len).map(|i| [i as f64, 0.0, 0.0]).collect();
        segments.push((Kind::Text, 0, text_len));
        let mut row = text_len;
        let (frame, w_grid) = frame_grid(lat_h, lat_w);
        let mut cursor = text_len as f64;
        for &(rh, rw) in refs {
            let (rf, _) = frame_grid(rh, rw);
            for p in &rf {
                positions.push([cursor, p[0], p[1]]);
            }
            segments.push((Kind::RefImage, row, rf.len()));
            row += rf.len();
            cursor += 1.0;
        }
        if let Some(at) = audio_t {
            if at > 0 {
                let (lo, hi) = (w_grid[0], *w_grid.last().unwrap());
                for ch in 0..2 {
                    for j in 0..at {
                        positions.push([cursor + j as f64, 0.0, if ch == 0 { lo } else { hi }]);
                    }
                }
                segments.push((Kind::Audio, row, 2 * at));
                row += 2 * at;
            }
        }
        for p in &frame {
            positions.push([cursor, p[0], p[1]]);
        }
        segments.push((Kind::Video, row, frame.len()));
        Self { text_len, text_tags, segments, positions }
    }

    pub fn seq_len(&self) -> usize {
        self.positions.len()
    }

    fn span(&self, kind: Kind) -> (usize, usize) {
        self.segments.iter().find(|s| s.0 == kind).map(|s| (s.1, s.2)).unwrap_or((0, 0))
    }

    pub fn video_span(&self) -> (usize, usize) {
        self.span(Kind::Video)
    }

    pub fn audio_span(&self) -> (usize, usize) {
        self.span(Kind::Audio)
    }

    /// AdaLN segments: text runs by tag at the target's timestep row, references at theirs,
    /// audio and the target at the target's.
    pub fn mod_segments(&self, t_row: usize, ref_row: usize) -> Vec<ModSeg> {
        let mut out = Vec::new();
        for &(kind, start, len) in &self.segments {
            if len == 0 {
                continue;
            }
            match kind {
                Kind::Text => {
                    let mut run = 0;
                    for i in 1..=len {
                        if i == len || self.text_tags[i] != self.text_tags[run] {
                            out.push(ModSeg { start: start + run, len: i - run, row: t_row * 3 + self.text_tags[run] });
                            run = i;
                        }
                    }
                }
                Kind::RefImage => out.push(ModSeg { start, len, row: ref_row * 3 + TAG_VISUAL }),
                Kind::Audio => out.push(ModSeg { start, len, row: t_row * 3 + TAG_AUDIO }),
                Kind::Video => out.push(ModSeg { start, len, row: t_row * 3 + TAG_VISUAL }),
            }
        }
        out
    }
}

/// Text-only tags.
pub fn text_tags(len: usize) -> Vec<usize> {
    vec![TAG_TEXT; len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_axis_is_centred() {
        // a square frame: ratio 1, coordinates k * 32 / n
        let a = axis(8, 8.0);
        assert_eq!(a, vec![0.0, 8.0, 16.0, 24.0]);
    }

    #[test]
    fn still_layout_matches_comfy_order() {
        let lay = Layout::new(text_tags(3), &[(4, 4)], Some(2), 4, 6);
        // text 3 + ref 4 + audio 4 + target 6
        assert_eq!(lay.seq_len(), 3 + 4 + 4 + 6);
        assert_eq!(lay.positions[3][0], 3.0); // first ref at t = text_len
        let (aa, an) = lay.audio_span();
        assert_eq!((aa, an), (7, 4));
        assert_eq!(lay.positions[aa][0], 4.0); // cursor after one ref
        assert_eq!(lay.positions[aa + 1][0], 5.0);
        assert_eq!(lay.positions[aa + 2][0], 4.0); // second channel restarts
        let (va, _) = lay.video_span();
        assert_eq!(lay.positions[va][0], 4.0);
        let segs = lay.mod_segments(0, 1);
        assert_eq!(segs[0].row, TAG_TEXT);
        assert_eq!(segs[1].row, 3 + TAG_VISUAL);
        assert_eq!(segs[2].row, TAG_AUDIO);
    }
}
