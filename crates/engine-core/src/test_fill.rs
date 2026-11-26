//! Test helper: fill the selection (or the whole active layer) with one colour. `edit.fill` moved
//! to `photocraft-engine-paint` in 0.24, which this crate's tests cannot install.

use photocraft_raster::from_rgba_into;

use crate::Session;

/// Fills the active layer with `hex` (`#rrggbb`) where the selection is, blended by its coverage,
/// as `edit.fill {"contents": "color"}` does at 100% Normal; the whole layer without a selection.
pub(crate) fn fill(s: &mut Session, hex: &str) {
    let h = hex.trim_start_matches('#');
    let byte = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).map_or(0.0, |v| f32::from(v) / 255.0);
    let rgba = [byte(0), byte(2), byte(4), 1.0];
    fill_rgba(s, rgba);
}

/// [`fill`] with an RGBA colour (0..=1).
pub(crate) fn fill_rgba(s: &mut Session, rgba: [f32; 4]) {
    s.edit("Fill", |doc, active| {
        let r = doc.bounds();
        let coverage = doc.selection.as_ref().map(|m| m.read_region(r));
        let Some(id) = *active else { return Ok(()) };
        let Some(surf) = doc.layer_mut(id).and_then(|l| l.surface_mut()) else { return Ok(()) };
        let n = surf.channels();
        let mut color = [0.0f32; 8];
        let used = from_rgba_into(&surf.format(), rgba, &mut color).min(n);
        let mut px = surf.read_region(r);
        for (i, p) in px.chunks_mut(n).enumerate() {
            let a = coverage.as_ref().map_or(1.0, |c| c.get(i).copied().unwrap_or(0.0));
            if a <= 0.0 {
                continue;
            }
            for (v, c) in p.iter_mut().zip(&color[..used]) {
                *v = *v * (1.0 - a) + c * a;
            }
        }
        surf.write_region(r, &px);
        Ok(())
    })
    .unwrap_or_else(|e| panic!("fill: {e}"));
}
