//! Pixel helpers for the generative commands: request sizes, resampling, feathering, dilation,
//! tone matching and the side-by-side pictures the judges see.

use photocraft_doc::Layer;
use photocraft_genai::{Gray8, Rgba8};
use photocraft_geom::Rect;

use crate::{EngineError, Result};

use super::{MAX_FILL_REQUEST_PIXELS, MAX_REQUEST_PIXELS, MIN_FILL_REQUEST_SIDE, MIN_SIDE, SIZE_STEP, gen_err};

/// The layer mask of a soft edge: the feathered request mask, its ramp dithered with
/// per-pixel noise (zero inside and outside, strongest mid-ramp) so the transition has no
/// visible line and no banding. Deterministic for a seed, so variations differ.
pub(super) fn dither_edge(mask: &[u8], seed: u64) -> Vec<f32> {
    mask.iter()
        .enumerate()
        .map(|(i, v)| {
            let m = f32::from(*v) / 255.0;
            if *v == 0 || *v == 255 {
                return m;
            }
            // splitmix64 of the pixel index and the seed → a uniform value in 0..1.
            let mut z = (i as u64).wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let n = (z >> 40) as f32 / (1u64 << 24) as f32;
            let amplitude = 0.6 * (1.0 - (2.0 * m - 1.0).abs());
            (m + (n - 0.5) * amplitude).clamp(0.0, 1.0)
        })
        .collect()
}

/// The request rectangle of a fill over `bounds`: the bounds grown by `margin` of their longer
/// side, on the canvas, on the 16-px grid the VAE and the patch grid want (so a template that
/// samples at the request's own size needs no padding and the result no resampling); refused
/// above the request cap.
pub(crate) fn fill_rect(bounds: Rect, margin: f64, canvas: Rect) -> Result<Rect> {
    let longer = i64::from(bounds.width()).max(i64::from(bounds.height()));
    let m = (margin * longer as f64).round().clamp(0.0, 65_536.0) as i32;
    let rect = Rect::new(bounds.x0.saturating_sub(m), bounds.y0.saturating_sub(m), bounds.x1.saturating_add(m), bounds.y1.saturating_add(m)).intersect(&canvas);
    if rect.is_empty() {
        return Err(EngineError::Other("the selection is empty".into()));
    }
    let rect = align_to_grid(rect, canvas);
    let pixels = u64::from(rect.width()) * u64::from(rect.height());
    if pixels > MAX_REQUEST_PIXELS {
        return Err(EngineError::Other(format!(
            "the selected area with its margin is {}×{} pixels; Generative Fill handles up to 16 megapixels at a time: select a smaller area or lower `margin`",
            rect.width(),
            rect.height()
        )));
    }
    Ok(rect)
}

/// `cov` (one value per pixel of `bounds`) laid over `rect`, zero elsewhere.
pub(super) fn embed_coverage(bounds: Rect, cov: &[f32], rect: Rect) -> Vec<f32> {
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let bw = bounds.width().max(1) as usize;
    let mut out = vec![0.0f32; w * h];
    for (i, c) in cov.iter().enumerate() {
        if *c <= 0.0 {
            continue;
        }
        let (x, y) = (bounds.x0 + (i % bw) as i32, bounds.y0 + (i / bw) as i32);
        if rect.contains(x, y)
            && let Some(o) = out.get_mut((y - rect.y0) as usize * w + (x - rect.x0) as usize)
        {
            *o = *c;
        }
    }
    out
}

/// Grow `rect` inside `canvas` until its sides are multiples of [`SIZE_STEP`], extending right
/// and down first, then left and up. A canvas too small for a side leaves that side as it is
/// (the server crops to its grid and the result is resampled back).
pub(crate) fn align_to_grid(rect: Rect, canvas: Rect) -> Rect {
    let grow = |lo: i32, hi: i32, clo: i32, chi: i32| -> (i32, i32) {
        let len = (hi - lo).max(1) as u32;
        let want = (len.div_ceil(SIZE_STEP) * SIZE_STEP) as i32;
        if want > chi - clo {
            return (lo, hi);
        }
        let hi2 = (lo + want).min(chi);
        let lo2 = (hi2 - want).max(clo);
        (lo2, lo2 + want)
    };
    let (x0, x1) = grow(rect.x0, rect.x1, canvas.x0, canvas.x1);
    let (y0, y1) = grow(rect.y0, rect.y1, canvas.y0, canvas.y1);
    Rect::new(x0, y0, x1, y1)
}

/// The size a `w`×`h` fill request is sent at: scaled down (aspect kept, multiples of 16) over
/// [`MAX_FILL_REQUEST_PIXELS`], scaled up when the longer side is under
/// [`MIN_FILL_REQUEST_SIDE`], otherwise unchanged.
pub(crate) fn request_size(w: u32, h: u32) -> (u32, u32) {
    request_size_within(w, h, MAX_FILL_REQUEST_PIXELS)
}

/// [`request_size`] with its own pixel cap (`maxPixels`).
pub(crate) fn request_size_within(w: u32, h: u32, max_pixels: u64) -> (u32, u32) {
    let (w, h) = fit_pixels(w, h, max_pixels);
    let longer = w.max(h);
    if longer == 0 || longer >= MIN_FILL_REQUEST_SIDE {
        return (w, h);
    }
    let scale = f64::from(MIN_FILL_REQUEST_SIDE) / f64::from(longer);
    let fit = |v: u32| (((f64::from(v) * scale).round() as u32) / SIZE_STEP * SIZE_STEP).max(MIN_SIDE);
    (fit(w), fit(h))
}

/// `w`×`h` shrunk (aspect kept, multiples of 16, at least 64 on a side) until it has at most
/// `max_pixels` pixels; unchanged when it already fits.
pub(crate) fn fit_pixels(w: u32, h: u32, max_pixels: u64) -> (u32, u32) {
    let pixels = u64::from(w) * u64::from(h);
    if pixels <= max_pixels || w == 0 || h == 0 {
        return (w, h);
    }
    let scale = (max_pixels as f64 / pixels as f64).sqrt();
    let fit = |v: u32| (((f64::from(v) * scale) as u32) / SIZE_STEP * SIZE_STEP).max(MIN_SIDE);
    (fit(w), fit(h))
}

/// Lanczos-3 kernel, widened by `scale` (> 1 when shrinking) so downsampling averages instead of
/// aliasing.
pub(super) fn lanczos3(x: f32) -> f32 {
    let x = x.abs();
    if x < 1e-6 {
        1.0
    } else if x >= 3.0 {
        0.0
    } else {
        let px = std::f32::consts::PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    }
}

/// Triangle (tent) kernel: no negative lobes, for masks.
pub(super) fn triangle(x: f32) -> f32 {
    (1.0 - x.abs()).max(0.0)
}

/// A resampling kernel and its radius in source pixels (at 1:1).
#[derive(Clone, Copy)]
pub(super) struct Filter {
    pub(super) kernel: fn(f32) -> f32,
    pub(super) radius: f32,
}

pub(super) const LANCZOS: Filter = Filter { kernel: lanczos3, radius: 3.0 };

pub(super) const TENT: Filter = Filter { kernel: triangle, radius: 1.0 };

/// Per-output-index taps of a separable resampler: (first source index, normalised weights).
pub(super) fn taps(src_len: usize, dst_len: usize, f: Filter) -> Vec<(usize, Vec<f32>)> {
    let scale = src_len as f32 / dst_len as f32;
    let widen = scale.max(1.0);
    let support = f.radius * widen;
    (0..dst_len)
        .map(|i| {
            let centre = (i as f32 + 0.5) * scale - 0.5;
            let lo = ((centre - support).floor().max(0.0)) as usize;
            let hi = ((centre + support).ceil() as usize).min(src_len.saturating_sub(1));
            let mut weights: Vec<f32> = (lo..=hi).map(|j| (f.kernel)((j as f32 - centre) / widen)).collect();
            let sum: f32 = weights.iter().sum();
            if sum.abs() > 1e-6 {
                for w in &mut weights {
                    *w /= sum;
                }
            } else if let Some(w) = weights.first_mut() {
                *w = 1.0;
            }
            (lo, weights)
        })
        .collect()
}

/// Separable resample of interleaved 8-bit samples (`ch` per pixel) from `sw`×`sh` to `dw`×`dh`.
pub(super) fn resample_u8(src: &[u8], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    if ch == 0 {
        return Vec::new();
    }
    let xt = taps(sw, dw, f);
    let yt = taps(sh, dh, f);
    // Horizontal pass into f32 rows, then vertical.
    let mut mid = vec![0.0f32; dw * sh * ch];
    for y in 0..sh {
        let row = &src[y * sw * ch..((y + 1) * sw * ch).min(src.len())];
        for ((lo, ws), out) in xt.iter().zip(mid[y * dw * ch..(y + 1) * dw * ch].chunks_exact_mut(ch)) {
            for (k, w) in ws.iter().enumerate() {
                let start = ((lo + k) * ch).min(row.len().saturating_sub(ch));
                match row.get(start..start + ch) {
                    Some(px) => {
                        for (o, p) in out.iter_mut().zip(px) {
                            *o += w * f32::from(*p);
                        }
                    }
                    // A row shorter than one pixel: what there is of it.
                    None => {
                        for (o, p) in out.iter_mut().zip(row.iter().skip(start)) {
                            *o += w * f32::from(*p);
                        }
                    }
                }
            }
        }
    }
    // The vertical pass a whole output row at a time: each sample sums the same taps in the
    // same order as one by one, but reads the source rows in sequence instead of a column.
    let row_len = dw * ch;
    let mut dst = vec![0u8; row_len * dh];
    let mut acc = vec![0.0f32; row_len];
    for (out, (lo, ws)) in dst.chunks_exact_mut(row_len.max(1)).zip(&yt) {
        acc.fill(0.0);
        for (k, w) in ws.iter().enumerate() {
            let Some(row) = mid.get((lo + k) * row_len..(lo + k + 1) * row_len) else { continue };
            for (a, m) in acc.iter_mut().zip(row) {
                *a += w * m;
            }
        }
        for (d, a) in out.iter_mut().zip(&acc) {
            *d = (a + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

/// Lanczos-3 resize (both ways), for requests over the size cap and for a backend that returns
/// a different size than it was given.
pub(crate) fn resize_rgba8(src: &Rgba8, w: u32, h: u32) -> Result<Rgba8> {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return Err(EngineError::Other("cannot resize an empty image".into()));
    }
    if (src.width, src.height) == (w, h) {
        return Ok(src.clone());
    }
    let out = resample_u8(&src.data, src.width as usize, src.height as usize, 4, w as usize, h as usize, LANCZOS);
    Rgba8::new(w, h, out).map_err(gen_err)
}

/// `img` at `w`×`h`: as it is when it already has that size, else resampled ([`resize_rgba8`]).
pub(crate) fn rgba8_at(img: Rgba8, w: u32, h: u32) -> Result<Rgba8> {
    if (img.width, img.height) == (w, h) { Ok(img) } else { resize_rgba8(&img, w, h) }
}

/// A mask at `w`×`h`: as it is when it already has that size, else resampled ([`resize_gray8`]).
pub(crate) fn gray8_at(mask: Gray8, w: u32, h: u32) -> Result<Gray8> {
    if (mask.width, mask.height) == (w, h) { Ok(mask) } else { resize_gray8(&mask, w, h) }
}

/// Soften a mask outward: full coverage at its edge falling to nothing about `2 × radius` pixels
/// beyond it, the inside untouched (two box blurs ≈ a triangle filter, the outer half of its
/// ramp doubled, then `max` with the original). The same shape ComfyUI's GrowMask + FeatherMask
/// make. As the latent noise mask it has the model re-render the band around the area and blend
/// it, with no step at the edge: a ramp that started at half coverage there (the first version)
/// left half of the tone difference as a line along the edge. As the layer mask (dithered, see
/// `dither_edge`) it fades the result into its surroundings across the same band.
pub(crate) fn feather_outward(mask: &Gray8, radius: usize) -> Result<Gray8> {
    let (w, h) = (mask.width as usize, mask.height as usize);
    if radius == 0 || w == 0 || h == 0 {
        return Ok(mask.clone());
    }
    let mut cur: Vec<f32> = mask.data.iter().map(|v| f32::from(*v)).collect();
    for _ in 0..2 {
        cur = box_blur(&cur, w, h, radius, true);
        cur = box_blur(&cur, w, h, radius, false);
    }
    let out: Vec<u8> = cur.iter().zip(&mask.data).map(|(b, o)| (*o).max((b * 2.0 + 0.5).clamp(0.0, 255.0) as u8)).collect();
    Gray8::new(mask.width, mask.height, out).map_err(gen_err)
}

/// One box-blur pass of radius `r` along rows (`horizontal`) or columns, edges clamped.
pub(super) fn box_blur(src: &[f32], w: usize, h: usize, r: usize, horizontal: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    if w == 0 || h == 0 || src.len() != w * h {
        return out;
    }
    let window = (2 * r + 1) as f32;
    if horizontal {
        for (row, out) in src.chunks_exact(w).zip(out.chunks_exact_mut(w)) {
            // Running sum over a window clamped to the row's ends.
            let mut sum = 0.0f32;
            for v in &row[..=r.min(w - 1)] {
                sum += v;
            }
            sum += row[0] * r as f32;
            for (i, o) in out.iter_mut().enumerate() {
                *o = sum / window;
                sum += row[(i + r + 1).min(w - 1)] - row[i.saturating_sub(r)];
            }
        }
    } else {
        // Every column's running sum at once, a row at a time (the same sums as column by
        // column, read in memory order).
        let row = |y: usize| &src[y * w..(y + 1) * w];
        let mut sum = vec![0.0f32; w];
        for y in 0..=r.min(h - 1) {
            for (s, v) in sum.iter_mut().zip(row(y)) {
                *s += v;
            }
        }
        for (s, v) in sum.iter_mut().zip(row(0)) {
            *s += v * r as f32;
        }
        for (y, out) in out.chunks_exact_mut(w).enumerate() {
            for (o, s) in out.iter_mut().zip(&sum) {
                *o = s / window;
            }
            let (entering, leaving) = (row((y + r + 1).min(h - 1)), row(y.saturating_sub(r)));
            for ((s, e), l) in sum.iter_mut().zip(entering).zip(leaving) {
                *s += e - l;
            }
        }
    }
    out
}

/// The outward feather of a fill's request mask: 4 % of the longer side, 6 to 48 pixels. The
/// model re-renders that band around the selection and the soft layer mask fades across it.
pub(crate) fn feather_radius(w: u32, h: u32) -> usize {
    ((f64::from(w.max(h)) * 0.04).round() as usize).clamp(6, 48)
}

/// The feather of an expand's request mask into the picture: 4 % of the longer side, 8 to 80
/// pixels (a band of twice that). The model re-renders that band of the original, so the new
/// area's tone is carried across the old edge instead of meeting it. Measured on the 1024-px
/// lighthouse: at 8 % the model took the licence to move the horizon inside the band (a ghosted
/// double horizon at one seed), at 2 % the old edge still read as a line; 4 % did neither.
pub(crate) fn outpaint_feather_radius(w: u32, h: u32) -> usize {
    ((f64::from(w.max(h)) * 0.04).round() as usize).clamp(8, 80)
}

/// Mask resize with a tent filter (no ringing, coverage stays within 0..=255).
pub(crate) fn resize_gray8(src: &Gray8, w: u32, h: u32) -> Result<Gray8> {
    if src.width == 0 || src.height == 0 || w == 0 || h == 0 {
        return Err(EngineError::Other("cannot resize an empty mask".into()));
    }
    if (src.width, src.height) == (w, h) {
        return Ok(src.clone());
    }
    let out = resample_u8(&src.data, src.width as usize, src.height as usize, 1, w as usize, h as usize, TENT);
    Gray8::new(w, h, out).map_err(gen_err)
}

/// Write RGBA8 pixels into a raster layer over `rect`, converting to the layer's own format
/// (gray, CMYK, 16-bit, float…); `src` must have `rect`'s size.
pub(crate) fn write_rgba8(layer: &mut Layer, src: &Rgba8, rect: Rect) -> Result<()> {
    if (src.width, src.height) != (rect.width(), rect.height()) {
        return Err(EngineError::Other("internal error: the image and its rectangle differ in size".into()));
    }
    let surf = crate::pixels_mut(layer)?;
    let fmt = surf.format();
    let n = fmt.channels();
    let mut px = vec![0.0f32; src.width as usize * src.height as usize * n];
    let mut tmp = [0.0f32; 8];
    for (i, s4) in src.data.as_chunks::<4>().0.iter().enumerate() {
        let rgba = [f32::from(s4[0]) / 255.0, f32::from(s4[1]) / 255.0, f32::from(s4[2]) / 255.0, f32::from(s4[3]) / 255.0];
        let used = photocraft_raster::from_rgba_into(&fmt, rgba, &mut tmp).min(n);
        if let (Some(dst), Some(src)) = (px.get_mut(i * n..i * n + used), tmp.get(..used)) {
            dst.copy_from_slice(src);
        }
    }
    surf.write_region(rect, &px);
    Ok(())
}

/// `image` with the pixels where `mask` is over half flattened to mid-grey (what AusBoss's
/// prompt writer is shown); `image` as it is when the sizes differ.
pub(super) fn grey_where(image: &Rgba8, mask: &Gray8) -> Rgba8 {
    let mut out = image.clone();
    if (mask.width, mask.height) != (image.width, image.height) {
        return out;
    }
    for (px, m) in out.data.as_chunks_mut::<4>().0.iter_mut().zip(&mask.data) {
        if *m > 127 {
            px[..3].copy_from_slice(&[128, 128, 128]);
        }
    }
    out
}

/// A whole-picture route's result moved back onto the picture outside the mask
/// (`generate_realign`), when it came back the picture's size.
pub(super) fn realigned(out: Rgba8, picture: &Rgba8, mask: &Gray8) -> Rgba8 {
    if (out.width, out.height) != (picture.width, picture.height) {
        return out;
    }
    let d = crate::generate_realign::drift(&out, picture, Some(mask), crate::generate_realign::max_shift(out.width, out.height));
    crate::generate_realign::shifted(&out, picture, d)
}

/// How far a result's fill is from its own edge: the distance between the mean colour of the
/// result inside the hole (mask at least [`TONE_MATCH_INSIDE`]) and the mean colour of the
/// picture in the band around it (mask above zero and below that), per channel, 0..1, plus
/// [`EDGE_TEXTURE_WEIGHT`] times the difference in texture (the mean absolute luminance
/// step between neighbours) between the two: a seed that drew a second wire on a plain sky
/// has the sky's colour but not its smoothness, and a plain fill on a checkered cloth has
/// the colour but not the texture. A removal wants its hole to look like its edge, so of
/// several seeds the lowest wins.
pub(super) fn edge_mismatch(result: &Rgba8, picture: &Rgba8, mask: &Gray8) -> f64 {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let n = w * h;
    if result.data.len() != n * 4 || picture.data.len() != n * 4 || mask.data.len() != n || w < 2 || h < 2 {
        return 0.0;
    }
    let luma = |d: &[u8], i: usize| (0.299 * f64::from(d[i * 4]) + 0.587 * f64::from(d[i * 4 + 1]) + 0.114 * f64::from(d[i * 4 + 2])) / 255.0;
    let (mut hole, mut band) = ([0.0f64; 3], [0.0f64; 3]);
    let (mut hn, mut bn) = (0usize, 0usize);
    let (mut hole_tex, mut band_tex) = (0.0f64, 0.0f64);
    let (mut htn, mut btn) = (0usize, 0usize);
    for (i, m) in mask.data.iter().enumerate() {
        let (x, y) = (i % w, i / w);
        if *m >= TONE_MATCH_INSIDE {
            for (acc, v) in hole.iter_mut().zip(&result.data[i * 4..i * 4 + 3]) {
                *acc += f64::from(*v);
            }
            hn += 1;
            if x + 1 < w && y + 1 < h {
                hole_tex += (luma(&result.data, i) - luma(&result.data, i + 1)).abs() + (luma(&result.data, i) - luma(&result.data, i + w)).abs();
                htn += 1;
            }
        } else if *m > 0 {
            for (acc, v) in band.iter_mut().zip(&picture.data[i * 4..i * 4 + 3]) {
                *acc += f64::from(*v);
            }
            bn += 1;
            if x + 1 < w && y + 1 < h {
                band_tex += (luma(&picture.data, i) - luma(&picture.data, i + 1)).abs() + (luma(&picture.data, i) - luma(&picture.data, i + w)).abs();
                btn += 1;
            }
        }
    }
    if hn == 0 || bn == 0 || htn == 0 || btn == 0 {
        return 0.0;
    }
    let colour = (0..3).map(|c| (hole[c] / hn as f64 - band[c] / bn as f64) / 255.0).map(|d| d * d).sum::<f64>().sqrt();
    let texture = (hole_tex / htn as f64 - band_tex / btn as f64).abs();
    colour + EDGE_TEXTURE_WEIGHT * texture
}

/// How much a texture difference weighs against a colour difference in [`edge_mismatch`]
/// (a luminance step is a few hundredths on a smooth surface, a few tenths on a pattern).
pub(super) const EDGE_TEXTURE_WEIGHT: f64 = 2.0;

/// What empty canvas is painted with when there is no picture to continue: mid-grey.
pub(super) const PREFILL: [u8; 4] = [128, 128, 128, 255];

/// Widen a coverage mask (one value per pixel, row-major) by `r` pixels: the maximum over a
/// square window, in two separable passes, so a selection's edge moves outward by `r`.
pub(crate) fn dilate(coverage: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 || coverage.len() != w * h {
        return coverage.to_vec();
    }
    // Rows, then columns (separable), each in constant time a pixel whatever the radius.
    let (mut fwd, mut bwd) = (Vec::new(), Vec::new());
    let mut rows = vec![0.0f32; w * h];
    for (src, out) in coverage.chunks_exact(w).zip(rows.chunks_exact_mut(w)) {
        max_line(src, r, out, &mut fwd, &mut bwd);
    }
    max_columns(&rows, w, h, r)
}

/// [`max_line`] down every column at once, a row at a time.
fn max_columns(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let k = 2 * r + 1;
    let padded = h + 2 * r;
    let zero = vec![0.0f32; w];
    let row = |y: usize| if y < r || y >= r + h { &zero[..] } else { &src[(y - r) * w..(y - r + 1) * w] };
    let mut fwd = vec![0.0f32; padded * w];
    let mut bwd = vec![0.0f32; padded * w];
    for y in 0..padded {
        let (done, cur) = fwd.split_at_mut(y * w);
        let cur = &mut cur[..w];
        if y.is_multiple_of(k) {
            cur.copy_from_slice(row(y));
        } else {
            for ((c, p), v) in cur.iter_mut().zip(&done[(y - 1) * w..]).zip(row(y)) {
                *c = p.max(*v);
            }
        }
    }
    for y in (0..padded).rev() {
        let (cur, done) = bwd.split_at_mut((y + 1) * w);
        let cur = &mut cur[y * w..];
        if y + 1 == padded || (y + 1).is_multiple_of(k) {
            cur.copy_from_slice(row(y));
        } else {
            for ((c, n), v) in cur.iter_mut().zip(&done[..w]).zip(row(y)) {
                *c = n.max(*v);
            }
        }
    }
    let mut out = vec![0.0f32; w * h];
    for (y, o) in out.chunks_exact_mut(w).enumerate() {
        for ((o, b), f) in o.iter_mut().zip(&bwd[y * w..(y + 1) * w]).zip(&fwd[(y + 2 * r) * w..(y + 2 * r + 1) * w]) {
            *o = 0.0f32.max(b.max(*f));
        }
    }
    out
}

/// The running maximum of `line` over a window of `r` either side (clamped to its ends, and
/// never below 0) into `out`, in constant time a sample (van Herk / Gil-Werman: maxima within
/// blocks of the window's width, forward and backward, then one of each per window).
fn max_line(line: &[f32], r: usize, out: &mut [f32], fwd: &mut Vec<f32>, bwd: &mut Vec<f32>) {
    let n = line.len();
    let k = 2 * r + 1;
    // The line padded with r zeros each side: a window clamped to the line is then a full
    // window of the padded one, and the zeros are the 0 every maximum starts from.
    let padded = n + 2 * r;
    let at = |i: usize| if i < r || i >= r + n { 0.0 } else { line[i - r] };
    fwd.clear();
    bwd.clear();
    fwd.resize(padded, 0.0);
    bwd.resize(padded, 0.0);
    for i in 0..padded {
        fwd[i] = if i.is_multiple_of(k) { at(i) } else { fwd[i - 1].max(at(i)) };
    }
    for i in (0..padded).rev() {
        bwd[i] = if i + 1 == padded || (i + 1).is_multiple_of(k) { at(i) } else { bwd[i + 1].max(at(i)) };
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = 0.0f32.max(bwd[i].max(fwd[i + 2 * r]));
    }
}

/// `result` with its tone matched to `original`'s at the hole's edge: the hole is where
/// `mask` is at least [`TONE_MATCH_INSIDE`]; just outside it the model re-rendered the band
/// and the difference between the two pictures there is its tone shift, which a membrane
/// carries smoothly across the hole and takes out (Pérez's seamless cloning with the result
/// as the source). Outside the hole the original comes back (the layer mask fades there
/// anyway). The correction moves a channel at most `most` (0..1; 1.0 is all the way). Both
/// pictures the same size; otherwise `result` as it was.
pub(super) fn tone_matched(result: &Rgba8, original: &Rgba8, mask: &Gray8, most: f32) -> Rgba8 {
    let (w, h) = (result.width as usize, result.height as usize);
    if (original.width, original.height) != (result.width, result.height) || (mask.width, mask.height) != (result.width, result.height) || w < 3 || h < 3 {
        return result.clone();
    }
    // A one-pixel unmasked margin keeps the problem Dirichlet (see `seamless_clone`).
    let hole: Vec<bool> =
        mask.data.iter().enumerate().map(|(i, m)| *m >= TONE_MATCH_INSIDE && i % w > 0 && i % w < w - 1 && i / w > 0 && i / w < h - 1).collect();
    if !hole.iter().any(|h| *h) {
        return result.clone();
    }
    let rgb = |img: &Rgba8| -> Vec<f32> {
        img.data.as_chunks::<4>().0.iter().flat_map(|p| [f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[2]) / 255.0]).collect()
    };
    let matched = photocraft_algo::poisson::seamless_clone(w, h, 3, &rgb(result), &rgb(original), &hole);
    let mut data = Vec::with_capacity(w * h * 4);
    for (px, src) in result.data.as_chunks::<4>().0.iter().zip(matched.as_chunks::<3>().0) {
        // The membrane's correction, at most `most` a channel.
        data.extend(src.iter().zip(px).map(|(v, r)| {
            let r = f32::from(*r) / 255.0;
            ((r + (v - r).clamp(-most, most)).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        }));
        data.push(px[3]);
    }
    Rgba8::new(result.width, result.height, data).unwrap_or_else(|_| result.clone())
}

/// Mask values from here up count as the hole for [`tone_matched`] (the request's mask is
/// 255 inside the selection and feathered outward).
pub(super) const TONE_MATCH_INSIDE: u8 = 200;

/// The most a split's tone matches (its removal's membrane, its grey-box fill's grid) may
/// move a channel, in levels of 255: a fill a few levels off is matched; one that came back
/// in other colours altogether is left showing it to the checks that judge it (and the fake
/// worker's flat colour stays its colour under a test, instead of becoming the picture's).
/// The Remove tools match all the way ([`FillJob::tone_most`]).
pub(crate) const TONE_MATCH_MAX: f32 = 40.0;

/// [`change_band`]: a pixel changed when the two pictures, each box-blurred twice at this
/// radius (texture the model redraws averages out, a shadow does not), differ by more than
/// [`SHADOW_CHANGE`] in some channel (0..1).
pub(super) const SHADOW_BLUR: usize = 3;

pub(crate) const SHADOW_CHANGE: f32 = 0.08;

/// The change an edit's band counts: a swapped garment against what it replaced (a yellow coat
/// over black jeans), not the light re-rendering a whole-picture route gives the rest (the 2.1
/// edit's jeans, which ran past the reach and refused the band at the shadow's 0.08).
pub(crate) const EDIT_BAND_CHANGE: f32 = 0.25;

/// How far from the hole the band may reach, in the hole's longer side (at least 16 px).
pub(super) const SHADOW_REACH: f64 = 0.75;

/// The band is taken only when it is at least this share of the hole (smaller is the model's
/// noise) and at most that (larger: the model shifted the whole picture's tone).
pub(super) const SHADOW_MIN_SHARE: f64 = 0.03;

pub(crate) const SHADOW_MAX_SHARE: f64 = 2.0;

/// The most an edit's or a fill's band may add, in the selection's area: a garment longer than
/// the one it replaces, not a second subject.
pub(crate) const EDIT_BAND_MAX_SHARE: f64 = 0.75;

/// What changed with the object, outside the hole (`mask` at [`TONE_MATCH_INSIDE`] or more):
/// the pixels where `result` differs from `original` ([`SHADOW_CHANGE`] after a blur),
/// connected to the hole's feathered edge, within [`SHADOW_REACH`] of it, as a mask (255 on
/// the band, widened by 2 px), or `None` when nothing or too much changed. A route that
/// regenerates the whole request removes a cast shadow or a reflection with its object; the
/// hole's mask kept it from showing (AusBoss's differential composite, catalogue §2.2). A band
/// larger than `max_share` of the hole is refused; `threshold` is the change that counts
/// ([`SHADOW_CHANGE`] for a shadow, [`EDIT_BAND_CHANGE`] for an edit).
pub(crate) fn change_band(result: &Rgba8, original: &Rgba8, mask: &Gray8, max_share: f64, threshold: f32) -> Option<Gray8> {
    let (w, h) = (result.width as usize, result.height as usize);
    if (original.width, original.height) != (result.width, result.height) || (mask.width, mask.height) != (result.width, result.height) || w < 3 || h < 3 {
        return None;
    }
    let blurred = |img: &Rgba8, c: usize| -> Vec<f32> {
        let mut v: Vec<f32> = img.data.as_chunks::<4>().0.iter().map(|p| f32::from(p.get(c).copied().unwrap_or(0)) / 255.0).collect();
        for _ in 0..2 {
            v = box_blur(&v, w, h, SHADOW_BLUR, true);
            v = box_blur(&v, w, h, SHADOW_BLUR, false);
        }
        v
    };
    let mut diff = vec![0.0f32; w * h];
    for c in 0..3 {
        for ((d, a), b) in diff.iter_mut().zip(blurred(result, c)).zip(blurred(original, c)) {
            *d = d.max((a - b).abs());
        }
    }
    let hole: Vec<bool> = mask.data.iter().map(|m| *m >= TONE_MATCH_INSIDE).collect();
    let hole_px = hole.iter().filter(|h| **h).count();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    for (i, _) in hole.iter().enumerate().filter(|(_, h)| **h) {
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    if hole_px == 0 {
        return None;
    }
    let reach = ((x1 - x0).max(y1 - y0) as f64 * SHADOW_REACH).max(16.0) as usize;
    let (lx0, ly0, lx1, ly1) = (x0.saturating_sub(reach), y0.saturating_sub(reach), (x1 + reach).min(w), (y1 + reach).min(h));
    let changed: Vec<f32> = diff
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let (x, y) = (i % w, i / w);
            let inside = hole.get(i).copied().unwrap_or(false);
            f32::from(u8::from(!inside && *d > threshold && x >= lx0 && x < lx1 && y >= ly0 && y < ly1))
        })
        .collect();
    // Specks out (an opening of one pixel), then the part connected to the hole's edge.
    let inverted: Vec<f32> = changed.iter().map(|c| 1.0 - c).collect();
    let eroded: Vec<f32> = dilate(&inverted, w, h, 1).iter().map(|c| 1.0 - c).collect();
    let opened = dilate(&eroded, w, h, 1);
    let open: Vec<bool> = opened.iter().zip(&changed).map(|(o, c)| *o >= 0.5 && *c >= 0.5).collect();
    let mut band = vec![false; w * h];
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| open.get(i).copied().unwrap_or(false) && mask.data.get(i).copied().unwrap_or(0) > 0).collect();
    for &i in &stack {
        if let Some(b) = band.get_mut(i) {
            *b = true;
        }
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let next = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
        for j in next.into_iter().flatten() {
            if open.get(j).copied().unwrap_or(false) && !band.get(j).copied().unwrap_or(true) {
                if let Some(b) = band.get_mut(j) {
                    *b = true;
                }
                stack.push(j);
            }
        }
    }
    // A band that runs into the request's edge (or the reach's) goes on outside it, where the
    // model was not asked: taking it would cut the shadow off in a straight line (the dog's,
    // measured 2026-10-27), so the shadow stays whole instead.
    let on_edge = |i: usize| {
        let (x, y) = (i % w, i / w);
        x == lx0 || y == ly0 || x + 1 == lx1 || y + 1 == ly1
    };
    if band.iter().enumerate().any(|(i, b)| *b && on_edge(i) && mask.data.get(i).copied().unwrap_or(0) == 0) {
        return None;
    }
    // Pixels the hole's own feather covers are the model's re-rendered rim, not a shadow.
    let taken = band.iter().zip(&mask.data).filter(|(b, m)| **b && **m == 0).count();
    if (taken as f64) < SHADOW_MIN_SHARE * hole_px as f64 || (taken as f64) > max_share * hole_px as f64 {
        return None;
    }
    let grown = dilate(&band.iter().map(|b| f32::from(u8::from(*b))).collect::<Vec<f32>>(), w, h, 2);
    Gray8::new(result.width, result.height, grown.iter().map(|g| if *g >= 0.5 { 255 } else { 0 }).collect()).ok()
}

/// `img` with the hole (`mask` at [`SMOOTH_INSIDE`] or more) smoothed over by a membrane from
/// its edge: the colours continue in, with no structure. An edit model copies what its
/// reference shows, so a hole with the object in it is redrawn and a flat grey one painted
/// back as a panel (both measured); a smooth continuation is neither. Unchanged when the
/// sizes differ or the hole is empty.
pub(super) fn smooth_covered(img: &Rgba8, mask: &Gray8) -> Rgba8 {
    let (w, h) = (img.width as usize, img.height as usize);
    if (mask.width, mask.height) != (img.width, img.height) || w < 3 || h < 3 {
        return img.clone();
    }
    let hole: Vec<bool> = mask.data.iter().enumerate().map(|(i, m)| *m >= SMOOTH_INSIDE && i % w > 0 && i % w < w - 1 && i / w > 0 && i / w < h - 1).collect();
    if !hole.iter().any(|h| *h) {
        return img.clone();
    }
    let rgb: Vec<f32> = img.data.as_chunks::<4>().0.iter().flat_map(|p| [f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[2]) / 255.0]).collect();
    let filled = photocraft_algo::poisson::membrane_fill(w, h, 3, &rgb, &hole);
    let mut data = img.data.clone();
    for (i, in_hole) in hole.iter().enumerate() {
        if *in_hole && let (Some(d), Some(s)) = (data.get_mut(i * 4..i * 4 + 3), filled.get(i * 3..i * 3 + 3)) {
            for (dv, sv) in d.iter_mut().zip(s) {
                *dv = (sv.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    Rgba8::new(img.width, img.height, data).unwrap_or_else(|_| img.clone())
}

/// Mask values from here up are the hole [`smooth_covered`] fills: the selection itself
/// (255), not the feathered band outside it.
pub(super) const SMOOTH_INSIDE: u8 = 250;

/// Whether a removal adapter left its paint in the hole: more than [`PAINT_LEFT_SHARE`] of
/// the pixels under `mask` (at [`SMOOTH_INSIDE`] or more) are saturated red in `result`.
pub(super) fn paint_left(result: &Rgba8, mask: &Gray8) -> bool {
    if (mask.width, mask.height) != (result.width, result.height) {
        return false;
    }
    let (mut hole, mut red) = (0usize, 0usize);
    for (i, m) in mask.data.iter().enumerate() {
        if *m < SMOOTH_INSIDE {
            continue;
        }
        hole += 1;
        if let Some(px) = result.data.get(i * 4..i * 4 + 3)
            && px[0] >= 150
            && i32::from(px[0]) - i32::from(px[1].max(px[2])) >= 70
        {
            red += 1;
        }
    }
    hole > 0 && red as f64 / hole as f64 > PAINT_LEFT_SHARE
}

/// The share of the hole that may be saturated red before the adapter's result is out.
pub(super) const PAINT_LEFT_SHARE: f64 = 0.05;

/// `img` with the hole (`mask` at [`SMOOTH_INSIDE`] or more) painted solid in `colour`
/// ("red", "green", "blue", or `#rrggbb`; anything else red): what a removal adapter trained
/// on a highlight expects to see.
pub(super) fn highlight_covered(img: &Rgba8, mask: &Gray8, colour: &str) -> Rgba8 {
    let rgb = match colour.trim().to_ascii_lowercase().as_str() {
        "green" => [0, 255, 0],
        "blue" => [0, 0, 255],
        hex if hex.len() == 7 && hex.starts_with('#') => {
            let v = |i: usize| u8::from_str_radix(hex.get(i..i + 2).unwrap_or("ff"), 16).unwrap_or(255);
            [v(1), v(3), v(5)]
        }
        _ => [255, 0, 0],
    };
    if (mask.width, mask.height) != (img.width, img.height) {
        return img.clone();
    }
    let mut data = img.data.clone();
    for (i, m) in mask.data.iter().enumerate() {
        if *m >= SMOOTH_INSIDE
            && let Some(px) = data.get_mut(i * 4..i * 4 + 4)
        {
            px.copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }
    Rgba8::new(img.width, img.height, data).unwrap_or_else(|_| img.clone())
}

/// Paint the pixels `coverage` selects (half or more) mid-grey: the request image shows a hole
/// where the model is to paint, not the object it is to replace.
pub(super) fn erase_covered(img: &mut Rgba8, coverage: &[f32]) {
    for (px, c) in img.data.as_chunks_mut::<4>().0.iter_mut().zip(coverage) {
        if *c >= 0.5 {
            px.copy_from_slice(&PREFILL);
        }
    }
}

/// `img` cut to `rect` (in its own pixels, clipped to it); `img` itself when the cut is empty.
pub(super) fn crop_rgba8(img: &Rgba8, rect: Rect) -> Rgba8 {
    let r = rect.intersect(&Rect::new(0, 0, img.width as i32, img.height as i32));
    if r.is_empty() {
        return img.clone();
    }
    let (w, h) = (r.width() as usize, r.height() as usize);
    let stride = img.width as usize * 4;
    let mut data = Vec::with_capacity(w * h * 4);
    for y in r.y0..r.y1 {
        let start = y as usize * stride + r.x0 as usize * 4;
        data.extend_from_slice(img.data.get(start..start + w * 4).unwrap_or(&[]));
    }
    Rgba8::new(r.width(), r.height(), data).unwrap_or_else(|_| img.clone())
}

/// The widest a [`side_by_side`]'s right half is made (a sliver of a reference scaled up to a
/// tall picture's height stays a bounded allocation; the request is shrunk to size after).
pub(super) const MAX_SIDE_BY_SIDE_WIDTH: u32 = 8192;

/// `left` and `right` side by side, `right` scaled to `left`'s height with its aspect kept:
/// what the rewriter of a fill from a reference and of a removal looks at. (It was once held to
/// `left`'s width, which squeezed a wide close-up beside narrower surroundings: a boat's hull
/// under its 384-px surroundings at two thirds of its width.)
pub(crate) fn side_by_side(left: &Rgba8, right: &Rgba8) -> Result<Rgba8> {
    let h = left.height.max(1);
    let rw = ((f64::from(right.width) * f64::from(h) / f64::from(right.height.max(1))).round() as u32).clamp(1, MAX_SIDE_BY_SIDE_WIDTH);
    let right = if (right.width, right.height) == (rw, h) { right.clone() } else { resize_rgba8(right, rw, h)? };
    let w = left.width + rw;
    let mut data = Vec::with_capacity(w as usize * h as usize * 4);
    let (ls, rs) = (left.width as usize * 4, rw as usize * 4);
    for y in 0..h as usize {
        data.extend_from_slice(left.data.get(y * ls..(y + 1) * ls).unwrap_or(&[]));
        data.extend_from_slice(right.data.get(y * rs..(y + 1) * rs).unwrap_or(&[]));
    }
    Rgba8::new(w, h, data).map_err(gen_err)
}

/// Paint the pixels outside `inner` (in image coordinates) with the nearest pixel of `inner`, so
/// the canvas the VAE encodes continues the picture's edge colours instead of meeting a grey
/// wall. The model never sees this area as content (its reference is the cropped picture, see
/// `Outpaint`, and the noise mask is full over it), but the VAE's latents next to a wall of grey
/// carry its tone into the picture's edge band, which came back as a dark line along the old
/// edge. The replicated streaks are what the early padded-reference variant turned into
/// content; with the reference cropped they are only ever noised away. (Qwen-Image-2.1's
/// in-context expand template, whose reference is the picture placed inside the canvas, never
/// sees this area either; a mid-grey prefill tried with the whole canvas as the reference came
/// back as black canvas, 2026-10-20.)
/// Everything of `img` outside `inner` flat [`PREFILL`] grey (the Outpaint LoRA's canvas).
pub(super) fn grey_outside(img: &mut Rgba8, inner: Rect) {
    let w = img.width as i32;
    for (i, px) in img.data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (x, y) = (i as i32 % w, i as i32 / w);
        if !(x >= inner.x0 && x < inner.x1 && y >= inner.y0 && y < inner.y1) {
            px.copy_from_slice(&PREFILL);
        }
    }
}

pub(super) fn prefill_outside(img: &mut Rgba8, inner: Rect) {
    let (w, h) = (img.width as i32, img.height as i32);
    let inner = inner.intersect(&Rect::new(0, 0, w, h));
    if inner.x0 == 0 && inner.y0 == 0 && inner.x1 == w && inner.y1 == h {
        return;
    }
    let empty = inner.x1 <= inner.x0 || inner.y1 <= inner.y0;
    let stride = img.width as usize * 4;
    for y in 0..h {
        let sy = if empty { 0 } else { y.clamp(inner.y0, inner.y1 - 1) };
        for x in 0..w {
            if x >= inner.x0 && x < inner.x1 && y >= inner.y0 && y < inner.y1 {
                continue;
            }
            let sx = if empty { 0 } else { x.clamp(inner.x0, inner.x1 - 1) };
            let src = sy as usize * stride + sx as usize * 4;
            let p = match img.data.get(src..src + 4) {
                Some(s) if !empty => [s[0], s[1], s[2], 255],
                _ => PREFILL,
            };
            let dst = y as usize * stride + x as usize * 4;
            if let Some(d) = img.data.get_mut(dst..dst + 4) {
                d.copy_from_slice(&p);
            }
        }
    }
}

#[cfg(test)]
#[path = "generate_pixel_ops_tests.rs"]
mod tests;
