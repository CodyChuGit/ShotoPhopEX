//! The pixel helpers' fast paths against the straightforward versions they replaced: the same
//! bytes (and the same `f32` bits) for every size, channel count and radius tried.

use super::{Filter, LANCZOS, TENT, box_blur, dilate, dim_outside, highlight_covered, resample_u8, taps, unboxed};
use photocraft_genai::Rgba8;
use photocraft_geom::Rect;

fn resample_u8_reference(src: &[u8], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize, f: Filter) -> Vec<u8> {
    let xt = taps(sw, dw, f);
    let yt = taps(sh, dh, f);
    // Horizontal pass into f32 rows, then vertical.
    let mut mid = vec![0.0f32; dw * sh * ch];
    for y in 0..sh {
        let row = &src[y * sw * ch..((y + 1) * sw * ch).min(src.len())];
        for (x, (lo, ws)) in xt.iter().enumerate() {
            let out = &mut mid[(y * dw + x) * ch..(y * dw + x + 1) * ch];
            for (k, w) in ws.iter().enumerate() {
                let start = ((lo + k) * ch).min(row.len().saturating_sub(ch));
                for (o, p) in out.iter_mut().zip(row.iter().skip(start)) {
                    *o += w * f32::from(*p);
                }
            }
        }
    }
    let mut dst = vec![0u8; dw * dh * ch];
    for (y, (lo, ws)) in yt.iter().enumerate() {
        for x in 0..dw {
            for c in 0..ch {
                let mut acc = 0.0f32;
                for (k, w) in ws.iter().enumerate() {
                    acc += w * mid.get(((lo + k) * dw + x) * ch + c).copied().unwrap_or(0.0);
                }
                if let Some(d) = dst.get_mut((y * dw + x) * ch + c) {
                    *d = (acc + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    dst
}

fn box_blur_reference(src: &[f32], w: usize, h: usize, r: usize, horizontal: bool) -> Vec<f32> {
    let (len, lines) = if horizontal { (w, h) } else { (h, w) };
    let at = |line: usize, i: usize| -> f32 {
        let idx = if horizontal { line * w + i } else { i * w + line };
        src.get(idx).copied().unwrap_or(0.0)
    };
    let mut out = vec![0.0f32; src.len()];
    let window = (2 * r + 1) as f32;
    for line in 0..lines {
        // Running sum over a window clamped to the line's ends.
        let mut sum = 0.0f32;
        for i in 0..=r.min(len.saturating_sub(1)) {
            sum += at(line, i);
        }
        sum += at(line, 0) * r as f32;
        for i in 0..len {
            let idx = if horizontal { line * w + i } else { i * w + line };
            if let Some(o) = out.get_mut(idx) {
                *o = sum / window;
            }
            let leaving = at(line, i.saturating_sub(r));
            let entering = at(line, (i + r + 1).min(len.saturating_sub(1)));
            sum += entering - leaving;
        }
    }
    out
}

fn dilate_reference(coverage: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 || coverage.len() != w * h {
        return coverage.to_vec();
    }
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        let (len, lines) = if horizontal { (w, h) } else { (h, w) };
        let at = |line: usize, i: usize| if horizontal { line * w + i } else { i * w + line };
        for line in 0..lines {
            for i in 0..len {
                let lo = i.saturating_sub(r);
                let hi = (i + r).min(len - 1);
                let mut m = 0.0f32;
                for j in lo..=hi {
                    m = m.max(src.get(at(line, j)).copied().unwrap_or(0.0));
                }
                if let Some(o) = out.get_mut(at(line, i)) {
                    *o = m;
                }
            }
        }
        out
    };
    pass(&pass(coverage, true), false)
}

/// A small deterministic generator (the tests need varied input, not quality randomness).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const SIZES: &[(usize, usize)] = &[(1, 1), (1, 7), (7, 1), (3, 3), (17, 5), (64, 48), (101, 37)];

#[test]
fn resample_matches_the_reference_both_ways_and_kernels() {
    let mut g = Lcg(7);
    for &(w, h) in SIZES {
        let rgba: Vec<u8> = (0..w * h * 4).map(|_| g.next() as u8).collect();
        let gray: Vec<u8> = (0..w * h).map(|_| g.next() as u8).collect();
        for (dw, dh) in [(1, 1), (w * 2, h), (w / 2 + 1, h * 3), (w + 3, h / 2 + 1), (w, h)] {
            assert_eq!(resample_u8(&rgba, w, h, 4, dw, dh, LANCZOS), resample_u8_reference(&rgba, w, h, 4, dw, dh, LANCZOS), "rgba {w}x{h} to {dw}x{dh}");
            assert_eq!(resample_u8(&gray, w, h, 1, dw, dh, TENT), resample_u8_reference(&gray, w, h, 1, dw, dh, TENT), "gray {w}x{h} to {dw}x{dh}");
        }
    }
}

#[test]
fn box_blur_and_dilate_match_the_reference_at_every_radius() {
    let mut g = Lcg(11);
    for &(w, h) in SIZES {
        let cov: Vec<f32> = (0..w * h).map(|_| if g.next().is_multiple_of(3) { 1.0 } else { (g.next() % 1000) as f32 / 1000.0 }).collect();
        for r in [0, 1, 2, 3, 6, 15, 48, 200] {
            for horizontal in [true, false] {
                assert_eq!(bits(&box_blur(&cov, w, h, r, horizontal)), bits(&box_blur_reference(&cov, w, h, r, horizontal)), "blur {w}x{h} r {r} {horizontal}");
            }
            assert_eq!(bits(&dilate(&cov, w, h, r)), bits(&dilate_reference(&cov, w, h, r)), "dilate {w}x{h} r {r}");
        }
    }
}

#[test]
fn empty_and_mismatched_inputs_do_not_panic() {
    assert!(resample_u8(&[], 0, 0, 0, 4, 4, LANCZOS).is_empty());
    assert_eq!(box_blur(&[0.5; 3], 2, 2, 1, false), vec![0.0; 3]);
    assert_eq!(dilate(&[0.5; 3], 2, 2, 1), vec![0.5; 3]);
}

/// The ops at a 2-megapixel size, against the reference: run with
/// `cargo test --release -p photocraft-engine --lib pixel_ops_speed -- --ignored --nocapture`.
#[test]
#[ignore = "timing, not a check"]
fn pixel_ops_speed() {
    let mut g = Lcg(3);
    let (w, h) = (1920, 1080);
    let rgba: Vec<u8> = (0..w * h * 4).map(|_| g.next() as u8).collect();
    let cov: Vec<f32> = (0..w * h).map(|_| (g.next() % 1000) as f32 / 1000.0).collect();
    let time = |what: &str, run: &dyn Fn() -> usize| {
        let t = std::time::Instant::now();
        let n = run();
        eprintln!("{what:<32} {:>8.1} ms ({n})", t.elapsed().as_secs_f64() * 1000.0);
    };
    time("resize reference", &|| resample_u8_reference(&rgba, w, h, 4, 1365, 768, LANCZOS).len());
    time("resize", &|| resample_u8(&rgba, w, h, 4, 1365, 768, LANCZOS).len());
    time("vertical blur r24 reference", &|| box_blur_reference(&cov, w, h, 24, false).len());
    time("vertical blur r24", &|| box_blur(&cov, w, h, 24, false).len());
    for r in [2, 16, 64] {
        time(&format!("dilate r{r} reference"), &|| dilate_reference(&cov, w, h, r).len());
        time(&format!("dilate r{r}"), &|| dilate(&cov, w, h, r).len());
    }
}

#[test]
fn dim_outside_darkens_only_what_is_not_selected() {
    // A 4×2 crop at (10, 20); the selection covers (11, 20) and (12, 21), its area (11, 20)-(13, 22).
    let mut img = Rgba8::new(4, 2, [200u8, 100, 40, 255].repeat(8)).unwrap();
    let area = Rect::new(11, 20, 13, 22);
    let coverage = [1.0, 0.0, 0.2, 0.9];
    dim_outside(&mut img, Rect::new(10, 20, 14, 22), &coverage, area);
    let kept: Vec<bool> = img.data.as_chunks::<4>().0.iter().map(|p| p[0] == 200).collect();
    assert_eq!(kept, [false, true, false, false, false, false, true, false]);
    // Darkened to a quarter, the alpha untouched.
    assert_eq!(&img.data[..4], &[50, 25, 10, 255]);
    // A coverage of the wrong length leaves the picture as it is.
    let mut same = Rgba8::new(1, 1, vec![9, 9, 9, 255]).unwrap();
    dim_outside(&mut same, Rect::new(0, 0, 1, 1), &[1.0, 1.0], Rect::new(0, 0, 1, 1));
    assert_eq!(same.data, [9, 9, 9, 255]);
}

#[test]
fn the_bbox_remover_s_box_sits_outside_the_hole_and_a_left_trace_goes_back_to_the_picture() {
    use photocraft_genai::Gray8;
    let (w, h) = (40u32, 30u32);
    let plain = Rgba8::new(w, h, vec![100; (w * h * 4) as usize]).unwrap();
    let mut m = vec![0u8; (w * h) as usize];
    for y in 10..20 {
        for x in 15..25 {
            m[(y * w + x) as usize] = 255;
        }
    }
    let mask = Gray8::new(w, h, m).unwrap();
    let boxed = highlight_covered(&plain, &mask, "red-box");
    let at = |img: &Rgba8, x: u32, y: u32| img.get(x, y).unwrap();
    assert_eq!(at(&boxed, 20, 15), [100, 100, 100, 100], "inside the hole as it was");
    assert_eq!(at(&boxed, 15 - 2, 15), [255, 0, 0, 255], "the box one pixel out, 3 wide");
    assert_eq!(at(&boxed, 15 - 4, 15), [255, 0, 0, 255]);
    assert_eq!(at(&boxed, 15 - 5, 15), [100, 100, 100, 100]);
    assert_eq!(at(&boxed, 15 - 1, 15), [100, 100, 100, 100]);
    // A result that kept the box is the picture there again.
    let out = unboxed(boxed.clone(), &boxed, &plain);
    assert_eq!(at(&out, 13, 15), [100, 100, 100, 100]);
    // The solid highlight still paints the hole.
    assert_eq!(at(&highlight_covered(&plain, &mask, "red"), 20, 15), [255, 0, 0, 255]);
}
