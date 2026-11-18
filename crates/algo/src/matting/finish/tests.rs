use super::*;

/// The picture, its alpha, and the true foreground and background.
type Ramp = (Vec<[f32; 3]>, Vec<f32>, [f32; 3], Vec<[f32; 3]>);

/// A red subject over a green-to-blue backdrop, its edge a horizontal alpha ramp.
fn ramp(w: usize, h: usize) -> Ramp {
    let fg = [0.8, 0.15, 0.1];
    let bg: Vec<[f32; 3]> = (0..w * h).map(|k| [0.05, 0.3 + 0.4 * (k % w) as f32 / w as f32, 0.6 - 0.3 * (k / w) as f32 / h as f32]).collect();
    let alpha: Vec<f32> = (0..w * h).map(|k| (1.0 - ((k % w) as f32 - 20.0) / 24.0).clamp(0.0, 1.0)).collect();
    let img = (0..w * h).map(|k| std::array::from_fn(|c| alpha[k] * fg[c] + (1.0 - alpha[k]) * bg[k][c])).collect();
    (img, alpha, fg, bg)
}

// Measured once (release, 4 threads, 2026-11-17): 1920 × 1080 in 0.22–0.30 s.
#[test]
fn foreground_estimation_recovers_the_subject_colour_at_the_edge() {
    let (w, h) = (64, 48);
    let (img, alpha, fg, bg) = ramp(w, h);
    let (f, b) = estimate_foreground(&img, &alpha, w, h, &ForegroundParams::default());
    assert_eq!((f.len(), b.len()), (w * h, w * h));
    for k in (0..w * h).filter(|k| alpha[*k] > 0.1 && alpha[*k] < 0.9) {
        for c in 0..3 {
            assert!((f[k][c] - fg[c]).abs() < 0.06, "F at {k}: {:?} vs {fg:?}", f[k]);
            // B is barely seen where the subject covers most of the pixel.
            assert!(alpha[k] > 0.6 || (b[k][c] - bg[k][c]).abs() < 0.08, "B at {k}: {:?} vs {:?}", b[k], bg[k]);
        }
        // The estimate still composes back to the picture.
        let mixed: [f32; 3] = std::array::from_fn(|c| alpha[k] * f[k][c] + (1.0 - alpha[k]) * b[k][c]);
        assert!((0..3).all(|c| (mixed[c] - img[k][c]).abs() < 0.02), "{mixed:?} vs {:?}", img[k]);
    }
    // Solid pixels keep their own colour.
    assert!((0..3).all(|c| (f[5 * w + 3][c] - img[5 * w + 3][c]).abs() < 1e-3));
}

#[test]
fn foreground_estimation_handles_tiny_and_empty_pictures() {
    let p = ForegroundParams::default();
    let (f, b) = estimate_foreground(&[[0.2, 0.4, 0.6]], &[1.0], 1, 1, &p);
    assert!((0..3).all(|c| (f[0][c] - [0.2, 0.4, 0.6][c]).abs() < 1e-3), "{f:?}");
    assert_eq!(b.len(), 1);
    assert_eq!(estimate_foreground(&[], &[], 0, 0, &p).0.len(), 0);
    let (f, _) = estimate_foreground(&[[0.5; 3]; 3], &[0.5; 3], 3, 1, &p);
    assert!(f.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn unmixing_changes_only_soft_pixels_and_takes_out_the_backdrop() {
    let (w, h) = (64, 48);
    let (img, alpha, _, _) = ramp(w, h);
    let mut px: Vec<u8> = img
        .iter()
        .zip(&alpha)
        .flat_map(|(q, a)| [q[0], q[1], q[2]].map(|v| (linear_to_srgb(v) * 255.0).round() as u8).into_iter().chain([(a * 255.0).round() as u8]))
        .collect();
    let before = px.clone();
    unmix_rgba8(&mut px, w, h, &alpha);
    for k in 0..w * h {
        assert_eq!(px[k * 4 + 3], before[k * 4 + 3], "alpha byte kept");
        if alpha[k] <= 0.0 || alpha[k] >= 1.0 {
            assert_eq!(px[k * 4..k * 4 + 3], before[k * 4..k * 4 + 3], "solid / clear pixel {k} kept");
        }
    }
    // A pixel a quarter covered was mostly backdrop; now it is the subject's red.
    let k = 10 * w + 38;
    assert!(alpha[k] > 0.2 && alpha[k] < 0.3);
    assert!(before[k * 4 + 1] > 120 && px[k * 4 + 1] < 120, "green tint gone: {:?} → {:?}", &before[k * 4..k * 4 + 4], &px[k * 4..k * 4 + 4]);
    assert!(px[k * 4] > 210, "red: {:?}", &px[k * 4..k * 4 + 4]);
    // Nothing soft: nothing to do.
    let mut hard = before.clone();
    unmix_rgba8(&mut hard, w, h, &alpha.iter().map(|a| a.round()).collect::<Vec<_>>());
    assert_eq!(hard, before);
}

/// Sets a `side`-square of `alpha` at (x, y) to `v`.
fn square(alpha: &mut [f32], w: usize, (x, y): (usize, usize), side: usize, v: f32) {
    for yy in y..y + side {
        alpha[yy * w + x..yy * w + x + side].fill(v);
    }
}

#[test]
fn despeckle_drops_specks_and_islands_but_keeps_the_subject_and_its_parts() {
    let (w, h) = (200, 200);
    let mut alpha = vec![0.0f32; w * h];
    // The subject: a disc with a soft rim.
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f32 - 70.0).powi(2) + (y as f32 - 70.0).powi(2)).sqrt();
            alpha[y * w + x] = (1.0 - (d - 40.0) / 6.0).clamp(0.0, 1.0);
        }
    }
    square(&mut alpha, w, (180, 30), 3, 1.0); // a speck far away
    square(&mut alpha, w, (185, 185), 5, 0.3); // a faint smudge, never above 0.5
    square(&mut alpha, w, (120, 150), 20, 1.0); // a held cup: separate but not small
    let before = alpha.clone();
    let removed = despeckle(&mut alpha, w, h, &DespeckleParams::default());
    assert_eq!(removed, 1);
    assert!(alpha[31 * w + 181] == 0.0 && alpha[187 * w + 187] == 0.0, "speck and smudge cleared");
    for y in 0..h {
        for x in 0..w {
            let k = y * w + x;
            let d = ((x as f32 - 70.0).powi(2) + (y as f32 - 70.0).powi(2)).sqrt();
            if d < 50.0 || (120..140).contains(&x) && (150..170).contains(&y) {
                assert_eq!(alpha[k], before[k], "subject and cup untouched at ({x}, {y})");
            }
        }
    }
}

#[test]
fn despeckle_scales_with_the_picture_and_spares_parts_of_a_small_subject() {
    let p = DespeckleParams::default();
    // A 5 × 5 speck beside a large subject: small in a 400² picture, not in a 100² one.
    for (side, gone) in [(400, true), (100, false)] {
        let mut alpha = vec![0.0f32; side * side];
        square(&mut alpha, side, (0, 0), 60, 1.0);
        square(&mut alpha, side, (side - 10, side - 10), 5, 1.0);
        despeckle(&mut alpha, side, side, &p);
        assert_eq!(alpha[(side - 8) * side + side - 8] == 0.0, gone, "{side}");
    }
    // Under the picture's limit but a large share of a small subject: kept.
    let mut alpha = vec![0.0f32; 400 * 400];
    square(&mut alpha, 400, (10, 10), 10, 1.0);
    square(&mut alpha, 400, (300, 300), 6, 1.0);
    assert_eq!(despeckle(&mut alpha, 400, 400, &p), 0);
    assert_eq!(alpha[302 * 400 + 302], 1.0);
    // A speck the detector itself found stays.
    let mut alpha = vec![0.0f32; 400 * 400];
    square(&mut alpha, 400, (0, 0), 60, 1.0);
    square(&mut alpha, 400, (390, 390), 5, 1.0);
    let keep: Vec<bool> = (0..400 * 400).map(|k| k == 392 * 400 + 392).collect();
    assert_eq!(despeckle_keeping(&mut alpha, 400, 400, &p, &keep), 0);
    assert_eq!(alpha[391 * 400 + 391], 1.0);
    // Nothing at or above 0.5: left alone.
    let mut faint = vec![0.2f32; 50 * 50];
    assert_eq!(despeckle(&mut faint, 50, 50, &p), 0);
    assert!(faint.iter().all(|a| *a == 0.2));
}

#[test]
fn despill_pulls_the_screen_channel_down_and_keeps_brightness() {
    let q = [0.4, 0.8, 0.3];
    let d = despill_pixel(q, Screen::Green, SpillLimit::Average, 1.0);
    assert!((d[1] - 0.35).abs() < 1e-6 && (d[0] - 0.625).abs() < 1e-6 && (d[2] - 0.525).abs() < 1e-6, "{d:?}");
    assert!((d.iter().sum::<f32>() - q.iter().sum::<f32>()).abs() < 1e-6);
    let m = despill_pixel(q, Screen::Green, SpillLimit::Max, 1.0);
    assert!((m[1] - 0.4).abs() < 1e-6 && (m.iter().sum::<f32>() - 1.5).abs() < 1e-6, "{m:?}");
    let half = despill_pixel(q, Screen::Green, SpillLimit::Average, 0.5);
    assert!((half[1] - 0.575).abs() < 1e-6, "{half:?}");
    assert_eq!(despill_pixel(q, Screen::Green, SpillLimit::Average, 0.0), q);
    // No spill, no change; blue works on blue.
    assert_eq!(despill_pixel([0.6, 0.3, 0.5], Screen::Green, SpillLimit::Average, 1.0), [0.6, 0.3, 0.5]);
    let b = despill_pixel([0.2, 0.4, 0.9], Screen::Blue, SpillLimit::Average, 1.0);
    assert!((b[2] - 0.3).abs() < 1e-6 && (b[0] - 0.5).abs() < 1e-6 && (b[1] - 0.7).abs() < 1e-6, "{b:?}");
    let mut px = vec![[0.4, 0.8, 0.3, 0.5]];
    despill(&mut px, Screen::Green, SpillLimit::Average, 1.0);
    assert_eq!(px[0][3], 0.5);
}

#[test]
fn screen_detection_reads_the_backdrop() {
    let n = 1000;
    let pick = |bg: [f32; 3], back: usize| -> (Vec<[f32; 3]>, Vec<f32>) {
        let rgb = (0..n).map(|k| if k < back { bg } else { [0.7, 0.5, 0.4] }).collect();
        let alpha = (0..n).map(|k| if k < back { 0.0 } else { 1.0 }).collect();
        (rgb, alpha)
    };
    let (rgb, a) = pick([0.1, 0.2, 0.8], 300);
    assert_eq!(detect_screen(&rgb, &a), Screen::Blue);
    let (rgb, a) = pick([0.1, 0.8, 0.2], 300);
    assert_eq!(detect_screen(&rgb, &a), Screen::Green);
    // Too little backdrop, or no clear lead: green.
    let (rgb, a) = pick([0.1, 0.2, 0.8], 5);
    assert_eq!(detect_screen(&rgb, &a), Screen::Green);
    let (rgb, a) = pick([0.1, 0.42, 0.45], 300);
    assert_eq!(detect_screen(&rgb, &a), Screen::Green);
}
