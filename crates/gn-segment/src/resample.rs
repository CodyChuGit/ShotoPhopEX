//! Bilinear resampling with PyTorch's `align_corners=False` convention and no antialiasing:
//! exactly what `F.interpolate(mode="bilinear")` and ComfyUI's `common_upscale` compute, so a
//! picture squashed to the model's 1008×1008 window, and a mask stretched back, match the
//! reference pixel for pixel.

/// Resample one plane from `sw`×`sh` to `dw`×`dh`.
pub fn bilinear(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || src.len() < sw * sh {
        return vec![0.0; dw * dh];
    }
    let taps = |len_src: usize, len_dst: usize| -> Vec<(usize, usize, f32)> {
        let scale = len_src as f32 / len_dst as f32;
        (0..len_dst)
            .map(|d| {
                let s = ((d as f32 + 0.5) * scale - 0.5).max(0.0);
                let i0 = (s.floor() as usize).min(len_src - 1);
                let i1 = (i0 + 1).min(len_src - 1);
                (i0, i1, s - i0 as f32)
            })
            .collect()
    };
    let xs = taps(sw, dw);
    let ys = taps(sh, dh);
    let mut out = Vec::with_capacity(dw * dh);
    for &(y0, y1, fy) in &ys {
        let r0 = &src[y0 * sw..y0 * sw + sw];
        let r1 = &src[y1 * sw..y1 * sw + sw];
        for &(x0, x1, fx) in &xs {
            let top = r0[x0] + (r0[x1] - r0[x0]) * fx;
            let bot = r1[x0] + (r1[x1] - r1[x0]) * fx;
            out.push(top + (bot - top) * fy);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_sizes_match() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(bilinear(&src, 2, 2, 2, 2), src);
    }

    #[test]
    fn upsample_2x_matches_torch_align_corners_false() {
        // torch: F.interpolate(tensor([[1,2],[3,4]]), scale_factor=2, mode="bilinear")
        // = [[1.00,1.25,1.75,2.00],[1.50,1.75,2.25,2.50],[2.50,2.75,3.25,3.50],[3.00,3.25,3.75,4.00]]
        let out = bilinear(&[1.0, 2.0, 3.0, 4.0], 2, 2, 4, 4);
        let want = [1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0];
        for (a, b) in out.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-6, "{out:?}");
        }
    }

    #[test]
    fn downsample_averages_neighbours() {
        // 4 → 2 with align_corners=False samples at source 0.5 and 2.5: the means of each pair.
        let out = bilinear(&[0.0, 2.0, 4.0, 6.0], 4, 1, 2, 1);
        assert_eq!(out, vec![1.0, 5.0]);
    }

    #[test]
    fn degenerate_sizes_do_not_panic() {
        assert_eq!(bilinear(&[], 0, 0, 2, 2).len(), 4);
        assert_eq!(bilinear(&[1.0], 1, 1, 0, 3).len(), 0);
        assert_eq!(bilinear(&[1.0], 2, 2, 1, 1), vec![0.0]);
    }
}
