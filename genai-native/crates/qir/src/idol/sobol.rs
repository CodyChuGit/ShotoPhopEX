//! torch's `SobolEngine(dimension=3, scramble=True, seed=123).draw(n)`, bit for bit: the
//! flow model's latent tokens sit at these fixed 3D anchors (its positional encoding), which
//! the checkpoint does not store. Ports `torch/quasirandom.py` and `SobolEngineOps.cpp`,
//! with torch's CPU generator (mt19937, `randint(2)` = `next_u32 % 2`).

use crate::comfy::Mt19937;

const MAXBIT: usize = 30;
/// Joe-Kuo direction numbers for the first three dimensions (`poly`, `initsobolstate`).
const POLY: [u64; 3] = [1, 3, 7];
const INIT: [&[u64]; 3] = [&[], &[1], &[1, 3]];

/// The first `n` points of torch's scrambled 3D Sobol sequence with `seed`, row-major `(n, 3)`.
pub fn scrambled_3d(n: usize, seed: u32) -> Vec<f32> {
    let dim = 3;
    // _sobol_engine_initialize_state_
    let mut ss = vec![[0u64; MAXBIT]; dim];
    ss[0] = [1; MAXBIT];
    for d in 1..dim {
        let p = POLY[d];
        let m = (64 - p.leading_zeros()) as usize - 1;
        ss[d][..m].copy_from_slice(INIT[d]);
        for j in m..MAXBIT {
            let mut newv = ss[d][j - m];
            let mut pow2 = 1u64;
            for k in 0..m {
                pow2 <<= 1;
                if (p >> (m - 1 - k)) & 1 == 1 {
                    newv ^= pow2 * ss[d][j - k - 1];
                }
            }
            ss[d][j] = newv;
        }
    }
    for row in ss.iter_mut() {
        for (j, v) in row.iter_mut().enumerate() {
            *v *= 1u64 << (MAXBIT - 1 - j);
        }
    }
    // _scramble: the shift, then the lower-triangular matrices, from one generator.
    let mut g = Mt19937::new(u64::from(seed));
    let mut shift = vec![0u64; dim];
    for s in shift.iter_mut() {
        for j in 0..MAXBIT {
            *s += u64::from(g.next_u32() % 2) << j;
        }
    }
    let mut ltm = vec![[[0u64; MAXBIT]; MAXBIT]; dim];
    for m in ltm.iter_mut() {
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                let bit = u64::from(g.next_u32() % 2);
                *v = if c < r { bit } else if c == r { 1 } else { 0 };
            }
        }
    }
    // _sobol_engine_scramble_
    for d in 0..dim {
        let dots: Vec<u64> = (0..MAXBIT).map(|p| (0..MAXBIT).map(|k| ltm[d][p][k] << (MAXBIT - 1 - k)).sum()).collect();
        for j in 0..MAXBIT {
            let vdj = ss[d][j];
            let (mut l, mut t2) = (1u64, 0u64);
            for p in (0..MAXBIT).rev() {
                let t1 = (dots[p] & vdj).count_ones() as u64 % 2;
                t2 += t1 * l;
                l <<= 1;
            }
            ss[d][j] = t2;
        }
    }
    // draw: the shift itself, then the Gray-code walk.
    let recip = 1.0f64 / (1u64 << MAXBIT) as f64;
    let mut quasi = shift;
    let mut out = Vec::with_capacity(n * dim);
    if n > 0 {
        out.extend(quasi.iter().map(|&q| (q as f64 * recip) as f32));
    }
    for i in 0..n.saturating_sub(1) as u64 {
        let l = (!i).trailing_zeros() as usize;
        for d in 0..dim {
            quasi[d] ^= ss[d][l];
        }
        out.extend(quasi.iter().map(|&q| (q as f64 * recip) as f32));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_the_reference() {
        // std::mt19937 with the default seed 5489: the 10000th output is 4123659995.
        let mut g = Mt19937::new(5489);
        let mut v = 0;
        for _ in 0..10000 {
            v = g.next_u32();
        }
        assert_eq!(v, 4_123_659_995);
    }

    #[test]
    fn points_are_in_the_unit_cube_and_distinct() {
        let p = scrambled_3d(8192, 123);
        assert_eq!(p.len(), 8192 * 3);
        assert!(p.iter().all(|&v| (0.0..1.0).contains(&v)));
        let mut rows: Vec<[u32; 3]> = p.chunks(3).map(|r| [r[0].to_bits(), r[1].to_bits(), r[2].to_bits()]).collect();
        rows.sort();
        rows.dedup();
        assert_eq!(rows.len(), 8192);
    }
}
