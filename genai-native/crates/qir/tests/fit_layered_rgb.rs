//! Fit the Layered RGBA VAE's latent-to-RGB map (16 → 3 plus a bias) by least squares, for
//! Fast previews: split real pictures, average each decoded layer over the 8×8 cells of the
//! latent grid, keep the opaque cells, and solve against the normalised latent. Prints the
//! table for `qir::preview`. Ignored: needs CUDA and ComfyUI's Layered files.
//!
//! `QIR_MODELS=<ComfyUI models> QIR_FIT_PICTURES=a.png;b.png cargo test --release --features
//! cuda,… --test fit_layered_rgb -- --ignored --nocapture`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::needless_range_loop)]

use std::sync::atomic::AtomicBool;

use candle_core::{DType, Device, Tensor};
use qwen_image_rs::layered_pipeline::{LayeredPipeline, RgbaImage};

#[test]
#[ignore = "needs CUDA and the Layered weights"]
fn fit_the_layered_latent_to_rgb_map() {
    let models = std::env::var("QIR_MODELS").expect("QIR_MODELS");
    let pictures = std::env::var("QIR_FIT_PICTURES").expect("QIR_FIT_PICTURES");
    let p = LayeredPipeline::load(std::path::Path::new(&models)).unwrap();
    // Normal equations: A^T A (17×17) and A^T y (17×3), rows = opaque latent cells.
    let mut ata = [[0f64; 17]; 17];
    let mut aty = [[0f64; 3]; 17];
    let mut rows = 0usize;
    for path in pictures.split(';') {
        let img = image::open(path).unwrap().to_rgba8();
        let img = RgbaImage::from_raw(img.width(), img.height(), img.into_raw()).unwrap();
        let mut retained: Option<Tensor> = None;
        let mut decoded: Vec<Tensor> = Vec::new();
        p.split_observed(&img, 3, "", 7, 20, |_, _| {}, &AtomicBool::new(false), |name, t| {
            if name == "retained" {
                retained = Some(t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?);
            } else if name.starts_with("decoded_") {
                decoded.push(t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?);
            }
            Ok(())
        })
        .unwrap();
        let retained = retained.unwrap(); // (layers, 16, h, w)
        let (n, c, h, w) = retained.dims4().unwrap();
        assert_eq!(c, 16);
        for (l, dec) in decoded.iter().enumerate().take(n) {
            let z = retained.get(l).unwrap().to_vec3::<f32>().unwrap();
            // (1, 4, 8h, 8w) in [-1, 1] → mean over each 8×8 cell.
            let pooled = dec.squeeze(0).unwrap().reshape((4, h, 8, w, 8)).unwrap().mean(4).unwrap().mean(2).unwrap().to_vec3::<f32>().unwrap();
            for y in 0..h {
                for x in 0..w {
                    let alpha = (pooled[3][y][x] + 1.0) / 2.0;
                    if alpha < 0.95 {
                        continue;
                    }
                    let mut a = [1f64; 17];
                    for k in 0..16 {
                        a[k] = f64::from(z[k][y][x]);
                    }
                    for i in 0..17 {
                        for j in 0..17 {
                            ata[i][j] += a[i] * a[j];
                        }
                        for k in 0..3 {
                            aty[i][k] += a[i] * f64::from(pooled[k][y][x]);
                        }
                    }
                    rows += 1;
                }
            }
        }
    }
    assert!(rows > 1000, "{rows} opaque cells");
    // Gauss-Jordan with a whisper of ridge.
    for (i, row) in ata.iter_mut().enumerate() {
        row[i] += 1e-6 * rows as f64;
    }
    let mut m = ata;
    let mut b = aty;
    for col in 0..17 {
        let piv = (col..17).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs())).unwrap();
        m.swap(col, piv);
        b.swap(col, piv);
        let d = m[col][col];
        for j in 0..17 {
            m[col][j] /= d;
        }
        for k in 0..3 {
            b[col][k] /= d;
        }
        for i in 0..17 {
            if i != col {
                let f = m[i][col];
                for j in 0..17 {
                    m[i][j] -= f * m[col][j];
                }
                for k in 0..3 {
                    b[i][k] -= f * b[col][k];
                }
            }
        }
    }
    println!("// {rows} opaque cells");
    for row in b.iter().take(16) {
        println!("    [{:.4}, {:.4}, {:.4}],", row[0], row[1], row[2]);
    }
    println!("bias [{:.4}, {:.4}, {:.4}]", b[16][0], b[16][1], b[16][2]);
}
