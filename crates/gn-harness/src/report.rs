//! Reports: a run folder's summary, and the comparison of two run folders (the plan's parity
//! table), as Markdown.

use std::path::Path;

use serde_json::Value;

use crate::engines::Record;
use crate::image::Image;
use crate::metrics::{self, Verdict};

/// Every `<id>.json` in a run folder, by id.
pub fn load_run(dir: &Path) -> crate::Result<Vec<Record>> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "json") {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            if let Ok(r) = serde_json::from_str::<Record>(&text) {
                out.push(r);
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn ms(v: Option<u64>) -> String {
    v.map_or("–".into(), |m| format!("{:.1} s", m as f64 / 1000.0))
}

/// One run's table: case, wall time, engine time, error.
pub fn summary(records: &[Record]) -> String {
    let mut s = String::from("| Case | Engine | Wall | Engine time | Seed | Note |\n|---|---|---:|---:|---:|---|\n");
    for r in records {
        let note = r.error.clone().unwrap_or_else(|| r.result.get("count").and_then(Value::as_u64).map_or(String::new(), |c| format!("{c} instance(s)")));
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            r.id,
            r.engine,
            ms(Some(r.wall_ms)),
            ms(r.engine_ms),
            r.seed.map_or("–".into(), |x| x.to_string()),
            note
        ));
    }
    let ok = records.iter().filter(|r| r.error.is_none()).count();
    let wall: u64 = records.iter().map(|r| r.wall_ms).sum();
    s.push_str(&format!("\n{ok} of {} cases ran; {} wall in total.\n", records.len(), ms(Some(wall))));
    s
}

/// The first instance's `[x, y, w, h]` from a SAM result.
fn first_bounds(v: &Value) -> Option<[f32; 4]> {
    let b = v.get("instances")?.as_array()?.first()?.get("bounds")?.as_array()?;
    let f = |i: usize| b.get(i).and_then(Value::as_f64).map(|x| x as f32);
    Some([f(0)?, f(1)?, f(2)?, f(3)?])
}

/// Two runs side by side: time and parity per case.
pub fn compare(a_dir: &Path, b_dir: &Path) -> crate::Result<String> {
    let (a, b) = (load_run(a_dir)?, load_run(b_dir)?);
    let (na, nb) = (a.first().map_or("A".to_string(), |r| r.engine.clone()), b.first().map_or("B".to_string(), |r| r.engine.clone()));
    let mut s = format!("| Case | {na} | {nb} | Speed | PSNR | SSIM | Bounds IoU | Parity |\n|---|---:|---:|---:|---:|---:|---:|:-:|\n");
    let (mut pass, mut total) = (0, 0);
    for ra in &a {
        let Some(rb) = b.iter().find(|r| r.id == ra.id) else { continue };
        total += 1;
        let speed = if rb.wall_ms > 0 { format!("{:.2}×", ra.wall_ms as f64 / rb.wall_ms as f64) } else { "–".into() };
        let (mut psnr, mut ssim, mut biou) = (None, None, None);
        if let (Some(ia), Some(ib)) = (&ra.image, &rb.image)
            && let (Ok(pa), Ok(pb)) = (Image::read(&a_dir.join(ia)), Image::read(&b_dir.join(ib)))
            && pa.same_size(&pb)
        {
            let (la, lb) = (pa.luma(), pb.luma());
            psnr = metrics::psnr(&la, &lb);
            ssim = metrics::ssim(&la, &lb, pa.width as usize, pa.height as usize);
        }
        if let (Some(ba), Some(bb)) = (first_bounds(&ra.result), first_bounds(&rb.result)) {
            biou = Some(metrics::bounds_iou(ba, bb));
        }
        let verdict = if ra.image.is_some() || rb.image.is_some() {
            Verdict::picture(psnr, ssim)
        } else {
            match biou {
                Some(i) if i >= 0.9 => Verdict::Pass,
                Some(_) => Verdict::Fail,
                None => Verdict::None,
            }
        };
        if verdict == Verdict::Pass {
            pass += 1;
        }
        let f = |v: Option<f32>, unit: &str| v.map_or("–".to_string(), |x| if x.is_infinite() { "∞".into() } else { format!("{x:.2}{unit}") });
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            ra.id,
            ms(Some(ra.wall_ms)),
            ms(Some(rb.wall_ms)),
            speed,
            f(psnr, " dB"),
            f(ssim, ""),
            f(biou, ""),
            verdict.glyph()
        ));
    }
    s.push_str(&format!(
        "\n{pass} of {total} cases at parity (pictures: PSNR ≥ 30 dB or SSIM ≥ 0.90; SAM: bounds IoU ≥ 0.90). Speed = {na} wall ÷ {nb} wall.\n"
    ));
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rec(id: &str, engine: &str, wall: u64, image: Option<&str>, result: Value) -> Record {
        Record { id: id.into(), engine: engine.into(), image: image.map(str::to_string), wall_ms: wall, engine_ms: None, seed: Some(1), result, error: None }
    }

    #[test]
    fn two_runs_compare_by_picture_and_by_bounds() {
        let d = std::env::temp_dir().join(format!("gn-report-{}", std::process::id()));
        let (a, b) = (d.join("a"), d.join("b"));
        let mut im = Image::new(16, 16);
        for (i, px) in im.data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *px = [(i * 3) as u8, 100, 50, 255];
        }
        im.write(&a.join("pic.png")).unwrap_or_else(|e| panic!("{e}"));
        im.write(&b.join("pic.png")).unwrap_or_else(|e| panic!("{e}"));
        for (dir, eng, wall) in [(&a, "comfy", 2000), (&b, "native", 1000)] {
            for r in [rec("pic", eng, wall, Some("pic.png"), json!({})), rec("sam", eng, wall, None, json!({"instances": [{"bounds": [0, 0, 10, 10]}]}))] {
                std::fs::write(dir.join(format!("{}.json", r.id)), serde_json::to_string(&r).unwrap_or_default()).unwrap_or_else(|e| panic!("{e}"));
            }
        }
        let md = compare(&a, &b).unwrap_or_else(|e| panic!("{e}"));
        assert!(md.contains("| pic | 2.0 s | 1.0 s | 2.00× | ∞ |"), "{md}");
        assert!(md.contains("| sam |") && md.contains("| 1.00 | ✓ |"), "{md}");
        assert!(md.contains("2 of 2 cases at parity"), "{md}");
        let sum = summary(&load_run(&a).unwrap_or_default());
        assert!(sum.contains("2 of 2 cases ran"), "{sum}");
        let _ = std::fs::remove_dir_all(d);
    }
}
