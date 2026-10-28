//! Replay the GPU parity manifest through the actual worker request routing.
use gn_diffusion::DiffusionEngine;
use gn_engine::{protocol::RunRequest, server::Engine};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};
fn main() -> anyhow::Result<()> {
    let manifest = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("manifest path required"))?;
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let root = value["models"].as_str().ok_or_else(|| anyhow::anyhow!("models required"))?;
    let out = PathBuf::from(value["out"].as_str().ok_or_else(|| anyhow::anyhow!("out required"))?);
    std::fs::create_dir_all(&out)?;
    let mut engine = DiffusionEngine::new(root.into(), false);
    let cases = value["cases"].as_array().ok_or_else(|| anyhow::anyhow!("cases required"))?;
    for case in cases {
        let mut req: RunRequest = serde_json::from_value(case["request"].clone())?;
        let name = req.model.split('/').next_back().ok_or_else(|| anyhow::anyhow!("template"))?.to_string();
        if std::env::var("CASE").is_ok_and(|c| c != name) {
            continue;
        }
        let paths = case["blobs"].as_array().ok_or_else(|| anyhow::anyhow!("blobs required"))?;
        let blobs = paths
            .iter()
            .map(|p| std::fs::read(p.as_str().ok_or_else(|| anyhow::anyhow!("blob path"))?).map_err(Into::into))
            .collect::<anyhow::Result<Vec<_>>>()?;
        for warm in 0..2 {
            req.seed = 42 + warm;
            let begin = Instant::now();
            let result = engine.run(&req, &blobs, &mut |f, m| eprintln!("{name} {warm} {f:.3} {m}"), &AtomicBool::new(false));
            let (done, bytes) = result.map_err(|e| anyhow::anyhow!("{:?}: {}", e.kind, e.message))?;
            let elapsed = begin.elapsed().as_secs_f64();
            let first = bytes.first().ok_or_else(|| anyhow::anyhow!("no image"))?;
            std::fs::write(out.join(format!("{name}-{warm}.rgba")), first)?;
            std::fs::write(out.join(format!("{name}-{warm}.json")), serde_json::to_vec_pretty(&serde_json::json!({"seconds":elapsed,"done":done}))?)?;
            eprintln!("BENCH {name} warm={warm} seconds={elapsed:.6}");
        }
    }
    Ok(())
}
