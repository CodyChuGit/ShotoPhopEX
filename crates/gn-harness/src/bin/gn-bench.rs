//! `gn-bench`: run a suite on an engine, or compare two runs.
//!
//! ```text
//! gn-bench run --engine comfy --suite t2i --fixtures <dir> --images <dir> --cli <photocraft-cli> --out <dir> [--comfy-path <install>] [--template <id>] [--only <substring>]
//! gn-bench run --engine sdcpp --suite t2i --fixtures <dir> --images <dir> --sd-cli <exe> --diffusion-model <file> --llm <file> --vae <file> --out <dir> [--steps N] [--cfg X] [--sd-args "…"] [--only <substring>]
//! gn-bench run --engine sam-ort --suite sam --fixtures <dir> --images <dir> --gn-sam <exe> --ort <dir> --models <dir> --out <dir> [--libs "<dir>;<dir>"] [--cpu] [--threshold 0.5] [--only <substring>]
//! gn-bench compare <run-a> <run-b> [--out <report.md>]
//! gn-bench summary <run>
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use gn_harness::engines::{ComfyCli, Engine, RunCtx, SamOrt, SdCli, run_suite};
use gn_harness::{Suite, report};

fn flags(args: &[String]) -> (Vec<String>, HashMap<String, String>) {
    let mut positional = Vec::new();
    let mut named = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(name) = a.strip_prefix("--") {
            let value = args.get(i + 1).cloned().unwrap_or_default();
            named.insert(name.to_string(), value);
            i += 2;
        } else {
            positional.push(a.clone());
            i += 1;
        }
    }
    (positional, named)
}

fn need<'a>(named: &'a HashMap<String, String>, key: &str) -> Result<&'a String, String> {
    named.get(key).filter(|v| !v.is_empty()).ok_or_else(|| format!("--{key} is required"))
}

fn main_inner() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (pos, named) = flags(&args);
    match pos.first().map(String::as_str) {
        Some("run") => {
            let suite_name = need(&named, "suite")?;
            let fixtures = PathBuf::from(need(&named, "fixtures")?);
            let suite = Suite::load(&fixtures.join(format!("{suite_name}.json")))?;
            let ctx = RunCtx { images: PathBuf::from(need(&named, "images")?), out: PathBuf::from(need(&named, "out")?), template: named.get("template").cloned().filter(|t| !t.is_empty()) };
            let engine: Box<dyn Engine> = match need(&named, "engine")?.as_str() {
                "comfy" => Box::new(ComfyCli { cli: PathBuf::from(need(&named, "cli")?), comfy_path: named.get("comfy-path").filter(|p| !p.is_empty()).map(PathBuf::from) }),
                "sdcpp" => {
                    let mut e = SdCli::krea2_turbo(PathBuf::from(need(&named, "sd-cli")?), PathBuf::from(need(&named, "diffusion-model")?), PathBuf::from(need(&named, "llm")?), PathBuf::from(need(&named, "vae")?));
                    if let Some(s) = named.get("steps").and_then(|v| v.parse().ok()) {
                        e.steps = s;
                    }
                    if let Some(c) = named.get("cfg").and_then(|v| v.parse().ok()) {
                        e.cfg = c;
                    }
                    if let Some(x) = named.get("sd-args") {
                        e.extra = x.split_whitespace().map(str::to_string).collect();
                    }
                    Box::new(e)
                }
                "sam-ort" => Box::new(SamOrt {
                    exe: PathBuf::from(need(&named, "gn-sam")?),
                    ort: PathBuf::from(need(&named, "ort")?),
                    libs: named.get("libs").map(|s| s.split(';').filter(|p| !p.is_empty()).map(PathBuf::from).collect()).unwrap_or_default(),
                    models: PathBuf::from(need(&named, "models")?),
                    cpu: named.contains_key("cpu"),
                    threshold: named.get("threshold").and_then(|t| t.parse().ok()).unwrap_or(0.5),
                }),
                other => return Err(format!("unknown engine `{other}` (comfy, sdcpp, sam-ort; native comes with the engine)")),
            };
            let only = named.get("only").map(String::as_str).filter(|s| !s.is_empty());
            println!("{} cases of `{}` on {} → {}", suite.cases.len(), suite.suite, engine.name(), ctx.out.display());
            let records = run_suite(engine.as_ref(), &suite.cases, &ctx, only, |r| match &r.error {
                Some(e) => println!("  {:<28} FAILED after {:.1} s: {e}", r.id, r.wall_ms as f64 / 1000.0),
                None => println!("  {:<28} {:.1} s{}", r.id, r.wall_ms as f64 / 1000.0, r.engine_ms.map_or(String::new(), |m| format!(" (engine {:.1} s)", m as f64 / 1000.0))),
            })?;
            let md = report::summary(&records);
            std::fs::write(ctx.out.join("summary.md"), &md).map_err(|e| format!("cannot write summary.md: {e}"))?;
            print!("\n{md}");
            Ok(())
        }
        Some("compare") => {
            let (a, b) = (pos.get(1).ok_or("compare needs two run folders")?, pos.get(2).ok_or("compare needs two run folders")?);
            let md = report::compare(&PathBuf::from(a), &PathBuf::from(b))?;
            if let Some(out) = named.get("out").filter(|o| !o.is_empty()) {
                std::fs::write(out, &md).map_err(|e| format!("cannot write {out}: {e}"))?;
            }
            print!("{md}");
            Ok(())
        }
        Some("summary") => {
            let dir = PathBuf::from(pos.get(1).ok_or("summary needs a run folder")?);
            print!("{}", report::summary(&report::load_run(&dir)?));
            Ok(())
        }
        _ => Err("usage: gn-bench run --engine comfy --suite t2i|edit|fill|sam --fixtures <dir> --images <dir> --cli <exe> --out <dir> [--comfy-path <install>] [--template <id>] [--only <text>]\n       gn-bench compare <run-a> <run-b> [--out <report.md>]\n       gn-bench summary <run>".into()),
    }
}

fn main() {
    if let Err(e) = main_inner() {
        eprintln!("gn-bench: {e}");
        std::process::exit(1);
    }
}
