//! Engines the harness can drive. Each runs one [`Case`] and leaves `<id>.png` (pictures) and
//! `<id>.json` (timings, the engine's own result) in the run folder.
//!
//! - [`ComfyCli`]: the ComfyUI route as PhotoshopEX runs it, through `photocraft-cli run` (the
//!   same templates, seeds and compositing the app uses), so the reference is "what the product
//!   does today". The engine's `ms` and the wall time are both recorded.
//! - sd.cpp (`sd-cli`) and the native worker come with plan Phases 0 §5.3 and 2.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::fixtures::{Case, Kind, Prompt, point_px, rect_px};
use crate::image::Image;

/// Where a run reads and writes.
pub struct RunCtx {
    /// The bench pictures (`krea2-fox.png`, …).
    pub images: PathBuf,
    /// The run folder.
    pub out: PathBuf,
    /// A template id to force (`krea2-turbo/image`, `qwen-2.1/image`, …); the engine's default
    /// otherwise.
    pub template: Option<String>,
}

/// What a run recorded, as `<id>.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub engine: String,
    /// The picture the case produced, relative to the run folder (`None` for a SAM case).
    pub image: Option<String>,
    pub wall_ms: u64,
    /// The engine's own time for the run, when it reports one.
    pub engine_ms: Option<u64>,
    pub seed: Option<u64>,
    /// The engine's result (instances for SAM, the command's JSON otherwise).
    pub result: Value,
    pub error: Option<String>,
}

pub trait Engine {
    fn name(&self) -> &str;
    fn run(&self, case: &Case, ctx: &RunCtx) -> Record;
}

/// PhotoshopEX's CLI driving ComfyUI.
pub struct ComfyCli {
    pub cli: PathBuf,
    /// A ComfyUI install for the app-managed sidecar: set as the `integrations.comfyPath`
    /// preference at the start of every invocation, so a server that is down is started by the
    /// first command that needs it (the CLI's session starts from default preferences).
    pub comfy_path: Option<PathBuf>,
    /// Further preferences set before every case (`integrations.allowResearchModels` = true
    /// for the Qwen-Image-2.1 templates, say).
    pub prefs: Vec<(String, Value)>,
}

/// The size of a bench picture (for fractions → pixels).
fn picture_size(path: &Path) -> crate::Result<(u32, u32)> {
    let im = Image::read(path)?;
    Ok((im.width, im.height))
}

/// The last `{"result": …}` line of the CLI's output.
fn last_result(stdout: &str) -> Option<Value> {
    stdout.lines().rev().filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok()).find_map(|v| v.get("result").cloned().or(Some(v)))
}

impl ComfyCli {
    /// The CLI invocation for `case`: arguments and whether a picture comes out.
    pub fn arguments(&self, case: &Case, ctx: &RunCtx) -> crate::Result<(Vec<String>, bool)> {
        let mut args: Vec<String> = vec!["run".into()];
        let tpl = |p: &mut Value| {
            if let Some(t) = &ctx.template {
                p["template"] = json!(t);
            }
        };
        let out = ctx.out.join(format!("{}.png", case.id));
        let mut sidecar: Vec<String> = match &self.comfy_path {
            Some(p) => {
                vec!["--cmd".into(), "prefs.set".into(), "--params".into(), json!({"path": "integrations.comfyPath", "value": p.to_string_lossy()}).to_string()]
            }
            None => Vec::new(),
        };
        for (path, value) in &self.prefs {
            sidecar.extend(["--cmd".into(), "prefs.set".into(), "--params".into(), json!({"path": path, "value": value}).to_string()]);
        }
        let produces = match &case.kind {
            Kind::T2i { prompt, seed, width, height } => {
                args.extend(["--new".into(), json!({"width": width, "height": height}).to_string()]);
                args.extend(sidecar.clone());
                let mut p = json!({"prompt": prompt, "seed": seed, "width": width, "height": height, "target": "layer"});
                tpl(&mut p);
                args.extend(["--cmd".into(), "generate.image".into(), "--params".into(), p.to_string()]);
                true
            }
            Kind::Edit { image, prompt, seed, references } => {
                args.push(ctx.images.join(image).to_string_lossy().into_owned());
                args.extend(sidecar.clone());
                let refs: Vec<Value> = references.iter().map(|r| json!({"path": ctx.images.join(r).to_string_lossy()})).collect();
                let mut p = json!({"prompt": prompt, "seed": seed, "wholePicture": true});
                if !refs.is_empty() {
                    p["references"] = json!(refs);
                }
                tpl(&mut p);
                args.extend(["--cmd".into(), "generate.edit".into(), "--params".into(), p.to_string()]);
                true
            }
            Kind::Fill { image, prompt, seed, rect } => {
                let path = ctx.images.join(image);
                let (w, h) = picture_size(&path)?;
                let [x, y, rw, rh] = rect_px(*rect, w, h);
                args.push(path.to_string_lossy().into_owned());
                args.extend(sidecar.clone());
                args.extend(["--cmd".into(), "select.rect".into(), "--params".into(), json!({"x": x, "y": y, "width": rw, "height": rh}).to_string()]);
                let mut p = json!({"prompt": prompt, "seed": seed});
                tpl(&mut p);
                args.extend(["--cmd".into(), "generate.fill".into(), "--params".into(), p.to_string()]);
                true
            }
            Kind::Sam { image, prompt, negatives } => {
                let path = ctx.images.join(image);
                let (w, h) = picture_size(&path)?;
                args.push(path.to_string_lossy().into_owned());
                args.extend(sidecar);
                let neg: Vec<[u32; 2]> = negatives.iter().map(|p| point_px(*p, w, h)).collect();
                let (cmd, mut p) = match prompt {
                    Prompt::Text(t) => ("select.byText", json!({"prompt": t})),
                    Prompt::Point(pt) => ("select.byPoint", json!({"points": [point_px(*pt, w, h)]})),
                    Prompt::Box(b) => ("select.byPoint", json!({"box": rect_px(*b, w, h)})),
                };
                if !neg.is_empty() {
                    p["negativePoints"] = json!(neg);
                }
                tpl(&mut p);
                args.extend(["--cmd".into(), cmd.into(), "--params".into(), p.to_string()]);
                false
            }
        };
        if produces {
            args.extend(["--out".into(), out.to_string_lossy().into_owned()]);
        }
        Ok((args, produces))
    }
}

impl Engine for ComfyCli {
    fn name(&self) -> &str {
        "comfy"
    }

    fn run(&self, case: &Case, ctx: &RunCtx) -> Record {
        let mut rec = Record {
            id: case.id.clone(),
            engine: self.name().into(),
            image: None,
            wall_ms: 0,
            engine_ms: None,
            seed: case.seed(),
            result: Value::Null,
            error: None,
        };
        let (args, produces) = match self.arguments(case, ctx) {
            Ok(a) => a,
            Err(e) => {
                rec.error = Some(e);
                return rec;
            }
        };
        let t0 = Instant::now();
        let out = Command::new(&self.cli).args(&args).output();
        rec.wall_ms = t0.elapsed().as_millis() as u64;
        match out {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
                rec.result = last_result(&stdout).unwrap_or(Value::Null);
                rec.engine_ms = rec.result.get("ms").and_then(Value::as_u64);
                if !o.status.success() {
                    let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" | ");
                    rec.error = Some(if tail.is_empty() { format!("exit {}", o.status) } else { tail });
                } else if produces {
                    let name = format!("{}.png", case.id);
                    if ctx.out.join(&name).is_file() {
                        rec.image = Some(name);
                    } else {
                        rec.error = Some("the CLI wrote no picture".into());
                    }
                }
            }
            Err(e) => rec.error = Some(format!("cannot run {}: {e}", self.cli.display())),
        }
        rec
    }
}

/// SAM 3 through ONNX Runtime (`gn-sam run`, crate `gn-perception`): the SAM suite only. The
/// result line has the same shape as PhotoshopEX's `select.*` answers, so `compare` reads it.
pub struct SamOrt {
    pub exe: PathBuf,
    /// The directory holding `onnxruntime.dll` and its providers.
    pub ort: PathBuf,
    /// Directories with the CUDA and cuDNN runtime libraries.
    pub libs: Vec<PathBuf>,
    /// The model directory (`vision-encoder.onnx`, …, `tokenizer.json`).
    pub models: PathBuf,
    pub cpu: bool,
    /// Passed through to `--threshold`; PhotoshopEX's default is 0.5.
    pub threshold: f32,
}

impl SamOrt {
    pub fn arguments(&self, case: &Case, ctx: &RunCtx) -> crate::Result<Vec<String>> {
        let Kind::Sam { image, prompt, negatives } = &case.kind else {
            return Err(format!("{}: sam-ort runs SAM cases only", case.id));
        };
        let path = ctx.images.join(image);
        let (w, h) = picture_size(&path)?;
        let mut args: Vec<String> = vec!["run".into(), "--ort".into(), self.ort.to_string_lossy().into_owned()];
        for l in &self.libs {
            args.extend(["--lib".into(), l.to_string_lossy().into_owned()]);
        }
        if self.cpu {
            args.push("--cpu".into());
        }
        args.extend(["--models".into(), self.models.to_string_lossy().into_owned(), "--image".into(), path.to_string_lossy().into_owned()]);
        match prompt {
            Prompt::Text(t) => args.extend(["--text".into(), t.clone()]),
            Prompt::Point(pt) => {
                let p = point_px(*pt, w, h);
                args.extend(["--point".into(), format!("{},{}", p[0], p[1])]);
            }
            Prompt::Box(b) => {
                let r = rect_px(*b, w, h);
                args.extend(["--box".into(), format!("{},{},{},{}", r[0], r[1], r[0] + r[2], r[1] + r[3])]);
            }
        }
        for n in negatives {
            let p = point_px(*n, w, h);
            args.extend(["--negative".into(), format!("{},{}", p[0], p[1])]);
        }
        args.extend(["--threshold".into(), self.threshold.to_string(), "--id".into(), case.id.clone()]);
        args.extend(["--out".into(), ctx.out.join(format!("{}.png", case.id)).to_string_lossy().into_owned()]);
        Ok(args)
    }
}

impl Engine for SamOrt {
    fn name(&self) -> &str {
        "sam-ort"
    }

    fn run(&self, case: &Case, ctx: &RunCtx) -> Record {
        let mut rec = Record { id: case.id.clone(), engine: self.name().into(), image: None, wall_ms: 0, engine_ms: None, seed: None, result: Value::Null, error: None };
        let args = match self.arguments(case, ctx) {
            Ok(a) => a,
            Err(e) => {
                rec.error = Some(e);
                return rec;
            }
        };
        let t0 = Instant::now();
        let out = Command::new(&self.exe).args(&args).output();
        rec.wall_ms = t0.elapsed().as_millis() as u64;
        match out {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
                rec.result = last_result(&stdout).unwrap_or(Value::Null);
                rec.engine_ms = rec.result.get("ms").and_then(Value::as_u64);
                if !o.status.success() {
                    let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" | ");
                    rec.error = Some(if tail.is_empty() { format!("exit {}", o.status) } else { tail });
                }
            }
            Err(e) => rec.error = Some(format!("cannot run {}: {e}", self.exe.display())),
        }
        rec
    }
}

/// stable-diffusion.cpp's `sd-cli` (plan §5.3): the same prompts and seeds through ggml. Text
/// to image for now (Krea 2 Turbo: a GGUF or repack DiT, the Qwen3-VL-4B encoder, the Wan
/// VAE); the Qwen Image Edit path for edit/fill comes with its files. `sd-cli` loads the
/// weights inside every call, so `engine_ms` is sd.cpp's own `generate_image` time (encode +
/// sampling + decode, after loading) and the sampling time is kept in `result`.
pub struct SdCli {
    pub exe: PathBuf,
    pub diffusion_model: PathBuf,
    pub llm: PathBuf,
    pub vae: PathBuf,
    /// Sampler settings; Krea 2 Turbo's recipe by default (8 steps, CFG 1, euler).
    pub steps: u32,
    pub cfg: f32,
    pub sampler: String,
    /// Extra flags appended as they are.
    pub extra: Vec<String>,
}

impl SdCli {
    pub fn krea2_turbo(exe: PathBuf, diffusion_model: PathBuf, llm: PathBuf, vae: PathBuf) -> Self {
        Self { exe, diffusion_model, llm, vae, steps: 8, cfg: 1.0, sampler: "euler".into(), extra: Vec::new() }
    }

    pub fn arguments(&self, case: &Case, ctx: &RunCtx) -> crate::Result<Vec<String>> {
        let Kind::T2i { prompt, seed, width, height } = &case.kind else {
            return Err("the sd.cpp runner does text to image only for now".into());
        };
        let out = ctx.out.join(format!("{}.png", case.id));
        let mut args: Vec<String> = vec![
            "--diffusion-model".into(),
            self.diffusion_model.to_string_lossy().into_owned(),
            "--llm".into(),
            self.llm.to_string_lossy().into_owned(),
            "--vae".into(),
            self.vae.to_string_lossy().into_owned(),
            "-p".into(),
            prompt.clone(),
            "-W".into(),
            width.to_string(),
            "-H".into(),
            height.to_string(),
            "--steps".into(),
            self.steps.to_string(),
            "--cfg-scale".into(),
            self.cfg.to_string(),
            "--sampling-method".into(),
            self.sampler.clone(),
            "--seed".into(),
            seed.to_string(),
            // ComfyUI's noise, so seeds mean the same thing across engines.
            "--rng".into(),
            "cpu".into(),
            "--backend".into(),
            "cuda0".into(),
            "--params-backend".into(),
            "cuda0".into(),
            "--flash-attn".into(),
            "on".into(),
            "-v".into(),
            "-o".into(),
            out.to_string_lossy().into_owned(),
        ];
        args.extend(self.extra.iter().cloned());
        Ok(args)
    }
}

/// `… completed in 9.70s` / `… taking 8.02s` → milliseconds.
fn seconds_after(text: &str, marker: &str) -> Option<u64> {
    text.lines().filter(|l| l.contains(marker)).find_map(|l| {
        let rest = l.split(marker).nth(1)?;
        let num: String = rest.trim_start().chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
        num.parse::<f64>().ok().map(|s| (s * 1000.0) as u64)
    })
}

impl Engine for SdCli {
    fn name(&self) -> &str {
        "sdcpp"
    }

    fn run(&self, case: &Case, ctx: &RunCtx) -> Record {
        let mut rec = Record {
            id: case.id.clone(),
            engine: self.name().into(),
            image: None,
            wall_ms: 0,
            engine_ms: None,
            seed: case.seed(),
            result: Value::Null,
            error: None,
        };
        let args = match self.arguments(case, ctx) {
            Ok(a) => a,
            Err(e) => {
                rec.error = Some(e);
                return rec;
            }
        };
        let t0 = Instant::now();
        let out = Command::new(&self.exe).args(&args).output();
        rec.wall_ms = t0.elapsed().as_millis() as u64;
        match out {
            Ok(o) => {
                let log = format!("{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
                // The log carries ANSI colour codes and `--- file:line` suffixes.
                let clean: String = log.lines().map(|l| l.split(" --- ").next().unwrap_or(l)).collect::<Vec<_>>().join("\n");
                rec.engine_ms = seconds_after(&clean, "generate_image completed in");
                rec.result = json!({
                    "samplingMs": seconds_after(&clean, "sampling completed, taking"),
                    "conditionMs": seconds_after(&clean, "get_learned_condition completed, taking"),
                    "decodeMs": seconds_after(&clean, "decode_first_stage completed, taking"),
                });
                let name = format!("{}.png", case.id);
                if !o.status.success() {
                    let tail: String = clean.lines().filter(|l| l.contains("[E]") || l.contains("error")).rev().take(3).collect::<Vec<_>>().join(" | ");
                    rec.error = Some(if tail.is_empty() { format!("exit {}", o.status) } else { tail });
                } else if ctx.out.join(&name).is_file() {
                    rec.image = Some(name);
                } else {
                    rec.error = Some("sd-cli wrote no picture".into());
                }
            }
            Err(e) => rec.error = Some(format!("cannot run {}: {e}", self.exe.display())),
        }
        rec
    }
}

/// Run a suite on an engine; one record per case, written as it completes.
pub fn run_suite(engine: &dyn Engine, cases: &[Case], ctx: &RunCtx, only: Option<&str>, mut progress: impl FnMut(&Record)) -> crate::Result<Vec<Record>> {
    std::fs::create_dir_all(&ctx.out).map_err(|e| format!("cannot create {}: {e}", ctx.out.display()))?;
    let mut out = Vec::new();
    for c in cases.iter().filter(|c| only.is_none_or(|o| c.id.contains(o))) {
        let rec = engine.run(c, ctx);
        let text = serde_json::to_string_pretty(&rec).map_err(|e| e.to_string())?;
        std::fs::write(ctx.out.join(format!("{}.json", c.id)), text).map_err(|e| format!("cannot write the record: {e}"))?;
        progress(&rec);
        out.push(rec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cli_arguments_follow_the_case() {
        let d = std::env::temp_dir().join(format!("gn-engines-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        Image::new(200, 100).write(&d.join("pic.png")).unwrap_or_else(|e| panic!("{e}"));
        let ctx = RunCtx { images: d.clone(), out: d.join("run"), template: Some("qwen-2.1/image".into()) };
        let e = ComfyCli { cli: PathBuf::from("photocraft-cli"), comfy_path: Some(PathBuf::from("C:/comfy")), prefs: vec![("integrations.allowResearchModels".into(), json!(true))] };
        let t2i = Case { id: "a".into(), kind: Kind::T2i { prompt: "a fox".into(), seed: 7, width: 1024, height: 768 } };
        let (args, produces) = e.arguments(&t2i, &ctx).unwrap_or_else(|e| panic!("{e}"));
        assert!(produces && args.contains(&"--new".to_string()) && args.contains(&"generate.image".to_string()));
        assert!(args.iter().any(|a| a.contains("integrations.comfyPath") && a.contains("C:/comfy")), "the sidecar preference comes first: {args:?}");
        assert!(args.iter().position(|a| a == "prefs.set") < args.iter().position(|a| a == "generate.image"));
        assert!(args.iter().any(|a| a.contains("\"seed\":7") && a.contains("\"template\":\"qwen-2.1/image\"")), "{args:?}");
        let fill = Case { id: "b".into(), kind: Kind::Fill { image: "pic.png".into(), prompt: "grass".into(), seed: 3, rect: [0.5, 0.0, 0.5, 1.0] } };
        let (args, _) = e.arguments(&fill, &ctx).unwrap_or_else(|e| panic!("{e}"));
        assert!(args.iter().any(|a| a == r#"{"height":100,"width":100,"x":100,"y":0}"#), "{args:?}");
        let sam = Case { id: "c".into(), kind: Kind::Sam { image: "pic.png".into(), prompt: Prompt::Point([0.5, 0.5]), negatives: vec![[0.0, 0.0]] } };
        let (args, produces) = e.arguments(&sam, &ctx).unwrap_or_else(|e| panic!("{e}"));
        assert!(!produces && args.contains(&"select.byPoint".to_string()));
        assert!(args.iter().any(|a| a.contains("[100,50]") && a.contains("negativePoints")), "{args:?}");
        assert_eq!(last_result("noise\n{\"id\":\"x\",\"result\":{\"ms\":5}}\n").and_then(|v| v["ms"].as_u64()), Some(5));
        // sd.cpp: the same case as a sd-cli command line, and its log's timings.
        let sd = SdCli::krea2_turbo(PathBuf::from("sd-cli"), PathBuf::from("krea.gguf"), PathBuf::from("llm.gguf"), PathBuf::from("vae.safetensors"));
        let args = sd.arguments(&t2i, &ctx).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            args.windows(2).any(|w| w == ["--seed", "7"]) && args.windows(2).any(|w| w == ["--rng", "cpu"]) && args.windows(2).any(|w| w == ["-W", "1024"]),
            "{args:?}"
        );
        assert!(sd.arguments(&fill, &ctx).is_err(), "text to image only, for now");
        let log = "[I] get_learned_condition completed, taking 1.10s\n[I] sampling completed, taking 8.02s\n[I] generate_image completed in 9.70s";
        assert_eq!((seconds_after(log, "sampling completed, taking"), seconds_after(log, "generate_image completed in")), (Some(8020), Some(9700)));
        let _ = std::fs::remove_dir_all(d);
    }
}
