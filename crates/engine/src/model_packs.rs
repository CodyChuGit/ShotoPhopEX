//! Get Models: one click downloads the model files this machine's memory profile runs
//! ([`crate::memory_profile`]) into the Models folder, in ComfyUI's layout (the owner's ask,
//! 2026-11-05: "a flow to download the models for one click install").
//!
//! The list of files, where each comes from (the authors' own Hugging Face pages; nothing is
//! re-hosted), its size and SHA-256 live in `model-packs.json` beside this crate. A pack is the
//! set a profile needs; extras are opt-in. NVIDIA's CUDA libraries the worker loads come from
//! NVIDIA's redistributable archives when no runtime folder is set.
//!
//! Downloads go through the system's `curl` (the app builds no TLS stack, as for App Icon's
//! download): to `<file>.part`, resumed with `-C -` after an interruption, then checked for its
//! size and SHA-256 before it is renamed into place. A file that fails its check is deleted and
//! fetched once more, then reported. A Hugging Face token, when given, goes to `curl` on its
//! standard input, never on its command line, and is never saved.
//!
//! `models.packs` lists what the current profile's pack holds and what is missing; `models.get`
//! downloads it as a background job (`jobs.cancel` stops it; a later run resumes).

use crate::commands::CommandSpec;
use crate::memory_profile::{self, Profile};
use crate::{EngineError, Result, Session};
use serde::Deserialize;
use serde_json::{Value, json};

pub const PACKS: &str = "models.packs";
pub const GET: &str = "models.get";

const MANIFEST: &str = include_str!("../model-packs.json");

/// One model file.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelFile {
    pub id: String,
    /// What people read ("Krea 2 Turbo").
    pub name: String,
    /// The Models folder's subfolder (`diffusion_models`, `loras/moss-image-edit`, …).
    pub folder: String,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    pub url: String,
    /// The page asks for a sign-in (and its licence accepted) before it serves the file.
    #[serde(default)]
    pub gated: bool,
}

/// One of NVIDIA's redistributable archives and the libraries taken from it.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeArchive {
    pub id: String,
    pub name: String,
    pub url: String,
    pub bytes: u64,
    pub sha256: String,
    /// Paths inside the archive, copied flat into the runtime folder.
    pub libraries: Vec<String>,
}

/// A set of files (a profile's pack, or an extra).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Pack {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// The memory profiles this pack is for (empty for an extra).
    #[serde(default)]
    pub profiles: Vec<String>,
    /// The systems it runs on, as `std::env::consts::OS` names them (empty for all): the MLX
    /// models are `["macos"]`.
    #[serde(default)]
    pub os: Vec<String>,
    pub files: Vec<String>,
}

impl Pack {
    /// Whether this system can run the pack.
    pub fn runs_here(&self) -> bool {
        self.os.is_empty() || self.os.iter().any(|o| o == std::env::consts::OS)
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub files: Vec<ModelFile>,
    #[serde(default)]
    pub runtime: Vec<RuntimeArchive>,
    pub packs: Vec<Pack>,
    #[serde(default)]
    pub extras: Vec<Pack>,
}

impl Manifest {
    pub fn file(&self, id: &str) -> Option<&ModelFile> {
        self.files.iter().find(|f| f.id == id)
    }
    /// The pack for `profile`; with none detected, the first pack (today's defaults).
    pub fn pack_for(&self, profile: Option<Profile>) -> Option<&Pack> {
        let name = profile.map(Profile::name);
        self.packs.iter().find(|p| name.is_some_and(|n| p.profiles.iter().any(|x| x == n))).or_else(|| self.packs.first())
    }
    pub fn pack(&self, id: &str) -> Option<&Pack> {
        self.packs.iter().chain(&self.extras).find(|p| p.id == id)
    }
}

/// The built-in manifest.
pub fn manifest() -> &'static Manifest {
    static M: std::sync::OnceLock<Manifest> = std::sync::OnceLock::new();
    // Checked by `the_manifest_is_complete`; a broken file offers nothing rather than panicking.
    M.get_or_init(|| serde_json::from_str(MANIFEST).unwrap_or_default())
}

/// Where downloads go when no Models folder is set: `Documents/ShotoPhopEX/models`.
pub fn default_models_root() -> Option<std::path::PathBuf> {
    home().map(|h| h.join("Documents").join("ShotoPhopEX").join("models"))
}

/// Where the CUDA libraries go when no runtime folder is set: `Documents/ShotoPhopEX/runtime`.
pub fn default_runtime_root() -> Option<std::path::PathBuf> {
    home().map(|h| h.join("Documents").join("ShotoPhopEX").join("runtime"))
}

fn home() -> Option<std::path::PathBuf> {
    ["USERPROFILE", "HOME"].into_iter().filter_map(std::env::var_os).find(|v| !v.is_empty()).map(std::path::PathBuf::from)
}

/// The folder files land in: the Models folder in use, else [`default_models_root`].
fn target_root(s: &Session) -> Option<std::path::PathBuf> {
    let r = s.prefs().integrations.models_root();
    if r.is_empty() { default_models_root() } else { Some(r.into()) }
}

/// A file counts as installed when it is there at its full size (hashing tens of gigabytes on
/// every look would take minutes; downloads are hashed as they finish).
pub fn installed(root: &std::path::Path, f: &ModelFile) -> bool {
    std::fs::metadata(path_of(root, f)).is_ok_and(|m| m.is_file() && m.len() == f.bytes)
}

/// Bytes still to fetch for `f`: none when installed, else what a resumable `.part` lacks.
fn left(root: Option<&std::path::Path>, f: &ModelFile) -> u64 {
    let Some(root) = root else { return f.bytes };
    if installed(root, f) {
        return 0;
    }
    let dest = path_of(root, f);
    let had = std::fs::metadata(part_of(&dest)).map(|m| m.len()).unwrap_or(0);
    f.bytes.saturating_sub(had.min(f.bytes))
}

fn part_of(dest: &std::path::Path) -> std::path::PathBuf {
    dest.with_file_name(format!("{}.part", dest.file_name().and_then(|n| n.to_str()).unwrap_or("download")))
}

fn path_of(root: &std::path::Path, f: &ModelFile) -> std::path::PathBuf {
    let mut p = root.to_path_buf();
    for part in f.folder.split('/').filter(|s| !s.is_empty() && *s != "." && *s != "..") {
        p.push(part);
    }
    p.join(&f.file)
}

/// The runtime is needed when no runtime folder is set (or found beside the exe). Windows
/// only: the archives are NVIDIA's Windows builds.
fn runtime_needed(s: &Session) -> bool {
    cfg!(windows) && s.prefs().integrations.runtime_folders().is_empty()
}

/// The ids of `pack`'s files plus the chosen extras', without duplicates, in order.
fn wanted(m: &Manifest, pack: &Pack, extras: &[String]) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    for id in pack.files.iter().chain(extras.iter().filter_map(|e| m.pack(e)).flat_map(|p| p.files.iter())) {
        if !ids.contains(id) {
            ids.push(id.clone());
        }
    }
    ids
}

fn file_json(root: Option<&std::path::Path>, f: &ModelFile) -> Value {
    json!({
        "id": f.id, "name": f.name, "folder": f.folder, "file": f.file, "bytes": f.bytes, "gated": f.gated,
        "installed": root.is_some_and(|r| installed(r, f)),
    })
}

fn pack_json(m: &Manifest, root: Option<&std::path::Path>, p: &Pack) -> Value {
    let files: Vec<Value> = p.files.iter().filter_map(|id| m.file(id)).map(|f| file_json(root, f)).collect();
    let missing: u64 = p.files.iter().filter_map(|id| m.file(id)).map(|f| left(root, f)).sum();
    let total: u64 = p.files.iter().filter_map(|id| m.file(id)).map(|f| f.bytes).sum();
    json!({"id": p.id, "name": p.name, "description": p.description, "files": files, "bytes": total, "missingBytes": missing})
}

fn run_packs(s: &mut Session, _p: &Value) -> Result<Value> {
    let m = manifest();
    let profile = memory_profile::active(s);
    let root = target_root(s);
    let pack = m.pack_for(profile).ok_or_else(|| EngineError::Other("no model pack is defined".into()))?;
    // The other packs and the add-ons; the profile's own pack is not offered twice.
    let extras: Vec<Value> = m.extras.iter().filter(|e| e.id != pack.id && e.runs_here()).map(|e| pack_json(m, root.as_deref(), e)).collect();
    let runtime_bytes: u64 = m.runtime.iter().map(|r| r.bytes).sum();
    Ok(json!({
        "profile": profile.map(Profile::name),
        "profileLabel": profile.map(Profile::label),
        "pack": pack_json(m, root.as_deref(), pack),
        "extras": extras,
        "modelsRoot": root.map(|r| r.to_string_lossy().into_owned()),
        "modelsRootSet": !s.prefs().integrations.models_root().is_empty(),
        "runtime": {"needed": runtime_needed(s) && !m.runtime.is_empty(), "bytes": runtime_bytes},
    }))
}

/// What one run downloads, captured before the job starts.
#[derive(Clone, Debug)]
struct Plan {
    root: std::path::PathBuf,
    files: Vec<ModelFile>,
    runtime: Option<(std::path::PathBuf, Vec<RuntimeArchive>)>,
    token: Option<String>,
    set_models_path: bool,
}

fn run_get(s: &mut Session, p: &Value) -> Result<Value> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (s, p);
        Err(EngineError::Other("not available in the web build".into()))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let bad = |m: &str| EngineError::BadParams { cmd: GET.into(), msg: m.into() };
        let m = manifest();
        let pack = match p.get("pack").and_then(Value::as_str) {
            Some(id) => m.pack(id).ok_or_else(|| bad("unknown `pack` (see models.packs)"))?,
            None => m.pack_for(memory_profile::active(s)).ok_or_else(|| bad("no model pack is defined"))?,
        };
        let extras: Vec<String> = match p.get("extras") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| bad("`extras` must be pack ids"))).collect::<Result<_>>()?,
            Some(_) => return Err(bad("`extras` must be a list of pack ids")),
        };
        if let Some(e) = extras.iter().find(|e| m.pack(e).is_none()) {
            return Err(bad(&format!("unknown extra `{e}`")));
        }
        if let Some(e) = extras.iter().find(|e| m.pack(e).is_some_and(|p| !p.runs_here())) {
            return Err(bad(&format!("`{e}` does not run on this system")));
        }
        let token = p.get("token").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
        if token.as_deref().is_some_and(|t| t.len() > 512 || t.chars().any(|c| c.is_control() || c.is_whitespace())) {
            return Err(bad("`token` is not a Hugging Face token"));
        }
        let root = target_root(s).ok_or_else(|| EngineError::Other("no Models folder is set and no Documents folder was found".into()))?;
        let files: Vec<ModelFile> = wanted(m, pack, &extras).iter().filter_map(|id| m.file(id)).filter(|f| !installed(&root, f)).cloned().collect();
        let want_runtime = p.get("runtime").and_then(Value::as_bool).unwrap_or(true) && runtime_needed(s) && !m.runtime.is_empty();
        let runtime = if want_runtime { default_runtime_root().map(|r| (r, m.runtime.clone())) } else { None };
        let plan = Plan { root, files, runtime, token, set_models_path: s.prefs().integrations.models_root().is_empty() };
        let label = format!("Get Models: {}", pack.name);
        crate::jobs::run(s, &label, false, move |ctx| download_all(&plan, ctx).map(|r| (plan, r)), apply_downloads)
    }
}

/// What a finished run did.
#[derive(Clone, Debug, Default)]
struct Outcome {
    downloaded: Vec<String>,
    runtime: bool,
}

fn apply_downloads(s: &mut Session, (plan, out): (Plan, Outcome)) -> Result<Value> {
    let root = plan.root.to_string_lossy().into_owned();
    let runtime = plan.runtime.as_ref().filter(|_| out.runtime).map(|(r, _)| r.to_string_lossy().into_owned());
    s.edit_prefs(|p| {
        if plan.set_models_path {
            p.integrations.models_path = root.clone();
        }
        if let Some(r) = &runtime {
            p.integrations.runtime_path = r.clone();
        }
    });
    Ok(json!({"downloaded": out.downloaded, "modelsRoot": root, "runtimePath": runtime}))
}

#[cfg(not(target_arch = "wasm32"))]
fn download_all(plan: &Plan, ctx: &crate::jobs::JobCtx) -> Result<Outcome> {
    let runtime_items: Vec<(String, u64)> = plan.runtime.iter().flat_map(|(_, a)| a.iter().map(|r| (r.name.clone(), r.bytes))).collect();
    let total: u64 = plan.files.iter().map(|f| f.bytes).sum::<u64>() + runtime_items.iter().map(|r| r.1).sum::<u64>();
    let mut done: u64 = 0;
    let mut out = Outcome::default();
    let report = |done: u64, now: u64, name: &str| {
        let f = if total == 0 { 1.0 } else { (done + now) as f32 / total as f32 };
        ctx.progress(f.min(0.999), &format!("{name}: {} of {}", gb(done + now), gb(total)));
    };
    for f in &plan.files {
        let dest = path_of(&plan.root, f);
        fetch_verified(&f.url, &dest, f.bytes, &f.sha256, plan.token.as_deref(), f.gated, ctx, &|now| report(done, now, &f.name))
            .map_err(|e| EngineError::Other(format!("{}: {e}", f.name)))?;
        done += f.bytes;
        out.downloaded.push(f.id.clone());
    }
    if let Some((dir, archives)) = &plan.runtime {
        for a in archives {
            let zip = dir.join("downloads").join(a.url.rsplit('/').next().unwrap_or("archive.zip"));
            fetch_verified(&a.url, &zip, a.bytes, &a.sha256, None, false, ctx, &|now| report(done, now, &a.name))
                .map_err(|e| EngineError::Other(format!("{}: {e}", a.name)))?;
            extract_libraries(&zip, dir, &a.libraries).map_err(|e| EngineError::Other(format!("{}: {e}", a.name)))?;
            let _ = std::fs::remove_file(&zip);
            done += a.bytes;
        }
        let _ = std::fs::remove_dir(dir.join("downloads"));
        out.runtime = true;
    }
    ctx.progress(1.0, "done");
    Ok(out)
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

/// Download `url` to `dest` through `<dest>.part` (resumed when one is there), check its size
/// and SHA-256, and move it into place. One more attempt from scratch when the check fails.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn fetch_verified(
    url: &str,
    dest: &std::path::Path,
    bytes: u64,
    sha256: &str,
    token: Option<&str>,
    gated: bool,
    ctx: &crate::jobs::JobCtx,
    progress: &dyn Fn(u64),
) -> std::result::Result<(), String> {
    let dir = dest.parent().ok_or("no folder")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("couldn't create {}: {e}", dir.display()))?;
    let part = part_of(dest);
    for attempt in 0..2 {
        // A part already longer than the file can't be resumed.
        if std::fs::metadata(&part).is_ok_and(|m| m.len() > bytes) {
            let _ = std::fs::remove_file(&part);
        }
        curl(url, &part, bytes, token, ctx, progress)
            .map_err(|e| if gated && token.is_none() { format!("{e} (this model needs a Hugging Face sign-in: add a token)") } else { e })?;
        let len = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let ok = len == bytes && sha256_of(&part, ctx)?.eq_ignore_ascii_case(sha256);
        if ok {
            return std::fs::rename(&part, dest).map_err(|e| format!("couldn't move the download into place: {e}"));
        }
        let _ = std::fs::remove_file(&part);
        if attempt == 1 {
            return Err(format!("the download didn't match its checksum twice (got {len} of {bytes} bytes)"));
        }
    }
    Err("the download didn't match its checksum".into())
}

/// The system's `curl` or `tar` by full path on Windows (System32, where Windows 10 and later
/// ship both), so a `curl.exe` or `tar.exe` beside the executable (a portable zip unpacked into
/// Downloads) or in a user-writable `PATH` entry is never the one run; the plain name elsewhere.
#[cfg(not(target_arch = "wasm32"))]
fn system_tool(name: &str) -> std::path::PathBuf {
    #[cfg(windows)]
    {
        let root = std::env::var_os("SystemRoot").map_or_else(|| std::path::PathBuf::from(r"C:\Windows"), std::path::PathBuf::from);
        root.join("System32").join(format!("{name}.exe"))
    }
    #[cfg(not(windows))]
    {
        std::path::PathBuf::from(name)
    }
}

/// Run `curl` to fetch `url` into `part` (resuming), reporting the part's size as it grows and
/// stopping it when the job is cancelled.
#[cfg(not(target_arch = "wasm32"))]
fn curl(
    url: &str,
    part: &std::path::Path,
    bytes: u64,
    token: Option<&str>,
    ctx: &crate::jobs::JobCtx,
    progress: &dyn Fn(u64),
) -> std::result::Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    if !url.starts_with("https://") {
        return Err("only https downloads are allowed".into());
    }
    if std::fs::metadata(part).is_ok_and(|m| m.len() == bytes) {
        return Ok(());
    }
    let mut cmd = Command::new(system_tool("curl"));
    cmd.args([
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "--continue-at",
        "-",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--output",
    ])
    .arg(part);
    if token.is_some() {
        // The header comes from standard input, so the token is never on a command line.
        cmd.args(["--header", "@-"]);
    }
    cmd.arg("--").arg(url).stdin(if token.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::null()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| format!("couldn't run curl: {e}"))?;
    if let (Some(t), Some(mut stdin)) = (token, child.stdin.take()) {
        let _ = writeln!(stdin, "Authorization: Bearer {t}");
    }
    loop {
        if ctx.cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err("cancelled".into());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                if status.success() {
                    return Ok(());
                }
                let mut err = String::new();
                if let Some(mut e) = child.stderr.take() {
                    let _ = std::io::Read::read_to_string(&mut e, &mut err);
                }
                let err = err.trim().trim_start_matches("curl: ").to_string();
                return Err(if err.is_empty() { format!("the download failed ({status})") } else { err });
            }
            Ok(None) => {
                progress(std::fs::metadata(part).map(|m| m.len()).unwrap_or(0));
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// The file's SHA-256 in lowercase hex, read in 8 MiB pieces (cancellable).
#[cfg(not(target_arch = "wasm32"))]
fn sha256_of(path: &std::path::Path, ctx: &crate::jobs::JobCtx) -> std::result::Result<String, String> {
    use sha2::Digest;
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        if ctx.cancelled() {
            return Err("cancelled".into());
        }
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Copy `libraries` (paths inside the zip) flat into `dir`, with the system's `tar` (Windows 10
/// and later ship bsdtar, which reads zip).
#[cfg(not(target_arch = "wasm32"))]
fn extract_libraries(zip: &std::path::Path, dir: &std::path::Path, libraries: &[String]) -> std::result::Result<(), String> {
    let tmp = dir.join("downloads").join("unpacked");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let mut cmd = std::process::Command::new(system_tool("tar"));
    cmd.arg("-xf").arg(zip).arg("-C").arg(&tmp);
    for l in libraries {
        cmd.arg(l);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().map_err(|e| format!("couldn't run tar: {e}"))?;
    if !out.status.success() {
        return Err(format!("couldn't unpack the archive: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    for l in libraries {
        let from = tmp.join(l);
        let name = from.file_name().ok_or("bad library path")?.to_owned();
        std::fs::rename(&from, dir.join(&name))
            .or_else(|_| std::fs::copy(&from, dir.join(&name)).map(drop))
            .map_err(|e| format!("couldn't place {}: {e}", name.to_string_lossy()))?;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

fn always(_s: &Session) -> std::result::Result<(), String> {
    Ok(())
}

fn native_only(_s: &Session) -> std::result::Result<(), String> {
    crate::generate_cmds::web_unavailable()
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: PACKS,
            label: "Model Packs",
            menu: &[],
            shortcut: None,
            params: r#"{} → {profile,profileLabel,pack:{id,name,description,files:[{id,name,folder,file,bytes,gated,installed}],bytes,missingBytes},extras:[pack…],modelsRoot,modelsRootSet,runtime:{needed,bytes}} (the memory profile's model pack and what is installed; a file counts as installed at its full size)"#,
            enabled: always,
            run: run_packs,
            journal: false,
        },
        CommandSpec {
            id: GET,
            label: "Get Models",
            menu: &[],
            shortcut: None,
            params: r#"{"pack"?:id (default: the memory profile's),"extras"?:[id],"runtime"?:bool=true (NVIDIA's CUDA libraries when no runtime folder is set, Windows),"token"?:"hf_…" (for gated models; used once, never saved)} → {downloaded:[id],modelsRoot,runtimePath} (a background job: downloads the missing files into the Models folder, or Documents/ShotoPhopEX/models when none is set (then set), resuming partial downloads and checking each file's SHA-256; jobs.cancel stops it)"#,
            enabled: native_only,
            run: run_get,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_is_complete() {
        let m: Manifest = serde_json::from_str(MANIFEST).expect("model-packs.json parses");
        assert!(!m.packs.is_empty());
        for f in &m.files {
            assert!(f.url.starts_with("https://"), "{} downloads over https", f.id);
            assert_eq!(f.sha256.len(), 64, "{} has a SHA-256", f.id);
            assert!(f.sha256.chars().all(|c| c.is_ascii_hexdigit()), "{} SHA-256 is hex", f.id);
            assert!(f.bytes > 0, "{} has a size", f.id);
            assert!(
                !f.file.contains(['/', '\\']) && !f.folder.contains('\\') && f.folder.split('/').all(|s| !s.is_empty() && s != "." && s != ".."),
                "{} paths stay in the Models folder",
                f.id
            );
            assert_eq!(m.files.iter().filter(|g| g.id == f.id).count(), 1, "{} is listed once", f.id);
        }
        for p in m.packs.iter().chain(&m.extras) {
            for id in &p.files {
                assert!(m.file(id).is_some(), "pack {} names an unknown file {id}", p.id);
            }
        }
        for r in &m.runtime {
            assert!(r.url.starts_with("https://") && r.sha256.len() == 64 && !r.libraries.is_empty(), "{}", r.id);
        }
        for p in m.packs.iter().chain(&m.extras) {
            assert!(p.os.iter().all(|o| ["windows", "linux", "macos"].contains(&o.as_str())), "pack {} names an unknown system", p.id);
        }
        // Every profile has a pack.
        for p in Profile::ALL {
            assert!(m.packs.iter().any(|k| k.profiles.iter().any(|x| x == p.name())), "no pack for {}", p.name());
        }
    }

    #[test]
    fn each_pack_holds_what_auto_runs_on_its_profile() {
        let m = manifest();
        for p in Profile::ALL {
            let pack = m.pack_for(Some(p)).unwrap();
            let files: Vec<&str> = pack.files.iter().filter_map(|id| m.file(id)).map(|f| f.file.as_str()).collect();
            let image = photocraft_genai::template::find(memory_profile::image_template(Some(p))).unwrap();
            if matches!(p, Profile::Gb8 | Profile::Mac8) {
                continue; // no image model fits yet
            }
            if p == Profile::Mac16 {
                // MLX runs MOSS Turbo as MOSS-Image-SFT with the 4-step LoRA (gn-diffusion's
                // mlx_recipe), not the template's Kromoss files.
                for f in [
                    "moss_image_sft_int8_convrot.safetensors",
                    "moss_image_sft_4step_v2.0.safetensors",
                    "adapter_model.safetensors",
                    "qwen3.5_9b_int8_convrot.safetensors",
                ] {
                    assert!(files.contains(&f), "mac16's pack lacks {f}");
                }
                continue;
            }
            for slot in &image.meta.models {
                let name = slot.default.rsplit('/').next().unwrap();
                assert!(files.contains(&name), "{}'s pack lacks {} for {}", p.name(), slot.default, image.meta.id);
            }
        }
    }

    #[test]
    fn packs_report_missing_files_and_the_default_folder() {
        let mut s = Session::new();
        s.edit_prefs(|p| p.integrations.models_path = std::env::temp_dir().join("pc-no-such-models").to_string_lossy().into_owned());
        let v = run_packs(&mut s, &json!({})).unwrap();
        assert_eq!(v["pack"]["id"], manifest().packs[0].id);
        assert!(v["pack"]["files"].as_array().unwrap().iter().all(|f| f["installed"] == false));
        assert_eq!(v["pack"]["missingBytes"], v["pack"]["bytes"]);
        memory_profile::record_detected(&mut s, Some(Profile::Gb12));
        let v = run_packs(&mut s, &json!({})).unwrap();
        assert_eq!(v["profile"], "12gb");
        assert!(v["pack"]["files"].as_array().unwrap().iter().any(|f| f["file"] == "kromoss_int8_convrot.safetensors"));
    }

    #[test]
    fn a_file_of_the_right_size_counts_as_installed() {
        let dir = std::env::temp_dir().join(format!("pc-packs-{}", std::process::id()));
        let f = ModelFile {
            id: "t".into(),
            name: "T".into(),
            folder: "loras/sub".into(),
            file: "x.safetensors".into(),
            bytes: 3,
            sha256: "0".repeat(64),
            url: "https://example.invalid/x".into(),
            gated: false,
        };
        assert!(!installed(&dir, &f));
        assert_eq!(left(Some(&dir), &f), 3);
        std::fs::create_dir_all(dir.join("loras").join("sub")).unwrap();
        std::fs::write(dir.join("loras").join("sub").join("x.safetensors.part"), b"a").unwrap();
        assert_eq!(left(Some(&dir), &f), 2, "a partial download counts what it already has");
        std::fs::write(dir.join("loras").join("sub").join("x.safetensors"), b"ab").unwrap();
        assert!(!installed(&dir, &f), "short file");
        std::fs::write(dir.join("loras").join("sub").join("x.safetensors"), b"abc").unwrap();
        assert!(installed(&dir, &f));
        assert_eq!(left(Some(&dir), &f), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_checksum_matches_a_known_value() {
        let p = std::env::temp_dir().join(format!("pc-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        let h = sha256_of(&p, &crate::jobs::JobCtx::default()).unwrap();
        assert_eq!(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn downloads_run_the_systems_own_curl_and_tar() {
        for name in ["curl", "tar"] {
            let p = system_tool(name);
            if cfg!(windows) {
                assert!(p.is_absolute() && p.ends_with(format!("System32/{name}.exe")), "{}", p.display());
            } else {
                assert_eq!(p, std::path::PathBuf::from(name));
            }
        }
    }

    #[test]
    fn get_rejects_unknown_packs_and_odd_tokens() {
        let mut s = Session::new();
        assert!(run_get(&mut s, &json!({"pack": "nope"})).is_err());
        assert!(run_get(&mut s, &json!({"extras": ["nope"]})).is_err());
        assert!(run_get(&mut s, &json!({"token": "hf_a b"})).is_err());
    }

    #[test]
    fn mac_only_extras_are_offered_only_on_a_mac() {
        let mut s = Session::new();
        s.edit_prefs(|p| p.integrations.models_path = std::env::temp_dir().join("pc-no-such-models").to_string_lossy().into_owned());
        let v = run_packs(&mut s, &json!({})).unwrap();
        let offered = v["extras"].as_array().unwrap().iter().any(|e| e["id"] == "flux3-klein-q4");
        assert_eq!(offered, cfg!(target_os = "macos"));
        let klein = manifest().pack("flux3-klein-q8").unwrap();
        assert_eq!(klein.os, ["macos"]);
        assert!(
            klein
                .files
                .iter()
                .filter(|id| *id != "flux2-vae")
                .all(|id| manifest().file(id).is_some_and(|f| f.folder.starts_with("mlx/flux-3-image-klein-q8/")))
        );
        if !cfg!(target_os = "macos") {
            let err = run_get(&mut s, &json!({"extras": ["flux3-klein-q4"]})).unwrap_err();
            assert!(err.to_string().contains("does not run on this system"), "{err}");
        }
    }
}
