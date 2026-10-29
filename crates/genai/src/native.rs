//! The native engine as the editor sees it (plan-native-engine §3.4): a [`GenerativeBackend`]
//! that talks to a worker process over `gn-engine`'s channel. The worker is this very
//! executable started with `--engine` ([`engine_main`]), so the editor and its engine ship as
//! one program; a CUDA fault ends the worker, never a document, and the next call starts a
//! fresh one.
//!
//! With a runtime folder and a model root configured (`integrations.runtimePath`,
//! `integrations.modelsPath`) the worker serves `gn-perception`'s engine: SAM 3.1 on candle
//! (plan Phase 1, Rust rebuild R1; a backend only with `native-diffusion`), and, built with the `native-diffusion` feature, `gn-diffusion`'s:
//! Qwen-Image-2.1 text to image from the snapshot under the model root (plan Phase 2), the two
//! behind one `gn_engine::Multi`. Without them it serves `gn-engine`'s fake engine, which is
//! enough to route a request end to end (and to test the editor without a GPU). Pixels cross as
//! binary frames; other values use the editor’s `Request`/`Response` types.
//! The runtime folders go on the worker's `PATH` too: the CUDA libraries candle links are found
//! there.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use gn_engine::client::{ClientError, EngineClient};
use gn_engine::protocol::{Blob, ErrorKind, RunRequest};
use serde::Serialize;
use serde_json::Value;

use crate::{Error, GenerativeBackend, Gray8, Health, Progress, Request, Response, Result, Rgba8, Task, Timings, template};

/// The engine worker, started on first use and restarted after it dies.
pub struct NativeBackend {
    program: PathBuf,
    args: Vec<String>,
    /// Directories put first on the worker's `PATH` (the runtime folders).
    path_dirs: Vec<PathBuf>,
    token: String,
    client: Mutex<Option<EngineClient>>,
    /// A worker we did not start (tests serve one in-process): never respawned.
    attached: bool,
    /// The file the worker's stderr goes to ([`Self::with_log`]); `None` inherits ours.
    log: Option<PathBuf>,
    /// What holds the client right now ([`IDLE`], [`STARTING`], [`RUNNING`]), so a status
    /// asked meanwhile answers without waiting for the lock.
    busy: AtomicU8,
    /// Why the last start failed, until one succeeds.
    start_error: Mutex<Option<String>>,
    timeout_ms: AtomicU64,
}

const IDLE: u8 = 0;
const STARTING: u8 = 1;
const RUNNING: u8 = 2;

/// The worker as the status bar and the engine page show it (`generate.engine.status`'s
/// `native`): `stopped` (no worker; the next generative command starts one), `starting`,
/// `running` (a request in flight), `ready`, `failed` (the last start failed, or the worker
/// answered unhealthy; `detail` says why).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeStatus {
    pub state: String,
    pub pid: Option<u32>,
    /// The worker's engines (`ort+qir`) and the card it runs on.
    pub engine: String,
    pub device: String,
    pub vram_total: Option<u64>,
    pub vram_free: Option<u64>,
    /// The model folders loaded now.
    pub resident: Vec<String>,
    pub detail: String,
}

/// Clears the busy mark when the holder of the client is done (also on an early return).
struct Busy<'a>(&'a AtomicU8);
impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.store(IDLE, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for NativeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeBackend").field("program", &self.program).field("args", &self.args).finish()
    }
}

/// The argument that turns this executable into the engine worker.
pub const ENGINE_FLAG: &str = "--engine";

/// Cooperative deadline: the protocol polls cancellation while waiting for worker replies.
struct Deadline<'a> {
    progress: &'a dyn Progress,
    started: std::time::Instant,
    timeout: std::time::Duration,
    expired: std::cell::Cell<bool>,
}
impl Progress for Deadline<'_> {
    fn report(&self, fraction: f32, message: &str) {
        self.progress.report(fraction, message);
    }
    fn cancelled(&self) -> bool {
        if self.progress.cancelled() {
            return true;
        }
        if self.started.elapsed() >= self.timeout {
            self.expired.set(true);
            return true;
        }
        false
    }
}

/// How a run that lost its worker mid-way begins its error.
const STOPPED: &str = "the engine stopped";

/// `attempt`, and once more when it lost its worker mid-run and a fresh one can be started
/// (`respawns`) and the run was not cancelled.
fn retry_after_stop<T>(respawns: bool, progress: &dyn Progress, attempt: impl Fn() -> Result<T>) -> Result<T> {
    match attempt() {
        Err(Error::Unavailable(m)) if m.starts_with(STOPPED) && respawns && !progress.cancelled() => {
            progress.report(0.0, "The engine restarted; trying again");
            attempt()
        }
        other => other,
    }
}

impl NativeBackend {
    /// A backend that starts `program args…` as its worker when first needed.
    pub fn new(program: PathBuf, args: Vec<String>) -> Self {
        Self {
            program,
            args,
            path_dirs: Vec::new(),
            token: gn_engine::client::new_token(),
            client: Mutex::new(None),
            attached: false,
            log: None,
            busy: AtomicU8::new(IDLE),
            start_error: Mutex::new(None),
            timeout_ms: AtomicU64::new(600_000),
        }
    }

    /// Change the generation deadline without unloading the worker’s models.
    pub fn set_timeout(&self, timeout: std::time::Duration) {
        self.timeout_ms.store(u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    /// Write the worker's stderr to `path` (truncated at each start), with `RUST_LOG=info`
    /// unless this process sets its own, so the engine's phase timings and refusals are there.
    pub fn with_log(mut self, path: PathBuf) -> Self {
        self.log = Some(path);
        self
    }

    pub fn log_path(&self) -> Option<&std::path::Path> {
        self.log.as_deref()
    }

    /// This executable with `--engine` (the editor and the CLI both answer it) and `extra` flags
    /// for the worker ([`worker_args`]).
    pub fn this_executable(extra: &[String]) -> Result<Self> {
        let exe = std::env::current_exe().map_err(|e| Error::Unavailable(format!("cannot find this executable: {e}")))?;
        let mut args = vec![ENGINE_FLAG.to_string()];
        args.extend(extra.iter().cloned());
        Ok(Self::new(exe, args))
    }

    /// Put `dirs` first on the worker's `PATH` ([`runtime_dirs`]).
    pub fn with_path_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.path_dirs = dirs;
        self
    }

    /// The worker's command line after the program.
    pub fn args(&self) -> &[String] {
        &self.args
    }

    /// Whether this backend joined a worker it did not start ([`Self::attached`]): its flags
    /// are not this process's to compare, and `stop` leaves it running.
    pub fn is_attached(&self) -> bool {
        self.attached
    }

    /// The worker's environment: `PATH` with the runtime folders first, when there are any.
    fn env(&self) -> Vec<(String, String)> {
        let mut env = Vec::new();
        if self.log.is_some() && std::env::var_os("RUST_LOG").is_none() {
            env.push(("RUST_LOG".to_string(), "info".to_string()));
        }
        if self.path_dirs.is_empty() {
            return env;
        }
        let current = std::env::var_os("PATH").unwrap_or_default();
        let joined = std::env::join_paths(self.path_dirs.iter().cloned().chain(std::env::split_paths(&current)));
        if let Ok(p) = joined {
            env.push(("PATH".to_string(), p.to_string_lossy().into_owned()));
        }
        env
    }

    /// A backend over a worker already listening (tests serve `gn_engine::fake::FakeEngine` on
    /// a thread).
    pub fn attached(port: u16, token: &str) -> Result<Self> {
        let client = EngineClient::connect(port, token).map_err(|e| Error::Unavailable(e.to_string()))?;
        let mut b = Self::new(PathBuf::new(), Vec::new());
        b.token = token.to_string();
        b.client = Mutex::new(Some(client));
        b.attached = true;
        Ok(b)
    }

    /// Run `f` on a live client, starting or restarting the worker first when needed. A call
    /// that finds the channel broken drops the client so the next one starts afresh.
    fn with_client<T>(&self, f: impl FnOnce(&mut EngineClient) -> std::result::Result<T, ClientError>) -> Result<T> {
        let mut guard = self.client.lock().unwrap_or_else(|e| e.into_inner());
        let dead = guard.as_mut().is_some_and(|c| !c.alive());
        if dead {
            *guard = None;
        }
        if guard.is_none() {
            if self.attached {
                return Err(Error::Unavailable("the engine is gone".into()));
            }
            let prev = self.busy.swap(STARTING, Ordering::SeqCst);
            let log = self.log.as_ref().and_then(|p| std::fs::File::create(p).ok());
            let spawned = EngineClient::spawn_logged(&self.program, &self.args, &self.token, &self.env(), log);
            self.busy.store(prev, Ordering::SeqCst);
            let mut start_error = self.start_error.lock().unwrap_or_else(|e| e.into_inner());
            match spawned {
                Ok(c) => {
                    *start_error = None;
                    *guard = Some(c);
                }
                Err(e) => {
                    *start_error = Some(e.to_string());
                    return Err(Error::Unavailable(e.to_string()));
                }
            }
        }
        let Some(client) = guard.as_mut() else { return Err(Error::Unavailable("no engine".into())) };
        match f(client) {
            Ok(v) => Ok(v),
            Err(ClientError::Gone(m)) => {
                *guard = None;
                Err(Error::Unavailable(format!("{STOPPED}: {m}")))
            }
            Err(e) => Err(map_err(e)),
        }
    }

    /// Stop the worker (the app is quitting, or Purge).
    pub fn stop(&self) {
        let mut guard = self.client.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = guard.take()
            && !self.attached
        {
            c.quit();
        }
    }

    pub fn pid(&self) -> Option<u32> {
        self.client.lock().ok().and_then(|g| g.as_ref().and_then(EngineClient::pid))
    }

    /// Start the worker now if none runs (it loads no model until a request asks).
    pub fn start(&self) -> Result<()> {
        self.with_client(|c| c.health()).map(|_| ())
    }

    /// Give the device back (unload every model) if a worker runs; never starts one.
    pub fn purge(&self) -> Result<()> {
        let mut guard = self.client.lock().unwrap_or_else(|e| e.into_inner());
        let Some(c) = guard.as_mut() else { return Ok(()) };
        match c.free() {
            Ok(()) => Ok(()),
            Err(ClientError::Gone(_)) => {
                *guard = None;
                Ok(())
            }
            Err(e) => Err(map_err(e)),
        }
    }

    /// The worker's state without waiting and without starting it: a request holding the
    /// channel answers `starting` or `running` at once; otherwise the live worker is asked.
    pub fn status(&self) -> NativeStatus {
        let mut guard = match self.client.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                let state = match self.busy.load(Ordering::SeqCst) {
                    STARTING => "starting",
                    _ => "running",
                };
                return NativeStatus { state: state.into(), ..NativeStatus::default() };
            }
        };
        let dead = guard.as_mut().is_some_and(|c| !c.alive());
        if dead {
            *guard = None;
        }
        let Some(c) = guard.as_mut() else {
            let start_error = self.start_error.lock().unwrap_or_else(|e| e.into_inner()).clone();
            return match start_error {
                Some(e) => NativeStatus { state: "failed".into(), detail: e, ..NativeStatus::default() },
                None => {
                    NativeStatus { state: "stopped".into(), detail: if dead { "the engine stopped".into() } else { String::new() }, ..NativeStatus::default() }
                }
            };
        };
        match c.health() {
            Ok(h) => NativeStatus {
                state: if h.ok { "ready" } else { "failed" }.into(),
                pid: c.pid(),
                engine: h.engine,
                device: h.device,
                vram_total: h.vram_total,
                vram_free: h.vram_free,
                resident: h.resident,
                detail: h.detail,
            },
            Err(e) => {
                *guard = None;
                NativeStatus { state: "stopped".into(), detail: format!("{STOPPED}: {e}"), ..NativeStatus::default() }
            }
        }
    }
}

fn map_err(e: ClientError) -> Error {
    match e {
        ClientError::Engine { kind: ErrorKind::Cancelled, .. } => Error::Cancelled,
        ClientError::Engine { kind: ErrorKind::BadRequest, message } => Error::Request(message),
        ClientError::Engine { message, .. } => Error::Server(message),
        ClientError::Gone(m) | ClientError::Start(m) => Error::Unavailable(m),
        ClientError::Protocol(m) => Error::Protocol(m),
    }
}

/// A task's wire name (`serde`'s lowercase spelling).
fn task_name(task: Task) -> String {
    serde_json::to_value(task).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| "image".into())
}

/// The request's pixels as blobs (image, mask, references, in that order) and their bytes.
fn blobs_of(req: &Request) -> (Option<Blob>, Option<Blob>, Vec<Blob>, Vec<Vec<u8>>) {
    let mut bytes = Vec::new();
    let mut next = |w: u32, h: u32, c: u8, data: &[u8]| -> Blob {
        let b = Blob { width: w, height: h, channels: c, index: bytes.len() as u32 };
        bytes.push(data.to_vec());
        b
    };
    let image = req.image.as_ref().map(|i| next(i.width, i.height, 4, &i.data));
    let mask = req.mask.as_ref().map(|m| next(m.width, m.height, 1, &m.data));
    let refs = req.references.iter().map(|r| next(r.width, r.height, 4, &r.data)).collect();
    (image, mask, refs, bytes)
}

/// A returned blob as the editor's RGBA (coverage comes back as grey with full alpha, the way
/// detector masks are represented in the editor).
fn picture_of(b: &Blob, bytes: &[u8]) -> Result<Rgba8> {
    match b.channels {
        4 => Rgba8::new(b.width, b.height, bytes.to_vec()),
        1 => {
            let g = Gray8::new(b.width, b.height, bytes.to_vec())?;
            let mut data = Vec::with_capacity(g.data.len() * 4);
            for v in &g.data {
                data.extend_from_slice(&[*v, *v, *v, 255]);
            }
            Rgba8::new(b.width, b.height, data)
        }
        c => Err(Error::Protocol(format!("a picture with {c} channels"))),
    }
}

impl GenerativeBackend for NativeBackend {
    fn health(&self) -> Health {
        let url = format!("{} {}", self.program.display(), self.args.join(" "));
        match self.with_client(|c| c.health()) {
            Ok(h) => Health {
                ok: h.ok,
                backend: format!("native/{}", h.engine),
                url,
                version: String::new(),
                device: h.device,
                vram_total: h.vram_total,
                vram_free: h.vram_free,
                queue_remaining: Some(0),
                error: None,
            },
            Err(e) => Health { ok: false, backend: "native".into(), url, error: Some(e.to_string()), ..Health::default() },
        }
    }

    fn run(&self, req: &Request, progress: &dyn Progress) -> Result<Response> {
        let encode = std::time::Instant::now();
        let tpl = template::find(&req.template)?;
        let (image, mask, references, bytes) = blobs_of(req);
        let (width, height) = req.size.unwrap_or((0, 0));
        let mut params = req.params.clone();
        params.insert("models".into(), serde_json::to_value(tpl.model_bindings(&req.models)?).map_err(|e| Error::Request(e.to_string()))?);
        let rr = RunRequest {
            model: req.template.clone(),
            task: task_name(tpl.meta.task),
            // The template’s prompt wrapper names the mask image for a fill.
            prompt: tpl.format_prompt(&req.prompt),
            negative: req.negative.clone(),
            seed: req.seed,
            steps: if req.steps == 0 { tpl.meta.defaults.steps } else { req.steps },
            guidance: if req.guidance == 0.0 { tpl.meta.defaults.guidance } else { req.guidance },
            width,
            height,
            image,
            mask,
            references,
            params,
        };
        let started = std::time::Instant::now();
        let encode_ms = encode.elapsed().as_millis() as u64;
        let timeout = std::time::Duration::from_millis(self.timeout_ms.load(Ordering::Relaxed));
        let deadline = Deadline { progress, started, timeout, expired: std::cell::Cell::new(false) };
        self.busy.store(RUNNING, Ordering::SeqCst);
        let _busy = Busy(&self.busy);
        let attempt = || {
            self.with_client(|c| {
                let mut report = |f: f32, m: &str| deadline.report(f, m);
                c.run(&rr, &bytes, &mut report, &|| deadline.cancelled())
            })
        };
        // A worker that stopped mid-run (it exits on a poisoned CUDA context) is replaced and
        // the run tried once more on the fresh one (plan-native-engine §12.2); a worker this
        // process only joined cannot be replaced from here.
        let result = retry_after_stop(!self.attached, &deadline, attempt);
        if deadline.expired.get() {
            return Err(Error::Timeout(timeout.as_secs()));
        }
        let (info, pictures) = result?;
        let download = std::time::Instant::now();
        let mut images = Vec::with_capacity(info.images.len());
        for (i, b) in info.images.iter().enumerate() {
            let data = pictures.get(i).ok_or_else(|| Error::Protocol(format!("picture {i} missing")))?;
            images.push(picture_of(b, data)?);
        }
        if images.is_empty() && info.texts.is_empty() {
            return Err(Error::NoOutput(req.template.clone()));
        }
        Ok(Response {
            images,
            texts: info.texts,
            seed: info.seed,
            run_id: format!("native-{}", info.seed),
            elapsed_ms: started.elapsed().as_millis() as u64,
            // IPC read/write is interleaved with worker progress; its residual is observable
            // as elapsed minus run, while download here measures RGBA materialization only.
            timings: Timings { encode_ms, run_ms: info.elapsed_ms, download_ms: download.elapsed().as_millis() as u64, ..Timings::default() },
        })
    }

    fn model_files(&self, folder: &str) -> Result<Vec<String>> {
        // The engine's index: a list under `installed`, or per-folder lists under `folders`.
        let v = self.with_client(|c| c.index(""))?;
        let list = |v: &Value| v.as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>()).unwrap_or_default();
        if let Some(f) = v.get("folders").and_then(|f| f.get(folder)) {
            return Ok(list(f));
        }
        Ok(v.get("installed").map(list).unwrap_or_default())
    }

    fn free(&self) -> Result<()> {
        self.with_client(|c| c.free())
    }
}

/// The worker flags the preferences translate into: `--runtime <dir;dir>` (where the NVIDIA
/// CUDA libraries live) and `--models <dir>` (the native model root, one
/// subfolder per model). Empty preferences give no flags, and the worker serves the fake engine.
pub fn worker_args(runtime_path: &str, models_path: &str) -> Vec<String> {
    let (r, m) = (runtime_path.trim(), models_path.trim());
    if r.is_empty() || m.is_empty() {
        return Vec::new();
    }
    vec!["--runtime".into(), r.to_string(), "--models".into(), m.to_string()]
}

/// The runtime folders of the preference (`;`-separated), as directories for the worker's `PATH`.
pub fn runtime_dirs(runtime_path: &str) -> Vec<PathBuf> {
    runtime_path.split(';').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from).collect()
}

/// The engines a worker with these flags serves: the perception engine (SAM 3.1 on candle)
/// with `--runtime` and `--models`, the diffusion engine (Qwen-Image-2.1, with the
/// `native-diffusion` feature) with `--models`. Empty without a model root.
fn engines_for(args: &[String]) -> Vec<Box<dyn gn_engine::Engine>> {
    let mut engines: Vec<Box<dyn gn_engine::Engine>> = Vec::new();
    // The diffusion engine first: the worker's health names the device of the first engine
    // that names one, and this one knows the card's name (and refuses a card it has no
    // kernels for, with a sentence in that health).
    #[cfg(feature = "native-diffusion")]
    {
        let missing = missing_cuda_libraries(&search_dirs());
        if missing.is_empty() {
            if let Some(d) = gn_diffusion::from_args(args) {
                engines.push(Box::new(d));
            }
        } else if args.iter().any(|a| a == "--models") {
            let why = format!("The native diffusion models are off: {} not found in the runtime folder or on PATH.", missing.join(", "));
            eprintln!("--engine: {why}");
            engines.push(Box::new(Refused(why)));
        }
    }
    if let Some(p) = gn_perception::engine::from_args(args) {
        engines.push(Box::new(p));
    }
    engines
}

/// The CUDA libraries a `native-diffusion` build loads on first use in the worker (the apps'
/// build scripts delay-load them); Windows names, CUDA 13. Elsewhere the dynamic loader finds
/// them and nothing is checked.
#[cfg(feature = "native-diffusion")]
const CUDA_LIBRARIES: &[&str] = if cfg!(windows) { &["cublas64_13.dll", "cublasLt64_13.dll", "curand64_10.dll"] } else { &[] };

/// Where Windows looks for a delay-loaded DLL that matters here: the executable's folder, then
/// `PATH` (which holds the runtime folders first, [`NativeBackend::with_path_dirs`]).
#[cfg(feature = "native-diffusion")]
fn search_dirs() -> Vec<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(PathBuf::from));
    exe_dir.into_iter().chain(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())).collect()
}

/// The diffusion engine's place when its runtime is missing: it serves nothing and its health
/// carries the reason, so the editor’s engine page can report it.
#[cfg(feature = "native-diffusion")]
struct Refused(String);

#[cfg(feature = "native-diffusion")]
impl gn_engine::Engine for Refused {
    fn name(&self) -> String {
        "qir".into()
    }
    fn health(&mut self) -> gn_engine::Health {
        gn_engine::Health { engine: "qir".into(), detail: self.0.clone(), ..gn_engine::Health::default() }
    }
    fn run(
        &mut self,
        _req: &RunRequest,
        _blobs: &[Vec<u8>],
        _progress: &mut dyn FnMut(f32, &str),
        _cancel: &std::sync::atomic::AtomicBool,
    ) -> std::result::Result<(gn_engine::DoneInfo, Vec<Vec<u8>>), gn_engine::EngineError> {
        Err(gn_engine::EngineError::new(ErrorKind::Unsupported, self.0.clone()))
    }
    fn free(&mut self) {}
    fn index(&mut self, dir: &str) -> Value {
        serde_json::json!({"dir": dir, "installed": []})
    }
}

/// The [`CUDA_LIBRARIES`] none of `dirs` holds.
#[cfg(feature = "native-diffusion")]
fn missing_cuda_libraries(dirs: &[PathBuf]) -> Vec<&'static str> {
    CUDA_LIBRARIES.iter().copied().filter(|dll| !dirs.iter().any(|d| d.join(dll).is_file())).collect()
}

/// `photocraft --engine [--token T] [--step-ms N] [--runtime <dir;dir> --models <dir>] [--cpu]
/// [--stay]`: this process becomes the worker. Reads the token from `GN_ENGINE_TOKEN` when not
/// given, binds a loopback port, prints the port line, serves until the editor says quit (or
/// goes away; with `--stay` it outlives its clients, for a worker started by hand that many
/// CLI processes join through `PHOTOCRAFT_NATIVE_ENGINE=port:token`). Returns the exit code.
/// With `--runtime` and `--models` it serves the perception engine (SAM 3.1 on candle)
/// and, when built with it, the diffusion engine; otherwise the fake engine.
pub fn engine_main(args: &[String]) -> i32 {
    // The engines report through `tracing`; `RUST_LOG` (env_logger directives, `info` for the
    // phase timings) turns the worker's stderr into their log. Silent otherwise.
    if let Ok(spec) = std::env::var("RUST_LOG")
        && !spec.trim().is_empty()
        && let Ok(filter) = tracing_subscriber::EnvFilter::try_new(spec.trim())
    {
        let _ = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).with_ansi(false).try_init();
    }
    let mut token = std::env::var("GN_ENGINE_TOKEN").unwrap_or_default();
    let mut step_ms = 5u64;
    let mut stay = false;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--stay" {
            stay = true;
            i += 1;
            continue;
        }
        match args.get(i).map(String::as_str) {
            Some("--token") => {
                token = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            Some("--step-ms") => {
                step_ms = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(5);
                i += 2;
            }
            _ => i += 1,
        }
    }
    if token.is_empty() {
        eprintln!("--engine: no token (GN_ENGINE_TOKEN or --token)");
        return 2;
    }
    let listener = match gn_engine::server::bind() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("--engine: cannot bind: {e}");
            return 1;
        }
    };
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    println!("{}", gn_engine::server::port_line(port));
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let engines = engines_for(args);
    let served = if engines.is_empty() {
        let mut engine =
            gn_engine::fake::FakeEngine { step_ms, installed: template::builtin().iter().map(|t| t.meta.id.clone()).collect(), ..Default::default() };
        gn_engine::server::serve_with(&listener, &token, &mut engine, stay)
    } else {
        let mut engine = gn_engine::Multi::new(engines);
        gn_engine::server::serve_with(&listener, &token, &mut engine, stay)
    };
    match served {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("--engine: {e}");
            1
        }
    }
}

/// Tests (this crate's and the engine's): a fake engine served on a thread in this process, as
/// the worker would be, knowing `installed` templates; its port and token.
#[cfg(feature = "fake-server")]
pub fn serve_fake(installed: Vec<String>, step_ms: u64) -> Result<(u16, String)> {
    let listener = gn_engine::server::bind().map_err(|e| Error::Unavailable(e.to_string()))?;
    let port = listener.local_addr().map_err(|e| Error::Unavailable(e.to_string()))?.port();
    let token = gn_engine::client::new_token();
    let t = token.clone();
    std::thread::spawn(move || {
        let mut engine = gn_engine::fake::FakeEngine { step_ms, installed, ..Default::default() };
        let _ = gn_engine::server::serve(&listener, &t, &mut engine);
    });
    Ok((port, token))
}

/// A fixture worker registered by the `fake-server` feature. Production builds have none.
#[doc(hidden)]
pub fn test_backend(key: &str) -> Result<Option<std::sync::Arc<NativeBackend>>> {
    #[cfg(feature = "fake-server")]
    {
        if let Some(b) = crate::fake::backend(key) {
            return Ok(Some(b));
        }
        if key.starts_with("fake-native:") || key.starts_with("http://") {
            return Err(Error::Unavailable(format!("native test worker at {key} is not available")));
        }
    }
    let _ = key;
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn served() -> (u16, String) {
        serve_fake(vec!["krea2-turbo/image".into(), "qwen-edit-2511/edit".into(), "sam3.1/segment".into()], 1).unwrap()
    }

    struct Cancel(AtomicBool, Mutex<Vec<(f32, String)>>);
    impl Progress for Cancel {
        fn report(&self, fraction: f32, message: &str) {
            self.1.lock().unwrap().push((fraction, message.to_string()));
        }
        fn cancelled(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }

    #[cfg(all(feature = "native-diffusion", windows))]
    #[test]
    fn the_worker_names_the_cuda_libraries_it_cannot_find() {
        let dir = std::env::temp_dir().join(format!("pc-cuda-libs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(missing_cuda_libraries(std::slice::from_ref(&dir)), CUDA_LIBRARIES.to_vec());
        for dll in CUDA_LIBRARIES {
            std::fs::write(dir.join(dll), b"").unwrap();
        }
        assert!(missing_cuda_libraries(&[PathBuf::from("Z:/nowhere"), dir.clone()]).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn requests_cross_the_channel_and_come_back_as_pictures() {
        let (port, token) = served();
        let b = NativeBackend::attached(port, &token).unwrap();
        let h = b.health();
        assert!(h.ok && h.backend == "native/fake", "{h:?}");
        // Text to image: the fake paints a solid picture of the asked size.
        let mut req = Request::new("krea2-turbo/image", "a fox");
        req.size = Some((48, 32));
        req.steps = 4;
        let p = Cancel(AtomicBool::new(false), Mutex::new(Vec::new()));
        let r = b.run(&req, &p).unwrap();
        assert_eq!((r.images.len(), r.images[0].width, r.images[0].height), (1, 48, 32));
        assert!(p.1.lock().unwrap().len() >= 4, "progress reports arrived");
        // An edit sends the picture across and gets the same size back.
        let mut edit = Request::new("qwen-edit-2511/edit", "make it winter");
        edit.image = Some(Rgba8::solid(20, 10, [10, 20, 30, 255]).unwrap());
        let r = b.run(&edit, &crate::NoProgress).unwrap();
        assert_eq!((r.images[0].width, r.images[0].height), (20, 10));
        assert_ne!(r.images[0].data, edit.image.as_ref().unwrap().data);
        // A detector's coverage comes back as a grey picture.
        let mut seg = Request::new("sam3.1/segment", "fox");
        seg.image = Some(Rgba8::solid(40, 40, [0, 0, 0, 255]).unwrap());
        let r = b.run(&seg, &crate::NoProgress).unwrap();
        let centre = r.images[0].get(20, 20).unwrap();
        let corner = r.images[0].get(0, 0).unwrap();
        assert_eq!((centre, corner), ([255, 255, 255, 255], [0, 0, 0, 255]));
        // A template the engine lacks is a server error, not a dead backend.
        let missing = Request::new("qwen-2.1/image", "x");
        assert!(matches!(b.run(&missing, &crate::NoProgress), Err(Error::Server(_))));
        assert!(b.health().ok);
        // Cancel reaches the run.
        let mut long = Request::new("krea2-turbo/image", "slow");
        long.size = Some((8, 8));
        long.steps = 1000;
        let c = Arc::new(Cancel(AtomicBool::new(false), Mutex::new(Vec::new())));
        let c2 = Arc::clone(&c);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            c2.0.store(true, Ordering::SeqCst);
        });
        assert!(matches!(b.run(&long, &*c), Err(Error::Cancelled)));
        assert_eq!(b.model_files("diffusion_models").unwrap().len(), 3);
        b.free().unwrap();
    }

    /// The status answers without starting a worker: stopped before any, failed with the reason
    /// after a start that could not happen, ready with what is resident once one runs; Purge
    /// never starts one.
    #[test]
    fn the_status_never_starts_a_worker_and_says_why_one_failed() {
        let b = NativeBackend::new(PathBuf::from("Z:/nowhere/photocraft.exe"), vec![ENGINE_FLAG.into()]);
        assert_eq!(b.status().state, "stopped");
        b.purge().unwrap();
        assert_eq!(b.status().state, "stopped", "purge started nothing");
        assert!(b.start().is_err());
        let st = b.status();
        assert!(st.state == "failed" && st.detail.contains("nowhere"), "{st:?}");
        let (port, token) = served();
        let a = NativeBackend::attached(port, &token).unwrap();
        let st = a.status();
        assert!(st.state == "ready" && st.engine == "fake" && st.resident.is_empty(), "{st:?}");
        let mut req = Request::new("krea2-turbo/image", "a fox");
        req.size = Some((8, 8));
        a.run(&req, &crate::NoProgress).unwrap();
        assert_eq!(a.status().resident, vec!["krea2-turbo/image".to_string()]);
        a.purge().unwrap();
        assert!(a.status().resident.is_empty());
        // A request holding the channel: the status answers at once.
        a.busy.store(RUNNING, Ordering::SeqCst);
        let held = a.client.lock().unwrap();
        assert_eq!(a.status().state, "running");
        drop(held);
        a.busy.store(IDLE, Ordering::SeqCst);
        // The log turns the worker's tracing on.
        let logged = NativeBackend::new(PathBuf::from("x"), Vec::new()).with_log(std::env::temp_dir().join("w.log"));
        if std::env::var_os("RUST_LOG").is_none() {
            assert!(logged.env().iter().any(|(k, v)| k == "RUST_LOG" && v == "info"));
        }
    }

    #[test]
    fn a_run_that_lost_its_worker_is_tried_once_more_on_a_fresh_one() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        let stopped_once = || {
            calls.set(calls.get() + 1);
            if calls.get() == 1 { Err(Error::Unavailable(format!("{STOPPED}: closed"))) } else { Ok(7) }
        };
        let p = Cancel(AtomicBool::new(false), Mutex::new(Vec::new()));
        assert_eq!(retry_after_stop(true, &p, stopped_once).unwrap(), 7);
        assert_eq!(calls.get(), 2);
        assert!(p.1.lock().unwrap().iter().any(|(_, m)| m.contains("trying again")));
        // A worker that cannot be replaced, a cancelled run, or another error: no second try.
        calls.set(0);
        assert!(retry_after_stop(false, &p, stopped_once).is_err());
        assert_eq!(calls.get(), 1);
        calls.set(0);
        p.0.store(true, Ordering::SeqCst);
        assert!(retry_after_stop(true, &p, stopped_once).is_err());
        assert_eq!(calls.get(), 1);
        let other = || -> Result<i32> { Err(Error::Server("x".into())) };
        p.0.store(false, Ordering::SeqCst);
        assert!(matches!(retry_after_stop(true, &p, other), Err(Error::Server(_))));
    }

    #[test]
    fn the_worker_flags_and_path_follow_the_preferences() {
        assert!(worker_args("", "m").is_empty() && worker_args("r", " ").is_empty());
        assert_eq!(worker_args(" a;b ", "m"), ["--runtime", "a;b", "--models", "m"]);
        assert_eq!(runtime_dirs("a; b ;;c"), [PathBuf::from("a"), PathBuf::from("b"), PathBuf::from("c")]);
        assert!(runtime_dirs("").is_empty());
        // Without a model root the worker serves the fake engine (no real engine is made).
        assert!(engines_for(&["--runtime".into(), "a".into()]).is_empty());
        let b = NativeBackend::new(PathBuf::from("x"), vec![ENGINE_FLAG.into()]).with_path_dirs(runtime_dirs("a;b"));
        let env = b.env();
        assert_eq!(env.len(), 1);
        assert!(env[0].0 == "PATH" && env[0].1.starts_with("a") && env[0].1.contains('b'), "{env:?}");
        assert!(NativeBackend::new(PathBuf::from("x"), Vec::new()).env().is_empty());
    }
}
