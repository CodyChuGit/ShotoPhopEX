//! The Generations library: every generative job the session runs, recorded as a
//! [`GenerationRecord`] that survives a restart, with its result pixels and a thumbnail on disk.
//!
//! The record is made by the command dispatcher, never by the commands themselves: a command
//! that ran inline (`Session::execute`: the CLI, tests, MCP with `wait`) is recorded when it
//! returns; one that started a background job is recorded when the job starts and completed in
//! `end_job`. The result's layers (the `layer` / `layers` / `document` fields every generative
//! command returns) are rendered with their masks into PNGs, on a writer thread for background
//! jobs and inline otherwise, so the library keeps what was made even after the document closes.
//!
//! On disk (`set_generations_dir`, the app passes `<config dir>/generations`):
//!
//! ```text
//! index.json            {"version": 1, "records": [GenerationRecord…]}
//! thumbs/<id>.png       the first result, at most 128 px on its longer side
//! results/<id>-<i>.png  every result layer, RGBA with its mask as alpha
//! ```
//!
//! Record ids are unique across processes (unix microseconds plus the process id, bumped past
//! every id already known), because job ids restart in every process. The index is written on a
//! one-second debounce from `Session::poll_jobs`, atomically, and merged with the file when
//! another process (the CLI) changed it meanwhile. At most [`MAX_RESULTS`] completed results or
//! [`MAX_BYTES`] of files are kept, oldest unpinned first; failed and cancelled rows go after 30
//! days. Without a directory (tests, the web) the library lives in memory only.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use photocraft_doc::{DocId, Document, LayerId};
use photocraft_genai::Rgba8;
use photocraft_geom::Rect;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::generate_cmds::{GenerativeInfo, bad, generative_info, opt_bool, opt_num, opt_str, parse_seed, short, web_unavailable};
use crate::jobs::{JobId, JobOutcome};
use crate::{EngineError, Result, Session};

pub const JOBS: &str = "generate.jobs";
pub const DISMISS: &str = "generate.jobs.dismiss";
pub const PIN: &str = "generate.jobs.pin";
pub const CLEAR: &str = "generate.jobs.clear";
pub const RECREATE: &str = "generate.recreate";
pub const PLACE: &str = "generate.place";
pub const EXPORT: &str = "generate.export";

/// Completed results kept before the oldest unpinned ones are dropped.
pub const MAX_RESULTS: usize = 500;
/// Bytes of result files kept before the oldest unpinned results are dropped.
pub const MAX_BYTES: u64 = 5 * 1024 * 1024 * 1024;
/// Failed and cancelled rows are kept this long.
const FAILED_KEEP_MS: u64 = 30 * 24 * 3600 * 1000;
/// A row still running after this long with no progress counts as stale.
const STALE_MS: u64 = 10 * 60 * 1000;
/// The index is written at most this often while things change.
const FLUSH_INTERVAL_MS: f64 = 1000.0;
/// Thumbnails fit in this square.
pub const THUMB_SIDE: u32 = 128;
/// Larger result layers are not stored (the commands cap requests at 16 MP anyway).
const MAX_CAPTURE_PIXELS: u64 = 16 * 1024 * 1024;
const INDEX_VERSION: u32 = 1;
/// The default file-name pattern of `generate.export`.
pub const DEFAULT_EXPORT_NAME: &str = "{model}_{date}_{seed}";

/// Commands the library records.
pub fn is_generative(id: &str) -> bool {
    matches!(
        id,
        "generate.fill"
            | "generate.image"
            | "generate.expand"
            | "generate.edit"
            | "generate.similar"
            | "generate.angles"
            | "generate.splitLayers"
            | "generate.removeBackground"
            | "select.byText"
            | "select.byPoint"
            | "select.subjectML"
            | RECREATE
    )
}

/// Where a generation stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GenStatus {
    #[default]
    Running,
    Done,
    Failed,
    Cancelled,
}

impl GenStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            GenStatus::Running => "running",
            GenStatus::Done => "done",
            GenStatus::Failed => "failed",
            GenStatus::Cancelled => "cancelled",
        }
    }
}

/// Why a generation failed, for a label the user can act on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GenFailureKind {
    ServerUnreachable,
    ModelFilesMissing,
    OutOfMemory,
    ServerError,
    Timeout,
    BadInput,
    /// The document changed or closed while it ran; the result was discarded.
    Discarded,
    /// The app closed while it ran.
    Interrupted,
    #[default]
    Other,
}

impl GenFailureKind {
    /// The English label (the UI translates it).
    pub fn label(self) -> &'static str {
        match self {
            GenFailureKind::ServerUnreachable => "Server unreachable",
            GenFailureKind::ModelFilesMissing => "Model files missing",
            GenFailureKind::OutOfMemory => "Out of memory on the server",
            GenFailureKind::ServerError => "Server error",
            GenFailureKind::Timeout => "Timed out",
            GenFailureKind::BadInput => "Invalid request",
            GenFailureKind::Discarded => "Result discarded",
            GenFailureKind::Interrupted => "Interrupted",
            GenFailureKind::Other => "Failed",
        }
    }
}

/// Sort a failure message into a [`GenFailureKind`] (the messages come from `photocraft-genai`
/// and the generate commands; matching is case-insensitive on stable phrases).
pub fn classify_failure(message: &str) -> GenFailureKind {
    let m = message.to_ascii_lowercase();
    let has = |s: &str| m.contains(s);
    if has("out of memory") || has("outofmemory") || has("cuda error: out") {
        GenFailureKind::OutOfMemory
    } else if has("update comfyui") {
        GenFailureKind::ServerError
    } else if has("cannot reach") || has("not available:") || has("connection refused") || has("no generative server") {
        GenFailureKind::ServerUnreachable
    } else if (has("missing") && (has("model") || has("file"))) || has("not installed") || has("value not in list") {
        GenFailureKind::ModelFilesMissing
    } else if has("did not finish within") || has("timed out") {
        GenFailureKind::Timeout
    } else if has("changed while it ran") || has("was closed") {
        GenFailureKind::Discarded
    } else if has("invalid parameters") || has("invalid generative request") {
        GenFailureKind::BadInput
    } else if has("server reported an error") || has("answered unexpectedly") || has("produced no output") {
        GenFailureKind::ServerError
    } else {
        GenFailureKind::Other
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenFailure {
    pub kind: GenFailureKind,
    pub message: String,
}

/// What a finished generation made.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenResult {
    /// Layer ids in the document it ran on (meaningful while that document is open).
    pub layers: Vec<u64>,
    pub width: u32,
    pub height: u32,
    /// Server time reported by the command.
    pub ms: u64,
    /// Result PNGs relative to the library directory, one per result layer.
    pub files: Vec<String>,
    /// The thumbnail PNG relative to the library directory ("" = none).
    pub thumbnail: String,
    pub bytes: u64,
    /// The files are on disk (false while the writer thread works).
    pub saved: bool,
    /// The result is a selection (`select.*`), not pixels.
    pub selection: bool,
}

/// One generation in the library.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GenerationRecord {
    pub id: u64,
    pub created_ms: u64,
    pub finished_ms: u64,
    pub command: String,
    pub label: String,
    /// The prompt as typed.
    pub prompt: String,
    /// The prompt the model got when the enhancer rewrote it.
    pub enhanced: String,
    pub template: String,
    pub seed: u64,
    pub count: u32,
    /// The command's params, for Recreate.
    pub params: Value,
    /// The document's name at the time.
    pub document: String,
    pub status: GenStatus,
    pub progress: f32,
    pub message: String,
    pub failure: Option<GenFailure>,
    pub result: Option<GenResult>,
    /// What the first result layer remembers (`generate.info`).
    pub info: Option<GenerativeInfo>,
    pub dismissed: bool,
    pub pinned: bool,
    /// The document it ran on (`DocId`), in this process only.
    #[serde(skip)]
    pub doc: Option<u64>,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Index {
    version: u32,
    records: Vec<GenerationRecord>,
}

/// Pixels to store for a finished generation: the document as it was and the layers to render.
pub(crate) struct Capture {
    doc: Arc<Document>,
    items: Vec<(LayerId, Rect)>,
}

/// What [`Session::capture_result`] found about a result.
struct Found {
    doc: Option<u64>,
    title: String,
    layers: Vec<u64>,
    info: Option<GenerativeInfo>,
    capture: Option<Capture>,
}

/// A writer thread's report.
struct WriteDone {
    id: u64,
    bytes: u64,
    error: Option<String>,
}

/// Limits of the library (see the module docs).
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub max_results: usize,
    pub max_bytes: u64,
}

impl Default for Caps {
    fn default() -> Self {
        Caps { max_results: MAX_RESULTS, max_bytes: MAX_BYTES }
    }
}

/// The library (see the module docs). `Session::generations`.
#[derive(Default)]
pub struct GenerationStore {
    dir: Option<PathBuf>,
    /// Oldest first.
    records: Vec<GenerationRecord>,
    /// Running job → record.
    by_job: BTreeMap<u64, u64>,
    /// The last jobs that ended (job, record), so the UI can match a `JobEvent` to its row.
    ended: std::collections::VecDeque<(u64, u64)>,
    /// Records this process removed (not resurrected by a merge with an older index).
    removed: BTreeSet<u64>,
    dirty: bool,
    last_flush_ms: f64,
    /// Modification time and length of the index as last read or written.
    stamp: Option<(std::time::SystemTime, u64)>,
    writes: Arc<Mutex<Vec<WriteDone>>>,
    #[cfg(not(target_arch = "wasm32"))]
    writers: Vec<std::thread::JoinHandle<()>>,
    /// `generate.recreate` names the command it really runs, so the row describes that.
    override_next: Option<(String, Value, String)>,
    last_id: u64,
    caps: Caps,
}

impl Drop for GenerationStore {
    fn drop(&mut self) {
        // Writers are short (a few PNG encodes): finish them so the CLI's last result lands.
        #[cfg(not(target_arch = "wasm32"))]
        for h in self.writers.drain(..) {
            let _ = h.join();
        }
        self.drain_writes();
        if self.dirty {
            self.flush();
        }
    }
}

impl std::fmt::Debug for GenerationStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationStore").field("dir", &self.dir).field("records", &self.records.len()).finish()
    }
}

/// Wall-clock milliseconds since the Unix epoch (0 on the web).
pub fn unix_ms() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
}

impl GenerationStore {
    /// The library directory, if one is attached.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Every record, oldest first.
    pub fn records(&self) -> &[GenerationRecord] {
        &self.records
    }

    pub fn get(&self, id: u64) -> Option<&GenerationRecord> {
        self.records.iter().find(|r| r.id == id)
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut GenerationRecord> {
        self.records.iter_mut().find(|r| r.id == id)
    }

    /// The record of job `job` (running, or among the last that ended), if it is a generation.
    pub fn record_for_job(&self, job: JobId) -> Option<u64> {
        self.by_job.get(&job.0).copied().or_else(|| self.ended.iter().rev().find(|(j, _)| *j == job.0).map(|(_, r)| *r))
    }

    /// The record of a job that is still running.
    fn running_record(&self, job: JobId) -> Option<u64> {
        self.by_job.get(&job.0).copied()
    }

    /// Change the limits (tests; a future preference).
    pub fn set_caps(&mut self, caps: Caps) {
        self.caps = caps;
        self.apply_caps();
    }

    /// Full path of a record's result file `index`.
    pub fn result_path(&self, id: u64, index: usize) -> Option<PathBuf> {
        let r = self.get(id)?.result.as_ref()?;
        Some(self.dir.as_ref()?.join(r.files.get(index)?))
    }

    /// Full path of a record's thumbnail.
    pub fn thumbnail_path(&self, id: u64) -> Option<PathBuf> {
        let r = self.get(id)?.result.as_ref().filter(|r| !r.thumbnail.is_empty())?;
        Some(self.dir.as_ref()?.join(&r.thumbnail))
    }

    /// The record whose result includes layer `layer` of document `doc` (this process).
    pub fn record_for_layer(&self, doc: DocId, layer: LayerId) -> Option<u64> {
        self.records.iter().rev().find(|r| r.doc == Some(doc.0) && r.result.as_ref().is_some_and(|x| x.layers.contains(&layer.0))).map(|r| r.id)
    }

    fn new_id(&mut self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        let pid = u64::from(std::process::id() % 1000);
        #[cfg(target_arch = "wasm32")]
        let pid = 0;
        let base = unix_ms().saturating_mul(1000).saturating_add(pid);
        let known = self.records.iter().map(|r| r.id).max().unwrap_or(0).max(self.last_id);
        let id = base.max(known.saturating_add(1));
        self.last_id = id;
        id
    }

    /// The running job of row `id` (to cancel it).
    pub fn job_for_record(&self, id: u64) -> Option<JobId> {
        self.by_job.iter().find(|(_, r)| **r == id).map(|(j, _)| JobId(*j))
    }

    /// Changes wait for the debounced index write.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Rows still running.
    pub fn running(&self) -> usize {
        self.records.iter().filter(|r| r.status == GenStatus::Running).count()
    }

    fn take_override(&mut self, command: &str, params: &Value, label: &str) -> (String, Value, String) {
        match (command == RECREATE, self.override_next.take()) {
            (true, Some(o)) => o,
            _ => (command.to_string(), params.clone(), label.to_string()),
        }
    }

    fn begin(&mut self, command: &str, label: &str, params: &Value, doc: Option<u64>, title: String) -> u64 {
        let id = self.new_id();
        let prompt = params.get("prompt").and_then(Value::as_str).unwrap_or_default().trim().to_string();
        let count = params.get("variations").and_then(Value::as_u64).map_or(1, |n| n.clamp(1, 64) as u32);
        let rec = GenerationRecord {
            id,
            created_ms: unix_ms(),
            command: command.to_string(),
            label: label.to_string(),
            prompt,
            template: params.get("template").and_then(Value::as_str).unwrap_or_default().to_string(),
            seed: params.get("seed").and_then(Value::as_u64).unwrap_or(0),
            count,
            params: params.clone(),
            document: title,
            status: GenStatus::Running,
            doc,
            ..GenerationRecord::default()
        };
        self.records.push(rec);
        self.dirty = true;
        id
    }

    fn finish_failed(&mut self, id: u64, message: &str) {
        if let Some(r) = self.get_mut(id) {
            r.status = GenStatus::Failed;
            r.finished_ms = unix_ms();
            r.failure = Some(GenFailure { kind: classify_failure(message), message: message.to_string() });
            r.message = message.to_string();
        }
        self.dirty = true;
    }

    fn finish_cancelled(&mut self, id: u64) {
        if let Some(r) = self.get_mut(id) {
            r.status = GenStatus::Cancelled;
            r.finished_ms = unix_ms();
            r.message = "Cancelled".into();
        }
        self.dirty = true;
    }

    fn finish_done(&mut self, id: u64, value: &Value, found: Found, background: bool) {
        let dir = self.dir.clone();
        let Some(r) = self.get_mut(id) else { return };
        r.status = GenStatus::Done;
        r.finished_ms = unix_ms();
        r.progress = 1.0;
        r.message.clear();
        if let Some(t) = value.get("template").and_then(Value::as_str) {
            r.template = t.to_string();
        }
        if let Some(seed) = value.get("seed").and_then(Value::as_u64) {
            r.seed = seed;
        }
        if r.doc.is_none() {
            r.doc = found.doc;
        }
        if r.document.is_empty() {
            r.document = found.title.clone();
        }
        if let Some(info) = &found.info {
            if r.prompt.is_empty() {
                r.prompt = info.prompt.clone();
            }
            if r.template.is_empty() {
                r.template = info.template.clone();
            }
            r.enhanced = info.enhanced.clone();
        }
        let files: Vec<String> = found.capture.as_ref().map_or_else(Vec::new, |c| (0..c.items.len()).map(|i| format!("results/{id}-{i}.png")).collect());
        if !found.layers.is_empty() {
            r.count = found.layers.len().min(u32::MAX as usize) as u32;
        }
        let thumbnail = if files.is_empty() || dir.is_none() { String::new() } else { format!("thumbs/{id}.png") };
        r.result = Some(GenResult {
            layers: found.layers.clone(),
            width: value.get("width").and_then(Value::as_u64).map_or(0, |v| v.min(u64::from(u32::MAX)) as u32),
            height: value.get("height").and_then(Value::as_u64).map_or(0, |v| v.min(u64::from(u32::MAX)) as u32),
            ms: value.get("ms").and_then(Value::as_u64).unwrap_or(0),
            files: if dir.is_some() { files.clone() } else { Vec::new() },
            thumbnail: thumbnail.clone(),
            bytes: 0,
            saved: dir.is_none() || files.is_empty(),
            selection: found.layers.is_empty() && value.get("selected").is_some(),
        });
        r.info = found.info;
        self.dirty = true;
        let (Some(dir), Some(capture)) = (dir, found.capture) else { return };
        if files.is_empty() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if background {
            let queue = self.writes.clone();
            let spawned = std::thread::Builder::new().name("photocraft-generations".into()).spawn(move || {
                let done = write_capture(&dir, id, &capture, &files, &thumbnail);
                queue.lock().unwrap_or_else(PoisonError::into_inner).push(done);
            });
            match spawned {
                Ok(h) => self.writers.push(h),
                Err(e) => {
                    if let Some(r) = self.get_mut(id) {
                        r.message = format!("the result could not be saved: {e}");
                    }
                }
            }
            return;
        }
        let _ = background;
        let done = write_capture(&dir, id, &capture, &files, &thumbnail);
        self.apply_write(done);
    }

    fn apply_write(&mut self, done: WriteDone) {
        if let Some(r) = self.get_mut(done.id) {
            if let Some(res) = r.result.as_mut() {
                res.saved = true;
                res.bytes = done.bytes;
            }
            if let Some(e) = done.error {
                r.message = e;
            }
        }
        self.dirty = true;
        self.apply_caps();
    }

    fn drain_writes(&mut self) {
        let done: Vec<WriteDone> = std::mem::take(&mut *self.writes.lock().unwrap_or_else(PoisonError::into_inner));
        for d in done {
            self.apply_write(d);
        }
    }

    /// Copy a running job's progress into its row (cheap; called every frame while jobs run).
    pub(crate) fn update_progress(&mut self, job: JobId, progress: f32, message: &str) {
        let Some(id) = self.running_record(job) else { return };
        if let Some(r) = self.get_mut(id) {
            r.progress = progress;
            if !message.is_empty() && r.message != message {
                message.clone_into(&mut r.message);
            }
        }
    }

    /// Collect finished writes and write the index when it is due. Cheap when idle.
    pub fn poll(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.writers.retain(|h| !h.is_finished());
        self.drain_writes();
        if self.dirty && crate::jobs::now_ms() - self.last_flush_ms >= FLUSH_INTERVAL_MS {
            self.flush();
        }
    }

    /// Write the index now (merging another process's changes first).
    pub fn flush(&mut self) {
        self.last_flush_ms = crate::jobs::now_ms();
        let Some(dir) = self.dir.clone() else {
            self.dirty = false;
            return;
        };
        let path = dir.join("index.json");
        if self.stamp != stamp_of(&path) {
            self.merge_from(&path, false);
        }
        self.apply_caps();
        let index = Index { version: INDEX_VERSION, records: self.records.clone() };
        let Ok(bytes) = serde_json::to_vec_pretty(&index) else { return };
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        if photocraft_format::atomic::atomic_write(&path, &bytes).is_ok() {
            self.stamp = stamp_of(&path);
            self.dirty = false;
        }
    }

    /// Re-read the index when another process changed it.
    pub fn refresh(&mut self) {
        let Some(path) = self.dir.as_ref().map(|d| d.join("index.json")) else { return };
        if self.stamp != stamp_of(&path) {
            self.merge_from(&path, false);
        }
    }

    /// Merge the index at `path` into memory: records we don't know are added (unless this
    /// process removed them); ours win. `startup`: rows left running by a process that is gone
    /// become "interrupted".
    fn merge_from(&mut self, path: &Path, startup: bool) {
        self.stamp = stamp_of(path);
        let Ok(text) = std::fs::read(path) else { return };
        let Ok(index) = serde_json::from_slice::<Index>(&text) else { return };
        let known: BTreeSet<u64> = self.records.iter().map(|r| r.id).collect();
        for mut r in index.records {
            if known.contains(&r.id) || self.removed.contains(&r.id) || r.id == 0 {
                continue;
            }
            if startup && r.status == GenStatus::Running {
                r.status = GenStatus::Failed;
                r.failure = Some(GenFailure { kind: GenFailureKind::Interrupted, message: "PhotoCraft closed while it ran".into() });
                self.dirty = true;
            }
            r.doc = None;
            self.records.push(r);
        }
        self.records.sort_by_key(|r| (r.created_ms, r.id));
        self.last_id = self.last_id.max(self.records.iter().map(|r| r.id).max().unwrap_or(0));
    }

    /// Attach the library directory: read its index (rows a closed process left running become
    /// "interrupted"). Nothing is created until there is something to write.
    pub fn set_dir(&mut self, dir: PathBuf) {
        let path = dir.join("index.json");
        self.dir = Some(dir);
        self.merge_from(&path, true);
        self.apply_caps();
    }

    fn remove(&mut self, id: u64) {
        let Some(i) = self.records.iter().position(|r| r.id == id) else { return };
        let r = self.records.remove(i);
        self.removed.insert(id);
        self.dirty = true;
        if let (Some(dir), Some(res)) = (&self.dir, &r.result) {
            for f in res.files.iter().chain(std::iter::once(&res.thumbnail)).filter(|f| !f.is_empty()) {
                let _ = std::fs::remove_file(dir.join(f));
            }
        }
    }

    /// Drop what is over the limits: old failures, then the oldest unpinned results.
    fn apply_caps(&mut self) {
        let now = unix_ms();
        let old: Vec<u64> = self
            .records
            .iter()
            .filter(|r| matches!(r.status, GenStatus::Failed | GenStatus::Cancelled) && !r.pinned && now.saturating_sub(r.created_ms) > FAILED_KEEP_MS)
            .map(|r| r.id)
            .collect();
        for id in old {
            self.remove(id);
        }
        let done = |r: &GenerationRecord| r.status == GenStatus::Done;
        let mut count = self.records.iter().filter(|r| done(r)).count();
        let mut bytes: u64 = self.records.iter().filter(|r| done(r)).filter_map(|r| r.result.as_ref()).map(|x| x.bytes).sum();
        let mut candidates: Vec<(u64, u64, u64)> =
            self.records.iter().filter(|r| done(r) && !r.pinned).map(|r| (r.created_ms, r.id, r.result.as_ref().map_or(0, |x| x.bytes))).collect();
        candidates.sort_unstable();
        let mut next = candidates.into_iter();
        while count > self.caps.max_results || bytes > self.caps.max_bytes {
            let Some((_, id, b)) = next.next() else { break };
            self.remove(id);
            count = count.saturating_sub(1);
            bytes = bytes.saturating_sub(b);
        }
    }
}

fn stamp_of(path: &Path) -> Option<(std::time::SystemTime, u64)> {
    let m = std::fs::metadata(path).ok()?;
    Some((m.modified().ok()?, m.len()))
}

/// Render a layer over `rect` as straight RGBA8, its enabled mask (with density) folded into
/// the alpha. `None` for a layer without pixels, an empty or an over-large rectangle.
pub(crate) fn render_layer(doc: &Document, id: LayerId, rect: Rect) -> Option<Rgba8> {
    let layer = doc.layer(id)?;
    let surf = layer.surface()?;
    if rect.is_empty() || u64::from(rect.width()) * u64::from(rect.height()) > MAX_CAPTURE_PIXELS {
        return None;
    }
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let fmt = surf.format();
    let n = fmt.channels().max(1);
    let px = surf.read_region(rect);
    let mask = layer.mask.as_ref().filter(|m| m.enabled).map(|m| (m.surface.read_region(rect), m.surface.channels().max(1), m.density.clamp(0.0, 1.0)));
    let mut out = Vec::with_capacity(w * h * 4);
    for i in 0..w * h {
        let p = px.get(i * n..i * n + n)?;
        let mut rgba = photocraft_raster::to_rgba(&fmt, p);
        if let Some((m, k, density)) = &mask {
            let v = m.get(i * k).copied().unwrap_or(1.0).clamp(0.0, 1.0);
            rgba[3] *= 1.0 - density * (1.0 - v);
        }
        for c in rgba {
            out.push((c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    Rgba8::new(w as u32, h as u32, out).ok()
}

/// The size of a thumbnail of a `w × h` image: at most [`THUMB_SIDE`] on its longer side.
pub fn thumb_size(w: u32, h: u32) -> (u32, u32) {
    let longer = w.max(h).max(1);
    if longer <= THUMB_SIDE {
        return (w.max(1), h.max(1));
    }
    let s = f64::from(THUMB_SIDE) / f64::from(longer);
    (((f64::from(w) * s).round() as u32).max(1), ((f64::from(h) * s).round() as u32).max(1))
}

/// Render and write a capture's result PNGs and thumbnail (on a writer thread or inline).
fn write_capture(dir: &Path, id: u64, capture: &Capture, files: &[String], thumbnail: &str) -> WriteDone {
    let mut done = WriteDone { id, bytes: 0, error: None };
    let mut first: Option<Rgba8> = None;
    for ((layer, rect), rel) in capture.items.iter().zip(files) {
        let Some(img) = render_layer(&capture.doc, *layer, *rect) else {
            done.error = Some("a result layer could not be stored".into());
            continue;
        };
        match write_png(&dir.join(rel), &img) {
            Ok(n) => done.bytes += n,
            Err(e) => done.error = Some(format!("the result could not be saved: {e}")),
        }
        if first.is_none() {
            first = Some(img);
        }
    }
    if let Some(img) = first.filter(|_| !thumbnail.is_empty()) {
        let (tw, th) = thumb_size(img.width, img.height);
        let thumb = if (tw, th) == (img.width, img.height) { Ok(img) } else { crate::generate_cmds::resize_rgba8(&img, tw, th) };
        match thumb.map_err(|e| e.to_string()).and_then(|t| write_png(&dir.join(thumbnail), &t)) {
            Ok(n) => done.bytes += n,
            Err(e) => done.error = Some(format!("the thumbnail could not be saved: {e}")),
        }
    }
    done
}

fn write_png(path: &Path, img: &Rgba8) -> std::result::Result<u64, String> {
    let png = photocraft_genai::png::encode_rgba8(img).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    photocraft_format::atomic::atomic_write(path, &png).map_err(|e| e.to_string())?;
    Ok(png.len() as u64)
}

/// The library directory the CLI uses: the desktop app's (`PHOTOCRAFT_CONFIG_DIR`, else
/// `PhotoCraftData` beside the executable in a portable copy, else the platform's per-user
/// config folder) plus `generations`, so `photocraft-cli` lists what the app made. The desktop
/// app resolves its own folder (`app_dirs.rs`) and passes it in.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_dir() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let portable = || {
        let exe = std::env::current_exe().ok()?;
        let dir = exe.parent()?;
        let marked = ["portable.txt", "PhotoCraft.portable"].iter().any(|m| dir.join(m).is_file());
        let data = dir.join("PhotoCraftData");
        (marked && data.is_dir()).then_some(data)
    };
    let base = var("PHOTOCRAFT_CONFIG_DIR").or_else(portable).or_else(|| {
        if cfg!(target_os = "macos") {
            var("HOME").map(|h| h.join("Library/Application Support/Photocraft"))
        } else if cfg!(windows) {
            var("APPDATA").map(|a| a.join("Photocraft"))
        } else {
            var("XDG_CONFIG_HOME").or_else(|| var("HOME").map(|h| h.join(".config"))).map(|c| c.join("photocraft"))
        }
    })?;
    Some(base.join("generations"))
}

/// A job label for a command that ran inline: the command's label and the prompt.
pub(crate) fn label_for(command: &str, params: &Value) -> String {
    let base = crate::commands::find(command).map_or(command, |s| s.label.trim_end_matches('…'));
    match params.get("prompt").and_then(Value::as_str).map(str::trim).filter(|p| !p.is_empty()) {
        Some(p) => format!("{base}: {}", short(p)),
        None => base.to_string(),
    }
}

/// `UTC date and time of `ms` as `YYYY-MM-DD_HHMMSS` (file names).
pub fn date_stamp(ms: u64) -> String {
    let secs = ms / 1000;
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's civil-from-days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}_{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// The file name (no extension) `generate.export` gives `rec`'s first result by default.
pub fn suggested_name(rec: &GenerationRecord) -> String {
    export_name(DEFAULT_EXPORT_NAME, rec, 1, false)
}

/// Expand a `generate.export` name pattern for result `n` (1-based) of `rec` (`many`: several
/// results are saved, so a pattern without `{n}` gets `_n`).
pub fn export_name(pattern: &str, rec: &GenerationRecord, n: usize, many: bool) -> String {
    let model = if rec.template.is_empty() { "generation".to_string() } else { rec.template.replace('/', "-") };
    let prompt: String = rec
        .prompt
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let prompt: String = prompt.chars().take(40).collect();
    let mut name = pattern
        .replace("{model}", &model)
        .replace("{template}", &model)
        .replace("{date}", &date_stamp(rec.created_ms))
        .replace("{seed}", &rec.seed.to_string())
        .replace("{prompt}", &prompt)
        .replace("{n}", &n.to_string());
    if many && !pattern.contains("{n}") {
        name.push_str(&format!("_{n}"));
    }
    let clean: String = name.chars().filter(|c| !c.is_control() && !matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*')).collect();
    let clean = clean.trim().trim_matches('.').to_string();
    if clean.is_empty() { format!("generation_{}", rec.id) } else { clean }
}

/// Is `e` worth a failed row? Requests refused before anything ran are not.
fn records_failure(e: &EngineError) -> bool {
    !matches!(e, EngineError::BadParams { .. } | EngineError::Disabled(..) | EngineError::UnknownCommand(_) | EngineError::NoDocument | EngineError::NoLayer(_))
}

impl Session {
    /// Attach the library directory (the desktop app and the CLI do; see the module docs).
    pub fn set_generations_dir(&mut self, dir: PathBuf) {
        self.generations.set_dir(dir);
    }

    /// Write the library's index now and wait for pending result files.
    pub fn flush_generations(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        for h in self.generations.writers.drain(..) {
            let _ = h.join();
        }
        self.generations.drain_writes();
        self.generations.flush();
    }

    /// The document `doc` (or the active one), its name and id.
    fn doc_title(&self, doc: Option<DocId>) -> (Option<u64>, String) {
        let st = doc.and_then(|d| self.docs.iter().find(|st| st.doc.id == d)).or_else(|| self.active());
        st.map_or((None, String::new()), |st| (Some(st.doc.id.0), st.doc.name.clone()))
    }

    /// A background generation started as job `job`.
    pub(crate) fn generation_started(&mut self, command: &str, params: &Value, label: &str, job: JobId, doc: Option<DocId>) {
        let (command, params, label) = self.generations.take_override(command, params, label);
        // A job that locks no document (Generate Image into a new one) takes its document from
        // the result.
        let (doc, title) = if doc.is_some() { self.doc_title(doc) } else { (None, String::new()) };
        let id = self.generations.begin(&command, &label, &params, doc, title);
        self.generations.by_job.insert(job.0, id);
    }

    /// A generation that ran inline ended with `r`.
    pub(crate) fn generation_inline(&mut self, command: &str, params: &Value, r: std::result::Result<&Value, &EngineError>) {
        if let Err(e) = r
            && !records_failure(e)
        {
            self.generations.override_next = None;
            return;
        }
        let label = label_for(command, params);
        let (command, params, label) = self.generations.take_override(command, params, &label);
        let (doc, title) = self.doc_title(None);
        let id = self.generations.begin(&command, &label, &params, doc, title);
        match r {
            Ok(v) => {
                let found = self.capture_result(v, None);
                self.generations.finish_done(id, v, found, false);
            }
            Err(EngineError::Cancelled) => self.generations.finish_cancelled(id),
            Err(e) => self.generations.finish_failed(id, &e.to_string()),
        }
    }

    /// Background job `job` (on document `doc`) ended.
    pub(crate) fn generation_job_ended(&mut self, job: JobId, doc: Option<DocId>, outcome: &JobOutcome) {
        let Some(id) = self.generations.by_job.remove(&job.0) else { return };
        self.generations.ended.push_back((job.0, id));
        if self.generations.ended.len() > 64 {
            self.generations.ended.pop_front();
        }
        match outcome {
            JobOutcome::Done(v) => {
                let found = self.capture_result(v, doc);
                self.generations.finish_done(id, v, found, true);
            }
            JobOutcome::Failed(msg) => self.generations.finish_failed(id, msg),
            JobOutcome::Cancelled => self.generations.finish_cancelled(id),
        }
    }

    /// Find a generative command's result layers (`layers`, `layer`, or the layers of a new
    /// `document`) and prepare their pixels for the library.
    fn capture_result(&self, v: &Value, doc: Option<DocId>) -> Found {
        let new_doc = v.get("document").and_then(Value::as_u64).and_then(|i| usize::try_from(i).ok());
        let index = new_doc.or_else(|| doc.and_then(|d| self.docs.iter().position(|st| st.doc.id == d))).or(self.active);
        let Some(st) = index.and_then(|i| self.docs.get(i)) else {
            return Found { doc: None, title: String::new(), layers: Vec::new(), info: None, capture: None };
        };
        let mut layers: Vec<u64> = v.get("layers").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
        if layers.is_empty()
            && let Some(l) = v.get("layer").and_then(Value::as_u64)
        {
            layers.push(l);
        }
        if layers.is_empty() && new_doc.is_some() {
            layers = st.doc.layers.iter().rev().filter(|l| generative_info(l).is_some()).map(|l| l.id.0).collect();
        }
        let info = layers.iter().find_map(|l| st.doc.layer(LayerId(*l)).and_then(generative_info));
        let canvas = st.doc.bounds();
        let items: Vec<(LayerId, Rect)> = layers
            .iter()
            .filter_map(|l| {
                let layer = st.doc.layer(LayerId(*l))?;
                layer.surface()?;
                let rect = match generative_info(layer) {
                    Some(i) if i.rect[2] > 0 && i.rect[3] > 0 => {
                        Rect::new(i.rect[0], i.rect[1], i.rect[0].saturating_add(i.rect[2]), i.rect[1].saturating_add(i.rect[3])).intersect(&canvas)
                    }
                    _ => canvas,
                };
                (!rect.is_empty()).then_some((LayerId(*l), rect))
            })
            .collect();
        let capture = (!items.is_empty()).then(|| Capture { doc: st.doc.clone(), items });
        Found { doc: Some(st.doc.id.0), title: st.doc.name.clone(), layers, info, capture }
    }
}

// ── Commands ──────────────────────────────────────────────────────────────

fn record_id(cmd: &str, p: &Value) -> Result<u64> {
    p.get("id").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "`id` is required: a generation id from generate.jobs"))
}

fn run_jobs(s: &mut Session, p: &Value) -> Result<Value> {
    let limit = opt_num(JOBS, p, "limit", 1.0, 10_000.0)?.map_or(100, |x| x as usize);
    let include_dismissed = opt_bool(JOBS, p, "includeDismissed", false)?;
    let status = match opt_str(JOBS, p, "status", 20)? {
        None => None,
        Some("running") => Some(GenStatus::Running),
        Some("done") => Some(GenStatus::Done),
        Some("failed") => Some(GenStatus::Failed),
        Some("cancelled") => Some(GenStatus::Cancelled),
        Some(other) => return Err(bad(JOBS, format!("`status` must be running, done, failed or cancelled (got `{other}`)"))),
    };
    s.generations.refresh();
    s.generations.poll();
    let list: Vec<&GenerationRecord> = s
        .generations
        .records
        .iter()
        .rev()
        .filter(|r| include_dismissed || !r.dismissed)
        .filter(|r| status.is_none_or(|st| r.status == st))
        .take(limit)
        .collect();
    let running = s.generations.records.iter().filter(|r| r.status == GenStatus::Running).count();
    Ok(json!({"jobs": list, "running": running, "total": s.generations.records.len(), "dir": s.generations.dir}))
}

fn run_dismiss(s: &mut Session, p: &Value) -> Result<Value> {
    let now = unix_ms();
    let ids: Vec<u64> = match (p.get("id").filter(|v| !v.is_null()), opt_str(DISMISS, p, "which", 20)?) {
        (Some(_), _) => {
            let id = record_id(DISMISS, p)?;
            s.generations.get(id).ok_or_else(|| EngineError::Other(format!("no generation {id}")))?;
            vec![id]
        }
        (None, Some(which)) => {
            let pick: fn(&GenerationRecord, u64) -> bool = match which {
                "completed" => |r, _| r.status == GenStatus::Done,
                "failed" => |r, _| matches!(r.status, GenStatus::Failed | GenStatus::Cancelled),
                "stale" => |r, now| r.status == GenStatus::Running && r.progress < 0.01 && now.saturating_sub(r.created_ms) > STALE_MS,
                "all" => |r, _| r.status != GenStatus::Running,
                other => return Err(bad(DISMISS, format!("`which` must be completed, failed, stale or all (got `{other}`)"))),
            };
            s.generations.records.iter().filter(|r| !r.dismissed && pick(r, now)).map(|r| r.id).collect()
        }
        (None, None) => return Err(bad(DISMISS, "give `id` or `which`")),
    };
    for id in &ids {
        if let Some(r) = s.generations.get_mut(*id) {
            r.dismissed = true;
            if r.status == GenStatus::Running {
                // A stale row: its job is gone (an earlier crash) or stuck.
                r.status = GenStatus::Failed;
                r.failure = Some(GenFailure { kind: GenFailureKind::Interrupted, message: "Dismissed while it seemed stuck".into() });
            }
        }
    }
    s.generations.dirty = true;
    Ok(json!({"dismissed": ids.len()}))
}

fn run_pin(s: &mut Session, p: &Value) -> Result<Value> {
    let id = record_id(PIN, p)?;
    let pinned = opt_bool(PIN, p, "pinned", true)?;
    let r = s.generations.get_mut(id).ok_or_else(|| EngineError::Other(format!("no generation {id}")))?;
    r.pinned = pinned;
    s.generations.dirty = true;
    Ok(json!({"id": id, "pinned": pinned}))
}

fn run_clear(s: &mut Session, p: &Value) -> Result<Value> {
    let which = opt_str(CLEAR, p, "which", 20)?.ok_or_else(|| bad(CLEAR, "`which` is required: completed, failed or all"))?;
    let pick: fn(&GenerationRecord) -> bool = match which {
        "completed" => |r| r.status == GenStatus::Done,
        "failed" => |r| matches!(r.status, GenStatus::Failed | GenStatus::Cancelled),
        "all" => |r| r.status != GenStatus::Running,
        other => return Err(bad(CLEAR, format!("`which` must be completed, failed or all (got `{other}`)"))),
    };
    let ids: Vec<u64> = s.generations.records.iter().filter(|r| pick(r)).map(|r| r.id).collect();
    for id in &ids {
        s.generations.remove(*id);
    }
    s.generations.flush();
    Ok(json!({"removed": ids.len()}))
}

/// The open document a record ran on, if it is still open.
fn open_doc_of(s: &Session, rec: &GenerationRecord) -> Option<usize> {
    let d = rec.doc?;
    s.docs.iter().position(|st| st.doc.id.0 == d)
}

fn run_recreate(s: &mut Session, p: &Value) -> Result<Value> {
    let id = record_id(RECREATE, p)?;
    let rec = s.generations.get(id).cloned().ok_or_else(|| EngineError::Other(format!("no generation {id}")))?;
    let seed = parse_seed(RECREATE, p)?;
    let variations = opt_num(RECREATE, p, "variations", 1.0, f64::from(crate::generate_cmds::MAX_VARIATIONS))?.map(|v| v as u32);
    let doc = open_doc_of(s, &rec);
    let first = rec.result.as_ref().and_then(|r| r.layers.first().copied());
    let similar_ok = matches!(rec.command.as_str(), "generate.fill" | "generate.edit" | "generate.expand" | "generate.image" | "generate.similar");
    // The result layer is still there: Generate Similar in place.
    if let (Some(i), Some(layer), true) = (doc, first, similar_ok)
        && s.docs.get(i).is_some_and(|st| st.doc.layer(LayerId(layer)).is_some())
    {
        let q = json!({"layer": layer, "seed": seed, "variations": variations.unwrap_or(rec.count.max(1))});
        return run_inner(s, Some(i), crate::generate_cmds::SIMILAR, q);
    }
    let mut q = rec.params.clone();
    if !q.is_object() {
        q = json!({});
    }
    if let Some(o) = q.as_object_mut() {
        o.insert("seed".into(), json!(seed));
        if let Some(v) = variations {
            o.insert("variations".into(), json!(v));
        }
    }
    let command =
        if rec.command == crate::generate_cmds::SIMILAR { rec.info.as_ref().map_or(rec.command.clone(), |i| i.command.clone()) } else { rec.command.clone() };
    if command == "generate.image" {
        if let Some(o) = q.as_object_mut()
            && (doc.is_none() || o.get("target").and_then(Value::as_str) == Some("document"))
        {
            o.insert("target".into(), json!("document"));
        }
        return run_inner(s, doc, "generate.image", q);
    }
    // The original command again on its document, when it can run there now (a fill needs a
    // selection, for instance).
    if let Some(i) = doc {
        s.set_active(i);
        match crate::commands::find(&command).map(|spec| (spec.enabled)(s)) {
            Some(Ok(())) => return run_inner(s, Some(i), &command, q),
            Some(Err(why)) if !matches!(command.as_str(), "generate.fill" | "generate.edit") => {
                return Err(EngineError::Disabled(command.clone(), why));
            }
            _ => {}
        }
    }
    // The document is closed (or the selection is gone): a fill's or an edit's prompt becomes
    // a Generate Image of it in a new document.
    if matches!(command.as_str(), "generate.fill" | "generate.edit") {
        let prompt = rec.info.as_ref().map_or(rec.prompt.clone(), |i| i.prompt.clone());
        if prompt.trim().is_empty() {
            return Err(EngineError::Other("its document is closed and it had no prompt; Place its result instead".into()));
        }
        let (w, h) = rec.result.as_ref().map_or((1024, 1024), |r| (r.width, r.height));
        let side = |v: u32| if v < 64 { 1024 } else { (v / 16 * 16).clamp(64, 2048) };
        let q = json!({"prompt": prompt, "seed": seed, "target": "document", "width": side(w), "height": side(h)});
        return run_inner(s, None, "generate.image", q).map(|mut v| {
            v["fallback"] = json!("image");
            v
        });
    }
    Err(EngineError::Other(format!("the document `{command}` ran on is closed; open it again and Recreate, or Place the result")))
}

/// Run `command` with `q` on document `doc` (made active) as `generate.recreate`'s work; the
/// library row describes `command`.
fn run_inner(s: &mut Session, doc: Option<usize>, command: &str, q: Value) -> Result<Value> {
    let spec = crate::commands::find(command).ok_or_else(|| EngineError::UnknownCommand(command.to_string()))?;
    if let Some(i) = doc {
        s.set_active(i);
    }
    if doc.is_some()
        && let Some(job) = s.active_job()
    {
        return Err(EngineError::Disabled(RECREATE.into(), format!("“{}” is still running on this document", job.label)));
    }
    // Generate Similar names its layer, so the active layer does not matter.
    if command != crate::generate_cmds::SIMILAR {
        (spec.enabled)(s).map_err(|why| EngineError::Disabled(command.to_string(), why))?;
    }
    s.generations.override_next = Some((command.to_string(), q.clone(), label_for(command, &q)));
    (spec.run)(s, &q)
}

fn run_place(s: &mut Session, p: &Value) -> Result<Value> {
    let id = record_id(PLACE, p)?;
    let index = opt_num(PLACE, p, "index", 0.0, 63.0)?.map_or(0, |v| v as usize);
    let target = match opt_str(PLACE, p, "target", 20)? {
        None => {
            if s.active().is_some() {
                "layer"
            } else {
                "document"
            }
        }
        Some(t @ ("layer" | "document")) => t,
        Some(other) => return Err(bad(PLACE, format!("`target` must be layer or document (got `{other}`)"))),
    };
    let rec = s.generations.get(id).cloned().ok_or_else(|| EngineError::Other(format!("no generation {id}")))?;
    let res = rec.result.as_ref().filter(|r| !r.files.is_empty()).ok_or_else(|| EngineError::Other("this generation has no stored result to place".into()))?;
    if index >= res.files.len() {
        return Err(bad(PLACE, format!("`index` must be below {} (the results it has)", res.files.len())));
    }
    let path = s.generations.result_path(id, index).ok_or_else(|| EngineError::Other("the library has no folder".into()))?;
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) if !res.saved => return Err(EngineError::Other("the result is still being saved; try again in a moment".into())),
        Err(e) => return Err(EngineError::Other(format!("the result file is missing ({e}); Recreate it instead"))),
    };
    let img = photocraft_genai::png::decode_rgba8(&bytes).map_err(crate::generate_cmds::gen_err)?;
    let mut info = rec.info.clone().unwrap_or_else(|| GenerativeInfo { command: rec.command.clone(), prompt: rec.prompt.clone(), ..GenerativeInfo::default() });
    let name = if info.name.is_empty() { rec.label.clone() } else { info.name.clone() };
    if target == "document" || s.active().is_none() {
        let mut doc =
            Document::new(name.clone(), photocraft_doc::Size::new(img.width, img.height), photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8);
        let mut layer = photocraft_doc::Layer::raster(name, doc.pixel_format());
        crate::generate_cmds::write_rgba8(&mut layer, &img, doc.bounds())?;
        info.rect = [0, 0, img.width as i32, img.height as i32];
        crate::generate_cmds::set_generative_info(&mut layer, &info);
        doc.layers.push(layer);
        let i = s.add_document(doc, None);
        return Ok(json!({"document": i, "width": img.width, "height": img.height}));
    }
    let (w, h) = (img.width as i32, img.height as i32);
    let nid = s.edit("Place Generation", move |doc, active| {
        let canvas = doc.bounds();
        let r = info.rect;
        let rect = if r[2] == w && r[3] == h && Rect::new(r[0], r[1], r[0] + w, r[1] + h).intersect(&canvas) == Rect::new(r[0], r[1], r[0] + w, r[1] + h) {
            Rect::new(r[0], r[1], r[0] + w, r[1] + h)
        } else {
            let (x, y) = ((canvas.width() as i32 - w) / 2, (canvas.height() as i32 - h) / 2);
            Rect::new(x, y, x + w, y + h)
        };
        let mut layer = photocraft_doc::Layer::raster(name, doc.pixel_format());
        crate::generate_cmds::write_rgba8(&mut layer, &img, rect)?;
        info.rect = [rect.x0, rect.y0, w, h];
        crate::generate_cmds::set_generative_info(&mut layer, &info);
        let nid = doc.insert_above(*active, layer);
        *active = Some(nid);
        Ok(nid)
    })?;
    Ok(json!({"layer": nid.0, "width": w, "height": h}))
}

fn run_export(s: &mut Session, p: &Value) -> Result<Value> {
    let id = record_id(EXPORT, p)?;
    let index = opt_num(EXPORT, p, "index", 0.0, 63.0)?.map(|v| v as usize);
    // The folder and the name pattern: the call's, else the preferences' (Preferences › AI
    // Integrations › Save results dir / naming).
    let prefs = &s.prefs().integrations;
    let (dir_pref, naming) = (prefs.save_results_dir.trim().to_string(), prefs.save_results_naming.trim().to_string());
    let dir = match opt_str(EXPORT, p, "dir", 4096)?.map(str::trim).filter(|d| !d.is_empty()) {
        Some(d) => d.to_string(),
        None if !dir_pref.is_empty() => dir_pref,
        None => return Err(bad(EXPORT, "`dir` is required when Preferences › AI Integrations › Save results dir is empty: the folder to save into")),
    };
    let pattern = match opt_str(EXPORT, p, "name", 200)?.map(str::trim).filter(|n| !n.is_empty()) {
        Some(n) => n.to_string(),
        None if !naming.is_empty() => naming,
        None => DEFAULT_EXPORT_NAME.to_string(),
    };
    let pattern = pattern.as_str();
    if pattern.contains(['/', '\\']) || pattern.contains("..") {
        return Err(bad(EXPORT, "`name` is a file name pattern, not a path"));
    }
    let rec = s.generations.get(id).cloned().ok_or_else(|| EngineError::Other(format!("no generation {id}")))?;
    let res = rec.result.as_ref().filter(|r| !r.files.is_empty()).ok_or_else(|| EngineError::Other("this generation has no stored result to save".into()))?;
    let picks: Vec<usize> = match index {
        Some(i) if i < res.files.len() => vec![i],
        Some(_) => return Err(bad(EXPORT, format!("`index` must be below {}", res.files.len()))),
        None => (0..res.files.len()).collect(),
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{}: {e}", dir.display())))?;
    let many = picks.len() > 1;
    let mut files = Vec::new();
    for i in picks {
        let src = s.generations.result_path(id, i).ok_or_else(|| EngineError::Other("the library has no folder".into()))?;
        let bytes = std::fs::read(&src).map_err(|e| EngineError::Other(format!("the result file is missing ({e})")))?;
        let stem = export_name(pattern, &rec, i + 1, many);
        let mut dst = dir.join(format!("{stem}.png"));
        let mut k = 1;
        while dst.exists() && k < 10_000 {
            dst = dir.join(format!("{stem}-{k}.png"));
            k += 1;
        }
        photocraft_format::atomic::atomic_write(&dst, &bytes).map_err(|e| EngineError::Other(format!("{}: {e}", dst.display())))?;
        files.push(dst.to_string_lossy().into_owned());
    }
    Ok(json!({"files": files}))
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: JOBS,
            label: "List Generations",
            menu: &[],
            shortcut: None,
            params: r#"{"limit":int=100,"includeDismissed":bool=false,"status":{"running|done|failed|cancelled"?}} → {"jobs":[{"id","createdMs","finishedMs","command","label","prompt","enhanced","template","seed","count","params","document","status","progress","message","failure":{"kind","message"}?,"result":{"layers","width","height","ms","files","thumbnail","bytes","saved","selection"}?,"info"?,"dismissed","pinned"}],"running","total","dir"} (the Generations library, newest first: every generative job with its prompt, template, progress, failure kind (serverUnreachable, modelFilesMissing, outOfMemory, serverError, timeout, badInput, discarded, interrupted, other) and stored result files, relative to `dir`; kept across restarts)"#,
            enabled: always,
            run: run_jobs,
            journal: false,
        },
        CommandSpec {
            id: DISMISS,
            label: "Dismiss Generations",
            menu: &[],
            shortcut: None,
            params: r#"{"id":{u64?},"which":{"completed|failed|stale|all"?}} → {"dismissed"} (hides rows from the Generations panel without deleting their files; stale = running for over ten minutes with no progress)"#,
            enabled: always,
            run: run_dismiss,
            journal: false,
        },
        CommandSpec {
            id: PIN,
            label: "Pin Generation",
            menu: &[],
            shortcut: None,
            params: r#"{"id":u64,"pinned":bool=true} → {"id","pinned"} (a pinned result is never dropped to stay within the library's limits)"#,
            enabled: always,
            run: run_pin,
            journal: false,
        },
        CommandSpec {
            id: CLEAR,
            label: "Clear Generations",
            menu: &[],
            shortcut: None,
            params: r#"{"which":"completed|failed|all"} → {"removed"} (deletes rows and their stored files; running ones stay)"#,
            enabled: always,
            run: run_clear,
            journal: false,
        },
        CommandSpec {
            id: RECREATE,
            label: "Recreate Generation",
            menu: &[],
            shortcut: None,
            params: r#"{"id":u64,"seed":{u64?=random},"variations":{int?}} → as the command it runs (Generate Similar on the result layer when it is still open; otherwise the original command again on its document with a new seed; a fill whose document closed becomes a Generate Image of its prompt in a new document (`fallback`: "image"); a background job)"#,
            enabled: |_| web_unavailable(),
            run: run_recreate,
            journal: true,
        },
        CommandSpec {
            id: PLACE,
            label: "Place Generation",
            menu: &[],
            shortcut: None,
            params: r#"{"id":u64,"index":int=0,"target":{"layer|document"?}} → {"layer"|"document","width","height"} (a stored result back as a new layer above the active one, at its original place when it fits, else centred; or as a new document; target defaults to layer when a document is open)"#,
            enabled: |_| web_unavailable(),
            run: run_place,
            journal: true,
        },
        CommandSpec {
            id: EXPORT,
            label: "Export Generation",
            menu: &[],
            shortcut: None,
            params: r#"{"id":u64,"dir":{path?=Preferences › AI Integrations › Save results dir},"index":{int?=all},"name":{str?=Save results naming, "{model}_{date}_{seed}"}} → {"files":[path]} (saves stored results as PNG files; without dir and with the preference empty it is an error (the app asks for a folder); name tokens {model}, {template}, {date} (UTC, YYYY-MM-DD_HHMMSS), {seed}, {prompt}, {n}; several results get _n; existing files are not replaced)"#,
            enabled: |_| web_unavailable(),
            run: run_export,
            journal: false,
        },
    ]
}

#[cfg(test)]
#[path = "generate_jobs_tests.rs"]
mod tests;
