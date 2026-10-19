//! The ONNX Runtime environment: found at run time, never bundled.
//!
//! The user (or the editor's preferences) names a directory that holds `onnxruntime.dll` /
//! `libonnxruntime.so` and its provider libraries, plus zero or more directories that hold the
//! CUDA and cuDNN runtime libraries the CUDA provider needs. Everything is loaded from there;
//! nothing is read from `PATH` by design, so two installed CUDA versions cannot fight.

use std::path::{Path, PathBuf};

use ort::ep;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

/// Where to find the runtime and which device to run on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// The directory holding `onnxruntime.dll` (or `libonnxruntime.so`) and the provider libraries.
    pub ort_dir: PathBuf,
    /// Directories to search for the CUDA toolkit runtime (`cudart`, `cublas`, `cufft`) and for
    /// cuDNN 9, in order; each library is taken from the first directory that has it.
    pub library_dirs: Vec<PathBuf>,
    /// Which device to run on.
    pub device: Device,
    /// Write ONNX Runtime's per-node profile of every session into this directory (one JSON
    /// trace per session, closed when the session drops). Diagnostics only.
    pub profile_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// The CUDA execution provider on GPU `ordinal`; falls back to the CPU provider for operators
    /// CUDA cannot run.
    Cuda { ordinal: i32 },
    /// The CPU provider only.
    Cpu,
}

/// An open runtime: one environment, sessions created from it.
pub struct Runtime {
    env: Environment,
    config: RuntimeConfig,
}

/// What went wrong opening the runtime or a model.
#[derive(Debug)]
pub enum RuntimeError {
    /// The runtime library was not where the configuration said.
    MissingLibrary(PathBuf),
    /// ONNX Runtime refused to load, or a session refused to open.
    Ort(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::MissingLibrary(p) => write!(f, "ONNX Runtime library not found at {}", p.display()),
            RuntimeError::Ort(e) => write!(f, "ONNX Runtime: {e}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<ort::Error> for RuntimeError {
    fn from(e: ort::Error) -> Self {
        RuntimeError::Ort(e.to_string())
    }
}

/// The runtime library's file name on this platform.
pub fn library_name() -> &'static str {
    if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// The CUDA and cuDNN libraries the CUDA provider loads, in dependency order.
fn cuda_libraries() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = ep::cuda::CUDA_DYLIBS.to_vec();
    v.extend_from_slice(ep::cuda::CUDNN_DYLIBS);
    v
}

/// Preload every CUDA/cuDNN library that can be found in `dirs`, so the provider resolves them
/// from there rather than from `PATH`. Returns the names that were not found anywhere; the caller
/// decides whether that is fatal (it is not when the CPU device was asked for).
///
/// After the known list, every other `cudnn*` library in those directories is loaded too: cuDNN
/// opens its engine sub-libraries by name at run time (9.24 added `cudnn_engines_tensor_ir`,
/// which `ort`'s list predates), and a sub-library it cannot find fails the first convolution.
pub fn preload_cuda(dirs: &[PathBuf]) -> Vec<&'static str> {
    let mut missing = Vec::new();
    let mut loaded: Vec<PathBuf> = Vec::new();
    for name in cuda_libraries() {
        let found = dirs.iter().map(|d| d.join(name)).find(|p| p.is_file());
        match found {
            Some(path) => {
                // A failure here surfaces again, with ORT's own message, when the session opens.
                let _ = ort::util::preload_dylib(&path);
                loaded.push(path);
            }
            None => missing.push(name),
        }
    }
    let (prefix, marker) = if cfg!(windows) { ("cudnn", "_9.dll") } else { ("libcudnn", ".so.9") };
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let mut extra: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(prefix) && n.contains(marker)))
            .filter(|p| !loaded.contains(p))
            .collect();
        extra.sort();
        for path in extra {
            let _ = ort::util::preload_dylib(&path);
            loaded.push(path);
        }
    }
    missing
}

impl Runtime {
    /// Load ONNX Runtime from `config.ort_dir` and build the environment.
    pub fn open(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        let lib = config.ort_dir.join(library_name());
        if !lib.is_file() {
            return Err(RuntimeError::MissingLibrary(lib));
        }
        if matches!(config.device, Device::Cuda { .. }) {
            // Missing pieces are not fatal here: the CUDA provider reports precisely which
            // library it could not load when the first session opens, which is the better message.
            let _ = preload_cuda(&config.library_dirs);
        }
        let builder = ort::init_from(&lib).map_err(|e| RuntimeError::Ort(e.to_string()))?;
        let env = builder.with_name("genai-native").build()?;
        Ok(Runtime { env, config })
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Open a model file as a session on the configured device.
    pub fn session(&self, model: &Path) -> Result<Session, RuntimeError> {
        if !model.is_file() {
            return Err(RuntimeError::MissingLibrary(model.to_path_buf()));
        }
        let mut builder = Session::builder(&self.env)?.with_optimization_level(GraphOptimizationLevel::Level3).map_err(ort_err)?;
        if let Some(dir) = &self.config.profile_dir {
            let stem = model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "session".into());
            builder = builder.with_profiling(dir.join(stem)).map_err(ort_err)?;
        }
        builder = match self.config.device {
            Device::Cuda { ordinal } => builder
                .with_execution_providers([ep::CUDA::default().with_device_id(ordinal).build().error_on_failure(), ep::CPU::default().build()])
                .map_err(ort_err)?,
            Device::Cpu => builder.with_execution_providers([ep::CPU::default().build()]).map_err(ort_err)?,
        };
        Ok(builder.commit_from_file(model)?)
    }

    /// Open a model that should stay on the CPU whatever the configured device (small graphs
    /// whose launch overhead on the GPU exceeds their work).
    pub fn cpu_session(&self, model: &Path) -> Result<Session, RuntimeError> {
        if !model.is_file() {
            return Err(RuntimeError::MissingLibrary(model.to_path_buf()));
        }
        Ok(Session::builder(&self.env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_execution_providers([ep::CPU::default().build()])
            .map_err(ort_err)?
            .commit_from_file(model)?)
    }
}

/// Any ORT-side error, as text (the builder's error type is not `ort::Error`).
fn ort_err(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Ort(e.to_string())
}

/// One input or output of a graph, for `inspect`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Port {
    pub name: String,
    pub dtype: String,
}

/// The inputs and outputs of a session.
pub fn ports(session: &Session) -> (Vec<Port>, Vec<Port>) {
    let port = |o: &ort::value::Outlet| Port { name: o.name().to_string(), dtype: format!("{}", o.dtype()) };
    (session.inputs().iter().map(port).collect(), session.outputs().iter().map(port).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_is_an_error_not_a_crash() {
        let cfg = RuntimeConfig { ort_dir: PathBuf::from("Z:/nowhere"), library_dirs: vec![], device: Device::Cpu, profile_dir: None };
        match Runtime::open(cfg) {
            Err(RuntimeError::MissingLibrary(p)) => assert!(p.ends_with(library_name())),
            other => panic!("expected MissingLibrary, got {:?}", other.err()),
        }
    }

    #[test]
    fn preload_reports_what_it_could_not_find() {
        let missing = preload_cuda(&[PathBuf::from("Z:/nowhere")]);
        assert_eq!(missing.len(), cuda_libraries().len());
    }
}
