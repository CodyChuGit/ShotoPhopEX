//! The worker process (plan-native-engine §3.4). The editor never runs a model in its own
//! address space: an engine runs as a child process and the two talk over a local channel. A
//! CUDA fault, an out-of-memory or a driver reset ends the child; the editor reports it and
//! respawns; documents are never at risk. The same channel serves every engine (our native
//! one, sd.cpp through FFI), so the editor has one code path.
//!
//! - [`protocol`]: the messages (JSON) and the frames that carry them and the pixels (binary).
//! - [`server`]: the loop a worker runs: one editor connection at a time, one run in flight,
//!   progress streamed, cancellation honoured mid-run.
//! - [`client`]: the editor side: spawn or connect, run with progress and cancel, respawn.
//! - [`fake`]: an engine that paints solid pictures, for tests and for wiring the editor before
//!   a real model exists.
//!
//! Transport: TCP on 127.0.0.1 with a random port the child prints on its first stdout line and
//! a token the parent hands it in the environment, on every platform with the standard library
//! alone (the app's control channel works the same way). A named pipe / Unix socket transport
//! can replace it behind the same [`protocol`] without touching callers.

#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing))]

pub mod client;
pub mod fake;
pub mod protocol;
pub mod server;

pub use client::EngineClient;
pub use protocol::{Blob, DoneInfo, ErrorKind, Health, Reply, Request, RunRequest};
pub use server::{Engine, EngineError, serve};
