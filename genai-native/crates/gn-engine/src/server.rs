//! The worker's loop: bind, announce the port, accept the editor, answer requests. One
//! connection at a time (the editor owns the engine), one run in flight; a reader thread keeps
//! listening while a run executes so `Cancel` reaches it mid-run.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::Value;

use crate::protocol::{self, Blob, DoneInfo, ErrorKind, Framed, Health, Preview, PreviewInfo, Reply, Request, RunRequest, write_json};

/// A failed run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineError {
    pub kind: ErrorKind,
    pub message: String,
}

impl EngineError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled, "cancelled")
    }
}

/// What an engine implements. `run` reports progress through `progress` and must return
/// `EngineError::cancelled()` soon after `cancel` turns true.
pub trait Engine: Send {
    fn name(&self) -> String;
    fn health(&mut self) -> Health;
    /// Produce pictures (as blobs: bytes + descriptors) for a request whose blobs are checked.
    fn run(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        cancel: &AtomicBool,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError>;
    /// [`Self::run`] for a request that asked for previews (`req.preview`): `preview` takes
    /// pictures of the run in progress. It must never change what the run produces. An engine
    /// without previews keeps this default and the run shows none.
    fn run_previewed(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        preview: &mut dyn FnMut(Preview),
        cancel: &AtomicBool,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), EngineError> {
        let _ = preview;
        self.run(req, blobs, progress, cancel)
    }
    /// Unload everything.
    fn free(&mut self);
    /// A light engine (its models load again in a second or two, the perception engine's) that
    /// gives the device back before another engine of the same worker runs (`Multi`): on a
    /// card shared with other processes the diffusion pipeline's working set and a resident
    /// detector together paged for minutes (2026-10-26).
    fn yields(&self) -> bool {
        false
    }
    /// What is installed under `dir` (engine-specific JSON).
    fn index(&mut self, dir: &str) -> Value {
        let _ = dir;
        Value::Null
    }
}

/// Bind a listener on loopback with a random port.
pub fn bind() -> std::io::Result<TcpListener> {
    TcpListener::bind(("127.0.0.1", 0))
}

/// The line a worker prints first so its parent learns the port.
pub fn port_line(port: u16) -> String {
    format!("gn-engine port={port}")
}

/// The port from such a line.
pub fn parse_port_line(line: &str) -> Option<u16> {
    line.trim().strip_prefix("gn-engine port=")?.parse().ok()
}

/// Serve `engine` on `listener` until the greeted client says `Quit` **or goes away** (an editor
/// that crashed must not leave a worker behind; it starts a new one next time). Each connection
/// must open with `Hello` carrying `token`. The engine serves one client at a time; a second
/// client is told so at once and closed, instead of waiting in the backlog; a client refused at
/// the greeting does not end the worker.
pub fn serve(listener: &TcpListener, token: &str, engine: &mut dyn Engine) -> std::io::Result<()> {
    serve_with(listener, token, engine, false)
}

/// [`serve`], and with `stay` the worker outlives its clients: a client that goes away ends
/// only its connection, and the next one finds the models still resident. Only `Quit` ends
/// the worker then. For a worker started by hand (`--engine --stay`) that a run of many short
/// processes shares, such as a bench through the CLI.
pub fn serve_with(listener: &TcpListener, token: &str, engine: &mut dyn Engine, stay: bool) -> std::io::Result<()> {
    let port = listener.local_addr()?.port();
    let engine = Mutex::new(engine);
    let quit = AtomicBool::new(false);
    std::thread::scope(|scope| -> std::io::Result<()> {
        for stream in listener.incoming() {
            if quit.load(Ordering::SeqCst) {
                break;
            }
            let stream = stream?;
            let (engine, quit, token) = (&engine, &quit, token);
            scope.spawn(move || {
                // The greeting comes first, before the engine is taken: a peer that never shows
                // the token cannot hold the engine away from the editor, nor make it allocate.
                let stream = match greet(stream, token) {
                    Ok(Some(stream)) => stream,
                    Ok(None) | Err(_) => return,
                };
                let Ok(mut guard) = engine.try_lock() else {
                    let _ =
                        write_json(&mut &stream, &Reply::Error { kind: ErrorKind::BadRequest, message: "another client is connected to this engine".into() });
                    return;
                };
                let ended = match serve_connection(stream, *guard) {
                    Ok(Ended::Quit) => true,
                    Ok(Ended::Left) => !stay,
                    Err(e) => {
                        eprintln!("gn-engine: connection ended: {e}");
                        true
                    }
                };
                if ended {
                    quit.store(true, Ordering::SeqCst);
                    drop(guard);
                    // Wake the accept loop so it sees the flag.
                    let _ = TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], port)), std::time::Duration::from_secs(1));
                }
            });
        }
        Ok(())
    })
}

/// How long a new connection has to greet before it is dropped.
const GREETING_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Read and check a connection's `Hello`. `Some(stream)` when it carries this protocol's version
/// and the token (the stream is then ready to be answered); `None` after telling the peer why not.
fn greet(stream: TcpStream, token: &str) -> std::io::Result<Option<TcpStream>> {
    stream.set_read_timeout(Some(GREETING_TIMEOUT))?;
    let mut reader = &stream;
    let hello = protocol::read_greeting::<Request>(&mut reader);
    let refusal = match hello {
        Ok(Some(Request::Hello { version, token: t })) if version == protocol::VERSION && protocol::same_secret(&t, token) => {
            stream.set_read_timeout(None)?;
            return Ok(Some(stream));
        }
        Ok(Some(Request::Hello { version, .. })) if version != protocol::VERSION => {
            format!("protocol version {version}, this engine speaks {}", protocol::VERSION)
        }
        Ok(None) => return Ok(None),
        Ok(Some(_)) | Err(_) => "the first message must be Hello with the token".to_string(),
    };
    let _ = write_json(&mut &stream, &Reply::Error { kind: ErrorKind::BadRequest, message: refusal });
    Ok(None)
}

/// How a connection ended.
enum Ended {
    /// The greeted client said `Quit`.
    Quit,
    /// The greeted client hung up (or its process died).
    Left,
}

/// One greeted connection.
fn serve_connection(stream: TcpStream, engine: &mut dyn Engine) -> std::io::Result<Ended> {
    stream.set_nodelay(true)?;
    let mut reader = stream.try_clone()?;
    let writer = Arc::new(Mutex::new(stream));
    send(&writer, &Reply::Hello { version: protocol::VERSION, engine: engine.name() })?;
    // Reader thread: every message into a channel; a Cancel also flips the flag at once.
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Framed<Request>>();
    let cancel_r = Arc::clone(&cancel);
    std::thread::spawn(move || {
        while let Ok(Some(f)) = protocol::read_message::<Request>(&mut reader) {
            if matches!(f.message, Request::Cancel) {
                cancel_r.store(true, Ordering::SeqCst);
            }
            if tx.send(f).is_err() {
                break;
            }
        }
    });
    while let Ok(f) = rx.recv() {
        match f.message {
            Request::Hello { .. } => send(&writer, &Reply::Error { kind: ErrorKind::BadRequest, message: "already greeted".into() })?,
            Request::Health => {
                let h = engine.health();
                send(&writer, &Reply::Health(h))?;
            }
            Request::Run(req) => {
                cancel.store(false, Ordering::SeqCst);
                let started = Instant::now();
                let reply = match protocol::check_blobs(&req.blob_refs(), &f.blobs) {
                    Err(e) => Err(EngineError::new(ErrorKind::BadRequest, e)),
                    Ok(()) => {
                        let w = Arc::clone(&writer);
                        let mut progress = |fraction: f32, message: &str| {
                            let _ = send(&w, &Reply::Progress { fraction, message: message.to_string() });
                        };
                        if req.preview.is_off() {
                            engine.run(&req, &f.blobs, &mut progress, &cancel)
                        } else {
                            let w = Arc::clone(&writer);
                            let mut preview = |p: Preview| {
                                let _ = send_preview(&w, p);
                            };
                            engine.run_previewed(&req, &f.blobs, &mut progress, &mut preview, &cancel)
                        }
                    }
                };
                match reply {
                    Ok((mut info, images)) => {
                        info.elapsed_ms = started.elapsed().as_millis() as u64;
                        let mut w = writer.lock().unwrap_or_else(|e| e.into_inner());
                        for bytes in &images {
                            protocol::write_frame(&mut *w, protocol::TAG_BLOB, bytes)?;
                        }
                        write_json(&mut *w, &Reply::Done(info))?;
                    }
                    Err(e) => send(&writer, &Reply::Error { kind: e.kind, message: e.message })?,
                }
            }
            // A Cancel with nothing running is a no-op (the flag is reset by the next run).
            Request::Cancel => {}
            Request::Free => {
                engine.free();
                send(&writer, &Reply::Freed)?;
            }
            Request::Index { dir } => {
                let v = engine.index(&dir);
                send(&writer, &Reply::Index(v))?;
            }
            Request::Quit => {
                let _ = send(&writer, &Reply::Bye);
                return Ok(Ended::Quit);
            }
        }
    }
    Ok(Ended::Left)
}

fn send(writer: &Arc<Mutex<TcpStream>>, reply: &Reply) -> std::io::Result<()> {
    let mut w = writer.lock().unwrap_or_else(|e| e.into_inner());
    write_json(&mut *w, reply)?;
    w.flush()
}

/// A preview as its blob frame and the message naming it; one with the wrong byte count is
/// dropped (a preview never fails a run).
fn send_preview(writer: &Arc<Mutex<TcpStream>>, p: Preview) -> std::io::Result<()> {
    let image = Blob { width: p.width, height: p.height, channels: 4, index: 0 };
    if p.rgba.len() != image.byte_len() || p.rgba.is_empty() {
        return Ok(());
    }
    let mut w = writer.lock().unwrap_or_else(|e| e.into_inner());
    protocol::write_frame(&mut *w, protocol::TAG_BLOB, &p.rgba)?;
    write_json(&mut *w, &Reply::Preview(PreviewInfo { step: p.step, steps: p.steps, output: p.output, image }))?;
    w.flush()
}

/// A solid RGBA picture (helpers for engines that return placeholders or tests).
pub fn solid(width: u32, height: u32, rgba: [u8; 4]) -> (Blob, Vec<u8>) {
    let n = width as usize * height as usize;
    let mut bytes = Vec::with_capacity(n * 4);
    for _ in 0..n {
        bytes.extend_from_slice(&rgba);
    }
    (Blob { width, height, channels: 4, index: 0 }, bytes)
}
