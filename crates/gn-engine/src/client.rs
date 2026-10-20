//! The editor's side: start a worker (or connect to one), greet it, run requests with progress
//! and cancellation, notice when it died.
//!
//! Reads go through a [`FrameReader`] that keeps partial bytes between polls, so the run loop
//! can poll for cancellation on a read timeout without ever losing part of a frame (a 4 MB
//! picture is many reads).

use std::io::{BufRead, BufReader, Read};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::protocol::{self, Blob, DoneInfo, ErrorKind, Framed, Health, Reply, Request, RunRequest, write_json};

/// Why a call failed on the editor's side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientError {
    /// The engine answered with an error (its kind and sentence).
    Engine { kind: ErrorKind, message: String },
    /// The channel broke: the worker died or hung up.
    Gone(String),
    /// The worker could not be started or greeted.
    Start(String),
    /// The worker answered with the wrong message for the call.
    Protocol(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Engine { message, .. } => f.write_str(message),
            Self::Gone(m) => write!(f, "the engine stopped: {m}"),
            Self::Start(m) => write!(f, "the engine could not start: {m}"),
            Self::Protocol(m) => write!(f, "the engine answered unexpectedly: {m}"),
        }
    }
}

/// Reads frames with `read` (never `read_exact`), keeping what arrived so far, so a timed-out
/// read loses nothing.
#[derive(Default)]
pub struct FrameReader {
    head: [u8; 5],
    head_filled: usize,
    body: Vec<u8>,
    body_filled: usize,
    /// Blob frames waiting for their JSON message.
    blobs: Vec<Vec<u8>>,
}

/// What one poll produced.
pub enum Polled<T> {
    /// A whole message (with its blobs).
    Message(Framed<T>),
    /// Nothing complete yet (the read timed out or would block).
    Pending,
    /// The peer closed the stream cleanly.
    Closed,
}

/// One frame's tag and bytes.
type Frame = (u8, Vec<u8>);

impl FrameReader {
    /// Pull bytes from `r` until a frame completes or the read would block: `Ok(None)` =
    /// would block, `Ok(Some(None))` = the stream closed, `Ok(Some(Some(frame)))` = a frame.
    fn poll_frame(&mut self, r: &mut impl Read) -> std::io::Result<Option<Option<Frame>>> {
        loop {
            if self.head_filled < 5 {
                match r.read(&mut self.head[self.head_filled..]) {
                    Ok(0) => return Ok(Some(None)),
                    Ok(n) => {
                        self.head_filled += n;
                        if self.head_filled == 5 {
                            let len = u32::from_le_bytes([self.head[0], self.head[1], self.head[2], self.head[3]]);
                            if len > protocol::MAX_FRAME {
                                return Err(std::io::Error::other(format!("frame of {len} bytes exceeds {}", protocol::MAX_FRAME)));
                            }
                            self.body = vec![0u8; len as usize];
                            self.body_filled = 0;
                        }
                    }
                    Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted) => {
                        return Ok(None);
                    }
                    Err(e) => return Err(e),
                }
            } else if self.body_filled < self.body.len() {
                match r.read(&mut self.body[self.body_filled..]) {
                    Ok(0) => return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "the stream closed inside a frame")),
                    Ok(n) => self.body_filled += n,
                    Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted) => {
                        return Ok(None);
                    }
                    Err(e) => return Err(e),
                }
            } else {
                let tag = self.head[4];
                let body = std::mem::take(&mut self.body);
                self.head_filled = 0;
                self.body_filled = 0;
                return Ok(Some(Some((tag, body))));
            }
        }
    }

    /// Pull frames until a JSON message completes (its blobs collected), the read would block,
    /// or the stream closes.
    pub fn poll<T: for<'de> serde::Deserialize<'de>>(&mut self, r: &mut impl Read) -> std::io::Result<Polled<T>> {
        loop {
            match self.poll_frame(r)? {
                None => return Ok(Polled::Pending),
                Some(None) => return Ok(Polled::Closed),
                Some(Some((protocol::TAG_BLOB, bytes))) => self.blobs.push(bytes),
                Some(Some((protocol::TAG_JSON, bytes))) => {
                    let message: T = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
                    return Ok(Polled::Message(Framed { message, blobs: std::mem::take(&mut self.blobs) }));
                }
                Some(Some((tag, _))) => return Err(std::io::Error::other(format!("unknown frame tag {tag}"))),
            }
        }
    }
}

/// A connection to one worker, and the child process when we started it.
pub struct EngineClient {
    reader: TcpStream,
    frames: FrameReader,
    writer: Arc<Mutex<TcpStream>>,
    child: Option<Child>,
    pub engine: String,
    pub port: u16,
}

/// How long a spawned worker may take to print its port.
pub const START_TIMEOUT: Duration = Duration::from_secs(30);
/// How often the run loop looks at `cancel` while the engine is quiet.
const CANCEL_POLL: Duration = Duration::from_millis(200);
/// How long the greeting may take.
const HELLO_TIMEOUT: Duration = Duration::from_secs(10);

impl EngineClient {
    /// Start `program args…` as the worker: it receives the token in `GN_ENGINE_TOKEN`, prints
    /// its port line first, and is greeted.
    pub fn spawn(program: &Path, args: &[String], token: &str) -> Result<Self, ClientError> {
        Self::spawn_with_env(program, args, token, &[])
    }

    /// [`Self::spawn`] with extra environment variables for the worker (a `PATH` that holds the
    /// CUDA libraries an engine links, for one).
    pub fn spawn_with_env(program: &Path, args: &[String], token: &str, env: &[(String, String)]) -> Result<Self, ClientError> {
        let mut cmd = Command::new(program);
        cmd.args(args).env("GN_ENGINE_TOKEN", token).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        for (k, v) in env {
            cmd.env(k, v);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().map_err(|e| ClientError::Start(format!("{}: {e}", program.display())))?;
        let Some(stdout) = child.stdout.take() else {
            let _ = child.kill();
            return Err(ClientError::Start("no stdout".into()));
        };
        // The port line arrives within the timeout or the child is not a worker.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut lines = BufReader::new(stdout).lines();
            let first = lines.next().and_then(|l| l.ok());
            let _ = tx.send(first);
            // Keep draining so the child never blocks on a full pipe.
            for _ in lines.by_ref() {}
        });
        let port = match rx.recv_timeout(START_TIMEOUT) {
            Ok(Some(line)) => crate::server::parse_port_line(&line).ok_or_else(|| ClientError::Start(format!("unexpected first line: {line}"))),
            Ok(None) => Err(ClientError::Start("the worker exited before announcing its port".into())),
            Err(_) => Err(ClientError::Start(format!("no port line within {} s", START_TIMEOUT.as_secs()))),
        };
        let port = match port {
            Ok(p) => p,
            Err(e) => {
                let _ = child.kill();
                return Err(e);
            }
        };
        let mut c = Self::connect(port, token)?;
        c.child = Some(child);
        Ok(c)
    }

    /// Connect to a worker already listening on `port`.
    pub fn connect(port: u16, token: &str) -> Result<Self, ClientError> {
        let stream = TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], port)), Duration::from_secs(5))
            .map_err(|e| ClientError::Start(format!("connect to port {port}: {e}")))?;
        let _ = stream.set_nodelay(true);
        let reader = stream.try_clone().map_err(|e| ClientError::Start(e.to_string()))?;
        let writer = Arc::new(Mutex::new(stream));
        let mut c = Self { reader, frames: FrameReader::default(), writer, child: None, engine: String::new(), port };
        c.send(&Request::Hello { version: protocol::VERSION, token: token.to_string() })?;
        // A greeting that does not come within the timeout is a worker that is not one.
        let _ = c.reader.set_read_timeout(Some(HELLO_TIMEOUT));
        let hello = c.recv().map_err(|e| ClientError::Start(format!("no greeting: {e}")));
        let _ = c.reader.set_read_timeout(None);
        match hello?.message {
            Reply::Hello { engine, .. } => c.engine = engine,
            Reply::Error { message, .. } => return Err(ClientError::Start(message)),
            other => return Err(ClientError::Protocol(format!("{other:?}"))),
        }
        Ok(c)
    }

    fn send(&self, req: &Request) -> Result<(), ClientError> {
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        write_json(&mut *w, req).map_err(|e| ClientError::Gone(e.to_string()))
    }

    /// The next message, blocking (with whatever read timeout is set: none outside the
    /// handshake, so `Pending` only happens when a timeout is).
    fn recv(&mut self) -> Result<Framed<Reply>, ClientError> {
        let started = Instant::now();
        loop {
            match self.frames.poll::<Reply>(&mut self.reader) {
                Ok(Polled::Message(f)) => return Ok(f),
                Ok(Polled::Pending) => {
                    if self.reader.read_timeout().ok().flatten().is_some_and(|t| started.elapsed() >= t) {
                        return Err(ClientError::Gone("no answer in time".into()));
                    }
                }
                Ok(Polled::Closed) => return Err(ClientError::Gone("the connection closed".into())),
                Err(e) => return Err(ClientError::Gone(e.to_string())),
            }
        }
    }

    pub fn health(&mut self) -> Result<Health, ClientError> {
        self.send(&Request::Health)?;
        match self.recv()?.message {
            Reply::Health(h) => Ok(h),
            other => Err(ClientError::Protocol(format!("{other:?}"))),
        }
    }

    /// Run a request: `blobs` are the pixels the request's descriptors name, in index order;
    /// `progress` gets every report; `cancel` is polled while waiting and, when true, sends
    /// `Cancel` once. Returns the result's descriptors and pixels.
    pub fn run(
        &mut self,
        req: &RunRequest,
        blobs: &[Vec<u8>],
        progress: &mut dyn FnMut(f32, &str),
        cancel: &dyn Fn() -> bool,
    ) -> Result<(DoneInfo, Vec<Vec<u8>>), ClientError> {
        protocol::check_blobs(&req.blob_refs(), blobs).map_err(|m| ClientError::Engine { kind: ErrorKind::BadRequest, message: m })?;
        {
            let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
            for b in blobs {
                protocol::write_frame(&mut *w, protocol::TAG_BLOB, b).map_err(|e| ClientError::Gone(e.to_string()))?;
            }
            write_json(&mut *w, &Request::Run(req.clone())).map_err(|e| ClientError::Gone(e.to_string()))?;
        }
        let mut sent_cancel = false;
        let _ = self.reader.set_read_timeout(Some(CANCEL_POLL));
        let result = loop {
            if !sent_cancel && cancel() {
                sent_cancel = true;
                let _ = self.send(&Request::Cancel);
            }
            match self.frames.poll::<Reply>(&mut self.reader) {
                Ok(Polled::Message(f)) => match f.message {
                    Reply::Progress { fraction, message } => progress(fraction, &message),
                    Reply::Done(info) => break Ok((info, f.blobs)),
                    Reply::Error { kind, message } => break Err(ClientError::Engine { kind, message }),
                    other => break Err(ClientError::Protocol(format!("{other:?}"))),
                },
                Ok(Polled::Pending) => continue,
                Ok(Polled::Closed) => break Err(ClientError::Gone("the connection closed during the run".into())),
                Err(e) => break Err(ClientError::Gone(e.to_string())),
            }
        };
        let _ = self.reader.set_read_timeout(None);
        result
    }

    pub fn free(&mut self) -> Result<(), ClientError> {
        self.send(&Request::Free)?;
        match self.recv()?.message {
            Reply::Freed => Ok(()),
            other => Err(ClientError::Protocol(format!("{other:?}"))),
        }
    }

    pub fn index(&mut self, dir: &str) -> Result<Value, ClientError> {
        self.send(&Request::Index { dir: dir.to_string() })?;
        match self.recv()?.message {
            Reply::Index(v) => Ok(v),
            other => Err(ClientError::Protocol(format!("{other:?}"))),
        }
    }

    /// Ask the worker to exit and wait briefly for it; kill it if it does not.
    pub fn quit(mut self) {
        let _ = self.send(&Request::Quit);
        let t0 = Instant::now();
        if let Some(child) = self.child.as_mut() {
            while t0.elapsed() < Duration::from_secs(3) {
                if matches!(child.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// End the worker now.
    pub fn kill(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Is the worker we started still running? (`true` for a worker we only connected to.)
    pub fn alive(&mut self) -> bool {
        match self.child.as_mut() {
            Some(c) => matches!(c.try_wait(), Ok(None)),
            None => true,
        }
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }
}

impl Drop for EngineClient {
    fn drop(&mut self) {
        // A worker we started must not outlive its client.
        if let Some(child) = self.child.as_mut()
            && !matches!(child.try_wait(), Ok(Some(_)))
        {
            if let Ok(mut w) = self.writer.lock() {
                let _ = write_json(&mut *w, &Request::Quit);
            }
            std::thread::sleep(Duration::from_millis(100));
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

/// A descriptor for RGBA pixels of a picture about to be sent (index assigned by position).
pub fn rgba_blob(width: u32, height: u32, index: u32) -> Blob {
    Blob { width, height, channels: 4, index }
}

/// A random token for a spawned worker.
pub fn new_token() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut x = t as u64 ^ (std::process::id() as u64).rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = String::with_capacity(64);
    for _ in 0..4 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.push_str(&format!("{x:016x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader that hands out one byte per call and times out between them: frames still
    /// arrive whole.
    struct Trickle {
        data: Vec<u8>,
        pos: usize,
        tick: bool,
    }
    impl Read for Trickle {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.tick = !self.tick;
            if self.tick {
                return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "tick"));
            }
            match (self.data.get(self.pos), buf.first_mut()) {
                (Some(b), Some(out)) => {
                    *out = *b;
                    self.pos += 1;
                    Ok(1)
                }
                _ => Ok(0),
            }
        }
    }

    #[test]
    fn frames_survive_timeouts_mid_frame() {
        let mut data = Vec::new();
        protocol::write_frame(&mut data, protocol::TAG_BLOB, &[9u8; 300]).unwrap();
        protocol::write_json(&mut data, &Reply::Freed).unwrap();
        let mut r = Trickle { data, pos: 0, tick: false };
        let mut fr = FrameReader::default();
        let mut pendings = 0;
        loop {
            match fr.poll::<Reply>(&mut r).unwrap() {
                Polled::Pending => pendings += 1,
                Polled::Message(f) => {
                    assert_eq!(f.message, Reply::Freed);
                    assert_eq!(f.blobs, vec![vec![9u8; 300]]);
                    break;
                }
                Polled::Closed => panic!("closed early"),
            }
        }
        assert!(pendings > 100, "{pendings} timeouts happened mid-frame");
        assert!(matches!(fr.poll::<Reply>(&mut r).unwrap(), Polled::Pending | Polled::Closed));
    }
}
