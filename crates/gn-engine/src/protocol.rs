//! Messages and frames.
//!
//! A **frame** is `u32 length (LE) | u8 tag | bytes`. Tag 0 carries one JSON message; tag 1
//! carries a **blob** (raw pixels). Blobs precede the JSON message that refers to them by
//! index, in order, and the index resets with every JSON message. A frame longer than
//! [`MAX_FRAME`] is refused (a 1 MP RGBA picture is 4 MB; 16 K × 16 K would be 1 GB).

use std::collections::BTreeMap;
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Protocol version both sides must agree on in `Hello`.
pub const VERSION: u32 = 1;
/// The largest frame either side accepts (256 MB: a 64 MP RGBA picture).
pub const MAX_FRAME: u32 = 256 * 1024 * 1024;
pub const TAG_JSON: u8 = 0;
pub const TAG_BLOB: u8 = 1;

/// Raw pixels carried in a blob frame: straight RGBA8 (`channels` 4) or coverage (`channels`
/// 1), row-major, `width * height * channels` bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    /// Which blob frame before this message holds the bytes.
    pub index: u32,
}

impl Blob {
    pub fn byte_len(&self) -> usize {
        self.width as usize * self.height as usize * self.channels as usize
    }
}

/// One generation, engine-agnostic (mirrors the editor's `Request`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRequest {
    /// What to run: a template id the engine knows (`krea2-turbo/image`, `qwen-2.1/edit`,
    /// `sam3.1/segment-point`, …).
    pub model: String,
    /// `fill | expand | image | edit | matte | segment | layers | text`.
    pub task: String,
    pub prompt: String,
    #[serde(default)]
    pub negative: String,
    pub seed: u64,
    #[serde(default)]
    pub steps: u32,
    #[serde(default)]
    pub guidance: f32,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    #[serde(default)]
    pub image: Option<Blob>,
    #[serde(default)]
    pub mask: Option<Blob>,
    #[serde(default)]
    pub references: Vec<Blob>,
    #[serde(default)]
    pub params: BTreeMap<String, Value>,
}

/// What the editor sends.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Request {
    Hello { version: u32, token: String },
    Health,
    Run(RunRequest),
    Cancel,
    Free,
    Index { dir: String },
    Quit,
}

/// The engine's state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Health {
    pub ok: bool,
    pub engine: String,
    pub device: String,
    pub vram_total: Option<u64>,
    pub vram_free: Option<u64>,
    /// Models loaded right now.
    #[serde(default)]
    pub resident: Vec<String>,
    /// Why the engine is not healthy, or what it refused, as a sentence for the editor (a card
    /// the build has no kernels for, the last load that failed); empty when there is nothing to say.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

/// A finished run; its pictures travel as blob frames before this message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DoneInfo {
    pub images: Vec<Blob>,
    #[serde(default)]
    pub texts: Vec<String>,
    pub seed: u64,
    pub elapsed_ms: u64,
}

/// Why a run failed, for the editor's failure taxonomy (plan §12.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorKind {
    Cancelled,
    /// The engine does not serve this model or task.
    Unsupported,
    ModelMissing,
    ModelUnreadable,
    OutOfMemory,
    BadRequest,
    Internal,
}

/// What the engine sends.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Reply {
    Hello { version: u32, engine: String },
    Health(Health),
    Progress { fraction: f32, message: String },
    Done(DoneInfo),
    Error { kind: ErrorKind, message: String },
    Freed,
    Index(Value),
    Bye,
}

/// Write one frame.
pub fn write_frame(w: &mut impl Write, tag: u8, bytes: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(bytes.len()).map_err(|_| std::io::Error::other("frame too large"))?;
    if len > MAX_FRAME {
        return Err(std::io::Error::other(format!("frame of {len} bytes exceeds {MAX_FRAME}")));
    }
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&[tag])?;
    w.write_all(bytes)?;
    w.flush()
}

/// Read one frame: its tag and bytes. `Ok(None)` at a clean end of stream.
pub fn read_frame(r: &mut impl Read) -> std::io::Result<Option<(u8, Vec<u8>)>> {
    let mut head = [0u8; 5];
    match r.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
    if len > MAX_FRAME {
        return Err(std::io::Error::other(format!("frame of {len} bytes exceeds {MAX_FRAME}")));
    }
    let mut bytes = vec![0u8; len as usize];
    r.read_exact(&mut bytes)?;
    Ok(Some((head[4], bytes)))
}

/// Write a JSON message (any serialisable) as a frame.
pub fn write_json<T: Serialize>(w: &mut impl Write, msg: &T) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(msg).map_err(std::io::Error::other)?;
    write_frame(w, TAG_JSON, &bytes)
}

/// A message with the blobs that preceded it.
#[derive(Debug)]
pub struct Framed<T> {
    pub message: T,
    pub blobs: Vec<Vec<u8>>,
}

/// Read blob frames until a JSON frame arrives, decode it as `T`. `Ok(None)` at a clean end of
/// stream.
pub fn read_message<T: for<'de> Deserialize<'de>>(r: &mut impl Read) -> std::io::Result<Option<Framed<T>>> {
    let mut blobs = Vec::new();
    loop {
        match read_frame(r)? {
            None => return Ok(None),
            Some((TAG_BLOB, bytes)) => blobs.push(bytes),
            Some((TAG_JSON, bytes)) => {
                let message: T = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
                return Ok(Some(Framed { message, blobs }));
            }
            Some((tag, _)) => return Err(std::io::Error::other(format!("unknown frame tag {tag}"))),
        }
    }
}

/// Check that every blob a message names exists and has the bytes its size promises.
pub fn check_blobs(refs: &[&Blob], blobs: &[Vec<u8>]) -> Result<(), String> {
    for b in refs {
        let bytes = blobs.get(b.index as usize).ok_or_else(|| format!("blob {} is missing", b.index))?;
        if bytes.len() != b.byte_len() {
            return Err(format!("blob {} is {} bytes for {}×{}×{}", b.index, bytes.len(), b.width, b.height, b.channels));
        }
    }
    Ok(())
}

impl RunRequest {
    /// Every blob the request names (image, mask, references).
    pub fn blob_refs(&self) -> Vec<&Blob> {
        self.image.iter().chain(self.mask.iter()).chain(self.references.iter()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_and_messages_round_trip() {
        let mut buf = Vec::new();
        write_frame(&mut buf, TAG_BLOB, &[1, 2, 3, 4]).unwrap();
        let req = Request::Run(RunRequest {
            model: "krea2-turbo/image".into(),
            task: "image".into(),
            prompt: "a fox".into(),
            negative: String::new(),
            seed: 7,
            steps: 0,
            guidance: 0.0,
            width: 2,
            height: 1,
            image: Some(Blob { width: 1, height: 1, channels: 4, index: 0 }),
            mask: None,
            references: Vec::new(),
            params: BTreeMap::new(),
        });
        write_json(&mut buf, &req).unwrap();
        let mut r = &buf[..];
        let f: Framed<Request> = read_message(&mut r).unwrap().unwrap();
        assert_eq!(f.message, req);
        assert_eq!(f.blobs, vec![vec![1, 2, 3, 4]]);
        assert!(read_message::<Request>(&mut r).unwrap().is_none(), "clean end of stream");
        if let Request::Run(rr) = &f.message {
            assert!(check_blobs(&rr.blob_refs(), &f.blobs).is_ok());
            assert!(check_blobs(&rr.blob_refs(), &[vec![1, 2]]).unwrap_err().contains("2 bytes"));
        }
        // An oversized frame is refused at both ends.
        let mut big = (MAX_FRAME + 1).to_le_bytes().to_vec();
        big.push(TAG_JSON);
        assert!(read_frame(&mut &big[..]).is_err());
    }
}
