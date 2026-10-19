//! Reading ranges of large files without mapping them (the loader forbids `unsafe`, and a
//! 20 GB repack must never be read whole just to look at its header).

use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

/// A file read in ranges; shared between threads behind a lock (one seek + read per call).
#[derive(Debug)]
pub struct RangeFile {
    file: Mutex<File>,
    pub len: u64,
    name: String,
}

impl RangeFile {
    pub fn open(path: &Path) -> crate::Result<Self> {
        let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let len = file.metadata().map(|m| m.len()).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { file: Mutex::new(file), len, name: path.display().to_string() })
    }

    /// Exactly `len` bytes at `offset`; a range past the end is an error, never a short read.
    pub fn read_at(&self, offset: u64, len: u64) -> crate::Result<Vec<u8>> {
        let end = offset.checked_add(len).ok_or_else(|| format!("{}: range overflow", self.name))?;
        if end > self.len {
            return Err(format!("{}: {offset}..{end} is past the end of the file ({})", self.name, self.len));
        }
        let mut buf = vec![0u8; len as usize];
        let mut f = self.file.lock().unwrap_or_else(|e| e.into_inner());
        f.seek(SeekFrom::Start(offset)).map_err(|e| format!("{}: seek: {e}", self.name))?;
        f.read_exact(&mut buf).map_err(|e| format!("{}: read: {e}", self.name))?;
        Ok(buf)
    }

    /// The first `len` bytes (or the whole file when shorter).
    pub fn prefix(&self, len: u64) -> crate::Result<Vec<u8>> {
        self.read_at(0, len.min(self.len))
    }
}
