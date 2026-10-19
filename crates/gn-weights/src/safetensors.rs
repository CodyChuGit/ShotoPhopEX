//! Safetensors: an 8-byte little-endian header length, a JSON header (`{"name": {"dtype",
//! "shape", "data_offsets": [begin, end]}, "__metadata__": {…}}`), then the tensor bytes. Only
//! the header is read on open; tensors are read in ranges on demand, each range checked
//! against the file first.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::dtype::DType;
use crate::io::RangeFile;
use crate::limits;

/// One tensor as the header describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Byte range inside the data section.
    pub begin: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.shape.iter().try_fold(1u64, |a, d| a.checked_mul(*d)).unwrap_or(u64::MAX)
    }

    pub fn bytes(&self) -> u64 {
        self.end - self.begin
    }
}

/// An open safetensors file: the header parsed and checked, the data read on demand.
#[derive(Debug)]
pub struct SafeTensors {
    path: PathBuf,
    file: RangeFile,
    data_start: u64,
    tensors: Vec<TensorInfo>,
    by_name: BTreeMap<String, usize>,
    metadata: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl SafeTensors {
    /// Open and check `path`. A file whose header lies about any length is refused with a
    /// sentence; nothing is read past the header until a tensor is asked for.
    pub fn open(path: &Path) -> crate::Result<Self> {
        let file = RangeFile::open(path)?;
        let name = path.display();
        let head: [u8; 8] = file.read_at(0, 8).map_err(|_| format!("{name}: shorter than a safetensors header"))?.try_into().unwrap_or([0; 8]);
        let header_len = u64::from_le_bytes(head);
        if header_len > limits::SAFETENSORS_HEADER || header_len.saturating_add(8) > file.len {
            return Err(format!("{name}: the header claims {header_len} bytes in a {}-byte file", file.len));
        }
        let header = file.read_at(8, header_len)?;
        let data_start = 8 + header_len;
        let data_len = file.len - data_start;
        let (tensors, metadata) = parse_header(&name.to_string(), &header, data_len)?;
        let by_name = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(Self { path: path.to_path_buf(), file, data_start, tensors, by_name, metadata })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).and_then(|i| self.tensors.get(*i))
    }

    /// A tensor's bytes, read from the file. Unknown types are refused here, not at open, so a
    /// file with one exotic tensor still serves the rest.
    pub fn bytes(&self, name: &str) -> crate::Result<Vec<u8>> {
        let t = self.info(name).ok_or_else(|| format!("{}: no tensor `{name}`", self.path.display()))?;
        if matches!(t.dtype, DType::Unknown(_)) {
            return Err(format!("{}: tensor `{name}` has an unknown type {}", self.path.display(), t.dtype));
        }
        self.file.read_at(self.data_start + t.begin, t.bytes())
    }

    /// The data section's size.
    pub fn data_len(&self) -> u64 {
        self.file.len - self.data_start
    }
}

/// The JSON header → tensor infos (checked against `data_len`) and the metadata strings.
pub fn parse_header(name: &str, header: &[u8], data_len: u64) -> crate::Result<(Vec<TensorInfo>, BTreeMap<String, String>)> {
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header).map_err(|e| format!("{name}: the header is not JSON: {e}"))?;
    if raw.len() as u64 > limits::TENSORS {
        return Err(format!("{name}: {} header entries", raw.len()));
    }
    let mut tensors = Vec::with_capacity(raw.len());
    let mut metadata = BTreeMap::new();
    for (key, value) in raw {
        if key == "__metadata__" {
            if let Some(obj) = value.as_object() {
                metadata = obj.iter().map(|(k, v)| (k.clone(), v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect();
            }
            continue;
        }
        let t: RawTensor = serde_json::from_value(value).map_err(|e| format!("{name}: tensor `{key}`: {e}"))?;
        if t.shape.len() > limits::DIMS {
            return Err(format!("{name}: tensor `{key}` has {} dimensions", t.shape.len()));
        }
        let [begin, end] = t.data_offsets;
        if begin > end || end > data_len {
            return Err(format!("{name}: tensor `{key}` lies outside the data ({begin}..{end} of {data_len})"));
        }
        let info = TensorInfo { name: key, dtype: DType::from_safetensors(&t.dtype), shape: t.shape, begin, end };
        // A known type's byte count must match its shape; unknown types are only indexed.
        if let Some(want) = info.dtype.byte_len(info.elements())
            && want != info.bytes()
        {
            return Err(format!("{name}: tensor `{}` is {} bytes for {} × {} elements", info.name, info.bytes(), info.dtype, info.elements()));
        }
        tensors.push(info);
    }
    Ok((tensors, metadata))
}

/// Write a safetensors file from `(name, dtype, shape, bytes)` tensors (tests, fixtures,
/// converted outputs).
pub fn write(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])], metadata: &BTreeMap<String, String>) -> crate::Result<()> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".into(), serde_json::to_value(metadata).map_err(|e| e.to_string())?);
    }
    let mut offset = 0u64;
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let end = offset + bytes.len() as u64;
        header.insert((*name).to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}));
        data.extend_from_slice(bytes);
        offset = end;
    }
    let mut text = serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|e| e.to_string())?;
    // The header is padded to 8 bytes with spaces, as the reference writer does.
    while text.len() % 8 != 0 {
        text.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + text.len() + data.len());
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(&text);
    out.extend_from_slice(&data);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gn-weights-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn raw(dir: &Path, header: &[u8], data: &[u8]) -> PathBuf {
        let p = dir.join("raw.safetensors");
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header);
        out.extend_from_slice(data);
        std::fs::write(&p, &out).unwrap();
        p
    }

    #[test]
    fn a_written_file_reads_back_and_checks_its_lengths() {
        let d = temp("st");
        let p = d.join("a.safetensors");
        let w: Vec<u8> = (0..24u8).collect();
        let s: Vec<u8> = 1.5f32.to_le_bytes().to_vec();
        let mut meta = BTreeMap::new();
        meta.insert("format".to_string(), "pt".to_string());
        write(&p, &[("blocks.0.weight", "F8_E4M3", &[4, 6], &w), ("blocks.0.scale_weight", "F32", &[1], &s)], &meta).unwrap();
        let st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.tensors().len(), 2);
        assert_eq!(st.metadata().get("format").map(String::as_str), Some("pt"));
        let t = st.info("blocks.0.weight").unwrap();
        assert_eq!((t.dtype.clone(), t.shape.clone(), t.elements(), t.bytes()), (DType::F8E4M3, vec![4, 6], 24, 24));
        assert_eq!(st.bytes("blocks.0.weight").unwrap(), w);
        assert_eq!(st.bytes("blocks.0.scale_weight").unwrap(), s);
        assert!(st.bytes("nope").is_err());
        assert_eq!(st.data_len(), 28);
        // A header that lies: offsets past the data; a shape that does not match the bytes.
        let p = raw(&d, br#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,16]}}"#, &[0u8; 8]);
        assert!(SafeTensors::open(&p).unwrap_err().contains("outside the data"));
        let p = raw(&d, br#"{"t":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#, &[0u8; 8]);
        assert!(SafeTensors::open(&p).unwrap_err().contains("8 bytes for"));
        // A giant header length is refused before anything is allocated.
        let p = d.join("giant.safetensors");
        std::fs::write(&p, (u64::MAX / 2).to_le_bytes()).unwrap();
        assert!(SafeTensors::open(&p).unwrap_err().contains("claims"));
        // An unknown dtype is indexed but not served.
        let p = raw(&d, br#"{"t":{"dtype":"C64","shape":[1],"data_offsets":[0,8]}}"#, &[0u8; 8]);
        let st = SafeTensors::open(&p).unwrap();
        assert!(matches!(st.info("t").unwrap().dtype, DType::Unknown(_)));
        assert!(st.bytes("t").unwrap_err().contains("unknown type"));
        let _ = std::fs::remove_dir_all(d);
    }
}
