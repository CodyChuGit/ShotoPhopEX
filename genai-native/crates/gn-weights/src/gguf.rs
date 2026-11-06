//! GGUF (ggml's container; versions 2 and 3): magic, version, tensor and metadata counts, the
//! metadata key-values, the tensor infos (name, dims, ggml type, offset), then the tensor data
//! aligned to `general.alignment` (32 by default). Only the header is read on open (in growing
//! prefixes, since its size is not written anywhere); tensor bytes are read in checked ranges.
//! Dequantisation belongs to the kernels.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::dtype::DType;
use crate::io::RangeFile;
use crate::limits;

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// An array's element type name and its length; the elements themselves are not kept (a
    /// tokenizer's vocabulary is six figures of strings nobody loading weights needs).
    Array(String, u64),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Self::U8(v) => u64::from(*v),
            Self::U16(v) => u64::from(*v),
            Self::U32(v) => u64::from(*v),
            Self::U64(v) => *v,
            Self::I8(v) if *v >= 0 => *v as u64,
            Self::I16(v) if *v >= 0 => *v as u64,
            Self::I32(v) if *v >= 0 => *v as u64,
            Self::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One tensor as the header describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    /// ggml order (innermost first).
    pub dims: Vec<u64>,
    /// Byte offset from the start of the data section.
    pub offset: u64,
    /// Bytes the tensor takes (`None` for a type whose size this crate does not know).
    pub bytes: Option<u64>,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().try_fold(1u64, |a, d| a.checked_mul(*d)).unwrap_or(u64::MAX)
    }
}

/// The parsed header.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub version: u32,
    pub alignment: u64,
    /// Where the data section starts (the header's end rounded up to the alignment).
    pub data_start: u64,
    pub tensors: Vec<TensorInfo>,
    pub metadata: BTreeMap<String, Value>,
}

/// An open GGUF file.
#[derive(Debug)]
pub struct Gguf {
    path: PathBuf,
    file: RangeFile,
    pub header: Header,
    by_name: BTreeMap<String, usize>,
}

/// The parse stopped because the prefix ended before the header did.
const NEED_MORE: &str = "header ends early";

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    name: &'a str,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> crate::Result<&[u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| format!("{}: header overflow", self.name))?;
        let s = self.bytes.get(self.pos..end).ok_or_else(|| format!("{}: {NEED_MORE}", self.name))?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> crate::Result<u8> {
        Ok(self.take(1)?.first().copied().unwrap_or(0))
    }
    fn u16(&mut self) -> crate::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap_or([0; 2])))
    }
    fn u32(&mut self) -> crate::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap_or([0; 4])))
    }
    fn u64(&mut self) -> crate::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap_or([0; 8])))
    }
    fn string(&mut self) -> crate::Result<String> {
        let n = self.u64()?;
        if n > limits::GGUF_STRING {
            return Err(format!("{}: a {n}-byte string in the header", self.name));
        }
        Ok(String::from_utf8_lossy(self.take(n as usize)?).into_owned())
    }
    /// A value of GGUF type `ty`; arrays are walked element by element and summarised.
    fn value(&mut self, ty: u32) -> crate::Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem = self.u32()?;
                let n = self.u64()?;
                if n > limits::GGUF_ARRAY {
                    return Err(format!("{}: an array of {n} in the header", self.name));
                }
                for _ in 0..n {
                    self.value(elem)?;
                }
                Value::Array(type_name(elem).to_string(), n)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(format!("{}: unknown metadata value type {other}", self.name)),
        })
    }
}

fn type_name(ty: u32) -> &'static str {
    match ty {
        0 => "u8",
        1 => "i8",
        2 => "u16",
        3 => "i16",
        4 => "u32",
        5 => "i32",
        6 => "f32",
        7 => "bool",
        8 => "string",
        9 => "array",
        10 => "u64",
        11 => "i64",
        12 => "f64",
        _ => "?",
    }
}

/// Parse a header from the file's first bytes. `Err` containing [`NEED_MORE`] means the prefix
/// was too short.
pub fn parse_header(name: &str, prefix: &[u8]) -> crate::Result<Header> {
    let mut c = Cursor { bytes: prefix, pos: 0, name };
    if c.take(4)? != b"GGUF" {
        return Err(format!("{name}: not a GGUF file"));
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("{name}: GGUF version {version} is not supported (2 and 3 are)"));
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    if n_tensors > limits::TENSORS || n_kv > limits::GGUF_KV {
        return Err(format!("{name}: {n_tensors} tensors and {n_kv} metadata entries claimed"));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        let v = c.value(ty)?;
        metadata.insert(key, v);
    }
    let alignment = metadata.get("general.alignment").and_then(Value::as_u64).filter(|a| *a > 0 && a.is_power_of_two()).unwrap_or(32);
    let mut tensors = Vec::with_capacity(n_tensors.min(4096) as usize);
    for _ in 0..n_tensors {
        let tname = c.string()?;
        let nd = c.u32()? as usize;
        if nd > limits::DIMS {
            return Err(format!("{name}: tensor `{tname}` has {nd} dimensions"));
        }
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(c.u64()?);
        }
        let dtype = DType::from_ggml(c.u32()?);
        let offset = c.u64()?;
        let mut info = TensorInfo { name: tname, dtype, dims, offset, bytes: None };
        info.bytes = info.dtype.byte_len(info.elements());
        tensors.push(info);
    }
    let header_end = c.pos as u64;
    let data_start = header_end.div_ceil(alignment) * alignment;
    Ok(Header { version, alignment, data_start, tensors, metadata })
}

impl Gguf {
    /// Open `path`: the header is read in growing prefixes (1 MB, 4 MB, … up to the cap) and
    /// every tensor's range checked against the file's length.
    pub fn open(path: &Path) -> crate::Result<Self> {
        let file = RangeFile::open(path)?;
        let name = path.display().to_string();
        let mut want = 1u64 << 20;
        let header = loop {
            let prefix = file.prefix(want)?;
            match parse_header(&name, &prefix) {
                Ok(h) => break h,
                Err(e) if e.contains(NEED_MORE) && want < file.len && want < limits::SAFETENSORS_HEADER => want *= 4,
                Err(e) => return Err(e),
            }
        };
        Self::checked(path, file, header)
    }

    /// A header parsed elsewhere against the file it came from.
    fn checked(path: &Path, file: RangeFile, header: Header) -> crate::Result<Self> {
        let name = path.display();
        if header.data_start > file.len {
            return Err(format!("{name}: the data section starts past the end of the file"));
        }
        let data_len = file.len - header.data_start;
        for t in &header.tensors {
            if let Some(b) = t.bytes
                && t.offset.checked_add(b).is_none_or(|end| end > data_len)
            {
                return Err(format!("{name}: tensor `{}` lies outside the data ({} + {b} of {data_len})", t.name, t.offset));
            }
        }
        let by_name = header.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(Self { path: path.to_path_buf(), file, header, by_name })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.header.tensors
    }
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.header.metadata
    }
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).and_then(|i| self.header.tensors.get(*i))
    }

    /// `general.architecture`, when the file says.
    pub fn architecture(&self) -> Option<&str> {
        self.header.metadata.get("general.architecture").and_then(Value::as_str)
    }

    /// A tensor's raw (possibly quantised) bytes.
    pub fn bytes(&self, name: &str) -> crate::Result<Vec<u8>> {
        let t = self.info(name).ok_or_else(|| format!("{}: no tensor `{name}`", self.path.display()))?;
        let b = t.bytes.ok_or_else(|| format!("{}: tensor `{name}` has a type of unknown size ({})", self.path.display(), t.dtype))?;
        self.file.read_at(self.header.data_start + t.offset, b)
    }
}

/// Write a minimal GGUF v3 file (tests and fixtures): string, u32 and f32 metadata; tensors as
/// raw bytes with their ggml type id, each padded to the alignment.
pub fn write(path: &Path, metadata: &[(&str, Value)], tensors: &[(&str, &[u64], u32, &[u8])]) -> crate::Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    let put_str = |out: &mut Vec<u8>, s: &str| {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };
    for (k, v) in metadata {
        put_str(&mut out, k);
        match v {
            Value::U32(x) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            Value::Str(s) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut out, s);
            }
            Value::F32(x) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            other => return Err(format!("gguf::write: unsupported metadata value {other:?}")),
        }
    }
    let alignment = 32u64;
    let mut offset = 0u64;
    let mut data = Vec::new();
    for (name, dims, ty, bytes) in tensors {
        put_str(&mut out, name);
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in *dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        data.extend_from_slice(bytes);
        offset += bytes.len() as u64;
        while !offset.is_multiple_of(alignment) {
            data.push(0);
            offset += 1;
        }
    }
    while !(out.len() as u64).is_multiple_of(alignment) {
        out.push(0);
    }
    out.extend_from_slice(&data);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_gguf_reads_back_with_its_blocks_checked() {
        let d = std::env::temp_dir().join(format!("gn-gguf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let p = d.join("a.gguf");
        let q8: Vec<u8> = (0..68u8).collect(); // two Q8_0 blocks of 32 elements
        let f32s: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        write(
            &p,
            &[("general.architecture", Value::Str("krea2".into())), ("general.alignment", Value::U32(32))],
            &[("blk.0.w", &[64], 8, &q8), ("blk.0.b", &[2], 0, &f32s)],
        )
        .unwrap();
        let g = Gguf::open(&p).unwrap();
        assert_eq!((g.header.version, g.architecture()), (3, Some("krea2")));
        assert_eq!(g.tensors().len(), 2);
        let w = g.info("blk.0.w").unwrap();
        assert_eq!((w.dtype.to_string(), w.elements(), w.bytes), ("Q8_0".into(), 64, Some(68)));
        assert_eq!(g.bytes("blk.0.w").unwrap(), q8);
        assert_eq!(g.bytes("blk.0.b").unwrap(), f32s);
        // A tensor that claims more than the file holds is refused at open.
        let short = d.join("short.gguf");
        write(&short, &[], &[("t", &[1024], 0, &[0u8; 4096])]).unwrap();
        let mut raw = std::fs::read(&short).unwrap();
        raw.truncate(raw.len() - 40);
        std::fs::write(&short, &raw).unwrap();
        assert!(Gguf::open(&short).unwrap_err().contains("outside the data"));
        let bad = d.join("bad.gguf");
        std::fs::write(&bad, b"NOPE and more bytes").unwrap();
        assert!(Gguf::open(&bad).unwrap_err().contains("not a GGUF"));
        // A prefix shorter than the header asks for more, which `open` supplies.
        assert!(parse_header("x", &std::fs::read(&p).unwrap()[..20]).unwrap_err().contains(NEED_MORE));
        let _ = std::fs::remove_dir_all(d);
    }
}
