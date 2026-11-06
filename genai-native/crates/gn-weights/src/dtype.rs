//! Element types across both formats: what a tensor's bytes mean and how many of them there are
//! per element (or per block, for the quantised GGUF types).

use std::fmt;

/// A tensor's element type. Safetensors names map one to one; GGUF's quantised block types keep
/// their ggml names.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    /// 8-bit float, 4 exponent bits, 3 mantissa bits (the `fp8` of the ComfyUI repacks).
    F8E4M3,
    F8E5M2,
    /// 8-bit power-of-two scale (MX block formats).
    F8E8M0,
    /// 4-bit float, two per byte (`float4_e2m1fn_x2`; NVFP4 files store these as `U8` or this).
    F4,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    /// A GGUF quantised block type by ggml name (`Q8_0`, `Q4_K`, `NVFP4`, …).
    Block(String),
    /// A name this crate does not know; the header is read, the data is refused.
    Unknown(String),
}

impl DType {
    /// The safetensors header name → type.
    pub fn from_safetensors(name: &str) -> Self {
        match name {
            "F64" => Self::F64,
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E8M0" => Self::F8E8M0,
            "F4" => Self::F4,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "I16" => Self::I16,
            "I8" => Self::I8,
            "U64" => Self::U64,
            "U32" => Self::U32,
            "U16" => Self::U16,
            "U8" => Self::U8,
            "BOOL" => Self::Bool,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// ggml's type id → type (the ids are part of the GGUF format).
    pub fn from_ggml(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => match ggml_block(other) {
                Some((name, _, _)) => Self::Block(name.to_string()),
                None => Self::Unknown(format!("ggml type {other}")),
            },
        }
    }

    /// Bytes per element for the dense types (`None` for block and unknown types; `F4` is half
    /// a byte, reported as `Some(1)` per two elements by [`DType::byte_len`]).
    pub fn element_bytes(&self) -> Option<usize> {
        Some(match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 | Self::I8 | Self::U8 | Self::Bool => 1,
            Self::F4 | Self::Block(_) | Self::Unknown(_) => return None,
        })
    }

    /// The bytes `count` elements take, when the type's size is known: dense types, `F4`
    /// (two per byte), and the GGUF block types (whole blocks; a count that is not a multiple
    /// of the block size is refused).
    pub fn byte_len(&self, count: u64) -> Option<u64> {
        if let Some(b) = self.element_bytes() {
            return count.checked_mul(b as u64);
        }
        match self {
            Self::F4 => Some(count.div_ceil(2)),
            Self::Block(name) => {
                let (_, block_elems, block_bytes) = ggml_block_by_name(name)?;
                if !count.is_multiple_of(block_elems) {
                    return None;
                }
                (count / block_elems).checked_mul(block_bytes)
            }
            _ => None,
        }
    }

    /// Is this one of the reduced-precision storage types a repack uses (anything below 16 bits
    /// per weight)?
    pub fn is_low_precision(&self) -> bool {
        matches!(self, Self::F8E4M3 | Self::F8E5M2 | Self::F4 | Self::I8 | Self::U8 | Self::Block(_))
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Block(n) | Self::Unknown(n) => f.write_str(n),
            other => write!(f, "{other:?}"),
        }
    }
}

/// ggml's quantised block types: (name, elements per block, bytes per block), by type id.
/// Source: ggml's `type_traits` table (MIT); the NVFP4 id is the one stable-diffusion.cpp
/// declares.
fn ggml_block(id: u32) -> Option<(&'static str, u64, u64)> {
    Some(match id {
        2 => ("Q4_0", 32, 18),
        3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22),
        7 => ("Q5_1", 32, 24),
        8 => ("Q8_0", 32, 34),
        9 => ("Q8_1", 32, 36),
        10 => ("Q2_K", 256, 84),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        15 => ("Q8_K", 256, 292),
        16 => ("IQ2_XXS", 256, 66),
        17 => ("IQ2_XS", 256, 74),
        18 => ("IQ3_XXS", 256, 98),
        19 => ("IQ1_S", 256, 50),
        20 => ("IQ4_NL", 32, 18),
        21 => ("IQ3_S", 256, 110),
        22 => ("IQ2_S", 256, 82),
        23 => ("IQ4_XS", 256, 136),
        29 => ("IQ1_M", 256, 56),
        34 => ("TQ1_0", 256, 54),
        35 => ("TQ2_0", 256, 66),
        39 => ("MXFP4", 32, 17),
        40 => ("NVFP4", 32, 17),
        _ => return None,
    })
}

fn ggml_block_by_name(name: &str) -> Option<(&'static str, u64, u64)> {
    (0..64u32).filter_map(ggml_block).find(|(n, _, _)| *n == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_follow_the_type() {
        assert_eq!(DType::from_safetensors("BF16").byte_len(10), Some(20));
        assert_eq!(DType::from_safetensors("F8_E4M3").byte_len(10), Some(10));
        assert_eq!(DType::from_safetensors("F4").byte_len(9), Some(5));
        assert_eq!(DType::from_ggml(8).to_string(), "Q8_0");
        assert_eq!(DType::from_ggml(8).byte_len(64), Some(68));
        assert_eq!(DType::from_ggml(8).byte_len(33), None, "a partial block");
        assert_eq!(DType::from_ggml(12).byte_len(256), Some(144));
        assert!(matches!(DType::from_ggml(99), DType::Unknown(_)));
        assert!(DType::from_ggml(99).byte_len(1).is_none());
        assert!(DType::from_safetensors("F8_E4M3").is_low_precision() && !DType::BF16.is_low_precision());
    }
}
