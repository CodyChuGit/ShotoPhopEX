//! The CLIP byte-pair tokenizer SAM 3's text encoder expects, read from `tokenizer.json`.
//!
//! The encoding mirrors ComfyUI's `SAM3Tokenizer` (`comfy/text_encoders/sam3_clip.py`): a window
//! of 32 positions, `<|startoftext|>` (49406), at most 30 word pieces, `<|endoftext|>` (49407),
//! then zeros; the attention mask is 1 through the end token and 0 over the padding.

use std::path::Path;

/// The window every SAM 3 text prompt is encoded into.
pub const CONTEXT: usize = 32;
pub const START: i64 = 49406;
pub const END: i64 = 49407;
pub const PAD: i64 = 0;

pub struct ClipTokenizer {
    inner: tokenizers::Tokenizer,
}

/// A prompt encoded for the text encoder: ids and mask, each `CONTEXT` long.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    pub ids: Vec<i64>,
    pub mask: Vec<i64>,
}

impl ClipTokenizer {
    pub fn open(path: &Path) -> Result<Self, String> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(ClipTokenizer { inner })
    }

    /// Encode one prompt into the 32-token window.
    pub fn encode(&self, text: &str) -> Result<Encoded, String> {
        // Word pieces only; the window's start and end tokens are placed here so the truncation
        // rule (30 pieces) is ours and matches ComfyUI's, not the file's post-processor.
        let enc = self.inner.encode(text, false).map_err(|e| e.to_string())?;
        let pieces: Vec<i64> = enc.get_ids().iter().take(CONTEXT - 2).map(|&id| i64::from(id)).collect();
        let mut ids = Vec::with_capacity(CONTEXT);
        ids.push(START);
        ids.extend_from_slice(&pieces);
        ids.push(END);
        let used = ids.len();
        ids.resize(CONTEXT, PAD);
        let mut mask = vec![1_i64; used];
        mask.resize(CONTEXT, 0);
        Ok(Encoded { ids, mask })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Option<ClipTokenizer> {
        let path = std::env::var_os("GN_SAM3_DIR").map(|d| Path::new(&d).join("tokenizer.json"))?;
        ClipTokenizer::open(&path).ok()
    }

    #[test]
    fn window_shape_and_special_tokens() {
        let Some(tok) = fixture() else { return };
        let e = tok.encode("fox").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(e.ids.len(), CONTEXT);
        assert_eq!(e.mask.len(), CONTEXT);
        assert_eq!(e.ids[0], START);
        // "fox" is one CLIP word piece (`fox</w>` = 3240).
        assert_eq!(e.ids[1], 3240);
        assert_eq!(e.ids[2], END);
        assert_eq!(&e.mask[..3], &[1, 1, 1]);
        assert_eq!(e.mask[3], 0);
        assert_eq!(e.ids[3], PAD);
    }

    #[test]
    fn long_prompts_are_truncated_not_rejected() {
        let Some(tok) = fixture() else { return };
        let long = "a ".repeat(100);
        let e = tok.encode(&long).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(e.ids.len(), CONTEXT);
        assert_eq!(e.ids[CONTEXT - 1], END);
        assert!(e.mask.iter().all(|&m| m == 1));
    }

    #[test]
    fn case_is_folded() {
        let Some(tok) = fixture() else { return };
        let a = tok.encode("Red Fox").unwrap_or_else(|e| panic!("{e}"));
        let b = tok.encode("red fox").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b);
    }
}
