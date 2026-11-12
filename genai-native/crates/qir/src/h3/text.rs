//! P-H3-Image's conditioning: Qwen3-VL-4B on the raw prompt (no chat template) with each
//! picture as `<Picture n>: ` and a vision block in front, the hidden states at five taps
//! through the text connector, and each row's AdaLN tag (ComfyUI `PH3ImageTEModel`,
//! `MiniMaxH3Tokenizer`, `PH3ImageConditioning`).

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, RmsNorm};
use tokenizers::Tokenizer;

use super::dit::{TAG_TEXT, TAG_VISUAL};
use crate::model::text_encoder::{QwenTextEncoder, TextConfig, IMAGE_PAD};
use crate::model::vision::VisionFeatures;
use crate::moss::weights::Weights;
use crate::Result;

pub const VISION_START: u32 = 151_652;
pub const VISION_END: u32 = 151_653;
/// The ids of an empty presentation.
pub const EMPTY_PROMPT: u32 = 151_643;
/// The hidden states the connector reads (transformers' `hidden_states` numbering); the last
/// one is the final layer's output after the final norm.
pub const TAPS: [usize; 5] = [13, 19, 25, 31, 36];
/// The pictures' copies for the encoder are scaled down to this area.
pub const VISION_PIXELS: usize = 1_000_000;
/// The default canvas side (8.5 MP).
pub const IMAGE_SIZE: usize = 2912;
pub const CANVAS_MULTIPLE: usize = 32;

/// The prompt as encoded: with pictures, a short instruction becomes `Edit <Picture 1>. …`;
/// one that already names the picture to edit, or an upsampler brief, is used as it is.
pub fn edit_prompt(prompt: &str, has_pictures: bool) -> String {
    if !has_pictures {
        return prompt.to_string();
    }
    let t = prompt.trim_start();
    let names_picture = t
        .strip_prefix("Edit <Picture ")
        .and_then(|r| {
            let digits = r.chars().take_while(char::is_ascii_digit).count();
            (digits > 0).then(|| r[digits..].starts_with('>'))
        })
        .unwrap_or(false);
    if names_picture || t.starts_with("subject_definitions:") {
        prompt.trim().to_string()
    } else {
        format!("Edit <Picture 1>. {prompt}")
    }
}

/// Python's `round` (half to even), as the size helpers use it.
fn round_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Aspect-preserving scale (down only) to `pixels`, each side rounded to the 32 grid
/// (`_size_within_area`).
pub fn size_within_area(width: usize, height: usize, pixels: usize) -> (usize, usize) {
    let scale = (pixels as f64 / (width.max(1) * height.max(1)) as f64).sqrt().min(1.0);
    let side = |s: usize| ((round_even(s as f64 * scale / CANVAS_MULTIPLE as f64) as usize) * CANVAS_MULTIPLE).max(CANVAS_MULTIPLE);
    (side(width), side(height))
}

/// The token ids: per picture `<Picture n>: `, `<|vision_start|>`, one `<|image_pad|>` the
/// picture's rows replace, `<|vision_end|>`; then the prompt. Text segments are tokenized one
/// by one, as ComfyUI does.
pub fn token_ids(tok: &Tokenizer, prompt: &str, pictures: usize) -> Result<Vec<u32>> {
    let enc = |s: &str| -> Result<Vec<u32>> {
        if s.is_empty() {
            return Ok(Vec::new());
        }
        Ok(tok.encode(s, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?.get_ids().to_vec())
    };
    let mut ids = Vec::new();
    for i in 1..=pictures {
        ids.extend(enc(&format!("<Picture {i}>: "))?);
        ids.extend([VISION_START, IMAGE_PAD, VISION_END]);
    }
    ids.extend(enc(prompt)?);
    if ids.is_empty() {
        ids.push(EMPTY_PROMPT);
    }
    Ok(ids)
}

/// Each row's AdaLN tag: the picture rows and their two flanking tokens are visual, the rest
/// text (`token_tags_from_embeds_info`).
pub fn row_tags(rows: usize, spans: &[(usize, usize)]) -> Vec<usize> {
    let mut tags = vec![TAG_TEXT; rows];
    for &(start, n) in spans {
        let a = start.saturating_sub(1);
        let b = (start + n + 1).min(rows);
        for t in tags.iter_mut().take(b).skip(a) {
            *t = TAG_VISUAL;
        }
    }
    tags
}

/// The text connector: standardize the concatenated taps, project, rescale, and put the sink
/// in the first row. All f32.
pub struct Connector {
    input_mean: Tensor,
    input_std: Tensor,
    proj: Tensor,
    output_mean: Tensor,
    output_std: Tensor,
    sink: Tensor,
}

impl Connector {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |n: &str| w.get_as(&format!("{prefix}.{n}"), DType::F32);
        Ok(Self {
            input_mean: g("input_mean")?,
            input_std: g("input_std")?,
            proj: g("proj.weight")?,
            output_mean: g("output_mean")?,
            output_std: g("output_std")?,
            sink: g("sink_embedding")?,
        })
    }

    /// `taps`: each `(T, C)`; returns `(T, out)` f32.
    pub fn forward(&self, taps: &[Tensor]) -> Result<Tensor> {
        let taps: Vec<Tensor> = taps.iter().map(|t| t.to_dtype(DType::F32)).collect::<candle_core::Result<_>>()?;
        let x = Tensor::cat(&taps, 1)?;
        let x = x.broadcast_sub(&self.input_mean)?.broadcast_div(&self.input_std)?;
        let x = x.matmul(&self.proj.t()?)?;
        let x = x.broadcast_mul(&self.output_std)?.broadcast_add(&self.output_mean)?;
        let t = x.dim(0)?;
        let sink = self.sink.unsqueeze(0)?;
        if t <= 1 {
            return Ok(sink);
        }
        Ok(Tensor::cat(&[sink, x.narrow(0, 1, t - 1)?], 0)?)
    }
}

/// The encoder's output for the DiT.
pub struct Conditioning {
    /// `(T, 5120)` f32.
    pub states: Tensor,
    /// Per row: [`TAG_TEXT`] or [`TAG_VISUAL`].
    pub tags: Vec<usize>,
}

pub struct H3Text {
    te: QwenTextEncoder,
    norm: RmsNorm,
    connector: Connector,
}

impl H3Text {
    /// From P-H3-Image's encoder file (`model.language_model.*`, `text_connector.*`), bf16 or
    /// INT8 ConvRot.
    pub fn load(w: &Weights) -> Result<Self> {
        let root = if w.contains("model.language_model.norm.weight") { "model.language_model" } else { "model" };
        let (te, norm) = QwenTextEncoder::load_weights(&TextConfig::qwen3vl_4b(), w, root)?;
        Ok(Self { te, norm, connector: Connector::load(w, "text_connector")? })
    }

    /// `ids` from [`token_ids`], `pictures` the vision tower's features, one per placeholder.
    pub fn encode(&self, ids: &[u32], pictures: &[&VisionFeatures]) -> Result<Conditioning> {
        let last = TAPS[TAPS.len() - 1];
        let (mut taps, spans) = if pictures.is_empty() {
            let t = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &Device::Cpu)?;
            (self.te.forward_taps(&t, &TAPS)?, Vec::new())
        } else {
            self.te.forward_with_images_taps(ids, pictures, &TAPS)?
        };
        // the last tap is the final layer's output, read after the final norm
        if let Some(t) = taps.last_mut() {
            *t = self.norm.forward(t)?;
        }
        debug_assert_eq!(last, 36);
        let taps: Vec<Tensor> = taps.iter().map(|t| t.squeeze(0)).collect::<candle_core::Result<_>>()?;
        let states = self.connector.forward(&taps)?;
        let rows = states.dim(0)?;
        Ok(Conditioning { states, tags: row_tags(rows, &spans) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_prompt_rules() {
        assert_eq!(edit_prompt("make it red", false), "make it red");
        assert_eq!(edit_prompt("make it red", true), "Edit <Picture 1>. make it red");
        assert_eq!(edit_prompt("  Edit <Picture 2> to be red ", true), "Edit <Picture 2> to be red");
        assert_eq!(edit_prompt("Edit <Picture x>", true), "Edit <Picture 1>. Edit <Picture x>");
        assert_eq!(edit_prompt("subject_definitions: a", true), "subject_definitions: a");
    }

    #[test]
    fn sizes_round_like_python() {
        assert_eq!(size_within_area(2912, 2912, 2912 * 2912), (2912, 2912));
        assert_eq!(size_within_area(4000, 3000, VISION_PIXELS), (1152, 864));
        // never up
        assert_eq!(size_within_area(512, 512, VISION_PIXELS), (512, 512));
        assert_eq!(round_even(2.5), 2.0);
        assert_eq!(round_even(3.5), 4.0);
    }

    #[test]
    fn tags_cover_vision_block() {
        // "<Picture 1>: " (3 ids) + start + 4 rows + end + prompt (2)
        let tags = row_tags(11, &[(4, 4)]);
        assert_eq!(tags, vec![1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn connector_math() {
        let d = Device::Cpu;
        let c = Connector {
            input_mean: Tensor::new(&[1f32, 0.0, 0.0, 0.0], &d).unwrap(),
            input_std: Tensor::new(&[2f32, 1.0, 1.0, 1.0], &d).unwrap(),
            proj: Tensor::new(&[[1f32, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]], &d).unwrap(),
            output_mean: Tensor::new(&[10f32, 0.0], &d).unwrap(),
            output_std: Tensor::new(&[1f32, 3.0], &d).unwrap(),
            sink: Tensor::new(&[7f32, 7.0], &d).unwrap(),
        };
        let a = Tensor::new(&[[3f32, 1.0], [5.0, 2.0]], &d).unwrap();
        let b = Tensor::new(&[[4f32, 1.0], [6.0, 1.0]], &d).unwrap();
        let out = c.forward(&[a, b]).unwrap().to_vec2::<f32>().unwrap();
        // row 1: x = [5, 2, 6, 1] -> [(5-1)/2, 2, 6, 1] = [2, 2, 6, 1] -> [4, 5] -> [14, 15]
        assert_eq!(out, vec![vec![7.0, 7.0], vec![14.0, 15.0]]);
    }
}
