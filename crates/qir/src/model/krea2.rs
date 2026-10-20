//! Krea 2 (plan Phase 4): what the model needs from the Qwen3-VL-4B encoder, as ComfyUI's
//! `krea2` text encoder hands it over, and (below) the single-stream DiT itself.
//!
//! The conditioning is a stack of twelve hidden states of the 4B language model
//! (`hidden_states[2, 5, 8, …, 35]`, transformers' numbering: `0` is the embedding), kept for
//! the tokens after the chat template's prefix (the system turn and `<|im_start|>user\n`),
//! shaped `(tokens, 12, 2560)`; the DiT's `txtfusion` adapter fuses the twelve per token.

use candle_core::Tensor;

use crate::Result;
use crate::model::text_encoder::QwenTextEncoder;

/// The hidden states Krea 2 taps (transformers' `hidden_states` indices).
pub const TAPS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];
/// The language model's width.
pub const TEXT_DIM: usize = 2560;
/// The chat template (ComfyUI's `KREA2_TEMPLATE`): the Qwen-Image system prompt.
pub const SYS_PROMPT: &str =
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";

const IM_START: u32 = 151_644;
const USER: u32 = 872;
const NEWLINE: u32 = 198;

/// The prompt in Krea 2's chat template.
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// How many leading tokens the encoder's output drops: up to the second `<|im_start|>`, and
/// through the `user\n` that follows it (ComfyUI's `Krea2TEModel.encode_token_weights`).
pub fn template_end(ids: &[u32]) -> usize {
    let mut end = 0;
    let mut seen = 0;
    for (i, t) in ids.iter().enumerate() {
        if *t == IM_START && seen < 2 {
            end = i;
            seen += 1;
        }
    }
    if ids.len() > end + 3 && ids.get(end + 1) == Some(&USER) && ids.get(end + 2) == Some(&NEWLINE) {
        end += 3;
    }
    end
}

/// The conditioning `(tokens', 12, 2560)` for `ids` (the templated prompt's tokens): the
/// twelve tapped states with the template's prefix dropped.
pub fn condition(te: &QwenTextEncoder, ids: &[u32], dev: &candle_core::Device) -> Result<Tensor> {
    let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), dev)?;
    let taps = te.forward_taps(&input, &TAPS)?;
    let end = template_end(ids);
    let keep = ids.len().saturating_sub(end);
    anyhow::ensure!(keep > 0, "the prompt has no tokens after the template");
    let rows: Vec<Tensor> = taps.iter().map(|t| t.narrow(1, end, keep)).collect::<candle_core::Result<_>>()?;
    // (1, keep, 2560) × 12 → (keep, 12, 2560)
    let stacked = Tensor::stack(&rows, 2)?; // (1, keep, 12, 2560)
    Ok(stacked.squeeze(0)?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prefix_ends_after_the_user_turn_opens() {
        // system turn … <|im_start|> user \n prompt …
        let ids = [IM_START, 8948, NEWLINE, 1, 2, 151_645, NEWLINE, IM_START, USER, NEWLINE, 100, 101, 151_645];
        assert_eq!(template_end(&ids), 10);
        // Without the `user\n` pair the cut sits at the second <|im_start|> itself.
        let ids = [IM_START, 8948, NEWLINE, 1, IM_START, 5, 6, 7, 8];
        assert_eq!(template_end(&ids), 4);
        assert_eq!(template_end(&[]), 0);
    }

    #[test]
    fn the_template_is_comfyuis() {
        let t = template("a fox");
        assert!(t.starts_with("<|im_start|>system\nDescribe the image by detailing") && t.ends_with("<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"));
    }
}
