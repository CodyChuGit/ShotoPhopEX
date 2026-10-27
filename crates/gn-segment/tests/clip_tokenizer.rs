//! The CLIP tokenizer assembled from ComfyUI's `sd1_tokenizer` gives the ONNX route's
//! tokenizer.json's ids (skipped when either is absent).

use std::path::Path;

use gn_segment::text::ClipTokenizer;

#[test]
fn the_assembled_clip_tokenizer_matches_the_onnx_routes() -> anyhow::Result<()> {
    let dir = Path::new(r"C:\Users\5090\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\sd1_tokenizer");
    let json = Path::new(r"C:\Users\5090\Documents\_eval\native-models\sam3.1\tokenizer.json");
    if !dir.is_dir() || !json.is_file() {
        return Ok(());
    }
    let (ours, theirs) = (ClipTokenizer::open(dir)?, ClipTokenizer::open(json)?);
    for p in [
        "dog",
        "the coffee cup on the left",
        "Red   SNEAKERS,  white laces!!",
        "café crème, naïve résumé, 東京",
        "it's they're we've I'm you'll he'd",
        "a very long prompt that runs past the thirty pieces the window keeps, with more words after the cut than before it",
    ] {
        assert_eq!(ours.encode(p)?, theirs.encode(p)?, "{p:?}");
    }
    Ok(())
}
