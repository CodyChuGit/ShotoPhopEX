//! The CLIP tokenizer assembled from ComfyUI's `sd1_tokenizer` gives the same ids as
//! the SAM 3.1 model folder's tokenizer.json (skipped when either is absent).

use gn_segment::text::ClipTokenizer;

#[test]
fn the_assembled_clip_tokenizer_matches_the_tokenizer_json() -> anyhow::Result<()> {
    let dir = &std::path::Path::new(&std::env::var_os("COMFY_DIR").unwrap_or_default()).join("comfy/sd1_tokenizer");
    let json = &std::path::Path::new(&std::env::var_os("NATIVE_MODELS").unwrap_or_default()).join("sam3.1/tokenizer.json");
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
