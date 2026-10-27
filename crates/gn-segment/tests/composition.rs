use anyhow::Result;
use gn_segment::text::ClipTokenizer;

#[test]
fn tokenizer_matches_existing_route() -> Result<()> {
    let path = std::path::Path::new(r"C:\Users\5090\Documents\_eval\native-models\sam3.1\tokenizer.json");
    if !path.is_file() {
        return Ok(());
    }
    let native = ClipTokenizer::open(path)?;
    let reference = gn_perception::tokenizer::ClipTokenizer::open(path).map_err(anyhow::Error::msg)?;
    for prompt in [String::new(), "coffee cup".into(), "DOG, café — a blue bench!".into(), "a ".repeat(100)] {
        let (ids, mask) = native.encode(&prompt)?;
        let expected = reference.encode(&prompt).map_err(anyhow::Error::msg)?;
        assert_eq!(ids.map(i64::from).as_slice(), expected.ids);
        assert_eq!(mask.map(i64::from).as_slice(), expected.mask);
    }
    Ok(())
}
