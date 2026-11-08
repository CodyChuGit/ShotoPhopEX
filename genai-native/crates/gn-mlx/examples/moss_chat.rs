//! A text answer from MOSS's Qwen3.5-9B encoder (the Mac's prompt enhancer), timed.
//! `cargo run --release -p gn-mlx --example moss_chat -- <models dir> <system> <prompt> [max tokens]`
//! (`MOSS_CHAT_IMAGE=<picture.png>` shows it a picture).
use std::path::PathBuf;
use std::time::Instant;

use gn_mlx::pipeline::MossMlx;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [models, system, prompt, rest @ ..] = a.as_slice() else {
        anyhow::bail!("usage: moss_chat <models dir> <system> <prompt> [max tokens]");
    };
    let models = PathBuf::from(models);
    let max = rest.first().map_or(Ok(160), |m| m.parse())?;
    let mut moss = MossMlx::new(&models, &models.join(".qir-cache").join("mlx"))?;
    let picture = match std::env::var("MOSS_CHAT_IMAGE") {
        Ok(p) => {
            let img = image::open(p)?.to_rgba8();
            Some(gn_mlx::pipeline::RefPicture { width: img.width() as usize, height: img.height() as usize, rgba: img.into_raw() })
        }
        Err(_) => None,
    };
    for _ in 0..2 {
        let t = Instant::now();
        let answer = moss.chat(system, prompt, picture.as_ref(), max, &|| false)?;
        println!("{:.1} s, {} words:\n{answer}\n", t.elapsed().as_secs_f32(), answer.split_whitespace().count());
    }
    Ok(())
}
