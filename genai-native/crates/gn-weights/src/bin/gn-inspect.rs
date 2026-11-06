//! `gn-inspect <file-or-folder> [--names N] [--grep text] [--json]`: what a model file (or a
//! models folder in ComfyUI's layout) holds, from its header alone.

use std::path::PathBuf;

use gn_weights::gguf::Gguf;
use gn_weights::index::{self, ModelIndex};
// `index::inspect` identifies from the tensors; the name is only shown.
use gn_weights::safetensors::SafeTensors;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut target = None;
    let (mut names, mut grep, mut json, mut dump) = (12usize, None::<String>, false, None::<String>);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--names" => {
                names = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(12);
                i += 2;
            }
            "--dump" => {
                dump = args.get(i + 1).cloned();
                i += 2;
            }
            "--grep" => {
                grep = args.get(i + 1).cloned();
                i += 2;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            other => {
                target = Some(PathBuf::from(other));
                i += 1;
            }
        }
    }
    let Some(target) = target else {
        eprintln!("usage: gn-inspect <file-or-folder> [--names N] [--grep text] [--json]");
        std::process::exit(2);
    };
    if target.is_dir() {
        match ModelIndex::scan(&target) {
            Ok(ix) => {
                if json {
                    println!("{}", serde_json::to_string_pretty(&ix).unwrap_or_default());
                    return;
                }
                for f in &ix.files {
                    let fp = f.fingerprint.as_ref();
                    println!(
                        "{:<16} {:<52} {:>8.0} MB  {:<14} {:<12} {:>5} tensors  {}",
                        f.folder,
                        f.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                        f.size as f64 / 1e6,
                        format!("{:?}", f.family),
                        format!("{:?}", f.precision),
                        fp.map_or(0, |fp| fp.tensors),
                        f.error.clone().unwrap_or_else(|| fp
                            .map(|fp| fp.dtypes.iter().map(|(k, v)| format!("{k}×{v}")).collect::<Vec<_>>().join(" "))
                            .unwrap_or_default())
                    );
                }
            }
            Err(e) => {
                eprintln!("gn-inspect: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    let ext = target.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let installed = index::inspect(&target, "");
    if json {
        println!("{}", serde_json::to_string_pretty(&installed).unwrap_or_default());
        return;
    }
    if let Some(e) = &installed.error {
        eprintln!("gn-inspect: {e}");
        std::process::exit(1);
    }
    println!("{}  {:.0} MB  family {:?}  precision {:?}", target.display(), installed.size as f64 / 1e6, installed.family, installed.precision);
    if let Some(fp) = &installed.fingerprint {
        println!(
            "  {} tensors, {:.0} MB of data; types: {}",
            fp.tensors,
            fp.data_bytes as f64 / 1e6,
            fp.dtypes.iter().map(|(k, v)| format!("{k}×{v}")).collect::<Vec<_>>().join(" ")
        );
        println!("  prefixes: {}", fp.prefixes.iter().map(|(k, v)| format!("{k}×{v}")).collect::<Vec<_>>().join(" "));
        for (k, v) in &fp.metadata {
            println!("  meta {k} = {v}");
        }
    }
    let matches = |n: &str| grep.as_deref().is_none_or(|g| n.contains(g));
    if ext == "gguf" {
        if let Ok(g) = Gguf::open(&target) {
            for (k, v) in g.metadata().iter().filter(|(k, _)| !k.starts_with("tokenizer.")) {
                println!("  kv {k} = {v:?}");
            }
            for t in g.tensors().iter().filter(|t| matches(&t.name)).take(names) {
                println!("  {:<60} {:<8} {:?} {} B", t.name, t.dtype.to_string(), t.dims, t.bytes.map_or("?".into(), |b| b.to_string()));
            }
        }
    } else if let Ok(st) = SafeTensors::open(&target) {
        for t in st.tensors().iter().filter(|t| matches(&t.name)).take(names) {
            println!("  {:<60} {:<8} {:?}", t.name, t.dtype.to_string(), t.shape);
        }
        // `--dump name`: a small tensor's bytes as text and as numbers (the `comfy_quant` blobs
        // are JSON; scales are a few floats).
        if let Some(name) = dump {
            match st.bytes(&name) {
                Ok(b) if b.len() <= 4096 => {
                    let text: String = b.iter().map(|c| if c.is_ascii_graphic() || *c == b' ' { *c as char } else { '.' }).collect();
                    println!("  dump {name} ({} bytes): {text}", b.len());
                    if let Some(t) = st.info(&name)
                        && t.dtype == gn_weights::DType::F32
                    {
                        let floats: Vec<String> = b.as_chunks::<4>().0.iter().take(16).map(|c| format!("{}", f32::from_le_bytes(*c))).collect();
                        println!("  as f32: {}", floats.join(" "));
                    }
                }
                Ok(b) => println!("  dump {name}: {} bytes (too large to print)", b.len()),
                Err(e) => println!("  dump: {e}"),
            }
        }
    }
}
