//! `gn-engine --serve [--step-ms N]`: a worker on 127.0.0.1 with a random port, announced on
//! the first stdout line (`gn-engine port=NNNN`); the token comes from `GN_ENGINE_TOKEN` (or
//! `--token`). Until the native engine exists it serves the fake engine; the editor's
//! `photocraft.exe --engine` calls the same `serve`.

use std::io::Write;

use gn_engine::fake::FakeEngine;
use gn_engine::server;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut token = std::env::var("GN_ENGINE_TOKEN").unwrap_or_default();
    let mut step_ms = 5u64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--token" => {
                token = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--step-ms" => {
                step_ms = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(5);
                i += 2;
            }
            _ => i += 1,
        }
    }
    if token.is_empty() {
        eprintln!("gn-engine: no token (GN_ENGINE_TOKEN or --token)");
        std::process::exit(2);
    }
    let listener = match server::bind() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("gn-engine: cannot bind: {e}");
            std::process::exit(1);
        }
    };
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    println!("{}", server::port_line(port));
    let _ = std::io::stdout().flush();
    let mut engine = FakeEngine { step_ms, ..FakeEngine::default() };
    if let Err(e) = server::serve(&listener, &token, &mut engine) {
        eprintln!("gn-engine: {e}");
        std::process::exit(1);
    }
}
