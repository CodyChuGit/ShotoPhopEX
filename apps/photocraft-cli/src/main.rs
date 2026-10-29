#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

fn main() {
    // comfy-photocraft: `photocraft-cli --engine …` is the native engine's worker process, so a
    // headless session has a worker next to it too.
    if std::env::args_os().nth(1).is_some_and(|a| a == photocraft_genai::native::ENGINE_FLAG) {
        let args: Vec<String> = std::env::args_os().skip(2).map(|a| a.to_string_lossy().into_owned()).collect();
        std::process::exit(photocraft_genai::native::engine_main(&args));
    }
    // `args_os`, not `args`: a non-Unicode argument (a Latin-1 file name on Linux) must be a
    // usage error, not a panic before anything is printed (issue #1108).
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let code = photocraft_cli::run_os(&args, &mut std::io::stdout(), &mut std::io::stderr());
    std::process::exit(code);
}
