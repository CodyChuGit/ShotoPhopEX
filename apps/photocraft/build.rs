//! Windows only: embed the app icon and version info (VERSIONINFO) into `photocraft.exe`.
//!
//! On every other target this does nothing. A missing resource compiler is a warning, so a
//! cross-compile from macOS or Linux still links, unless `PHOTOCRAFT_REQUIRE_WINRES=1` (set by the
//! release workflow) turns it into an error.

#[path = "src/build_channel.rs"]
mod build_channel;

/// Embeds the Windows resources and delay-loads the print-only system DLLs (#5884, #8272).
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Fork: the "Sp" tile (`assets/app-icon/sp/`) instead of upstream's kitsune.
    println!("cargo:rerun-if-changed=../../assets/app-icon/sp/sp.ico");
    println!("cargo:rerun-if-env-changed=PHOTOCRAFT_REQUIRE_WINRES");
    // Fork: a Linux native build finds the CUDA libraries its release ships in `runtime/` beside
    // the binary (packaging/linux/package-native.sh; the folder the engine's runtime preference
    // falls back to, as on Windows) before the system's.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") && std::env::var_os("CARGO_FEATURE_NATIVE_DIFFUSION").is_some() {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/runtime");
    }
    println!("cargo:rerun-if-env-changed=PHOTOCRAFT_BUILD_CHANNEL");
    println!("cargo:rustc-check-cfg=cfg(photocraft_nightly_v1)");
    let nightly = match std::env::var("PHOTOCRAFT_BUILD_CHANNEL").as_deref() {
        Ok("nightly-v1") => {
            println!("cargo:rustc-cfg=photocraft_nightly_v1");
            true
        }
        // An empty value (an unset CI input) means the default stable build.
        Ok("stable" | "") | Err(std::env::VarError::NotPresent) => false,
        _ => {
            eprintln!("unsupported PHOTOCRAFT_BUILD_CHANNEL (expected stable or nightly-v1)");
            std::process::exit(1);
        }
    };
    println!("cargo:rerun-if-env-changed=PHOTOCRAFT_VERSION");
    let built_version = if nightly {
        match std::env::var("PHOTOCRAFT_VERSION") {
            Ok(version) if build_channel::is_nightly_version(&version) => version,
            _ => {
                eprintln!("nightly-v1 requires an explicit PHOTOCRAFT_VERSION feed identity");
                std::process::exit(1);
            }
        }
    } else {
        std::env::var("CARGO_PKG_VERSION").unwrap_or_default()
    };
    println!("cargo:rustc-env=PHOTOCRAFT_BUILT_VERSION={built_version}");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    // Printing to PDF (winprint's `PdfCreateRenderer`) imports Windows.Data.Pdf.dll, and its
    // printer options (the Print Ticket API) import prntvpt.dll; stripped Windows images (NTLite,
    // some N/LTSC builds) lack them. A load-time import stops the exe from starting at all
    // ("windows.data.pdf.dll was not found", #5884; "prntvpt.dll", #8272), so load them on first
    // use; the print commands check for them first and report an error instead.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/DELAYLOAD:windows.data.pdf.dll");
        // Fork: the same for the CUDA libraries candle's cudarc imports when the native engine
        // is built in (`native-diffusion`): only the `--engine` worker calls them, with the
        // runtime folder on its PATH, so the editor starts on a PC that has none
        // (plan-native-engine §10). cuBLAS carries the toolkit's major version
        // (`CUDARC_CUDA_VERSION` 13030 → `cublas64_13.dll`); cuRAND stays `curand64_10.dll`.
        println!("cargo:rustc-link-arg-bins=/DELAYLOAD:prntvpt.dll");
        println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");
        if std::env::var_os("CARGO_FEATURE_NATIVE_DIFFUSION").is_some() {
            let major = std::env::var("CUDARC_CUDA_VERSION").ok().and_then(|v| v.trim().parse::<u32>().ok()).map_or(13, |v| v / 1000);
            println!("cargo:rustc-link-arg-bins=/DELAYLOAD:cublas64_{major}.dll");
            println!("cargo:rustc-link-arg-bins=/DELAYLOAD:cublasLt64_{major}.dll");
            println!("cargo:rustc-link-arg-bins=/DELAYLOAD:curand64_10.dll");
        }
        println!("cargo:rustc-link-arg-bins=delayimp.lib");
    }
    let mut res = winresource::WindowsResource::new();
    res.set_icon("../../assets/app-icon/sp/sp.ico")
        .set("ProductName", "ShotoPhopEX")
        .set("FileDescription", "ShotoPhopEX image editor")
        .set("CompanyName", "ShotoPhopEX contributors")
        .set("LegalCopyright", "Copyright (c) the PhotoCraft authors and the ShotoPhopEX contributors. MIT OR Apache-2.0.")
        .set("OriginalFilename", "photocraft.exe")
        .set("InternalName", "photocraft");
    if nightly {
        // Launcher verifies the PE ProductVersion against the exact feed identity, not FileVersion.
        res.set("ProductVersion", &built_version);
    }
    if let Err(e) = res.compile() {
        if std::env::var_os("PHOTOCRAFT_REQUIRE_WINRES").is_some() {
            panic!("embedding Windows resources failed: {e}");
        }
        println!("cargo:warning=photocraft.exe built without icon/version resources: {e}");
    }
}
