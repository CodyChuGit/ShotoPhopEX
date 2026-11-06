//! gn-weights' GGUF header parser (metadata values, nested arrays, tensor infos).
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(h) = gn_weights::gguf::parse_header("fuzz", data) {
        for t in &h.tensors {
            let _ = t.elements();
        }
    }
});
