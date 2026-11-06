//! gn-weights' safetensors header check: the 8-byte length, then the JSON header against the
//! data that follows. Input layout: the whole file.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((len, rest)) = data.split_first_chunk::<8>() else { return };
    let n = (u64::from_le_bytes(*len) as usize).min(rest.len());
    let (header, body) = rest.split_at(n);
    if let Ok((tensors, _)) = gn_weights::safetensors::parse_header("fuzz", header, body.len() as u64) {
        for t in &tensors {
            assert!(t.begin <= t.end && t.end <= body.len() as u64, "a checked tensor lies inside the data");
            let _ = t.elements();
        }
    }
});
