//! The editor <-> worker link: the pre-auth greeting reader (anyone on the machine can connect
//! to the loopback port) and the framed message readers, both directions. Nothing may panic,
//! and nothing may allocate more than the frame caps allow.
#![no_main]

use gn_engine::protocol::{self, Reply, Request};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = protocol::read_greeting::<Request>(&mut &data[..]);
    let mut r = &data[..];
    while let Ok(Some(m)) = protocol::read_message::<Request>(&mut r) {
        if let Request::Run(req) = &m.message {
            let _ = protocol::check_blobs(&req.blob_refs(), &m.blobs);
        }
    }
    let mut r = &data[..];
    while let Ok(Some(_)) = protocol::read_message::<Reply>(&mut r) {}
});
