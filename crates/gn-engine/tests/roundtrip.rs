//! The worker as a real child process: spawn, greet, run with progress, cancel mid-run, free,
//! survive a kill, big pictures both ways, a wrong token.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use gn_engine::client::{ClientError, EngineClient, new_token, rgba_blob};
use gn_engine::protocol::{ErrorKind, RunRequest};

fn exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_gn-engine"))
}

fn request(model: &str, task: &str, steps: u32) -> RunRequest {
    RunRequest {
        model: model.into(),
        task: task.into(),
        prompt: "a fox".into(),
        negative: String::new(),
        seed: 7,
        steps,
        guidance: 0.0,
        width: 64,
        height: 32,
        image: None,
        mask: None,
        references: Vec::new(),
        params: BTreeMap::new(),
    }
}

#[test]
fn a_spawned_worker_runs_reports_progress_and_quits() {
    let token = new_token();
    let mut c = EngineClient::spawn(&exe(), &["--serve".into(), "--step-ms".into(), "2".into()], &token).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(c.engine, "fake");
    assert!(c.alive() && c.pid().is_some());
    let h = c.health().unwrap_or_else(|e| panic!("{e}"));
    assert!(h.ok && h.resident.is_empty());
    let mut reports = Vec::new();
    let (info, images) =
        c.run(&request("fake/image", "image", 8), &[], &mut |f, m| reports.push((f, m.to_string())), &|| false).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(info.images.len(), 1);
    assert_eq!((info.images[0].width, info.images[0].height, info.images[0].channels), (64, 32, 4));
    assert_eq!(images[0].len(), 64 * 32 * 4);
    assert!(reports.len() >= 8 && reports.iter().any(|(_, m)| m == "Loading"), "{reports:?}");
    assert!((reports.last().map(|r| r.0).unwrap_or(0.0) - 1.0).abs() < 1e-6);
    assert_eq!(c.health().map(|h| h.resident).unwrap_or_default(), vec!["fake/image".to_string()]);
    c.free().unwrap_or_else(|e| panic!("{e}"));
    assert!(c.health().map(|h| h.resident.is_empty()).unwrap_or(false));
    let idx = c.index("C:/models").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(idx["dir"], "C:/models");
    // A model the engine lacks is a typed error, not a dead worker.
    match c.run(&request("nope/model", "image", 1), &[], &mut |_, _| {}, &|| false) {
        Err(ClientError::Engine { kind: ErrorKind::ModelMissing, .. }) => {}
        other => panic!("{other:?}"),
    }
    assert!(c.alive());
    c.quit();
}

#[test]
fn cancel_reaches_a_running_worker_and_a_killed_one_is_noticed() {
    let token = new_token();
    let mut c = EngineClient::spawn(&exe(), &["--serve".into(), "--step-ms".into(), "50".into()], &token).unwrap_or_else(|e| panic!("{e}"));
    let flag = Arc::new(AtomicBool::new(false));
    let f2 = Arc::clone(&flag);
    let t0 = Instant::now();
    let mut n = 0;
    let r = c.run(
        &request("fake/image", "image", 200),
        &[],
        &mut |_, _| {
            n += 1;
            if n == 3 {
                f2.store(true, Ordering::SeqCst);
            }
        },
        &|| flag.load(Ordering::SeqCst),
    );
    assert!(matches!(r, Err(ClientError::Engine { kind: ErrorKind::Cancelled, .. })), "{r:?}");
    assert!(t0.elapsed().as_secs() < 5, "cancelled well before 200 × 50 ms");
    // The worker is still fine after a cancel.
    assert!(c.health().map(|h| h.ok).unwrap_or(false));
    // Killed from outside: the next call says so instead of hanging.
    c.kill();
    assert!(!c.alive());
    assert!(matches!(c.health(), Err(ClientError::Gone(_))));
}

#[test]
fn pictures_travel_both_ways_and_the_token_is_checked() {
    let token = new_token();
    let mut c = EngineClient::spawn(&exe(), &["--serve".into()], &token).unwrap_or_else(|e| panic!("{e}"));
    // A 1 MP input (4 MB) through an edit: the fake tints it and sends it back whole.
    let (w, h) = (1024u32, 1024u32);
    let input: Vec<u8> = (0..w * h).flat_map(|i| [(i % 251) as u8, (i % 241) as u8, (i % 239) as u8, 255]).collect();
    let mut req = request("fake/edit", "edit", 2);
    req.image = Some(rgba_blob(w, h, 0));
    let (info, out) = c.run(&req, std::slice::from_ref(&input), &mut |_, _| {}, &|| false).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((info.images[0].width, info.images[0].height), (w, h));
    assert_eq!(out[0].len(), input.len());
    assert_ne!(out[0], input, "tinted");
    // A segment returns coverage.
    let mut seg = request("fake/segment", "segment", 1);
    seg.image = Some(rgba_blob(100, 60, 0));
    let (info, out) = c.run(&seg, &[vec![0u8; 100 * 60 * 4]], &mut |_, _| {}, &|| false).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(info.images[0].channels, 1);
    assert_eq!(out[0].len(), 100 * 60);
    assert_eq!(out[0][30 * 100 + 50], 255, "the centre is inside the disc");
    assert_eq!(out[0][0], 0, "the corner is outside");
    // A blob whose bytes do not match its descriptor is refused by the client before sending.
    let mut bad = request("fake/edit", "edit", 1);
    bad.image = Some(rgba_blob(10, 10, 0));
    assert!(matches!(c.run(&bad, &[vec![0u8; 7]], &mut |_, _| {}, &|| false), Err(ClientError::Engine { kind: ErrorKind::BadRequest, .. })));
    // A second client with the wrong token is turned away; the right one still works.
    let port = c.port;
    assert!(matches!(EngineClient::connect(port, "wrong"), Err(ClientError::Start(_))));
    assert!(c.health().is_ok());
    c.quit();
}

/// A client that hangs up without `Quit` (what a crashed editor amounts to) ends the worker.
#[test]
fn a_worker_whose_client_vanishes_exits_by_itself() {
    use std::io::BufRead as _;
    let token = new_token();
    // Spawned by hand so no client `Drop` can send Quit or kill it.
    let mut child = std::process::Command::new(exe())
        .arg("--serve")
        .env("GN_ENGINE_TOKEN", &token)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut first = String::new();
    std::io::BufReader::new(child.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let port = gn_engine::server::parse_port_line(&first).unwrap();
    {
        let mut c = EngineClient::connect(port, &token).unwrap_or_else(|e| panic!("{e}"));
        assert!(c.health().is_ok());
        // Dropped here: the socket closes, nothing is sent.
    }
    let t0 = Instant::now();
    let mut exited = None;
    while t0.elapsed().as_secs() < 5 {
        if let Ok(Some(status)) = child.try_wait() {
            exited = Some(status);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if exited.is_none() {
        let _ = child.kill();
    }
    let _ = child.wait();
    assert!(exited.is_some(), "the worker should exit once its greeted client left");
}

#[test]
fn a_stay_worker_outlives_its_clients_until_quit() {
    use std::io::BufRead as _;
    let token = new_token();
    let mut child = std::process::Command::new(exe())
        .args(["--serve", "--stay"])
        .env("GN_ENGINE_TOKEN", &token)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut first = String::new();
    std::io::BufReader::new(child.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let port = gn_engine::server::parse_port_line(&first).unwrap();
    // Two clients in turn, each going away without a word (a connected client's drop closes
    // the socket and sends nothing): the worker serves the next one.
    for _ in 0..2 {
        let mut c = EngineClient::connect(port, &token).unwrap_or_else(|e| panic!("{e}"));
        assert!(c.health().is_ok());
        drop(c);
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
    assert!(matches!(child.try_wait(), Ok(None)), "a stay worker keeps running after its clients left");
    // Quit still ends it.
    let c = EngineClient::connect(port, &token).unwrap_or_else(|e| panic!("{e}"));
    c.quit();
    let t0 = Instant::now();
    let mut exited = false;
    while t0.elapsed().as_secs() < 5 {
        if let Ok(Some(_)) = child.try_wait() {
            exited = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if !exited {
        let _ = child.kill();
    }
    let _ = child.wait();
    assert!(exited, "Quit ends a stay worker");
}
