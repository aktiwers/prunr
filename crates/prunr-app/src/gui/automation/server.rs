//! The control socket: one JSON request per line on 127.0.0.1:port, one
//! JSON reply per line. Requests are handed to the UI thread and answered
//! from there; the thread wakes the event loop so an idle app still
//! renders the frames a reply waits for.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::Duration;

use super::{Pending, Request, Response};

/// How long one request may take, long enough for an erase or a model load.
const REPLY_TIMEOUT: Duration = Duration::from_secs(600);

pub fn spawn(port: u16, ctx: egui::Context) -> Option<mpsc::Receiver<Pending>> {
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(port, %e, "control socket bind failed");
            return None;
        }
    };
    let (tx, rx) = mpsc::channel();
    let spawned = std::thread::Builder::new().name("prunr-control".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            serve(stream, &tx, &ctx);
        }
    });
    if let Err(e) = spawned {
        tracing::warn!(%e, "control socket thread failed");
        return None;
    }
    tracing::info!(port, "control socket listening");
    Some(rx)
}

fn serve(stream: TcpStream, tx: &mpsc::Sender<Pending>, ctx: &egui::Context) {
    let Ok(mut writer) = stream.try_clone() else { return };
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Err(e) => Response::err(format!("bad request: {e}")),
            Ok(request) => {
                let (reply, reply_rx) = mpsc::channel();
                if tx.send(Pending { request, reply }).is_err() {
                    break;
                }
                ctx.request_repaint();
                reply_rx.recv_timeout(REPLY_TIMEOUT).unwrap_or_else(|_| Response::err("timed out"))
            }
        };
        let Ok(mut json) = serde_json::to_string(&response) else { break };
        json.push('\n');
        if writer.write_all(json.as_bytes()).is_err() {
            break;
        }
    }
}
