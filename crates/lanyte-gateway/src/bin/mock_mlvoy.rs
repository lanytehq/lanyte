use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Duration;

use ipcprims::peer::async_connect;
use lanyte_common::channels;
use serde_json::Value;

const DEFAULT_FRAME: &str = r#"{
  "type": "mail_search_request",
  "request_id": "123e4567-e89b-42d3-a456-426614174000",
  "delegation_id": "dev-test",
  "query": "hello"
}"#;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let socket_path = args.next().ok_or_else(usage)?;

    let mut frame_file = None;
    let mut wait_ms = 0u64;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => {
                frame_file = Some(args.next().ok_or_else(usage)?);
            }
            "--wait-ms" => {
                let value = args.next().ok_or_else(usage)?;
                wait_ms = value
                    .parse()
                    .map_err(|_| format!("invalid --wait-ms value: {value}"))?;
            }
            _ => return Err(usage()),
        }
    }

    let frames = load_frames(frame_file.as_deref())?;
    let client = async_connect(&socket_path, &[channels::MAIL])
        .await
        .map_err(|err| format!("connect failed: {err}"))?;
    let (tx, mut rx) = client.into_split();

    for frame in &frames {
        tx.send_json(channels::MAIL, frame)
            .await
            .map_err(|err| format!("send failed: {err}"))?;
    }

    if wait_ms > 0 {
        let _ = tokio::time::timeout(Duration::from_millis(wait_ms), rx.recv()).await;
    }

    Ok(())
}

fn load_frames(path: Option<&str>) -> Result<Vec<Value>, String> {
    let payload = match path {
        Some(path) => fs::read_to_string(path)
            .map_err(|err| format!("failed to read frames file {path}: {err}"))?,
        None => DEFAULT_FRAME.to_owned(),
    };

    let json: Value =
        serde_json::from_str(&payload).map_err(|err| format!("invalid frame JSON: {err}"))?;

    match json {
        Value::Array(frames) if !frames.is_empty() => Ok(frames),
        Value::Array(_) => Err("frames JSON array must not be empty".to_owned()),
        value @ Value::Object(_) => Ok(vec![value]),
        _ => Err("frames JSON must be an object or non-empty array".to_owned()),
    }
}

fn usage() -> String {
    "usage: mock_mlvoy <socket-path> [--frames <json-file>] [--wait-ms <milliseconds>]".to_owned()
}
