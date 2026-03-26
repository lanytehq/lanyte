#![cfg(unix)]

use std::process::Command;
use std::time::Duration;

use ipcprims::peer::async_connect;
use lanyte_common::channels;
use lanyte_gateway::test_support::{spawn_test_gateway, write_all_peer_schemas, TempGatewayDir};
use serde_json::json;

#[tokio::test]
async fn gateway_accepts_peer_and_forwards_validated_frames() {
    let dir = TempGatewayDir::new("forward");
    let (handle, mut events) =
        spawn_test_gateway(&dir, &[channels::MAIL]).expect("spawn should succeed");

    let client = async_connect(dir.socket_path(), &[channels::MAIL])
        .await
        .expect("client should connect");
    let (tx, _rx) = client.into_split();

    tx.send_json(channels::MAIL, &json!({})).await.unwrap();

    let ev = tokio::time::timeout(Duration::from_secs(2), events.recv())
        .await
        .expect("event should arrive")
        .expect("channel should be open");

    assert_eq!(ev.peer_id, "peer-1");
    assert_eq!(ev.channel, channels::MAIL);
    assert_eq!(ev.payload, b"{}".to_vec());

    handle.cancel();
    handle.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn gateway_disconnects_peer_when_schema_is_missing() {
    let dir = TempGatewayDir::new("missing-schema");
    let (handle, mut events) = spawn_test_gateway(&dir, &[]).expect("spawn should succeed");

    let client = async_connect(dir.socket_path(), &[channels::PROXY])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();

    tx.send_json(channels::PROXY, &json!({})).await.unwrap();

    let maybe = tokio::time::timeout(Duration::from_millis(200), events.recv()).await;
    assert!(maybe.is_err(), "no gateway event should be forwarded");

    let res = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await;
    assert!(res.is_ok(), "disconnect should arrive");

    handle.cancel();
    handle.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn gateway_advertises_core_peer_id_to_clients() {
    let dir = TempGatewayDir::new("core-peer-id");
    let (handle, _events) =
        spawn_test_gateway(&dir, &[channels::MAIL]).expect("spawn should succeed");

    let client = async_connect(dir.socket_path(), &[channels::MAIL])
        .await
        .expect("client should connect");

    assert_eq!(client.id(), "lanyte-core");

    handle.cancel();
    handle.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn mock_mlvoy_binary_connects_and_forwards_mail_frames() {
    let dir = TempGatewayDir::new("mock-mlvoy");
    write_all_peer_schemas(&dir.schemas_dir());
    let (handle, mut events) = spawn_test_gateway(&dir, &[]).expect("spawn should succeed");

    let socket_path = dir.socket_path();
    let status = tokio::task::spawn_blocking(move || {
        Command::new(env!("CARGO_BIN_EXE_mock_mlvoy"))
            .arg(socket_path)
            .status()
    })
    .await
    .expect("command task should join")
    .expect("mock mlvoy should launch");

    assert!(status.success(), "mock mlvoy should exit successfully");

    let ev = tokio::time::timeout(Duration::from_secs(2), events.recv())
        .await
        .expect("event should arrive")
        .expect("channel should be open");

    let payload: serde_json::Value =
        serde_json::from_slice(&ev.payload).expect("payload should be valid JSON");
    assert_eq!(ev.peer_id, "peer-1");
    assert_eq!(ev.channel, channels::MAIL);
    assert_eq!(payload["type"], "mail_search_request");
    assert_eq!(payload["delegation_id"], "dev-test");

    handle.cancel();
    handle.wait().await.expect("gateway should exit");
}
