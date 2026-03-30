#![cfg(unix)]

use std::process::Command;
use std::time::Duration;

use ipcprims::peer::async_connect;
use lanyte_common::channels;
use lanyte_gateway::test_support::{
    spawn_test_gateway, write_all_peer_schemas, write_schema, TempGatewayDir,
};
use serde_json::json;

async fn recv_event(
    events: &mut tokio::sync::mpsc::Receiver<lanyte_gateway::GatewayEvent>,
) -> lanyte_gateway::GatewayEvent {
    tokio::time::timeout(Duration::from_secs(2), events.recv())
        .await
        .expect("event should arrive")
        .expect("channel should be open")
}

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

    let ev = recv_event(&mut events).await;

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
async fn gateway_disconnects_peer_when_payload_fails_schema_validation() {
    const STRICT_MAIL_SCHEMA: &str = r#"{
        "type":"object",
        "properties":{"kind":{"type":"string"}},
        "required":["kind"],
        "additionalProperties":false
    }"#;

    let dir = TempGatewayDir::new("invalid-schema");
    write_schema(&dir.schemas_dir(), channels::MAIL, STRICT_MAIL_SCHEMA);
    let (handle, mut events) = spawn_test_gateway(&dir, &[]).expect("spawn should succeed");

    let client = async_connect(dir.socket_path(), &[channels::MAIL])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();

    tx.send_json(channels::MAIL, &json!({}))
        .await
        .expect("invalid payload should still send over the wire");

    let maybe = tokio::time::timeout(Duration::from_millis(200), events.recv()).await;
    assert!(maybe.is_err(), "invalid payload should not be forwarded");

    let disconnect = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("disconnect should arrive")
        .expect_err("schema validation failure should disconnect peer");
    let _ = disconnect;

    tokio::time::timeout(Duration::from_secs(2), async {
        while handle.peer_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("invalid peer should be removed");

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

    let ev = recv_event(&mut events).await;

    let payload: serde_json::Value =
        serde_json::from_slice(&ev.payload).expect("payload should be valid JSON");
    assert_eq!(ev.peer_id, "peer-1");
    assert_eq!(ev.channel, channels::MAIL);
    assert_eq!(payload["type"], "mail_search_request");
    assert_eq!(payload["delegation_id"], "dev-test");

    handle.cancel();
    handle.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn gateway_routes_events_from_multiple_peers() {
    let dir = TempGatewayDir::new("multi-peer");
    let (handle, mut events) =
        spawn_test_gateway(&dir, &[channels::MAIL, channels::ADMIN]).expect("spawn should succeed");

    let client_one = async_connect(dir.socket_path(), &[channels::MAIL])
        .await
        .expect("first client should connect");
    let (tx_one, _rx_one) = client_one.into_split();

    let client_two = async_connect(dir.socket_path(), &[channels::ADMIN])
        .await
        .expect("second client should connect");
    let (tx_two, _rx_two) = client_two.into_split();

    tx_one
        .send_json(channels::MAIL, &json!({"source":"one"}))
        .await
        .expect("mail frame should send");
    tx_two
        .send_json(channels::ADMIN, &json!({"source":"two"}))
        .await
        .expect("admin frame should send");

    let first = recv_event(&mut events).await;
    let second = recv_event(&mut events).await;
    let observed = [first, second];

    assert!(observed.iter().any(|event| {
        event.peer_id == "peer-1"
            && event.channel == channels::MAIL
            && event.payload == br#"{"source":"one"}"#.to_vec()
    }));
    assert!(observed.iter().any(|event| {
        event.peer_id == "peer-2"
            && event.channel == channels::ADMIN
            && event.payload == br#"{"source":"two"}"#.to_vec()
    }));

    handle.cancel();
    handle.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn gateway_removes_peer_after_disconnect() {
    let dir = TempGatewayDir::new("peer-cleanup");
    let (handle, _events) =
        spawn_test_gateway(&dir, &[channels::MAIL]).expect("spawn should succeed");

    let client = async_connect(dir.socket_path(), &[channels::MAIL])
        .await
        .expect("client should connect");
    let (tx, rx) = client.into_split();

    tokio::time::timeout(Duration::from_secs(2), async {
        while handle.peer_count() != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("peer should register");

    drop(tx);
    drop(rx);

    tokio::time::timeout(Duration::from_secs(2), async {
        while handle.peer_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("peer should be removed after disconnect");

    handle.cancel();
    handle.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn gateway_shutdown_disconnects_connected_peers() {
    let dir = TempGatewayDir::new("structured-shutdown");
    let (handle, _events) =
        spawn_test_gateway(&dir, &[channels::MAIL]).expect("spawn should succeed");

    let client = async_connect(dir.socket_path(), &[channels::MAIL])
        .await
        .expect("client should connect");
    let (_tx, mut rx) = client.into_split();

    handle.cancel();

    let err = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("peer disconnect should arrive")
        .expect_err("gateway shutdown should disconnect peers");
    assert!(
        err.to_string().contains("disconnected"),
        "shutdown should surface peer disconnect: {err}"
    );

    handle.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn gateway_stops_when_downstream_channel_closes() {
    let dir = TempGatewayDir::new("downstream-closed");
    let (handle, events) =
        spawn_test_gateway(&dir, &[channels::MAIL]).expect("spawn should succeed");
    drop(events);

    let client = async_connect(dir.socket_path(), &[channels::MAIL])
        .await
        .expect("client should connect");
    let (tx, _rx) = client.into_split();
    tx.send_json(channels::MAIL, &json!({"shutdown":"trigger"}))
        .await
        .expect("mail frame should send");

    tokio::time::timeout(Duration::from_secs(2), handle.wait())
        .await
        .expect("gateway should stop after downstream closes")
        .expect("gateway should exit cleanly");
}
