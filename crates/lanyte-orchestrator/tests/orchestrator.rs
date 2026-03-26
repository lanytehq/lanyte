#![cfg(unix)]

use std::time::Duration;

use ipcprims::peer::async_connect;
use lanyte_common::channels;
use lanyte_gateway::test_support::{spawn_test_gateway, TempGatewayDir};
use lanyte_gateway::{PeerResponse, PeerSendError};
use lanyte_orchestrator::Orchestrator;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn mail_event_reaches_mail_handler() {
    let dir = TempGatewayDir::new("orchestrator-mail");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::MAIL]).expect("gateway should spawn");
    let (obs_tx, mut obs_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let orchestrator =
        Orchestrator::new(events, cancel.clone(), gateway.responder()).with_test_observer(obs_tx);
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::MAIL])
        .await
        .expect("client should connect");
    let (tx, _rx) = client.into_split();
    tx.send_json(
        channels::MAIL,
        &serde_json::json!({
            "type": "mail_search_request",
            "request_id": "123e4567-e89b-42d3-a456-426614174000",
            "delegation_id": "dev-test",
            "query": "hello"
        }),
    )
    .await
    .expect("mail frame should send");

    let observed = tokio::time::timeout(Duration::from_secs(2), obs_rx.recv())
        .await
        .expect("mail observation should arrive")
        .expect("mail observation should exist");

    assert_eq!(observed.handler, "mail");
    assert_eq!(observed.channel, channels::MAIL);

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn admin_status_request_round_trip_returns_expected_fields() {
    let dir = TempGatewayDir::new("orchestrator-admin");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::ADMIN]).expect("gateway should spawn");
    let cancel = CancellationToken::new();
    let orchestrator = Orchestrator::new(events, cancel.clone(), gateway.responder());
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::ADMIN])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();
    tx.send_json(
        channels::ADMIN,
        &serde_json::json!({
            "type": "admin_status_request",
            "request_id": "550e8400-e29b-41d4-a716-446655440000"
        }),
    )
    .await
    .expect("admin request should send");

    let frame = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("admin response should arrive")
        .expect("admin response frame should be valid");
    let payload: serde_json::Value =
        serde_json::from_slice(frame.payload.as_ref()).expect("response payload should be JSON");

    assert_eq!(frame.channel, channels::ADMIN);
    assert_eq!(payload["type"], "admin_status_response");
    assert_eq!(
        payload["request_id"],
        "550e8400-e29b-41d4-a716-446655440000"
    );
    assert_eq!(payload["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(payload["skill_count"], 0);
    assert!(
        payload["peer_count"]
            .as_u64()
            .expect("peer_count should be u64")
            >= 1
    );
    assert!(
        payload["uptime_secs"]
            .as_u64()
            .expect("uptime_secs should be u64")
            < 5
    );
    assert!(payload.get("gate_pending").is_none());
    assert!(payload.get("healthy").is_none());

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn sending_to_unknown_peer_returns_typed_error() {
    let dir = TempGatewayDir::new("orchestrator-missing-peer");
    let (gateway, _events) =
        spawn_test_gateway(&dir, &[channels::ADMIN]).expect("gateway should spawn");

    let err = gateway
        .responder()
        .send(PeerResponse {
            peer_id: "missing-peer".to_owned(),
            channel: channels::ADMIN,
            payload: b"{}".to_vec(),
        })
        .await
        .expect_err("missing peer should return an error");

    assert_eq!(err, PeerSendError::UnknownPeer("missing-peer".to_owned()));

    gateway.cancel();
    gateway.wait().await.expect("gateway should exit");
}
