#![cfg(unix)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::{env, fs};

use ipcprims::peer::async_connect;
use lanyte_common::{channels, ProviderKind};
use lanyte_gateway::test_support::{spawn_test_gateway, TempGatewayDir};
use lanyte_gateway::{PeerResponse, PeerSendError};
use lanyte_llm::{
    BackendCapabilities, CompletionRequest, CompletionResponse, HealthStatus, LlmBackend, LlmError,
    LlmStream, Usage,
};
use lanyte_orchestrator::{ConfiguredBackends, Orchestrator};
use lanyte_state::{StatePaths, StateStore};
use lanyte_telemetry::AuditRecordKind;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
struct FixedBackend {
    name: &'static str,
    response: CompletionResponse,
}

fn configured_backends(
    default_provider: ProviderKind,
    backends: impl IntoIterator<Item = (ProviderKind, FixedBackend)>,
) -> ConfiguredBackends {
    let backends = backends
        .into_iter()
        .map(|(provider, backend)| (provider, Arc::new(backend) as Arc<dyn LlmBackend>))
        .collect::<BTreeMap<_, _>>();
    ConfiguredBackends::new(default_provider, backends).expect("backends should configure")
}

fn temp_state_root(label: &str) -> std::path::PathBuf {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time should advance")
        .as_nanos();
    env::temp_dir().join(format!(
        "lanyte-orchestrator-{label}-{}-{now}",
        std::process::id()
    ))
}

impl LlmBackend for FixedBackend {
    fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        Ok(self.response.clone())
    }

    fn stream(&self, _request: CompletionRequest) -> Result<LlmStream, LlmError> {
        unimplemented!("AGI-005 tests exercise the synchronous completion path only")
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_streaming: true,
            supports_tool_use: false,
            supports_system_prompt: true,
            supports_parallel_tool_calls: false,
            supports_web_search: false,
            supports_image_generation: false,
            max_context_tokens: 200_000,
        }
    }

    fn health(&self) -> HealthStatus {
        HealthStatus::Healthy
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

#[tokio::test]
async fn mail_event_reaches_mail_handler() {
    let dir = TempGatewayDir::new("orchestrator-mail");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::MAIL]).expect("gateway should spawn");
    let (obs_tx, mut obs_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let orchestrator = Orchestrator::new(events, cancel.clone(), gateway.responder(), None)
        .with_test_observer(obs_tx);
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
    let orchestrator = Orchestrator::new(events, cancel.clone(), gateway.responder(), None);
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

#[tokio::test]
async fn llm_complete_command_round_trip_returns_expected_fields() {
    let dir = TempGatewayDir::new("orchestrator-command-llm");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::COMMAND]).expect("gateway should spawn");
    let cancel = CancellationToken::new();
    let backend = FixedBackend {
        name: "claude",
        response: CompletionResponse {
            response_id: Some("resp-test-1".to_owned()),
            text: "hello from llm".to_owned(),
            stop_reason: lanyte_llm::StopReason::EndTurn,
            usage: Some(Usage {
                input_tokens: 14,
                output_tokens: 4,
                reasoning_tokens: None,
            }),
        },
    };
    let orchestrator = Orchestrator::new(
        events,
        cancel.clone(),
        gateway.responder(),
        Some(configured_backends(
            ProviderKind::Claude,
            [(ProviderKind::Claude, backend)],
        )),
    );
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::COMMAND])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();
    tx.send_json(
        channels::COMMAND,
        &serde_json::json!({
            "type": "invoke",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "command": "llm.complete",
            "args": {
                "prompt": "Say hello",
                "system_prompt": "You are terse",
                "max_tokens": 32
            }
        }),
    )
    .await
    .expect("command frame should send");

    let frame = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("command response should arrive")
        .expect("command response frame should be valid");
    let payload: serde_json::Value =
        serde_json::from_slice(frame.payload.as_ref()).expect("response payload should be JSON");

    assert_eq!(frame.channel, channels::COMMAND);
    assert_eq!(payload["type"], "invoke_result");
    assert_eq!(
        payload["request_id"],
        "11111111-1111-4111-8111-111111111111"
    );
    assert_eq!(payload["command"], "llm.complete");
    assert_eq!(payload["result"]["backend"], "claude");
    assert_eq!(payload["result"]["intent"], "deliver_result");
    assert_eq!(payload["result"]["text"], "hello from llm");
    assert_eq!(payload["result"]["stop_reason"], "end_turn");
    assert_eq!(payload["result"]["response_id"], "resp-test-1");
    assert_eq!(payload["result"]["usage"]["input_tokens"], 14);
    assert_eq!(payload["result"]["usage"]["output_tokens"], 4);

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn llm_complete_command_returns_invoke_error_when_backend_unconfigured() {
    let dir = TempGatewayDir::new("orchestrator-command-no-llm");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::COMMAND]).expect("gateway should spawn");
    let cancel = CancellationToken::new();
    let orchestrator = Orchestrator::new(events, cancel.clone(), gateway.responder(), None);
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::COMMAND])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();
    tx.send_json(
        channels::COMMAND,
        &serde_json::json!({
            "type": "invoke",
            "request_id": "22222222-2222-4222-8222-222222222222",
            "command": "llm.complete",
            "args": {
                "prompt": "Say hello"
            }
        }),
    )
    .await
    .expect("command frame should send");

    let frame = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("command error should arrive")
        .expect("command error frame should be valid");
    let payload: serde_json::Value =
        serde_json::from_slice(frame.payload.as_ref()).expect("response payload should be JSON");

    assert_eq!(frame.channel, channels::COMMAND);
    assert_eq!(payload["type"], "invoke_error");
    assert_eq!(
        payload["request_id"],
        "22222222-2222-4222-8222-222222222222"
    );
    assert_eq!(payload["command"], "llm.complete");
    assert_eq!(payload["error_code"], "internal_error");
    assert_eq!(payload["retryable"], false);
    assert!(payload["message"]
        .as_str()
        .unwrap_or_default()
        .contains("not configured"));

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn llm_complete_command_uses_configured_default_provider() {
    let dir = TempGatewayDir::new("orchestrator-command-default-provider");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::COMMAND]).expect("gateway should spawn");
    let cancel = CancellationToken::new();
    let orchestrator = Orchestrator::new(
        events,
        cancel.clone(),
        gateway.responder(),
        Some(configured_backends(
            ProviderKind::OpenAi,
            [
                (
                    ProviderKind::Claude,
                    FixedBackend {
                        name: "claude",
                        response: CompletionResponse {
                            response_id: Some("resp-claude".to_owned()),
                            text: "from claude".to_owned(),
                            stop_reason: lanyte_llm::StopReason::EndTurn,
                            usage: None,
                        },
                    },
                ),
                (
                    ProviderKind::OpenAi,
                    FixedBackend {
                        name: "openai",
                        response: CompletionResponse {
                            response_id: Some("resp-openai".to_owned()),
                            text: "from openai".to_owned(),
                            stop_reason: lanyte_llm::StopReason::EndTurn,
                            usage: None,
                        },
                    },
                ),
            ],
        )),
    );
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::COMMAND])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();
    tx.send_json(
        channels::COMMAND,
        &serde_json::json!({
            "type": "invoke",
            "request_id": "44444444-4444-4444-8444-444444444444",
            "command": "llm.complete",
            "args": {
                "prompt": "Say hello"
            }
        }),
    )
    .await
    .expect("command frame should send");

    let frame = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("command response should arrive")
        .expect("command response frame should be valid");
    let payload: serde_json::Value =
        serde_json::from_slice(frame.payload.as_ref()).expect("response payload should be JSON");

    assert_eq!(payload["type"], "invoke_result");
    assert_eq!(payload["result"]["backend"], "openai");
    assert_eq!(payload["result"]["text"], "from openai");

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn llm_complete_command_provider_override_chooses_non_default_backend() {
    let dir = TempGatewayDir::new("orchestrator-command-provider-override");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::COMMAND]).expect("gateway should spawn");
    let cancel = CancellationToken::new();
    let orchestrator = Orchestrator::new(
        events,
        cancel.clone(),
        gateway.responder(),
        Some(configured_backends(
            ProviderKind::Claude,
            [
                (
                    ProviderKind::Claude,
                    FixedBackend {
                        name: "claude",
                        response: CompletionResponse {
                            response_id: Some("resp-claude-override".to_owned()),
                            text: "from claude".to_owned(),
                            stop_reason: lanyte_llm::StopReason::EndTurn,
                            usage: None,
                        },
                    },
                ),
                (
                    ProviderKind::OpenAi,
                    FixedBackend {
                        name: "openai",
                        response: CompletionResponse {
                            response_id: Some("resp-openai-override".to_owned()),
                            text: "from openai".to_owned(),
                            stop_reason: lanyte_llm::StopReason::EndTurn,
                            usage: None,
                        },
                    },
                ),
            ],
        )),
    );
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::COMMAND])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();
    tx.send_json(
        channels::COMMAND,
        &serde_json::json!({
            "type": "invoke",
            "request_id": "55555555-5555-4555-8555-555555555555",
            "command": "llm.complete",
            "args": {
                "prompt": "Say hello",
                "provider": "openai"
            }
        }),
    )
    .await
    .expect("command frame should send");

    let frame = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("command response should arrive")
        .expect("command response frame should be valid");
    let payload: serde_json::Value =
        serde_json::from_slice(frame.payload.as_ref()).expect("response payload should be JSON");

    assert_eq!(payload["type"], "invoke_result");
    assert_eq!(payload["result"]["backend"], "openai");
    assert_eq!(payload["result"]["text"], "from openai");

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn llm_complete_command_returns_invoke_error_when_provider_is_unconfigured() {
    let dir = TempGatewayDir::new("orchestrator-command-provider-unconfigured");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::COMMAND]).expect("gateway should spawn");
    let cancel = CancellationToken::new();
    let orchestrator = Orchestrator::new(
        events,
        cancel.clone(),
        gateway.responder(),
        Some(configured_backends(
            ProviderKind::Claude,
            [(
                ProviderKind::Claude,
                FixedBackend {
                    name: "claude",
                    response: CompletionResponse {
                        response_id: Some("resp-only-claude".to_owned()),
                        text: "from claude".to_owned(),
                        stop_reason: lanyte_llm::StopReason::EndTurn,
                        usage: None,
                    },
                },
            )],
        )),
    );
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::COMMAND])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();
    tx.send_json(
        channels::COMMAND,
        &serde_json::json!({
            "type": "invoke",
            "request_id": "66666666-6666-4666-8666-666666666666",
            "command": "llm.complete",
            "args": {
                "prompt": "Say hello",
                "provider": "openai"
            }
        }),
    )
    .await
    .expect("command frame should send");

    let frame = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("command error should arrive")
        .expect("command error frame should be valid");
    let payload: serde_json::Value =
        serde_json::from_slice(frame.payload.as_ref()).expect("response payload should be JSON");

    assert_eq!(payload["type"], "invoke_error");
    assert_eq!(payload["error_code"], "invalid_args");
    assert_eq!(payload["retryable"], false);
    assert!(payload["message"]
        .as_str()
        .unwrap_or_default()
        .contains("openai"));

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn llm_complete_command_returns_invoke_error_when_provider_is_invalid() {
    let dir = TempGatewayDir::new("orchestrator-command-provider-invalid");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::COMMAND]).expect("gateway should spawn");
    let cancel = CancellationToken::new();
    let orchestrator = Orchestrator::new(events, cancel.clone(), gateway.responder(), None);
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::COMMAND])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();
    tx.send_json(
        channels::COMMAND,
        &serde_json::json!({
            "type": "invoke",
            "request_id": "77777777-7777-4777-8777-777777777777",
            "command": "llm.complete",
            "args": {
                "prompt": "Say hello",
                "provider": "bogus"
            }
        }),
    )
    .await
    .expect("command frame should send");

    let frame = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("command error should arrive")
        .expect("command error frame should be valid");
    let payload: serde_json::Value =
        serde_json::from_slice(frame.payload.as_ref()).expect("response payload should be JSON");

    assert_eq!(payload["type"], "invoke_error");
    assert_eq!(payload["error_code"], "invalid_args");
    assert!(payload["message"]
        .as_str()
        .unwrap_or_default()
        .contains("provider"));

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
}

#[tokio::test]
async fn llm_complete_command_appends_runtime_audit_records() {
    let dir = TempGatewayDir::new("orchestrator-command-llm-audit");
    let (gateway, events) =
        spawn_test_gateway(&dir, &[channels::COMMAND]).expect("gateway should spawn");
    let state_root = temp_state_root("audit");
    let audit_store = Arc::new(Mutex::new(
        StateStore::open(StatePaths::new(&state_root)).expect("state store should open"),
    ));
    let cancel = CancellationToken::new();
    let backend = FixedBackend {
        name: "claude",
        response: CompletionResponse {
            response_id: Some("resp-test-2".to_owned()),
            text: "audited hello".to_owned(),
            stop_reason: lanyte_llm::StopReason::EndTurn,
            usage: Some(Usage {
                input_tokens: 8,
                output_tokens: 2,
                reasoning_tokens: None,
            }),
        },
    };
    let orchestrator = Orchestrator::new(
        events,
        cancel.clone(),
        gateway.responder(),
        Some(configured_backends(
            ProviderKind::Claude,
            [(ProviderKind::Claude, backend)],
        )),
    )
    .with_audit_store(audit_store.clone());
    let orchestrator_task = tokio::spawn(orchestrator.run());

    let client = async_connect(dir.socket_path(), &[channels::COMMAND])
        .await
        .expect("client should connect");
    let (tx, mut rx) = client.into_split();
    tx.send_json(
        channels::COMMAND,
        &serde_json::json!({
            "type": "invoke",
            "request_id": "33333333-3333-4333-8333-333333333333",
            "command": "llm.complete",
            "args": {
                "prompt": "Say hello",
                "max_tokens": 32
            }
        }),
    )
    .await
    .expect("command frame should send");

    let _frame = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("command response should arrive")
        .expect("command response frame should be valid");

    let records = audit_store
        .lock()
        .expect("audit store lock should succeed")
        .audit_records("33333333-3333-4333-8333-333333333333")
        .expect("audit records should load");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].kind, AuditRecordKind::Effect);
    assert_eq!(records[1].kind, AuditRecordKind::Outcome);
    assert_eq!(records[1].prev_hash, records[0].entry_hash);

    cancel.cancel();
    gateway.cancel();
    orchestrator_task
        .await
        .expect("orchestrator task should join")
        .expect("orchestrator should exit cleanly");
    gateway.wait().await.expect("gateway should exit");
    let _ = fs::remove_dir_all(&state_root);
}
