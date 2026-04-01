//! Event-loop skeleton for the Lanyte orchestrator.

mod model;

use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use lanyte_common::{channels, ChannelId};
use lanyte_gateway::{GatewayEvent, PeerResponder, PeerResponse};
use lanyte_llm::{CompletionRequest, CompletionResponse, LlmBackend, LlmError, StopReason};
use lanyte_telemetry::AuditEvent;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub use model::{
    ActionError, ActionOutcome, ActionStatus, ContentPart, EntityMessage, EntityRef, Envelope,
    GateDecision, GateProposal, MessageIntent, OrchestratorEffect, OrchestratorEvent, SystemNotice,
    TimerKind,
};

const LLM_COMPLETE_COMMAND: &str = "llm.complete";
const DEFAULT_LLM_MAX_TOKENS: u32 = 256;

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("invalid telemetry audit event: {0}")]
    InvalidAuditEvent(&'static str),
}

pub struct Orchestrator {
    events: mpsc::Receiver<GatewayEvent>,
    cancel: CancellationToken,
    responder: PeerResponder,
    llm: Option<Arc<dyn LlmBackend>>,
    started_at: Instant,
    #[cfg(any(test, feature = "test-support"))]
    observer: Option<mpsc::UnboundedSender<test_support::HandlerObservation>>,
}

impl Orchestrator {
    #[must_use]
    pub fn new(
        events: mpsc::Receiver<GatewayEvent>,
        cancel: CancellationToken,
        responder: PeerResponder,
        llm: Option<Arc<dyn LlmBackend>>,
    ) -> Self {
        Self {
            events,
            cancel,
            responder,
            llm,
            started_at: Instant::now(),
            #[cfg(any(test, feature = "test-support"))]
            observer: None,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    #[must_use]
    pub fn with_test_observer(
        mut self,
        observer: mpsc::UnboundedSender<test_support::HandlerObservation>,
    ) -> Self {
        self.observer = Some(observer);
        self
    }

    pub async fn run(mut self) -> Result<(), OrchestratorError> {
        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => {
                    tracing::info!("orchestrator shutting down");
                    return Ok(());
                }
                event = self.events.recv() => {
                    let event = match event {
                        Some(event) => event,
                        None => {
                            tracing::info!("gateway channel closed, shutting down");
                            return Ok(());
                        }
                    };
                    self.dispatch(event).await;
                }
            }
        }
    }

    async fn dispatch(&self, event: GatewayEvent) {
        match event.channel {
            channels::COMMAND => self.handle_command(&event).await,
            channels::TELEMETRY => self.handle_telemetry(&event).await,
            channels::ERROR => self.handle_error(&event).await,
            channels::MAIL => self.handle_mail(&event).await,
            channels::PROXY => self.handle_proxy(&event).await,
            channels::ADMIN => self.handle_admin(&event).await,
            channels::SKILL_IO => self.handle_skill(&event).await,
            unknown => tracing::info!(
                peer_id = %event.peer_id,
                channel = unknown,
                "dropping unknown channel"
            ),
        }
    }

    async fn handle_command(&self, event: &GatewayEvent) {
        self.observe("command", event);
        let command_request: CommandInvokeRequest = match serde_json::from_slice(&event.payload) {
            Ok(request) => request,
            Err(err) => {
                tracing::warn!(
                    peer_id = %event.peer_id,
                    channel = event.channel,
                    error = %err,
                    "failed to deserialize command payload"
                );
                return;
            }
        };

        if command_request.kind != "invoke" {
            tracing::info!(
                peer_id = %event.peer_id,
                channel = event.channel,
                request_type = %command_request.kind,
                "command payload logged and dropped"
            );
            return;
        }

        if command_request.command != LLM_COMPLETE_COMMAND {
            self.send_command_error(
                &event.peer_id,
                CommandInvokeError {
                    kind: "invoke_error",
                    request_id: command_request.request_id,
                    command: command_request.command.clone(),
                    error_code: "unknown_command",
                    message: format!("unsupported command: {}", command_request.command),
                    retryable: false,
                },
            )
            .await;
            return;
        }

        let args: LlmCompleteArgs =
            match serde_json::from_value::<LlmCompleteArgs>(command_request.args.clone()) {
                Ok(args) if !args.prompt.trim().is_empty() && args.max_tokens > 0 => args,
                Ok(_) => {
                    self.send_command_error(
                        &event.peer_id,
                        CommandInvokeError {
                            kind: "invoke_error",
                            request_id: command_request.request_id,
                            command: command_request.command,
                            error_code: "invalid_args",
                            message:
                                "prompt must be non-empty and max_tokens must be greater than zero"
                                    .to_owned(),
                            retryable: false,
                        },
                    )
                    .await;
                    return;
                }
                Err(err) => {
                    self.send_command_error(
                        &event.peer_id,
                        CommandInvokeError {
                            kind: "invoke_error",
                            request_id: command_request.request_id,
                            command: command_request.command,
                            error_code: "invalid_args",
                            message: format!("invalid llm.complete args: {err}"),
                            retryable: false,
                        },
                    )
                    .await;
                    return;
                }
            };

        let ingress = Self::llm_complete_ingress_event(
            event.peer_id.clone(),
            event.channel,
            &command_request,
            &args,
        );
        let completion_effect = match Self::request_completion_effect(&ingress) {
            Ok(effect) => effect,
            Err(err) => {
                self.send_command_error(
                    &event.peer_id,
                    CommandInvokeError {
                        kind: "invoke_error",
                        request_id: command_request.request_id,
                        command: command_request.command,
                        error_code: "internal_error",
                        message: format!(
                            "failed to derive completion request from ingress event: {err}"
                        ),
                        retryable: false,
                    },
                )
                .await;
                return;
            }
        };
        let OrchestratorEffect::RequestCompletion {
            envelope: completion_envelope,
            request: completion_request,
            ..
        } = completion_effect
        else {
            unreachable!("request_completion_effect always returns RequestCompletion");
        };
        let OrchestratorEvent::IngressMessage {
            envelope: ingress_envelope,
            ..
        } = &ingress
        else {
            unreachable!("llm_complete_ingress_event always returns IngressMessage");
        };

        let Some(backend) = self.llm.clone() else {
            self.send_command_error(
                &event.peer_id,
                CommandInvokeError {
                    kind: "invoke_error",
                    request_id: command_request.request_id.clone(),
                    command: command_request.command.clone(),
                    error_code: "internal_error",
                    message: "LLM backend is not configured".to_owned(),
                    retryable: false,
                },
            )
            .await;
            return;
        };

        let backend_name = backend.name();
        let completion =
            match tokio::task::spawn_blocking(move || backend.complete(completion_request)).await {
                Ok(Ok(response)) => response,
                Ok(Err(err)) => {
                    let (error_code, retryable) = map_llm_error(&err);
                    self.send_command_error(
                        &event.peer_id,
                        CommandInvokeError {
                            kind: "invoke_error",
                            request_id: command_request.request_id,
                            command: command_request.command,
                            error_code,
                            message: err.to_string(),
                            retryable,
                        },
                    )
                    .await;
                    return;
                }
                Err(err) => {
                    self.send_command_error(
                        &event.peer_id,
                        CommandInvokeError {
                            kind: "invoke_error",
                            request_id: command_request.request_id,
                            command: command_request.command,
                            error_code: "internal_error",
                            message: format!("LLM task failed to join: {err}"),
                            retryable: true,
                        },
                    )
                    .await;
                    return;
                }
            };

        let reply_effect = Self::emit_message_effect(
            ingress_envelope,
            &completion_envelope,
            backend_name,
            &completion,
        );
        self.send_command_result(
            &event.peer_id,
            CommandInvokeResult {
                kind: "invoke_result",
                request_id: command_request.request_id,
                command: command_request.command,
                result: Self::command_result_payload(&reply_effect, backend_name, &completion),
            },
        )
        .await;
    }

    async fn handle_telemetry(&self, event: &GatewayEvent) {
        self.observe("telemetry", event);
        let payload: Value = match serde_json::from_slice(&event.payload) {
            Ok(payload) => payload,
            Err(err) => {
                tracing::warn!(
                    peer_id = %event.peer_id,
                    channel = event.channel,
                    error = %err,
                    "failed to deserialize telemetry payload"
                );
                return;
            }
        };

        match payload.get("type").and_then(Value::as_str) {
            Some("audit_event") => {
                let audit_event: AuditEvent = match serde_json::from_value(payload) {
                    Ok(event) => event,
                    Err(err) => {
                        tracing::warn!(
                            peer_id = %event.peer_id,
                            channel = event.channel,
                            error = %err,
                            "failed to deserialize audit_event payload"
                        );
                        return;
                    }
                };
                if let Err(err) = audit_event.validate() {
                    tracing::warn!(
                        peer_id = %event.peer_id,
                        channel = event.channel,
                        error = %err,
                        "invalid audit_event payload"
                    );
                    return;
                }
                lanyte_telemetry::emit_audit_event(&audit_event);
            }
            Some("metric_event") | Some("trace_span") | Some("health_report") => {
                tracing::info!(
                    peer_id = %event.peer_id,
                    channel = event.channel,
                    telemetry_type = payload.get("type").and_then(|value| value.as_str()).unwrap_or("unknown"),
                    "telemetry payload logged and dropped"
                );
            }
            other => {
                tracing::warn!(
                    peer_id = %event.peer_id,
                    channel = event.channel,
                    telemetry_type = other.unwrap_or("unknown"),
                    "unknown telemetry payload type"
                );
            }
        }
    }

    async fn handle_error(&self, event: &GatewayEvent) {
        self.observe("error", event);
        tracing::info!(peer_id = %event.peer_id, channel = event.channel, "peer error received");
    }

    async fn handle_mail(&self, event: &GatewayEvent) {
        self.observe("mail", event);
        tracing::info!(peer_id = %event.peer_id, channel = event.channel, "mail event received");
    }

    async fn handle_proxy(&self, event: &GatewayEvent) {
        self.observe("proxy", event);
        tracing::info!(peer_id = %event.peer_id, channel = event.channel, "proxy event received");
    }

    async fn handle_admin(&self, event: &GatewayEvent) {
        self.observe("admin", event);
        tracing::info!(peer_id = %event.peer_id, channel = event.channel, "admin event received");

        let request: AdminStatusRequest = match serde_json::from_slice(&event.payload) {
            Ok(request) => request,
            Err(err) => {
                tracing::warn!(
                    peer_id = %event.peer_id,
                    channel = event.channel,
                    error = %err,
                    "failed to deserialize admin payload"
                );
                return;
            }
        };

        if request.kind != "admin_status_request" {
            tracing::info!(
                peer_id = %event.peer_id,
                channel = event.channel,
                request_type = %request.kind,
                "admin payload logged and dropped"
            );
            return;
        }

        let response = AdminStatusResponse {
            kind: "admin_status_response",
            request_id: request.request_id,
            version: env!("CARGO_PKG_VERSION"),
            uptime_secs: self.started_at.elapsed().as_secs(),
            peer_count: self.responder.peer_count() as u64,
            skill_count: 0,
        };

        let payload = match serde_json::to_vec(&response) {
            Ok(payload) => payload,
            Err(err) => {
                tracing::warn!(
                    peer_id = %event.peer_id,
                    channel = event.channel,
                    error = %err,
                    "failed to serialize admin status response"
                );
                return;
            }
        };

        if let Err(err) = self
            .responder
            .send(PeerResponse {
                peer_id: event.peer_id.clone(),
                channel: channels::ADMIN,
                payload,
            })
            .await
        {
            tracing::warn!(
                peer_id = %event.peer_id,
                channel = event.channel,
                error = %err,
                "failed to send admin status response"
            );
        }
    }

    async fn handle_skill(&self, event: &GatewayEvent) {
        self.observe("skill", event);
        tracing::info!(peer_id = %event.peer_id, channel = event.channel, "skill event received");
    }

    fn llm_complete_ingress_event(
        peer_id: String,
        channel: ChannelId,
        request: &CommandInvokeRequest,
        args: &LlmCompleteArgs,
    ) -> OrchestratorEvent {
        OrchestratorEvent::IngressMessage {
            envelope: Envelope {
                id: Uuid::new_v4(),
                occurred_at: Utc::now(),
                conversation_id: None,
                turn_id: None,
                action_id: None,
                causation_id: None,
                correlation_id: Some(request.request_id.clone()),
                external_ref: Some(request.request_id.clone()),
                source: EntityRef::Peer { peer_id, channel },
                target: Some(EntityRef::System {
                    component: "orchestrator".to_owned(),
                }),
                trust_ref: None,
                gate_ref: None,
            },
            message: EntityMessage {
                intent: MessageIntent::Ask,
                parts: vec![
                    ContentPart::Text {
                        text: args.prompt.clone(),
                    },
                    ContentPart::Structured {
                        value: serde_json::json!({
                            "system_prompt": args.system_prompt,
                            "max_tokens": args.max_tokens,
                        }),
                    },
                ],
            },
        }
    }

    fn request_completion_effect(
        event: &OrchestratorEvent,
    ) -> Result<OrchestratorEffect, &'static str> {
        let OrchestratorEvent::IngressMessage { envelope, message } = event else {
            unreachable!("AGI-005 only derives completion requests from ingress messages");
        };
        let request = completion_request_from_message(message)?;

        Ok(OrchestratorEffect::RequestCompletion {
            envelope: Envelope {
                id: Uuid::new_v4(),
                occurred_at: Utc::now(),
                conversation_id: envelope.conversation_id,
                turn_id: envelope.turn_id,
                action_id: envelope.action_id,
                causation_id: Some(envelope.id),
                correlation_id: envelope.correlation_id.clone(),
                external_ref: envelope.external_ref.clone(),
                source: EntityRef::System {
                    component: "orchestrator".to_owned(),
                },
                target: Some(EntityRef::System {
                    component: "llm".to_owned(),
                }),
                trust_ref: envelope.trust_ref.clone(),
                gate_ref: envelope.gate_ref.clone(),
            },
            request,
        })
    }

    fn emit_message_effect(
        ingress_envelope: &Envelope,
        completion_envelope: &Envelope,
        backend_name: &str,
        completion: &CompletionResponse,
    ) -> OrchestratorEffect {
        OrchestratorEffect::EmitMessage {
            envelope: Envelope {
                id: Uuid::new_v4(),
                occurred_at: Utc::now(),
                conversation_id: ingress_envelope.conversation_id,
                turn_id: ingress_envelope.turn_id,
                action_id: ingress_envelope.action_id,
                causation_id: Some(completion_envelope.id),
                correlation_id: ingress_envelope.correlation_id.clone(),
                external_ref: completion
                    .response_id
                    .clone()
                    .or_else(|| ingress_envelope.external_ref.clone()),
                source: EntityRef::System {
                    component: "orchestrator".to_owned(),
                },
                target: Some(ingress_envelope.source.clone()),
                trust_ref: ingress_envelope.trust_ref.clone(),
                gate_ref: ingress_envelope.gate_ref.clone(),
            },
            message: EntityMessage {
                intent: MessageIntent::DeliverResult,
                parts: vec![
                    ContentPart::Text {
                        text: completion.text.clone(),
                    },
                    ContentPart::Annotation {
                        kind: "llm_response".to_owned(),
                        payload: serde_json::json!({
                            "backend": backend_name,
                            "stop_reason": stop_reason_value(completion.stop_reason),
                            "response_id": completion.response_id,
                            "usage": completion.usage,
                        }),
                    },
                ],
            },
        }
    }

    fn command_result_payload(
        effect: &OrchestratorEffect,
        backend_name: &str,
        completion: &CompletionResponse,
    ) -> Value {
        let OrchestratorEffect::EmitMessage { message, .. } = effect else {
            unreachable!("command replies must be built from EmitMessage effects");
        };

        let mut result = serde_json::json!({
            "backend": backend_name,
            "intent": message.intent,
            "text": extract_text(message),
            "stop_reason": stop_reason_value(completion.stop_reason),
        });

        if let Some(response_id) = &completion.response_id {
            result["response_id"] = Value::String(response_id.clone());
        }
        if let Some(usage) = &completion.usage {
            result["usage"] = serde_json::to_value(usage).expect("usage should serialize");
        }

        result
    }

    async fn send_command_result(&self, peer_id: &str, response: CommandInvokeResult) {
        self.send_json(peer_id, channels::COMMAND, &response, "command result")
            .await;
    }

    async fn send_command_error(&self, peer_id: &str, response: CommandInvokeError) {
        self.send_json(peer_id, channels::COMMAND, &response, "command error")
            .await;
    }

    async fn send_json<T: Serialize>(
        &self,
        peer_id: &str,
        channel: ChannelId,
        response: &T,
        context: &'static str,
    ) {
        let payload = match serde_json::to_vec(response) {
            Ok(payload) => payload,
            Err(err) => {
                tracing::warn!(
                    peer_id = %peer_id,
                    channel,
                    error = %err,
                    context,
                    "failed to serialize response"
                );
                return;
            }
        };

        if let Err(err) = self
            .responder
            .send(PeerResponse {
                peer_id: peer_id.to_owned(),
                channel,
                payload,
            })
            .await
        {
            tracing::warn!(
                peer_id = %peer_id,
                channel,
                error = %err,
                context,
                "failed to send response"
            );
        }
    }

    fn observe(&self, handler: &'static str, event: &GatewayEvent) {
        #[cfg(not(any(test, feature = "test-support")))]
        let _ = (handler, event);

        #[cfg(any(test, feature = "test-support"))]
        if let Some(observer) = &self.observer {
            let _ = observer.send(test_support::HandlerObservation {
                handler,
                peer_id: event.peer_id.clone(),
                channel: event.channel,
            });
        }
    }
}

fn extract_text(message: &EntityMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn completion_request_from_message(
    message: &EntityMessage,
) -> Result<CompletionRequest, &'static str> {
    let prompt = extract_text(message);
    if prompt.trim().is_empty() {
        return Err("ingress message is missing prompt text");
    }

    let options = message
        .parts
        .iter()
        .find_map(|part| match part {
            ContentPart::Structured { value } => Some(value.clone()),
            _ => None,
        })
        .ok_or("ingress message is missing llm.complete options")?;
    let options = serde_json::from_value::<LlmCompleteMessageOptions>(options)
        .map_err(|_| "ingress message has invalid llm.complete options")?;

    Ok(CompletionRequest {
        system_prompt: options.system_prompt,
        previous_response_id: None,
        messages: vec![lanyte_llm::Message::user(prompt)],
        tools: Vec::new(),
        tool_results: Vec::new(),
        max_tokens: Some(options.max_tokens),
        thinking_budget_tokens: None,
        temperature: None,
        parallel_tool_calls: None,
    })
}

fn map_llm_error(err: &LlmError) -> (&'static str, bool) {
    match err {
        LlmError::RateLimited { .. } => ("rate_limited", true),
        LlmError::ServiceUnavailable | LlmError::Http(_) | LlmError::Upstream { .. } => {
            ("internal_error", true)
        }
        LlmError::MissingApiKey
        | LlmError::AuthenticationFailed
        | LlmError::InvalidModel
        | LlmError::Unsupported(_)
        | LlmError::InvalidResponse(_)
        | LlmError::UnsupportedMessageRole(_)
        | LlmError::Json(_) => ("internal_error", false),
    }
}

fn stop_reason_value(stop_reason: StopReason) -> &'static str {
    match stop_reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::ToolUse => "tool_use",
        StopReason::ContentFiltered => "content_filtered",
    }
}

#[derive(Debug, Deserialize)]
struct AdminStatusRequest {
    #[serde(rename = "type")]
    kind: String,
    request_id: String,
}

#[derive(Debug, Serialize)]
struct AdminStatusResponse<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    request_id: String,
    version: &'a str,
    uptime_secs: u64,
    peer_count: u64,
    skill_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct CommandInvokeRequest {
    #[serde(rename = "type")]
    kind: String,
    request_id: String,
    command: String,
    #[serde(default)]
    args: Value,
}

#[derive(Debug, Serialize)]
struct CommandInvokeResult {
    #[serde(rename = "type")]
    kind: &'static str,
    request_id: String,
    command: String,
    result: Value,
}

#[derive(Debug, Serialize)]
struct CommandInvokeError {
    #[serde(rename = "type")]
    kind: &'static str,
    request_id: String,
    command: String,
    error_code: &'static str,
    message: String,
    retryable: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmCompleteArgs {
    prompt: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default = "default_llm_max_tokens")]
    max_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LlmCompleteMessageOptions {
    #[serde(default)]
    system_prompt: Option<String>,
    max_tokens: u32,
}

const fn default_llm_max_tokens() -> u32 {
    DEFAULT_LLM_MAX_TOKENS
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use lanyte_common::ChannelId;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct HandlerObservation {
        pub handler: &'static str,
        pub peer_id: String,
        pub channel: ChannelId,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lanyte_common::channels;
    use lanyte_gateway::{PeerResponder, PeerResponse, PeerSendError};
    use tokio::sync::mpsc;

    fn make_orchestrator() -> (
        Orchestrator,
        mpsc::UnboundedReceiver<test_support::HandlerObservation>,
    ) {
        let (_events_tx, events_rx) = mpsc::channel(4);
        let (obs_tx, obs_rx) = mpsc::unbounded_channel();
        (
            Orchestrator::new(
                events_rx,
                CancellationToken::new(),
                PeerResponder::empty_for_tests(),
                None,
            )
            .with_test_observer(obs_tx),
            obs_rx,
        )
    }

    #[tokio::test]
    async fn dispatch_routes_mail_to_mail_handler() {
        let (orchestrator, mut observations) = make_orchestrator();
        orchestrator
            .dispatch(GatewayEvent {
                peer_id: "peer-1".to_owned(),
                channel: channels::MAIL,
                payload: b"{}".to_vec(),
            })
            .await;

        let observed = observations
            .recv()
            .await
            .expect("mail observation should exist");
        assert_eq!(observed.handler, "mail");
    }

    #[tokio::test]
    async fn dispatch_routes_telemetry_to_telemetry_handler() {
        let (orchestrator, mut observations) = make_orchestrator();
        orchestrator
            .dispatch(GatewayEvent {
                peer_id: "peer-1".to_owned(),
                channel: channels::TELEMETRY,
                payload: serde_json::to_vec(
                    &lanyte_telemetry::AuditEvent::new(
                        "550e8400-e29b-41d4-a716-446655440000",
                        "peer-1",
                        "2026-03-11T12:00:00.000Z",
                        "test.audit",
                        lanyte_telemetry::ZERO_HASH,
                    )
                    .to_payload(),
                )
                .expect("payload should serialize"),
            })
            .await;

        let observed = observations
            .recv()
            .await
            .expect("telemetry observation should exist");
        assert_eq!(observed.handler, "telemetry");
    }

    #[tokio::test]
    async fn unknown_channel_is_dropped() {
        let (orchestrator, mut observations) = make_orchestrator();
        orchestrator
            .dispatch(GatewayEvent {
                peer_id: "peer-1".to_owned(),
                channel: 999,
                payload: Vec::new(),
            })
            .await;

        assert!(observations.try_recv().is_err());
    }

    #[tokio::test]
    async fn run_exits_when_event_channel_closes() {
        let (events_tx, events_rx) = mpsc::channel(1);
        drop(events_tx);

        Orchestrator::new(
            events_rx,
            CancellationToken::new(),
            PeerResponder::empty_for_tests(),
            None,
        )
        .run()
        .await
        .expect("run should exit cleanly");
    }

    #[tokio::test]
    async fn run_exits_on_cancellation() {
        let (_events_tx, events_rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        cancel.cancel();

        Orchestrator::new(events_rx, cancel, PeerResponder::empty_for_tests(), None)
            .run()
            .await
            .expect("run should exit cleanly");
    }

    #[tokio::test]
    async fn admin_send_failure_does_not_halt_dispatch() {
        let (_events_tx, events_rx) = mpsc::channel(1);
        let responder = PeerResponder::channel_closed_for_tests();
        let (obs_tx, _obs_rx) = mpsc::unbounded_channel();
        let orchestrator =
            Orchestrator::new(events_rx, CancellationToken::new(), responder.clone(), None)
                .with_test_observer(obs_tx);

        orchestrator
            .handle_admin(&GatewayEvent {
                peer_id: "peer-1".to_owned(),
                channel: channels::ADMIN,
                payload: serde_json::to_vec(&serde_json::json!({
                    "type": "admin_status_request",
                    "request_id": "550e8400-e29b-41d4-a716-446655440000"
                }))
                .expect("request should serialize"),
            })
            .await;

        let err = responder
            .send(PeerResponse {
                peer_id: "peer-1".to_owned(),
                channel: channels::ADMIN,
                payload: Vec::new(),
            })
            .await
            .expect_err("response channel should be closed");
        assert_eq!(err, PeerSendError::ChannelClosed);
    }

    #[test]
    fn llm_complete_command_builds_request_completion_effect() {
        let command_request = CommandInvokeRequest {
            kind: "invoke".to_owned(),
            request_id: "550e8400-e29b-41d4-a716-446655440000".to_owned(),
            command: LLM_COMPLETE_COMMAND.to_owned(),
            args: serde_json::json!({ "prompt": "Summarize this", "max_tokens": 48 }),
        };
        let args = LlmCompleteArgs {
            prompt: "Summarize this".to_owned(),
            system_prompt: Some("You are concise".to_owned()),
            max_tokens: 48,
        };

        let ingress = Orchestrator::llm_complete_ingress_event(
            "peer-1".to_owned(),
            channels::COMMAND,
            &command_request,
            &args,
        );
        let OrchestratorEvent::IngressMessage { message, .. } = &ingress else {
            panic!("expected ingress message event");
        };
        assert_eq!(extract_text(message), "Summarize this");
        let options = message
            .parts
            .iter()
            .find_map(|part| match part {
                ContentPart::Structured { value } => Some(value.clone()),
                _ => None,
            })
            .expect("structured llm options should be present");
        assert_eq!(
            serde_json::from_value::<LlmCompleteMessageOptions>(options)
                .expect("llm options should deserialize"),
            LlmCompleteMessageOptions {
                system_prompt: Some("You are concise".to_owned()),
                max_tokens: 48,
            }
        );

        let effect = Orchestrator::request_completion_effect(&ingress)
            .expect("completion request should derive from ingress event");

        let OrchestratorEffect::RequestCompletion { envelope, request } = effect else {
            panic!("expected RequestCompletion effect");
        };

        assert_eq!(
            envelope.correlation_id.as_deref(),
            Some(command_request.request_id.as_str())
        );
        assert_eq!(request.system_prompt.as_deref(), Some("You are concise"));
        assert_eq!(request.max_tokens, Some(48));
        assert_eq!(
            request.messages,
            vec![lanyte_llm::Message::user("Summarize this")]
        );
    }

    #[test]
    fn completion_response_becomes_deliver_result_message() {
        let request_envelope = Envelope {
            id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            conversation_id: None,
            turn_id: None,
            action_id: None,
            causation_id: None,
            correlation_id: Some("550e8400-e29b-41d4-a716-446655440000".to_owned()),
            external_ref: Some("550e8400-e29b-41d4-a716-446655440000".to_owned()),
            source: EntityRef::Peer {
                peer_id: "peer-1".to_owned(),
                channel: channels::COMMAND,
            },
            target: Some(EntityRef::System {
                component: "orchestrator".to_owned(),
            }),
            trust_ref: None,
            gate_ref: None,
        };
        let completion_envelope = Envelope {
            id: Uuid::new_v4(),
            occurred_at: Utc::now(),
            conversation_id: None,
            turn_id: None,
            action_id: None,
            causation_id: Some(request_envelope.id),
            correlation_id: Some("550e8400-e29b-41d4-a716-446655440000".to_owned()),
            external_ref: Some("550e8400-e29b-41d4-a716-446655440000".to_owned()),
            source: EntityRef::System {
                component: "orchestrator".to_owned(),
            },
            target: Some(EntityRef::System {
                component: "llm".to_owned(),
            }),
            trust_ref: None,
            gate_ref: None,
        };
        let response = CompletionResponse {
            response_id: Some("resp-1".to_owned()),
            text: "hello back".to_owned(),
            stop_reason: lanyte_llm::StopReason::EndTurn,
            usage: Some(lanyte_llm::Usage {
                input_tokens: 12,
                output_tokens: 3,
                reasoning_tokens: None,
            }),
        };

        let effect = Orchestrator::emit_message_effect(
            &request_envelope,
            &completion_envelope,
            "test-backend",
            &response,
        );
        let OrchestratorEffect::EmitMessage { envelope, message } = effect else {
            panic!("expected EmitMessage effect");
        };

        assert_eq!(
            envelope.correlation_id.as_deref(),
            Some("550e8400-e29b-41d4-a716-446655440000")
        );
        assert_eq!(envelope.causation_id, Some(completion_envelope.id));
        assert_eq!(
            envelope.target,
            Some(EntityRef::Peer {
                peer_id: "peer-1".to_owned(),
                channel: channels::COMMAND,
            })
        );
        assert_eq!(message.intent, MessageIntent::DeliverResult);
        assert_eq!(extract_text(&message), "hello back");
    }
}
