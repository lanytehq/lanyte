//! Event-loop skeleton for the Lanyte orchestrator.

use std::time::Instant;

use lanyte_common::channels;
use lanyte_gateway::{GatewayEvent, PeerResponder, PeerResponse};
use lanyte_telemetry::AuditEvent;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("invalid telemetry audit event: {0}")]
    InvalidAuditEvent(&'static str),
}

pub struct Orchestrator {
    events: mpsc::Receiver<GatewayEvent>,
    cancel: CancellationToken,
    responder: PeerResponder,
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
    ) -> Self {
        Self {
            events,
            cancel,
            responder,
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
        tracing::info!(peer_id = %event.peer_id, channel = event.channel, "command event received");
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

        Orchestrator::new(events_rx, cancel, PeerResponder::empty_for_tests())
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
            Orchestrator::new(events_rx, CancellationToken::new(), responder.clone())
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
}
