use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::sync::Mutex;

use chrono::Utc;
use lanyte_driver_codex::{confine_workspace, CodexAppServerDriver, CodexLaunchSpec, CodexSession};
use lanyte_gateway::GatewayEvent;
use lanyte_mission::{
    AttemptRecord, AttemptState, AttemptTerminalReason, CapabilityName, EventSource,
    EventSourceKind, HarnessDriver, HarnessSelection, LifecycleEvent, LifecyclePayload,
    MissionControlRequest, MissionControlResult, MissionPhase, MissionTerminalReason,
    MissionTransition, ObservationLevel, PrincipalRef, RecoveryRelation, LIFECYCLE_EVENT_SCHEMA,
};
use lanyte_state::NewMissionProjectionReceipt;
use lanyte_telemetry::AuditEnvelopeRef;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{CommandInvokeError, CommandInvokeRequest, Orchestrator};
use crate::mission::{caller_principal, MissionCommandError};

const CODEX_DRIVER_ID: &str = "driver.codex.app_server";

fn live_sessions() -> &'static Mutex<HashMap<Uuid, CodexSession>> {
    static LIVE: OnceLock<Mutex<HashMap<Uuid, CodexSession>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

impl Orchestrator {
    pub(super) async fn handle_mission_launch(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
    ) {
        match self.parse_control(&command_request) {
            Ok(request) => {
                self.reply_control(
                    event,
                    command_request,
                    self.launch_codex(event, request).await,
                )
                .await
            }
            Err(err) => self.reply_control_error(event, command_request, err).await,
        }
    }

    pub(super) async fn handle_mission_observe(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
    ) {
        match self.parse_control(&command_request) {
            Ok(request) => {
                self.reply_control(
                    event,
                    command_request,
                    self.observe_codex(event, request).await,
                )
                .await
            }
            Err(err) => self.reply_control_error(event, command_request, err).await,
        }
    }

    pub(super) async fn handle_mission_close(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
    ) {
        match self.parse_control(&command_request) {
            Ok(request) => {
                self.reply_control(
                    event,
                    command_request,
                    self.close_codex(event, request).await,
                )
                .await
            }
            Err(err) => self.reply_control_error(event, command_request, err).await,
        }
    }

    fn parse_control(
        &self,
        command_request: &CommandInvokeRequest,
    ) -> Result<MissionControlRequest, MissionCommandError> {
        let request: MissionControlRequest = serde_json::from_value(command_request.args.clone())
            .map_err(|err| {
            MissionCommandError::invalid_args(format!("invalid mission control request: {err}"))
        })?;
        if request.operation() != command_request.command
            || request.request_id().to_string() != command_request.request_id
        {
            return Err(MissionCommandError::invalid_args(
                "outer command/request_id must match the mission control payload",
            ));
        }
        Ok(request)
    }

    async fn reply_control(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
        outcome: Result<MissionControlResult, MissionCommandError>,
    ) {
        match outcome {
            Ok(result) => match serde_json::to_value(&result) {
                Ok(result) => {
                    self.send_command_result(
                        &event.peer_id,
                        super::CommandInvokeResult {
                            kind: "invoke_result",
                            request_id: command_request.request_id,
                            command: command_request.command,
                            result,
                        },
                    )
                    .await;
                }
                Err(err) => {
                    self.reply_control_error(
                        event,
                        command_request,
                        MissionCommandError::internal(format!(
                            "failed to encode mission result: {err}"
                        )),
                    )
                    .await;
                }
            },
            Err(err) => self.reply_control_error(event, command_request, err).await,
        }
    }

    async fn reply_control_error(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
        err: MissionCommandError,
    ) {
        self.send_command_error(
            &event.peer_id,
            CommandInvokeError {
                kind: "invoke_error",
                request_id: command_request.request_id,
                command: command_request.command,
                error_code: err.code.as_str(),
                message: err.message,
                retryable: false,
            },
        )
        .await;
    }

    async fn launch_codex(
        &self,
        event: &GatewayEvent,
        request: MissionControlRequest,
    ) -> Result<MissionControlResult, MissionCommandError> {
        let MissionControlRequest::Launch {
            request_id,
            idempotency_key,
            expected_revision,
            body,
        } = request
        else {
            return Err(MissionCommandError::internal(
                "launch handler received a non-launch control request",
            ));
        };
        let Some(service) = &self.mission_service else {
            return Err(MissionCommandError::internal(
                "mission service is not configured",
            ));
        };
        let caller = service.authenticate(
            event
                .client_auth_token
                .as_ref()
                .map(lanyte_gateway::ClientAuthToken::expose),
        )?;
        let fingerprint = mutation_fingerprint(
            "mission.launch",
            &caller,
            &serde_json::to_value(&body).unwrap_or_default(),
        )?;
        if let Some(replayed) = service.replay_mutation(&idempotency_key, &fingerprint)? {
            return replayed_control_result(&replayed);
        }
        let stored = service.visible_projection(&body.mission_id.to_string(), &caller)?;
        let before = stored.mission.clone();
        if before.revision != expected_revision {
            return Err(MissionCommandError::invalid_args(format!(
                "stale mission revision: expected {expected_revision}, actual {}",
                before.revision
            )));
        }
        if before.phase != MissionPhase::Created {
            return Err(MissionCommandError::invalid_args(
                "mission.launch requires a created mission with no live attempt",
            ));
        }
        let paths = service.state_paths()?;
        let workspace = confine_workspace(
            PathBuf::from(&body.workspace).as_path(),
            &paths.workspace_root(),
        )
        .map_err(|err| MissionCommandError::invalid_args(err.to_string()))?;
        let driver = CodexAppServerDriver::new(CodexLaunchSpec {
            workspace: workspace.clone(),
            allowed_root: paths.workspace_root(),
            pin_dir: paths.pin_dir(),
            binary_path: body.binary.map(PathBuf::from),
        });
        let descriptor = driver.descriptor();
        driver
            .capabilities()
            .require_usable_at(
                Utc::now(),
                &descriptor,
                std::env::consts::OS,
                &[
                    CapabilityName::Create,
                    CapabilityName::Identify,
                    CapabilityName::Observe,
                    CapabilityName::Close,
                ],
            )
            .map_err(|err| MissionCommandError::invalid_args(err.to_string()))?;
        let attempt_id = Uuid::new_v4();
        let session = driver
            .create(attempt_id)
            .await
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        let now = Utc::now();
        let authorizer = caller_principal(&caller);
        let mut mission = before.clone();
        mission.revision = expected_revision + 1;
        mission.updated_at = now;
        mission.phase = MissionPhase::Active;
        mission.authorizer = Some(authorizer.clone());
        mission.authorization_ref = Some(caller.trust_ref.clone());
        mission.harness_selection = Some(HarnessSelection {
            harness_id: "codex".to_owned(),
            driver_id: CODEX_DRIVER_ID.to_owned(),
            model: None,
            workspace_ref: workspace.display().to_string(),
            environment_ref: None,
        });
        mission.current_attempt_id = Some(attempt_id);
        mission.attempts.push(AttemptRecord {
            attempt_id,
            ordinal: 1,
            generation: 1,
            fencing_token_sha256: format!("{:x}", Sha256::digest(attempt_id.as_bytes())),
            recovery_relation: RecoveryRelation::Initial,
            predecessor_attempt_id: None,
            state: AttemptState::Running,
            driver_id: Some(CODEX_DRIVER_ID.to_owned()),
            harness_session_id: Some(session.harness_session_id.clone()),
            started_at: Some(now),
            ended_at: None,
            terminal_reason: None,
            evidence_ref: Some(format!(
                "codex:{}:{}",
                session.binary.version, session.binary.digest
            )),
        });
        if let Err(err) = (MissionTransition {
            expected_revision,
            from: MissionPhase::Created,
            to: MissionPhase::Active,
        })
        .check(&before, &mission)
        {
            let mut session = session;
            let _ = session.close().await;
            return Err(MissionCommandError::invalid_args(err.to_string()));
        }

        let history = service.lifecycle_history(&body.mission_id.to_string())?;
        let report = driver.capabilities();
        let payloads = vec![
            LifecyclePayload::AuthorizationBound {
                authorizer: PrincipalRef {
                    kind: authorizer.kind,
                    subject: authorizer.subject.clone(),
                    attestation_ref: caller.trust_ref.clone(),
                },
            },
            LifecyclePayload::MissionPhaseChanged {
                from: MissionPhase::Created,
                to: MissionPhase::Active,
                reason: Some("codex launch".to_owned()),
            },
            LifecyclePayload::AttemptCreated {
                attempt_id,
                ordinal: 1,
                generation: 1,
                recovery_relation: RecoveryRelation::Initial,
                predecessor_attempt_id: None,
            },
            LifecyclePayload::DriverCapabilityEvaluated {
                attempt_id,
                generation: 1,
                driver_id: CODEX_DRIVER_ID.to_owned(),
                capability: CapabilityName::Create,
                availability: report.availability,
                fidelity: lanyte_mission::CapabilityFidelity::Native,
                report_id: report.report_id,
            },
            LifecyclePayload::AttemptStateChanged {
                attempt_id,
                generation: 1,
                from: AttemptState::Starting,
                to: AttemptState::Running,
                reason: Some("thread/start".to_owned()),
            },
        ];
        let receipts = chain_receipts(
            &history,
            &mission,
            &caller,
            EventSourceKind::VerifiedAttestation,
            payloads,
        )?;
        let result = MissionControlResult::launch(
            request_id,
            idempotency_key.clone(),
            expected_revision,
            mission.clone(),
        )
        .map_err(MissionCommandError::internal)?;
        if let Err(err) = service.persist_update_events(
            expected_revision,
            mission,
            receipts,
            Some(lanyte_state::MissionMutationIdempotency {
                key: idempotency_key,
                request_fingerprint: fingerprint,
                operation: "mission.launch".to_owned(),
                result_json: serde_json::to_string(&result)
                    .map_err(|err| MissionCommandError::internal(err.to_string()))?,
            }),
        ) {
            let mut session = session;
            let _ = session.close().await;
            return Err(err);
        }

        live_sessions().lock().await.insert(attempt_id, session);
        Ok(result)
    }

    async fn observe_codex(
        &self,
        event: &GatewayEvent,
        request: MissionControlRequest,
    ) -> Result<MissionControlResult, MissionCommandError> {
        let MissionControlRequest::Observe {
            request_id, body, ..
        } = request
        else {
            return Err(MissionCommandError::internal(
                "observe handler received a non-observe control request",
            ));
        };
        let Some(service) = &self.mission_service else {
            return Err(MissionCommandError::internal(
                "mission service is not configured",
            ));
        };
        let caller = service.authenticate(
            event
                .client_auth_token
                .as_ref()
                .map(lanyte_gateway::ClientAuthToken::expose),
        )?;
        let mission = service.visible_mission(&body.mission_id.to_string(), &caller)?;
        let attempt_id = mission
            .current_attempt_id
            .ok_or_else(|| MissionCommandError::invalid_args("mission has no live attempt"))?;
        let mut sessions = live_sessions().lock().await;
        let session = sessions.get_mut(&attempt_id).ok_or_else(|| {
            MissionCommandError::invalid_args("codex session is not in this kernel")
        })?;
        let mut events = Vec::new();
        while let Some(event) = session
            .observe()
            .await
            .map_err(|err| MissionCommandError::internal(err.to_string()))?
        {
            events.push(event);
            if events.len() >= 32 {
                break;
            }
        }
        MissionControlResult::observe(request_id, body.mission_id, attempt_id, events)
            .map_err(MissionCommandError::internal)
    }

    async fn close_codex(
        &self,
        event: &GatewayEvent,
        request: MissionControlRequest,
    ) -> Result<MissionControlResult, MissionCommandError> {
        let MissionControlRequest::Close {
            request_id,
            idempotency_key,
            expected_revision,
            body,
        } = request
        else {
            return Err(MissionCommandError::internal(
                "close handler received a non-close control request",
            ));
        };
        let Some(service) = &self.mission_service else {
            return Err(MissionCommandError::internal(
                "mission service is not configured",
            ));
        };
        let caller = service.authenticate(
            event
                .client_auth_token
                .as_ref()
                .map(lanyte_gateway::ClientAuthToken::expose),
        )?;
        let fingerprint = mutation_fingerprint(
            "mission.close",
            &caller,
            &serde_json::json!({ "mission_id": body.mission_id }),
        )?;
        if let Some(replayed) = service.replay_mutation(&idempotency_key, &fingerprint)? {
            return replayed_control_result(&replayed);
        }
        let stored = service.visible_projection(&body.mission_id.to_string(), &caller)?;
        let before = stored.mission.clone();
        if before.revision != expected_revision {
            return Err(MissionCommandError::invalid_args(format!(
                "stale mission revision: expected {expected_revision}, actual {}",
                before.revision
            )));
        }
        let attempt_id = before
            .current_attempt_id
            .ok_or_else(|| MissionCommandError::invalid_args("mission has no live attempt"))?;
        let mut sessions = live_sessions().lock().await;
        let mut session = sessions.remove(&attempt_id).ok_or_else(|| {
            MissionCommandError::invalid_args(
                "codex session is not in this kernel; close will not claim cancellation",
            )
        })?;
        session
            .close()
            .await
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        drop(sessions);

        let now = Utc::now();
        let mut mission = before.clone();
        mission.revision = expected_revision + 1;
        mission.updated_at = now;
        mission.phase = MissionPhase::Cancelled;
        mission.terminal_reason = Some(MissionTerminalReason::OperatorCancelled);
        mission.current_attempt_id = None;
        if let Some(attempt) = mission
            .attempts
            .iter_mut()
            .find(|attempt| attempt.attempt_id == attempt_id)
        {
            attempt.state = AttemptState::Completed;
            attempt.ended_at = Some(now);
            attempt.terminal_reason = Some(AttemptTerminalReason::OperatorCancelled);
        }
        MissionTransition {
            expected_revision,
            from: MissionPhase::Active,
            to: MissionPhase::Cancelled,
        }
        .check(&before, &mission)
        .map_err(|err| MissionCommandError::invalid_args(err.to_string()))?;

        let history = service.lifecycle_history(&body.mission_id.to_string())?;
        let payloads = vec![
            LifecyclePayload::AttemptStateChanged {
                attempt_id,
                generation: 1,
                from: AttemptState::Running,
                to: AttemptState::Completed,
                reason: Some("operator close".to_owned()),
            },
            LifecyclePayload::MissionPhaseChanged {
                from: MissionPhase::Active,
                to: MissionPhase::Cancelled,
                reason: Some("operator close".to_owned()),
            },
            LifecyclePayload::MissionTerminal {
                phase: MissionPhase::Cancelled,
                reason: MissionTerminalReason::OperatorCancelled,
                terminal_entry_hash: "0".repeat(64),
            },
        ];
        let mut receipts = chain_receipts(
            &history,
            &mission,
            &caller,
            EventSourceKind::OperatorCommand,
            payloads.clone(),
        )?;
        if let Some(last) = receipts.last_mut() {
            if let LifecyclePayload::MissionTerminal {
                terminal_entry_hash,
                ..
            } = &mut last.event.payload
            {
                *terminal_entry_hash = last.event.entry_hash.clone();
            }
            last.event.entry_hash = hash_lifecycle(&last.event)?;
            mission.terminal_entry_hash = Some(last.event.entry_hash.clone());
        }
        let _ = payloads;
        let result = MissionControlResult::close(
            request_id,
            idempotency_key.clone(),
            expected_revision,
            body.mission_id,
            attempt_id,
        )
        .map_err(MissionCommandError::internal)?;
        service.persist_update_events(
            expected_revision,
            mission,
            receipts,
            Some(lanyte_state::MissionMutationIdempotency {
                key: idempotency_key,
                request_fingerprint: fingerprint,
                operation: "mission.close".to_owned(),
                result_json: serde_json::to_string(&result)
                    .map_err(|err| MissionCommandError::internal(err.to_string()))?,
            }),
        )?;
        Ok(result)
    }
}

fn hash_lifecycle(event: &LifecycleEvent) -> Result<String, MissionCommandError> {
    let mut material = serde_json::to_value(event)
        .map_err(|err| MissionCommandError::internal(err.to_string()))?;
    material
        .as_object_mut()
        .ok_or_else(|| MissionCommandError::internal("lifecycle event was not an object"))?
        .remove("entry_hash");
    Ok(format!(
        "{:x}",
        Sha256::digest(material.to_string().as_bytes())
    ))
}

fn chain_receipts(
    history: &[LifecycleEvent],
    mission: &lanyte_mission::MissionRecord,
    caller: &crate::mission::VerifiedSession,
    source_kind: EventSourceKind,
    payloads: Vec<LifecyclePayload>,
) -> Result<Vec<NewMissionProjectionReceipt>, MissionCommandError> {
    let mut previous = history.last().map(|event| event.entry_hash.clone());
    let mut sequence = u64::try_from(history.len() + 1)
        .map_err(|_| MissionCommandError::internal("lifecycle sequence overflow"))?;
    let mut receipts = Vec::new();
    for payload in payloads {
        let event_id = Uuid::new_v4();
        let mut event = LifecycleEvent {
            event_schema: LIFECYCLE_EVENT_SCHEMA.to_owned(),
            event_id,
            mission_id: mission.mission_id,
            sequence,
            previous_entry_hash: previous.clone(),
            entry_hash: "0".repeat(64),
            occurred_at: mission.updated_at,
            recorded_at: mission.updated_at,
            event_type: payload.event_type().to_owned(),
            source: EventSource {
                kind: source_kind,
                subject: caller.subject.clone(),
                producer_version: env!("CARGO_PKG_VERSION").to_owned(),
                assurance: ObservationLevel::KernelObserved,
                evidence_ref: Some(caller.trust_ref.clone()),
            },
            payload,
        };
        event.entry_hash = hash_lifecycle(&event)?;
        previous = Some(event.entry_hash.clone());
        sequence += 1;
        receipts.push(NewMissionProjectionReceipt {
            event,
            envelope: AuditEnvelopeRef {
                action_id: Some(event_id.to_string()),
                correlation_id: Some(mission.mission_id.to_string()),
                trust_ref: Some(caller.trust_ref.clone()),
                ..AuditEnvelopeRef::default()
            },
            verification: None,
        });
    }
    Ok(receipts)
}

fn replayed_control_result(json: &str) -> Result<MissionControlResult, MissionCommandError> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|err| MissionCommandError::internal(err.to_string()))?;
    let request_id = value
        .get("request_id")
        .and_then(|value| value.as_str())
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or_else(|| MissionCommandError::internal("replayed result missing request_id"))?;
    match value.get("operation").and_then(|value| value.as_str()) {
        Some("mission.launch") => {
            let record = serde_json::from_value(
                value
                    .pointer("/body/record")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            )
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
            let key = value
                .get("idempotency_key")
                .and_then(|value| value.as_str())
                .unwrap_or("replayed-launch-key")
                .to_owned();
            let expected = value
                .get("expected_revision")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            MissionControlResult::launch(request_id, key, expected, record)
                .map_err(MissionCommandError::internal)
        }
        Some("mission.close") => {
            let mission_id = value
                .pointer("/body/mission_id")
                .and_then(|value| value.as_str())
                .and_then(|value| Uuid::parse_str(value).ok())
                .ok_or_else(|| {
                    MissionCommandError::internal("replayed close missing mission_id")
                })?;
            let attempt_id = value
                .pointer("/body/attempt_id")
                .and_then(|value| value.as_str())
                .and_then(|value| Uuid::parse_str(value).ok())
                .ok_or_else(|| {
                    MissionCommandError::internal("replayed close missing attempt_id")
                })?;
            let key = value
                .get("idempotency_key")
                .and_then(|value| value.as_str())
                .unwrap_or("replayed-close-key")
                .to_owned();
            let expected = value
                .get("expected_revision")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            MissionControlResult::close(request_id, key, expected, mission_id, attempt_id)
                .map_err(MissionCommandError::internal)
        }
        other => Err(MissionCommandError::internal(format!(
            "unsupported replayed operation: {other:?}"
        ))),
    }
}

fn mutation_fingerprint(
    operation: &str,
    caller: &crate::mission::VerifiedSession,
    body: &serde_json::Value,
) -> Result<String, MissionCommandError> {
    let encoded = serde_json::to_string(&serde_json::json!({
        "operation": operation,
        "caller": caller.subject,
        "body": body,
    }))
    .map_err(|err| MissionCommandError::internal(err.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(encoded.as_bytes())))
}
