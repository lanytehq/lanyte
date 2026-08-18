use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::sync::Mutex;

use chrono::Utc;
use lanyte_driver_codex::{CodexAppServerDriver, CodexLaunchSpec, CodexSession};
use lanyte_gateway::GatewayEvent;
use lanyte_mission::{
    AttemptRecord, AttemptState, AttemptTerminalReason, EventSource, EventSourceKind,
    HarnessSelection, LifecycleEvent, LifecyclePayload, MissionControlRequest,
    MissionControlResult, MissionPhase, MissionTerminalReason, MissionTransition, ObservationLevel,
    RecoveryRelation, LIFECYCLE_EVENT_SCHEMA,
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
        let workspace = PathBuf::from(&body.workspace);
        if !workspace.is_dir() {
            return Err(MissionCommandError::invalid_args(
                "workspace must be an existing directory",
            ));
        }
        let driver = CodexAppServerDriver::new(CodexLaunchSpec {
            workspace: workspace.clone(),
            binary_path: body.binary.map(PathBuf::from),
        });
        let attempt_id = Uuid::new_v4();
        let session = driver
            .create(attempt_id)
            .await
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        let now = Utc::now();
        let authorizer = caller_principal(&caller);
        let authorization_ref = format!("authorizations/launch/{attempt_id}");
        let mut mission = before.clone();
        mission.revision = expected_revision + 1;
        mission.updated_at = now;
        mission.phase = MissionPhase::Active;
        mission.authorizer = Some(authorizer.clone());
        mission.authorization_ref = Some(authorization_ref);
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

        let event_id = Uuid::new_v4();
        let mut lifecycle = LifecycleEvent {
            event_schema: LIFECYCLE_EVENT_SCHEMA.to_owned(),
            event_id,
            mission_id: mission.mission_id,
            sequence: 2,
            previous_entry_hash: Some(stored.audit_entry_hash.clone()),
            entry_hash: "0".repeat(64),
            occurred_at: now,
            recorded_at: now,
            event_type: "attempt_created".to_owned(),
            source: EventSource {
                kind: EventSourceKind::VerifiedAttestation,
                subject: caller.subject.clone(),
                producer_version: env!("CARGO_PKG_VERSION").to_owned(),
                assurance: ObservationLevel::KernelObserved,
                evidence_ref: Some(caller.trust_ref.clone()),
            },
            payload: LifecyclePayload::AttemptCreated {
                attempt_id,
                ordinal: 1,
                generation: 1,
                recovery_relation: RecoveryRelation::Initial,
                predecessor_attempt_id: None,
            },
        };
        lifecycle.entry_hash = hash_lifecycle(&lifecycle)?;
        if let Err(err) = service.persist_update(
            expected_revision,
            mission.clone(),
            NewMissionProjectionReceipt {
                event: lifecycle,
                envelope: AuditEnvelopeRef {
                    action_id: Some(event_id.to_string()),
                    correlation_id: Some(mission.mission_id.to_string()),
                    trust_ref: Some(caller.trust_ref.clone()),
                    ..AuditEnvelopeRef::default()
                },
                verification: None,
            },
        ) {
            let mut session = session;
            let _ = session.close().await;
            return Err(err);
        }

        live_sessions().lock().await.insert(attempt_id, session);
        MissionControlResult::launch(request_id, idempotency_key, expected_revision, mission)
            .map_err(MissionCommandError::internal)
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
        let event_id = Uuid::new_v4();
        let mut lifecycle = LifecycleEvent {
            event_schema: LIFECYCLE_EVENT_SCHEMA.to_owned(),
            event_id,
            mission_id: mission.mission_id,
            sequence: 3,
            previous_entry_hash: Some(stored.audit_entry_hash.clone()),
            entry_hash: "0".repeat(64),
            occurred_at: now,
            recorded_at: now,
            event_type: "mission_terminal".to_owned(),
            source: EventSource {
                kind: EventSourceKind::OperatorCommand,
                subject: caller.subject.clone(),
                producer_version: env!("CARGO_PKG_VERSION").to_owned(),
                assurance: ObservationLevel::KernelObserved,
                evidence_ref: Some(caller.trust_ref.clone()),
            },
            payload: LifecyclePayload::MissionTerminal {
                phase: MissionPhase::Cancelled,
                reason: MissionTerminalReason::OperatorCancelled,
                terminal_entry_hash: "0".repeat(64),
            },
        };
        let entry_hash = hash_lifecycle(&lifecycle)?;
        if let LifecyclePayload::MissionTerminal {
            terminal_entry_hash,
            ..
        } = &mut lifecycle.payload
        {
            *terminal_entry_hash = entry_hash.clone();
        }
        lifecycle.entry_hash = entry_hash.clone();
        mission.terminal_entry_hash = Some(entry_hash);

        MissionTransition {
            expected_revision,
            from: MissionPhase::Active,
            to: MissionPhase::Cancelled,
        }
        .check(&before, &mission)
        .map_err(|err| MissionCommandError::invalid_args(err.to_string()))?;

        service.persist_update(
            expected_revision,
            mission,
            NewMissionProjectionReceipt {
                event: lifecycle,
                envelope: AuditEnvelopeRef {
                    action_id: Some(event_id.to_string()),
                    correlation_id: Some(body.mission_id.to_string()),
                    trust_ref: Some(caller.trust_ref.clone()),
                    ..AuditEnvelopeRef::default()
                },
                verification: None,
            },
        )?;

        MissionControlResult::close(
            request_id,
            idempotency_key,
            expected_revision,
            body.mission_id,
            attempt_id,
        )
        .map_err(MissionCommandError::internal)
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
