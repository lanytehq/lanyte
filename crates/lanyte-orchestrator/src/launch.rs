use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::sync::Mutex;

use chrono::Utc;
use lanyte_driver_codex::{CodexAppServerDriver, CodexLaunchSpec, CodexSession};
use lanyte_gateway::GatewayEvent;
use lanyte_mission::{
    AttemptRecord, AttemptState, EventSource, EventSourceKind, HarnessDriver, HarnessSelection,
    LifecycleEvent, LifecyclePayload, MissionPhase, ObservationLevel, RecoveryRelation, Validate,
    LIFECYCLE_EVENT_SCHEMA,
};
use lanyte_state::NewMissionProjectionReceipt;
use lanyte_telemetry::AuditEnvelopeRef;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{CommandInvokeError, CommandInvokeRequest, Orchestrator};
use crate::mission::MissionCommandError;

fn live_sessions() -> &'static Mutex<HashMap<Uuid, CodexSession>> {
    static LIVE: OnceLock<Mutex<HashMap<Uuid, CodexSession>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Debug, Deserialize)]
struct LaunchEnvelope {
    operation: String,
    body: LaunchArgs,
}

#[derive(Debug, Deserialize)]
struct LaunchArgs {
    mission_id: String,
    workspace: String,
    #[serde(default)]
    binary: Option<String>,
}

impl Orchestrator {
    pub(super) async fn handle_mission_launch(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
    ) {
        match self.launch_codex(event, &command_request).await {
            Ok(result) => {
                self.send_command_result(
                    &event.peer_id,
                    super::CommandInvokeResult {
                        kind: "invoke_result",
                        request_id: command_request.request_id.clone(),
                        command: command_request.command.clone(),
                        result,
                    },
                )
                .await;
            }
            Err(err) => {
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
        }
    }

    async fn launch_codex(
        &self,
        event: &GatewayEvent,
        command_request: &CommandInvokeRequest,
    ) -> Result<Value, MissionCommandError> {
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
        let args = match serde_json::from_value::<LaunchEnvelope>(command_request.args.clone()) {
            Ok(envelope) if envelope.operation == "mission.launch" => envelope.body,
            _ => serde_json::from_value(command_request.args.clone())
                .map_err(|err| MissionCommandError::internal(err.to_string()))?,
        };
        let mut mission = service.visible_mission(&args.mission_id, &caller)?;
        if mission.phase != MissionPhase::Created {
            return Err(MissionCommandError::internal(
                "mission.launch requires a created mission with no live attempt",
            ));
        }
        let workspace = PathBuf::from(&args.workspace);
        if !workspace.is_dir() {
            return Err(MissionCommandError::internal(
                "workspace must be an existing directory",
            ));
        }
        let driver = CodexAppServerDriver::new(CodexLaunchSpec {
            workspace: workspace.clone(),
            binary_path: args.binary.map(PathBuf::from),
        });
        let attempt_id = Uuid::new_v4();
        let session = driver
            .create(attempt_id)
            .await
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        let now = Utc::now();
        let expected_revision = mission.revision;
        mission.revision += 1;
        mission.updated_at = now;
        mission.phase = MissionPhase::Active;
        mission.harness_selection = Some(HarnessSelection {
            harness_id: "codex".to_owned(),
            driver_id: driver.descriptor().driver_id,
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
            driver_id: Some(session.binary.path.display().to_string()),
            harness_session_id: Some(session.harness_session_id.clone()),
            started_at: Some(now),
            ended_at: None,
            terminal_reason: None,
            evidence_ref: Some(format!("codex:{}", session.binary.version)),
        });
        mission
            .validate()
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;

        let event_id = Uuid::new_v4();
        let mut lifecycle = LifecycleEvent {
            event_schema: LIFECYCLE_EVENT_SCHEMA.to_owned(),
            event_id,
            mission_id: mission.mission_id,
            sequence: 2,
            previous_entry_hash: None,
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
        let mut material = serde_json::to_value(&lifecycle)
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        material
            .as_object_mut()
            .ok_or_else(|| MissionCommandError::internal("lifecycle event was not an object"))?
            .remove("entry_hash");
        lifecycle.entry_hash = format!("{:x}", Sha256::digest(material.to_string().as_bytes()));
        service.persist_update(
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
        )?;

        live_sessions().lock().await.insert(attempt_id, session);

        Ok(serde_json::json!({
            "mission_id": mission.mission_id,
            "attempt_id": attempt_id,
            "phase": "active",
            "harness_session_id": mission.attempts[0].harness_session_id,
            "binary": mission.attempts[0].driver_id,
        }))
    }

    pub(super) async fn handle_mission_observe(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
    ) {
        match self.observe_codex(event, &command_request).await {
            Ok(result) => {
                self.send_command_result(
                    &event.peer_id,
                    super::CommandInvokeResult {
                        kind: "invoke_result",
                        request_id: command_request.request_id.clone(),
                        command: command_request.command.clone(),
                        result,
                    },
                )
                .await;
            }
            Err(err) => {
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
        }
    }

    pub(super) async fn handle_mission_close(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
    ) {
        match self.close_codex(event, &command_request).await {
            Ok(result) => {
                self.send_command_result(
                    &event.peer_id,
                    super::CommandInvokeResult {
                        kind: "invoke_result",
                        request_id: command_request.request_id.clone(),
                        command: command_request.command.clone(),
                        result,
                    },
                )
                .await;
            }
            Err(err) => {
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
        }
    }

    async fn observe_codex(
        &self,
        event: &GatewayEvent,
        command_request: &CommandInvokeRequest,
    ) -> Result<Value, MissionCommandError> {
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
        let mission_id = command_request
            .args
            .get("body")
            .and_then(|body| body.get("mission_id"))
            .or_else(|| command_request.args.get("mission_id"))
            .and_then(Value::as_str)
            .ok_or_else(|| MissionCommandError::internal("mission_id required"))?;
        let mission = service.visible_mission(mission_id, &caller)?;
        let attempt_id = mission
            .current_attempt_id
            .ok_or_else(|| MissionCommandError::internal("mission has no live attempt"))?;
        let mut sessions = live_sessions().lock().await;
        let session = sessions
            .get_mut(&attempt_id)
            .ok_or_else(|| MissionCommandError::internal("codex session is not in this kernel"))?;
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
        Ok(serde_json::json!({ "mission_id": mission_id, "attempt_id": attempt_id, "events": events }))
    }

    async fn close_codex(
        &self,
        event: &GatewayEvent,
        command_request: &CommandInvokeRequest,
    ) -> Result<Value, MissionCommandError> {
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
        let mission_id = command_request
            .args
            .get("body")
            .and_then(|body| body.get("mission_id"))
            .or_else(|| command_request.args.get("mission_id"))
            .and_then(Value::as_str)
            .ok_or_else(|| MissionCommandError::internal("mission_id required"))?;
        let mission = service.visible_mission(mission_id, &caller)?;
        let attempt_id = mission
            .current_attempt_id
            .ok_or_else(|| MissionCommandError::internal("mission has no live attempt"))?;
        let mut sessions = live_sessions().lock().await;
        let mut session = sessions
            .remove(&attempt_id)
            .ok_or_else(|| MissionCommandError::internal("codex session is not in this kernel"))?;
        session
            .close()
            .await
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        Ok(serde_json::json!({
            "mission_id": mission_id,
            "attempt_id": attempt_id,
            "closed": true
        }))
    }
}
