use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::sync::Mutex;

use chrono::Utc;
use lanyte_driver_codex::{
    confine_workspace, CloseOutcome, CodexAppServerDriver, CodexLaunchSpec, CodexSession,
};
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

use tokio_util::sync::CancellationToken;

use super::{CommandInvokeError, CommandInvokeRequest, Orchestrator};
use crate::mission::{caller_principal, MissionCommandError, MissionService};

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
            &serde_json::json!({
                "expected_revision": expected_revision,
                "body": body,
            }),
        )?;
        let stored = service.visible_projection(&body.mission_id.to_string(), &caller)?;
        if let Some(replayed) = service.completed_mutation(&idempotency_key, &fingerprint)? {
            return replayed_control_result(&replayed);
        }
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
        let reservation = lanyte_state::MissionMutationIdempotency {
            key: idempotency_key.clone(),
            request_fingerprint: fingerprint.clone(),
            operation: "mission.launch".to_owned(),
            result_json: String::new(),
            owner_token: String::new(),
        };
        let owner_token =
            match service.reserve_mutation(&body.mission_id.to_string(), &reservation)? {
                lanyte_state::MutationReserve::Replay(replayed) => {
                    return replayed_control_result(&replayed);
                }
                lanyte_state::MutationReserve::Owned(token) => token,
            };
        let mut reservation_guard = ReservationGuard::new(service, &idempotency_key, &owner_token);
        let heartbeat = MutationHeartbeat::start(service, &idempotency_key, &owner_token);
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
        let report = driver.capabilities();
        report
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
        service.renew_mutation(&idempotency_key, &owner_token)?;
        let session = driver
            .create(attempt_id)
            .await
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        service.renew_mutation(&idempotency_key, &owner_token)?;
        if session.binary.digest != report.validity_condition.executable_sha256 {
            return Err(MissionCommandError::internal(
                "pinned executable digest does not match the gated capability report",
            ));
        }
        let now = Utc::now();
        let authorizer = caller_principal(&caller);
        let mut mission = before.clone();
        mission.revision = expected_revision + 1;
        mission.updated_at = now;
        mission.phase = MissionPhase::Active;
        mission.authorizer = Some(authorizer.clone());
        mission.authorization_ref = Some(format!("mission.launch/{request_id}"));
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
            lease_expires_at: None,
            deadman_at: None,
            last_observed_at: None,
            last_observation_source: None,
            lease_generation: None,
            process_tree_ref: None,
            ownership_established_at: None,
            harness_thread_id: None,
            harness_turn_id: None,
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
        let report_path = paths
            .pin_dir()
            .join(format!("{}.capability.json", session.binary.digest));
        let encoded = serde_json::to_string(&report)
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        std::fs::write(&report_path, encoded)
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
        let payloads = vec![
            (
                EventSourceKind::OperatorCommand,
                LifecyclePayload::AuthorizationBound {
                    authorizer: PrincipalRef {
                        kind: authorizer.kind,
                        subject: authorizer.subject.clone(),
                        attestation_ref: format!("mission.launch/{request_id}"),
                    },
                },
            ),
            (
                EventSourceKind::KernelObserved,
                LifecyclePayload::MissionPhaseChanged {
                    from: MissionPhase::Created,
                    to: MissionPhase::Active,
                    reason: Some("codex launch".to_owned()),
                },
            ),
            (
                EventSourceKind::KernelObserved,
                LifecyclePayload::AttemptCreated {
                    attempt_id,
                    ordinal: 1,
                    generation: 1,
                    recovery_relation: RecoveryRelation::Initial,
                    predecessor_attempt_id: None,
                },
            ),
            (
                EventSourceKind::DriverReported,
                LifecyclePayload::DriverCapabilityEvaluated {
                    attempt_id,
                    generation: 1,
                    driver_id: CODEX_DRIVER_ID.to_owned(),
                    capability: CapabilityName::Create,
                    availability: report.availability,
                    fidelity: lanyte_mission::CapabilityFidelity::Native,
                    report_id: report.report_id,
                },
            ),
            (
                EventSourceKind::DriverReported,
                LifecyclePayload::DriverCapabilityEvaluated {
                    attempt_id,
                    generation: 1,
                    driver_id: CODEX_DRIVER_ID.to_owned(),
                    capability: CapabilityName::Identify,
                    availability: report.availability,
                    fidelity: lanyte_mission::CapabilityFidelity::Native,
                    report_id: report.report_id,
                },
            ),
            (
                EventSourceKind::DriverReported,
                LifecyclePayload::DriverCapabilityEvaluated {
                    attempt_id,
                    generation: 1,
                    driver_id: CODEX_DRIVER_ID.to_owned(),
                    capability: CapabilityName::Observe,
                    availability: report.availability,
                    fidelity: lanyte_mission::CapabilityFidelity::Native,
                    report_id: report.report_id,
                },
            ),
            (
                EventSourceKind::DriverReported,
                LifecyclePayload::DriverCapabilityEvaluated {
                    attempt_id,
                    generation: 1,
                    driver_id: CODEX_DRIVER_ID.to_owned(),
                    capability: CapabilityName::Close,
                    availability: report.availability,
                    fidelity: lanyte_mission::CapabilityFidelity::Native,
                    report_id: report.report_id,
                },
            ),
            (
                EventSourceKind::KernelObserved,
                LifecyclePayload::AttemptStateChanged {
                    attempt_id,
                    generation: 1,
                    from: AttemptState::Starting,
                    to: AttemptState::Running,
                    reason: Some("thread/start".to_owned()),
                },
            ),
        ];
        let receipts = chain_receipts(&history, &mission, &caller, payloads, None)?;
        let result = MissionControlResult::launch(
            request_id,
            idempotency_key.clone(),
            expected_revision,
            mission.clone(),
        )
        .map_err(MissionCommandError::internal)?;
        service.renew_mutation(&idempotency_key, &owner_token)?;
        if let Err(err) = heartbeat.stop().await {
            let mut session = session;
            let _ = session.close().await;
            return Err(err);
        }
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
                owner_token,
            }),
        ) {
            let mut session = session;
            let _ = session.close().await;
            return Err(err);
        }

        live_sessions().lock().await.insert(attempt_id, session);
        reservation_guard.disarm();
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
        if session.overflowed() {
            return Err(MissionCommandError::invalid_args(
                "observation overflow; oldest events were dropped",
            ));
        }
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
            &serde_json::json!({
                "expected_revision": expected_revision,
                "mission_id": body.mission_id,
            }),
        )?;
        let stored = service.visible_projection(&body.mission_id.to_string(), &caller)?;
        if let Some(replayed) = service.completed_mutation(&idempotency_key, &fingerprint)? {
            return replayed_control_result(&replayed);
        }
        let before = stored.mission.clone();
        let already_cancelling = before.attempts.iter().any(|attempt| {
            attempt.state == AttemptState::Cancelling
                && before.current_attempt_id == Some(attempt.attempt_id)
        });
        if !already_cancelling && before.revision != expected_revision {
            return Err(MissionCommandError::invalid_args(format!(
                "stale mission revision: expected {expected_revision}, actual {}",
                before.revision
            )));
        }
        if before.phase != MissionPhase::Active {
            return Err(MissionCommandError::invalid_args(
                "mission.close requires an active mission",
            ));
        }
        let reservation = lanyte_state::MissionMutationIdempotency {
            key: idempotency_key.clone(),
            request_fingerprint: fingerprint.clone(),
            operation: "mission.close".to_owned(),
            result_json: String::new(),
            owner_token: String::new(),
        };
        let owner_token =
            match service.reserve_mutation(&body.mission_id.to_string(), &reservation)? {
                lanyte_state::MutationReserve::Replay(replayed) => {
                    return replayed_control_result(&replayed);
                }
                lanyte_state::MutationReserve::Owned(token) => token,
            };
        let mut reservation_guard = ReservationGuard::new(service, &idempotency_key, &owner_token);
        let attempt_id = before
            .current_attempt_id
            .ok_or_else(|| MissionCommandError::invalid_args("mission has no live attempt"))?;
        let prior_outcome = {
            let mut sessions = live_sessions().lock().await;
            match sessions.get_mut(&attempt_id) {
                Some(session) => {
                    if let Some(CloseOutcome::Terminated(status)) = session.retained_close_outcome()
                    {
                        Some(CloseOutcome::Terminated(status))
                    } else {
                        session
                            .poll_exit()
                            .map_err(|err| MissionCommandError::internal(err.to_string()))?
                            .map(CloseOutcome::AlreadyExited)
                    }
                }
                None => None,
            }
        };
        let skip_cancelling = already_cancelling || prior_outcome.is_some();
        let from_state = if matches!(prior_outcome, Some(CloseOutcome::AlreadyExited(_)))
            && !already_cancelling
        {
            AttemptState::Running
        } else {
            AttemptState::Cancelling
        };
        let (cancelling, terminal_expected) = if skip_cancelling {
            reservation_guard.disarm();
            (before.clone(), before.revision)
        } else {
            let now = Utc::now();
            let mut cancelling = before.clone();
            cancelling.revision = expected_revision + 1;
            cancelling.updated_at = now;
            if let Some(attempt) = cancelling
                .attempts
                .iter_mut()
                .find(|attempt| attempt.attempt_id == attempt_id)
            {
                attempt.state = AttemptState::Cancelling;
            }
            let cancelling_receipts = chain_receipts(
                &service.lifecycle_history(&body.mission_id.to_string())?,
                &cancelling,
                &caller,
                vec![(
                    EventSourceKind::OperatorCommand,
                    LifecyclePayload::AttemptStateChanged {
                        attempt_id,
                        generation: 1,
                        from: AttemptState::Running,
                        to: AttemptState::Cancelling,
                        reason: Some("operator close".to_owned()),
                    },
                )],
                None,
            )?;
            service.persist_update_events(
                expected_revision,
                cancelling.clone(),
                cancelling_receipts,
                None,
            )?;
            reservation_guard.disarm();
            (cancelling, expected_revision + 1)
        };
        let mut sessions = live_sessions().lock().await;
        let close_outcome = if let Some(outcome) = prior_outcome {
            Some(outcome)
        } else if let Some(session) = sessions.get_mut(&attempt_id) {
            Some(
                session
                    .close()
                    .await
                    .map_err(|err| MissionCommandError::internal(err.to_string()))?,
            )
        } else if already_cancelling {
            None
        } else {
            return Err(MissionCommandError::invalid_args(
                "codex session is not in this kernel; close will not claim cancellation",
            ));
        };

        let now = Utc::now();
        let disposition = CloseDisposition::from_outcome(from_state, close_outcome);
        let phase = disposition.phase;
        let terminal_reason = disposition.terminal_reason;
        let attempt_state = disposition.attempt_state;
        let attempt_reason = disposition.attempt_reason;
        let operator_killed = disposition.operator_killed;
        let reap_ref = disposition.reap_detail.clone();
        let mut mission = cancelling.clone();
        mission.revision = terminal_expected + 1;
        mission.updated_at = now;
        mission.phase = phase;
        mission.terminal_reason = Some(terminal_reason);
        mission.terminal_entry_hash = Some("0".repeat(64));
        mission.current_attempt_id = None;
        if let Some(attempt) = mission
            .attempts
            .iter_mut()
            .find(|attempt| attempt.attempt_id == attempt_id)
        {
            attempt.state = attempt_state;
            attempt.ended_at = Some(now);
            attempt.terminal_reason = Some(attempt_reason);
        }
        MissionTransition {
            expected_revision: terminal_expected,
            from: MissionPhase::Active,
            to: phase,
        }
        .check(&cancelling, &mission)
        .map_err(|err| MissionCommandError::invalid_args(err.to_string()))?;

        let history = service.lifecycle_history(&body.mission_id.to_string())?;
        let payloads = vec![
            (
                EventSourceKind::KernelObserved,
                LifecyclePayload::AttemptStateChanged {
                    attempt_id,
                    generation: 1,
                    from: from_state,
                    to: attempt_state,
                    reason: reap_ref.clone(),
                },
            ),
            (
                if operator_killed {
                    EventSourceKind::OperatorCommand
                } else {
                    EventSourceKind::KernelObserved
                },
                LifecyclePayload::MissionPhaseChanged {
                    from: MissionPhase::Active,
                    to: phase,
                    reason: reap_ref.clone(),
                },
            ),
            (
                if operator_killed {
                    EventSourceKind::OperatorCommand
                } else {
                    EventSourceKind::KernelObserved
                },
                LifecyclePayload::MissionTerminal {
                    phase,
                    reason: terminal_reason,
                    terminal_entry_hash: "0".repeat(64),
                },
            ),
        ];
        let mut receipts = chain_receipts(&history, &mission, &caller, payloads, None)?;
        if let Some(last) = receipts.last_mut() {
            let digest = last.event.entry_hash.clone();
            if let LifecyclePayload::MissionTerminal {
                terminal_entry_hash,
                ..
            } = &mut last.event.payload
            {
                *terminal_entry_hash = digest.clone();
            }
            mission.terminal_entry_hash = Some(digest);
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
        if let Err(err) = service.persist_update_events(
            terminal_expected,
            mission,
            receipts,
            Some(lanyte_state::MissionMutationIdempotency {
                key: idempotency_key.clone(),
                request_fingerprint: fingerprint,
                operation: "mission.close".to_owned(),
                result_json: serde_json::to_string(&result)
                    .map_err(|err| MissionCommandError::internal(err.to_string()))?,
                owner_token: owner_token.clone(),
            }),
        ) {
            let _ = service.release_mutation(&idempotency_key, &owner_token);
            return Err(err);
        }
        sessions.remove(&attempt_id);
        reservation_guard.disarm();
        Ok(result)
    }
}

fn hash_lifecycle(event: &LifecycleEvent) -> Result<String, MissionCommandError> {
    let mut material = serde_json::to_value(event)
        .map_err(|err| MissionCommandError::internal(err.to_string()))?;
    let object = material
        .as_object_mut()
        .ok_or_else(|| MissionCommandError::internal("lifecycle event was not an object"))?;
    object.remove("entry_hash");
    if let Some(payload) = object
        .get_mut("payload")
        .and_then(|value| value.as_object_mut())
    {
        payload.remove("terminal_entry_hash");
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(material.to_string().as_bytes())
    ))
}

fn chain_receipts(
    history: &[LifecycleEvent],
    mission: &lanyte_mission::MissionRecord,
    caller: &crate::mission::VerifiedSession,
    payloads: Vec<(EventSourceKind, LifecyclePayload)>,
    kernel_evidence: Option<&str>,
) -> Result<Vec<NewMissionProjectionReceipt>, MissionCommandError> {
    let mut previous = history.last().map(|event| event.entry_hash.clone());
    let mut sequence = u64::try_from(history.len() + 1)
        .map_err(|_| MissionCommandError::internal("lifecycle sequence overflow"))?;
    let mut receipts = Vec::new();
    for (source_kind, payload) in payloads {
        let event_id = Uuid::new_v4();
        let evidence_ref = if source_kind == EventSourceKind::KernelObserved {
            kernel_evidence
                .map(str::to_owned)
                .or_else(|| Some(format!("lifecycle/{event_id}")))
        } else {
            Some(caller.trust_ref.clone())
        };
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
                evidence_ref,
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

struct MutationHeartbeat {
    cancel: CancellationToken,
    failed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl MutationHeartbeat {
    fn start(service: &MissionService, key: &str, owner_token: &str) -> Self {
        let cancel = CancellationToken::new();
        let failed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let service = service.clone();
        let key = key.to_owned();
        let owner_token = owner_token.to_owned();
        let child_cancel = cancel.clone();
        let failed_flag = std::sync::Arc::clone(&failed);
        let join = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(15));
            interval.tick().await;
            loop {
                tokio::select! {
                    () = child_cancel.cancelled() => break,
                    _ = interval.tick() => {
                        if service.renew_mutation(&key, &owner_token).is_err() {
                            failed_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                            break;
                        }
                    }
                }
            }
        });
        Self {
            cancel,
            failed,
            join: Some(join),
        }
    }

    fn failed(&self) -> bool {
        self.failed.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn stop(mut self) -> Result<(), MissionCommandError> {
        self.cancel.cancel();
        if let Some(join) = self.join.take() {
            join.await
                .map_err(|_| MissionCommandError::internal("mutation heartbeat task failed"))?;
        }
        if self.failed() {
            return Err(MissionCommandError::internal(
                "mutation lease renewal failed",
            ));
        }
        Ok(())
    }
}

impl Drop for MutationHeartbeat {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

struct ReservationGuard<'a> {
    service: &'a crate::mission::MissionService,
    key: String,
    fingerprint: String,
    armed: bool,
}

impl<'a> ReservationGuard<'a> {
    fn new(service: &'a crate::mission::MissionService, key: &str, fingerprint: &str) -> Self {
        Self {
            service,
            key: key.to_owned(),
            fingerprint: fingerprint.to_owned(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ReservationGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.service.release_mutation(&self.key, &self.fingerprint);
        }
    }
}

fn mutation_fingerprint(
    operation: &str,
    caller: &crate::mission::VerifiedSession,
    body: &serde_json::Value,
) -> Result<String, MissionCommandError> {
    let encoded = serde_json::to_string(&serde_json::json!({
        "operation": operation,
        "caller": {
            "issuer": caller.issuer,
            "subject": caller.subject,
            "role": caller.role,
            "scope": caller.scope,
        },
        "body": body,
    }))
    .map_err(|err| MissionCommandError::internal(err.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(encoded.as_bytes())))
}

struct CloseDisposition {
    phase: MissionPhase,
    terminal_reason: MissionTerminalReason,
    attempt_state: AttemptState,
    attempt_reason: AttemptTerminalReason,
    reap_detail: Option<String>,
    operator_killed: bool,
}

impl CloseDisposition {
    fn from_outcome(from_state: AttemptState, outcome: Option<CloseOutcome>) -> Self {
        match outcome {
            Some(CloseOutcome::Terminated(status)) => Self {
                phase: MissionPhase::Cancelled,
                terminal_reason: MissionTerminalReason::OperatorCancelled,
                attempt_state: AttemptState::Cancelled,
                attempt_reason: AttemptTerminalReason::ProcessReaped,
                reap_detail: Some(reap_detail("reap:terminated", status)),
                operator_killed: true,
            },
            Some(CloseOutcome::AlreadyExited(status)) => {
                let (attempt_state, attempt_reason) = if status.success() {
                    if from_state == AttemptState::Cancelling {
                        (
                            AttemptState::Failed,
                            AttemptTerminalReason::HarnessCompleted,
                        )
                    } else {
                        (
                            AttemptState::Completed,
                            AttemptTerminalReason::HarnessCompleted,
                        )
                    }
                } else {
                    (AttemptState::Crashed, AttemptTerminalReason::HarnessCrashed)
                };
                Self {
                    phase: MissionPhase::Failed,
                    terminal_reason: MissionTerminalReason::InternalError,
                    attempt_state,
                    attempt_reason,
                    reap_detail: Some(reap_detail("reap:already-exited", status)),
                    operator_killed: false,
                }
            }
            None => Self {
                phase: MissionPhase::Failed,
                terminal_reason: MissionTerminalReason::InternalError,
                attempt_state: AttemptState::Lost,
                attempt_reason: AttemptTerminalReason::OutcomeUnknown,
                reap_detail: Some("reap:unknown-handle-lost".to_owned()),
                operator_killed: false,
            },
        }
    }
}

fn reap_detail(prefix: &str, status: std::process::ExitStatus) -> String {
    let mut detail = format!("{prefix}:code={:?}", status.code());
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        detail.push_str(&format!(":signal={:?}", status.signal()));
    }
    detail
}

#[cfg(test)]
mod close_disposition_tests {
    use super::*;
    use crate::mission::SessionVerifier;

    #[cfg(unix)]
    fn status_from_raw(raw: i32) -> std::process::ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(raw)
    }

    #[cfg(unix)]
    #[test]
    fn prior_success_from_running_completes_the_attempt() {
        let disposition = CloseDisposition::from_outcome(
            AttemptState::Running,
            Some(CloseOutcome::AlreadyExited(status_from_raw(0))),
        );
        assert_eq!(disposition.phase, MissionPhase::Failed);
        assert_eq!(
            disposition.terminal_reason,
            MissionTerminalReason::InternalError
        );
        assert_eq!(disposition.attempt_state, AttemptState::Completed);
        assert_eq!(
            disposition.attempt_reason,
            AttemptTerminalReason::HarnessCompleted
        );
        assert!(!disposition.operator_killed);
        assert!(from_state_legal(
            AttemptState::Running,
            disposition.attempt_state
        ));
    }

    #[cfg(unix)]
    #[test]
    fn prior_success_from_cancelling_cannot_complete() {
        let disposition = CloseDisposition::from_outcome(
            AttemptState::Cancelling,
            Some(CloseOutcome::AlreadyExited(status_from_raw(0))),
        );
        assert_eq!(disposition.attempt_state, AttemptState::Failed);
        assert_eq!(
            disposition.attempt_reason,
            AttemptTerminalReason::HarnessCompleted
        );
        assert_eq!(disposition.phase, MissionPhase::Failed);
        assert!(from_state_legal(
            AttemptState::Cancelling,
            disposition.attempt_state
        ));
    }

    #[cfg(unix)]
    #[test]
    fn prior_nonzero_exit_crashes_the_attempt() {
        let disposition = CloseDisposition::from_outcome(
            AttemptState::Running,
            Some(CloseOutcome::AlreadyExited(status_from_raw(1 << 8))),
        );
        assert_eq!(disposition.attempt_state, AttemptState::Crashed);
        assert_eq!(
            disposition.attempt_reason,
            AttemptTerminalReason::HarnessCrashed
        );
        assert_eq!(disposition.phase, MissionPhase::Failed);
        assert!(!disposition.operator_killed);
        assert!(from_state_legal(
            AttemptState::Running,
            disposition.attempt_state
        ));
    }

    #[cfg(unix)]
    #[test]
    fn terminated_close_is_operator_cancelled() {
        let disposition = CloseDisposition::from_outcome(
            AttemptState::Cancelling,
            Some(CloseOutcome::Terminated(status_from_raw(libc_sigterm()))),
        );
        assert_eq!(disposition.phase, MissionPhase::Cancelled);
        assert_eq!(
            disposition.terminal_reason,
            MissionTerminalReason::OperatorCancelled
        );
        assert_eq!(disposition.attempt_state, AttemptState::Cancelled);
        assert_eq!(
            disposition.attempt_reason,
            AttemptTerminalReason::ProcessReaped
        );
        assert!(disposition.operator_killed);
    }

    #[test]
    fn missing_handle_is_lost_and_unknown() {
        let disposition = CloseDisposition::from_outcome(AttemptState::Cancelling, None);
        assert_eq!(disposition.attempt_state, AttemptState::Lost);
        assert_eq!(
            disposition.attempt_reason,
            AttemptTerminalReason::OutcomeUnknown
        );
        assert_eq!(disposition.phase, MissionPhase::Failed);
        assert!(!disposition.operator_killed);
    }

    fn from_state_legal(from: AttemptState, to: AttemptState) -> bool {
        from.can_transition_to(to)
    }

    #[cfg(unix)]
    fn libc_sigterm() -> i32 {
        15
    }

    struct TestVerifier;

    impl crate::mission::SessionVerifier for TestVerifier {
        fn verify(&self, token: &str) -> Result<crate::mission::VerifiedSession, String> {
            if token != "valid-session-secret" {
                return Err("invalid attestation".to_owned());
            }
            Ok(crate::mission::VerifiedSession {
                issuer: "lanyte-attest".to_owned(),
                subject: "operator-subject".to_owned(),
                session_id: Uuid::parse_str("00000000-0000-4000-8000-000000000001").unwrap(),
                role: "operator".to_owned(),
                scope: "lanytehq".to_owned(),
                jti: Uuid::parse_str("00000000-0000-4000-9000-000000000001").unwrap(),
                context_sha256: "1".repeat(64),
                token_sha256: "2".repeat(64),
                verification_policy_sha256: "3".repeat(64),
                trust_ref: "lanyte-attest://lanyte-attest/sessions/test".to_owned(),
            })
        }
    }

    fn test_event() -> lanyte_gateway::GatewayEvent {
        lanyte_gateway::GatewayEvent {
            peer_id: "peer-1".to_owned(),
            channel: lanyte_common::channels::COMMAND,
            payload: Vec::new(),
            client_auth_token: Some(lanyte_gateway::ClientAuthToken::from_test_secret(
                "valid-session-secret",
            )),
        }
    }

    #[tokio::test]
    async fn close_retry_keeps_terminated_after_terminal_persist_failure() {
        use std::path::PathBuf;
        use std::sync::{Arc, Mutex};

        use lanyte_mission::{
            MissionCreateBody, MissionLaunchBody, MissionPhase, NormalizedHarnessEvent,
            RecoveryPolicy,
        };
        use lanyte_state::{StatePaths, StateStore};

        let root = std::env::temp_dir().join(format!("lanyte-close-retry-{}", Uuid::new_v4()));
        let paths = StatePaths::new(&root);
        let store = Arc::new(Mutex::new(StateStore::open(paths.clone()).expect("store")));
        let service = crate::MissionService::new(Arc::clone(&store), Arc::new(TestVerifier));
        let (_tx, rx) = tokio::sync::mpsc::channel(4);
        let orchestrator = crate::Orchestrator::new(
            rx,
            tokio_util::sync::CancellationToken::new(),
            lanyte_gateway::PeerResponder::empty_for_tests(),
            None,
        )
        .with_mission_service(service.clone());

        let created = match service
            .handle(
                MissionControlRequest::create(
                    Uuid::new_v4(),
                    "create:close-retry".to_owned(),
                    MissionCreateBody {
                        goal: "Prove close retry keeps operator termination".to_owned(),
                        policy_id: "policy.local".to_owned(),
                        deadline_at: None,
                        recovery_policy: RecoveryPolicy::AskOperator,
                    },
                )
                .expect("create request"),
                Some("valid-session-secret"),
            )
            .expect("create")
        {
            MissionControlResult::Record { record, .. } => *record,
            _ => panic!("expected created record"),
        };

        let workspace = paths.workspace_root().join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let binary = workspace.join("fake-codex");
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../lanyte-driver-codex/tests/fixtures/fake-codex-app-server.py");
        let staging = binary.with_extension("partial");
        std::fs::copy(fixture, &staging).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        if let Ok(file) = std::fs::File::open(&staging) {
            let _ = file.sync_all();
        }
        std::fs::rename(staging, &binary).unwrap();

        let launched = orchestrator
            .launch_codex(
                &test_event(),
                MissionControlRequest::launch(
                    Uuid::new_v4(),
                    "launch:close-retry".to_owned(),
                    created.revision,
                    MissionLaunchBody {
                        mission_id: created.mission_id,
                        workspace: workspace.display().to_string(),
                        binary: Some(binary.display().to_string()),
                    },
                )
                .expect("launch request"),
            )
            .await
            .expect("launch");
        let launched = match launched {
            MissionControlResult::Record { record, .. } => *record,
            _ => panic!("expected launched record"),
        };
        assert_eq!(launched.phase, MissionPhase::Active);

        let mut saw_started = false;
        let mut saw_tool = false;
        let mut saw_exited = false;
        for _ in 0..20 {
            let observed = orchestrator
                .observe_codex(
                    &test_event(),
                    MissionControlRequest::observe(Uuid::new_v4(), created.mission_id)
                        .expect("observe request"),
                )
                .await
                .expect("observe");
            if let MissionControlResult::Observe { events, .. } = observed {
                for event in events {
                    match event {
                        NormalizedHarnessEvent::Started { .. } => saw_started = true,
                        NormalizedHarnessEvent::ToolProposed { .. } => saw_tool = true,
                        NormalizedHarnessEvent::Exited { .. } => saw_exited = true,
                    }
                }
            }
            if saw_started && saw_tool && saw_exited {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(saw_started && saw_tool && saw_exited);

        service.fail_next_terminal_persist();
        let first = orchestrator
            .close_codex(
                &test_event(),
                MissionControlRequest::close(
                    Uuid::new_v4(),
                    "close:retry-terminated".to_owned(),
                    launched.revision,
                    created.mission_id,
                )
                .expect("close request"),
            )
            .await
            .expect_err("terminal persist must fail once");
        assert!(
            first.message.contains("injected terminal persist failure"),
            "unexpected first close error: {} ({:?})",
            first.message,
            first.code
        );

        let midway = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("midway mission");
        assert_eq!(midway.phase, MissionPhase::Active);
        assert_eq!(midway.attempts[0].state, AttemptState::Cancelling);

        let closed = orchestrator
            .close_codex(
                &test_event(),
                MissionControlRequest::close(
                    Uuid::new_v4(),
                    "close:retry-terminated".to_owned(),
                    launched.revision,
                    created.mission_id,
                )
                .expect("retry close"),
            )
            .await
            .expect("retry must keep Terminated");
        let MissionControlResult::Close {
            request_id: committed_request_id,
            ..
        } = &closed
        else {
            panic!("expected close result");
        };

        let caller = TestVerifier.verify("valid-session-secret").expect("caller");
        let finished = service
            .visible_mission(&created.mission_id.to_string(), &caller)
            .expect("finished mission");
        assert_eq!(finished.phase, MissionPhase::Cancelled);
        assert_eq!(
            finished.terminal_reason,
            Some(MissionTerminalReason::OperatorCancelled)
        );
        assert_eq!(finished.attempts[0].state, AttemptState::Cancelled);
        assert_eq!(
            finished.attempts[0].terminal_reason,
            Some(AttemptTerminalReason::ProcessReaped)
        );

        let history = service
            .lifecycle_history(&created.mission_id.to_string())
            .expect("history");
        lanyte_mission::validate_history(&finished, &history).expect("history must validate");
        assert_eq!(history[0].previous_entry_hash, None);
        for (index, event) in history.iter().enumerate() {
            assert_eq!(event.sequence, u64::try_from(index + 1).unwrap());
            if index > 0 {
                assert_eq!(
                    event.previous_entry_hash.as_deref(),
                    Some(history[index - 1].entry_hash.as_str())
                );
            }
            if event.source.kind == EventSourceKind::KernelObserved {
                assert_eq!(
                    event.source.evidence_ref.as_deref(),
                    Some(format!("lifecycle/{}", event.event_id).as_str())
                );
            }
        }
        let terminal = history
            .iter()
            .rev()
            .find(|event| {
                matches!(
                    event.payload,
                    lanyte_mission::LifecyclePayload::MissionTerminal { .. }
                )
            })
            .expect("terminal event");
        assert_eq!(
            finished.terminal_entry_hash.as_deref(),
            Some(terminal.entry_hash.as_str())
        );

        let audit = store
            .lock()
            .expect("store lock")
            .audit_records(&created.mission_id.to_string())
            .expect("audit");
        let stored = store
            .lock()
            .expect("store lock")
            .mission(&created.mission_id.to_string())
            .expect("projection")
            .expect("projection present");
        let mission_events: Vec<_> = audit
            .iter()
            .filter(|record| record.kind == lanyte_telemetry::AuditRecordKind::MissionEvent)
            .collect();
        assert_eq!(mission_events.len(), history.len());
        assert_eq!(
            stored.audit_entry_id,
            history.last().unwrap().event_id.to_string()
        );
        assert_eq!(
            stored.audit_entry_hash,
            mission_events.last().unwrap().entry_hash
        );
        for event in &history {
            assert!(
                mission_events
                    .iter()
                    .any(|record| record.entry_id == event.event_id.to_string()),
                "lifecycle {} must resolve to an audit entry",
                event.event_id
            );
        }

        let replayed = orchestrator
            .close_codex(
                &test_event(),
                MissionControlRequest::close(
                    Uuid::new_v4(),
                    "close:retry-terminated".to_owned(),
                    launched.revision,
                    created.mission_id,
                )
                .expect("replay close"),
            )
            .await
            .expect("completed close must replay");
        assert_eq!(replayed, closed);
        if let MissionControlResult::Close { request_id, .. } = replayed {
            assert_eq!(request_id, *committed_request_id);
        }

        let after_replay = service
            .visible_mission(&created.mission_id.to_string(), &caller)
            .expect("replayed mission");
        assert_eq!(after_replay, finished);
        let history_after = service
            .lifecycle_history(&created.mission_id.to_string())
            .expect("history after replay");
        assert_eq!(history_after, history);
        let audit_after = store
            .lock()
            .expect("store lock")
            .audit_records(&created.mission_id.to_string())
            .expect("audit after replay");
        assert_eq!(audit_after.len(), audit.len());
        let _ = std::fs::remove_dir_all(root);
    }
}
