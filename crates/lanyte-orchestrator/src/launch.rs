use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::sync::Mutex;

use chrono::{Duration, Utc};
use lanyte_driver_codex::{
    confine_workspace, probe_process_tree, terminate_process_tree, CloseOutcome,
    CodexAppServerDriver, CodexLaunchSpec, CodexSession, InterruptAttempt, ProcessTreeKill,
};
use lanyte_gateway::GatewayEvent;
use lanyte_mission::{
    AttemptRecord, AttemptState, AttemptStateCause, AttemptTerminalReason, CancelProgress,
    CapabilityName, EventSource, EventSourceKind, FallbackCancelOutcome, FallbackCancelProgress,
    HarnessDriver, HarnessSelection, LeasePolicy, LeaseTickKind, LifecycleEvent, LifecyclePayload,
    MissionControlRequest, MissionControlResult, MissionPhase, MissionTerminalReason,
    MissionTransition, ObservationLevel, ObservationSource, PrincipalRef, ProtocolCancelOutcome,
    ProtocolCancelProgress, RecoveryRelation, LIFECYCLE_EVENT_SCHEMA,
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

    pub(super) async fn handle_mission_cancel(
        &self,
        event: &GatewayEvent,
        command_request: CommandInvokeRequest,
    ) {
        match self.parse_control(&command_request) {
            Ok(request) => {
                self.reply_control(
                    event,
                    command_request,
                    self.cancel_mission(event, request).await,
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
                    CapabilityName::Cancel,
                    CapabilityName::LocalProcessTermination,
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
        if !mission.lease_policy.enabled {
            mission.lease_policy = LeasePolicy {
                enabled: true,
                lease_seconds: Some(600),
                deadman_seconds: Some(300),
            };
        }
        let lease_seconds = mission.lease_policy.lease_seconds.unwrap_or(600);
        let deadman_seconds = mission.lease_policy.deadman_seconds.unwrap_or(300);
        let lease_expires_at = now + Duration::seconds(lease_seconds as i64);
        let deadman_at = now + Duration::seconds(deadman_seconds as i64);
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
            lease_expires_at: Some(lease_expires_at),
            deadman_at: Some(deadman_at),
            last_observed_at: Some(now),
            last_observation_source: Some(ObservationSource::DriverEvent),
            lease_generation: Some(1),
            process_tree_ref: session.process_tree_ref.clone(),
            ownership_established_at: Some(now),
            harness_thread_id: Some(session.harness_session_id.clone()),
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
                EventSourceKind::KernelObserved,
                LifecyclePayload::LeaseStarted {
                    attempt_id,
                    generation: 1,
                    lease_generation: 1,
                    lease_expires_at,
                    deadman_at,
                    observed_at: now,
                    observation_source: ObservationSource::DriverEvent,
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
                EventSourceKind::DriverReported,
                LifecyclePayload::DriverCapabilityEvaluated {
                    attempt_id,
                    generation: 1,
                    driver_id: CODEX_DRIVER_ID.to_owned(),
                    capability: CapabilityName::Cancel,
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
                    capability: CapabilityName::LocalProcessTermination,
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
                    cause: Some(lanyte_mission::AttemptStateCause::HarnessCompleted),
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

    async fn cancel_mission(
        &self,
        event: &GatewayEvent,
        request: MissionControlRequest,
    ) -> Result<MissionControlResult, MissionCommandError> {
        let MissionControlRequest::Cancel {
            request_id,
            idempotency_key,
            expected_revision,
            body,
        } = request
        else {
            return Err(MissionCommandError::internal(
                "cancel handler received a non-cancel control request",
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
            "mission.cancel",
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
        if before.phase.is_terminal() {
            return Err(MissionCommandError::invalid_args(
                "mission is already terminal",
            ));
        }
        let already_cancelling = before.attempts.iter().any(|attempt| {
            attempt.state == AttemptState::Cancelling
                && before.current_attempt_id == Some(attempt.attempt_id)
        });
        let pending_cancel =
            service.incomplete_mutation(&body.mission_id.to_string(), "mission.cancel")?;
        if let Some(pending) = pending_cancel.as_ref() {
            if pending.key != idempotency_key {
                return Err(MissionCommandError::invalid_args(
                    "a cancel mutation is already in flight for this mission",
                ));
            }
            if parse_pending_cancel(&pending.result_json).is_none() {
                return Err(MissionCommandError::internal(
                    "stored cancel mutation is not a typed pending stub",
                ));
            }
        }
        let resume_stored = pending_cancel
            .as_ref()
            .is_some_and(|pending| pending.key == idempotency_key);
        if !(already_cancelling && resume_stored) && before.revision != expected_revision {
            return Err(MissionCommandError::invalid_args(format!(
                "stale mission revision: expected {expected_revision}, actual {}",
                before.revision
            )));
        }
        let reservation = lanyte_state::MissionMutationIdempotency {
            key: idempotency_key.clone(),
            request_fingerprint: fingerprint.clone(),
            operation: "mission.cancel".to_owned(),
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

        if before.attempts.is_empty() && before.phase == MissionPhase::Created {
            let now = Utc::now();
            let mut mission = before.clone();
            mission.revision = expected_revision + 1;
            mission.updated_at = now;
            mission.phase = MissionPhase::Cancelled;
            mission.terminal_reason = Some(MissionTerminalReason::OperatorCancelled);
            mission.terminal_entry_hash = Some("0".repeat(64));
            let payloads = vec![
                (
                    EventSourceKind::OperatorCommand,
                    LifecyclePayload::CancelRequested {
                        attempt_id: None,
                        generation: None,
                        lease_generation: None,
                    },
                ),
                (
                    EventSourceKind::OperatorCommand,
                    LifecyclePayload::MissionPhaseChanged {
                        from: MissionPhase::Created,
                        to: MissionPhase::Cancelled,
                        reason: Some("operator cancel".to_owned()),
                    },
                ),
                (
                    EventSourceKind::KernelObserved,
                    LifecyclePayload::MissionTerminal {
                        phase: MissionPhase::Cancelled,
                        reason: MissionTerminalReason::OperatorCancelled,
                        terminal_entry_hash: "0".repeat(64),
                    },
                ),
            ];
            let mut receipts = chain_receipts(
                &service.lifecycle_history(&body.mission_id.to_string())?,
                &mission,
                &caller,
                payloads,
                None,
            )?;
            if let Some(hash) = receipts
                .last()
                .map(|receipt| receipt.event.entry_hash.clone())
            {
                mission.terminal_entry_hash = Some(hash.clone());
                if let Some(LifecyclePayload::MissionTerminal {
                    terminal_entry_hash,
                    ..
                }) = receipts
                    .last_mut()
                    .map(|receipt| &mut receipt.event.payload)
                {
                    *terminal_entry_hash = hash;
                }
            }
            let progress = CancelProgress {
                requested: true,
                protocol: None,
                fallback: None,
            };
            let result = MissionControlResult::Cancel {
                request_id,
                idempotency_key: idempotency_key.clone(),
                expected_revision,
                record: Box::new(mission.clone()),
                progress,
            };
            service.persist_update_events(
                expected_revision,
                mission,
                receipts,
                Some(lanyte_state::MissionMutationIdempotency {
                    key: idempotency_key,
                    request_fingerprint: fingerprint,
                    operation: "mission.cancel".to_owned(),
                    result_json: serde_json::to_string(&result)
                        .map_err(|err| MissionCommandError::internal(err.to_string()))?,
                    owner_token,
                }),
            )?;
            reservation_guard.disarm();
            return Ok(result);
        }

        let attempt_id = before
            .current_attempt_id
            .ok_or_else(|| MissionCommandError::invalid_args("mission has no live attempt"))?;
        let attempt = before
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == attempt_id)
            .cloned()
            .ok_or_else(|| MissionCommandError::invalid_args("current attempt is missing"))?;
        let lease_generation = attempt.lease_generation.unwrap_or(1);
        let history = service.lifecycle_history(&body.mission_id.to_string())?;
        let already_requested = history.iter().any(|event| {
            matches!(
                &event.payload,
                LifecyclePayload::CancelRequested {
                    attempt_id: Some(id),
                    generation: Some(generation),
                    ..
                } if *id == attempt_id && *generation == attempt.generation
            )
        });
        let already_dispatched = history.iter().any(|event| {
            matches!(
                &event.payload,
                LifecyclePayload::ProcessTerminationAttempted {
                    attempt_id: id,
                    outcome: FallbackCancelOutcome::KillDispatched,
                    ..
                } if *id == attempt_id
            )
        });
        let now = Utc::now();
        let mut cancelling = before.clone();
        if !already_cancelling && !already_requested {
            cancelling.revision = expected_revision + 1;
            cancelling.updated_at = now;
            if let Some(live) = cancelling
                .attempts
                .iter_mut()
                .find(|item| item.attempt_id == attempt_id)
            {
                live.state = AttemptState::Cancelling;
            }
            let receipts = chain_receipts(
                &history,
                &cancelling,
                &caller,
                vec![
                    (
                        EventSourceKind::OperatorCommand,
                        LifecyclePayload::CancelRequested {
                            attempt_id: Some(attempt_id),
                            generation: Some(attempt.generation),
                            lease_generation: Some(lease_generation),
                        },
                    ),
                    (
                        EventSourceKind::OperatorCommand,
                        LifecyclePayload::AttemptStateChanged {
                            attempt_id,
                            generation: attempt.generation,
                            from: attempt.state,
                            to: AttemptState::Cancelling,
                            reason: Some("operator cancel".to_owned()),
                            cause: Some(AttemptStateCause::OperatorCancel),
                        },
                    ),
                ],
                None,
            )?;
            service.persist_update_events(
                expected_revision,
                cancelling.clone(),
                receipts,
                Some(pending_cancel_idempotency(
                    &idempotency_key,
                    &fingerprint,
                    request_id,
                    expected_revision,
                    &owner_token,
                )),
            )?;
            reservation_guard.disarm();
            service.crash_after_cancelling_persist()?;
        }

        let protocol = {
            let mut sessions = live_sessions().lock().await;
            match sessions.get_mut(&attempt_id) {
                Some(session) => Some(
                    session
                        .interrupt_turn(
                            attempt.harness_thread_id.as_deref(),
                            attempt.harness_turn_id.as_deref(),
                        )
                        .await,
                ),
                None => None,
            }
        };
        let mut protocol_progress = match protocol {
            None => ProtocolCancelProgress {
                outcome: ProtocolCancelOutcome::Unavailable,
                thread_id: attempt.harness_thread_id.clone(),
                turn_id: attempt.harness_turn_id.clone(),
            },
            Some(InterruptAttempt::Unavailable) => ProtocolCancelProgress {
                outcome: ProtocolCancelOutcome::Unavailable,
                thread_id: attempt.harness_thread_id.clone(),
                turn_id: attempt.harness_turn_id.clone(),
            },
            Some(InterruptAttempt::RequestAccepted { thread_id, turn_id }) => {
                ProtocolCancelProgress {
                    outcome: ProtocolCancelOutcome::RequestAccepted,
                    thread_id: Some(thread_id),
                    turn_id: Some(turn_id),
                }
            }
            Some(InterruptAttempt::Timeout { thread_id, turn_id }) => ProtocolCancelProgress {
                outcome: ProtocolCancelOutcome::Timeout,
                thread_id: Some(thread_id),
                turn_id: Some(turn_id),
            },
            Some(InterruptAttempt::Interrupted { thread_id, turn_id }) => ProtocolCancelProgress {
                outcome: ProtocolCancelOutcome::Interrupted,
                thread_id: Some(thread_id),
                turn_id: Some(turn_id),
            },
            Some(InterruptAttempt::UnrelatedCompletion { thread_id, turn_id }) => {
                ProtocolCancelProgress {
                    outcome: ProtocolCancelOutcome::UnrelatedCompletion,
                    thread_id: Some(thread_id),
                    turn_id: Some(turn_id),
                }
            }
            Some(InterruptAttempt::Failed { detail }) => ProtocolCancelProgress {
                outcome: ProtocolCancelOutcome::Failed,
                thread_id: attempt.harness_thread_id.clone(),
                turn_id: Some(detail),
            },
        };
        let force_nonterminal = service.take_force_nonterminal_cancel();
        let mut interrupted =
            protocol_progress.outcome == ProtocolCancelOutcome::Interrupted && !force_nonterminal;
        let tree_ref = attempt.process_tree_ref.clone();
        let fallback = if interrupted {
            None
        } else if already_dispatched {
            tree_ref.as_deref().map(probe_process_tree)
        } else {
            let mut sessions = live_sessions().lock().await;
            sessions
                .get_mut(&attempt_id)
                .map(CodexSession::kill_process_tree)
                .or_else(|| tree_ref.as_deref().map(terminate_process_tree))
        };
        let mut fallback_progress = fallback.map(|outcome| FallbackCancelProgress {
            outcome: match outcome {
                ProcessTreeKill::Cleared => FallbackCancelOutcome::Cleared,
                ProcessTreeKill::KillDispatched => FallbackCancelOutcome::KillDispatched,
                ProcessTreeKill::Survivors => FallbackCancelOutcome::Survivors,
                ProcessTreeKill::Unknown => FallbackCancelOutcome::Unknown,
            },
        });
        let mut cleared = fallback_progress
            .as_ref()
            .is_some_and(|progress| progress.outcome == FallbackCancelOutcome::Cleared);
        if force_nonterminal {
            protocol_progress.outcome = ProtocolCancelOutcome::RequestAccepted;
            fallback_progress = Some(FallbackCancelProgress {
                outcome: FallbackCancelOutcome::KillDispatched,
            });
            interrupted = false;
            cleared = false;
        }
        let mut evidence = vec![(
            EventSourceKind::DriverReported,
            LifecyclePayload::ProtocolCancelAttempted {
                attempt_id,
                generation: attempt.generation,
                lease_generation,
                thread_id: protocol_progress.thread_id.clone(),
                turn_id: protocol_progress.turn_id.clone(),
                outcome: protocol_progress.outcome.clone(),
            },
        )];
        if let Some(fallback_progress) = &fallback_progress {
            evidence.push((
                EventSourceKind::KernelObserved,
                LifecyclePayload::ProcessTerminationAttempted {
                    attempt_id,
                    generation: attempt.generation,
                    lease_generation,
                    outcome: fallback_progress.outcome.clone(),
                },
            ));
        }

        let now = Utc::now();
        let mut mission = cancelling.clone();
        mission.revision += 1;
        mission.updated_at = now;
        if let Some(live) = mission
            .attempts
            .iter_mut()
            .find(|item| item.attempt_id == attempt_id)
        {
            if let Some(thread_id) = &protocol_progress.thread_id {
                live.harness_thread_id = Some(thread_id.clone());
            }
            if protocol_progress.outcome == ProtocolCancelOutcome::Interrupted {
                if let Some(turn_id) = &protocol_progress.turn_id {
                    live.harness_turn_id = Some(turn_id.clone());
                }
            } else if live.harness_turn_id.is_none() {
                if let Some(turn_id) = &protocol_progress.turn_id {
                    if protocol_progress.outcome != ProtocolCancelOutcome::Failed {
                        live.harness_turn_id = Some(turn_id.clone());
                    }
                }
            }
        }
        if interrupted || cleared {
            mission.phase = MissionPhase::Cancelled;
            mission.terminal_reason = Some(MissionTerminalReason::OperatorCancelled);
            mission.current_attempt_id = None;
            if let Some(live) = mission
                .attempts
                .iter_mut()
                .find(|item| item.attempt_id == attempt_id)
            {
                live.state = AttemptState::Cancelled;
                live.ended_at = Some(now);
                live.terminal_reason = Some(if interrupted {
                    AttemptTerminalReason::ProtocolCancelled
                } else {
                    AttemptTerminalReason::ProcessReaped
                });
            }
            evidence.push((
                EventSourceKind::KernelObserved,
                LifecyclePayload::AttemptStateChanged {
                    attempt_id,
                    generation: attempt.generation,
                    from: AttemptState::Cancelling,
                    to: AttemptState::Cancelled,
                    reason: Some("cancel confirmed".to_owned()),
                    cause: Some(if interrupted {
                        AttemptStateCause::ProtocolInterrupt
                    } else {
                        AttemptStateCause::ProcessExit
                    }),
                },
            ));
            evidence.push((
                EventSourceKind::KernelObserved,
                LifecyclePayload::MissionPhaseChanged {
                    from: cancelling.phase,
                    to: MissionPhase::Cancelled,
                    reason: Some("cancel confirmed".to_owned()),
                },
            ));
            evidence.push((
                EventSourceKind::KernelObserved,
                LifecyclePayload::MissionTerminal {
                    phase: MissionPhase::Cancelled,
                    reason: MissionTerminalReason::OperatorCancelled,
                    terminal_entry_hash: "0".repeat(64),
                },
            ));
            if interrupted || cleared {
                live_sessions().lock().await.remove(&attempt_id);
            }
        }
        let mut receipts = chain_receipts(
            &service.lifecycle_history(&body.mission_id.to_string())?,
            &mission,
            &caller,
            evidence,
            None,
        )?;
        if mission.phase == MissionPhase::Cancelled {
            if let Some(hash) = receipts
                .last()
                .map(|receipt| receipt.event.entry_hash.clone())
            {
                mission.terminal_entry_hash = Some(hash.clone());
                if let Some(LifecyclePayload::MissionTerminal {
                    terminal_entry_hash,
                    ..
                }) = receipts
                    .last_mut()
                    .map(|receipt| &mut receipt.event.payload)
                {
                    *terminal_entry_hash = hash;
                }
            }
        }
        let progress = CancelProgress {
            requested: true,
            protocol: Some(protocol_progress),
            fallback: fallback_progress,
        };
        let result = MissionControlResult::Cancel {
            request_id,
            idempotency_key: idempotency_key.clone(),
            expected_revision,
            record: Box::new(mission.clone()),
            progress,
        };
        let complete = mission.phase == MissionPhase::Cancelled;
        service.persist_update_events(
            cancelling.revision,
            mission,
            receipts,
            Some(if complete {
                lanyte_state::MissionMutationIdempotency {
                    key: idempotency_key,
                    request_fingerprint: fingerprint,
                    operation: "mission.cancel".to_owned(),
                    result_json: serde_json::to_string(&result)
                        .map_err(|err| MissionCommandError::internal(err.to_string()))?,
                    owner_token,
                }
            } else {
                pending_cancel_idempotency(
                    &idempotency_key,
                    &fingerprint,
                    request_id,
                    expected_revision,
                    &owner_token,
                )
            }),
        )?;
        reservation_guard.disarm();
        Ok(result)
    }

    pub(super) async fn reconcile_restarts(&self) {
        self.apply_supervisor_clock(Utc::now(), true).await;
    }

    pub(super) async fn tick_leases(&self) {
        self.apply_supervisor_clock(Utc::now(), false).await;
    }

    #[cfg(test)]
    pub(super) async fn tick_leases_at(&self, now: chrono::DateTime<Utc>) {
        self.apply_supervisor_clock(now, false).await;
    }

    async fn apply_supervisor_clock(&self, now: chrono::DateTime<Utc>, restart: bool) {
        let Some(service) = &self.mission_service else {
            return;
        };
        let missions = match service.supervised_missions() {
            Ok(missions) => missions,
            Err(err) => {
                tracing::warn!(error = %err.message, "supervisor scan failed");
                return;
            }
        };
        for mission in missions {
            if let Err(err) = self.fold_supervisor_clock(service, mission, now, restart) {
                tracing::warn!(error = %err.message, "supervisor clock fold failed");
            }
        }
    }

    fn fold_supervisor_clock(
        &self,
        service: &MissionService,
        before: lanyte_mission::MissionRecord,
        now: chrono::DateTime<Utc>,
        restart: bool,
    ) -> Result<(), MissionCommandError> {
        if !before.lease_policy.enabled || before.phase.is_terminal() {
            return Ok(());
        }
        let Some(attempt_id) = before.current_attempt_id else {
            return Ok(());
        };
        let Some(attempt) = before
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == attempt_id)
            .cloned()
        else {
            return Ok(());
        };
        if !attempt.state.is_live() {
            return Ok(());
        }
        if attempt.state == AttemptState::Cancelling {
            if let Some(tree_ref) = &attempt.process_tree_ref {
                let history = service.lifecycle_history(&before.mission_id.to_string())?;
                let already_dispatched = history.iter().any(|event| {
                    matches!(
                        &event.payload,
                        LifecyclePayload::ProcessTerminationAttempted {
                            attempt_id: id,
                            outcome: FallbackCancelOutcome::KillDispatched,
                            ..
                        } if *id == attempt_id
                    )
                });
                let outcome = match live_sessions().try_lock() {
                    Ok(mut sessions) => {
                        if let Some(session) = sessions.get_mut(&attempt_id) {
                            let _ = session.poll_exit();
                            let outcome = if already_dispatched {
                                session.probe_process_tree()
                            } else {
                                session.kill_process_tree()
                            };
                            let _ = session.poll_exit();
                            outcome
                        } else if already_dispatched {
                            probe_process_tree(tree_ref)
                        } else {
                            terminate_process_tree(tree_ref)
                        }
                    }
                    Err(_) if already_dispatched => probe_process_tree(tree_ref),
                    Err(_) => terminate_process_tree(tree_ref),
                };
                if outcome == ProcessTreeKill::Cleared {
                    let mut mission = before.clone();
                    mission.revision = before.revision + 1;
                    mission.updated_at = now;
                    mission.phase = MissionPhase::Cancelled;
                    mission.terminal_reason = Some(MissionTerminalReason::OperatorCancelled);
                    mission.current_attempt_id = None;
                    if let Some(live) = mission
                        .attempts
                        .iter_mut()
                        .find(|item| item.attempt_id == attempt_id)
                    {
                        live.state = AttemptState::Cancelled;
                        live.ended_at = Some(now);
                        live.terminal_reason = Some(AttemptTerminalReason::ProcessReaped);
                    }
                    let mut receipts = chain_receipts_from(
                        &service.lifecycle_history(&before.mission_id.to_string())?,
                        &mission,
                        &mission.supervisor.subject,
                        Some("lanyte://kernel/supervisor"),
                        vec![
                            (
                                EventSourceKind::KernelObserved,
                                LifecyclePayload::ProcessTerminationAttempted {
                                    attempt_id,
                                    generation: attempt.generation,
                                    lease_generation: attempt.lease_generation.unwrap_or(1),
                                    outcome: FallbackCancelOutcome::Cleared,
                                },
                            ),
                            (
                                EventSourceKind::KernelObserved,
                                LifecyclePayload::AttemptStateChanged {
                                    attempt_id,
                                    generation: attempt.generation,
                                    from: AttemptState::Cancelling,
                                    to: AttemptState::Cancelled,
                                    reason: Some("membership cleared".to_owned()),
                                    cause: Some(AttemptStateCause::ProcessExit),
                                },
                            ),
                            (
                                EventSourceKind::KernelObserved,
                                LifecyclePayload::MissionPhaseChanged {
                                    from: before.phase,
                                    to: MissionPhase::Cancelled,
                                    reason: Some("membership cleared".to_owned()),
                                },
                            ),
                            (
                                EventSourceKind::KernelObserved,
                                LifecyclePayload::MissionTerminal {
                                    phase: MissionPhase::Cancelled,
                                    reason: MissionTerminalReason::OperatorCancelled,
                                    terminal_entry_hash: "0".repeat(64),
                                },
                            ),
                        ],
                        Some("sysprims/cleared"),
                    )?;
                    if let Some(hash) = receipts
                        .last()
                        .map(|receipt| receipt.event.entry_hash.clone())
                    {
                        mission.terminal_entry_hash = Some(hash.clone());
                        if let Some(LifecyclePayload::MissionTerminal {
                            terminal_entry_hash,
                            ..
                        }) = receipts
                            .last_mut()
                            .map(|receipt| &mut receipt.event.payload)
                        {
                            *terminal_entry_hash = hash;
                        }
                    }
                    let pending = service
                        .incomplete_mutation(&before.mission_id.to_string(), "mission.cancel")?;
                    let idempotency = match pending {
                        Some(pending) => {
                            let stub =
                                parse_pending_cancel(&pending.result_json).ok_or_else(|| {
                                    MissionCommandError::internal(
                                        "stored cancel mutation is not a typed pending stub",
                                    )
                                })?;
                            let result = MissionControlResult::Cancel {
                                request_id: stub.request_id,
                                idempotency_key: pending.key.clone(),
                                expected_revision: stub.expected_revision,
                                record: Box::new(mission.clone()),
                                progress: CancelProgress {
                                    requested: true,
                                    protocol: None,
                                    fallback: Some(FallbackCancelProgress {
                                        outcome: FallbackCancelOutcome::Cleared,
                                    }),
                                },
                            };
                            Some(lanyte_state::MissionMutationIdempotency {
                                key: pending.key,
                                request_fingerprint: pending.request_fingerprint,
                                operation: pending.operation,
                                result_json: serde_json::to_string(&result).map_err(|err| {
                                    MissionCommandError::internal(err.to_string())
                                })?,
                                owner_token: pending.owner_token,
                            })
                        }
                        None => None,
                    };
                    service.persist_update_events(
                        before.revision,
                        mission,
                        receipts,
                        idempotency,
                    )?;
                } else if !already_dispatched {
                    let mut mission = before.clone();
                    mission.revision = before.revision + 1;
                    mission.updated_at = now;
                    let receipts = chain_receipts_from(
                        &history,
                        &mission,
                        &mission.supervisor.subject,
                        Some("lanyte://kernel/supervisor"),
                        vec![(
                            EventSourceKind::KernelObserved,
                            LifecyclePayload::ProcessTerminationAttempted {
                                attempt_id,
                                generation: attempt.generation,
                                lease_generation: attempt.lease_generation.unwrap_or(1),
                                outcome: match outcome {
                                    ProcessTreeKill::Survivors => FallbackCancelOutcome::Survivors,
                                    ProcessTreeKill::Unknown => FallbackCancelOutcome::Unknown,
                                    _ => FallbackCancelOutcome::KillDispatched,
                                },
                            },
                        )],
                        Some("sysprims/dispatched"),
                    )?;
                    service.persist_update_events(before.revision, mission, receipts, None)?;
                }
            }
            return Ok(());
        }
        let Some(lease_generation) = attempt.lease_generation else {
            return Ok(());
        };
        let Some(lease_expires_at) = attempt.lease_expires_at else {
            return Ok(());
        };
        let Some(deadman_at) = attempt.deadman_at else {
            return Ok(());
        };
        let history = service.lifecycle_history(&before.mission_id.to_string())?;
        let mut payloads = Vec::new();
        let overdue = now >= deadman_at || now >= lease_expires_at;
        if restart
            && overdue
            && !history.iter().any(|event| {
                matches!(
                    &event.payload,
                    LifecyclePayload::RestartReconciled {
                        attempt_id: id,
                        lease_generation: generation,
                        overdue: true,
                        ..
                    } if *id == attempt_id && *generation == lease_generation
                )
            })
        {
            payloads.push((
                EventSourceKind::KernelObserved,
                LifecyclePayload::RestartReconciled {
                    attempt_id,
                    generation: attempt.generation,
                    lease_generation,
                    overdue: true,
                },
            ));
        }
        let mut next_state = attempt.state;
        if now >= deadman_at
            && !history_has_timer_edge(
                &history,
                attempt_id,
                LeaseTickKind::DeadmanFired,
                deadman_at,
            )
        {
            payloads.push((
                EventSourceKind::KernelObserved,
                LifecyclePayload::LeaseTick {
                    attempt_id,
                    generation: attempt.generation,
                    kind: LeaseTickKind::DeadmanFired,
                    prior_lease_generation: lease_generation,
                    result_lease_generation: lease_generation,
                    prior_lease_expires_at: lease_expires_at,
                    prior_deadman_at: deadman_at,
                    result_lease_expires_at: lease_expires_at,
                    result_deadman_at: deadman_at,
                    observed_at: now,
                    observation_source: ObservationSource::KernelClock,
                },
            ));
            if matches!(next_state, AttemptState::Running | AttemptState::Waiting) {
                payloads.push((
                    EventSourceKind::KernelObserved,
                    LifecyclePayload::AttemptStateChanged {
                        attempt_id,
                        generation: attempt.generation,
                        from: next_state,
                        to: AttemptState::Unresponsive,
                        reason: Some("deadman fired".to_owned()),
                        cause: Some(AttemptStateCause::DeadmanSilence),
                    },
                ));
                if before.phase == MissionPhase::Active {
                    payloads.push((
                        EventSourceKind::KernelObserved,
                        LifecyclePayload::MissionPhaseChanged {
                            from: MissionPhase::Active,
                            to: MissionPhase::RecoveryPending,
                            reason: Some("deadman fired".to_owned()),
                        },
                    ));
                }
                next_state = AttemptState::Unresponsive;
            }
        }
        let expire = now >= lease_expires_at
            && !history_has_timer_edge(
                &history,
                attempt_id,
                LeaseTickKind::Expired,
                lease_expires_at,
            );
        if expire {
            payloads.push((
                EventSourceKind::KernelObserved,
                LifecyclePayload::LeaseTick {
                    attempt_id,
                    generation: attempt.generation,
                    kind: LeaseTickKind::Expired,
                    prior_lease_generation: lease_generation,
                    result_lease_generation: lease_generation,
                    prior_lease_expires_at: lease_expires_at,
                    prior_deadman_at: deadman_at,
                    result_lease_expires_at: lease_expires_at,
                    result_deadman_at: deadman_at,
                    observed_at: now,
                    observation_source: ObservationSource::KernelClock,
                },
            ));
            if next_state.is_live() {
                let stand_down =
                    before.recovery_policy == lanyte_mission::RecoveryPolicy::StandDown;
                let next_phase = if stand_down {
                    MissionPhase::DeadlineExceeded
                } else {
                    MissionPhase::RecoveryPending
                };
                payloads.push((
                    EventSourceKind::KernelObserved,
                    LifecyclePayload::AttemptStateChanged {
                        attempt_id,
                        generation: attempt.generation,
                        from: next_state,
                        to: AttemptState::TimedOut,
                        reason: Some("lease expired".to_owned()),
                        cause: Some(AttemptStateCause::LeaseExpired),
                    },
                ));
                if before.phase != next_phase {
                    payloads.push((
                        EventSourceKind::KernelObserved,
                        LifecyclePayload::MissionPhaseChanged {
                            from: before.phase,
                            to: next_phase,
                            reason: Some("lease expired".to_owned()),
                        },
                    ));
                }
                if stand_down {
                    payloads.push((
                        EventSourceKind::KernelObserved,
                        LifecyclePayload::MissionTerminal {
                            phase: MissionPhase::DeadlineExceeded,
                            reason: MissionTerminalReason::MissionDeadlineExceeded,
                            terminal_entry_hash: "0".repeat(64),
                        },
                    ));
                }
                next_state = AttemptState::TimedOut;
            }
        }
        if payloads.is_empty() {
            return Ok(());
        }
        let mut mission = before.clone();
        mission.revision = before.revision + 1;
        mission.updated_at = now;
        if let Some(live) = mission
            .attempts
            .iter_mut()
            .find(|item| item.attempt_id == attempt_id)
        {
            live.state = next_state;
            if next_state == AttemptState::TimedOut {
                live.ended_at = Some(now);
                live.terminal_reason = Some(AttemptTerminalReason::AttemptTimedOut);
            }
        }
        if next_state == AttemptState::Unresponsive && mission.phase == MissionPhase::Active {
            mission.phase = MissionPhase::RecoveryPending;
        }
        if next_state == AttemptState::TimedOut {
            let stand_down = mission.recovery_policy == lanyte_mission::RecoveryPolicy::StandDown;
            mission.phase = if stand_down {
                MissionPhase::DeadlineExceeded
            } else {
                MissionPhase::RecoveryPending
            };
            if stand_down {
                mission.terminal_reason = Some(MissionTerminalReason::MissionDeadlineExceeded);
            }
            mission.current_attempt_id = None;
        }
        let mut receipts = chain_receipts_from(
            &history,
            &mission,
            &mission.supervisor.subject,
            Some("lanyte://kernel/supervisor"),
            payloads,
            Some("lease/supervisor"),
        )?;
        if mission.phase == MissionPhase::DeadlineExceeded {
            if let Some(hash) = receipts
                .last()
                .map(|receipt| receipt.event.entry_hash.clone())
            {
                mission.terminal_entry_hash = Some(hash.clone());
                if let Some(LifecyclePayload::MissionTerminal {
                    terminal_entry_hash,
                    ..
                }) = receipts
                    .last_mut()
                    .map(|receipt| &mut receipt.event.payload)
                {
                    *terminal_entry_hash = hash;
                }
            }
        }
        service.persist_update_events(before.revision, mission, receipts, None)?;
        Ok(())
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
        drop(sessions);
        if !events.is_empty() {
            self.fold_driver_observation(&caller, &mission, attempt_id, &events)?;
        }
        MissionControlResult::observe(request_id, body.mission_id, attempt_id, events)
            .map_err(MissionCommandError::internal)
    }

    fn fold_driver_observation(
        &self,
        caller: &crate::mission::VerifiedSession,
        before: &lanyte_mission::MissionRecord,
        attempt_id: Uuid,
        events: &[lanyte_mission::NormalizedHarnessEvent],
    ) -> Result<(), MissionCommandError> {
        let Some(service) = &self.mission_service else {
            return Ok(());
        };
        if !before.lease_policy.enabled {
            return Ok(());
        }
        let Some(attempt) = before
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == attempt_id)
        else {
            return Ok(());
        };
        let now = events
            .iter()
            .map(observation_time)
            .max()
            .unwrap_or_else(Utc::now);
        let exited = events.iter().rev().find_map(|event| match event {
            lanyte_mission::NormalizedHarnessEvent::Exited { success, .. } => Some(*success),
            _ => None,
        });
        let started_turn = events.iter().rev().find_map(|event| match event {
            lanyte_mission::NormalizedHarnessEvent::TurnProgress {
                turn_id, status, ..
            } if status == "started" => Some(turn_id.clone()),
            _ => None,
        });
        let mut payloads = Vec::new();
        let mut mission = before.clone();
        mission.revision = before.revision + 1;
        mission.updated_at = now;
        if let Some(success) = exited {
            if let Some(live) = mission
                .attempts
                .iter_mut()
                .find(|item| item.attempt_id == attempt_id)
            {
                if live.state.is_live() {
                    payloads.push((
                        EventSourceKind::DriverReported,
                        LifecyclePayload::AttemptStateChanged {
                            attempt_id,
                            generation: live.generation,
                            from: live.state,
                            to: if success {
                                AttemptState::Completed
                            } else {
                                AttemptState::Crashed
                            },
                            reason: Some("observed process exit".to_owned()),
                            cause: Some(AttemptStateCause::ProcessExit),
                        },
                    ));
                    live.state = if success {
                        AttemptState::Completed
                    } else {
                        AttemptState::Crashed
                    };
                    live.ended_at = Some(now);
                    live.terminal_reason = Some(if success {
                        AttemptTerminalReason::HarnessCompleted
                    } else {
                        AttemptTerminalReason::HarnessCrashed
                    });
                    mission.phase = MissionPhase::RecoveryPending;
                    mission.current_attempt_id = None;
                    payloads.push((
                        EventSourceKind::KernelObserved,
                        LifecyclePayload::MissionPhaseChanged {
                            from: before.phase,
                            to: MissionPhase::RecoveryPending,
                            reason: Some("observed process exit".to_owned()),
                        },
                    ));
                }
            }
        } else {
            let Some(prior_generation) = attempt.lease_generation else {
                return Ok(());
            };
            let Some(prior_lease) = attempt.lease_expires_at else {
                return Ok(());
            };
            let Some(prior_deadman) = attempt.deadman_at else {
                return Ok(());
            };
            let lease_seconds = before.lease_policy.lease_seconds.unwrap_or(600);
            let deadman_seconds = before.lease_policy.deadman_seconds.unwrap_or(300);
            let result_lease = now + Duration::seconds(lease_seconds as i64);
            let result_deadman = now + Duration::seconds(deadman_seconds as i64);
            let clocks_moved = result_deadman > prior_deadman || result_lease > prior_lease;
            if let Some(live) = mission
                .attempts
                .iter_mut()
                .find(|item| item.attempt_id == attempt_id)
            {
                if live.state == AttemptState::Unresponsive {
                    payloads.push((
                        EventSourceKind::KernelObserved,
                        LifecyclePayload::AttemptStateChanged {
                            attempt_id,
                            generation: live.generation,
                            from: AttemptState::Unresponsive,
                            to: AttemptState::Running,
                            reason: Some("driver observation".to_owned()),
                            cause: Some(AttemptStateCause::HarnessCompleted),
                        },
                    ));
                    live.state = AttemptState::Running;
                    if mission.phase == MissionPhase::RecoveryPending {
                        payloads.push((
                            EventSourceKind::KernelObserved,
                            LifecyclePayload::MissionPhaseChanged {
                                from: MissionPhase::RecoveryPending,
                                to: MissionPhase::Active,
                                reason: Some("driver observation".to_owned()),
                            },
                        ));
                        mission.phase = MissionPhase::Active;
                    }
                }
                if let Some(turn_id) = started_turn.clone() {
                    live.harness_turn_id = Some(turn_id);
                }
                if clocks_moved {
                    live.lease_generation = Some(prior_generation + 1);
                    live.lease_expires_at = Some(result_lease);
                    live.deadman_at = Some(result_deadman);
                    live.last_observed_at = Some(now);
                    live.last_observation_source = Some(ObservationSource::DriverEvent);
                    payloads.push((
                        EventSourceKind::KernelObserved,
                        LifecyclePayload::LeaseTick {
                            attempt_id,
                            generation: attempt.generation,
                            kind: LeaseTickKind::Renewed,
                            prior_lease_generation: prior_generation,
                            result_lease_generation: prior_generation + 1,
                            prior_lease_expires_at: prior_lease,
                            prior_deadman_at: prior_deadman,
                            result_lease_expires_at: result_lease,
                            result_deadman_at: result_deadman,
                            observed_at: now,
                            observation_source: ObservationSource::DriverEvent,
                        },
                    ));
                }
            }
        }
        if payloads.is_empty() && started_turn.is_none() {
            return Ok(());
        }
        if payloads.is_empty() {
            if let Some(live) = mission
                .attempts
                .iter_mut()
                .find(|item| item.attempt_id == attempt_id)
            {
                live.harness_turn_id = started_turn;
            }
        }
        if payloads.is_empty() {
            return Ok(());
        }
        let receipts = chain_receipts(
            &service.lifecycle_history(&before.mission_id.to_string())?,
            &mission,
            caller,
            payloads,
            None,
        )?;
        service.persist_update_events(before.revision, mission, receipts, None)?;
        Ok(())
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
                        cause: Some(lanyte_mission::AttemptStateCause::OperatorCancel),
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
                    cause: Some(lanyte_mission::AttemptStateCause::OperatorCancel),
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

fn observation_time(event: &lanyte_mission::NormalizedHarnessEvent) -> chrono::DateTime<Utc> {
    match event {
        lanyte_mission::NormalizedHarnessEvent::Started { occurred_at, .. }
        | lanyte_mission::NormalizedHarnessEvent::ToolProposed { occurred_at, .. }
        | lanyte_mission::NormalizedHarnessEvent::Exited { occurred_at, .. }
        | lanyte_mission::NormalizedHarnessEvent::TurnProgress { occurred_at, .. } => *occurred_at,
    }
}

fn history_has_timer_edge(
    history: &[LifecycleEvent],
    attempt_id: Uuid,
    kind: LeaseTickKind,
    deadline: chrono::DateTime<Utc>,
) -> bool {
    history.iter().any(|event| match &event.payload {
        LifecyclePayload::LeaseTick {
            attempt_id: id,
            kind: tick_kind,
            prior_deadman_at,
            prior_lease_expires_at,
            ..
        } if *id == attempt_id && *tick_kind == kind => match kind {
            LeaseTickKind::DeadmanFired => *prior_deadman_at == deadline,
            LeaseTickKind::Expired => *prior_lease_expires_at == deadline,
            LeaseTickKind::Renewed => false,
        },
        _ => false,
    })
}

fn chain_receipts(
    history: &[LifecycleEvent],
    mission: &lanyte_mission::MissionRecord,
    caller: &crate::mission::VerifiedSession,
    payloads: Vec<(EventSourceKind, LifecyclePayload)>,
    kernel_evidence: Option<&str>,
) -> Result<Vec<NewMissionProjectionReceipt>, MissionCommandError> {
    chain_receipts_from(
        history,
        mission,
        &caller.subject,
        Some(caller.trust_ref.as_str()),
        payloads,
        kernel_evidence,
    )
}

fn chain_receipts_from(
    history: &[LifecycleEvent],
    mission: &lanyte_mission::MissionRecord,
    subject: &str,
    trust_ref: Option<&str>,
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
            trust_ref.map(str::to_owned)
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
                subject: subject.to_owned(),
                producer_version: env!("CARGO_PKG_VERSION").to_owned(),
                assurance: match source_kind {
                    EventSourceKind::KernelObserved => ObservationLevel::KernelObserved,
                    EventSourceKind::DriverReported | EventSourceKind::HarnessReported => {
                        ObservationLevel::DriverObserved
                    }
                    EventSourceKind::OperatorCommand | EventSourceKind::VerifiedAttestation => {
                        ObservationLevel::ResourceAttested
                    }
                },
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
                trust_ref: trust_ref.map(str::to_owned),
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
        Some("mission.cancel") => {
            let record = serde_json::from_value(
                value
                    .pointer("/body/record")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            )
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
            let progress = serde_json::from_value(
                value
                    .pointer("/body/progress")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
            )
            .map_err(|err| MissionCommandError::internal(err.to_string()))?;
            let key = value
                .get("idempotency_key")
                .and_then(|value| value.as_str())
                .unwrap_or("replayed-cancel-key")
                .to_owned();
            let expected = value
                .get("expected_revision")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            Ok(MissionControlResult::Cancel {
                request_id,
                idempotency_key: key,
                expected_revision: expected,
                record: Box::new(record),
                progress,
            })
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

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct PendingCancelStub {
    kind: String,
    request_id: Uuid,
    expected_revision: u64,
}

fn parse_pending_cancel(result_json: &str) -> Option<PendingCancelStub> {
    let stub = serde_json::from_str::<PendingCancelStub>(result_json).ok()?;
    (stub.kind == "pending_cancel").then_some(stub)
}

fn pending_cancel_idempotency(
    key: &str,
    fingerprint: &str,
    request_id: Uuid,
    expected_revision: u64,
    owner_token: &str,
) -> lanyte_state::MissionMutationIdempotency {
    lanyte_state::MissionMutationIdempotency {
        key: key.to_owned(),
        request_fingerprint: fingerprint.to_owned(),
        operation: "mission.cancel".to_owned(),
        result_json: serde_json::json!({
            "kind": "pending_cancel",
            "request_id": request_id,
            "expected_revision": expected_revision,
        })
        .to_string(),
        owner_token: owner_token.to_owned(),
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
        let mut saw_turn = false;
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
                        NormalizedHarnessEvent::Exited { .. } => {}
                        NormalizedHarnessEvent::TurnProgress { .. } => saw_turn = true,
                    }
                }
            }
            if saw_started && saw_tool && saw_turn {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(saw_started && saw_tool && saw_turn);

        let latest = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("latest mission");
        let close_revision = latest.revision;

        service.fail_next_terminal_persist();
        let first = orchestrator
            .close_codex(
                &test_event(),
                MissionControlRequest::close(
                    Uuid::new_v4(),
                    "close:retry-terminated".to_owned(),
                    close_revision,
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

    #[tokio::test]
    async fn cancel_created_mission_without_attempt() {
        let (root, orchestrator, service, created, _) = launched_fixture(false).await;
        let cancelled = orchestrator
            .cancel_mission(
                &test_event(),
                MissionControlRequest::cancel(
                    Uuid::new_v4(),
                    "cancel:created-no-attempt".to_owned(),
                    created.revision,
                    created.mission_id,
                )
                .expect("cancel request"),
            )
            .await
            .expect("cancel created");
        let MissionControlResult::Cancel {
            record, progress, ..
        } = cancelled
        else {
            panic!("expected cancel result");
        };
        assert_eq!(record.phase, MissionPhase::Cancelled);
        assert!(progress.requested);
        assert!(progress.protocol.is_none());
        let history = service
            .lifecycle_history(&created.mission_id.to_string())
            .expect("history");
        lanyte_mission::validate_history(&record, &history).expect("history must validate");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn supervisor_clock_marks_deadman_then_expiry() {
        let (root, orchestrator, service, created, launched) = launched_fixture(true).await;
        let started = launched.attempts[0].started_at.expect("started");
        orchestrator
            .tick_leases_at(started + Duration::seconds(301))
            .await;
        let after_deadman = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("after deadman");
        assert_eq!(after_deadman.attempts[0].state, AttemptState::Unresponsive);
        assert_eq!(after_deadman.phase, MissionPhase::RecoveryPending);
        orchestrator
            .tick_leases_at(started + Duration::seconds(601))
            .await;
        let after_expire = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("after expire");
        assert_eq!(after_expire.phase, MissionPhase::RecoveryPending);
        assert_eq!(after_expire.attempts[0].state, AttemptState::TimedOut);
        let history = service
            .lifecycle_history(&created.mission_id.to_string())
            .expect("history");
        lanyte_mission::validate_history(&after_expire, &history).expect("history must validate");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn live_protocol_cancel_binds_turn_and_folds() {
        let (root, orchestrator, service, created, launched) = launched_fixture(true).await;
        for _ in 0..20 {
            let _ = orchestrator
                .observe_codex(
                    &test_event(),
                    MissionControlRequest::observe(Uuid::new_v4(), created.mission_id)
                        .expect("observe"),
                )
                .await;
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let latest = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("latest");
        let cancelled = orchestrator
            .cancel_mission(
                &test_event(),
                MissionControlRequest::cancel(
                    Uuid::new_v4(),
                    "cancel:live-protocol-1".to_owned(),
                    latest.revision,
                    created.mission_id,
                )
                .expect("cancel"),
            )
            .await
            .expect("cancel live");
        let MissionControlResult::Cancel {
            record, progress, ..
        } = cancelled
        else {
            panic!("expected cancel result");
        };
        assert!(progress.requested);
        assert_eq!(
            progress.protocol.as_ref().map(|item| item.outcome.clone()),
            Some(lanyte_mission::ProtocolCancelOutcome::Interrupted)
        );
        assert_eq!(record.phase, MissionPhase::Cancelled);
        assert_eq!(record.attempts[0].state, AttemptState::Cancelled);
        assert_eq!(
            record.attempts[0].harness_turn_id.as_deref(),
            Some("turn_test")
        );
        let history = service
            .lifecycle_history(&created.mission_id.to_string())
            .expect("history");
        lanyte_mission::validate_history(&record, &history).expect("history must validate");
        let _ = launched;
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn cancel_retries_original_revision_after_pipeline_crash() {
        let (root, orchestrator, service, created, launched) = launched_fixture(true).await;
        for _ in 0..20 {
            let _ = orchestrator
                .observe_codex(
                    &test_event(),
                    MissionControlRequest::observe(Uuid::new_v4(), created.mission_id)
                        .expect("observe"),
                )
                .await;
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let latest = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("latest");
        service.fail_after_cancelling_persist();
        let original = MissionControlRequest::cancel(
            Uuid::new_v4(),
            "cancel:crash-resume-01".to_owned(),
            latest.revision,
            created.mission_id,
        )
        .expect("cancel");
        let original_id = original.request_id();
        let first = orchestrator
            .cancel_mission(&test_event(), original)
            .await
            .expect_err("pipeline crash");
        assert!(first.message.contains("injected cancel pipeline crash"));
        let midway = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("midway");
        assert_eq!(midway.attempts[0].state, AttemptState::Cancelling);
        orchestrator.tick_leases().await;
        let after_supervisor = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("after supervisor");
        assert_eq!(after_supervisor.phase, MissionPhase::Cancelled);
        let retried = orchestrator
            .cancel_mission(
                &test_event(),
                MissionControlRequest::cancel(
                    Uuid::new_v4(),
                    "cancel:crash-resume-01".to_owned(),
                    latest.revision,
                    created.mission_id,
                )
                .expect("retry cancel"),
            )
            .await
            .expect("exact replay after supervisor complete");
        let MissionControlResult::Cancel {
            request_id,
            expected_revision,
            record,
            ..
        } = retried
        else {
            panic!("expected cancel result");
        };
        assert_eq!(request_id, original_id);
        assert_eq!(expected_revision, latest.revision);
        assert_eq!(record.phase, MissionPhase::Cancelled);
        let _ = launched;
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn distinct_cancel_key_is_rejected_while_original_is_pending() {
        let (root, orchestrator, service, created, launched) = launched_fixture(true).await;
        for _ in 0..20 {
            let _ = orchestrator
                .observe_codex(
                    &test_event(),
                    MissionControlRequest::observe(Uuid::new_v4(), created.mission_id)
                        .expect("observe"),
                )
                .await;
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let latest = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("latest");
        service.fail_after_cancelling_persist();
        let original = MissionControlRequest::cancel(
            Uuid::new_v4(),
            "cancel:original-key-01".to_owned(),
            latest.revision,
            created.mission_id,
        )
        .expect("cancel A");
        let original_id = original.request_id();
        let _ = orchestrator
            .cancel_mission(&test_event(), original)
            .await
            .expect_err("pipeline crash");
        let midway = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("midway");
        let hijack = orchestrator
            .cancel_mission(
                &test_event(),
                MissionControlRequest::cancel(
                    Uuid::new_v4(),
                    "cancel:other-key-0002".to_owned(),
                    midway.revision,
                    created.mission_id,
                )
                .expect("cancel B"),
            )
            .await
            .expect_err("distinct key must not take over");
        assert!(hijack.message.contains("already in flight"));
        orchestrator.tick_leases().await;
        let retried = orchestrator
            .cancel_mission(
                &test_event(),
                MissionControlRequest::cancel(
                    Uuid::new_v4(),
                    "cancel:original-key-01".to_owned(),
                    latest.revision,
                    created.mission_id,
                )
                .expect("replay A"),
            )
            .await
            .expect("original key replays");
        let MissionControlResult::Cancel {
            request_id, record, ..
        } = retried
        else {
            panic!("expected cancel result");
        };
        assert_eq!(request_id, original_id);
        assert_eq!(record.phase, MissionPhase::Cancelled);
        let _ = launched;
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn nonterminal_cancel_keeps_pending_stub_until_supervisor_completes() {
        let (root, orchestrator, service, created, launched) = launched_fixture(true).await;
        for _ in 0..20 {
            let _ = orchestrator
                .observe_codex(
                    &test_event(),
                    MissionControlRequest::observe(Uuid::new_v4(), created.mission_id)
                        .expect("observe"),
                )
                .await;
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let latest = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("latest");
        service.force_nonterminal_cancel();
        let original = MissionControlRequest::cancel(
            Uuid::new_v4(),
            "cancel:nonterminal-key-01".to_owned(),
            latest.revision,
            created.mission_id,
        )
        .expect("cancel A");
        let original_id = original.request_id();
        let accepted = orchestrator
            .cancel_mission(&test_event(), original)
            .await
            .expect("accepted nonterminal cancel");
        let MissionControlResult::Cancel { record, .. } = accepted else {
            panic!("expected cancel result");
        };
        assert_ne!(record.phase, MissionPhase::Cancelled);
        let midway = service
            .visible_mission(
                &created.mission_id.to_string(),
                &TestVerifier.verify("valid-session-secret").unwrap(),
            )
            .expect("midway");
        let hijack = orchestrator
            .cancel_mission(
                &test_event(),
                MissionControlRequest::cancel(
                    Uuid::new_v4(),
                    "cancel:other-key-0003".to_owned(),
                    midway.revision,
                    created.mission_id,
                )
                .expect("cancel B"),
            )
            .await
            .expect_err("distinct key must not take over");
        assert!(hijack.message.contains("already in flight"));
        orchestrator.tick_leases().await;
        let retried = orchestrator
            .cancel_mission(
                &test_event(),
                MissionControlRequest::cancel(
                    Uuid::new_v4(),
                    "cancel:nonterminal-key-01".to_owned(),
                    latest.revision,
                    created.mission_id,
                )
                .expect("replay A"),
            )
            .await
            .expect("original key replays");
        let MissionControlResult::Cancel {
            request_id, record, ..
        } = retried
        else {
            panic!("expected cancel result");
        };
        assert_eq!(request_id, original_id);
        assert_eq!(record.phase, MissionPhase::Cancelled);
        let _ = launched;
        let _ = std::fs::remove_dir_all(root);
    }

    async fn launched_fixture(
        launch: bool,
    ) -> (
        std::path::PathBuf,
        crate::Orchestrator,
        crate::MissionService,
        lanyte_mission::MissionRecord,
        lanyte_mission::MissionRecord,
    ) {
        use std::path::PathBuf;
        use std::sync::{Arc, Mutex};

        use lanyte_mission::{MissionCreateBody, MissionLaunchBody, RecoveryPolicy};
        use lanyte_state::{StatePaths, StateStore};

        let root = std::env::temp_dir().join(format!("lanyte-wave3-{}", Uuid::new_v4()));
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
                    format!("create:wave3:{}", Uuid::new_v4()),
                    MissionCreateBody {
                        goal: "Exercise lease and cancel".to_owned(),
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
        if !launch {
            return (root, orchestrator, service, created.clone(), created);
        }
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
                    format!("launch:wave3:{}", Uuid::new_v4()),
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
        (root, orchestrator, service, created, launched)
    }
}
