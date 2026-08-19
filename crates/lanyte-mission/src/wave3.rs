//! Persist-time Wave 3 semantic layer (SEM-C / SEM-L / SEM-P / SEM-A).

use std::collections::{HashMap, HashSet};

use chrono::{Duration, Utc};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    AttemptState, AttemptStateCause, EventSourceKind, FallbackCancelOutcome, InvariantError,
    LeaseTickKind, LifecycleEvent, LifecyclePayload, MissionPhase, MissionRecord,
    ObservationSource, Principal, ProtocolCancelOutcome,
};

const MUTATING_OPERATIONS: &[&str] = &[
    "mission.create",
    "mission.launch",
    "mission.close",
    "mission.cancel",
];

#[derive(Debug, Clone, Deserialize)]
pub struct ControlBinding {
    pub operation: String,
    pub idempotency_key: String,
    pub request_fingerprint: String,
    pub original_result_hash: String,
    pub evidence_ref: String,
    pub request: Value,
    pub result: Value,
}

type Fence = (uuid::Uuid, u64, u64);

pub fn semantic_violation_codes(
    mission: &MissionRecord,
    events: &[LifecycleEvent],
) -> Vec<&'static str> {
    semantic_violation_codes_with_control(mission, events, &[])
}

pub fn semantic_violation_codes_with_control(
    mission: &MissionRecord,
    events: &[LifecycleEvent],
    records: &[ControlBinding],
) -> Vec<&'static str> {
    semantic_violation_codes_for_fixture(mission, events, records, &[])
}

pub fn semantic_violation_codes_for_fixture(
    mission: &MissionRecord,
    events: &[LifecycleEvent],
    records: &[ControlBinding],
    reports: &[crate::DriverCapabilityReport],
) -> Vec<&'static str> {
    let mut violations = wave3_violations(mission, events, records);
    for report in reports {
        let mut names = HashSet::new();
        for capability in &report.capabilities {
            if !names.insert(capability.name) {
                note(
                    &mut violations,
                    "SEM-D01",
                    "capability names must be unique in a report",
                );
            }
        }
    }
    violations.into_iter().map(|err| err.field).collect()
}

pub(crate) fn validate_wave3_semantics(
    mission: &MissionRecord,
    events: &[LifecycleEvent],
) -> Result<(), InvariantError> {
    match wave3_violations(mission, events, &[]).into_iter().next() {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

fn wave3_violations(
    mission: &MissionRecord,
    events: &[LifecycleEvent],
    records: &[ControlBinding],
) -> Vec<InvariantError> {
    let mut violations = Vec::new();
    let lease_enabled = mission.lease_policy.enabled;
    let lease_seconds = mission.lease_policy.lease_seconds;
    let deadman_seconds = mission.lease_policy.deadman_seconds;
    let attempts: HashMap<_, _> = mission
        .attempts
        .iter()
        .map(|attempt| (attempt.attempt_id, attempt))
        .collect();

    let mut cancel_requests: HashMap<Fence, u64> = HashMap::new();
    let mut protocol_proofs: HashMap<Fence, u64> = HashMap::new();
    let mut process_proofs: HashMap<Fence, u64> = HashMap::new();
    let mut running_leases: HashMap<uuid::Uuid, RunningLease> = HashMap::new();
    let mut consumed_timer_edges: HashSet<(uuid::Uuid, u64, LeaseTickKind, chrono::DateTime<Utc>)> =
        HashSet::new();
    let mut consumed_restarts: HashSet<(uuid::Uuid, u64)> = HashSet::new();
    let mut cancel_state_edges: HashMap<uuid::Uuid, CancelEdge> = HashMap::new();
    let mut cancel_requested_seen = false;
    let mut non_success_process = false;

    for event in events {
        let sequence = event.sequence;
        match &event.payload {
            LifecyclePayload::AttemptCreated {
                attempt_id,
                ordinal,
                generation,
                recovery_relation,
                predecessor_attempt_id,
            } => {
                if !attempts.get(attempt_id).is_some_and(|attempt| {
                    attempt.ordinal == *ordinal
                        && attempt.generation == *generation
                        && attempt.recovery_relation == *recovery_relation
                        && attempt.predecessor_attempt_id == *predecessor_attempt_id
                }) {
                    note(
                        &mut violations,
                        "SEM-T09",
                        "attempt_created must match the attempt record identity",
                    );
                }
            }
            LifecyclePayload::AuthorizationBound {
                authorizer,
                authorization_ref,
            } => {
                let Some(durable) = mission.authorizer.as_ref() else {
                    note(
                        &mut violations,
                        "SEM-A01",
                        "authorization_bound requires a durable authorizer",
                    );
                    continue;
                };
                if !authorizer.matches_principal(durable)
                    || Some(authorization_ref.as_str()) != mission.authorization_ref.as_deref()
                    || event.source.evidence_ref.as_deref()
                        != Some(authorizer.attestation_ref.as_str())
                {
                    note(
                        &mut violations,
                        "SEM-A01",
                        "authorization_bound must match the durable attested principal and source evidence",
                    );
                }
            }
            LifecyclePayload::CancelRequested {
                attempt_id,
                generation,
                lease_generation,
                authorizer,
                authorization_ref,
            } => {
                cancel_requested_seen = true;
                let durable: &Principal = mission.authorizer.as_ref().unwrap_or(&mission.initiator);
                if event.source.kind != EventSourceKind::OperatorCommand
                    || event.source.assurance != crate::ObservationLevel::ResourceAttested
                    || event.source.evidence_ref.is_none()
                    || event.source.subject != durable.subject
                    || !authorizer.matches_principal(durable)
                    || authorization_ref.as_deref() != mission.authorization_ref.as_deref()
                {
                    note(
                        &mut violations,
                        "SEM-A04",
                        "cancel_requested requires attested operator evidence bound to the durable principal",
                    );
                }
                if let (Some(attempt_id), Some(generation), Some(lease_generation)) =
                    (*attempt_id, *generation, *lease_generation)
                {
                    let key = (attempt_id, generation, lease_generation);
                    if cancel_requests.contains_key(&key) {
                        note(
                            &mut violations,
                            "SEM-C08",
                            "a fence may emit at most one cancel_requested",
                        );
                    }
                    cancel_requests.insert(key, sequence);
                    check_live_lease(
                        lease_enabled,
                        &running_leases,
                        Some(attempt_id),
                        Some(lease_generation),
                        &mut violations,
                    );
                }
            }
            LifecyclePayload::ProtocolCancelAttempted {
                attempt_id,
                generation,
                lease_generation,
                thread_id,
                turn_id,
                outcome,
            } => {
                if !matches!(
                    event.source.kind,
                    EventSourceKind::DriverReported | EventSourceKind::HarnessReported
                ) {
                    note(
                        &mut violations,
                        "SEM-P01",
                        "protocol cancel evidence must stay driver/harness",
                    );
                }
                let key = (*attempt_id, *generation, *lease_generation);
                if cancel_requests
                    .get(&key)
                    .copied()
                    .is_none_or(|request_seq| sequence <= request_seq)
                {
                    note(
                        &mut violations,
                        "SEM-C01",
                        "protocol cancel attempt requires an earlier same-fence cancel_requested",
                    );
                }
                if *outcome == ProtocolCancelOutcome::Interrupted {
                    let attempt = attempts.get(attempt_id).copied();
                    let matches_attempt = attempt.is_some_and(|attempt| {
                        attempt.generation == *generation
                            && attempt.lease_generation == Some(*lease_generation)
                            && thread_id.as_deref().is_some_and(|value| !value.is_empty())
                            && turn_id.as_deref().is_some_and(|value| !value.is_empty())
                            && attempt.harness_thread_id.as_deref() == thread_id.as_deref()
                            && attempt.harness_turn_id.as_deref() == turn_id.as_deref()
                    });
                    if matches_attempt {
                        protocol_proofs
                            .insert((*attempt_id, *generation, *lease_generation), sequence);
                    } else {
                        note(
                            &mut violations,
                            "SEM-C02",
                            "interrupted proof must bind attempt, lease, thread, and turn",
                        );
                    }
                }
                check_live_lease(
                    lease_enabled,
                    &running_leases,
                    Some(*attempt_id),
                    Some(*lease_generation),
                    &mut violations,
                );
            }
            LifecyclePayload::ProcessTerminationAttempted {
                attempt_id,
                generation,
                lease_generation,
                outcome,
            } => {
                if event.source.kind != EventSourceKind::KernelObserved {
                    note(
                        &mut violations,
                        "SEM-P01",
                        "process membership evidence must be kernel-observed",
                    );
                }
                let key = (*attempt_id, *generation, *lease_generation);
                if cancel_requests
                    .get(&key)
                    .copied()
                    .is_none_or(|request_seq| sequence <= request_seq)
                {
                    note(
                        &mut violations,
                        "SEM-C01",
                        "process cancel attempt requires an earlier same-fence cancel_requested",
                    );
                }
                if *outcome == FallbackCancelOutcome::Cleared
                    && attempts.get(attempt_id).is_some_and(|attempt| {
                        attempt.generation == *generation
                            && attempt.lease_generation == Some(*lease_generation)
                    })
                {
                    process_proofs.insert((*attempt_id, *generation, *lease_generation), sequence);
                } else if matches!(
                    *outcome,
                    FallbackCancelOutcome::KillDispatched
                        | FallbackCancelOutcome::Survivors
                        | FallbackCancelOutcome::Unknown
                ) {
                    non_success_process = true;
                }
                check_live_lease(
                    lease_enabled,
                    &running_leases,
                    Some(*attempt_id),
                    Some(*lease_generation),
                    &mut violations,
                );
            }
            LifecyclePayload::LeaseStarted {
                attempt_id,
                generation,
                lease_generation,
                lease_expires_at,
                deadman_at,
                observed_at,
                observation_source,
            } => {
                if !lease_enabled {
                    note(
                        &mut violations,
                        "SEM-L04",
                        "lease events require an enabled lease policy",
                    );
                }
                if event.source.kind != EventSourceKind::KernelObserved {
                    note(
                        &mut violations,
                        "SEM-L09",
                        "lease_started must be kernel-observed",
                    );
                }
                if *observed_at > event.occurred_at || event.occurred_at > event.recorded_at {
                    note(
                        &mut violations,
                        "SEM-M02",
                        "lease timestamps cannot move backward",
                    );
                }
                let created_ok = events.iter().any(|prior| {
                    prior.sequence < event.sequence
                        && matches!(
                            &prior.payload,
                            LifecyclePayload::AttemptCreated {
                                attempt_id: id,
                                generation: created_generation,
                                ..
                            } if id == attempt_id && created_generation == generation
                        )
                });
                if !created_ok
                    || *lease_generation != 1
                    || running_leases.contains_key(attempt_id)
                    || attempts
                        .get(attempt_id)
                        .is_some_and(|attempt| attempt.generation != *generation)
                {
                    note(
                        &mut violations,
                        "SEM-L11",
                        "lease_started must be the generation-1 anchor",
                    );
                }
                if let Some(seconds) = lease_seconds {
                    if *lease_expires_at != *observed_at + Duration::seconds(seconds as i64) {
                        note(
                            &mut violations,
                            "SEM-L06",
                            "lease deadline must derive from observed_at",
                        );
                    }
                }
                if let Some(seconds) = deadman_seconds {
                    if *deadman_at != *observed_at + Duration::seconds(seconds as i64) {
                        note(
                            &mut violations,
                            "SEM-L06",
                            "deadman deadline must derive from observed_at",
                        );
                    }
                }
                running_leases.insert(
                    *attempt_id,
                    RunningLease {
                        lease_generation: *lease_generation,
                        lease_expires_at: *lease_expires_at,
                        deadman_at: *deadman_at,
                        last_observed_at: *observed_at,
                        last_observation_source: *observation_source,
                    },
                );
            }
            LifecyclePayload::LeaseTick {
                attempt_id,
                kind,
                prior_lease_generation,
                result_lease_generation,
                prior_lease_expires_at,
                prior_deadman_at,
                result_lease_expires_at,
                result_deadman_at,
                observed_at,
                observation_source,
                ..
            } => {
                if !lease_enabled {
                    note(
                        &mut violations,
                        "SEM-L04",
                        "lease events require an enabled lease policy",
                    );
                }
                if event.source.kind != EventSourceKind::KernelObserved {
                    note(
                        &mut violations,
                        "SEM-L09",
                        "lease_tick must be kernel-observed",
                    );
                }
                if *observed_at > event.occurred_at || event.occurred_at > event.recorded_at {
                    note(
                        &mut violations,
                        "SEM-M02",
                        "lease timestamps cannot move backward",
                    );
                }
                let Some(running) = running_leases.get(attempt_id).cloned() else {
                    note(
                        &mut violations,
                        "SEM-L11",
                        "lease_tick requires a lease_started anchor",
                    );
                    continue;
                };
                if *prior_lease_generation != running.lease_generation
                    || *prior_lease_expires_at != running.lease_expires_at
                    || *prior_deadman_at != running.deadman_at
                {
                    note(
                        &mut violations,
                        "SEM-L01",
                        "tick prior lease_generation is stale",
                    );
                    note(
                        &mut violations,
                        "SEM-L08",
                        "tick prior tuple must match the running lease",
                    );
                }
                match kind {
                    LeaseTickKind::Renewed => {
                        if *result_lease_generation != prior_lease_generation + 1 {
                            note(
                                &mut violations,
                                "SEM-L01",
                                "renewed ticks must advance lease_generation",
                            );
                        }
                        if *observation_source == ObservationSource::KernelClock {
                            note(&mut violations, "SEM-L09", "kernel clock cannot renew");
                        }
                        if *observation_source == ObservationSource::ProcessProbe
                            && *result_lease_expires_at != *prior_lease_expires_at
                        {
                            note(
                                &mut violations,
                                "SEM-L06",
                                "process probe may move deadman only",
                            );
                        }
                        if let Some(seconds) = deadman_seconds {
                            if *result_deadman_at
                                != *observed_at + Duration::seconds(seconds as i64)
                            {
                                note(
                                    &mut violations,
                                    "SEM-L06",
                                    "renewed deadman must derive from observed_at",
                                );
                            }
                        }
                        if matches!(
                            *observation_source,
                            ObservationSource::DriverEvent | ObservationSource::HarnessEvent
                        ) {
                            if let Some(seconds) = lease_seconds {
                                if *result_lease_expires_at
                                    != *observed_at + Duration::seconds(seconds as i64)
                                {
                                    note(
                                        &mut violations,
                                        "SEM-L06",
                                        "renewed lease deadline must derive from observed_at",
                                    );
                                }
                            }
                        }
                        let moved = *result_deadman_at > *prior_deadman_at
                            || (*observation_source != ObservationSource::ProcessProbe
                                && *result_lease_expires_at > *prior_lease_expires_at);
                        if !moved {
                            note(
                                &mut violations,
                                "SEM-L10",
                                "renewed ticks must move a permitted clock",
                            );
                        }
                        running_leases.insert(
                            *attempt_id,
                            RunningLease {
                                lease_generation: *result_lease_generation,
                                lease_expires_at: *result_lease_expires_at,
                                deadman_at: *result_deadman_at,
                                last_observed_at: *observed_at,
                                last_observation_source: *observation_source,
                            },
                        );
                    }
                    LeaseTickKind::DeadmanFired | LeaseTickKind::Expired => {
                        if *prior_lease_generation != *result_lease_generation
                            || *prior_lease_expires_at != *result_lease_expires_at
                            || *prior_deadman_at != *result_deadman_at
                        {
                            note(
                                &mut violations,
                                "SEM-L08",
                                "fire/expire ticks cannot rewrite clocks",
                            );
                        }
                        if *observation_source != ObservationSource::KernelClock {
                            note(
                                &mut violations,
                                "SEM-L09",
                                "fire/expire ticks are kernel-clock only",
                            );
                        }
                        let deadline = if *kind == LeaseTickKind::DeadmanFired {
                            running.deadman_at
                        } else {
                            running.lease_expires_at
                        };
                        if event.occurred_at < deadline {
                            note(
                                &mut violations,
                                "SEM-L07",
                                "fire/expire cannot occur before the deadline",
                            );
                        }
                        let edge = (*attempt_id, running.lease_generation, *kind, deadline);
                        if !consumed_timer_edges.insert(edge) {
                            note(
                                &mut violations,
                                "SEM-L10",
                                "fire/expire edges cannot be emitted twice",
                            );
                        }
                    }
                }
            }
            LifecyclePayload::RestartReconciled {
                attempt_id,
                lease_generation,
                overdue,
                ..
            } => {
                if !lease_enabled {
                    note(
                        &mut violations,
                        "SEM-L04",
                        "lease events require an enabled lease policy",
                    );
                }
                if event.source.kind != EventSourceKind::KernelObserved {
                    note(
                        &mut violations,
                        "SEM-L09",
                        "restart_reconciled must be kernel-observed",
                    );
                }
                if !*overdue {
                    note(
                        &mut violations,
                        "SEM-L07",
                        "restart_reconciled exists only for overdue restarts",
                    );
                }
                check_live_lease(
                    lease_enabled,
                    &running_leases,
                    Some(*attempt_id),
                    Some(*lease_generation),
                    &mut violations,
                );
                let generation = running_leases
                    .get(attempt_id)
                    .map(|running| running.lease_generation)
                    .unwrap_or(*lease_generation);
                if !consumed_restarts.insert((*attempt_id, generation)) {
                    note(
                        &mut violations,
                        "SEM-L03",
                        "at most one overdue restart per lease generation",
                    );
                }
                if let Some(running) = running_leases.get(attempt_id) {
                    if event.occurred_at < running.deadman_at
                        && event.occurred_at < running.lease_expires_at
                    {
                        note(
                            &mut violations,
                            "SEM-L07",
                            "overdue restart must be at or after a deadline",
                        );
                    }
                }
            }
            LifecyclePayload::AttemptStateChanged {
                attempt_id,
                generation,
                to,
                cause,
                ..
            } => {
                if matches!(
                    *to,
                    AttemptState::Crashed | AttemptState::TimedOut | AttemptState::Lost
                ) && *cause == Some(AttemptStateCause::DeadmanSilence)
                {
                    note(
                        &mut violations,
                        "SEM-C05",
                        "deadman silence cannot fold crashed/timed_out/lost",
                    );
                }
                if *to == AttemptState::Cancelled {
                    cancel_state_edges.insert(
                        *attempt_id,
                        CancelEdge {
                            sequence,
                            source: event.source.kind,
                            cause: *cause,
                            generation: *generation,
                        },
                    );
                }
            }
            _ => {}
        }
    }

    let wave3_cancel = cancel_requested_seen
        || !protocol_proofs.is_empty()
        || !process_proofs.is_empty()
        || non_success_process;
    for attempt in attempts.values() {
        if !wave3_cancel || attempt.state != AttemptState::Cancelled {
            continue;
        }
        let key = (
            attempt.attempt_id,
            attempt.generation,
            attempt.lease_generation.unwrap_or(1),
        );
        let request_seq = cancel_requests.get(&key).copied();
        let proto_seq = protocol_proofs.get(&key).copied();
        let proc_seq = process_proofs.get(&key).copied();
        let proof = [proto_seq, proc_seq].into_iter().flatten().min();
        if request_seq.is_none()
            || proof.is_none_or(|seq| request_seq.is_some_and(|req| seq <= req))
        {
            note(
                &mut violations,
                "SEM-C01",
                "cancelled attempts need request-before-proof",
            );
        }
        if proto_seq.is_none() && proc_seq.is_none() {
            note(
                &mut violations,
                "SEM-C03",
                "process fallback folds cancelled only on cleared",
            );
            note(
                &mut violations,
                "SEM-C02",
                "cancelled attempts need protocol or process proof",
            );
        }
        let expected_cause = if proto_seq.is_some() {
            Some(AttemptStateCause::ProtocolInterrupt)
        } else {
            Some(AttemptStateCause::ProcessExit)
        };
        let Some(edge) = cancel_state_edges.get(&attempt.attempt_id) else {
            note(
                &mut violations,
                "SEM-C10",
                "cancelled attempts need a kernel cancelled edge",
            );
            continue;
        };
        if edge.source != EventSourceKind::KernelObserved
            || edge.generation != attempt.generation
            || edge.cause != expected_cause
            || proof.is_some_and(|seq| edge.sequence <= seq.max(request_seq.unwrap_or(0)))
        {
            note(
                &mut violations,
                "SEM-C10",
                "cancelled edge must follow matching same-fence proof",
            );
        }
        if mission.phase == MissionPhase::Cancelled
            && events
                .last()
                .is_some_and(|event| event.sequence <= edge.sequence)
        {
            note(
                &mut violations,
                "SEM-C10",
                "mission_terminal must follow the cancelled edge",
            );
        }
    }

    if wave3_cancel && mission.phase.is_terminal() && mission.phase != MissionPhase::Cancelled {
        note(
            &mut violations,
            "SEM-C01",
            "cancel_requested cannot fold an incompatible terminal phase",
        );
    }
    if wave3_cancel && mission.phase == MissionPhase::Cancelled {
        let no_attempt = attempts.is_empty();
        if (!cancel_requested_seen || !no_attempt)
            && process_proofs.is_empty()
            && protocol_proofs.is_empty()
        {
            if non_success_process {
                note(
                    &mut violations,
                    "SEM-C03",
                    "process fallback folds cancelled only on cleared",
                );
            }
            note(
                &mut violations,
                "SEM-C02",
                "mission cancelled requires protocol, process, or created-no-attempt proof",
            );
        }
        if !attempts.is_empty()
            && attempts
                .values()
                .all(|attempt| attempt.state != AttemptState::Cancelled)
        {
            note(
                &mut violations,
                "SEM-C09",
                "cancelled missions must bind a cancelled attempt",
            );
        }
    }

    for attempt in attempts.values() {
        if !lease_enabled {
            if attempt.lease_expires_at.is_some()
                || attempt.deadman_at.is_some()
                || attempt.lease_generation.is_some()
            {
                note(
                    &mut violations,
                    "SEM-L04",
                    "disabled lease cannot carry runtime fields",
                );
            }
            continue;
        }
        if attempt.state.is_live()
            && (attempt.lease_expires_at.is_none()
                || attempt.deadman_at.is_none()
                || attempt.lease_generation.is_none()
                || attempt.last_observed_at.is_none()
                || attempt.last_observation_source.is_none())
        {
            note(
                &mut violations,
                "SEM-L04",
                "live leased attempts need wall deadlines and observation",
            );
        }
        if let Some(running) = running_leases.get(&attempt.attempt_id) {
            if Some(running.lease_generation) != attempt.lease_generation
                || Some(running.lease_expires_at) != attempt.lease_expires_at
                || Some(running.deadman_at) != attempt.deadman_at
                || Some(running.last_observed_at) != attempt.last_observed_at
                || Some(running.last_observation_source) != attempt.last_observation_source
            {
                note(
                    &mut violations,
                    "SEM-L08",
                    "final lease projection must match the attempt record",
                );
            }
        } else if attempt.lease_generation.is_some() {
            note(
                &mut violations,
                "SEM-L11",
                "enabled lease runtime requires lease_started",
            );
        }
    }
    if let (Some(last), updated) = (events.last(), mission.updated_at) {
        if updated < last.recorded_at {
            note(
                &mut violations,
                "SEM-M06",
                "mission updated_at cannot precede the last event",
            );
        }
    }
    if mission.phase.is_terminal() {
        match events.last() {
            Some(event) => match &event.payload {
                LifecyclePayload::MissionTerminal {
                    phase,
                    reason,
                    terminal_entry_hash,
                } if *phase == mission.phase
                    && Some(*reason) == mission.terminal_reason
                    && Some(terminal_entry_hash.as_str())
                        == mission.terminal_entry_hash.as_deref()
                    && terminal_entry_hash == &event.entry_hash => {}
                _ => {
                    note(
                        &mut violations,
                        "SEM-C06",
                        "terminal history must end with matching mission_terminal",
                    );
                }
            },
            None => {
                note(
                    &mut violations,
                    "SEM-C06",
                    "terminal history must end with matching mission_terminal",
                );
            }
        }
    }
    check_control_bindings(mission, events, records, &mut violations);
    violations
}

#[derive(Clone)]
struct RunningLease {
    lease_generation: u64,
    lease_expires_at: chrono::DateTime<Utc>,
    deadman_at: chrono::DateTime<Utc>,
    last_observed_at: chrono::DateTime<Utc>,
    last_observation_source: ObservationSource,
}

struct CancelEdge {
    sequence: u64,
    source: EventSourceKind,
    cause: Option<AttemptStateCause>,
    generation: u64,
}

fn check_control_bindings(
    mission: &MissionRecord,
    events: &[LifecycleEvent],
    records: &[ControlBinding],
    violations: &mut Vec<InvariantError>,
) {
    if records.is_empty() {
        return;
    }
    let mut fingerprints_by_key: HashMap<String, HashSet<String>> = HashMap::new();
    let mut results_by_key_fp: HashMap<(String, String), HashSet<String>> = HashMap::new();
    let mut cancel_evidence: HashSet<String> = HashSet::new();
    for record in records {
        let request_hash = control_content_hash(&record.request);
        let result_hash = control_content_hash(&record.result);
        let request_id = record.request.get("request_id").and_then(Value::as_str);
        let result_id = record.result.get("request_id").and_then(Value::as_str);
        let request_op = record.request.get("operation").and_then(Value::as_str);
        let result_op = record.result.get("operation").and_then(Value::as_str);
        let request_key = record
            .request
            .get("idempotency_key")
            .and_then(Value::as_str);
        let result_key = record.result.get("idempotency_key").and_then(Value::as_str);
        let request_kind = record.request.get("kind").and_then(Value::as_str);
        let result_kind = record.result.get("kind").and_then(Value::as_str);
        let result_original = record
            .result
            .get("original_result_hash")
            .and_then(Value::as_str);
        let request_fp = record
            .request
            .get("request_fingerprint")
            .and_then(Value::as_str);
        let result_fp = record
            .result
            .get("request_fingerprint")
            .and_then(Value::as_str);
        let identity_ok = control_identities_match(mission.mission_id, record);
        let ids_ok =
            request_id.is_some_and(canonical_uuid_v4) && result_id.is_some_and(canonical_uuid_v4);
        if !MUTATING_OPERATIONS.contains(&record.operation.as_str())
            || request_kind != Some("request")
            || result_kind != Some("result")
            || request_op != Some(record.operation.as_str())
            || result_op != Some(record.operation.as_str())
            || request_key != Some(record.idempotency_key.as_str())
            || result_key != Some(record.idempotency_key.as_str())
            || request_id != result_id
            || !ids_ok
            || request_fp != Some(record.request_fingerprint.as_str())
            || result_fp != Some(record.request_fingerprint.as_str())
            || !record
                .request
                .get("original_result_hash")
                .is_some_and(Value::is_null)
            || result_original != Some(record.original_result_hash.as_str())
            || request_hash.as_deref() != Some(record.request_fingerprint.as_str())
            || result_hash.as_deref() != Some(record.original_result_hash.as_str())
            || !identity_ok
        {
            note(
                violations,
                "SEM-A05",
                "mutating control records must be schema-shaped replay bindings for this mission",
            );
            continue;
        }
        fingerprints_by_key
            .entry(record.idempotency_key.clone())
            .or_default()
            .insert(record.request_fingerprint.clone());
        results_by_key_fp
            .entry((
                record.idempotency_key.clone(),
                record.request_fingerprint.clone(),
            ))
            .or_default()
            .insert(record.original_result_hash.clone());
        if record.operation == "mission.cancel" {
            cancel_evidence.insert(record.evidence_ref.clone());
        }
    }
    if fingerprints_by_key.values().any(|values| values.len() > 1)
        || results_by_key_fp.values().any(|values| values.len() > 1)
    {
        note(
            violations,
            "SEM-A05",
            "idempotency key must map to one fingerprint and original result",
        );
    }
    for event in events {
        if !matches!(event.payload, LifecyclePayload::CancelRequested { .. }) {
            continue;
        }
        if event
            .source
            .evidence_ref
            .as_ref()
            .is_none_or(|evidence| !cancel_evidence.contains(evidence))
        {
            note(
                violations,
                "SEM-A05",
                "cancel_requested must name a mission.cancel control binding",
            );
        }
    }
}

fn control_identities_match(mission_id: uuid::Uuid, record: &ControlBinding) -> bool {
    let history = mission_id.to_string();
    let result_id = result_control_mission_id(&record.result);
    if result_id.as_deref() != Some(history.as_str()) {
        return false;
    }
    if record.operation == "mission.create" {
        return true;
    }
    request_control_mission_id(&record.request).as_deref() == Some(history.as_str())
}

fn canonical_uuid_v4(value: &str) -> bool {
    uuid::Uuid::parse_str(value).ok().is_some_and(|parsed| {
        parsed.get_version() == Some(uuid::Version::Random) && parsed.to_string() == value
    })
}

fn request_control_mission_id(request: &Value) -> Option<String> {
    request
        .get("body")
        .and_then(|body| body.get("mission_id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn result_control_mission_id(result: &Value) -> Option<String> {
    result
        .get("body")
        .and_then(|body| body.get("record"))
        .and_then(|record| record.get("mission_id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            result
                .get("body")
                .and_then(|body| body.get("mission_id"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn control_content_hash(value: &Value) -> Option<String> {
    let Value::Object(map) = value else {
        return None;
    };
    let mut filtered = serde_json::Map::new();
    for (key, item) in map {
        if matches!(
            key.as_str(),
            "request_id" | "request_fingerprint" | "original_result_hash"
        ) {
            continue;
        }
        filtered.insert(key.clone(), item.clone());
    }
    let blob = serde_json::to_string(&canonical_value(&Value::Object(filtered))).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(blob.as_bytes());
    Some(format!("{:x}", hasher.finalize()))
}

fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            let mut sorted = serde_json::Map::new();
            for key in keys {
                sorted.insert(key.clone(), canonical_value(&map[&key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_value).collect()),
        other => other.clone(),
    }
}

fn check_live_lease(
    lease_enabled: bool,
    running: &HashMap<uuid::Uuid, RunningLease>,
    attempt_id: Option<uuid::Uuid>,
    lease_generation: Option<u64>,
    violations: &mut Vec<InvariantError>,
) {
    if !lease_enabled {
        return;
    }
    let Some(attempt_id) = attempt_id else {
        return;
    };
    let Some(running) = running.get(&attempt_id) else {
        note(
            violations,
            "SEM-L11",
            "fenced cancel/restart requires lease_started",
        );
        return;
    };
    if lease_generation.is_some_and(|generation| generation != running.lease_generation) {
        note(
            violations,
            "SEM-L01",
            "fenced mutation lease_generation is stale",
        );
    }
}

fn note(violations: &mut Vec<InvariantError>, field: &'static str, message: &'static str) {
    violations.push(InvariantError { field, message });
}
