use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use thiserror::Error;
use uuid::{Uuid, Version};

use crate::{
    AttemptRecord, AttemptState, CapabilityFidelity, DriverCapabilityReport, DriverDescriptor,
    EventSourceKind, LifecycleEvent, LifecyclePayload, Mission, MissionPhase, MissionRecord,
    MissionTerminalReason, Principal, PrincipalKind, RecoveryRelation, DRIVER_CAPABILITIES_SCHEMA,
    LIFECYCLE_EVENT_SCHEMA, MISSION_RECORD_SCHEMA,
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{field}: {message}")]
pub struct InvariantError {
    pub field: &'static str,
    pub message: &'static str,
}

impl InvariantError {
    const fn new(field: &'static str, message: &'static str) -> Self {
        Self { field, message }
    }
}

pub trait Validate {
    fn validate(&self) -> Result<(), InvariantError>;
}

fn is_uuid_v4(value: Uuid) -> bool {
    value.get_version() == Some(Version::Random)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_principal(principal: &Principal) -> Result<(), InvariantError> {
    if !nonempty(&principal.subject) {
        return Err(InvariantError::new(
            "principal.subject",
            "must not be empty",
        ));
    }
    match principal.kind {
        PrincipalKind::AttestedSession => {
            if principal
                .role
                .as_deref()
                .is_none_or(|value| !nonempty(value))
                || principal
                    .scope
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
                || principal.attestation.is_none()
            {
                return Err(InvariantError::new(
                    "principal",
                    "attested sessions require role, scope, and attestation",
                ));
            }
            let attestation = principal.attestation.as_ref().ok_or_else(|| {
                InvariantError::new(
                    "principal.attestation",
                    "attested session requires attestation",
                )
            })?;
            if !is_uuid_v4(attestation.session_id) || !is_uuid_v4(attestation.jti) {
                return Err(InvariantError::new(
                    "principal.attestation",
                    "session_id and jti must be UUID v4",
                ));
            }
            for digest in [
                &attestation.context_sha256,
                &attestation.token_sha256,
                &attestation.verification_policy_sha256,
            ] {
                if !is_sha256(digest) {
                    return Err(InvariantError::new(
                        "principal.attestation",
                        "attestation digests must be lowercase SHA-256",
                    ));
                }
            }
        }
        PrincipalKind::Service => {
            if principal.role.is_some()
                || principal.scope.is_some()
                || principal.attestation.is_some()
            {
                return Err(InvariantError::new(
                    "principal",
                    "service principals cannot carry session authority",
                ));
            }
        }
    }
    Ok(())
}

fn validate_attempt(attempt: &AttemptRecord) -> Result<(), InvariantError> {
    if !is_uuid_v4(attempt.attempt_id) {
        return Err(InvariantError::new("attempt.attempt_id", "must be UUID v4"));
    }
    if attempt.ordinal == 0 || attempt.generation == 0 {
        return Err(InvariantError::new(
            "attempt.ordinal",
            "ordinal and generation start at one",
        ));
    }
    if !is_sha256(&attempt.fencing_token_sha256) {
        return Err(InvariantError::new(
            "attempt.fencing_token_sha256",
            "must be lowercase SHA-256",
        ));
    }
    match attempt.recovery_relation {
        crate::RecoveryRelation::Initial if attempt.predecessor_attempt_id.is_some() => {
            return Err(InvariantError::new(
                "attempt.predecessor_attempt_id",
                "initial attempt cannot have a predecessor",
            ));
        }
        crate::RecoveryRelation::Resumes | crate::RecoveryRelation::Relaunches
            if attempt.predecessor_attempt_id.is_none() =>
        {
            return Err(InvariantError::new(
                "attempt.predecessor_attempt_id",
                "successor attempt requires a predecessor",
            ));
        }
        _ => {}
    }
    if let (Some(started_at), Some(ended_at)) = (attempt.started_at, attempt.ended_at) {
        if ended_at < started_at {
            return Err(InvariantError::new(
                "attempt.ended_at",
                "cannot precede started_at",
            ));
        }
    }
    if attempt.state.is_live() {
        if attempt.ended_at.is_some() || attempt.terminal_reason.is_some() {
            return Err(InvariantError::new(
                "attempt.state",
                "live attempt cannot carry terminal fields",
            ));
        }
    } else if attempt.ended_at.is_none() || attempt.terminal_reason.is_none() {
        return Err(InvariantError::new(
            "attempt.state",
            "terminal attempt requires ended_at and terminal_reason",
        ));
    }
    Ok(())
}

impl Validate for AttemptRecord {
    fn validate(&self) -> Result<(), InvariantError> {
        validate_attempt(self)
    }
}

impl Validate for MissionRecord {
    fn validate(&self) -> Result<(), InvariantError> {
        if self.mission_schema != MISSION_RECORD_SCHEMA
            && self.mission_schema != crate::MISSION_RECORD_SCHEMA_V0
        {
            return Err(InvariantError::new(
                "mission_schema",
                "unsupported schema identifier",
            ));
        }
        if !is_uuid_v4(self.mission_id) || !is_uuid_v4(self.evidence_chain_id) {
            return Err(InvariantError::new(
                "mission_id",
                "mission and evidence chain ids must be UUID v4",
            ));
        }
        if self.mission_id != self.evidence_chain_id {
            return Err(InvariantError::new(
                "evidence_chain_id",
                "must equal mission_id",
            ));
        }
        if !nonempty(&self.goal) || !nonempty(&self.policy_id) {
            return Err(InvariantError::new(
                "goal",
                "goal and policy_id must not be empty",
            ));
        }
        if self.updated_at < self.created_at
            || self
                .deadline_at
                .is_some_and(|value| value < self.created_at)
        {
            return Err(InvariantError::new(
                "updated_at",
                "mission timestamps cannot move backward",
            ));
        }
        validate_principal(&self.initiator)?;
        validate_principal(&self.supervisor)?;
        if let Some(authorizer) = &self.authorizer {
            validate_principal(authorizer)?;
            if authorizer.kind != PrincipalKind::AttestedSession {
                return Err(InvariantError::new(
                    "authorizer.kind",
                    "mission authorization requires an attested session",
                ));
            }
        }
        if self.authorizer.is_some() != self.authorization_ref.is_some() {
            return Err(InvariantError::new(
                "authorization_ref",
                "authorizer and authorization_ref must appear together",
            ));
        }
        if !nonempty(&self.operating_role.role) || !nonempty(&self.operating_role.scope) {
            return Err(InvariantError::new(
                "operating_role",
                "role and scope must not be empty",
            ));
        }
        if self.lease_policy.enabled {
            if self
                .lease_policy
                .lease_seconds
                .is_none_or(|value| value == 0)
                || self
                    .lease_policy
                    .deadman_seconds
                    .is_none_or(|value| value == 0)
            {
                return Err(InvariantError::new(
                    "lease_policy",
                    "enabled policy requires positive lease and deadman values",
                ));
            }
        } else if self.lease_policy.lease_seconds.is_some()
            || self.lease_policy.deadman_seconds.is_some()
        {
            return Err(InvariantError::new(
                "lease_policy",
                "disabled policy cannot carry timing values",
            ));
        }

        let mut attempt_ids = HashSet::new();
        let mut previous_generation = None;
        let mut live_attempts = Vec::new();
        for (index, attempt) in self.attempts.iter().enumerate() {
            validate_attempt(attempt)?;
            let expected_ordinal = u32::try_from(index + 1).map_err(|_| {
                InvariantError::new("attempt.ordinal", "attempt collection is too large")
            })?;
            if attempt.ordinal != expected_ordinal {
                return Err(InvariantError::new(
                    "attempt.ordinal",
                    "attempt ordinals must be contiguous",
                ));
            }
            if !attempt_ids.insert(attempt.attempt_id) {
                return Err(InvariantError::new("attempt", "attempt ids must be unique"));
            }
            if previous_generation.is_some_and(|previous| attempt.generation <= previous) {
                return Err(InvariantError::new(
                    "attempt.generation",
                    "attempt generations must be unique and increasing",
                ));
            }
            previous_generation = Some(attempt.generation);
            if let Some(predecessor_id) = attempt.predecessor_attempt_id {
                let predecessor = self.attempts[..index]
                    .iter()
                    .find(|candidate| candidate.attempt_id == predecessor_id)
                    .ok_or_else(|| {
                        InvariantError::new(
                            "attempt.predecessor_attempt_id",
                            "must name an earlier attempt",
                        )
                    })?;
                if predecessor.state != AttemptState::Replaced {
                    return Err(InvariantError::new(
                        "attempt.predecessor_attempt_id",
                        "predecessor must be replaced before successor creation",
                    ));
                }
            }
            if attempt.state.is_live() {
                live_attempts.push(attempt.attempt_id);
            }
        }
        if live_attempts.len() > 1 {
            return Err(InvariantError::new(
                "attempts",
                "at most one attempt may be live",
            ));
        }
        if self.current_attempt_id != live_attempts.first().copied() {
            return Err(InvariantError::new(
                "current_attempt_id",
                "must identify the sole live attempt",
            ));
        }

        if self.phase == MissionPhase::Created
            && (self.revision != 0
                || !self.attempts.is_empty()
                || self.authorizer.is_some()
                || self.authorization_ref.is_some()
                || self.current_attempt_id.is_some()
                || self.terminal_reason.is_some()
                || self.terminal_entry_hash.is_some())
        {
            return Err(InvariantError::new(
                "phase",
                "created mission must be revision zero with no authority, attempts, or terminal fields",
            ));
        }
        if matches!(self.phase, MissionPhase::Active | MissionPhase::Waiting)
            && (self.current_attempt_id.is_none() || self.authorizer.is_none())
        {
            return Err(InvariantError::new(
                "phase",
                "active or waiting mission requires authority and a live attempt",
            ));
        }
        if self.phase.is_terminal() {
            if self.current_attempt_id.is_some()
                || self.terminal_reason.is_none()
                || self
                    .terminal_entry_hash
                    .as_deref()
                    .is_none_or(|value| !is_sha256(value))
            {
                return Err(InvariantError::new(
                    "phase",
                    "terminal mission requires reason and terminal hash with no live attempt",
                ));
            }
            if !terminal_reason_matches(self.phase, self.terminal_reason) {
                return Err(InvariantError::new(
                    "terminal_reason",
                    "does not match terminal mission phase",
                ));
            }
        } else if self.terminal_reason.is_some() || self.terminal_entry_hash.is_some() {
            return Err(InvariantError::new(
                "terminal_reason",
                "non-terminal mission cannot carry terminal fields",
            ));
        }
        Ok(())
    }
}

impl TryFrom<MissionRecord> for Mission {
    type Error = InvariantError;

    fn try_from(record: MissionRecord) -> Result<Self, Self::Error> {
        record.validate()?;
        Ok(Self { record })
    }
}

fn terminal_reason_matches(phase: MissionPhase, reason: Option<MissionTerminalReason>) -> bool {
    matches!(
        (phase, reason),
        (
            MissionPhase::Completed,
            Some(MissionTerminalReason::GoalSatisfied)
        ) | (
            MissionPhase::Cancelled,
            Some(MissionTerminalReason::OperatorCancelled)
        ) | (
            MissionPhase::DeadlineExceeded,
            Some(MissionTerminalReason::MissionDeadlineExceeded)
        ) | (
            MissionPhase::BudgetExhausted,
            Some(MissionTerminalReason::BudgetExhausted)
        ) | (
            MissionPhase::Failed,
            Some(
                MissionTerminalReason::PolicyDenied
                    | MissionTerminalReason::RestoreExhausted
                    | MissionTerminalReason::RequiredDriverCapabilityUnavailable
                    | MissionTerminalReason::InternalError
            )
        )
    )
}

impl Validate for DriverCapabilityReport {
    fn validate(&self) -> Result<(), InvariantError> {
        if self.capabilities_schema != DRIVER_CAPABILITIES_SCHEMA {
            return Err(InvariantError::new(
                "capabilities_schema",
                "unsupported schema identifier",
            ));
        }
        if !is_uuid_v4(self.report_id) || self.expires_at <= self.observed_at {
            return Err(InvariantError::new(
                "report_id",
                "report id must be UUID v4 and expiry must follow observation",
            ));
        }
        if !nonempty(&self.driver_id)
            || !nonempty(&self.driver_version)
            || !nonempty(&self.harness_kind)
            || !nonempty(&self.evidence_ref)
            || self.capabilities.is_empty()
        {
            return Err(InvariantError::new(
                "driver_capabilities",
                "driver identity, evidence, and at least one capability are required",
            ));
        }
        if self.validity_condition.kind != "executable-version-platform-match"
            || self.validity_condition.executable_version != self.driver_version
            || !nonempty(&self.validity_condition.platform)
            || !nonempty(&self.validity_condition.probe_ref)
        {
            return Err(InvariantError::new(
                "validity_condition",
                "must bind the observed driver version to a platform probe",
            ));
        }
        let mut names = HashSet::new();
        for capability in &self.capabilities {
            if !names.insert(capability.name) {
                return Err(InvariantError::new(
                    "capabilities",
                    "capability names must be unique",
                ));
            }
            if capability.fidelity == CapabilityFidelity::Unsupported
                && capability
                    .limitation
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
            {
                return Err(InvariantError::new(
                    "capability.limitation",
                    "unsupported capability requires a limitation",
                ));
            }
            if capability
                .limitation
                .as_deref()
                .is_some_and(|value| !nonempty(value))
                || capability
                    .evidence_ref
                    .as_deref()
                    .is_some_and(|value| !nonempty(value))
            {
                return Err(InvariantError::new(
                    "capability",
                    "optional limitation and evidence values cannot be empty",
                ));
            }
        }
        for digest in [
            &self.validity_condition.executable_sha256,
            &self.validity_condition.configuration_sha256,
        ] {
            if !is_sha256(digest) {
                return Err(InvariantError::new(
                    "validity_condition",
                    "executable and configuration digests must be lowercase SHA-256",
                ));
            }
        }
        Ok(())
    }
}

impl DriverCapabilityReport {
    pub fn require_usable_at(
        &self,
        now: DateTime<Utc>,
        descriptor: &DriverDescriptor,
        platform: &str,
        required: &[crate::CapabilityName],
    ) -> Result<(), InvariantError> {
        self.validate()?;
        if now < self.observed_at || now >= self.expires_at {
            return Err(InvariantError::new(
                "expires_at",
                "capability evidence is not current at the observation time",
            ));
        }
        if self.driver_id != descriptor.driver_id
            || self.driver_version != descriptor.driver_version
            || self.harness_kind != descriptor.harness_kind
            || self.validity_condition.executable_version != descriptor.driver_version
            || self.validity_condition.platform != platform
        {
            return Err(InvariantError::new(
                "validity_condition",
                "capability evidence does not match the selected driver and platform",
            ));
        }
        if required
            .iter()
            .any(|capability| !self.advertises_usable(*capability))
        {
            return Err(InvariantError::new(
                "capabilities",
                "required capability is unavailable, lossy, or unsupported",
            ));
        }
        Ok(())
    }

    pub fn require_recovery_usable_at(
        &self,
        now: DateTime<Utc>,
        descriptor: &DriverDescriptor,
        platform: &str,
        relation: RecoveryRelation,
    ) -> Result<(), InvariantError> {
        use crate::CapabilityName::{Cancel, Identify, Resume, TerminalStatus};

        match relation {
            RecoveryRelation::Initial => Err(InvariantError::new(
                "recovery_relation",
                "initial attempt is not a recovery operation",
            )),
            RecoveryRelation::Resumes => {
                self.require_usable_at(now, descriptor, platform, &[Resume])
            }
            RecoveryRelation::Relaunches => self.require_usable_at(
                now,
                descriptor,
                platform,
                &[Identify, Cancel, TerminalStatus],
            ),
        }
    }
}

impl Validate for LifecycleEvent {
    fn validate(&self) -> Result<(), InvariantError> {
        if self.event_schema != LIFECYCLE_EVENT_SCHEMA {
            return Err(InvariantError::new(
                "event_schema",
                "unsupported schema identifier",
            ));
        }
        if !is_uuid_v4(self.event_id) || !is_uuid_v4(self.mission_id) || self.sequence == 0 {
            return Err(InvariantError::new(
                "event_id",
                "event and mission ids must be UUID v4 and sequence starts at one",
            ));
        }
        if !is_sha256(&self.entry_hash)
            || self
                .previous_entry_hash
                .as_deref()
                .is_some_and(|value| !is_sha256(value))
            || self.recorded_at < self.occurred_at
        {
            return Err(InvariantError::new(
                "entry_hash",
                "event hashes and timestamps must be canonical",
            ));
        }
        if (self.sequence == 1) != self.previous_entry_hash.is_none() {
            return Err(InvariantError::new(
                "previous_entry_hash",
                "only the first event may omit the previous hash",
            ));
        }
        if self.event_type != self.payload.event_type() {
            return Err(InvariantError::new("event_type", "must match payload type"));
        }
        if !nonempty(&self.source.subject) || !nonempty(&self.source.producer_version) {
            return Err(InvariantError::new(
                "source",
                "subject and producer version must not be empty",
            ));
        }
        if self
            .source
            .evidence_ref
            .as_deref()
            .is_some_and(|value| !nonempty(value))
        {
            return Err(InvariantError::new(
                "source.evidence_ref",
                "evidence reference cannot be empty",
            ));
        }
        if matches!(
            self.source.kind,
            EventSourceKind::VerifiedAttestation | EventSourceKind::OperatorCommand
        ) && self
            .source
            .evidence_ref
            .as_deref()
            .is_none_or(|value| !nonempty(value))
        {
            return Err(InvariantError::new(
                "source.evidence_ref",
                "authoritative source requires evidence",
            ));
        }
        validate_lifecycle_payload(&self.payload)?;
        Ok(())
    }
}

pub fn validate_history(
    mission: &MissionRecord,
    events: &[LifecycleEvent],
) -> Result<(), InvariantError> {
    mission.validate()?;
    if !matches!(
        events.first().map(|event| &event.payload),
        Some(LifecyclePayload::MissionCreated { .. })
    ) {
        return Err(InvariantError::new(
            "events",
            "mission history must begin with mission_created",
        ));
    }
    let mut previous_hash = None;
    let mut previous_occurred_at = None;
    let mut previous_recorded_at = None;
    let mut phase = MissionPhase::Created;
    let mut authorizer_subject = None;
    let mut attempts = HashMap::new();
    let mut latest_generation = None;
    for (index, event) in events.iter().enumerate() {
        event.validate()?;
        if event.mission_id != mission.mission_id {
            return Err(InvariantError::new(
                "event.mission_id",
                "event must belong to the mission",
            ));
        }
        let expected_sequence = u64::try_from(index + 1)
            .map_err(|_| InvariantError::new("event.sequence", "event history is too large"))?;
        if event.sequence != expected_sequence || event.previous_entry_hash != previous_hash {
            return Err(InvariantError::new(
                "event.sequence",
                "event sequence and previous hash must be contiguous",
            ));
        }
        if previous_occurred_at.is_some_and(|value| event.occurred_at < value)
            || previous_recorded_at.is_some_and(|value| event.recorded_at < value)
        {
            return Err(InvariantError::new(
                "event.occurred_at",
                "event history timestamps cannot move backward",
            ));
        }
        match &event.payload {
            LifecyclePayload::MissionCreated { .. } if index != 0 => {
                return Err(InvariantError::new(
                    "events",
                    "mission_created may appear only as the first event",
                ));
            }
            LifecyclePayload::MissionCreated { .. } => {}
            LifecyclePayload::AuthorizationBound { authorizer } => {
                if !matches!(
                    event.source.kind,
                    EventSourceKind::VerifiedAttestation | EventSourceKind::OperatorCommand
                ) || authorizer_subject.is_some()
                {
                    return Err(InvariantError::new(
                        "payload.authorizer",
                        "authorization must be bound once by an authoritative event",
                    ));
                }
                authorizer_subject = Some(authorizer.subject.as_str());
            }
            LifecyclePayload::MissionPhaseChanged { from, to, .. } => {
                if *from != phase || (*to == MissionPhase::Active && authorizer_subject.is_none()) {
                    return Err(InvariantError::new(
                        "payload.from",
                        "mission phase event does not follow authoritative history",
                    ));
                }
                phase = *to;
            }
            LifecyclePayload::AttemptCreated {
                attempt_id,
                generation,
                predecessor_attempt_id,
                ..
            } => {
                if attempts.contains_key(attempt_id)
                    || latest_generation.is_some_and(|value| *generation <= value)
                    || predecessor_attempt_id.is_some_and(|predecessor_id| {
                        attempts.get(&predecessor_id).map(|(_, state)| *state)
                            != Some(AttemptState::Replaced)
                    })
                {
                    return Err(InvariantError::new(
                        "payload.generation",
                        "attempt creation must advance the generation after a replaced predecessor",
                    ));
                }
                attempts.insert(*attempt_id, (*generation, AttemptState::Starting));
                latest_generation = Some(*generation);
            }
            LifecyclePayload::AttemptStateChanged {
                attempt_id,
                generation,
                from,
                to,
                ..
            } => {
                let Some((attempt_generation, state)) = attempts.get_mut(attempt_id) else {
                    return Err(InvariantError::new(
                        "payload.attempt_id",
                        "attempt state event requires a prior attempt creation",
                    ));
                };
                if *attempt_generation != *generation
                    || *state != *from
                    || latest_generation.is_some_and(|value| *generation < value)
                    || (*to == AttemptState::Running && authorizer_subject.is_none())
                {
                    return Err(InvariantError::new(
                        "payload.generation",
                        "attempt state event must use the current generation and folded state",
                    ));
                }
                *state = *to;
                latest_generation = Some(*generation);
            }
            LifecyclePayload::DriverCapabilityEvaluated {
                attempt_id,
                generation,
                ..
            } => {
                if attempts
                    .get(attempt_id)
                    .is_none_or(|(attempt_generation, _)| attempt_generation != generation)
                    || latest_generation.is_some_and(|value| *generation < value)
                {
                    return Err(InvariantError::new(
                        "payload.generation",
                        "capability event must use the current attempt generation",
                    ));
                }
                latest_generation = Some(*generation);
            }
            LifecyclePayload::MissionTerminal {
                phase: terminal_phase,
                ..
            } if *terminal_phase != phase => {
                return Err(InvariantError::new(
                    "payload.phase",
                    "terminal event must match the folded mission phase",
                ));
            }
            LifecyclePayload::RecoveryRequested { .. }
            | LifecyclePayload::RecoveryPointRecorded { .. }
            | LifecyclePayload::MissionTerminal { .. } => {}
        }
        previous_hash = Some(event.entry_hash.clone());
        previous_occurred_at = Some(event.occurred_at);
        previous_recorded_at = Some(event.recorded_at);
    }
    if phase != mission.phase
        || authorizer_subject
            != mission
                .authorizer
                .as_ref()
                .map(|authorizer| authorizer.subject.as_str())
    {
        return Err(InvariantError::new(
            "events",
            "folded authority and mission phase must match the projection",
        ));
    }
    if attempts.len() != mission.attempts.len()
        || mission.attempts.iter().any(|attempt| {
            attempts.get(&attempt.attempt_id) != Some(&(attempt.generation, attempt.state))
        })
    {
        return Err(InvariantError::new(
            "events",
            "folded attempt generations and states must match the projection",
        ));
    }
    if let Some(terminal_hash) = &mission.terminal_entry_hash {
        if previous_hash.as_ref() != Some(terminal_hash) {
            return Err(InvariantError::new(
                "terminal_entry_hash",
                "terminal mission hash must equal the final lifecycle entry hash",
            ));
        }
        match events.last().map(|event| &event.payload) {
            Some(LifecyclePayload::MissionTerminal {
                phase,
                reason,
                terminal_entry_hash,
            }) if *phase == mission.phase
                && Some(*reason) == mission.terminal_reason
                && terminal_entry_hash == terminal_hash => {}
            _ => {
                return Err(InvariantError::new(
                    "events",
                    "terminal mission history must end with its matching terminal event",
                ));
            }
        }
    } else if events
        .iter()
        .any(|event| matches!(&event.payload, LifecyclePayload::MissionTerminal { .. }))
    {
        return Err(InvariantError::new(
            "events",
            "non-terminal mission cannot contain a terminal event",
        ));
    }
    Ok(())
}

fn validate_lifecycle_payload(payload: &LifecyclePayload) -> Result<(), InvariantError> {
    match payload {
        LifecyclePayload::MissionCreated { revision } if *revision != 0 => Err(
            InvariantError::new("payload.revision", "created revision must be zero"),
        ),
        LifecyclePayload::AuthorizationBound { authorizer } => {
            if authorizer.kind != PrincipalKind::AttestedSession
                || !nonempty(&authorizer.subject)
                || !nonempty(&authorizer.attestation_ref)
            {
                Err(InvariantError::new(
                    "payload.authorizer",
                    "authorization must reference an attested session",
                ))
            } else {
                Ok(())
            }
        }
        LifecyclePayload::MissionPhaseChanged { from, to, reason } => {
            if !from.can_transition_to(*to) {
                Err(InvariantError::new(
                    "payload.to",
                    "mission phase transition is illegal",
                ))
            } else if reason.as_deref().is_some_and(|value| !nonempty(value)) {
                Err(InvariantError::new(
                    "payload.reason",
                    "transition reason cannot be empty",
                ))
            } else {
                Ok(())
            }
        }
        LifecyclePayload::AttemptCreated {
            attempt_id,
            ordinal,
            generation,
            recovery_relation,
            predecessor_attempt_id,
        } => {
            if !is_uuid_v4(*attempt_id) || *ordinal == 0 || *generation == 0 {
                return Err(InvariantError::new(
                    "payload.attempt_id",
                    "attempt id must be UUID v4 and counters start at one",
                ));
            }
            let relation_matches = matches!(
                (recovery_relation, predecessor_attempt_id),
                (RecoveryRelation::Initial, None)
                    | (
                        RecoveryRelation::Resumes | RecoveryRelation::Relaunches,
                        Some(_)
                    )
            );
            if !relation_matches || predecessor_attempt_id.is_some_and(|value| !is_uuid_v4(value)) {
                return Err(InvariantError::new(
                    "payload.predecessor_attempt_id",
                    "recovery relation and predecessor must agree",
                ));
            }
            Ok(())
        }
        LifecyclePayload::AttemptStateChanged {
            attempt_id,
            generation,
            from,
            to,
            reason,
        } => {
            if !is_uuid_v4(*attempt_id) || *generation == 0 {
                return Err(InvariantError::new(
                    "payload.attempt_id",
                    "attempt id must be UUID v4 and generation starts at one",
                ));
            }
            if !from.can_transition_to(*to) {
                return Err(InvariantError::new(
                    "payload.to",
                    "attempt state transition is illegal",
                ));
            }
            if reason.as_deref().is_some_and(|value| !nonempty(value)) {
                return Err(InvariantError::new(
                    "payload.reason",
                    "transition reason cannot be empty",
                ));
            }
            Ok(())
        }
        LifecyclePayload::DriverCapabilityEvaluated {
            attempt_id,
            generation,
            driver_id,
            report_id,
            ..
        } => {
            if !is_uuid_v4(*attempt_id)
                || !is_uuid_v4(*report_id)
                || *generation == 0
                || !nonempty(driver_id)
            {
                return Err(InvariantError::new(
                    "payload.driver_capability_evaluated",
                    "attempt, report, generation, and driver identity must be canonical",
                ));
            }
            Ok(())
        }
        LifecyclePayload::RecoveryRequested {
            predecessor_attempt_id,
            relation,
        } => {
            if !is_uuid_v4(*predecessor_attempt_id) || *relation == RecoveryRelation::Initial {
                return Err(InvariantError::new(
                    "payload.relation",
                    "recovery requires a predecessor and resume or relaunch relation",
                ));
            }
            Ok(())
        }
        LifecyclePayload::RecoveryPointRecorded { recovery_point_ref } => {
            if !nonempty(recovery_point_ref) {
                return Err(InvariantError::new(
                    "payload.recovery_point_ref",
                    "recovery point reference cannot be empty",
                ));
            }
            Ok(())
        }
        LifecyclePayload::MissionTerminal {
            phase,
            reason,
            terminal_entry_hash,
        } => {
            if !phase.is_terminal()
                || !terminal_reason_matches(*phase, Some(*reason))
                || !is_sha256(terminal_entry_hash)
            {
                return Err(InvariantError::new(
                    "payload.mission_terminal",
                    "terminal phase, reason, and hash must agree",
                ));
            }
            Ok(())
        }
        LifecyclePayload::MissionCreated { .. } => Ok(()),
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use chrono::{TimeZone, Utc};
    use uuid::Uuid;

    use crate::{
        AttemptRecord, AttemptState, AttestationRef, BudgetPolicy, LeasePolicy, MissionPhase,
        MissionRecord, OperatingRole, Principal, PrincipalKind, RecoveryPolicy, RecoveryRelation,
        MISSION_RECORD_SCHEMA,
    };

    pub fn uuid(value: u128) -> Uuid {
        Uuid::from_u128(value | (4 << 76) | (8 << 60))
    }

    pub fn created_mission() -> MissionRecord {
        MissionRecord {
            mission_schema: MISSION_RECORD_SCHEMA.to_owned(),
            mission_id: uuid(1),
            revision: 0,
            goal: "Build a domain contract".to_owned(),
            policy_id: "policy.default".to_owned(),
            created_at: Utc.with_ymd_and_hms(2026, 8, 17, 14, 0, 0).unwrap(),
            updated_at: Utc.with_ymd_and_hms(2026, 8, 17, 14, 0, 0).unwrap(),
            initiator: Principal {
                kind: PrincipalKind::AttestedSession,
                subject: "operator-1".to_owned(),
                role: Some("entarch".to_owned()),
                scope: Some("lanytehq".to_owned()),
                attestation: Some(AttestationRef {
                    issuer: "lanyte-attest".to_owned(),
                    session_id: uuid(2),
                    jti: uuid(3),
                    context_sha256: "b".repeat(64),
                    token_sha256: "c".repeat(64),
                    verification_policy_sha256: "d".repeat(64),
                    trust_ref: "attestations/3".to_owned(),
                }),
            },
            authorizer: None,
            authorization_ref: None,
            supervisor: Principal {
                kind: PrincipalKind::Service,
                subject: "lanyte-kernel".to_owned(),
                role: None,
                scope: None,
                attestation: None,
            },
            operating_role: OperatingRole {
                role: "entarch".to_owned(),
                scope: "lanytehq".to_owned(),
            },
            phase: MissionPhase::Created,
            terminal_reason: None,
            deadline_at: None,
            lease_policy: LeasePolicy {
                enabled: false,
                lease_seconds: None,
                deadman_seconds: None,
            },
            budget_policy: BudgetPolicy {
                wall_clock_seconds: None,
                token_limit: None,
                cost_micros: None,
                action_limit: None,
            },
            harness_selection: None,
            recovery_policy: RecoveryPolicy::AskOperator,
            recovery_point_ref: None,
            attempts: Vec::new(),
            current_attempt_id: None,
            evidence_chain_id: uuid(1),
            terminal_entry_hash: None,
        }
    }

    pub fn running_attempt(id: u128, ordinal: u32, generation: u64) -> AttemptRecord {
        AttemptRecord {
            attempt_id: uuid(id),
            ordinal,
            generation,
            fencing_token_sha256: "a".repeat(64),
            recovery_relation: RecoveryRelation::Initial,
            predecessor_attempt_id: None,
            state: AttemptState::Running,
            driver_id: Some("driver.local".to_owned()),
            harness_session_id: Some(format!("session-{ordinal}")),
            started_at: Some(Utc.with_ymd_and_hms(2026, 8, 17, 14, 1, 0).unwrap()),
            ended_at: None,
            terminal_reason: None,
            evidence_ref: None,
            lease_expires_at: None,
            deadman_at: None,
            last_observed_at: None,
            last_observation_source: None,
            lease_generation: None,
            process_tree_ref: None,
            ownership_established_at: None,
            harness_thread_id: None,
            harness_turn_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::fixtures::created_mission;
    use super::*;
    use crate::{
        CapabilityName, DriverAvailability, DriverCapability, DriverValidityCondition,
        EnforcementLevel, EventSource, LifecyclePayload, ObservationLevel, PrincipalRef,
        ReplaySupport, DRIVER_CAPABILITIES_SCHEMA, LIFECYCLE_EVENT_SCHEMA,
    };

    fn event(
        event_id: u128,
        mission_id: Uuid,
        sequence: u64,
        previous_entry_hash: Option<String>,
        event_type: &str,
        payload: LifecyclePayload,
    ) -> LifecycleEvent {
        LifecycleEvent {
            event_schema: LIFECYCLE_EVENT_SCHEMA.to_owned(),
            event_id: fixtures::uuid(event_id),
            mission_id,
            sequence,
            previous_entry_hash,
            entry_hash: format!("{sequence:064x}"),
            occurred_at: Utc
                .with_ymd_and_hms(2026, 8, 17, 14, sequence as u32, 0)
                .unwrap(),
            recorded_at: Utc
                .with_ymd_and_hms(2026, 8, 17, 14, sequence as u32, 1)
                .unwrap(),
            event_type: event_type.to_owned(),
            source: EventSource {
                kind: EventSourceKind::DriverReported,
                subject: "driver.test".to_owned(),
                producer_version: "0.1.0".to_owned(),
                assurance: ObservationLevel::DriverObserved,
                evidence_ref: None,
            },
            payload,
        }
    }

    #[test]
    fn created_mission_is_valid() {
        created_mission().validate().unwrap();
    }

    #[test]
    fn evidence_chain_must_match_mission() {
        let mut mission = created_mission();
        mission.evidence_chain_id = fixtures::uuid(9);
        assert_eq!(mission.validate().unwrap_err().field, "evidence_chain_id");
    }

    #[test]
    fn service_principal_cannot_carry_role() {
        let mut mission = created_mission();
        mission.supervisor.role = Some("entarch".to_owned());
        assert_eq!(mission.validate().unwrap_err().field, "principal");
    }

    #[test]
    fn created_phase_has_no_authorizer() {
        let mut mission = created_mission();
        mission.authorizer = Some(mission.initiator.clone());
        mission.authorization_ref = Some("authorizations/1".to_owned());
        assert_eq!(mission.validate().unwrap_err().field, "phase");
    }

    #[test]
    fn two_live_attempts_are_rejected() {
        let mut mission = created_mission();
        mission.phase = MissionPhase::RecoveryPending;
        mission.revision = 2;
        mission.attempts = vec![
            fixtures::running_attempt(8, 1, 1),
            fixtures::running_attempt(9, 2, 2),
        ];
        mission.current_attempt_id = Some(mission.attempts[1].attempt_id);
        assert_eq!(mission.validate().unwrap_err().field, "attempts");
    }

    #[test]
    fn attempt_generations_must_increase() {
        let mut mission = created_mission();
        mission.phase = MissionPhase::RecoveryPending;
        mission.revision = 2;
        mission.attempts = vec![
            fixtures::running_attempt(8, 1, 2),
            fixtures::running_attempt(9, 2, 1),
        ];
        mission.current_attempt_id = Some(mission.attempts[1].attempt_id);
        assert_eq!(mission.validate().unwrap_err().field, "attempt.generation");
    }

    #[test]
    fn service_principal_cannot_authorize() {
        let mut mission = created_mission();
        mission.phase = MissionPhase::Suspended;
        mission.revision = 1;
        mission.authorizer = Some(mission.supervisor.clone());
        mission.authorization_ref = Some("authorizations/1".to_owned());
        assert_eq!(mission.validate().unwrap_err().field, "authorizer.kind");
    }

    #[test]
    fn non_v4_mission_id_is_rejected() {
        let mut mission = created_mission();
        mission.mission_id = Uuid::nil();
        mission.evidence_chain_id = Uuid::nil();
        assert_eq!(mission.validate().unwrap_err().field, "mission_id");
    }

    #[test]
    fn unknown_record_fields_are_rejected_on_deserialize() {
        let mut value = serde_json::to_value(created_mission()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("ambient_role".to_owned(), serde_json::json!("entarch"));
        assert!(serde_json::from_value::<MissionRecord>(value).is_err());
    }

    #[test]
    fn checked_aggregate_rejects_invalid_wire_record() {
        let mut record = created_mission();
        record.goal.clear();
        assert!(Mission::try_from(record).is_err());
    }

    #[test]
    fn lossy_required_capability_fails_closed() {
        let mut report = DriverCapabilityReport {
            capabilities_schema: DRIVER_CAPABILITIES_SCHEMA.to_owned(),
            report_id: fixtures::uuid(6),
            driver_id: "driver.test".to_owned(),
            driver_version: "0.1.0".to_owned(),
            harness_kind: "test".to_owned(),
            observed_at: Utc.with_ymd_and_hms(2026, 8, 17, 14, 0, 0).unwrap(),
            expires_at: Utc.with_ymd_and_hms(2026, 8, 18, 14, 0, 0).unwrap(),
            availability: DriverAvailability::Available,
            capabilities: vec![DriverCapability {
                name: CapabilityName::Resume,
                fidelity: CapabilityFidelity::Lossy,
                observation: ObservationLevel::DriverObserved,
                enforcement: EnforcementLevel::RequestOnly,
                replay: ReplaySupport::Idempotent,
                limitation: Some("relaunches rather than resuming".to_owned()),
                evidence_ref: Some("probes/resume".to_owned()),
            }],
            validity_condition: DriverValidityCondition {
                kind: "executable-version-platform-match".to_owned(),
                executable_version: "0.1.0".to_owned(),
                executable_sha256: "e".repeat(64),
                configuration_sha256: "f".repeat(64),
                platform: "test-platform".to_owned(),
                probe_ref: "probes/test-driver".to_owned(),
            },
            evidence_ref: "reports/test-driver".to_owned(),
        };
        let descriptor = DriverDescriptor {
            driver_id: "driver.test".to_owned(),
            driver_version: "0.1.0".to_owned(),
            harness_kind: "test".to_owned(),
        };
        assert_eq!(
            report
                .require_usable_at(
                    Utc.with_ymd_and_hms(2026, 8, 17, 15, 0, 0).unwrap(),
                    &descriptor,
                    "test-platform",
                    &[CapabilityName::Resume],
                )
                .unwrap_err()
                .field,
            "capabilities"
        );
        report.capabilities[0].fidelity = CapabilityFidelity::Native;
        report
            .require_recovery_usable_at(
                Utc.with_ymd_and_hms(2026, 8, 17, 15, 0, 0).unwrap(),
                &descriptor,
                "test-platform",
                RecoveryRelation::Resumes,
            )
            .unwrap();
        assert_eq!(
            report
                .require_recovery_usable_at(
                    Utc.with_ymd_and_hms(2026, 8, 17, 15, 0, 0).unwrap(),
                    &descriptor,
                    "test-platform",
                    RecoveryRelation::Relaunches,
                )
                .unwrap_err()
                .field,
            "capabilities"
        );
        assert_eq!(
            report
                .require_usable_at(
                    Utc.with_ymd_and_hms(2026, 8, 18, 14, 0, 0).unwrap(),
                    &descriptor,
                    "test-platform",
                    &[CapabilityName::Resume],
                )
                .unwrap_err()
                .field,
            "expires_at"
        );
    }

    #[test]
    fn lifecycle_payload_rejects_illegal_transition() {
        let event = LifecycleEvent {
            event_schema: LIFECYCLE_EVENT_SCHEMA.to_owned(),
            event_id: fixtures::uuid(20),
            mission_id: fixtures::uuid(1),
            sequence: 1,
            previous_entry_hash: None,
            entry_hash: "1".repeat(64),
            occurred_at: Utc.with_ymd_and_hms(2026, 8, 17, 14, 0, 0).unwrap(),
            recorded_at: Utc.with_ymd_and_hms(2026, 8, 17, 14, 0, 1).unwrap(),
            event_type: "mission_phase_changed".to_owned(),
            source: EventSource {
                kind: EventSourceKind::DriverReported,
                subject: "driver.test".to_owned(),
                producer_version: "0.1.0".to_owned(),
                assurance: ObservationLevel::DriverObserved,
                evidence_ref: None,
            },
            payload: LifecyclePayload::MissionPhaseChanged {
                from: MissionPhase::Created,
                to: MissionPhase::Completed,
                reason: None,
            },
        };
        assert_eq!(event.validate().unwrap_err().field, "payload.to");
    }

    #[test]
    fn lifecycle_history_requires_contiguous_hash_links() {
        let mission = created_mission();
        let created = LifecycleEvent {
            event_schema: LIFECYCLE_EVENT_SCHEMA.to_owned(),
            event_id: fixtures::uuid(20),
            mission_id: mission.mission_id,
            sequence: 1,
            previous_entry_hash: None,
            entry_hash: "1".repeat(64),
            occurred_at: mission.created_at,
            recorded_at: mission.created_at,
            event_type: "mission_created".to_owned(),
            source: EventSource {
                kind: EventSourceKind::VerifiedAttestation,
                subject: "operator-1".to_owned(),
                producer_version: "0.1.0".to_owned(),
                assurance: ObservationLevel::ResourceAttested,
                evidence_ref: Some("attestations/1".to_owned()),
            },
            payload: LifecyclePayload::MissionCreated { revision: 0 },
        };
        let mut authorization = created.clone();
        authorization.event_id = fixtures::uuid(21);
        authorization.sequence = 2;
        authorization.previous_entry_hash = Some("9".repeat(64));
        authorization.entry_hash = "2".repeat(64);
        authorization.event_type = "authorization_bound".to_owned();
        authorization.payload = LifecyclePayload::AuthorizationBound {
            authorizer: PrincipalRef {
                kind: PrincipalKind::AttestedSession,
                subject: "operator-1".to_owned(),
                attestation_ref: "attestations/1".to_owned(),
            },
        };
        assert_eq!(
            validate_history(&mission, &[created, authorization])
                .unwrap_err()
                .field,
            "event.sequence"
        );
    }

    #[test]
    fn lifecycle_history_rejects_claimed_authorization() {
        let mut mission = created_mission();
        mission.revision = 1;
        mission.phase = MissionPhase::Suspended;
        mission.authorizer = Some(mission.initiator.clone());
        mission.authorization_ref = Some("authorizations/1".to_owned());

        let mut created = event(
            30,
            mission.mission_id,
            1,
            None,
            "mission_created",
            LifecyclePayload::MissionCreated { revision: 0 },
        );
        created.source.kind = EventSourceKind::VerifiedAttestation;
        created.source.assurance = ObservationLevel::ResourceAttested;
        created.source.evidence_ref = Some("attestations/created".to_owned());
        let authorization = event(
            31,
            mission.mission_id,
            2,
            Some(created.entry_hash.clone()),
            "authorization_bound",
            LifecyclePayload::AuthorizationBound {
                authorizer: PrincipalRef {
                    kind: PrincipalKind::AttestedSession,
                    subject: mission.initiator.subject.clone(),
                    attestation_ref: "attestations/1".to_owned(),
                },
            },
        );
        assert_eq!(
            validate_history(&mission, &[created, authorization])
                .unwrap_err()
                .field,
            "payload.authorizer"
        );
    }

    #[test]
    fn lifecycle_history_rejects_stale_event_generation() {
        let mission = created_mission();
        let created = event(
            40,
            mission.mission_id,
            1,
            None,
            "mission_created",
            LifecyclePayload::MissionCreated { revision: 0 },
        );
        let attempt_created = event(
            41,
            mission.mission_id,
            2,
            Some(created.entry_hash.clone()),
            "attempt_created",
            LifecyclePayload::AttemptCreated {
                attempt_id: fixtures::uuid(50),
                ordinal: 1,
                generation: 2,
                recovery_relation: RecoveryRelation::Initial,
                predecessor_attempt_id: None,
            },
        );
        let capability = event(
            42,
            mission.mission_id,
            3,
            Some(attempt_created.entry_hash.clone()),
            "driver_capability_evaluated",
            LifecyclePayload::DriverCapabilityEvaluated {
                attempt_id: fixtures::uuid(50),
                generation: 1,
                driver_id: "driver.test".to_owned(),
                capability: CapabilityName::Create,
                availability: DriverAvailability::Available,
                fidelity: CapabilityFidelity::Native,
                report_id: fixtures::uuid(51),
            },
        );
        assert_eq!(
            validate_history(&mission, &[created, attempt_created, capability])
                .unwrap_err()
                .field,
            "payload.generation"
        );
    }
}
