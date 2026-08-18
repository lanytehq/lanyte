use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const MISSION_RECORD_SCHEMA: &str =
    "https://schemas.3leaps.dev/agentic/mission/v0/mission-record.schema.json";
pub const DRIVER_CAPABILITIES_SCHEMA: &str =
    "https://schemas.3leaps.dev/agentic/mission/v0/driver-capabilities.schema.json";
pub const LIFECYCLE_EVENT_SCHEMA: &str =
    "https://schemas.3leaps.dev/agentic/mission/v0/lifecycle-event.schema.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrincipalKind {
    AttestedSession,
    Service,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestationRef {
    pub issuer: String,
    pub session_id: Uuid,
    pub jti: Uuid,
    pub context_sha256: String,
    pub token_sha256: String,
    pub verification_policy_sha256: String,
    pub trust_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Principal {
    pub kind: PrincipalKind,
    pub subject: String,
    pub role: Option<String>,
    pub scope: Option<String>,
    pub attestation: Option<AttestationRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatingRole {
    pub role: String,
    pub scope: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissionPhase {
    Created,
    Active,
    Waiting,
    RecoveryPending,
    Suspended,
    Completed,
    Cancelled,
    Failed,
    DeadlineExceeded,
    BudgetExhausted,
}

impl MissionPhase {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::Cancelled
                | Self::Failed
                | Self::DeadlineExceeded
                | Self::BudgetExhausted
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissionTerminalReason {
    GoalSatisfied,
    OperatorCancelled,
    MissionDeadlineExceeded,
    BudgetExhausted,
    PolicyDenied,
    RestoreExhausted,
    RequiredDriverCapabilityUnavailable,
    InternalError,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeasePolicy {
    pub enabled: bool,
    pub lease_seconds: Option<u64>,
    pub deadman_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetPolicy {
    pub wall_clock_seconds: Option<u64>,
    pub token_limit: Option<u64>,
    pub cost_micros: Option<u64>,
    pub action_limit: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HarnessSelection {
    pub harness_id: String,
    pub driver_id: String,
    pub model: Option<String>,
    pub workspace_ref: String,
    pub environment_ref: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryPolicy {
    ResumeOrRelaunch,
    ResumeOnly,
    AskOperator,
    StandDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryRelation {
    Initial,
    Resumes,
    Relaunches,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptState {
    Starting,
    Running,
    Waiting,
    Unresponsive,
    Cancelling,
    Completed,
    Cancelled,
    Replaced,
    Failed,
    Crashed,
    TimedOut,
    Lost,
}

impl AttemptState {
    #[must_use]
    pub const fn is_live(self) -> bool {
        matches!(
            self,
            Self::Starting | Self::Running | Self::Waiting | Self::Unresponsive | Self::Cancelling
        )
    }

    #[must_use]
    pub const fn is_terminal(self) -> bool {
        !self.is_live()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptTerminalReason {
    HarnessCompleted,
    OperatorCancelled,
    ProtocolCancelled,
    ProcessReaped,
    SpawnFailed,
    IdentifyFailed,
    HarnessCrashed,
    AttemptTimedOut,
    ConnectivityLost,
    ReplacedBySuccessor,
    OutcomeUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptRecord {
    pub attempt_id: Uuid,
    pub ordinal: u32,
    pub generation: u64,
    pub fencing_token_sha256: String,
    pub recovery_relation: RecoveryRelation,
    pub predecessor_attempt_id: Option<Uuid>,
    pub state: AttemptState,
    pub driver_id: Option<String>,
    pub harness_session_id: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub terminal_reason: Option<AttemptTerminalReason>,
    pub evidence_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MissionRecord {
    pub mission_schema: String,
    pub mission_id: Uuid,
    pub revision: u64,
    pub goal: String,
    pub policy_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub initiator: Principal,
    pub authorizer: Option<Principal>,
    pub authorization_ref: Option<String>,
    pub supervisor: Principal,
    pub operating_role: OperatingRole,
    pub phase: MissionPhase,
    pub terminal_reason: Option<MissionTerminalReason>,
    pub deadline_at: Option<DateTime<Utc>>,
    pub lease_policy: LeasePolicy,
    pub budget_policy: BudgetPolicy,
    pub harness_selection: Option<HarnessSelection>,
    pub recovery_policy: RecoveryPolicy,
    pub recovery_point_ref: Option<String>,
    pub attempts: Vec<AttemptRecord>,
    pub current_attempt_id: Option<Uuid>,
    pub evidence_chain_id: Uuid,
    pub terminal_entry_hash: Option<String>,
}

/// Invariant-bearing mission aggregate.
///
/// Wire or persistence DTOs must pass [`crate::Validate`] through
/// `TryFrom<MissionRecord>` before entering domain logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mission {
    pub(crate) record: MissionRecord,
}

impl Mission {
    #[must_use]
    pub const fn record(&self) -> &MissionRecord {
        &self.record
    }

    #[must_use]
    pub fn into_record(self) -> MissionRecord {
        self.record
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverAvailability {
    Available,
    TemporarilyUnavailable,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityName {
    Create,
    Identify,
    Heartbeat,
    Cancel,
    Resume,
    Close,
    Observe,
    DeliverInput,
    ApprovalEvents,
    ToolEffectEvents,
    Artifacts,
    Checkpoint,
    UsageMetering,
    TerminalStatus,
    LocalProcessTermination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityFidelity {
    Native,
    Mapped,
    Lossy,
    Unsupported,
}

impl CapabilityFidelity {
    #[must_use]
    pub const fn is_usable(self) -> bool {
        matches!(self, Self::Native | Self::Mapped)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationLevel {
    Claim,
    DriverObserved,
    KernelObserved,
    ResourceAttested,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnforcementLevel {
    None,
    RequestOnly,
    LocalProcessControl,
    ProtocolConfirmed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplaySupport {
    None,
    Idempotent,
    Cursor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverCapability {
    pub name: CapabilityName,
    pub fidelity: CapabilityFidelity,
    pub observation: ObservationLevel,
    pub enforcement: EnforcementLevel,
    pub replay: ReplaySupport,
    pub limitation: Option<String>,
    pub evidence_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverValidityCondition {
    pub kind: String,
    pub executable_version: String,
    pub executable_sha256: String,
    pub configuration_sha256: String,
    pub platform: String,
    pub probe_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverCapabilityReport {
    pub capabilities_schema: String,
    pub report_id: Uuid,
    pub driver_id: String,
    pub driver_version: String,
    pub harness_kind: String,
    pub observed_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub availability: DriverAvailability,
    pub capabilities: Vec<DriverCapability>,
    pub validity_condition: DriverValidityCondition,
    pub evidence_ref: String,
}

impl DriverCapabilityReport {
    #[must_use]
    pub fn capability(&self, name: CapabilityName) -> Option<&DriverCapability> {
        self.capabilities
            .iter()
            .find(|capability| capability.name == name)
    }

    #[must_use]
    pub(crate) fn advertises_usable(&self, name: CapabilityName) -> bool {
        self.availability == DriverAvailability::Available
            && self
                .capability(name)
                .is_some_and(|capability| capability.fidelity.is_usable())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSourceKind {
    KernelObserved,
    VerifiedAttestation,
    DriverReported,
    HarnessReported,
    OperatorCommand,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventSource {
    pub kind: EventSourceKind,
    pub subject: String,
    pub producer_version: String,
    pub assurance: ObservationLevel,
    pub evidence_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleEvent {
    pub event_schema: String,
    pub event_id: Uuid,
    pub mission_id: Uuid,
    pub sequence: u64,
    pub previous_entry_hash: Option<String>,
    pub entry_hash: String,
    pub occurred_at: DateTime<Utc>,
    pub recorded_at: DateTime<Utc>,
    pub event_type: String,
    pub source: EventSource,
    pub payload: LifecyclePayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum LifecyclePayload {
    MissionCreated {
        revision: u64,
    },
    AuthorizationBound {
        authorizer: PrincipalRef,
    },
    MissionPhaseChanged {
        from: MissionPhase,
        to: MissionPhase,
        reason: Option<String>,
    },
    AttemptCreated {
        attempt_id: Uuid,
        ordinal: u32,
        generation: u64,
        recovery_relation: RecoveryRelation,
        predecessor_attempt_id: Option<Uuid>,
    },
    AttemptStateChanged {
        attempt_id: Uuid,
        generation: u64,
        from: AttemptState,
        to: AttemptState,
        reason: Option<String>,
    },
    DriverCapabilityEvaluated {
        attempt_id: Uuid,
        generation: u64,
        driver_id: String,
        capability: CapabilityName,
        availability: DriverAvailability,
        fidelity: CapabilityFidelity,
        report_id: Uuid,
    },
    RecoveryRequested {
        predecessor_attempt_id: Uuid,
        relation: RecoveryRelation,
    },
    RecoveryPointRecorded {
        recovery_point_ref: String,
    },
    MissionTerminal {
        phase: MissionPhase,
        reason: MissionTerminalReason,
        terminal_entry_hash: String,
    },
}

impl LifecyclePayload {
    #[must_use]
    pub const fn event_type(&self) -> &'static str {
        match self {
            Self::MissionCreated { .. } => "mission_created",
            Self::AuthorizationBound { .. } => "authorization_bound",
            Self::MissionPhaseChanged { .. } => "mission_phase_changed",
            Self::AttemptCreated { .. } => "attempt_created",
            Self::AttemptStateChanged { .. } => "attempt_state_changed",
            Self::DriverCapabilityEvaluated { .. } => "driver_capability_evaluated",
            Self::RecoveryRequested { .. } => "recovery_requested",
            Self::RecoveryPointRecorded { .. } => "recovery_point_recorded",
            Self::MissionTerminal { .. } => "mission_terminal",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrincipalRef {
    pub kind: PrincipalKind,
    pub subject: String,
    pub attestation_ref: String,
}
