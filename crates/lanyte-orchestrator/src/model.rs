use chrono::{DateTime, Utc};
use lanyte_common::{ChannelId, ProviderKind};
use lanyte_llm::{CompletionRequest, StreamEvent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EntityRef {
    Human {
        handle: String,
    },
    SupervisedSession {
        role: String,
        session_id: Uuid,
        supervisor: String,
    },
    AutonomousAgent {
        agent_id: String,
    },
    Peer {
        peer_id: String,
        channel: ChannelId,
    },
    Skill {
        skill_id: String,
        invocation_id: Option<Uuid>,
    },
    System {
        component: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    pub id: Uuid,
    pub occurred_at: DateTime<Utc>,
    pub conversation_id: Option<Uuid>,
    pub turn_id: Option<Uuid>,
    pub action_id: Option<Uuid>,
    pub causation_id: Option<Uuid>,
    pub correlation_id: Option<String>,
    pub external_ref: Option<String>,
    pub source: EntityRef,
    pub target: Option<EntityRef>,
    pub trust_ref: Option<String>,
    pub gate_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityMessage {
    pub intent: MessageIntent,
    pub parts: Vec<ContentPart>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageIntent {
    Inform,
    Ask,
    ProposeAction,
    Approve,
    Deny,
    DeliverResult,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Structured {
        value: Value,
    },
    ResourceRef {
        kind: String,
        uri: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolResult {
        id: String,
        name: String,
        value: Value,
    },
    Annotation {
        kind: String,
        payload: Value,
    },
    ApprovalToken {
        token_ref: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OrchestratorEvent {
    IngressMessage {
        envelope: Envelope,
        message: EntityMessage,
    },
    LlmStream {
        envelope: Envelope,
        event: StreamEvent,
    },
    ActionResult {
        envelope: Envelope,
        outcome: ActionOutcome,
    },
    GateResolved {
        envelope: Envelope,
        decision: GateDecision,
    },
    TimerFired {
        envelope: Envelope,
        timer_kind: TimerKind,
    },
    StateLoaded {
        envelope: Envelope,
        snapshot_ref: String,
    },
    SystemNotice {
        envelope: Envelope,
        notice: SystemNotice,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OrchestratorEffect {
    EmitMessage {
        envelope: Envelope,
        message: EntityMessage,
    },
    RequestCompletion {
        envelope: Envelope,
        provider: Option<ProviderKind>,
        request: CompletionRequest,
    },
    InvokeSkill {
        envelope: Envelope,
        skill_id: String,
        input: Value,
    },
    SendPeerRequest {
        envelope: Envelope,
        channel: ChannelId,
        payload: Value,
    },
    PersistState {
        envelope: Envelope,
        checkpoint_ref: String,
    },
    AppendMemory {
        envelope: Envelope,
        record_kind: String,
        payload: Value,
    },
    OpenGate {
        envelope: Envelope,
        proposal: GateProposal,
    },
    ScheduleTimer {
        envelope: Envelope,
        timer_kind: TimerKind,
        at: DateTime<Utc>,
    },
    EmitTelemetry {
        envelope: Envelope,
        payload: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcome {
    pub status: ActionStatus,
    pub result: Option<Value>,
    pub error: Option<ActionError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionStatus {
    Proposed,
    AwaitingApproval,
    Approved,
    InProgress,
    Succeeded,
    Failed,
    Cancelled,
    Expired,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GateDecision {
    Approved { token_ref: Option<String> },
    Denied { reason: Option<String> },
    Expired,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateProposal {
    pub action_summary: String,
    pub risk_class: String,
    pub requested_by: EntityRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimerKind {
    Retry,
    Wake,
    LeaseExpiry,
    Deadline,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemNotice {
    pub kind: String,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}
