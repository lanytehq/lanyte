//! Telemetry primitives for tracing initialization and audit-event modeling.

mod audit;
mod telemetry;

pub use audit::{
    compute_entry_hash, genesis_prev_hash, AuditEnvelopeRef, AuditEvent, AuditRecord,
    AuditRecordKind, AuditSeverity, NewAuditRecord, ZERO_HASH,
};
pub use telemetry::{init_tracing, TelemetryError, DEFAULT_LOG_FILTER};

/// Emit a structured audit event through the active tracing subscriber.
pub fn emit_audit_event(event: &AuditEvent) {
    tracing::info!(
        entry_id = %event.entry_id,
        peer_id = %event.peer_id,
        timestamp = %event.timestamp,
        action = %event.action,
        actor = event.actor.as_deref().unwrap_or(""),
        prev_hash = %event.prev_hash,
        severity = ?event.severity,
        details = %serde_json::Value::Object(event.details.clone()),
        "audit event"
    );
}

/// Emit an internal orchestrator-owned audit record through the active tracing subscriber.
pub fn emit_audit_record(record: &AuditRecord) {
    tracing::info!(
        entry_id = %record.entry_id,
        session_id = %record.session_id,
        timestamp = %record.timestamp,
        action = %record.action,
        record_kind = ?record.kind,
        prev_hash = %record.prev_hash,
        entry_hash = %record.entry_hash,
        severity = ?record.severity,
        correlation_id = record.envelope.correlation_id.as_deref().unwrap_or(""),
        trust_ref = record.envelope.trust_ref.as_deref().unwrap_or(""),
        gate_ref = record.envelope.gate_ref.as_deref().unwrap_or(""),
        payload = %record.payload,
        verification = %record
            .verification
            .clone()
            .unwrap_or(serde_json::Value::Null),
        "audit record"
    );
}
