//! Telemetry primitives for tracing initialization and audit-event modeling.

mod audit;
mod telemetry;

pub use audit::{AuditEvent, AuditSeverity, ZERO_HASH};
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
