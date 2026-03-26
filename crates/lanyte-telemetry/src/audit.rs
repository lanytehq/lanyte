use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditSeverity {
    Info,
    Notice,
    Warning,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub entry_id: String,
    pub peer_id: String,
    pub timestamp: String,
    pub action: String,
    pub actor: Option<String>,
    pub prev_hash: String,
    pub details: Map<String, Value>,
    pub severity: AuditSeverity,
}

impl AuditEvent {
    #[must_use]
    pub fn new(
        entry_id: impl Into<String>,
        peer_id: impl Into<String>,
        timestamp: impl Into<String>,
        action: impl Into<String>,
        prev_hash: impl Into<String>,
    ) -> Self {
        Self {
            entry_id: entry_id.into(),
            peer_id: peer_id.into(),
            timestamp: timestamp.into(),
            action: action.into(),
            actor: None,
            prev_hash: prev_hash.into(),
            details: Map::new(),
            severity: AuditSeverity::Info,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !is_uuid_v4(&self.entry_id) {
            return Err("entry_id must be UUID v4 lowercase hex");
        }
        if self.peer_id.trim().is_empty() {
            return Err("peer_id must not be empty");
        }
        if self.timestamp.trim().is_empty() {
            return Err("timestamp must not be empty");
        }
        if self.action.trim().is_empty() {
            return Err("action must not be empty");
        }
        if !is_sha256_hex(&self.prev_hash) {
            return Err("prev_hash must be 64 lowercase hex chars");
        }
        Ok(())
    }

    /// Build the telemetry-channel wire payload shape with `type = audit_event`.
    #[must_use]
    pub fn to_payload(&self) -> Value {
        let mut root = Map::new();
        root.insert("type".to_owned(), Value::String("audit_event".to_owned()));
        root.insert("entry_id".to_owned(), Value::String(self.entry_id.clone()));
        root.insert("peer_id".to_owned(), Value::String(self.peer_id.clone()));
        root.insert(
            "timestamp".to_owned(),
            Value::String(self.timestamp.clone()),
        );
        root.insert("action".to_owned(), Value::String(self.action.clone()));
        root.insert(
            "prev_hash".to_owned(),
            Value::String(self.prev_hash.clone()),
        );
        if let Some(actor) = self.actor.clone() {
            root.insert("actor".to_owned(), Value::String(actor));
        }
        if !self.details.is_empty() {
            root.insert("details".to_owned(), Value::Object(self.details.clone()));
        }
        root.insert(
            "severity".to_owned(),
            Value::String(
                match self.severity {
                    AuditSeverity::Info => "info",
                    AuditSeverity::Notice => "notice",
                    AuditSeverity::Warning => "warning",
                    AuditSeverity::Critical => "critical",
                }
                .to_owned(),
            ),
        );
        Value::Object(root)
    }
}

fn is_sha256_hex(input: &str) -> bool {
    input.len() == 64
        && input
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn is_uuid_v4(input: &str) -> bool {
    if input.len() != 36 {
        return false;
    }

    for (index, ch) in input.chars().enumerate() {
        match index {
            8 | 13 | 18 | 23 => {
                if ch != '-' {
                    return false;
                }
            }
            14 => {
                if ch != '4' {
                    return false;
                }
            }
            19 => {
                if !matches!(ch, '8' | '9' | 'a' | 'b') {
                    return false;
                }
            }
            _ => {
                if !ch.is_ascii_hexdigit() || ch.is_ascii_uppercase() {
                    return false;
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_event_uses_expected_defaults() {
        let event = AuditEvent::new(
            "550e8400-e29b-41d4-a716-446655440000",
            "lanyte-core",
            "2026-02-24T20:00:00.123Z",
            "gateway.peer_connected",
            ZERO_HASH,
        );

        assert_eq!(event.severity, AuditSeverity::Info);
        assert!(event.actor.is_none());
        assert!(event.details.is_empty());
    }

    #[test]
    fn validate_rejects_invalid_prev_hash() {
        let event = AuditEvent::new(
            "550e8400-e29b-41d4-a716-446655440000",
            "lanyte-core",
            "2026-02-24T20:00:00.123Z",
            "gateway.peer_connected",
            "invalid",
        );
        assert_eq!(
            event.validate(),
            Err("prev_hash must be 64 lowercase hex chars")
        );
    }

    #[test]
    fn validate_rejects_invalid_entry_id() {
        let event = AuditEvent::new(
            "not-a-uuid",
            "lanyte-core",
            "2026-02-24T20:00:00.123Z",
            "gateway.peer_connected",
            ZERO_HASH,
        );
        assert_eq!(
            event.validate(),
            Err("entry_id must be UUID v4 lowercase hex")
        );
    }

    #[test]
    fn payload_contains_schema_discriminator() {
        let event = AuditEvent::new(
            "550e8400-e29b-41d4-a716-446655440000",
            "lanyte-core",
            "2026-02-24T20:00:00.123Z",
            "gateway.peer_connected",
            ZERO_HASH,
        );
        let payload = event.to_payload();
        assert_eq!(payload["type"], Value::String("audit_event".to_owned()));
        assert_eq!(payload["severity"], Value::String("info".to_owned()));
    }
}
