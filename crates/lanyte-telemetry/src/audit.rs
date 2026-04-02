use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

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
            Value::String(audit_severity_name(self.severity).to_owned()),
        );
        Value::Object(root)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditRecordKind {
    Effect,
    Outcome,
    GateDecision,
    Verification,
    SessionAttestation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AuditEnvelopeRef {
    pub conversation_id: Option<String>,
    pub turn_id: Option<String>,
    pub action_id: Option<String>,
    pub causation_id: Option<String>,
    pub correlation_id: Option<String>,
    pub external_ref: Option<String>,
    pub trust_ref: Option<String>,
    pub gate_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewAuditRecord {
    pub entry_id: String,
    pub session_id: String,
    pub timestamp: String,
    pub kind: AuditRecordKind,
    pub action: String,
    pub severity: AuditSeverity,
    pub envelope: AuditEnvelopeRef,
    pub payload: Value,
    pub verification: Option<Value>,
}

impl NewAuditRecord {
    #[must_use]
    pub fn finalize(self, prev_hash: impl Into<String>) -> AuditRecord {
        let prev_hash = prev_hash.into();
        let entry_hash = compute_entry_hash(&self, &prev_hash);
        AuditRecord {
            entry_id: self.entry_id,
            session_id: self.session_id,
            timestamp: self.timestamp,
            kind: self.kind,
            action: self.action,
            severity: self.severity,
            envelope: self.envelope,
            payload: self.payload,
            verification: self.verification,
            prev_hash,
            entry_hash,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditRecord {
    pub entry_id: String,
    pub session_id: String,
    pub timestamp: String,
    pub kind: AuditRecordKind,
    pub action: String,
    pub severity: AuditSeverity,
    pub envelope: AuditEnvelopeRef,
    pub payload: Value,
    pub verification: Option<Value>,
    pub prev_hash: String,
    pub entry_hash: String,
}

impl AuditRecord {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !is_uuid_v4(&self.entry_id) {
            return Err("entry_id must be UUID v4 lowercase hex");
        }
        if !is_uuid_v4(&self.session_id) {
            return Err("session_id must be UUID v4 lowercase hex");
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
        if !is_sha256_hex(&self.entry_hash) {
            return Err("entry_hash must be 64 lowercase hex chars");
        }
        if self.entry_hash != compute_entry_hash(&self.as_new_record(), &self.prev_hash) {
            return Err("entry_hash does not match canonical hash surface");
        }
        Ok(())
    }

    #[must_use]
    pub fn to_jsonl_line(&self) -> String {
        let mut line = serde_json::to_string(self).expect("audit record should serialize");
        line.push('\n');
        line
    }

    #[must_use]
    pub fn as_new_record(&self) -> NewAuditRecord {
        NewAuditRecord {
            entry_id: self.entry_id.clone(),
            session_id: self.session_id.clone(),
            timestamp: self.timestamp.clone(),
            kind: self.kind,
            action: self.action.clone(),
            severity: self.severity,
            envelope: self.envelope.clone(),
            payload: self.payload.clone(),
            verification: self.verification.clone(),
        }
    }
}

#[must_use]
pub fn genesis_prev_hash(session_id: &str) -> String {
    sha256_hex(format!("genesis:{session_id}").as_bytes())
}

/// Hash contract for AGI-006.
///
/// `entry_hash` is computed as `sha256(canonical_json(record_without_entry_hash))`.
/// The serialized surface includes `prev_hash`, `session_id`, envelope correlation metadata,
/// payload, and optional verification payload. The record's own `entry_hash` field is excluded.
/// The first record in a session chain uses `prev_hash = sha256("genesis:<session_id>")`.
#[must_use]
pub fn compute_entry_hash(record: &NewAuditRecord, prev_hash: &str) -> String {
    let surface = RecordHashSurface {
        entry_id: &record.entry_id,
        session_id: &record.session_id,
        timestamp: &record.timestamp,
        kind: record.kind,
        action: &record.action,
        severity: record.severity,
        envelope: &record.envelope,
        payload: &record.payload,
        verification: record.verification.as_ref(),
        prev_hash,
    };
    let bytes = serde_json::to_vec(&surface).expect("hash surface should serialize");
    sha256_hex(&bytes)
}

#[derive(Serialize)]
struct RecordHashSurface<'a> {
    entry_id: &'a str,
    session_id: &'a str,
    timestamp: &'a str,
    kind: AuditRecordKind,
    action: &'a str,
    severity: AuditSeverity,
    envelope: &'a AuditEnvelopeRef,
    payload: &'a Value,
    verification: Option<&'a Value>,
    prev_hash: &'a str,
}

fn audit_severity_name(severity: AuditSeverity) -> &'static str {
    match severity {
        AuditSeverity::Info => "info",
        AuditSeverity::Notice => "notice",
        AuditSeverity::Warning => "warning",
        AuditSeverity::Critical => "critical",
    }
}

fn sha256_hex(input: &[u8]) -> String {
    format!("{:x}", Sha256::digest(input))
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

    const ENTRY_ID: &str = "550e8400-e29b-41d4-a716-446655440000";
    const SESSION_ID: &str = "550e8400-e29b-41d4-a716-446655440001";

    #[test]
    fn new_event_uses_expected_defaults() {
        let event = AuditEvent::new(
            ENTRY_ID,
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
            ENTRY_ID,
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
            ENTRY_ID,
            "lanyte-core",
            "2026-02-24T20:00:00.123Z",
            "gateway.peer_connected",
            ZERO_HASH,
        );
        let payload = event.to_payload();
        assert_eq!(payload["type"], Value::String("audit_event".to_owned()));
        assert_eq!(payload["severity"], Value::String("info".to_owned()));
    }

    #[test]
    fn genesis_hash_is_session_scoped() {
        let first = genesis_prev_hash(SESSION_ID);
        let second = genesis_prev_hash(SESSION_ID);
        let third = genesis_prev_hash("550e8400-e29b-41d4-a716-446655440002");

        assert_eq!(first, second);
        assert_ne!(first, third);
        assert_ne!(first, ZERO_HASH);
    }

    #[test]
    fn record_hash_changes_when_payload_changes() {
        let base = NewAuditRecord {
            entry_id: ENTRY_ID.to_owned(),
            session_id: SESSION_ID.to_owned(),
            timestamp: "2026-04-01T19:24:00.000Z".to_owned(),
            kind: AuditRecordKind::Effect,
            action: "orchestrator.request_completion".to_owned(),
            severity: AuditSeverity::Info,
            envelope: AuditEnvelopeRef {
                action_id: Some("550e8400-e29b-41d4-a716-446655440003".to_owned()),
                ..AuditEnvelopeRef::default()
            },
            payload: serde_json::json!({"backend":"claude","max_tokens":256}),
            verification: None,
        };
        let changed = NewAuditRecord {
            payload: serde_json::json!({"backend":"grok","max_tokens":256}),
            ..base.clone()
        };
        let prev_hash = genesis_prev_hash(SESSION_ID);

        assert_ne!(
            compute_entry_hash(&base, &prev_hash),
            compute_entry_hash(&changed, &prev_hash)
        );
    }

    #[test]
    fn validate_rejects_mismatched_entry_hash() {
        let mut record = NewAuditRecord {
            entry_id: ENTRY_ID.to_owned(),
            session_id: SESSION_ID.to_owned(),
            timestamp: "2026-04-01T19:24:00.000Z".to_owned(),
            kind: AuditRecordKind::Verification,
            action: "verify.tool_execution".to_owned(),
            severity: AuditSeverity::Notice,
            envelope: AuditEnvelopeRef {
                action_id: Some("550e8400-e29b-41d4-a716-446655440003".to_owned()),
                trust_ref: Some("trust:session-1".to_owned()),
                ..AuditEnvelopeRef::default()
            },
            payload: serde_json::json!({"status":"verified"}),
            verification: Some(serde_json::json!({"strategy":"diff"})),
        }
        .finalize(genesis_prev_hash(SESSION_ID));
        record.entry_hash = ZERO_HASH.to_owned();

        assert_eq!(
            record.validate(),
            Err("entry_hash does not match canonical hash surface")
        );
    }

    #[test]
    fn jsonl_export_is_one_record_per_line() {
        let record = NewAuditRecord {
            entry_id: ENTRY_ID.to_owned(),
            session_id: SESSION_ID.to_owned(),
            timestamp: "2026-04-01T19:24:00.000Z".to_owned(),
            kind: AuditRecordKind::Outcome,
            action: "tool.run".to_owned(),
            severity: AuditSeverity::Warning,
            envelope: AuditEnvelopeRef::default(),
            payload: serde_json::json!({"status":"failed"}),
            verification: None,
        }
        .finalize(genesis_prev_hash(SESSION_ID));

        let line = record.to_jsonl_line();
        assert!(line.ends_with('\n'));
        let trimmed = line.trim_end_matches('\n');
        let decoded: AuditRecord = serde_json::from_str(trimmed).expect("record should decode");
        assert_eq!(decoded, record);
    }
}
