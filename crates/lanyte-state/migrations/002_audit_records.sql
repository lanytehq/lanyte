CREATE TABLE IF NOT EXISTS audit_records (
    entry_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    chain_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    record_kind TEXT NOT NULL,
    action TEXT NOT NULL,
    severity TEXT NOT NULL,
    conversation_id TEXT,
    turn_id TEXT,
    action_id TEXT,
    causation_id TEXT,
    correlation_id TEXT,
    external_ref TEXT,
    trust_ref TEXT,
    gate_ref TEXT,
    payload_json TEXT NOT NULL,
    verification_json TEXT,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(session_id, chain_index)
);

CREATE INDEX IF NOT EXISTS idx_audit_records_session_chain ON audit_records(session_id, chain_index);
CREATE INDEX IF NOT EXISTS idx_audit_records_action_id ON audit_records(action_id);
CREATE INDEX IF NOT EXISTS idx_audit_records_correlation_id ON audit_records(correlation_id);

CREATE TRIGGER IF NOT EXISTS audit_records_no_update
BEFORE UPDATE ON audit_records
BEGIN
    SELECT RAISE(FAIL, 'audit_records is append-only');
END;

CREATE TRIGGER IF NOT EXISTS audit_records_no_delete
BEFORE DELETE ON audit_records
BEGIN
    SELECT RAISE(FAIL, 'audit_records is append-only');
END;

CREATE TRIGGER IF NOT EXISTS audit_records_no_replace
BEFORE INSERT ON audit_records
WHEN EXISTS (SELECT 1 FROM audit_records WHERE entry_id = NEW.entry_id)
BEGIN
    SELECT RAISE(FAIL, 'audit_records is append-only');
END;
