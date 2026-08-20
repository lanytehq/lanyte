CREATE TABLE IF NOT EXISTS mission_control_bindings (
    evidence_ref TEXT PRIMARY KEY NOT NULL,
    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    original_result_hash TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mission_control_bindings_mission_id
ON mission_control_bindings(mission_id);
