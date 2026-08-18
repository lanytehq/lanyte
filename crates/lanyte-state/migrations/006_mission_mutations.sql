CREATE TABLE IF NOT EXISTS mission_mutations (
    idempotency_key TEXT PRIMARY KEY NOT NULL,
    request_fingerprint TEXT NOT NULL,
    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
    operation TEXT NOT NULL,
    result_json TEXT NOT NULL,
    reserved_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_mission_mutations_mission_id
ON mission_mutations(mission_id);
