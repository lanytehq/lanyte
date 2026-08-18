CREATE TABLE IF NOT EXISTS mission_requests (
    idempotency_key TEXT PRIMARY KEY NOT NULL,
    request_fingerprint TEXT NOT NULL,
    mission_id TEXT NOT NULL UNIQUE REFERENCES missions(mission_id)
);

CREATE INDEX IF NOT EXISTS idx_mission_requests_mission_id
ON mission_requests(mission_id);
