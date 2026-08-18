CREATE TABLE IF NOT EXISTS missions (
    mission_id TEXT PRIMARY KEY NOT NULL,
    mission_schema TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    goal TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    operating_role TEXT NOT NULL,
    operating_scope TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    evidence_chain_id TEXT NOT NULL,
    record_json TEXT NOT NULL,
    receipt_entry_id TEXT NOT NULL UNIQUE,
    receipt_entry_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_missions_phase
ON missions(phase);

CREATE INDEX IF NOT EXISTS idx_missions_operating_role_scope
ON missions(operating_role, operating_scope);

CREATE INDEX IF NOT EXISTS idx_missions_created_at
ON missions(created_at);
