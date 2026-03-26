CREATE TABLE IF NOT EXISTS memory_entries (
    entry_id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    topic TEXT,
    payload_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    supersedes_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY(supersedes_id) REFERENCES memory_entries(entry_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_entries_timestamp ON memory_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_entries_agent_id ON memory_entries(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_entries_topic ON memory_entries(topic);

CREATE TABLE IF NOT EXISTS state_metadata (
    key TEXT PRIMARY KEY NOT NULL,
    value INTEGER NOT NULL
);

CREATE TRIGGER IF NOT EXISTS memory_entries_no_update
BEFORE UPDATE ON memory_entries
BEGIN
    SELECT RAISE(FAIL, 'memory_entries is append-only');
END;

CREATE TRIGGER IF NOT EXISTS memory_entries_no_delete
BEFORE DELETE ON memory_entries
BEGIN
    SELECT RAISE(FAIL, 'memory_entries is append-only');
END;

-- Prevent REPLACE-style overwrites by rejecting inserts that target an existing entry_id.
CREATE TRIGGER IF NOT EXISTS memory_entries_no_replace
BEFORE INSERT ON memory_entries
WHEN EXISTS (SELECT 1 FROM memory_entries WHERE entry_id = NEW.entry_id)
BEGIN
    SELECT RAISE(FAIL, 'memory_entries is append-only');
END;

