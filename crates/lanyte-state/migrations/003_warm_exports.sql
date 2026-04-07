CREATE TABLE IF NOT EXISTS warm_exports (
    session_id TEXT PRIMARY KEY NOT NULL,
    archive_path TEXT NOT NULL,
    format_version TEXT NOT NULL,
    genesis_prev_hash TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    terminal_entry_hash TEXT NOT NULL,
    latest_record_timestamp TEXT NOT NULL,
    exported_at TEXT NOT NULL,
    hot_deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_warm_exports_hot_deleted_at ON warm_exports(hot_deleted_at);

INSERT OR IGNORE INTO state_metadata(key, value) VALUES ('allow_audit_delete', '0');

DROP TRIGGER IF EXISTS audit_records_no_delete;

CREATE TRIGGER IF NOT EXISTS audit_records_no_delete
BEFORE DELETE ON audit_records
WHEN COALESCE((SELECT value FROM state_metadata WHERE key = 'allow_audit_delete'), '0') != '1'
BEGIN
    SELECT RAISE(FAIL, 'audit_records is append-only');
END;
