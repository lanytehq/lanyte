//! Hot-tier state store bootstrap for lanyte core.
//!
//! This crate owns SQLite setup (paths, WAL mode, schema, append-only guards)
//! and is the boundary through which core accesses memory state.

use std::fs;
use std::path::{Path, PathBuf};

use lanyte_common::env as common_env;
use lanyte_telemetry::{
    genesis_prev_hash, AuditEnvelopeRef, AuditRecord, AuditRecordKind, AuditSeverity,
    NewAuditRecord,
};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use thiserror::Error;

pub const LANYTE_STATE_ROOT_ENV: &str = "LANYTE_STATE_ROOT";
pub const DEFAULT_STATE_ROOT: &str = "/var/lib/lanyte/state";

const MIGRATION_001: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/001_initial.sql"
));
const MIGRATION_002: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/002_audit_records.sql"
));
const MIGRATIONS: &[(i64, &str)] = &[(1, MIGRATION_001), (2, MIGRATION_002)];

const HOT_TIER_DIR: &str = "hot";
const WARM_TIER_DIR: &str = "warm";
const COLD_TIER_DIR: &str = "cold";
const HOT_TIER_DB_FILE: &str = "memory.sqlite3";

#[derive(Debug, Error)]
pub enum StateError {
    #[error(transparent)]
    Common(#[from] lanyte_common::CommonError),

    #[error("filesystem error: {0}")]
    Io(#[from] std::io::Error),

    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("invalid audit record: {0}")]
    InvalidAuditRecord(String),

    #[error("failed to encode audit JSON: {0}")]
    AuditJson(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, StateError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatePaths {
    root_dir: PathBuf,
    hot_dir: PathBuf,
    warm_dir: PathBuf,
    cold_dir: PathBuf,
    hot_db_path: PathBuf,
}

impl StatePaths {
    #[must_use]
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        let root_dir = root_dir.into();
        let hot_dir = root_dir.join(HOT_TIER_DIR);
        let warm_dir = root_dir.join(WARM_TIER_DIR);
        let cold_dir = root_dir.join(COLD_TIER_DIR);
        let hot_db_path = hot_dir.join(HOT_TIER_DB_FILE);

        Self {
            root_dir,
            hot_dir,
            warm_dir,
            cold_dir,
            hot_db_path,
        }
    }

    pub fn from_env() -> Result<Self> {
        let root = common_env::read_env_var_utf8(LANYTE_STATE_ROOT_ENV)?
            .unwrap_or_else(|| DEFAULT_STATE_ROOT.to_owned());
        let normalized = common_env::normalize_nonempty(root, LANYTE_STATE_ROOT_ENV)?;
        Ok(Self::new(normalized))
    }

    pub fn ensure_layout(&self) -> Result<()> {
        fs::create_dir_all(&self.hot_dir)?;
        fs::create_dir_all(&self.warm_dir)?;
        fs::create_dir_all(&self.cold_dir)?;
        Ok(())
    }

    #[must_use]
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    #[must_use]
    pub fn hot_dir(&self) -> &Path {
        &self.hot_dir
    }

    #[must_use]
    pub fn warm_dir(&self) -> &Path {
        &self.warm_dir
    }

    #[must_use]
    pub fn cold_dir(&self) -> &Path {
        &self.cold_dir
    }

    #[must_use]
    pub fn hot_db_path(&self) -> &Path {
        &self.hot_db_path
    }
}

/// Store boundary for the hot-tier memory DB.
pub struct StateStore {
    paths: StatePaths,
    connection: Connection,
}

impl StateStore {
    pub fn open_default() -> Result<Self> {
        let paths = StatePaths::from_env()?;
        Self::open(paths)
    }

    pub fn open(paths: StatePaths) -> Result<Self> {
        paths.ensure_layout()?;
        let mut connection = Connection::open(paths.hot_db_path())?;
        configure_sqlite(&connection)?;
        apply_migrations(&mut connection)?;

        Ok(Self { paths, connection })
    }

    #[must_use]
    pub fn paths(&self) -> &StatePaths {
        &self.paths
    }

    pub fn schema_version(&self) -> Result<i64> {
        let version = self.connection.query_row(
            "SELECT value FROM state_metadata WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )?;
        Ok(version)
    }

    pub fn append_audit_record(&mut self, record: NewAuditRecord) -> Result<AuditRecord> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let latest: Option<(i64, String)> = tx
            .query_row(
                "SELECT chain_index, entry_hash FROM audit_records WHERE session_id = ?1 ORDER BY chain_index DESC LIMIT 1",
                (&record.session_id,),
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let (chain_index, prev_hash) = match latest {
            Some((chain_index, entry_hash)) => (chain_index + 1, entry_hash),
            None => (0, genesis_prev_hash(&record.session_id)),
        };
        let record = record.finalize(prev_hash);
        record
            .validate()
            .map_err(|err| StateError::InvalidAuditRecord(err.to_owned()))?;
        let session_id = record.session_id.clone();
        let payload_json = serde_json::to_string(&record.payload)?;
        let verification_json = record
            .verification
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;

        tx.execute(
            "INSERT INTO audit_records(entry_id, session_id, chain_index, timestamp, record_kind, action, severity, conversation_id, turn_id, action_id, causation_id, correlation_id, external_ref, trust_ref, gate_ref, payload_json, verification_json, prev_hash, entry_hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
            params![
                &record.entry_id,
                &record.session_id,
                chain_index,
                &record.timestamp,
                audit_record_kind_name(record.kind),
                &record.action,
                audit_severity_name(record.severity),
                &record.envelope.conversation_id,
                &record.envelope.turn_id,
                &record.envelope.action_id,
                &record.envelope.causation_id,
                &record.envelope.correlation_id,
                &record.envelope.external_ref,
                &record.envelope.trust_ref,
                &record.envelope.gate_ref,
                payload_json,
                verification_json,
                &record.prev_hash,
                &record.entry_hash,
            ],
        )?;
        tx.commit()?;

        self.audit_records(&session_id)?
            .into_iter()
            .last()
            .ok_or_else(|| {
                StateError::InvalidAuditRecord(
                    "appended audit record missing after commit".to_owned(),
                )
            })
    }

    pub fn audit_records(&self, session_id: &str) -> Result<Vec<AuditRecord>> {
        let mut stmt = self.connection.prepare(
            "SELECT entry_id, session_id, timestamp, record_kind, action, severity, conversation_id, turn_id, action_id, causation_id, correlation_id, external_ref, trust_ref, gate_ref, payload_json, verification_json, prev_hash, entry_hash FROM audit_records WHERE session_id = ?1 ORDER BY chain_index ASC",
        )?;
        let rows = stmt.query_map((session_id,), |row| {
            let payload_json: String = row.get(14)?;
            let verification_json: Option<String> = row.get(15)?;
            Ok::<_, rusqlite::Error>(AuditRecord {
                entry_id: row.get(0)?,
                session_id: row.get(1)?,
                timestamp: row.get(2)?,
                kind: parse_audit_record_kind(&row.get::<_, String>(3)?)?,
                action: row.get(4)?,
                severity: parse_audit_severity(&row.get::<_, String>(5)?)?,
                envelope: AuditEnvelopeRef {
                    conversation_id: row.get(6)?,
                    turn_id: row.get(7)?,
                    action_id: row.get(8)?,
                    causation_id: row.get(9)?,
                    correlation_id: row.get(10)?,
                    external_ref: row.get(11)?,
                    trust_ref: row.get(12)?,
                    gate_ref: row.get(13)?,
                },
                payload: parse_json_value(&payload_json)?,
                verification: verification_json
                    .as_deref()
                    .map(parse_json_value)
                    .transpose()?,
                prev_hash: row.get(16)?,
                entry_hash: row.get(17)?,
            })
        })?;

        let records = rows
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(StateError::from)?;
        validate_audit_chain(session_id, &records)?;
        Ok(records)
    }

    pub fn export_audit_jsonl(&self, session_id: &str) -> Result<String> {
        Ok(self
            .audit_records(session_id)?
            .into_iter()
            .map(|record| record.to_jsonl_line())
            .collect())
    }
}

fn audit_record_kind_name(kind: AuditRecordKind) -> &'static str {
    match kind {
        AuditRecordKind::Effect => "effect",
        AuditRecordKind::Outcome => "outcome",
        AuditRecordKind::GateDecision => "gate_decision",
        AuditRecordKind::Verification => "verification",
        AuditRecordKind::SessionAttestation => "session_attestation",
    }
}

fn audit_severity_name(severity: AuditSeverity) -> &'static str {
    match severity {
        AuditSeverity::Info => "info",
        AuditSeverity::Notice => "notice",
        AuditSeverity::Warning => "warning",
        AuditSeverity::Critical => "critical",
    }
}

fn parse_audit_record_kind(input: &str) -> rusqlite::Result<AuditRecordKind> {
    match input {
        "effect" => Ok(AuditRecordKind::Effect),
        "outcome" => Ok(AuditRecordKind::Outcome),
        "gate_decision" => Ok(AuditRecordKind::GateDecision),
        "verification" => Ok(AuditRecordKind::Verification),
        "session_attestation" => Ok(AuditRecordKind::SessionAttestation),
        other => Err(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown audit record kind: {other}"),
            )),
        )),
    }
}

fn parse_audit_severity(input: &str) -> rusqlite::Result<AuditSeverity> {
    match input {
        "info" => Ok(AuditSeverity::Info),
        "notice" => Ok(AuditSeverity::Notice),
        "warning" => Ok(AuditSeverity::Warning),
        "critical" => Ok(AuditSeverity::Critical),
        other => Err(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown audit severity: {other}"),
            )),
        )),
    }
}

fn parse_json_value(input: &str) -> rusqlite::Result<serde_json::Value> {
    serde_json::from_str(input).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(err))
    })
}

fn validate_audit_chain(session_id: &str, records: &[AuditRecord]) -> Result<()> {
    let mut expected_prev_hash = genesis_prev_hash(session_id);
    for record in records {
        if record.session_id != session_id {
            return Err(StateError::InvalidAuditRecord(format!(
                "record session mismatch: expected {session_id}, got {}",
                record.session_id
            )));
        }
        record
            .validate()
            .map_err(|err| StateError::InvalidAuditRecord(err.to_owned()))?;
        if record.prev_hash != expected_prev_hash {
            return Err(StateError::InvalidAuditRecord(format!(
                "broken audit chain for session {session_id}: expected prev_hash {expected_prev_hash}, got {}",
                record.prev_hash
            )));
        }
        expected_prev_hash = record.entry_hash.clone();
    }
    Ok(())
}

fn configure_sqlite(connection: &Connection) -> Result<()> {
    connection.pragma_update(None, "journal_mode", "WAL")?;
    connection.pragma_update(None, "synchronous", "NORMAL")?;
    connection.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

fn apply_migrations(connection: &mut Connection) -> Result<()> {
    let mut current = current_schema_version(connection)?;

    for (version, sql) in MIGRATIONS {
        if *version <= current {
            continue;
        }

        let tx = connection.transaction()?;
        tx.execute_batch(sql)?;
        tx.execute(
            "INSERT OR REPLACE INTO state_metadata(key, value) VALUES ('schema_version', ?1)",
            (*version,),
        )?;
        tx.commit()?;

        current = *version;
    }

    Ok(())
}

fn current_schema_version(connection: &Connection) -> Result<i64> {
    if !table_exists(connection, "state_metadata")? {
        return Ok(0);
    }

    let maybe_version: Option<i64> = connection
        .query_row(
            "SELECT value FROM state_metadata WHERE key = 'schema_version' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;

    Ok(maybe_version.unwrap_or(0))
}

fn table_exists(connection: &Connection, name: &str) -> Result<bool> {
    let exists: Option<i64> = connection
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
            (name,),
            |row| row.get(0),
        )
        .optional()?;
    Ok(exists == Some(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::ffi::OsString;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    const TEST_ENTRY_ID_A: &str = "550e8400-e29b-41d4-a716-446655440000";
    const TEST_ENTRY_ID_B: &str = "550e8400-e29b-41d4-a716-446655440001";
    const TEST_SESSION_ID: &str = "550e8400-e29b-41d4-a716-446655440002";
    const TEST_ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    static TEMP_ROOT_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_state_root() -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let pid = std::process::id();
        let sequence = TEMP_ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
        env::temp_dir().join(format!("lanyte-state-test-{pid}-{now}-{sequence}"))
    }

    #[test]
    fn path_layout_derives_expected_locations() {
        let root = PathBuf::from("/tmp/lanyte-state");
        let paths = StatePaths::new(&root);

        assert_eq!(paths.root_dir(), Path::new("/tmp/lanyte-state"));
        assert_eq!(paths.hot_dir(), Path::new("/tmp/lanyte-state/hot"));
        assert_eq!(paths.warm_dir(), Path::new("/tmp/lanyte-state/warm"));
        assert_eq!(paths.cold_dir(), Path::new("/tmp/lanyte-state/cold"));
        assert_eq!(
            paths.hot_db_path(),
            Path::new("/tmp/lanyte-state/hot/memory.sqlite3")
        );
    }

    #[test]
    fn non_utf8_env_value_is_rejected() {
        let err = common_env::map_env_var_result(
            LANYTE_STATE_ROOT_ENV,
            Err(env::VarError::NotUnicode(OsString::from("bad-bytes"))),
        )
        .expect_err("must fail");

        match StateError::from(err) {
            StateError::Common(lanyte_common::CommonError::InvalidEnvironment { key, reason }) => {
                assert_eq!(key, LANYTE_STATE_ROOT_ENV);
                assert!(reason.contains("UTF-8"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn open_bootstraps_hot_tier_sqlite() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let store = StateStore::open(paths.clone()).expect("state store should open");

        assert!(paths.hot_db_path().exists());
        assert_eq!(store.schema_version().expect("schema version query"), 2);
    }

    #[test]
    fn reopen_is_idempotent_and_does_not_rerun_migrations() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);

        let store_a = StateStore::open(paths.clone()).expect("state store should open");
        assert_eq!(store_a.schema_version().expect("schema version query"), 2);

        let store_b = StateStore::open(paths).expect("state store should open again");
        assert_eq!(store_b.schema_version().expect("schema version query"), 2);
    }

    #[test]
    fn hot_tier_bootstrap_installs_append_only_triggers() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let store = StateStore::open(paths).expect("state store should open");

        for trigger in [
            "memory_entries_no_update",
            "memory_entries_no_delete",
            "memory_entries_no_replace",
            "audit_records_no_update",
            "audit_records_no_delete",
            "audit_records_no_replace",
        ] {
            let exists: Option<i64> = store
                .connection
                .query_row(
                    "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?1 LIMIT 1",
                    (trigger,),
                    |row| row.get(0),
                )
                .optional()
                .expect("trigger query should succeed");
            assert!(exists.is_some(), "missing expected trigger: {trigger}");
        }
    }

    #[test]
    fn append_only_guards_reject_update_and_delete() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let store = StateStore::open(paths).expect("state store should open");

        store
            .connection
            .execute(
                "INSERT INTO memory_entries(entry_id, timestamp, agent_id, topic, payload_json, prev_hash) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                (
                    "550e8400-e29b-41d4-a716-446655440000",
                    "2026-02-24T20:00:00.123Z",
                    "agent-devlead",
                    "test.topic",
                    "{}",
                    "0000000000000000000000000000000000000000000000000000000000000000",
                ),
            )
            .expect("insert should succeed");

        let update_result = store.connection.execute(
            "UPDATE memory_entries SET topic = 'other' WHERE entry_id = ?1",
            ("550e8400-e29b-41d4-a716-446655440000",),
        );
        assert!(
            update_result.is_err(),
            "update should be blocked by trigger"
        );

        let delete_result = store.connection.execute(
            "DELETE FROM memory_entries WHERE entry_id = ?1",
            ("550e8400-e29b-41d4-a716-446655440000",),
        );
        assert!(
            delete_result.is_err(),
            "delete should be blocked by trigger"
        );
    }

    #[test]
    fn append_only_guards_reject_insert_or_replace() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let store = StateStore::open(paths).expect("state store should open");

        store
            .connection
            .execute(
                "INSERT INTO memory_entries(entry_id, timestamp, agent_id, topic, payload_json, prev_hash) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                (
                    "550e8400-e29b-41d4-a716-446655440000",
                    "2026-02-24T20:00:00.123Z",
                    "agent-a",
                    "topic.a",
                    "{\"v\":\"a\"}",
                    TEST_ZERO_HASH,
                ),
            )
            .expect("initial insert should succeed");

        let replace_result = store.connection.execute(
            "INSERT OR REPLACE INTO memory_entries(entry_id, timestamp, agent_id, topic, payload_json, prev_hash) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            (
                "550e8400-e29b-41d4-a716-446655440000",
                "2026-02-24T21:00:00.123Z",
                "agent-b",
                "topic.b",
                "{\"v\":\"b\"}",
                TEST_ZERO_HASH,
            ),
        );
        assert!(
            replace_result.is_err(),
            "INSERT OR REPLACE should be blocked by append-only trigger"
        );

        let (agent_id, topic, payload_json): (String, String, String) = store
            .connection
            .query_row(
                "SELECT agent_id, topic, payload_json FROM memory_entries WHERE entry_id = ?1",
                ("550e8400-e29b-41d4-a716-446655440000",),
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("original row should remain unchanged");

        assert_eq!(agent_id, "agent-a");
        assert_eq!(topic, "topic.a");
        assert_eq!(payload_json, "{\"v\":\"a\"}");
    }

    #[test]
    fn append_audit_record_builds_session_hash_chain() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");

        let first = store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_A.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-04-01T19:24:00.000Z".to_owned(),
                kind: AuditRecordKind::Effect,
                action: "orchestrator.request_completion".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef {
                    action_id: Some("550e8400-e29b-41d4-a716-446655440003".to_owned()),
                    correlation_id: Some("req-1".to_owned()),
                    trust_ref: Some("trust:session-1".to_owned()),
                    ..AuditEnvelopeRef::default()
                },
                payload: serde_json::json!({"backend":"claude"}),
                verification: None,
            })
            .expect("first audit record should append");
        let second = store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_B.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-04-01T19:25:00.000Z".to_owned(),
                kind: AuditRecordKind::Verification,
                action: "verify.request_completion".to_owned(),
                severity: AuditSeverity::Notice,
                envelope: AuditEnvelopeRef {
                    action_id: Some("550e8400-e29b-41d4-a716-446655440003".to_owned()),
                    causation_id: Some(TEST_ENTRY_ID_A.to_owned()),
                    correlation_id: Some("req-1".to_owned()),
                    trust_ref: Some("trust:session-1".to_owned()),
                    gate_ref: Some("gate:approval-1".to_owned()),
                    ..AuditEnvelopeRef::default()
                },
                payload: serde_json::json!({"status":"verified"}),
                verification: Some(serde_json::json!({"strategy":"snapshot"})),
            })
            .expect("second audit record should append");

        assert_eq!(first.prev_hash, genesis_prev_hash(TEST_SESSION_ID));
        assert_eq!(second.prev_hash, first.entry_hash);
        assert_eq!(second.envelope.gate_ref.as_deref(), Some("gate:approval-1"));
    }

    #[test]
    fn audit_records_round_trip_and_export_jsonl() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");

        let appended = store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_A.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-04-01T19:24:00.000Z".to_owned(),
                kind: AuditRecordKind::Outcome,
                action: "tool.run".to_owned(),
                severity: AuditSeverity::Warning,
                envelope: AuditEnvelopeRef {
                    conversation_id: Some("550e8400-e29b-41d4-a716-446655440010".to_owned()),
                    turn_id: Some("550e8400-e29b-41d4-a716-446655440011".to_owned()),
                    action_id: Some("550e8400-e29b-41d4-a716-446655440012".to_owned()),
                    causation_id: Some("550e8400-e29b-41d4-a716-446655440013".to_owned()),
                    correlation_id: Some("req-2".to_owned()),
                    external_ref: Some("resp-2".to_owned()),
                    trust_ref: Some("trust:session-2".to_owned()),
                    gate_ref: Some("gate:approval-2".to_owned()),
                },
                payload: serde_json::json!({"tool":"grep","status":"failed"}),
                verification: Some(serde_json::json!({"status":"failed","details":[]})),
            })
            .expect("audit record should append");

        let records = store
            .audit_records(TEST_SESSION_ID)
            .expect("audit records should load");
        assert_eq!(records, vec![appended.clone()]);

        let jsonl = store
            .export_audit_jsonl(TEST_SESSION_ID)
            .expect("jsonl export should succeed");
        let exported: AuditRecord = serde_json::from_str(jsonl.trim_end())
            .expect("jsonl line should decode to audit record");
        assert_eq!(exported, appended);
    }

    #[test]
    fn audit_append_only_guards_reject_update_and_delete() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");

        store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_A.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-04-01T19:24:00.000Z".to_owned(),
                kind: AuditRecordKind::Effect,
                action: "orchestrator.emit_message".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"text":"ok"}),
                verification: None,
            })
            .expect("audit record should append");

        let update_result = store.connection.execute(
            "UPDATE audit_records SET action = 'other' WHERE entry_id = ?1",
            (TEST_ENTRY_ID_A,),
        );
        assert!(
            update_result.is_err(),
            "audit record update should be blocked by trigger"
        );

        let delete_result = store.connection.execute(
            "DELETE FROM audit_records WHERE entry_id = ?1",
            (TEST_ENTRY_ID_A,),
        );
        assert!(
            delete_result.is_err(),
            "audit record delete should be blocked by trigger"
        );
    }

    #[test]
    fn audit_records_reject_broken_prev_hash_linkage() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");

        let first = store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_A.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-04-01T19:24:00.000Z".to_owned(),
                kind: AuditRecordKind::Effect,
                action: "orchestrator.request_completion".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"backend":"claude"}),
                verification: None,
            })
            .expect("first audit record should append");

        store
            .connection
            .execute(
                "INSERT INTO audit_records(entry_id, session_id, chain_index, timestamp, record_kind, action, severity, payload_json, prev_hash, entry_hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                (
                    TEST_ENTRY_ID_B,
                    TEST_SESSION_ID,
                    1_i64,
                    "2026-04-01T19:25:00.000Z",
                    "outcome",
                    "orchestrator.request_completion.outcome",
                    "info",
                    "{\"status\":\"succeeded\"}",
                    TEST_ZERO_HASH,
                    first.entry_hash.as_str(),
                ),
            )
            .expect("tampered insert should succeed at sqlite layer");

        let err = store
            .audit_records(TEST_SESSION_ID)
            .expect_err("broken chain should fail closed");
        assert!(
            err.to_string().contains("broken audit chain")
                || err
                    .to_string()
                    .contains("entry_hash does not match canonical hash surface")
        );
    }

    #[test]
    fn audit_records_reject_invalid_genesis_anchor() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let store = StateStore::open(paths).expect("state store should open");

        store
            .connection
            .execute(
                "INSERT INTO audit_records(entry_id, session_id, chain_index, timestamp, record_kind, action, severity, payload_json, prev_hash, entry_hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                (
                    TEST_ENTRY_ID_A,
                    TEST_SESSION_ID,
                    0_i64,
                    "2026-04-01T19:24:00.000Z",
                    "effect",
                    "orchestrator.request_completion",
                    "info",
                    "{\"backend\":\"claude\"}",
                    TEST_ZERO_HASH,
                    TEST_ZERO_HASH,
                ),
            )
            .expect("tampered insert should succeed at sqlite layer");

        let err = store
            .audit_records(TEST_SESSION_ID)
            .expect_err("invalid genesis anchor should fail closed");
        assert!(
            err.to_string()
                .contains("entry_hash does not match canonical hash surface")
                || err.to_string().contains("broken audit chain")
        );
    }
}
