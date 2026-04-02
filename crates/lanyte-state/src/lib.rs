//! Hot-tier state store bootstrap for lanyte core.
//!
//! This crate owns SQLite setup (paths, WAL mode, schema, append-only guards)
//! and is the boundary through which core accesses memory state.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
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
const MIGRATION_003: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/003_warm_exports.sql"
));
const MIGRATIONS: &[(i64, &str)] = &[(1, MIGRATION_001), (2, MIGRATION_002), (3, MIGRATION_003)];

const HOT_TIER_DIR: &str = "hot";
const WARM_TIER_DIR: &str = "warm";
const COLD_TIER_DIR: &str = "cold";
const HOT_TIER_DB_FILE: &str = "memory.sqlite3";
const WARM_EXPORT_FORMAT_VERSION: &str = "1.0";
const AUDIT_RECORDS_NO_DELETE_TRIGGER_SQL: &str = "CREATE TRIGGER IF NOT EXISTS audit_records_no_delete\nBEFORE DELETE ON audit_records\nWHEN COALESCE((SELECT value FROM state_metadata WHERE key = 'allow_audit_delete'), '0') != '1'\nBEGIN\n    SELECT RAISE(FAIL, 'audit_records is append-only');\nEND;";
pub const DEFAULT_HOT_RETENTION_DAYS: u64 = 30;

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

    #[error("invalid warm export: {0}")]
    InvalidWarmExport(String),

    #[error("session not found in hot tier: {0}")]
    SessionNotFound(String),

    #[error("invalid eviction policy: {0}")]
    InvalidEvictionPolicy(String),

    #[error("timestamp parse error: {0}")]
    TimestampParse(#[from] chrono::ParseError),
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

    #[must_use]
    pub fn warm_export_path(&self, session_id: &str) -> PathBuf {
        self.warm_dir
            .join(format!("audit-session-{session_id}.jsonl"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgeBasedEvictionPolicy {
    pub max_age: Duration,
}

impl Default for AgeBasedEvictionPolicy {
    fn default() -> Self {
        Self {
            max_age: Duration::from_secs(DEFAULT_HOT_RETENTION_DAYS * 24 * 60 * 60),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmExportMetadata {
    pub session_id: String,
    pub archive_path: PathBuf,
    pub format_version: String,
    pub genesis_prev_hash: String,
    pub record_count: usize,
    pub terminal_entry_hash: String,
    pub latest_record_timestamp: String,
    pub exported_at: String,
    pub hot_deleted_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictedSession {
    pub export: WarmExportMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WarmChainHeader {
    #[serde(rename = "_type")]
    pub line_type: String,
    pub session_id: String,
    pub genesis_prev_hash: String,
    pub record_count: usize,
    pub terminal_entry_hash: String,
    pub exported_at: String,
    pub format_version: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct WarmAuditRecordLine {
    #[serde(rename = "_type")]
    line_type: String,
    #[serde(flatten)]
    record: AuditRecord,
}

pub struct SessionExporter<'a> {
    store: &'a StateStore,
}

pub struct SessionEvictor<'a> {
    store: &'a mut StateStore,
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

    #[must_use]
    pub fn session_exporter(&self) -> SessionExporter<'_> {
        SessionExporter { store: self }
    }

    #[must_use]
    pub fn session_evictor(&mut self) -> SessionEvictor<'_> {
        SessionEvictor { store: self }
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
        let records = self.audit_records(session_id)?;
        let export = build_warm_export(session_id, &records, Utc::now());
        render_warm_export(&export.header, &export.records)
    }

    pub fn warm_export_metadata(&self, session_id: &str) -> Result<Option<WarmExportMetadata>> {
        self.connection
            .query_row(
                "SELECT session_id, archive_path, format_version, genesis_prev_hash, record_count, terminal_entry_hash, latest_record_timestamp, exported_at, hot_deleted_at FROM warm_exports WHERE session_id = ?1 LIMIT 1",
                (session_id,),
                |row| {
                    Ok(WarmExportMetadata {
                        session_id: row.get(0)?,
                        archive_path: PathBuf::from(row.get::<_, String>(1)?),
                        format_version: row.get(2)?,
                        genesis_prev_hash: row.get(3)?,
                        record_count: row.get::<_, i64>(4)? as usize,
                        terminal_entry_hash: row.get(5)?,
                        latest_record_timestamp: row.get(6)?,
                        exported_at: row.get(7)?,
                        hot_deleted_at: row.get(8)?,
                    })
                },
            )
            .optional()
            .map_err(StateError::from)
    }

    fn upsert_warm_export_metadata(&self, metadata: &WarmExportMetadata) -> Result<()> {
        self.connection.execute(
            "INSERT INTO warm_exports(session_id, archive_path, format_version, genesis_prev_hash, record_count, terminal_entry_hash, latest_record_timestamp, exported_at, hot_deleted_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) ON CONFLICT(session_id) DO UPDATE SET archive_path = excluded.archive_path, format_version = excluded.format_version, genesis_prev_hash = excluded.genesis_prev_hash, record_count = excluded.record_count, terminal_entry_hash = excluded.terminal_entry_hash, latest_record_timestamp = excluded.latest_record_timestamp, exported_at = excluded.exported_at, hot_deleted_at = excluded.hot_deleted_at",
            params![
                &metadata.session_id,
                metadata.archive_path.to_string_lossy().to_string(),
                &metadata.format_version,
                &metadata.genesis_prev_hash,
                metadata.record_count as i64,
                &metadata.terminal_entry_hash,
                &metadata.latest_record_timestamp,
                &metadata.exported_at,
                &metadata.hot_deleted_at,
            ],
        )?;
        Ok(())
    }

    fn eviction_candidates(&self, cutoff: &str) -> Result<Vec<String>> {
        let mut stmt = self.connection.prepare(
            "SELECT session_id FROM audit_records GROUP BY session_id HAVING MAX(timestamp) < ?1 ORDER BY MAX(timestamp) ASC",
        )?;
        let rows = stmt.query_map((cutoff,), |row| row.get::<_, String>(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(StateError::from)
    }

    fn delete_hot_session(
        &mut self,
        session_id: &str,
        metadata: &WarmExportMetadata,
        deleted_at: &str,
    ) -> Result<()> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch("DROP TRIGGER IF EXISTS audit_records_no_delete")?;
        tx.execute(
            "DELETE FROM audit_records WHERE session_id = ?1",
            (session_id,),
        )?;
        tx.execute(
            "INSERT INTO warm_exports(session_id, archive_path, format_version, genesis_prev_hash, record_count, terminal_entry_hash, latest_record_timestamp, exported_at, hot_deleted_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) ON CONFLICT(session_id) DO UPDATE SET archive_path = excluded.archive_path, format_version = excluded.format_version, genesis_prev_hash = excluded.genesis_prev_hash, record_count = excluded.record_count, terminal_entry_hash = excluded.terminal_entry_hash, latest_record_timestamp = excluded.latest_record_timestamp, exported_at = excluded.exported_at, hot_deleted_at = excluded.hot_deleted_at",
            params![
                session_id,
                metadata.archive_path.to_string_lossy().to_string(),
                &metadata.format_version,
                &metadata.genesis_prev_hash,
                metadata.record_count as i64,
                &metadata.terminal_entry_hash,
                &metadata.latest_record_timestamp,
                &metadata.exported_at,
                deleted_at,
            ],
        )?;
        tx.execute_batch(AUDIT_RECORDS_NO_DELETE_TRIGGER_SQL)?;
        tx.commit()?;
        Ok(())
    }
}

impl<'a> SessionExporter<'a> {
    pub fn export_to_warm(&self, session_id: &str) -> Result<WarmExportMetadata> {
        let records = self.store.audit_records(session_id)?;
        if records.is_empty() {
            return Err(StateError::SessionNotFound(session_id.to_owned()));
        }

        let export = build_warm_export(session_id, &records, Utc::now());
        let final_path = self.store.paths.warm_export_path(session_id);
        let temp_path = self.store.paths.warm_dir().join(format!(
            ".audit-session-{session_id}-{}.tmp",
            std::process::id()
        ));
        let rendered = render_warm_export(&export.header, &export.records)?;
        write_temp_export(&temp_path, &rendered)?;

        let mut metadata = verify_warm_export_file(&temp_path)?;
        if let Err(err) = fs::rename(&temp_path, &final_path) {
            let _ = fs::remove_file(&temp_path);
            return Err(StateError::Io(err));
        }

        metadata.archive_path = final_path.clone();
        self.store.upsert_warm_export_metadata(&metadata)?;
        self.store.warm_export_metadata(session_id)?.ok_or_else(|| {
            StateError::InvalidWarmExport("warm export metadata missing after write".to_owned())
        })
    }

    pub fn verify_warm_export(&self, path: &Path) -> Result<WarmExportMetadata> {
        verify_warm_export_file(path)
    }
}

impl<'a> SessionEvictor<'a> {
    pub fn evict_older_than(
        &mut self,
        now: DateTime<Utc>,
        policy: &AgeBasedEvictionPolicy,
    ) -> Result<Vec<EvictedSession>> {
        let cutoff = retention_cutoff(now, policy)?;
        let candidates = self.store.eviction_candidates(&cutoff)?;
        let mut evicted = Vec::new();

        for session_id in candidates {
            let export = self.store.session_exporter().export_to_warm(&session_id)?;
            let verified = self
                .store
                .session_exporter()
                .verify_warm_export(&export.archive_path)?;
            let deleted_at = now.to_rfc3339_opts(SecondsFormat::Millis, true);
            self.store
                .delete_hot_session(&session_id, &verified, &deleted_at)?;

            let export = self
                .store
                .warm_export_metadata(&session_id)?
                .ok_or_else(|| {
                    StateError::InvalidWarmExport(
                        "warm export metadata missing after eviction".to_owned(),
                    )
                })?;
            evicted.push(EvictedSession { export });
        }

        Ok(evicted)
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

fn retention_cutoff(now: DateTime<Utc>, policy: &AgeBasedEvictionPolicy) -> Result<String> {
    let max_age = chrono::Duration::from_std(policy.max_age).map_err(|_| {
        StateError::InvalidEvictionPolicy("max_age is too large to convert".to_owned())
    })?;
    Ok((now - max_age).to_rfc3339_opts(SecondsFormat::Millis, true))
}

struct WarmExportBuild {
    header: WarmChainHeader,
    records: Vec<AuditRecord>,
}

fn build_warm_export(
    session_id: &str,
    records: &[AuditRecord],
    exported_at: DateTime<Utc>,
) -> WarmExportBuild {
    let terminal_entry_hash = records
        .last()
        .expect("warm export requires at least one record")
        .entry_hash
        .clone();
    WarmExportBuild {
        header: WarmChainHeader {
            line_type: "chain_header".to_owned(),
            session_id: session_id.to_owned(),
            genesis_prev_hash: genesis_prev_hash(session_id),
            record_count: records.len(),
            terminal_entry_hash,
            exported_at: exported_at.to_rfc3339_opts(SecondsFormat::Millis, true),
            format_version: WARM_EXPORT_FORMAT_VERSION.to_owned(),
        },
        records: records.to_vec(),
    }
}

fn render_warm_export(header: &WarmChainHeader, records: &[AuditRecord]) -> Result<String> {
    let mut rendered = String::new();
    rendered.push_str(&serde_json::to_string(header)?);
    rendered.push('\n');
    for record in records {
        rendered.push_str(&serde_json::to_string(&WarmAuditRecordLine {
            line_type: "audit_record".to_owned(),
            record: record.clone(),
        })?);
        rendered.push('\n');
    }
    Ok(rendered)
}

fn write_temp_export(path: &Path, rendered: &str) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(rendered.as_bytes())?;
    writer.flush()?;
    Ok(())
}

fn verify_warm_export_file(path: &Path) -> Result<WarmExportMetadata> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let Some(header_line) = lines.next() else {
        return Err(StateError::InvalidWarmExport(
            "warm export is missing chain header".to_owned(),
        ));
    };
    let header: WarmChainHeader = serde_json::from_str(&header_line?)?;
    if header.line_type != "chain_header" {
        return Err(StateError::InvalidWarmExport(
            "first JSONL line must be chain_header".to_owned(),
        ));
    }
    if header.format_version != WARM_EXPORT_FORMAT_VERSION {
        return Err(StateError::InvalidWarmExport(format!(
            "unsupported warm export format version: {}",
            header.format_version
        )));
    }
    if header.genesis_prev_hash != genesis_prev_hash(&header.session_id) {
        return Err(StateError::InvalidWarmExport(
            "header genesis_prev_hash does not match session".to_owned(),
        ));
    }

    let mut records = Vec::new();
    for line in lines {
        let line = line?;
        let record_line: WarmAuditRecordLine = serde_json::from_str(&line)?;
        if record_line.line_type != "audit_record" {
            return Err(StateError::InvalidWarmExport(
                "all non-header JSONL lines must be audit_record".to_owned(),
            ));
        }
        records.push(record_line.record);
    }
    if records.is_empty() {
        return Err(StateError::InvalidWarmExport(
            "warm export must contain at least one audit record".to_owned(),
        ));
    }
    validate_audit_chain(&header.session_id, &records)?;
    if header.record_count != records.len() {
        return Err(StateError::InvalidWarmExport(format!(
            "header record_count {} does not match actual {}",
            header.record_count,
            records.len()
        )));
    }
    let terminal_entry_hash = records
        .last()
        .expect("verified warm export has records")
        .entry_hash
        .clone();
    if header.terminal_entry_hash != terminal_entry_hash {
        return Err(StateError::InvalidWarmExport(
            "header terminal_entry_hash does not match chain tip".to_owned(),
        ));
    }
    let latest_record_timestamp = records
        .iter()
        .map(|record| record.timestamp.as_str())
        .max()
        .expect("verified warm export has records")
        .to_owned();

    Ok(WarmExportMetadata {
        session_id: header.session_id,
        archive_path: path.to_path_buf(),
        format_version: header.format_version,
        genesis_prev_hash: header.genesis_prev_hash,
        record_count: header.record_count,
        terminal_entry_hash: header.terminal_entry_hash,
        latest_record_timestamp,
        exported_at: header.exported_at,
        hot_deleted_at: None,
    })
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
    use chrono::{DateTime, Utc};
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
        assert_eq!(store.schema_version().expect("schema version query"), 3);
    }

    #[test]
    fn reopen_is_idempotent_and_does_not_rerun_migrations() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);

        let store_a = StateStore::open(paths.clone()).expect("state store should open");
        assert_eq!(store_a.schema_version().expect("schema version query"), 3);

        let store_b = StateStore::open(paths).expect("state store should open again");
        assert_eq!(store_b.schema_version().expect("schema version query"), 3);
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
        let mut lines = jsonl.lines();
        let header: WarmChainHeader =
            serde_json::from_str(lines.next().expect("header line")).expect("header should decode");
        assert_eq!(header.line_type, "chain_header");
        assert_eq!(header.session_id, TEST_SESSION_ID);
        assert_eq!(header.record_count, 1);
        let record_line: WarmAuditRecordLine =
            serde_json::from_str(lines.next().expect("record line")).expect("record should decode");
        assert_eq!(record_line.line_type, "audit_record");
        assert_eq!(record_line.record, appended);
        assert!(lines.next().is_none());
    }

    #[test]
    fn session_exporter_writes_standalone_verified_jsonl() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths.clone()).expect("state store should open");

        store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_A.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-03-01T00:00:00.000Z".to_owned(),
                kind: AuditRecordKind::Effect,
                action: "orchestrator.request_completion".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"backend":"claude"}),
                verification: None,
            })
            .expect("first audit record should append");
        let second = store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_B.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-03-02T00:00:00.000Z".to_owned(),
                kind: AuditRecordKind::Outcome,
                action: "orchestrator.request_completion.outcome".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"status":"succeeded"}),
                verification: None,
            })
            .expect("second audit record should append");

        let export = store
            .session_exporter()
            .export_to_warm(TEST_SESSION_ID)
            .expect("warm export should succeed");
        let verified = store
            .session_exporter()
            .verify_warm_export(&export.archive_path)
            .expect("warm export should verify");

        assert_eq!(verified.session_id, TEST_SESSION_ID);
        assert_eq!(verified.record_count, 2);
        assert_eq!(verified.terminal_entry_hash, second.entry_hash);
        assert_eq!(
            verified.genesis_prev_hash,
            genesis_prev_hash(TEST_SESSION_ID)
        );
        assert_eq!(
            store
                .warm_export_metadata(TEST_SESSION_ID)
                .expect("warm export metadata should load"),
            Some(verified.clone())
        );
        assert_eq!(
            store
                .audit_records(TEST_SESSION_ID)
                .expect("hot records should remain")
                .len(),
            2
        );
    }

    #[test]
    fn session_evictor_moves_only_old_sessions_by_latest_timestamp() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");

        store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_A.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-03-01T00:00:00.000Z".to_owned(),
                kind: AuditRecordKind::Effect,
                action: "orchestrator.request_completion".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"backend":"claude"}),
                verification: None,
            })
            .expect("old session record should append");
        store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_B.to_owned(),
                session_id: "550e8400-e29b-41d4-a716-446655440099".to_owned(),
                timestamp: "2026-05-10T00:00:00.000Z".to_owned(),
                kind: AuditRecordKind::Effect,
                action: "orchestrator.request_completion".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"backend":"grok"}),
                verification: None,
            })
            .expect("recent session record should append");

        let evicted = store
            .session_evictor()
            .evict_older_than(
                DateTime::parse_from_rfc3339("2026-05-15T00:00:00.000Z")
                    .expect("timestamp should parse")
                    .with_timezone(&Utc),
                &AgeBasedEvictionPolicy::default(),
            )
            .expect("eviction should succeed");

        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].export.session_id, TEST_SESSION_ID);
        assert!(evicted[0].export.hot_deleted_at.is_some());
        assert!(store
            .audit_records(TEST_SESSION_ID)
            .expect("old session should load")
            .is_empty());
        assert_eq!(
            store
                .audit_records("550e8400-e29b-41d4-a716-446655440099")
                .expect("recent session should remain")
                .len(),
            1
        );
    }

    #[test]
    fn eviction_failure_does_not_delete_hot_session() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths.clone()).expect("state store should open");

        store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_A.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-03-01T00:00:00.000Z".to_owned(),
                kind: AuditRecordKind::Effect,
                action: "orchestrator.request_completion".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"backend":"claude"}),
                verification: None,
            })
            .expect("session record should append");

        fs::create_dir_all(paths.warm_export_path(TEST_SESSION_ID))
            .expect("conflicting warm export path directory should be created");
        let err = store
            .session_evictor()
            .evict_older_than(
                DateTime::parse_from_rfc3339("2026-05-15T00:00:00.000Z")
                    .expect("timestamp should parse")
                    .with_timezone(&Utc),
                &AgeBasedEvictionPolicy::default(),
            )
            .expect_err("eviction should fail when archive path is invalid");

        assert!(matches!(err, StateError::Io(_)));
        assert_eq!(
            store
                .audit_records(TEST_SESSION_ID)
                .expect("hot records should remain after failed eviction")
                .len(),
            1
        );
        assert!(store
            .warm_export_metadata(TEST_SESSION_ID)
            .expect("warm export metadata should load")
            .is_none());
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
