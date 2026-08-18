//! Hot-tier state store bootstrap for lanyte core.
//!
//! This crate owns SQLite setup (paths, WAL mode, schema, append-only guards)
//! and is the boundary through which core accesses memory state.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use chrono::{DateTime, SecondsFormat, Utc};
use lanyte_common::env as common_env;
use lanyte_mission::{
    validate_history, EventSourceKind, LifecycleEvent, LifecyclePayload, MissionPhase,
    MissionRecord, Validate,
};
use lanyte_telemetry::{
    genesis_prev_hash, AuditEnvelopeRef, AuditRecord, AuditRecordKind, AuditSeverity,
    NewAuditRecord,
};
use rusqlite::{
    params, params_from_iter, Connection, OptionalExtension, Transaction, TransactionBehavior,
};
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
const MIGRATION_004: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/004_missions.sql"
));
const MIGRATION_005: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/005_mission_requests.sql"
));
const MIGRATION_006: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/006_mission_mutations.sql"
));
const MIGRATIONS: &[(i64, &str)] = &[
    (1, MIGRATION_001),
    (2, MIGRATION_002),
    (3, MIGRATION_003),
    (4, MIGRATION_004),
    (5, MIGRATION_005),
    (6, MIGRATION_006),
];

const HOT_TIER_DIR: &str = "hot";
const WARM_TIER_DIR: &str = "warm";
const COLD_TIER_DIR: &str = "cold";
const HOT_TIER_DB_FILE: &str = "memory.sqlite3";
const WARM_EXPORT_FORMAT_VERSION: &str = "1.0";
const AUDIT_RECORDS_NO_DELETE_TRIGGER_SQL: &str = "CREATE TRIGGER IF NOT EXISTS audit_records_no_delete\nBEFORE DELETE ON audit_records\nWHEN COALESCE((SELECT value FROM state_metadata WHERE key = 'allow_audit_delete'), '0') != '1'\nBEGIN\n    SELECT RAISE(FAIL, 'audit_records is append-only');\nEND;";
pub const DEFAULT_HOT_RETENTION_DAYS: u64 = 30;
pub const MAX_MISSION_LIST_LIMIT: u16 = 200;
const MISSION_LIST_CURSOR_VERSION: u8 = 1;

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

    #[error("invalid mission projection: {0}")]
    InvalidMissionProjection(String),

    #[error("mission create idempotency conflict for key {key}")]
    MissionIdempotencyConflict { key: String },

    #[error("invalid mission list request: {0}")]
    InvalidMissionList(String),

    #[error("failed to encode audit JSON: {0}")]
    AuditJson(#[from] serde_json::Error),

    #[error("invalid warm export: {0}")]
    InvalidWarmExport(String),

    #[error("session not found in hot tier: {0}")]
    SessionNotFound(String),

    #[error("eviction race detected for session {session_id}: hot tier advanced after export")]
    EvictionConflict { session_id: String },

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
        fs::create_dir_all(self.workspace_root())?;
        fs::create_dir_all(self.pin_dir())?;
        Ok(())
    }

    #[must_use]
    pub fn workspace_root(&self) -> PathBuf {
        self.root_dir.join("workspaces")
    }

    #[must_use]
    pub fn pin_dir(&self) -> PathBuf {
        self.root_dir.join("pins")
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

#[derive(Debug, Clone, PartialEq)]
pub struct NewMissionProjectionReceipt {
    pub event: LifecycleEvent,
    pub envelope: AuditEnvelopeRef,
    pub verification: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredMissionProjection {
    pub mission: MissionRecord,
    pub audit_entry_id: String,
    pub audit_entry_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionProjectionWrite {
    pub projection: StoredMissionProjection,
    pub receipt: AuditRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionCreateIdempotency {
    pub key: String,
    pub request_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionMutationIdempotency {
    pub key: String,
    pub request_fingerprint: String,
    pub operation: String,
    pub result_json: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IdempotentMissionWrite {
    pub write: MissionProjectionWrite,
    pub replayed: bool,
    pub replayed_result_json: Option<String>,
}

/// An opaque, filter-bound position in one caller-scoped mission listing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MissionListCursor {
    version: u8,
    created_at: String,
    mission_id: String,
    snapshot_sequence: i64,
    operating_role: String,
    operating_scope: String,
    phases: Vec<String>,
}

/// Caller-scoped mission query. An empty phase list includes every phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionListFilter {
    pub operating_role: String,
    pub operating_scope: String,
    pub phases: Vec<MissionPhase>,
    pub limit: u16,
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionListPage {
    pub projections: Vec<StoredMissionProjection>,
    pub next_cursor: Option<String>,
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

    pub fn reserve_mutation(
        &mut self,
        mission_id: &str,
        idempotency: &MissionMutationIdempotency,
    ) -> Result<Option<String>> {
        validate_mission_create_idempotency(&idempotency.key, &idempotency.request_fingerprint)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, String)> = tx
            .query_row(
                "SELECT request_fingerprint, result_json FROM mission_mutations WHERE idempotency_key = ?1 LIMIT 1",
                [&idempotency.key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((stored, result)) = existing {
            if stored != idempotency.request_fingerprint {
                return Err(StateError::MissionIdempotencyConflict {
                    key: idempotency.key.clone(),
                });
            }
            tx.commit()?;
            if result.is_empty() {
                return Err(StateError::InvalidMissionProjection(
                    "identical mutation is already in flight".to_owned(),
                ));
            }
            return Ok(Some(result));
        }
        tx.execute(
            "INSERT INTO mission_mutations(idempotency_key, request_fingerprint, mission_id, operation, result_json) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                &idempotency.key,
                &idempotency.request_fingerprint,
                mission_id,
                &idempotency.operation,
                "",
            ],
        )?;
        tx.commit()?;
        Ok(None)
    }

    pub fn release_mutation(&mut self, key: &str, fingerprint: &str) -> Result<()> {
        self.connection.execute(
            "DELETE FROM mission_mutations WHERE idempotency_key = ?1 AND request_fingerprint = ?2 AND result_json = ''",
            params![key, fingerprint],
        )?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn replay_mutation(&self, key: &str, fingerprint: &str) -> Result<Option<String>> {
        validate_mission_create_idempotency(key, fingerprint)?;
        let existing: Option<(String, String)> = self
            .connection
            .query_row(
                "SELECT request_fingerprint, result_json FROM mission_mutations WHERE idempotency_key = ?1 LIMIT 1",
                [key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        match existing {
            None => Ok(None),
            Some((stored, result)) if stored == fingerprint => Ok(Some(result)),
            Some(_) => Err(StateError::MissionIdempotencyConflict {
                key: key.to_owned(),
            }),
        }
    }

    pub fn schema_version(&self) -> Result<i64> {
        let version = self.connection.query_row(
            "SELECT value FROM state_metadata WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )?;
        Ok(version)
    }

    pub fn create_mission(
        &mut self,
        mission: MissionRecord,
        receipt: NewMissionProjectionReceipt,
    ) -> Result<MissionProjectionWrite> {
        Ok(self.create_mission_inner(mission, receipt, None)?.write)
    }

    /// Persist an initial mission projection, receipt, and idempotency binding atomically.
    /// A replay with the same key and fingerprint returns the original write; a changed
    /// fingerprint for an existing key fails before any new receipt can be appended.
    pub fn create_mission_idempotent(
        &mut self,
        mission: MissionRecord,
        receipt: NewMissionProjectionReceipt,
        idempotency: MissionCreateIdempotency,
    ) -> Result<IdempotentMissionWrite> {
        validate_mission_create_idempotency(&idempotency.key, &idempotency.request_fingerprint)?;
        let outcome = self.create_mission_inner(
            mission,
            receipt,
            Some((&idempotency.key, &idempotency.request_fingerprint)),
        )?;
        Ok(outcome)
    }

    /// Replace a mission projection and append one receipt in the same transaction.
    pub fn update_mission(
        &mut self,
        expected_revision: u64,
        mission: MissionRecord,
        receipt: NewMissionProjectionReceipt,
    ) -> Result<MissionProjectionWrite> {
        Ok(self
            .update_mission_with_events(expected_revision, mission, vec![receipt], None)?
            .write)
    }

    /// Replace a mission projection and append one or more hash-linked receipts.
    pub fn update_mission_with_events(
        &mut self,
        expected_revision: u64,
        mission: MissionRecord,
        receipts: Vec<NewMissionProjectionReceipt>,
        idempotency: Option<MissionMutationIdempotency>,
    ) -> Result<IdempotentMissionWrite> {
        if let Some(idempotency) = &idempotency {
            validate_mission_create_idempotency(
                &idempotency.key,
                &idempotency.request_fingerprint,
            )?;
        }
        mission
            .validate()
            .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
        if mission.revision != expected_revision.saturating_add(1) {
            return Err(StateError::InvalidMissionProjection(
                "updated mission revision must be expected_revision + 1".to_owned(),
            ));
        }
        if receipts.is_empty() {
            return Err(StateError::InvalidMissionProjection(
                "mission update requires at least one lifecycle receipt".to_owned(),
            ));
        }
        for receipt in &receipts {
            receipt
                .event
                .validate()
                .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
            if !matches!(
                receipt.event.source.kind,
                EventSourceKind::VerifiedAttestation
                    | EventSourceKind::OperatorCommand
                    | EventSourceKind::KernelObserved
                    | EventSourceKind::DriverReported
            ) {
                return Err(StateError::InvalidMissionProjection(
                    "update receipt must be authoritative".to_owned(),
                ));
            }
        }

        let mission_id = mission.mission_id.to_string();
        let projection_json = serde_json::to_string(&mission)
            .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;

        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(idempotency) = &idempotency {
            let existing: Option<(String, String)> = tx
                .query_row(
                    "SELECT request_fingerprint, result_json FROM mission_mutations WHERE idempotency_key = ?1 LIMIT 1",
                    [&idempotency.key],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((stored_fingerprint, stored_result)) = existing {
                if stored_fingerprint != idempotency.request_fingerprint {
                    return Err(StateError::MissionIdempotencyConflict {
                        key: idempotency.key.clone(),
                    });
                }
                let projection =
                    load_mission_projection_tx(&tx, &mission_id)?.ok_or_else(|| {
                        StateError::InvalidMissionProjection(
                            "idempotency binding references a missing mission projection"
                                .to_owned(),
                        )
                    })?;
                let receipt =
                    load_audit_record_tx(&tx, &projection.audit_entry_id)?.ok_or_else(|| {
                        StateError::InvalidMissionProjection(
                            "idempotency binding references a missing mission receipt".to_owned(),
                        )
                    })?;
                tx.commit()?;
                return Ok(IdempotentMissionWrite {
                    write: MissionProjectionWrite {
                        projection,
                        receipt,
                    },
                    replayed: true,
                    replayed_result_json: Some(stored_result),
                });
            }
        }

        let mut history = load_lifecycle_history_tx(&tx, &mission_id)?;
        history.extend(receipts.iter().map(|receipt| receipt.event.clone()));
        validate_history(&mission, &history)
            .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;

        let updated = tx.execute(
            "UPDATE missions SET revision = ?1, phase = ?2, updated_at = ?3, record_json = ?4, \
             receipt_entry_id = ?5, receipt_entry_hash = ?6 \
             WHERE mission_id = ?7 AND revision = ?8",
            params![
                i64::try_from(mission.revision).map_err(|_| {
                    StateError::InvalidMissionProjection(
                        "mission revision exceeds SQLite integer range".to_owned(),
                    )
                })?,
                mission_phase_name(mission.phase),
                mission
                    .updated_at
                    .to_rfc3339_opts(SecondsFormat::Millis, true),
                &projection_json,
                "",
                "",
                &mission_id,
                i64::try_from(expected_revision).map_err(|_| {
                    StateError::InvalidMissionProjection(
                        "expected revision exceeds SQLite integer range".to_owned(),
                    )
                })?,
            ],
        )?;
        if updated != 1 {
            return Err(StateError::InvalidMissionProjection(
                "mission update did not match the expected revision".to_owned(),
            ));
        }

        let mut last_audit = None;
        for receipt in receipts {
            let event_json = serde_json::to_value(&receipt.event)
                .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
            let audit_input = NewAuditRecord {
                entry_id: receipt.event.event_id.to_string(),
                session_id: mission_id.clone(),
                timestamp: receipt
                    .event
                    .recorded_at
                    .to_rfc3339_opts(SecondsFormat::Millis, true),
                kind: AuditRecordKind::MissionEvent,
                action: receipt.event.event_type.clone(),
                severity: AuditSeverity::Notice,
                envelope: receipt.envelope,
                payload: event_json,
                verification: receipt.verification,
            };
            last_audit = Some(append_audit_record_tx(&tx, audit_input)?);
        }
        let audit = last_audit.ok_or_else(|| {
            StateError::InvalidMissionProjection("mission update produced no receipts".to_owned())
        })?;
        tx.execute(
            "UPDATE missions SET receipt_entry_id = ?1, receipt_entry_hash = ?2 WHERE mission_id = ?3",
            params![&audit.entry_id, &audit.entry_hash, &mission_id],
        )?;
        if let Some(idempotency) = &idempotency {
            let updated = tx.execute(
                "UPDATE mission_mutations SET result_json = ?1 WHERE idempotency_key = ?2 AND request_fingerprint = ?3",
                params![&idempotency.result_json, &idempotency.key, &idempotency.request_fingerprint],
            )?;
            if updated == 0 {
                tx.execute(
                    "INSERT INTO mission_mutations(idempotency_key, request_fingerprint, mission_id, operation, result_json) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        &idempotency.key,
                        &idempotency.request_fingerprint,
                        &mission_id,
                        &idempotency.operation,
                        &idempotency.result_json,
                    ],
                )?;
            }
        }
        tx.commit()?;
        Ok(IdempotentMissionWrite {
            write: MissionProjectionWrite {
                projection: StoredMissionProjection {
                    mission,
                    audit_entry_id: audit.entry_id.clone(),
                    audit_entry_hash: audit.entry_hash.clone(),
                },
                receipt: audit,
            },
            replayed: false,
            replayed_result_json: None,
        })
    }

    fn create_mission_inner(
        &mut self,
        mission: MissionRecord,
        receipt: NewMissionProjectionReceipt,
        idempotency: Option<(&str, &str)>,
    ) -> Result<IdempotentMissionWrite> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some((idempotency_key, request_fingerprint)) = idempotency {
            let existing: Option<(String, String)> = tx
                .query_row(
                    "SELECT request_fingerprint, mission_id FROM mission_requests WHERE idempotency_key = ?1 LIMIT 1",
                    (idempotency_key,),
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((stored_fingerprint, stored_mission_id)) = existing {
                if stored_fingerprint != request_fingerprint {
                    return Err(StateError::MissionIdempotencyConflict {
                        key: idempotency_key.to_owned(),
                    });
                }
                let projection =
                    load_mission_projection_tx(&tx, &stored_mission_id)?.ok_or_else(|| {
                        StateError::InvalidMissionProjection(
                            "idempotency binding references a missing mission projection"
                                .to_owned(),
                        )
                    })?;
                let receipt =
                    load_audit_record_tx(&tx, &projection.audit_entry_id)?.ok_or_else(|| {
                        StateError::InvalidMissionProjection(
                            "idempotency binding references a missing mission receipt".to_owned(),
                        )
                    })?;
                validate_projection_receipt_binding(&projection, &receipt)?;
                tx.commit()?;
                return Ok(IdempotentMissionWrite {
                    write: MissionProjectionWrite {
                        projection,
                        receipt,
                    },
                    replayed: true,
                    replayed_result_json: None,
                });
            }
        }
        mission
            .validate()
            .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
        if mission.phase != MissionPhase::Created || mission.revision != 0 {
            return Err(StateError::InvalidMissionProjection(
                "initial projection must be a revision-zero created mission".to_owned(),
            ));
        }
        if !matches!(
            &receipt.event.payload,
            LifecyclePayload::MissionCreated { revision: 0 }
        ) || !matches!(
            receipt.event.source.kind,
            EventSourceKind::VerifiedAttestation | EventSourceKind::OperatorCommand
        ) {
            return Err(StateError::InvalidMissionProjection(
                "initial receipt must be an authoritative mission_created event at revision zero"
                    .to_owned(),
            ));
        }
        validate_history(&mission, std::slice::from_ref(&receipt.event))
            .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;

        let mission_id = mission.mission_id.to_string();
        let projection_json = serde_json::to_string(&mission)
            .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
        let event_json = serde_json::to_value(&receipt.event)
            .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
        let audit_input = NewAuditRecord {
            entry_id: receipt.event.event_id.to_string(),
            session_id: mission_id.clone(),
            timestamp: receipt
                .event
                .recorded_at
                .to_rfc3339_opts(SecondsFormat::Millis, true),
            kind: AuditRecordKind::MissionEvent,
            action: receipt.event.event_type.clone(),
            severity: AuditSeverity::Notice,
            envelope: receipt.envelope,
            payload: event_json,
            verification: receipt.verification,
        };

        let chain_exists: i64 = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM audit_records WHERE session_id = ?1)",
            (&mission_id,),
            |row| row.get(0),
        )?;
        if chain_exists != 0 {
            return Err(StateError::InvalidMissionProjection(
                "mission creation requires an empty receipt chain".to_owned(),
            ));
        }
        let audit = append_audit_record_tx(&tx, audit_input)?;
        tx.execute(
            "INSERT INTO missions(mission_id, mission_schema, revision, goal, policy_id, phase, operating_role, operating_scope, created_at, updated_at, evidence_chain_id, record_json, receipt_entry_id, receipt_entry_hash) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                &mission_id,
                &mission.mission_schema,
                i64::try_from(mission.revision).map_err(|_| {
                    StateError::InvalidMissionProjection(
                        "mission revision exceeds SQLite integer range".to_owned(),
                    )
                })?,
                &mission.goal,
                &mission.policy_id,
                mission_phase_name(mission.phase),
                &mission.operating_role.role,
                &mission.operating_role.scope,
                mission
                    .created_at
                    .to_rfc3339_opts(SecondsFormat::Millis, true),
                mission
                    .updated_at
                    .to_rfc3339_opts(SecondsFormat::Millis, true),
                mission.evidence_chain_id.to_string(),
                &projection_json,
                &audit.entry_id,
                &audit.entry_hash,
            ],
        )?;
        if let Some((idempotency_key, request_fingerprint)) = idempotency {
            tx.execute(
                "INSERT INTO mission_requests(idempotency_key, request_fingerprint, mission_id) VALUES (?1, ?2, ?3)",
                params![idempotency_key, request_fingerprint, &mission_id],
            )?;
        }
        tx.commit()?;

        Ok(IdempotentMissionWrite {
            write: MissionProjectionWrite {
                projection: StoredMissionProjection {
                    mission,
                    audit_entry_id: audit.entry_id.clone(),
                    audit_entry_hash: audit.entry_hash.clone(),
                },
                receipt: audit,
            },
            replayed: false,
            replayed_result_json: None,
        })
    }

    pub fn mission(&self, mission_id: &str) -> Result<Option<StoredMissionProjection>> {
        load_mission_projection(&self.connection, mission_id)
    }

    /// List projections only within the caller's operating role and scope.
    /// Ordering by `(created_at, mission_id)` gives each returned cursor a stable boundary.
    pub fn list_missions(&self, request: MissionListFilter) -> Result<MissionListPage> {
        validate_mission_list_request(&request)?;
        let cursor = request
            .cursor
            .as_deref()
            .map(|value| parse_mission_list_cursor(value, &request))
            .transpose()?;
        let snapshot_sequence = match cursor.as_ref() {
            Some(cursor) => cursor.snapshot_sequence,
            None => match mission_list_snapshot_boundary(&self.connection, &request)? {
                Some(snapshot_sequence) => snapshot_sequence,
                None => {
                    return Ok(MissionListPage {
                        projections: Vec::new(),
                        next_cursor: None,
                    });
                }
            },
        };

        let mut sql = String::from(
            "SELECT mission_id, revision, phase, operating_role, operating_scope, record_json, receipt_entry_id, receipt_entry_hash \
             FROM missions WHERE operating_role = ? AND operating_scope = ?",
        );
        let mut values = vec![
            rusqlite::types::Value::Text(request.operating_role.clone()),
            rusqlite::types::Value::Text(request.operating_scope.clone()),
        ];
        if !request.phases.is_empty() {
            sql.push_str(" AND phase IN (");
            for (index, phase) in request.phases.iter().enumerate() {
                if index != 0 {
                    sql.push_str(", ");
                }
                sql.push('?');
                values.push(rusqlite::types::Value::Text(
                    mission_phase_name(*phase).to_owned(),
                ));
            }
            sql.push(')');
        }
        if let Some(cursor) = cursor {
            sql.push_str(" AND (created_at > ? OR (created_at = ? AND mission_id > ?))");
            values.push(rusqlite::types::Value::Text(cursor.created_at.clone()));
            values.push(rusqlite::types::Value::Text(cursor.created_at));
            values.push(rusqlite::types::Value::Text(cursor.mission_id));
        }
        sql.push_str(" AND rowid <= ?");
        values.push(rusqlite::types::Value::Integer(snapshot_sequence));
        sql.push_str(" ORDER BY created_at ASC, mission_id ASC LIMIT ?");
        values.push(rusqlite::types::Value::Integer(
            i64::from(request.limit) + 1,
        ));

        let mut statement = self.connection.prepare(&sql)?;
        let rows = statement.query_map(params_from_iter(values), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
            ))
        })?;
        let mut missions = rows
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .map(
                |(
                    mission_id,
                    revision,
                    phase,
                    operating_role,
                    operating_scope,
                    projection_json,
                    audit_entry_id,
                    audit_entry_hash,
                )| {
                    stored_mission_projection_from_row(
                        &mission_id,
                        revision,
                        &phase,
                        &projection_json,
                        audit_entry_id,
                        audit_entry_hash,
                        Some((&operating_role, &operating_scope)),
                    )
                },
            )
            .collect::<Result<Vec<_>>>()?;
        let has_more = missions.len() > usize::from(request.limit);
        if has_more {
            let _extra = missions.pop().expect("length was checked");
            // The next request starts after the last item returned, not the look-ahead row.
            let next_cursor = missions
                .last()
                .map(|last| format_mission_list_cursor(&last.mission, &request, snapshot_sequence))
                .expect("a look-ahead implies a returned item");
            return Ok(MissionListPage {
                projections: missions,
                next_cursor: Some(next_cursor),
            });
        }
        Ok(MissionListPage {
            projections: missions,
            next_cursor: None,
        })
    }

    pub fn append_audit_record(&mut self, record: NewAuditRecord) -> Result<AuditRecord> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let record = append_audit_record_tx(&tx, record)?;
        tx.commit()?;
        Ok(record)
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
            "SELECT records.session_id \
             FROM audit_records AS records \
             WHERE NOT EXISTS (SELECT 1 FROM missions WHERE missions.mission_id = records.session_id) \
             GROUP BY records.session_id \
             HAVING MAX(records.timestamp) < ?1 \
             ORDER BY MAX(records.timestamp) ASC",
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
        let hot_summary = load_session_chain_summary_tx(&tx, session_id)?
            .ok_or_else(|| StateError::SessionNotFound(session_id.to_owned()))?;
        if hot_summary.record_count != metadata.record_count
            || hot_summary.terminal_entry_hash != metadata.terminal_entry_hash
            || hot_summary.latest_record_timestamp != metadata.latest_record_timestamp
        {
            return Err(StateError::EvictionConflict {
                session_id: session_id.to_owned(),
            });
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct SessionChainSummary {
    record_count: usize,
    terminal_entry_hash: String,
    latest_record_timestamp: String,
}

impl SessionExporter<'_> {
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

impl SessionEvictor<'_> {
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

fn load_mission_projection(
    connection: &Connection,
    mission_id: &str,
) -> Result<Option<StoredMissionProjection>> {
    let stored = connection
        .query_row(
            "SELECT revision, phase, record_json, receipt_entry_id, receipt_entry_hash \
             FROM missions WHERE mission_id = ?1 LIMIT 1",
            (mission_id,),
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()?;
    stored
        .map(
            |(revision, phase, projection_json, audit_entry_id, audit_entry_hash)| {
                stored_mission_projection_from_row(
                    mission_id,
                    revision,
                    &phase,
                    &projection_json,
                    audit_entry_id,
                    audit_entry_hash,
                    None,
                )
            },
        )
        .transpose()
}

fn load_mission_projection_tx(
    tx: &Transaction<'_>,
    mission_id: &str,
) -> Result<Option<StoredMissionProjection>> {
    let stored = tx
        .query_row(
            "SELECT revision, phase, record_json, receipt_entry_id, receipt_entry_hash \
             FROM missions WHERE mission_id = ?1 LIMIT 1",
            (mission_id,),
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()?;
    stored
        .map(
            |(revision, phase, projection_json, audit_entry_id, audit_entry_hash)| {
                stored_mission_projection_from_row(
                    mission_id,
                    revision,
                    &phase,
                    &projection_json,
                    audit_entry_id,
                    audit_entry_hash,
                    None,
                )
            },
        )
        .transpose()
}

fn stored_mission_projection_from_row(
    mission_id: &str,
    revision: i64,
    phase: &str,
    projection_json: &str,
    audit_entry_id: String,
    audit_entry_hash: String,
    operating_scope: Option<(&str, &str)>,
) -> Result<StoredMissionProjection> {
    let mission: MissionRecord = serde_json::from_str(projection_json)
        .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
    mission
        .validate()
        .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
    if mission.mission_id.to_string() != mission_id
        || i64::try_from(mission.revision).ok() != Some(revision)
        || mission_phase_name(mission.phase) != phase
    {
        return Err(StateError::InvalidMissionProjection(
            "indexed mission fields do not match stored projection".to_owned(),
        ));
    }
    if let Some((role, scope)) = operating_scope {
        if mission.operating_role.role != role || mission.operating_role.scope != scope {
            return Err(StateError::InvalidMissionProjection(
                "caller-scope fields do not match stored projection".to_owned(),
            ));
        }
    }
    Ok(StoredMissionProjection {
        mission,
        audit_entry_id,
        audit_entry_hash,
    })
}

fn load_audit_record_tx(tx: &Transaction<'_>, entry_id: &str) -> Result<Option<AuditRecord>> {
    tx.query_row(
        "SELECT entry_id, session_id, timestamp, record_kind, action, severity, conversation_id, turn_id, action_id, causation_id, correlation_id, external_ref, trust_ref, gate_ref, payload_json, verification_json, prev_hash, entry_hash FROM audit_records WHERE entry_id = ?1 LIMIT 1",
        (entry_id,),
        |row| {
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
        },
    )
    .optional()
    .map_err(StateError::from)
}

fn validate_projection_receipt_binding(
    projection: &StoredMissionProjection,
    receipt: &AuditRecord,
) -> Result<()> {
    receipt
        .validate()
        .map_err(|err| StateError::InvalidAuditRecord(err.to_owned()))?;
    if receipt.entry_id != projection.audit_entry_id
        || receipt.entry_hash != projection.audit_entry_hash
        || receipt.session_id != projection.mission.mission_id.to_string()
        || receipt.kind != AuditRecordKind::MissionEvent
    {
        return Err(StateError::InvalidMissionProjection(
            "mission projection receipt binding is invalid".to_owned(),
        ));
    }
    Ok(())
}

fn load_lifecycle_history_tx(
    tx: &Transaction<'_>,
    mission_id: &str,
) -> Result<Vec<LifecycleEvent>> {
    let mut stmt = tx.prepare(
        "SELECT payload_json FROM audit_records WHERE session_id = ?1 AND record_kind = ?2 ORDER BY chain_index ASC",
    )?;
    let rows = stmt.query_map(params![mission_id, "mission_event"], |row| {
        row.get::<_, String>(0)
    })?;
    let mut events = Vec::new();
    for payload in rows {
        let payload = payload?;
        let value: serde_json::Value = serde_json::from_str(&payload)?;
        let event: LifecycleEvent = serde_json::from_value(value)
            .map_err(|err| StateError::InvalidMissionProjection(err.to_string()))?;
        events.push(event);
    }
    Ok(events)
}

fn validate_mission_create_idempotency(key: &str, fingerprint: &str) -> Result<()> {
    let valid_key = (16..=256).contains(&key.len())
        && key.bytes().enumerate().all(|(index, byte)| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' => true,
            b'.' | b'_' | b':' | b'-' => index > 0,
            _ => false,
        });
    if !valid_key {
        return Err(StateError::InvalidMissionProjection(
            "idempotency key does not match the mission contract".to_owned(),
        ));
    }
    if fingerprint.len() != 64 || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(StateError::InvalidMissionProjection(
            "request fingerprint must be a 64-character hexadecimal SHA-256 digest".to_owned(),
        ));
    }
    Ok(())
}

fn validate_mission_list_request(request: &MissionListFilter) -> Result<()> {
    if request.operating_role.is_empty() || request.operating_scope.is_empty() {
        return Err(StateError::InvalidMissionList(
            "operating role and scope are required".to_owned(),
        ));
    }
    if request.limit == 0 || request.limit > MAX_MISSION_LIST_LIMIT {
        return Err(StateError::InvalidMissionList(format!(
            "limit must be in 1..={MAX_MISSION_LIST_LIMIT}"
        )));
    }
    if request.phases.len() > 10 {
        return Err(StateError::InvalidMissionList(
            "phases must contain at most 10 values".to_owned(),
        ));
    }
    for (index, phase) in request.phases.iter().enumerate() {
        if request.phases[..index].contains(phase) {
            return Err(StateError::InvalidMissionList(
                "phases must be unique".to_owned(),
            ));
        }
    }
    request
        .cursor
        .as_deref()
        .map(|value| parse_mission_list_cursor(value, request))
        .transpose()?;
    Ok(())
}

fn mission_list_snapshot_boundary(
    connection: &Connection,
    request: &MissionListFilter,
) -> Result<Option<i64>> {
    let mut sql = String::from(
        "SELECT MAX(rowid) FROM missions \
         WHERE operating_role = ? AND operating_scope = ?",
    );
    let mut values = vec![
        rusqlite::types::Value::Text(request.operating_role.clone()),
        rusqlite::types::Value::Text(request.operating_scope.clone()),
    ];
    if !request.phases.is_empty() {
        sql.push_str(" AND phase IN (");
        for (index, phase) in request.phases.iter().enumerate() {
            if index != 0 {
                sql.push_str(", ");
            }
            sql.push('?');
            values.push(rusqlite::types::Value::Text(
                mission_phase_name(*phase).to_owned(),
            ));
        }
        sql.push(')');
    }
    connection
        .query_row(&sql, params_from_iter(values), |row| row.get(0))
        .map_err(StateError::from)
}

fn format_mission_list_cursor(
    mission: &MissionRecord,
    request: &MissionListFilter,
    snapshot_sequence: i64,
) -> String {
    let cursor = MissionListCursor {
        version: MISSION_LIST_CURSOR_VERSION,
        created_at: mission
            .created_at
            .to_rfc3339_opts(SecondsFormat::Millis, true),
        mission_id: mission.mission_id.to_string(),
        snapshot_sequence,
        operating_role: request.operating_role.clone(),
        operating_scope: request.operating_scope.clone(),
        phases: canonical_mission_phase_names(&request.phases),
    };
    URL_SAFE_NO_PAD.encode(
        serde_json::to_vec(&cursor).expect("mission list cursor contains only serializable fields"),
    )
}

fn parse_mission_list_cursor(
    value: &str,
    request: &MissionListFilter,
) -> Result<MissionListCursor> {
    if value.is_empty() || value.len() > 1024 {
        return Err(StateError::InvalidMissionList(
            "cursor must contain 1..=1024 characters".to_owned(),
        ));
    }
    let encoded = URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| StateError::InvalidMissionList("cursor encoding is invalid".to_owned()))?;
    let cursor: MissionListCursor = serde_json::from_slice(&encoded)
        .map_err(|_| StateError::InvalidMissionList("cursor payload is invalid".to_owned()))?;
    if cursor.version != MISSION_LIST_CURSOR_VERSION {
        return Err(StateError::InvalidMissionList(
            "cursor version is unsupported".to_owned(),
        ));
    }
    if !is_canonical_mission_cursor_position(&cursor.created_at, &cursor.mission_id) {
        return Err(StateError::InvalidMissionList(
            "cursor must contain a canonical millisecond timestamp and UUIDv4 mission ID"
                .to_owned(),
        ));
    }
    if cursor.snapshot_sequence <= 0 {
        return Err(StateError::InvalidMissionList(
            "cursor snapshot sequence must be positive".to_owned(),
        ));
    }
    if cursor.operating_role != request.operating_role
        || cursor.operating_scope != request.operating_scope
        || cursor.phases != canonical_mission_phase_names(&request.phases)
    {
        return Err(StateError::InvalidMissionList(
            "cursor does not match the mission list filters".to_owned(),
        ));
    }
    Ok(cursor)
}

fn is_canonical_mission_cursor_position(created_at: &str, mission_id: &str) -> bool {
    DateTime::parse_from_rfc3339(created_at).is_ok_and(|timestamp| {
        timestamp
            .with_timezone(&Utc)
            .to_rfc3339_opts(SecondsFormat::Millis, true)
            == created_at
    }) && is_uuid_v4(mission_id)
}

fn canonical_mission_phase_names(phases: &[MissionPhase]) -> Vec<String> {
    let mut names = phases
        .iter()
        .map(|phase| mission_phase_name(*phase).to_owned())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names
}

fn is_uuid_v4(value: &str) -> bool {
    value.len() == 36
        && value.as_bytes().get(8) == Some(&b'-')
        && value.as_bytes().get(13) == Some(&b'-')
        && value.as_bytes().get(18) == Some(&b'-')
        && value.as_bytes().get(23) == Some(&b'-')
        && value.as_bytes().get(14) == Some(&b'4')
        && matches!(value.as_bytes().get(19), Some(b'8' | b'9' | b'a' | b'b'))
        && value
            .bytes()
            .enumerate()
            .all(|(index, byte)| matches!(index, 8 | 13 | 18 | 23) || byte.is_ascii_hexdigit())
}

fn append_audit_record_tx(tx: &Transaction<'_>, record: NewAuditRecord) -> Result<AuditRecord> {
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
    Ok(record)
}

const fn mission_phase_name(phase: MissionPhase) -> &'static str {
    match phase {
        MissionPhase::Created => "created",
        MissionPhase::Active => "active",
        MissionPhase::Waiting => "waiting",
        MissionPhase::RecoveryPending => "recovery_pending",
        MissionPhase::Suspended => "suspended",
        MissionPhase::Completed => "completed",
        MissionPhase::Cancelled => "cancelled",
        MissionPhase::Failed => "failed",
        MissionPhase::DeadlineExceeded => "deadline_exceeded",
        MissionPhase::BudgetExhausted => "budget_exhausted",
    }
}

fn audit_record_kind_name(kind: AuditRecordKind) -> &'static str {
    match kind {
        AuditRecordKind::Effect => "effect",
        AuditRecordKind::Outcome => "outcome",
        AuditRecordKind::GateDecision => "gate_decision",
        AuditRecordKind::Verification => "verification",
        AuditRecordKind::SessionAttestation => "session_attestation",
        AuditRecordKind::MissionEvent => "mission_event",
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
        "mission_event" => Ok(AuditRecordKind::MissionEvent),
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

fn load_session_chain_summary_tx(
    tx: &rusqlite::Transaction<'_>,
    session_id: &str,
) -> Result<Option<SessionChainSummary>> {
    tx.query_row(
        "SELECT 
            (SELECT COUNT(*) FROM audit_records WHERE session_id = ?1),
            tip.timestamp,
            tip.entry_hash
         FROM audit_records AS tip
         WHERE tip.session_id = ?1
           AND tip.chain_index = (SELECT MAX(chain_index) FROM audit_records WHERE session_id = ?1)",
        (session_id,),
        |row| {
            Ok(SessionChainSummary {
                record_count: row.get::<_, i64>(0)? as usize,
                latest_record_timestamp: row.get(1)?,
                terminal_entry_hash: row.get(2)?,
            })
        },
    )
    .optional()
    .map_err(StateError::from)
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

    fn created_mission() -> MissionRecord {
        serde_json::from_value(serde_json::json!({
            "mission_schema": lanyte_mission::MISSION_RECORD_SCHEMA,
            "mission_id": TEST_SESSION_ID,
            "revision": 0,
            "goal": "Persist one mission projection",
            "policy_id": "policy.default",
            "created_at": "2026-08-17T20:00:00Z",
            "updated_at": "2026-08-17T20:00:00Z",
            "initiator": {
                "kind": "attested_session",
                "subject": "operator-1",
                "role": "entarch",
                "scope": "lanytehq",
                "attestation": {
                    "issuer": "lanyte-attest",
                    "session_id": "550e8400-e29b-41d4-a716-446655440003",
                    "jti": "550e8400-e29b-41d4-a716-446655440004",
                    "context_sha256": "a".repeat(64),
                    "token_sha256": "b".repeat(64),
                    "verification_policy_sha256": "c".repeat(64),
                    "trust_ref": "attestations/1"
                }
            },
            "authorizer": null,
            "authorization_ref": null,
            "supervisor": {
                "kind": "service",
                "subject": "lanyte",
                "role": null,
                "scope": null,
                "attestation": null
            },
            "operating_role": {
                "role": "entarch",
                "scope": "lanytehq"
            },
            "phase": "created",
            "terminal_reason": null,
            "deadline_at": null,
            "lease_policy": {
                "enabled": false,
                "lease_seconds": null,
                "deadman_seconds": null
            },
            "budget_policy": {
                "wall_clock_seconds": null,
                "token_limit": null,
                "cost_micros": null,
                "action_limit": null
            },
            "harness_selection": null,
            "recovery_policy": "ask_operator",
            "recovery_point_ref": null,
            "attempts": [],
            "current_attempt_id": null,
            "evidence_chain_id": TEST_SESSION_ID,
            "terminal_entry_hash": null
        }))
        .expect("mission fixture should deserialize")
    }

    fn created_mission_receipt(entry_id: &str) -> NewMissionProjectionReceipt {
        let event = serde_json::from_value(serde_json::json!({
            "event_schema": lanyte_mission::LIFECYCLE_EVENT_SCHEMA,
            "event_id": entry_id,
            "mission_id": TEST_SESSION_ID,
            "sequence": 1,
            "previous_entry_hash": null,
            "entry_hash": "d".repeat(64),
            "occurred_at": "2026-08-17T20:00:00Z",
            "recorded_at": "2026-08-17T20:00:00Z",
            "event_type": "mission_created",
            "source": {
                "kind": "verified_attestation",
                "subject": "operator-1",
                "producer_version": "0.1.0",
                "assurance": "resource_attested",
                "evidence_ref": "attestations/1"
            },
            "payload": {
                "type": "mission_created",
                "revision": 0
            }
        }))
        .expect("lifecycle fixture should deserialize");
        NewMissionProjectionReceipt {
            event,
            envelope: AuditEnvelopeRef {
                action_id: Some(entry_id.to_owned()),
                correlation_id: Some(TEST_SESSION_ID.to_owned()),
                trust_ref: Some("attestations/1".to_owned()),
                ..AuditEnvelopeRef::default()
            },
            verification: None,
        }
    }

    fn created_mission_for(
        mission_id: &str,
        operating_role: &str,
        operating_scope: &str,
        timestamp: &str,
    ) -> MissionRecord {
        let mut value = serde_json::to_value(created_mission()).expect("fixture should encode");
        value["mission_id"] = serde_json::json!(mission_id);
        value["evidence_chain_id"] = serde_json::json!(mission_id);
        value["created_at"] = serde_json::json!(timestamp);
        value["updated_at"] = serde_json::json!(timestamp);
        value["operating_role"]["role"] = serde_json::json!(operating_role);
        value["operating_role"]["scope"] = serde_json::json!(operating_scope);
        serde_json::from_value(value).expect("mission fixture should deserialize")
    }

    fn created_mission_receipt_for(
        mission_id: &str,
        entry_id: &str,
        timestamp: &str,
    ) -> NewMissionProjectionReceipt {
        let receipt = created_mission_receipt(entry_id);
        let mut event = serde_json::to_value(receipt.event).expect("fixture should encode");
        event["mission_id"] = serde_json::json!(mission_id);
        event["occurred_at"] = serde_json::json!(timestamp);
        event["recorded_at"] = serde_json::json!(timestamp);
        NewMissionProjectionReceipt {
            event: serde_json::from_value(event).expect("receipt fixture should deserialize"),
            envelope: AuditEnvelopeRef {
                action_id: Some(entry_id.to_owned()),
                correlation_id: Some(mission_id.to_owned()),
                trust_ref: Some("attestations/1".to_owned()),
                ..AuditEnvelopeRef::default()
            },
            verification: None,
        }
    }

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
        assert_eq!(store.schema_version().expect("schema version query"), 6);
    }

    #[test]
    fn reopen_is_idempotent_and_does_not_rerun_migrations() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);

        let store_a = StateStore::open(paths.clone()).expect("state store should open");
        assert_eq!(store_a.schema_version().expect("schema version query"), 6);

        let store_b = StateStore::open(paths).expect("state store should open again");
        assert_eq!(store_b.schema_version().expect("schema version query"), 6);
    }

    #[test]
    fn version_three_store_upgrades_to_mission_projection_schema() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        drop(StateStore::open(paths.clone()).expect("state store should open"));

        let connection =
            Connection::open(paths.hot_db_path()).expect("database should reopen directly");
        connection
            .execute_batch(
                "DROP TABLE missions; \
                 UPDATE state_metadata SET value = 3 WHERE key = 'schema_version';",
            )
            .expect("version-three fixture should prepare");
        drop(connection);

        let mut store = StateStore::open(paths).expect("version-three store should upgrade");
        assert_eq!(store.schema_version().expect("schema version query"), 6);
        store
            .create_mission(created_mission(), created_mission_receipt(TEST_ENTRY_ID_A))
            .expect("upgraded store should persist missions");
    }

    #[test]
    fn version_four_store_upgrades_to_mission_request_schema() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        drop(StateStore::open(paths.clone()).expect("state store should open"));

        let connection =
            Connection::open(paths.hot_db_path()).expect("database should reopen directly");
        connection
            .execute_batch(
                "DROP TABLE mission_requests; \
                 UPDATE state_metadata SET value = 4 WHERE key = 'schema_version';",
            )
            .expect("version-four fixture should prepare");
        drop(connection);

        let mut store = StateStore::open(paths).expect("version-four store should upgrade");
        assert_eq!(store.schema_version().expect("schema version query"), 6);
        store
            .create_mission_idempotent(
                created_mission(),
                created_mission_receipt(TEST_ENTRY_ID_A),
                MissionCreateIdempotency {
                    key: "mission-create-upgrade-key".to_owned(),
                    request_fingerprint: "a".repeat(64),
                },
            )
            .expect("upgraded store should persist idempotent missions");
    }

    #[test]
    fn mission_projection_and_receipt_commit_atomically() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");
        let mission = created_mission();

        let write = store
            .create_mission(mission.clone(), created_mission_receipt(TEST_ENTRY_ID_A))
            .expect("mission and receipt should commit");

        assert_eq!(write.projection.mission, mission);
        assert_eq!(write.receipt.kind, AuditRecordKind::MissionEvent);
        assert_eq!(write.receipt.session_id, TEST_SESSION_ID);
        assert_eq!(write.projection.audit_entry_hash, write.receipt.entry_hash);

        let stored = store
            .mission(TEST_SESSION_ID)
            .expect("mission query should succeed")
            .expect("mission should exist");
        assert_eq!(stored, write.projection);
        let records = store
            .audit_records(TEST_SESSION_ID)
            .expect("mission receipt chain should load");
        assert_eq!(records, vec![write.receipt]);
    }

    #[test]
    fn duplicate_projection_does_not_append_a_receipt() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");
        let mission = created_mission();
        store
            .create_mission(mission.clone(), created_mission_receipt(TEST_ENTRY_ID_A))
            .expect("first mission should commit");

        let result = store.create_mission(mission, created_mission_receipt(TEST_ENTRY_ID_B));
        assert!(result.is_err(), "duplicate mission must fail");

        let records = store
            .audit_records(TEST_SESSION_ID)
            .expect("mission receipt chain should remain valid");
        assert_eq!(records.len(), 1, "failed projection must roll back receipt");
        assert_eq!(records[0].entry_id, TEST_ENTRY_ID_A);
    }

    #[test]
    fn idempotent_create_replays_original_write_and_rejects_conflicts() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");
        let idempotency = MissionCreateIdempotency {
            key: "mission-create-replay-key".to_owned(),
            request_fingerprint: "a".repeat(64),
        };

        let initial = store
            .create_mission_idempotent(
                created_mission(),
                created_mission_receipt(TEST_ENTRY_ID_A),
                idempotency.clone(),
            )
            .expect("initial create should commit");
        assert!(!initial.replayed);

        let replay = store
            .create_mission_idempotent(
                created_mission(),
                created_mission_receipt(TEST_ENTRY_ID_B),
                idempotency.clone(),
            )
            .expect("matching replay should return original write");
        assert!(replay.replayed);
        assert_eq!(replay.write, initial.write);
        assert_eq!(
            store
                .audit_records(TEST_SESSION_ID)
                .expect("mission receipt chain should load")
                .len(),
            1,
            "matching replay must not append another receipt"
        );

        let conflict = store.create_mission_idempotent(
            created_mission(),
            created_mission_receipt(TEST_ENTRY_ID_B),
            MissionCreateIdempotency {
                key: idempotency.key,
                request_fingerprint: "b".repeat(64),
            },
        );
        assert!(matches!(
            conflict,
            Err(StateError::MissionIdempotencyConflict { .. })
        ));
        assert_eq!(
            store
                .audit_records(TEST_SESSION_ID)
                .expect("mission receipt chain should remain valid")
                .len(),
            1,
            "conflict must not append another receipt"
        );
    }

    #[test]
    fn list_missions_is_caller_scoped_filtered_and_cursor_stable() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");
        let first_id = "550e8400-e29b-41d4-a716-446655440010";
        let second_id = "550e8400-e29b-41d4-a716-446655440011";
        let other_id = "550e8400-e29b-41d4-a716-446655440012";
        let inserted_after_snapshot_id = "00000000-0000-4000-8000-000000000013";
        store
            .create_mission(
                created_mission_for(first_id, "entarch", "lanytehq", "2026-08-17T20:00:00Z"),
                created_mission_receipt_for(
                    first_id,
                    "550e8400-e29b-41d4-a716-446655440020",
                    "2026-08-17T20:00:00Z",
                ),
            )
            .expect("first mission should persist");
        store
            .create_mission(
                created_mission_for(second_id, "entarch", "lanytehq", "2026-08-17T20:01:00Z"),
                created_mission_receipt_for(
                    second_id,
                    "550e8400-e29b-41d4-a716-446655440021",
                    "2026-08-17T20:01:00Z",
                ),
            )
            .expect("second mission should persist");
        store
            .create_mission(
                created_mission_for(other_id, "other", "scope", "2026-08-17T20:00:30Z"),
                created_mission_receipt_for(
                    other_id,
                    "550e8400-e29b-41d4-a716-446655440022",
                    "2026-08-17T20:00:30Z",
                ),
            )
            .expect("other caller mission should persist");

        let first_page = store
            .list_missions(MissionListFilter {
                operating_role: "entarch".to_owned(),
                operating_scope: "lanytehq".to_owned(),
                phases: vec![MissionPhase::Created],
                limit: 1,
                cursor: None,
            })
            .expect("first page should load");
        assert_eq!(first_page.projections.len(), 1);
        assert_eq!(
            first_page.projections[0].mission.mission_id.to_string(),
            first_id
        );
        let cursor = first_page
            .next_cursor
            .clone()
            .expect("first page should have a cursor");
        assert!(!cursor.contains('|'), "cursor must be opaque");
        store
            .create_mission(
                created_mission_for(
                    inserted_after_snapshot_id,
                    "entarch",
                    "lanytehq",
                    "2026-08-17T20:01:00Z",
                ),
                created_mission_receipt_for(
                    inserted_after_snapshot_id,
                    "550e8400-e29b-41d4-a716-446655440023",
                    "2026-08-17T20:01:00Z",
                ),
            )
            .expect("post-snapshot mission should persist");
        let second_page = store
            .list_missions(MissionListFilter {
                operating_role: "entarch".to_owned(),
                operating_scope: "lanytehq".to_owned(),
                phases: vec![MissionPhase::Created],
                limit: 1,
                cursor: Some(cursor.clone()),
            })
            .expect("second page should load");
        assert_eq!(second_page.projections.len(), 1);
        assert_eq!(
            second_page.projections[0].mission.mission_id.to_string(),
            second_id
        );
        assert!(
            second_page.next_cursor.is_none(),
            "a mission inserted after the first page must not enter that snapshot"
        );

        for mismatched in [
            MissionListFilter {
                operating_role: "entarch".to_owned(),
                operating_scope: "other-scope".to_owned(),
                phases: vec![MissionPhase::Created],
                limit: 1,
                cursor: Some(cursor.clone()),
            },
            MissionListFilter {
                operating_role: "entarch".to_owned(),
                operating_scope: "lanytehq".to_owned(),
                phases: vec![MissionPhase::Active],
                limit: 1,
                cursor: Some(cursor.clone()),
            },
        ] {
            assert!(matches!(
                store.list_missions(mismatched),
                Err(StateError::InvalidMissionList(_))
            ));
        }

        let hidden = store
            .list_missions(MissionListFilter {
                operating_role: "entarch".to_owned(),
                operating_scope: "lanytehq".to_owned(),
                phases: vec![MissionPhase::Active],
                limit: 1,
                cursor: None,
            })
            .expect("phase-filtered page should load");
        assert!(hidden.projections.is_empty());
        let other_caller = store
            .list_missions(MissionListFilter {
                operating_role: "other".to_owned(),
                operating_scope: "scope".to_owned(),
                phases: Vec::new(),
                limit: 1,
                cursor: None,
            })
            .expect("other caller page should load");
        assert_eq!(other_caller.projections.len(), 1);
        assert_eq!(
            other_caller.projections[0].mission.mission_id.to_string(),
            other_id
        );

        let invalid_limit = store.list_missions(MissionListFilter {
            operating_role: "entarch".to_owned(),
            operating_scope: "lanytehq".to_owned(),
            phases: Vec::new(),
            limit: MAX_MISSION_LIST_LIMIT + 1,
            cursor: None,
        });
        assert!(matches!(
            invalid_limit,
            Err(StateError::InvalidMissionList(_))
        ));

        store
            .connection
            .execute(
                "UPDATE missions SET record_json = '{}' WHERE mission_id = ?1",
                (first_id,),
            )
            .expect("fixture corruption should apply");
        let corrupted = store.list_missions(MissionListFilter {
            operating_role: "entarch".to_owned(),
            operating_scope: "lanytehq".to_owned(),
            phases: Vec::new(),
            limit: 10,
            cursor: None,
        });
        assert!(matches!(
            corrupted,
            Err(StateError::InvalidMissionProjection(_))
        ));
    }

    #[test]
    fn projection_insert_failure_rolls_back_audit_append() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");
        store
            .connection
            .execute_batch(
                "CREATE TRIGGER fail_mission_insert \
                 BEFORE INSERT ON missions \
                 BEGIN \
                     SELECT RAISE(FAIL, 'forced mission insert failure'); \
                 END;",
            )
            .expect("failure trigger should install");

        let result =
            store.create_mission(created_mission(), created_mission_receipt(TEST_ENTRY_ID_A));
        assert!(result.is_err(), "forced projection failure must surface");
        assert!(store
            .mission(TEST_SESSION_ID)
            .expect("mission query should succeed")
            .is_none());
        assert!(store
            .audit_records(TEST_SESSION_ID)
            .expect("receipt chain query should succeed")
            .is_empty());
    }

    #[test]
    fn claimed_create_receipt_cannot_persist_a_projection() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");
        let mut receipt = created_mission_receipt(TEST_ENTRY_ID_A);
        receipt.event.source.kind = EventSourceKind::DriverReported;
        receipt.event.source.assurance = lanyte_mission::ObservationLevel::DriverObserved;
        receipt.event.source.evidence_ref = None;

        let result = store.create_mission(created_mission(), receipt);
        assert!(matches!(
            result,
            Err(StateError::InvalidMissionProjection(_))
        ));
        assert!(store
            .mission(TEST_SESSION_ID)
            .expect("mission query should succeed")
            .is_none());
        assert!(store
            .audit_records(TEST_SESSION_ID)
            .expect("empty receipt chain should load")
            .is_empty());
    }

    #[test]
    fn mission_projection_survives_reopen_with_receipt_binding() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mission = created_mission();
        let expected_hash = {
            let mut store = StateStore::open(paths.clone()).expect("state store should open");
            store
                .create_mission(mission.clone(), created_mission_receipt(TEST_ENTRY_ID_A))
                .expect("mission should commit")
                .receipt
                .entry_hash
        };

        let store = StateStore::open(paths).expect("state store should reopen");
        let stored = store
            .mission(TEST_SESSION_ID)
            .expect("mission query should succeed")
            .expect("mission should survive reopen");
        assert_eq!(stored.mission, mission);
        assert_eq!(stored.audit_entry_hash, expected_hash);
        assert_eq!(
            store
                .audit_records(TEST_SESSION_ID)
                .expect("receipt chain should survive reopen")
                .len(),
            1
        );
    }

    #[test]
    fn mission_receipt_chain_is_not_generic_session_eviction() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");
        store
            .create_mission(created_mission(), created_mission_receipt(TEST_ENTRY_ID_A))
            .expect("mission should commit");

        let evicted = store
            .session_evictor()
            .evict_older_than(
                DateTime::parse_from_rfc3339("2026-10-01T00:00:00.000Z")
                    .expect("valid time")
                    .with_timezone(&Utc),
                &AgeBasedEvictionPolicy::default(),
            )
            .expect("eviction should succeed");

        assert!(evicted.is_empty());
        assert!(store
            .mission(TEST_SESSION_ID)
            .expect("mission query should succeed")
            .is_some());
        assert_eq!(
            store
                .audit_records(TEST_SESSION_ID)
                .expect("mission receipt chain should remain hot")
                .len(),
            1
        );
        assert!(store
            .warm_export_metadata(TEST_SESSION_ID)
            .expect("warm metadata query should succeed")
            .is_none());
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
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-03-02T00:00:00.000Z".to_owned(),
                kind: AuditRecordKind::Outcome,
                action: "orchestrator.request_completion.outcome".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"status":"succeeded"}),
                verification: None,
            })
            .expect("old session tip should append");
        store
            .append_audit_record(NewAuditRecord {
                entry_id: "550e8400-e29b-41d4-a716-446655440099".to_owned(),
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
        assert_eq!(evicted[0].export.record_count, 2);
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
    fn eviction_aborts_if_session_advances_after_export() {
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
            .expect("initial record should append");

        let export = store
            .session_exporter()
            .export_to_warm(TEST_SESSION_ID)
            .expect("warm export should succeed");

        store
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
            .expect("concurrent append should succeed");

        let err = store
            .delete_hot_session(TEST_SESSION_ID, &export, "2026-05-15T00:00:00.000Z")
            .expect_err("stale export metadata should fail deletion");

        assert!(matches!(
            err,
            StateError::EvictionConflict { ref session_id } if session_id == TEST_SESSION_ID
        ));
        assert_eq!(
            store
                .audit_records(TEST_SESSION_ID)
                .expect("hot records should remain after conflict")
                .len(),
            2
        );
        assert!(store
            .warm_export_metadata(TEST_SESSION_ID)
            .expect("warm export metadata should load")
            .and_then(|metadata| metadata.hot_deleted_at)
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

    #[test]
    fn append_audit_record_rejects_noncanonical_timestamp() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);
        let mut store = StateStore::open(paths).expect("state store should open");

        let err = store
            .append_audit_record(NewAuditRecord {
                entry_id: TEST_ENTRY_ID_A.to_owned(),
                session_id: TEST_SESSION_ID.to_owned(),
                timestamp: "2026-04-02 12:00:00".to_owned(),
                kind: AuditRecordKind::Effect,
                action: "orchestrator.request_completion".to_owned(),
                severity: AuditSeverity::Info,
                envelope: AuditEnvelopeRef::default(),
                payload: serde_json::json!({"backend":"claude"}),
                verification: None,
            })
            .expect_err("noncanonical timestamp should be rejected");

        assert!(err
            .to_string()
            .contains("timestamp must be RFC3339 UTC with millisecond precision"));
    }
}
