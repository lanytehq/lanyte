//! Hot-tier state store bootstrap for lanyte core.
//!
//! This crate owns SQLite setup (paths, WAL mode, schema, append-only guards)
//! and is the boundary through which core accesses memory state.

use std::fs;
use std::path::{Path, PathBuf};

use lanyte_common::env as common_env;
use rusqlite::{Connection, OptionalExtension};
use thiserror::Error;

pub const LANYTE_STATE_ROOT_ENV: &str = "LANYTE_STATE_ROOT";
pub const DEFAULT_STATE_ROOT: &str = "/var/lib/lanyte/state";

const MIGRATION_001: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/001_initial.sql"
));
const MIGRATIONS: &[(i64, &str)] = &[(1, MIGRATION_001)];

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
        assert_eq!(store.schema_version().expect("schema version query"), 1);
    }

    #[test]
    fn reopen_is_idempotent_and_does_not_rerun_migrations() {
        let root = temp_state_root();
        let paths = StatePaths::new(&root);

        let store_a = StateStore::open(paths.clone()).expect("state store should open");
        assert_eq!(store_a.schema_version().expect("schema version query"), 1);

        let store_b = StateStore::open(paths).expect("state store should open again");
        assert_eq!(store_b.schema_version().expect("schema version query"), 1);
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
}
