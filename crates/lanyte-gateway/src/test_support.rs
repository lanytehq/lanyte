use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use lanyte_common::{GatewayConfig, PEER_SERVICE_CHANNELS};
use tokio::sync::mpsc;

use crate::{spawn, GatewayError, GatewayEvent, GatewayHandle};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(1);

const PERMISSIVE_TEST_SCHEMA: &str = r#"{"type":"object","additionalProperties":true}"#;

/// RAII temp directory for gateway tests. Removed on drop, including after panic.
#[derive(Debug)]
pub struct TempGatewayDir {
    path: PathBuf,
}

impl TempGatewayDir {
    pub fn new(tag: &str) -> Self {
        let unique = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let epoch_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        // Keep paths short: macOS UDS paths cap at 104 bytes.
        let path = PathBuf::from("/tmp").join(format!("ltgw-{tag}-{unique}-{epoch_nanos}"));
        std::fs::create_dir_all(path.join("schemas")).expect("test temp dir should be creatable");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn socket_path(&self) -> PathBuf {
        self.path.join("gateway.sock")
    }

    pub fn schemas_dir(&self) -> PathBuf {
        self.path.join("schemas")
    }
}

impl Drop for TempGatewayDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub fn write_test_schema(dir: &Path, channel: u16) {
    std::fs::create_dir_all(dir).expect("schema dir should be creatable");
    let path = dir.join(format!("channel_{channel}.schema.json"));
    std::fs::write(path, PERMISSIVE_TEST_SCHEMA).expect("schema should be writable");
}

pub fn write_all_peer_schemas(dir: &Path) {
    for channel in PEER_SERVICE_CHANNELS {
        write_test_schema(dir, channel);
    }
}

pub fn spawn_test_gateway(
    dir: &TempGatewayDir,
    channels: &[u16],
) -> Result<(GatewayHandle, mpsc::Receiver<GatewayEvent>), GatewayError> {
    for &channel in channels {
        write_test_schema(&dir.schemas_dir(), channel);
    }

    spawn(GatewayConfig {
        core_peer_id: "lanyte-core".to_owned(),
        socket_path: dir.socket_path(),
        crucible_schemas_dir: dir.schemas_dir(),
    })
}
