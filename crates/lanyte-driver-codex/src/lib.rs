//! Codex App Server driver. Owns a child `codex app-server` process and maps
//! JSON-RPC observations to [`lanyte_mission::NormalizedHarnessEvent`].

mod protocol;
mod spawn;

pub use protocol::{map_notification, CodexProtocolError, JsonRpcLine};
pub use spawn::{scrub_child_env, CodexBinary, CodexLaunchSpec, SpawnError};

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::Utc;
use lanyte_mission::{
    CapabilityFidelity, CapabilityName, DriverAvailability, DriverCapability,
    DriverCapabilityReport, DriverDescriptor, DriverValidityCondition, EnforcementLevel,
    HarnessDriver, NormalizedHarnessEvent, ObservationLevel, ReplaySupport,
    DRIVER_CAPABILITIES_SCHEMA,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use uuid::Uuid;

const DRIVER_ID: &str = "driver.codex.app_server";
const DRIVER_VERSION: &str = "0.1.0";
const HARNESS_KIND: &str = "codex";

#[derive(Debug, Error)]
pub enum CodexDriverError {
    #[error("spawn failed: {0}")]
    Spawn(#[from] SpawnError),
    #[error("protocol failed: {0}")]
    Protocol(#[from] CodexProtocolError),
    #[error("child stdin is closed")]
    StdinClosed,
    #[error("child stdout is closed")]
    StdoutClosed,
    #[error("timed out waiting for Codex App Server")]
    Timeout,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Live Codex App Server session owned by the kernel process.
pub struct CodexSession {
    pub attempt_id: Uuid,
    pub harness_session_id: String,
    pub binary: CodexBinary,
    child: Child,
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
    next_id: AtomicU64,
}

impl CodexSession {
    pub async fn identify(&mut self) -> Result<String, CodexDriverError> {
        Ok(self.harness_session_id.clone())
    }

    pub async fn observe(&mut self) -> Result<Option<NormalizedHarnessEvent>, CodexDriverError> {
        let mut stdout = self.stdout.lock().await;
        let mut line = String::new();
        let read = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            stdout.read_line(&mut line),
        )
        .await;
        match read {
            Ok(Ok(0)) => Ok(None),
            Ok(Ok(_)) => Ok(map_notification(self.attempt_id, line.trim_end())),
            Ok(Err(err)) => Err(err.into()),
            Err(_) => Ok(None),
        }
    }

    pub async fn close(&mut self) -> Result<(), CodexDriverError> {
        let _ = self
            .request("shutdown", json!({}))
            .await;
        let _ = self.child.start_kill();
        let _ = self.child.wait().await;
        Ok(())
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, CodexDriverError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let message = json!({"id": id, "method": method, "params": params});
        let mut encoded = serde_json::to_string(&message)?;
        encoded.push('\n');
        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(encoded.as_bytes()).await?;
            stdin.flush().await?;
        }
        let mut stdout = self.stdout.lock().await;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if tokio::time::Instant::now() > deadline {
                return Err(CodexDriverError::Timeout);
            }
            let mut line = String::new();
            let n = stdout.read_line(&mut line).await?;
            if n == 0 {
                return Err(CodexDriverError::StdoutClosed);
            }
            let parsed: Value = serde_json::from_str(line.trim_end())?;
            if parsed.get("id") == Some(&json!(id)) {
                if let Some(error) = parsed.get("error") {
                    return Err(CodexProtocolError::Remote(error.to_string()).into());
                }
                return Ok(parsed.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }
}

pub struct CodexAppServerDriver {
    spec: CodexLaunchSpec,
}

impl CodexAppServerDriver {
    #[must_use]
    pub fn new(spec: CodexLaunchSpec) -> Self {
        Self { spec }
    }

    pub async fn create(&self, attempt_id: Uuid) -> Result<CodexSession, CodexDriverError> {
        let binary = CodexBinary::resolve(&self.spec.binary_path)?;
        let mut command = Command::new(&binary.path);
        command
            .args(["app-server"])
            .current_dir(&self.spec.workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .envs(scrub_child_env(&self.spec.workspace));
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().ok_or(CodexDriverError::StdinClosed)?;
        let stdout = child.stdout.take().ok_or(CodexDriverError::StdoutClosed)?;
        let session = CodexSession {
            attempt_id,
            harness_session_id: String::new(),
            binary,
            child,
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            next_id: AtomicU64::new(1),
        };
        let init = session
            .request(
                "initialize",
                json!({
                    "clientInfo": {
                        "name": "lanyte",
                        "title": "Lanyte",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
            .await?;
        let _ = init;
        let started = session
            .request(
                "thread/start",
                json!({ "cwd": self.spec.workspace.display().to_string() }),
            )
            .await?;
        let thread_id = started
            .get("thread")
            .and_then(|thread| thread.get("id"))
            .or_else(|| started.get("threadId"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let mut session = session;
        session.harness_session_id = thread_id;
        Ok(session)
    }
}

impl HarnessDriver for CodexAppServerDriver {
    fn descriptor(&self) -> DriverDescriptor {
        DriverDescriptor {
            driver_id: DRIVER_ID.to_owned(),
            driver_version: DRIVER_VERSION.to_owned(),
            harness_kind: HARNESS_KIND.to_owned(),
        }
    }

    fn capabilities(&self) -> DriverCapabilityReport {
        let now = Utc::now();
        let binary = CodexBinary::resolve(&self.spec.binary_path).ok();
        let version = binary
            .as_ref()
            .map(|bin| bin.version.clone())
            .unwrap_or_else(|| "unknown".to_owned());
        let digest = binary
            .as_ref()
            .map(|bin| bin.digest.clone())
            .unwrap_or_else(|| "0".repeat(64));
        DriverCapabilityReport {
            capabilities_schema: DRIVER_CAPABILITIES_SCHEMA.to_owned(),
            report_id: Uuid::new_v4(),
            driver_id: DRIVER_ID.to_owned(),
            driver_version: DRIVER_VERSION.to_owned(),
            harness_kind: HARNESS_KIND.to_owned(),
            observed_at: now,
            expires_at: now + chrono::Duration::hours(1),
            availability: if binary.is_some() {
                DriverAvailability::Available
            } else {
                DriverAvailability::TemporarilyUnavailable
            },
            capabilities: [
                CapabilityName::Create,
                CapabilityName::Identify,
                CapabilityName::Observe,
                CapabilityName::Close,
            ]
            .into_iter()
            .map(|name| DriverCapability {
                name,
                fidelity: CapabilityFidelity::Native,
                observation: ObservationLevel::KernelObserved,
                enforcement: EnforcementLevel::ProtocolConfirmed,
                replay: ReplaySupport::None,
                limitation: None,
                evidence_ref: Some("probes/codex-app-server".to_owned()),
            })
            .collect(),
            validity_condition: DriverValidityCondition {
                kind: "executable-version-platform-match".to_owned(),
                executable_version: version,
                executable_sha256: digest,
                configuration_sha256: format!("{:x}", Sha256::digest(path_bytes(&self.spec.workspace))),
                platform: std::env::consts::OS.to_owned(),
                probe_ref: "probes/codex-app-server".to_owned(),
            },
            evidence_ref: "capability-reports/codex-app-server".to_owned(),
        }
    }
}

#[cfg(unix)]
fn path_bytes(path: &std::path::Path) -> &[u8] {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes()
}

#[cfg(not(unix))]
fn path_bytes(path: &std::path::Path) -> &[u8] {
    path.to_string_lossy().as_bytes()
}
