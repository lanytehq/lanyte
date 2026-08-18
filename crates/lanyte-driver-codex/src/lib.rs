//! Codex App Server driver. Owns a child `codex app-server` process and maps
//! JSON-RPC observations to [`lanyte_mission::NormalizedHarnessEvent`].

mod process;
mod protocol;
mod spawn;

pub use process::{
    capture_process_tree_handle, format_process_tree_handle, format_process_tree_ref,
    parse_process_tree_handle, parse_process_tree_ref, probe_process_tree, terminate_process_tree,
    ProcessTreeHandle,
};
pub use protocol::{map_notification, CodexProtocolError, JsonRpcLine};
pub use spawn::{confine_workspace, scrub_child_env, CodexBinary, CodexLaunchSpec, SpawnError};

#[derive(Debug, Clone, Copy)]
pub enum CloseOutcome {
    AlreadyExited(std::process::ExitStatus),
    Terminated(std::process::ExitStatus),
}

impl CloseOutcome {
    #[must_use]
    pub fn status(&self) -> std::process::ExitStatus {
        match self {
            Self::AlreadyExited(status) | Self::Terminated(status) => *status,
        }
    }
}

use std::collections::{HashMap, VecDeque};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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

type PendingResponses =
    Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value, CodexDriverError>>>>>;

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
    stdout: Option<Mutex<BufReader<ChildStdout>>>,
    events: Arc<Mutex<VecDeque<NormalizedHarnessEvent>>>,
    overflowed: Arc<std::sync::atomic::AtomicBool>,
    next_id: AtomicU64,
    last_close: Option<CloseOutcome>,
    pub process_tree_ref: Option<String>,
    active_turn_id: Arc<Mutex<Option<String>>>,
    pending: PendingResponses,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterruptAttempt {
    Unavailable,
    RequestAccepted { thread_id: String, turn_id: String },
    Interrupted { thread_id: String, turn_id: String },
    Timeout { thread_id: String, turn_id: String },
    UnrelatedCompletion { thread_id: String, turn_id: String },
    Failed { detail: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessTreeKill {
    KillDispatched,
    Cleared,
    Survivors,
    Unknown,
}

impl CodexSession {
    pub async fn identify(&mut self) -> Result<String, CodexDriverError> {
        Ok(self.harness_session_id.clone())
    }

    pub async fn observe(&mut self) -> Result<Option<NormalizedHarnessEvent>, CodexDriverError> {
        Ok(self.events.lock().await.pop_front())
    }

    #[must_use]
    pub fn overflowed(&self) -> bool {
        self.overflowed.load(Ordering::SeqCst)
    }

    fn start_observation_pump(&mut self) {
        let Some(stdout) = self.stdout.take() else {
            return;
        };
        let events = Arc::clone(&self.events);
        let overflowed = Arc::clone(&self.overflowed);
        let active_turn_id = Arc::clone(&self.active_turn_id);
        let pending = Arc::clone(&self.pending);
        let attempt_id = self.attempt_id;
        tokio::spawn(async move {
            let mut stdout = stdout.lock().await;
            loop {
                let mut line = String::new();
                match stdout.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = line.trim_end();
                        if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                            if let Some(id) = parsed.get("id").and_then(Value::as_u64) {
                                if let Some(tx) = pending.lock().await.remove(&id) {
                                    let result = if let Some(error) = parsed.get("error") {
                                        Err(CodexProtocolError::Remote(error.to_string()).into())
                                    } else {
                                        Ok(parsed.get("result").cloned().unwrap_or(Value::Null))
                                    };
                                    let _ = tx.send(result);
                                    continue;
                                }
                            }
                        }
                        if let Some(event) = map_notification(attempt_id, trimmed) {
                            if let NormalizedHarnessEvent::TurnProgress {
                                turn_id, status, ..
                            } = &event
                            {
                                let mut turn = active_turn_id.lock().await;
                                if status == "started" {
                                    *turn = Some(turn_id.clone());
                                } else if status == "interrupted" || status == "completed" {
                                    *turn = None;
                                }
                            }
                            let mut queue = events.lock().await;
                            if queue.len() >= 256 {
                                queue.pop_front();
                                overflowed.store(true, Ordering::SeqCst);
                            }
                            queue.push_back(event);
                        }
                    }
                    Err(_) => break,
                }
            }
        });
    }

    pub fn poll_exit(&mut self) -> Result<Option<std::process::ExitStatus>, CodexDriverError> {
        self.child.try_wait().map_err(Into::into)
    }

    #[must_use]
    pub fn retained_close_outcome(&self) -> Option<CloseOutcome> {
        self.last_close
    }

    pub async fn close(&mut self) -> Result<CloseOutcome, CodexDriverError> {
        if let Some(CloseOutcome::Terminated(status)) = self.last_close {
            return Ok(CloseOutcome::Terminated(status));
        }
        if let Ok(Some(status)) = self.child.try_wait() {
            let outcome = CloseOutcome::AlreadyExited(status);
            self.last_close = Some(outcome);
            return Ok(outcome);
        }
        if !self.harness_session_id.is_empty() {
            let _ = self
                .notify(
                    "thread/unsubscribe",
                    json!({ "threadId": self.harness_session_id }),
                )
                .await;
        }
        self.child.start_kill()?;
        let status = tokio::time::timeout(std::time::Duration::from_secs(5), self.child.wait())
            .await
            .map_err(|_| CodexDriverError::Timeout)??;
        let outcome = CloseOutcome::Terminated(status);
        self.last_close = Some(outcome);
        Ok(outcome)
    }

    pub async fn active_turn_id(&self) -> Option<String> {
        self.active_turn_id.lock().await.clone()
    }

    pub async fn interrupt_turn(
        &mut self,
        thread_id: Option<&str>,
        turn_id: Option<&str>,
    ) -> InterruptAttempt {
        let turn_id = match self
            .active_turn_id()
            .await
            .or_else(|| turn_id.map(str::to_owned))
        {
            Some(turn_id) if !turn_id.is_empty() => turn_id,
            _ => return InterruptAttempt::Unavailable,
        };
        let thread_id = thread_id
            .map(str::to_owned)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                if self.harness_session_id.is_empty() {
                    None
                } else {
                    Some(self.harness_session_id.clone())
                }
            });
        let Some(thread_id) = thread_id else {
            return InterruptAttempt::Unavailable;
        };
        let cursor = self.events.lock().await.len();
        if let Err(err) = self
            .request(
                "turn/interrupt",
                json!({ "threadId": thread_id, "turnId": turn_id }),
            )
            .await
        {
            return match err {
                CodexDriverError::Timeout => InterruptAttempt::Timeout { thread_id, turn_id },
                other => InterruptAttempt::Failed {
                    detail: other.to_string(),
                },
            };
        }
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let outcome = matching_interrupt_outcome(
                self.events.lock().await.iter().skip(cursor),
                &thread_id,
                &turn_id,
            );
            if let Some(outcome) = outcome {
                return outcome;
            }
            if tokio::time::Instant::now() >= deadline {
                return InterruptAttempt::RequestAccepted { thread_id, turn_id };
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    pub async fn interrupt_active_turn(&mut self) -> InterruptAttempt {
        self.interrupt_turn(None, None).await
    }

    pub fn kill_process_tree(&mut self) -> ProcessTreeKill {
        if let Some(tree_ref) = &self.process_tree_ref {
            return terminate_process_tree(tree_ref);
        }
        match self.child.id() {
            Some(pid) => capture_process_tree_handle(pid)
                .map(|handle| terminate_process_tree(&format_process_tree_handle(&handle)))
                .unwrap_or(ProcessTreeKill::Unknown),
            None => ProcessTreeKill::Unknown,
        }
    }

    pub fn probe_process_tree(&self) -> ProcessTreeKill {
        self.process_tree_ref
            .as_deref()
            .map(probe_process_tree)
            .unwrap_or(ProcessTreeKill::Unknown)
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), CodexDriverError> {
        let mut encoded = serde_json::to_string(&json!({"method": method, "params": params}))?;
        encoded.push('\n');
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(encoded.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, CodexDriverError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let message = json!({"id": id, "method": method, "params": params});
        let mut encoded = serde_json::to_string(&message)?;
        encoded.push('\n');
        if self.stdout.is_none() {
            let (tx, rx) = tokio::sync::oneshot::channel();
            self.pending.lock().await.insert(id, tx);
            {
                let mut stdin = self.stdin.lock().await;
                stdin.write_all(encoded.as_bytes()).await?;
                stdin.flush().await?;
            }
            return tokio::time::timeout(std::time::Duration::from_secs(10), rx)
                .await
                .map_err(|_| CodexDriverError::Timeout)?
                .map_err(|_| CodexDriverError::StdoutClosed)?;
        }
        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(encoded.as_bytes()).await?;
            stdin.flush().await?;
        }
        let stdout = self.stdout.as_ref().ok_or(CodexDriverError::StdoutClosed)?;
        let mut stdout = stdout.lock().await;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(CodexDriverError::Timeout);
            }
            let mut line = String::new();
            let n = tokio::time::timeout(remaining, stdout.read_line(&mut line))
                .await
                .map_err(|_| CodexDriverError::Timeout)??;
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
            if let Some(event) = map_notification(self.attempt_id, line.trim_end()) {
                self.events.lock().await.push_back(event);
            }
        }
    }
}

fn matching_interrupt_outcome<'a>(
    mut events: impl Iterator<Item = &'a NormalizedHarnessEvent>,
    thread_id: &str,
    turn_id: &str,
) -> Option<InterruptAttempt> {
    events.find_map(|event| {
        if let NormalizedHarnessEvent::TurnProgress {
            turn_id: event_turn,
            thread_id: event_thread,
            status,
            ..
        } = event
        {
            if event_turn != turn_id {
                return None;
            }
            if event_thread.as_deref() != Some(thread_id) {
                return Some(InterruptAttempt::UnrelatedCompletion {
                    thread_id: thread_id.to_owned(),
                    turn_id: turn_id.to_owned(),
                });
            }
            if status == "interrupted" {
                return Some(InterruptAttempt::Interrupted {
                    thread_id: thread_id.to_owned(),
                    turn_id: turn_id.to_owned(),
                });
            }
            if status == "completed" {
                return Some(InterruptAttempt::UnrelatedCompletion {
                    thread_id: thread_id.to_owned(),
                    turn_id: turn_id.to_owned(),
                });
            }
        }
        None
    })
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
        let workspace = confine_workspace(&self.spec.workspace, &self.spec.allowed_root)?;
        let binary = CodexBinary::resolve(&self.spec.binary_path, &workspace)?
            .pin_copy(&self.spec.pin_dir)?;
        let mut command = Command::new(&binary.path);
        command
            .args(["app-server"])
            .current_dir(&workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env_clear()
            .envs(scrub_child_env(&workspace));
        #[cfg(unix)]
        {
            command.process_group(0);
        }
        let mut child = command.spawn()?;
        let process_tree_ref = child
            .id()
            .and_then(capture_process_tree_handle)
            .map(|handle| format_process_tree_handle(&handle));
        if process_tree_ref.is_none() {
            let _ = child.start_kill();
            return Err(CodexDriverError::Io(std::io::Error::other(
                "could not capture authoritative process-group birth identity",
            )));
        }
        let stdin = child.stdin.take().ok_or(CodexDriverError::StdinClosed)?;
        let stdout = child.stdout.take().ok_or(CodexDriverError::StdoutClosed)?;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    line.clear();
                }
            });
        }
        let session = CodexSession {
            attempt_id,
            harness_session_id: String::new(),
            binary,
            child,
            stdin: Mutex::new(stdin),
            stdout: Some(Mutex::new(BufReader::new(stdout))),
            events: Arc::new(Mutex::new(VecDeque::new())),
            overflowed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            next_id: AtomicU64::new(1),
            last_close: None,
            process_tree_ref,
            active_turn_id: Arc::new(Mutex::new(None)),
            pending: Arc::new(Mutex::new(HashMap::new())),
        };
        if let Err(err) = session
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
            .await
        {
            let mut session = session;
            let _ = session.close().await;
            return Err(err);
        }
        if let Err(err) = session.notify("initialized", json!({})).await {
            let mut session = session;
            let _ = session.close().await;
            return Err(err);
        }
        let started = match session
            .request(
                "thread/start",
                json!({ "cwd": workspace.display().to_string() }),
            )
            .await
        {
            Ok(started) => started,
            Err(err) => {
                let mut session = session;
                let _ = session.close().await;
                return Err(err);
            }
        };
        let thread_id = started
            .get("thread")
            .and_then(|thread| thread.get("id"))
            .or_else(|| started.get("threadId"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let mut session = session;
        session.harness_session_id = thread_id.clone();
        {
            let mut events = session.events.lock().await;
            let already_started = events
                .iter()
                .any(|event| matches!(event, NormalizedHarnessEvent::Started { .. }));
            if !already_started {
                events.push_back(NormalizedHarnessEvent::Started {
                    occurred_at: Utc::now(),
                    attempt_id,
                    harness_session_id: thread_id,
                    detail: Some("thread/start".to_owned()),
                });
            }
        }
        session.start_observation_pump();
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
        let binary = CodexBinary::resolve(&self.spec.binary_path, &self.spec.workspace).ok();
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
                (
                    CapabilityName::Create,
                    ObservationLevel::KernelObserved,
                    EnforcementLevel::ProtocolConfirmed,
                ),
                (
                    CapabilityName::Identify,
                    ObservationLevel::KernelObserved,
                    EnforcementLevel::ProtocolConfirmed,
                ),
                (
                    CapabilityName::Observe,
                    ObservationLevel::DriverObserved,
                    EnforcementLevel::ProtocolConfirmed,
                ),
                (
                    CapabilityName::Close,
                    ObservationLevel::KernelObserved,
                    EnforcementLevel::ProtocolConfirmed,
                ),
                (
                    CapabilityName::Cancel,
                    ObservationLevel::DriverObserved,
                    EnforcementLevel::ProtocolConfirmed,
                ),
                (
                    CapabilityName::LocalProcessTermination,
                    ObservationLevel::KernelObserved,
                    EnforcementLevel::LocalProcessControl,
                ),
            ]
            .into_iter()
            .map(|(name, observation, enforcement)| DriverCapability {
                name,
                fidelity: CapabilityFidelity::Native,
                observation,
                enforcement,
                replay: ReplaySupport::None,
                limitation: Some(format!("codex-cli:{version}")),
                evidence_ref: Some("probes/codex-app-server".to_owned()),
            })
            .collect(),
            validity_condition: DriverValidityCondition {
                kind: "executable-version-platform-match".to_owned(),
                executable_version: DRIVER_VERSION.to_owned(),
                executable_sha256: digest,
                configuration_sha256: format!(
                    "{:x}",
                    Sha256::digest(path_bytes(&self.spec.workspace))
                ),
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

#[cfg(test)]
mod interrupt_proof_tests {
    use super::*;

    fn turn(thread: Option<&str>, turn: &str, status: &str) -> NormalizedHarnessEvent {
        NormalizedHarnessEvent::TurnProgress {
            occurred_at: Utc::now(),
            attempt_id: Uuid::nil(),
            thread_id: thread.map(str::to_owned),
            turn_id: turn.to_owned(),
            status: status.to_owned(),
        }
    }

    #[test]
    fn missing_thread_is_not_protocol_confirmed() {
        let events = [turn(None, "turn_1", "interrupted")];
        assert!(matches!(
            matching_interrupt_outcome(events.iter(), "thr", "turn_1"),
            Some(InterruptAttempt::UnrelatedCompletion { .. })
        ));
    }

    #[test]
    fn stale_events_before_cursor_are_ignored() {
        let events = [
            turn(Some("thr"), "turn_1", "interrupted"),
            turn(Some("thr"), "turn_1", "completed"),
        ];
        assert!(matching_interrupt_outcome(events.iter().skip(1), "thr", "turn_1").is_some());
        assert!(matches!(
            matching_interrupt_outcome(events.iter().skip(1), "thr", "turn_1"),
            Some(InterruptAttempt::UnrelatedCompletion { .. })
        ));
        assert!(matching_interrupt_outcome(events.iter().skip(2), "thr", "turn_1").is_none());
    }

    #[test]
    fn matching_interrupted_requires_exact_thread_and_turn() {
        let events = [turn(Some("thr"), "turn_1", "interrupted")];
        assert!(matches!(
            matching_interrupt_outcome(events.iter(), "thr", "turn_1"),
            Some(InterruptAttempt::Interrupted { .. })
        ));
        assert!(matches!(
            matching_interrupt_outcome(events.iter(), "other", "turn_1"),
            Some(InterruptAttempt::UnrelatedCompletion { .. })
        ));
    }
}
