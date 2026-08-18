use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use uuid::Uuid;

use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpawnError {
    #[error("codex binary not found: {0}")]
    MissingBinary(String),
    #[error("failed to execute `{0}`: {1}")]
    Exec(String, String),
    #[error("workspace is outside the allowed root: {0}")]
    WorkspaceEscape(String),
    #[error("workspace must be an existing directory: {0}")]
    WorkspaceMissing(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexBinary {
    pub path: PathBuf,
    pub version: String,
    pub digest: String,
}

impl CodexBinary {
    pub fn resolve(explicit: &Option<PathBuf>, workspace: &Path) -> Result<Self, SpawnError> {
        let path = if let Some(path) = explicit {
            if !path.is_absolute() {
                return Err(SpawnError::MissingBinary(path.display().to_string()));
            }
            path.clone()
        } else {
            which("codex").ok_or_else(|| SpawnError::MissingBinary("codex".to_owned()))?
        };
        if !path.is_file() {
            return Err(SpawnError::MissingBinary(path.display().to_string()));
        }
        let path = resolve_native_executable(&path)?;
        let output = run_version_probe(&path, workspace)?;
        let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        let bytes = std::fs::read(&path)
            .map_err(|err| SpawnError::Exec(path.display().to_string(), err.to_string()))?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        Ok(Self {
            path,
            version,
            digest,
        })
    }

    pub fn pin_copy(&self, pin_dir: &Path) -> Result<Self, SpawnError> {
        std::fs::create_dir_all(pin_dir)
            .map_err(|err| SpawnError::Exec(pin_dir.display().to_string(), err.to_string()))?;
        let pinned = pin_dir.join(&self.digest);
        install_executable(&self.path, &pinned)?;
        let bytes = std::fs::read(&pinned)
            .map_err(|err| SpawnError::Exec(pinned.display().to_string(), err.to_string()))?;
        let digest = format!("{:x}", Sha256::digest(bytes));
        if digest != self.digest {
            return Err(SpawnError::Exec(
                pinned.display().to_string(),
                "pinned copy digest does not match the resolved binary".to_owned(),
            ));
        }
        Ok(Self {
            path: pinned,
            version: self.version.clone(),
            digest,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CodexLaunchSpec {
    pub workspace: PathBuf,
    pub allowed_root: PathBuf,
    pub pin_dir: PathBuf,
    pub binary_path: Option<PathBuf>,
}

pub fn confine_workspace(requested: &Path, allowed_root: &Path) -> Result<PathBuf, SpawnError> {
    let root = allowed_root
        .canonicalize()
        .map_err(|_| SpawnError::WorkspaceMissing(allowed_root.display().to_string()))?;
    if !root.is_dir() {
        return Err(SpawnError::WorkspaceMissing(root.display().to_string()));
    }
    let workspace = requested
        .canonicalize()
        .map_err(|_| SpawnError::WorkspaceMissing(requested.display().to_string()))?;
    if !workspace.is_dir() {
        return Err(SpawnError::WorkspaceMissing(
            workspace.display().to_string(),
        ));
    }
    if !workspace.starts_with(&root) {
        return Err(SpawnError::WorkspaceEscape(workspace.display().to_string()));
    }
    Ok(workspace)
}

/// Environment for the child: no ambient tokens, only a constrained PATH and
/// a workspace HOME. Callers must not re-inject GH/LANYTE session secrets.
#[must_use]
pub fn scrub_child_env(workspace: &Path) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.insert(
        "PATH".to_owned(),
        "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin".to_owned(),
    );
    env.insert("HOME".to_owned(), workspace.display().to_string());
    env.insert("TERM".to_owned(), "dumb".to_owned());
    env.insert("LANG".to_owned(), "C".to_owned());
    env
}

fn run_version_probe(path: &Path, workspace: &Path) -> Result<std::process::Output, SpawnError> {
    let mut last = None;
    for attempt in 0..8 {
        match Command::new(path)
            .arg("--version")
            .env_clear()
            .envs(scrub_child_env(workspace))
            .output()
        {
            Ok(output) => return Ok(output),
            Err(err) if is_text_file_busy(&err) && attempt + 1 < 8 => {
                std::thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1) as u64));
                last = Some(err);
            }
            Err(err) => {
                return Err(SpawnError::Exec(
                    path.display().to_string(),
                    err.to_string(),
                ));
            }
        }
    }
    Err(SpawnError::Exec(
        path.display().to_string(),
        last.map(|err| err.to_string())
            .unwrap_or_else(|| "version probe failed".to_owned()),
    ))
}

fn is_text_file_busy(err: &std::io::Error) -> bool {
    err.raw_os_error() == Some(26)
}

fn install_executable(source: &Path, dest: &Path) -> Result<(), SpawnError> {
    if dest.is_file() {
        return Ok(());
    }
    let staging_name = format!(
        "{}.{}.partial",
        dest.file_name().unwrap_or_default().to_string_lossy(),
        Uuid::new_v4()
    );
    let staging = dest.with_file_name(staging_name);
    write_exclusive_copy(source, &staging)?;
    if dest.is_file() {
        let _ = std::fs::remove_file(&staging);
        return Ok(());
    }
    match std::fs::rename(&staging, dest) {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = std::fs::remove_file(&staging);
            if dest.is_file() {
                Ok(())
            } else {
                Err(SpawnError::Exec(
                    dest.display().to_string(),
                    err.to_string(),
                ))
            }
        }
    }
}

fn write_exclusive_copy(source: &Path, staging: &Path) -> Result<(), SpawnError> {
    let mut input = std::fs::File::open(source)
        .map_err(|err| SpawnError::Exec(source.display().to_string(), err.to_string()))?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staging)
        .map_err(|err| SpawnError::Exec(staging.display().to_string(), err.to_string()))?;
    io::copy(&mut input, &mut output)
        .map_err(|err| SpawnError::Exec(staging.display().to_string(), err.to_string()))?;
    output
        .flush()
        .map_err(|err| SpawnError::Exec(staging.display().to_string(), err.to_string()))?;
    output
        .sync_all()
        .map_err(|err| SpawnError::Exec(staging.display().to_string(), err.to_string()))?;
    drop(output);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(staging, std::fs::Permissions::from_mode(0o500))
            .map_err(|err| SpawnError::Exec(staging.display().to_string(), err.to_string()))?;
    }
    Ok(())
}

fn resolve_native_executable(path: &Path) -> Result<PathBuf, SpawnError> {
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if looks_like_native(&resolved) {
        return Ok(resolved);
    }
    if let Some(native) = native_from_npm_wrapper(&resolved) {
        return Ok(native);
    }
    Err(SpawnError::Exec(
        resolved.display().to_string(),
        "codex path is a script wrapper, not a pinned native executable".to_owned(),
    ))
}

fn looks_like_native(path: &Path) -> bool {
    !matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("js" | "mjs" | "cjs" | "ts")
    )
}

fn native_from_npm_wrapper(wrapper: &Path) -> Option<PathBuf> {
    let package_root = wrapper.parent()?.parent()?;
    let (package, triple) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => ("@openai/codex-darwin-arm64", "aarch64-apple-darwin"),
        ("macos", "x86_64") => ("@openai/codex-darwin-x64", "x86_64-apple-darwin"),
        ("linux", "aarch64") => ("@openai/codex-linux-arm64", "aarch64-unknown-linux-musl"),
        ("linux", "x86_64") => ("@openai/codex-linux-x64", "x86_64-unknown-linux-musl"),
        _ => return None,
    };
    let candidate = package_root
        .join("node_modules")
        .join(package)
        .join("vendor")
        .join(triple)
        .join("bin")
        .join("codex");
    candidate.is_file().then_some(candidate)
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(name);
        candidate.is_file().then_some(candidate)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_omits_token_names() {
        let env = scrub_child_env(Path::new("/tmp/ws"));
        assert!(!env.keys().any(|key| key.contains("TOKEN")));
        assert!(!env.keys().any(|key| key.contains("SECRET")));
        assert_eq!(env.get("HOME").map(String::as_str), Some("/tmp/ws"));
    }

    #[test]
    fn confine_rejects_escape_and_accepts_child() {
        let root = std::env::temp_dir().join(format!("lanyte-ws-root-{}", uuid_like()));
        let inside = root.join("mission");
        std::fs::create_dir_all(&inside).unwrap();
        assert_eq!(
            confine_workspace(&inside, &root).unwrap(),
            inside.canonicalize().unwrap()
        );
        let outside = std::env::temp_dir().join(format!("lanyte-ws-out-{}", uuid_like()));
        std::fs::create_dir_all(&outside).unwrap();
        assert!(matches!(
            confine_workspace(&outside, &root),
            Err(SpawnError::WorkspaceEscape(_))
        ));
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    fn uuid_like() -> String {
        format!("{}", std::process::id())
    }

    #[test]
    fn pin_copy_concurrent_same_digest_publishes_once() {
        let root = std::env::temp_dir().join(format!("lanyte-pin-race-{}", uuid_like()));
        let pin_dir = root.join("pins");
        std::fs::create_dir_all(&pin_dir).unwrap();
        let source = root.join("codex-src");
        std::fs::write(&source, b"#!/bin/sh\necho fake-codex\n").unwrap();
        let digest = format!("{:x}", Sha256::digest(std::fs::read(&source).unwrap()));
        let binary = CodexBinary {
            path: source,
            version: "fake".to_owned(),
            digest: digest.clone(),
        };

        let results: Vec<_> = std::thread::scope(|scope| {
            (0..8)
                .map(|_| {
                    let binary = binary.clone();
                    let pin_dir = pin_dir.clone();
                    scope.spawn(move || binary.pin_copy(&pin_dir))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("thread"))
                .collect()
        });

        assert!(
            results.iter().all(Result::is_ok),
            "concurrent pin_copy must not fail: {results:?}"
        );
        let published: std::collections::BTreeSet<_> = results
            .into_iter()
            .map(|result| result.unwrap().path)
            .collect();
        assert_eq!(published.len(), 1);
        assert_eq!(published.iter().next().unwrap(), &pin_dir.join(&digest));
        let leftovers: Vec<_> = std::fs::read_dir(&pin_dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name())
            .filter(|name| name.to_string_lossy().ends_with(".partial"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "no leftover staging files: {leftovers:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
