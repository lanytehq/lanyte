use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

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
        let output = Command::new(&path)
            .arg("--version")
            .env_clear()
            .envs(scrub_child_env(workspace))
            .output()
            .map_err(|err| SpawnError::Exec(path.display().to_string(), err.to_string()))?;
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
        if !pinned.is_file() {
            std::fs::copy(&self.path, &pinned)
                .map_err(|err| SpawnError::Exec(pinned.display().to_string(), err.to_string()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&pinned, std::fs::Permissions::from_mode(0o500)).map_err(
                    |err| SpawnError::Exec(pinned.display().to_string(), err.to_string()),
                )?;
            }
        }
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
}
