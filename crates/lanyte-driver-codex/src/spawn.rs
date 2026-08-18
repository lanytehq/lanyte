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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexBinary {
    pub path: PathBuf,
    pub version: String,
    pub digest: String,
}

impl CodexBinary {
    pub fn resolve(explicit: &Option<PathBuf>) -> Result<Self, SpawnError> {
        let path = if let Some(path) = explicit {
            path.clone()
        } else {
            which("codex").ok_or_else(|| SpawnError::MissingBinary("codex".to_owned()))?
        };
        if !path.is_file() {
            return Err(SpawnError::MissingBinary(path.display().to_string()));
        }
        let output = Command::new(&path)
            .arg("--version")
            .output()
            .map_err(|err| SpawnError::Exec(path.display().to_string(), err.to_string()))?;
        let version = String::from_utf8_lossy(&output.stdout)
            .trim()
            .to_owned();
        let bytes = std::fs::read(&path)
            .map_err(|err| SpawnError::Exec(path.display().to_string(), err.to_string()))?;
        Ok(Self {
            path,
            version,
            digest: format!("{:x}", Sha256::digest(bytes)),
        })
    }
}

#[derive(Debug, Clone)]
pub struct CodexLaunchSpec {
    pub workspace: PathBuf,
    pub binary_path: Option<PathBuf>,
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
}
