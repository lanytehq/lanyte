use std::env;
use std::fmt;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::env as common_env;
use crate::error::{CommonError, Result};

pub const LANYTE_CORE_PEER_ID_ENV: &str = "LANYTE_CORE_PEER_ID";
pub const LANYTE_CONFIG_PATH_ENV: &str = "LANYTE_CONFIG_PATH";
pub const LANYTE_GATEWAY_SOCKET_PATH_ENV: &str = "LANYTE_GATEWAY_SOCKET_PATH";
pub const LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV: &str = "LANYTE_CRUCIBLE_SCHEMAS_DIR";
pub const LLM_CONFIG: &str = "llm";
pub const LLM_CLAUDE_MODEL_ENV: &str = "LANYTE_LLM_CLAUDE_MODEL";
pub const LLM_CLAUDE_API_KEY_ENV: &str = "LANYTE_LLM_CLAUDE_API_KEY";
pub const LLM_GROK_MODEL_ENV: &str = "LANYTE_LLM_GROK_MODEL";
pub const LLM_GROK_API_KEY_ENV: &str = "LANYTE_LLM_GROK_API_KEY";
pub const LLM_OPENAI_MODEL_ENV: &str = "LANYTE_LLM_OPENAI_MODEL";
pub const LLM_OPENAI_API_KEY_ENV: &str = "LANYTE_LLM_OPENAI_API_KEY";
pub const LLM_OPENAI_BASE_URL_ENV: &str = "LANYTE_LLM_OPENAI_BASE_URL";

pub const DEFAULT_CORE_PEER_ID: &str = "lanyte-core";
pub const DEFAULT_CONFIG_RELATIVE_PATH: &str = ".config/lanytehq/config.toml";
pub const DEFAULT_GATEWAY_SOCKET_PATH: &str = "/tmp/lanyte.sock";
pub const DEFAULT_CRUCIBLE_SCHEMAS_DIR: &str = "../lanyte-crucible/schemas/ipc";
pub const DEFAULT_CLAUDE_MODEL: &str = "claude-sonnet-4-6";
pub const DEFAULT_GROK_MODEL: &str = "grok-4.20-beta-latest-reasoning";
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4";
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

const CONFIG_GATEWAY_CORE_PEER_ID_FIELD: &str = "gateway.core_peer_id";
const CONFIG_GATEWAY_SOCKET_PATH_FIELD: &str = "gateway.socket_path";
const CONFIG_GATEWAY_CRUCIBLE_SCHEMAS_DIR_FIELD: &str = "gateway.crucible_schemas_dir";
const CONFIG_LLM_CLAUDE_MODEL_FIELD: &str = "llm.claude.model";
const CONFIG_LLM_GROK_MODEL_FIELD: &str = "llm.grok.model";
const CONFIG_LLM_OPENAI_MODEL_FIELD: &str = "llm.openai.model";
const CONFIG_LLM_OPENAI_BASE_URL_FIELD: &str = "llm.openai.base_url";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LanyteConfig {
    pub gateway: GatewayConfig,
    pub llm: LlmConfig,
}

impl LanyteConfig {
    pub fn load() -> Result<Self> {
        Self::load_with_lookup(common_env::read_env_var_utf8)
    }

    pub fn from_env() -> Result<Self> {
        Self::from_lookup(common_env::read_env_var_utf8)
    }

    pub fn validate(&self) -> Result<()> {
        self.gateway.validate()?;
        self.llm.validate()?;
        Ok(())
    }

    fn from_lookup<F>(mut lookup: F) -> Result<Self>
    where
        F: FnMut(&'static str) -> Result<Option<String>>,
    {
        let mut config = Self::default();
        config.gateway.apply_env(&mut lookup)?;
        config.llm.apply_env(&mut lookup)?;
        config.validate()?;
        Ok(config)
    }

    fn load_with_lookup<F>(mut lookup: F) -> Result<Self>
    where
        F: FnMut(&'static str) -> Result<Option<String>>,
    {
        let mut config = Self::default();
        let config_path = match lookup(LANYTE_CONFIG_PATH_ENV)? {
            Some(path) => PathBuf::from(common_env::normalize_nonempty(
                path,
                LANYTE_CONFIG_PATH_ENV,
            )?),
            None => default_config_path()?,
        };
        config.apply_config_file(&config_path)?;
        config.gateway.apply_env(&mut lookup)?;
        config.llm.apply_env(&mut lookup)?;
        config.validate()?;
        Ok(config)
    }

    fn apply_config_file(&mut self, path: &Path) -> Result<()> {
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(source) => {
                return Err(CommonError::ConfigFileRead {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        let file_config = parse_file_config(path, &contents)?;

        if let Some(gateway) = file_config.gateway {
            self.gateway.apply_file(gateway)?;
        }
        if let Some(llm) = file_config.llm {
            self.llm.apply_file(llm)?;
        }

        Ok(())
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    gateway: Option<FileGatewayConfig>,
    #[serde(default)]
    llm: Option<FileLlmConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileGatewayConfig {
    #[serde(default)]
    core_peer_id: Option<String>,
    #[serde(default)]
    socket_path: Option<String>,
    #[serde(default)]
    crucible_schemas_dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileLlmConfig {
    #[serde(default)]
    claude: Option<FileClaudeConfig>,
    #[serde(default)]
    grok: Option<FileGrokConfig>,
    #[serde(default)]
    openai: Option<FileOpenAiConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileClaudeConfig {
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileGrokConfig {
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileOpenAiConfig {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
}

fn default_config_path() -> Result<PathBuf> {
    let home = env::var_os("HOME").ok_or_else(|| CommonError::ConfigPathUnavailable {
        reason: "HOME is not set".to_owned(),
    })?;
    Ok(PathBuf::from(home).join(DEFAULT_CONFIG_RELATIVE_PATH))
}

fn parse_file_config(path: &Path, contents: &str) -> Result<FileConfig> {
    toml::from_str(contents).map_err(|err: toml::de::Error| CommonError::ConfigFileParse {
        path: path.to_path_buf(),
        reason: err.to_string(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub core_peer_id: String,
    pub socket_path: PathBuf,
    pub crucible_schemas_dir: PathBuf,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            core_peer_id: DEFAULT_CORE_PEER_ID.to_owned(),
            socket_path: PathBuf::from(DEFAULT_GATEWAY_SOCKET_PATH),
            crucible_schemas_dir: PathBuf::from(DEFAULT_CRUCIBLE_SCHEMAS_DIR),
        }
    }
}

impl GatewayConfig {
    fn apply_file(&mut self, config: FileGatewayConfig) -> Result<()> {
        if let Some(core_peer_id) = config.core_peer_id {
            self.core_peer_id =
                common_env::normalize_nonempty(core_peer_id, CONFIG_GATEWAY_CORE_PEER_ID_FIELD)?;
        }
        if let Some(socket_path) = config.socket_path {
            self.socket_path = PathBuf::from(common_env::normalize_nonempty(
                socket_path,
                CONFIG_GATEWAY_SOCKET_PATH_FIELD,
            )?);
        }
        if let Some(crucible_schemas_dir) = config.crucible_schemas_dir {
            self.crucible_schemas_dir = PathBuf::from(common_env::normalize_nonempty(
                crucible_schemas_dir,
                CONFIG_GATEWAY_CRUCIBLE_SCHEMAS_DIR_FIELD,
            )?);
        }
        Ok(())
    }

    fn apply_env<F>(&mut self, lookup: &mut F) -> Result<()>
    where
        F: FnMut(&'static str) -> Result<Option<String>>,
    {
        if let Some(peer_id) = lookup(LANYTE_CORE_PEER_ID_ENV)? {
            self.core_peer_id = common_env::normalize_nonempty(peer_id, LANYTE_CORE_PEER_ID_ENV)?;
        }
        if let Some(socket_path) = lookup(LANYTE_GATEWAY_SOCKET_PATH_ENV)? {
            self.socket_path = PathBuf::from(common_env::normalize_nonempty(
                socket_path,
                LANYTE_GATEWAY_SOCKET_PATH_ENV,
            )?);
        }
        if let Some(schemas_dir) = lookup(LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV)? {
            self.crucible_schemas_dir = PathBuf::from(common_env::normalize_nonempty(
                schemas_dir,
                LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV,
            )?);
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.core_peer_id.trim().is_empty() {
            return Err(CommonError::EmptyConfigValue {
                field: LANYTE_CORE_PEER_ID_ENV,
            });
        }
        if self.socket_path.as_os_str().is_empty() {
            return Err(CommonError::EmptyConfigValue {
                field: LANYTE_GATEWAY_SOCKET_PATH_ENV,
            });
        }
        if self.crucible_schemas_dir.as_os_str().is_empty() {
            return Err(CommonError::EmptyConfigValue {
                field: LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LlmConfig {
    pub claude: ClaudeConfig,
    pub grok: GrokConfig,
    pub openai: OpenAiConfig,
}

impl LlmConfig {
    fn apply_file(&mut self, config: FileLlmConfig) -> Result<()> {
        if let Some(claude) = config.claude {
            self.claude.apply_file(claude)?;
        }
        if let Some(grok) = config.grok {
            self.grok.apply_file(grok)?;
        }
        if let Some(openai) = config.openai {
            self.openai.apply_file(openai)?;
        }
        Ok(())
    }

    fn apply_env<F>(&mut self, lookup: &mut F) -> Result<()>
    where
        F: FnMut(&'static str) -> Result<Option<String>>,
    {
        if let Some(model) = lookup(LLM_CLAUDE_MODEL_ENV)? {
            self.claude.model = common_env::normalize_nonempty(model, LLM_CLAUDE_MODEL_ENV)?;
        }
        if let Some(api_key) = lookup(LLM_CLAUDE_API_KEY_ENV)? {
            self.claude.api_key = Some(common_env::normalize_nonempty(
                api_key,
                LLM_CLAUDE_API_KEY_ENV,
            )?);
        }
        if let Some(model) = lookup(LLM_GROK_MODEL_ENV)? {
            self.grok.model = common_env::normalize_nonempty(model, LLM_GROK_MODEL_ENV)?;
        }
        if let Some(api_key) = lookup(LLM_GROK_API_KEY_ENV)? {
            self.grok.api_key = Some(common_env::normalize_nonempty(
                api_key,
                LLM_GROK_API_KEY_ENV,
            )?);
        }
        if let Some(model) = lookup(LLM_OPENAI_MODEL_ENV)? {
            self.openai.model = common_env::normalize_nonempty(model, LLM_OPENAI_MODEL_ENV)?;
        }
        if let Some(api_key) = lookup(LLM_OPENAI_API_KEY_ENV)? {
            self.openai.api_key = Some(common_env::normalize_nonempty(
                api_key,
                LLM_OPENAI_API_KEY_ENV,
            )?);
        }
        if let Some(base_url) = lookup(LLM_OPENAI_BASE_URL_ENV)? {
            self.openai.base_url =
                common_env::normalize_nonempty(base_url, LLM_OPENAI_BASE_URL_ENV)?;
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.claude.model.trim().is_empty() {
            return Err(CommonError::EmptyConfigValue {
                field: LLM_CLAUDE_MODEL_ENV,
            });
        }
        if let Some(api_key) = self.claude.api_key.as_deref() {
            if api_key.trim().is_empty() {
                return Err(CommonError::EmptyConfigValue {
                    field: LLM_CLAUDE_API_KEY_ENV,
                });
            }
        }
        if self.grok.model.trim().is_empty() {
            return Err(CommonError::EmptyConfigValue {
                field: LLM_GROK_MODEL_ENV,
            });
        }
        if let Some(api_key) = self.grok.api_key.as_deref() {
            if api_key.trim().is_empty() {
                return Err(CommonError::EmptyConfigValue {
                    field: LLM_GROK_API_KEY_ENV,
                });
            }
        }
        if self.openai.model.trim().is_empty() {
            return Err(CommonError::EmptyConfigValue {
                field: LLM_OPENAI_MODEL_ENV,
            });
        }
        if let Some(api_key) = self.openai.api_key.as_deref() {
            if api_key.trim().is_empty() {
                return Err(CommonError::EmptyConfigValue {
                    field: LLM_OPENAI_API_KEY_ENV,
                });
            }
        }
        if self.openai.base_url.trim().is_empty() {
            return Err(CommonError::EmptyConfigValue {
                field: LLM_OPENAI_BASE_URL_ENV,
            });
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeConfig {
    pub model: String,
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_CLAUDE_MODEL.to_owned(),
            api_key: None,
        }
    }
}

impl fmt::Debug for ClaudeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redacted_api_key = self.api_key.as_ref().map(|_| "<redacted>");
        f.debug_struct("ClaudeConfig")
            .field("model", &self.model)
            .field("api_key", &redacted_api_key)
            .finish()
    }
}

impl ClaudeConfig {
    fn apply_file(&mut self, config: FileClaudeConfig) -> Result<()> {
        if let Some(model) = config.model {
            self.model = common_env::normalize_nonempty(model, CONFIG_LLM_CLAUDE_MODEL_FIELD)?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrokConfig {
    pub model: String,
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
}

impl Default for GrokConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_GROK_MODEL.to_owned(),
            api_key: None,
        }
    }
}

impl fmt::Debug for GrokConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redacted_api_key = self.api_key.as_ref().map(|_| "<redacted>");
        f.debug_struct("GrokConfig")
            .field("model", &self.model)
            .field("api_key", &redacted_api_key)
            .finish()
    }
}

impl GrokConfig {
    fn apply_file(&mut self, config: FileGrokConfig) -> Result<()> {
        if let Some(model) = config.model {
            self.model = common_env::normalize_nonempty(model, CONFIG_LLM_GROK_MODEL_FIELD)?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiConfig {
    pub model: String,
    pub base_url: String,
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_OPENAI_MODEL.to_owned(),
            base_url: DEFAULT_OPENAI_BASE_URL.to_owned(),
            api_key: None,
        }
    }
}

impl fmt::Debug for OpenAiConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redacted_api_key = self.api_key.as_ref().map(|_| "<redacted>");
        f.debug_struct("OpenAiConfig")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &redacted_api_key)
            .finish()
    }
}

impl OpenAiConfig {
    fn apply_file(&mut self, config: FileOpenAiConfig) -> Result<()> {
        if let Some(model) = config.model {
            self.model = common_env::normalize_nonempty(model, CONFIG_LLM_OPENAI_MODEL_FIELD)?;
        }
        if let Some(base_url) = config.base_url {
            self.base_url =
                common_env::normalize_nonempty(base_url, CONFIG_LLM_OPENAI_BASE_URL_FIELD)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::env;
    use std::ffi::OsString;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_ID: AtomicUsize = AtomicUsize::new(0);

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let path = env::temp_dir().join(format!(
                "lanyte-common-config-test-{}-{}",
                process::id(),
                NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("temp dir should be created");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn config_from_pairs(pairs: &[(&str, &str)]) -> Result<LanyteConfig> {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        LanyteConfig::from_lookup(|key| Ok(map.get(key).map(|value| value.to_string())))
    }

    fn load_config_from_pairs(pairs: &[(&str, &str)]) -> Result<LanyteConfig> {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        LanyteConfig::load_with_lookup(|key| Ok(map.get(key).map(|value| value.to_string())))
    }

    #[test]
    fn default_config_matches_crt_bootstrap_expectations() {
        let cfg = LanyteConfig::default();
        assert_eq!(cfg.gateway.core_peer_id, "lanyte-core");
        assert_eq!(cfg.gateway.socket_path, PathBuf::from("/tmp/lanyte.sock"));
        assert_eq!(
            cfg.gateway.crucible_schemas_dir,
            PathBuf::from("../lanyte-crucible/schemas/ipc")
        );
        assert_eq!(cfg.llm.claude.model, "claude-sonnet-4-6");
        assert!(cfg.llm.claude.api_key.is_none());
        assert_eq!(cfg.llm.grok.model, "grok-4.20-beta-latest-reasoning");
        assert!(cfg.llm.grok.api_key.is_none());
        assert_eq!(cfg.llm.openai.model, "gpt-5.4");
        assert_eq!(cfg.llm.openai.base_url, "https://api.openai.com/v1");
        assert!(cfg.llm.openai.api_key.is_none());
    }

    #[test]
    fn env_values_override_defaults() {
        let cfg = config_from_pairs(&[
            (LANYTE_CORE_PEER_ID_ENV, "core-a"),
            (LANYTE_GATEWAY_SOCKET_PATH_ENV, "/tmp/custom.sock"),
            (LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV, "/opt/crucible/schemas/ipc"),
            (LLM_CLAUDE_MODEL_ENV, "claude-sonnet-4-6-20250101"),
            (LLM_CLAUDE_API_KEY_ENV, "test-key"),
            (LLM_GROK_MODEL_ENV, "grok-4.20-beta-latest-non-reasoning"),
            (LLM_GROK_API_KEY_ENV, "grok-key"),
            (LLM_OPENAI_MODEL_ENV, "gpt-4.1"),
            (LLM_OPENAI_API_KEY_ENV, "openai-key"),
            (LLM_OPENAI_BASE_URL_ENV, "http://127.0.0.1:11434/v1"),
        ])
        .expect("config should parse");

        assert_eq!(cfg.gateway.core_peer_id, "core-a");
        assert_eq!(cfg.gateway.socket_path, PathBuf::from("/tmp/custom.sock"));
        assert_eq!(
            cfg.gateway.crucible_schemas_dir,
            PathBuf::from("/opt/crucible/schemas/ipc")
        );
        assert_eq!(cfg.llm.claude.model, "claude-sonnet-4-6-20250101");
        assert_eq!(cfg.llm.claude.api_key.as_deref(), Some("test-key"));
        assert_eq!(cfg.llm.grok.model, "grok-4.20-beta-latest-non-reasoning");
        assert_eq!(cfg.llm.grok.api_key.as_deref(), Some("grok-key"));
        assert_eq!(cfg.llm.openai.model, "gpt-4.1");
        assert_eq!(cfg.llm.openai.api_key.as_deref(), Some("openai-key"));
        assert_eq!(cfg.llm.openai.base_url, "http://127.0.0.1:11434/v1");
    }

    #[test]
    fn load_uses_defaults_when_config_file_is_missing() {
        let temp_dir = TestDir::new();
        let missing_path = temp_dir.path().join("missing.toml");

        let cfg = load_config_from_pairs(&[(
            LANYTE_CONFIG_PATH_ENV,
            missing_path.to_str().expect("path should be utf-8"),
        )])
        .expect("missing config file should not fail");

        assert_eq!(cfg, LanyteConfig::default());
    }

    #[test]
    fn config_file_values_override_defaults() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        fs::write(
            &config_path,
            r#"
[gateway]
core_peer_id = "core-from-file"
socket_path = "/tmp/file.sock"
crucible_schemas_dir = "/opt/lanyte/crucible/schemas/ipc"

[llm.claude]
model = "claude-file"

[llm.grok]
model = "grok-file"

[llm.openai]
model = "gpt-file"
base_url = "https://example.test/v1"
"#,
        )
        .expect("config file should be written");

        let cfg = load_config_from_pairs(&[(
            LANYTE_CONFIG_PATH_ENV,
            config_path.to_str().expect("path should be utf-8"),
        )])
        .expect("config should load");

        assert_eq!(cfg.gateway.core_peer_id, "core-from-file");
        assert_eq!(cfg.gateway.socket_path, PathBuf::from("/tmp/file.sock"));
        assert_eq!(
            cfg.gateway.crucible_schemas_dir,
            PathBuf::from("/opt/lanyte/crucible/schemas/ipc")
        );
        assert_eq!(cfg.llm.claude.model, "claude-file");
        assert_eq!(cfg.llm.grok.model, "grok-file");
        assert_eq!(cfg.llm.openai.model, "gpt-file");
        assert_eq!(cfg.llm.openai.base_url, "https://example.test/v1");
    }

    #[test]
    fn env_values_override_config_file() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        fs::write(
            &config_path,
            r#"
[gateway]
core_peer_id = "core-from-file"

[llm.openai]
model = "gpt-file"
base_url = "https://example.test/v1"
"#,
        )
        .expect("config file should be written");

        let cfg = load_config_from_pairs(&[
            (
                LANYTE_CONFIG_PATH_ENV,
                config_path.to_str().expect("path should be utf-8"),
            ),
            (LANYTE_CORE_PEER_ID_ENV, "core-from-env"),
            (LLM_OPENAI_MODEL_ENV, "gpt-from-env"),
            (LLM_OPENAI_BASE_URL_ENV, "http://127.0.0.1:11434/v1"),
        ])
        .expect("config should load");

        assert_eq!(cfg.gateway.core_peer_id, "core-from-env");
        assert_eq!(cfg.llm.openai.model, "gpt-from-env");
        assert_eq!(cfg.llm.openai.base_url, "http://127.0.0.1:11434/v1");
    }

    #[test]
    fn malformed_config_file_returns_path_and_parse_location() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        fs::write(&config_path, "[gateway\ncore_peer_id = \"oops\"\n")
            .expect("config file should be written");

        let err = load_config_from_pairs(&[(
            LANYTE_CONFIG_PATH_ENV,
            config_path.to_str().expect("path should be utf-8"),
        )])
        .expect_err("malformed config should fail");

        match err {
            CommonError::ConfigFileParse { path, reason } => {
                assert_eq!(path, config_path);
                assert!(
                    reason.contains("line"),
                    "reason did not include location: {reason}"
                );
                assert!(
                    reason.contains("column"),
                    "reason did not include column: {reason}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn config_file_rejects_unknown_fields_like_api_keys() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        fs::write(
            &config_path,
            r#"
[llm.openai]
api_key = "should-not-be-here"
"#,
        )
        .expect("config file should be written");

        let err = load_config_from_pairs(&[(
            LANYTE_CONFIG_PATH_ENV,
            config_path.to_str().expect("path should be utf-8"),
        )])
        .expect_err("unknown fields should fail");

        match err {
            CommonError::ConfigFileParse { reason, .. } => {
                assert!(
                    reason.contains("unknown field`api_key`")
                        || reason.contains("unknown field `api_key`")
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn whitespace_only_env_value_is_rejected() {
        let err = config_from_pairs(&[(LANYTE_CORE_PEER_ID_ENV, "   ")]).expect_err("must fail");
        match err {
            CommonError::EmptyConfigValue { field } => {
                assert_eq!(field, LANYTE_CORE_PEER_ID_ENV);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn non_utf8_env_value_is_rejected() {
        let err = common_env::map_env_var_result(
            LANYTE_CORE_PEER_ID_ENV,
            Err(env::VarError::NotUnicode(OsString::from("bad-bytes"))),
        )
        .expect_err("must fail");
        match err {
            CommonError::InvalidEnvironment { key, reason } => {
                assert_eq!(key, LANYTE_CORE_PEER_ID_ENV);
                assert!(reason.contains("UTF-8"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let cfg = config_from_pairs(&[(LLM_CLAUDE_API_KEY_ENV, "top-secret-key")])
            .expect("config should parse");
        let debug_view = format!("{cfg:?}");
        assert!(!debug_view.contains("top-secret-key"));
        assert!(debug_view.contains("<redacted>"));
    }

    #[test]
    fn serialization_omits_api_key_field() {
        let cfg = config_from_pairs(&[(LLM_CLAUDE_API_KEY_ENV, "top-secret-key")])
            .expect("config should parse");
        let serialized = serde_json::to_string(&cfg).expect("serialization should work");
        assert!(!serialized.contains("top-secret-key"));
        assert!(!serialized.contains("api_key"));
    }
}
