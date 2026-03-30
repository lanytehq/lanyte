use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

#[cfg(feature = "seclusor-secrets")]
use seclusor_crypto::{decrypt, load_identity_file};
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
pub const DEFAULT_SECRETS_AGE_FILENAME: &str = "secrets.age";
pub const DEFAULT_SECRETS_TOML_FILENAME: &str = "secrets.toml";
pub const DEFAULT_SECLUSOR_IDENTITY_RELATIVE_PATH: &str = ".config/seclusor/identity.txt";
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
const SECRETS_LLM_CLAUDE_API_KEY_FIELD: &str = "llm.claude.api_key";
const CONFIG_LLM_GROK_MODEL_FIELD: &str = "llm.grok.model";
const SECRETS_LLM_GROK_API_KEY_FIELD: &str = "llm.grok.api_key";
const CONFIG_LLM_OPENAI_MODEL_FIELD: &str = "llm.openai.model";
const CONFIG_LLM_OPENAI_BASE_URL_FIELD: &str = "llm.openai.base_url";
const SECRETS_LLM_OPENAI_API_KEY_FIELD: &str = "llm.openai.api_key";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigWithProvenance {
    pub config: LanyteConfig,
    pub provenance: BTreeMap<String, ConfigSource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    Default,
    ConfigFile,
    SecretsFile,
    EnvVar(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LanyteConfig {
    pub gateway: GatewayConfig,
    pub llm: LlmConfig,
}

impl LanyteConfig {
    /// Load layered runtime configuration with the precedence:
    /// defaults -> config.toml -> secrets.toml -> secrets.age -> env vars.
    pub fn load() -> Result<Self> {
        Ok(Self::load_with_provenance()?.config)
    }

    pub fn load_with_provenance() -> Result<ConfigWithProvenance> {
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
        let mut provenance = default_provenance();
        config.gateway.apply_env(&mut lookup, &mut provenance)?;
        config.llm.apply_env(&mut lookup, &mut provenance)?;
        config.validate()?;
        Ok(config)
    }

    fn load_with_lookup<F>(mut lookup: F) -> Result<ConfigWithProvenance>
    where
        F: FnMut(&'static str) -> Result<Option<String>>,
    {
        let mut config = Self::default();
        let mut provenance = default_provenance();
        let config_path = match lookup(LANYTE_CONFIG_PATH_ENV)? {
            Some(path) => PathBuf::from(common_env::normalize_nonempty(
                path,
                LANYTE_CONFIG_PATH_ENV,
            )?),
            None => default_config_path()?,
        };
        config.apply_config_file(&config_path, &mut provenance)?;
        let secrets_dir = config_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        config.apply_secrets_file(
            &secrets_dir.join(DEFAULT_SECRETS_TOML_FILENAME),
            &mut provenance,
        )?;
        config.apply_encrypted_secrets_file(
            &secrets_dir.join(DEFAULT_SECRETS_AGE_FILENAME),
            &trusted_seclusor_identity_path()?,
            &mut provenance,
        )?;
        config.gateway.apply_env(&mut lookup, &mut provenance)?;
        config.llm.apply_env(&mut lookup, &mut provenance)?;
        config.validate()?;
        Ok(ConfigWithProvenance { config, provenance })
    }

    fn apply_config_file(
        &mut self,
        path: &Path,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
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
            self.gateway.apply_file(gateway, provenance)?;
        }
        if let Some(llm) = file_config.llm {
            self.llm.apply_file(llm, provenance)?;
        }

        Ok(())
    }

    fn apply_secrets_file(
        &mut self,
        path: &Path,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(source) => {
                return Err(CommonError::SecretsFileRead {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };

        let secrets = parse_secrets_file(path, &contents)?;
        self.apply_secrets(secrets, provenance)
    }

    fn apply_secrets(
        &mut self,
        secrets: FileSecrets,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(llm) = secrets.llm {
            self.llm.apply_secrets(llm, provenance)?;
        }
        Ok(())
    }

    #[cfg(feature = "seclusor-secrets")]
    fn apply_encrypted_secrets_file(
        &mut self,
        path: &Path,
        identity_path: &Path,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        let ciphertext = match fs::read(path) {
            Ok(contents) => contents,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(source) => {
                return Err(CommonError::SecretsFileRead {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };

        let identities =
            load_identity_file(identity_path).map_err(|err| CommonError::SecretsIdentityLoad {
                path: identity_path.to_path_buf(),
                reason: err.to_string(),
            })?;
        let plaintext =
            decrypt(&ciphertext, &identities).map_err(|err| CommonError::SecretsFileDecrypt {
                path: path.to_path_buf(),
                reason: err.to_string(),
            })?;
        let contents =
            String::from_utf8(plaintext).map_err(|err| CommonError::SecretsFileParse {
                path: path.to_path_buf(),
                reason: format!("decrypted secrets are not valid UTF-8: {err}"),
            })?;
        let secrets = parse_secrets_file(path, &contents)?;
        self.apply_secrets(secrets, provenance)
    }

    #[cfg(not(feature = "seclusor-secrets"))]
    fn apply_encrypted_secrets_file(
        &mut self,
        path: &Path,
        _identity_path: &Path,
        _provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        match fs::metadata(path) {
            Ok(metadata) if metadata.is_file() => {
                tracing::info!(
                    path = %path.display(),
                    "skipping encrypted secrets file because lanyte-common was built without seclusor-secrets"
                );
            }
            Ok(_) => {}
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                tracing::debug!(
                    path = %path.display(),
                    error = %source,
                    "unable to inspect encrypted secrets file while seclusor-secrets is disabled"
                );
            }
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
struct FileSecrets {
    #[serde(default)]
    llm: Option<FileSecretsLlm>,
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
struct FileSecretsLlm {
    #[serde(default)]
    claude: Option<FileSecretsClaudeConfig>,
    #[serde(default)]
    grok: Option<FileSecretsGrokConfig>,
    #[serde(default)]
    openai: Option<FileSecretsOpenAiConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileClaudeConfig {
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSecretsClaudeConfig {
    #[serde(default)]
    api_key: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileGrokConfig {
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSecretsGrokConfig {
    #[serde(default)]
    api_key: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileOpenAiConfig {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSecretsOpenAiConfig {
    #[serde(default)]
    api_key: Option<String>,
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

fn parse_secrets_file(path: &Path, contents: &str) -> Result<FileSecrets> {
    toml::from_str(contents).map_err(|err: toml::de::Error| CommonError::SecretsFileParse {
        path: path.to_path_buf(),
        reason: err.to_string(),
    })
}

fn trusted_seclusor_identity_path() -> Result<PathBuf> {
    let home = env::var_os("HOME").ok_or_else(|| CommonError::ConfigPathUnavailable {
        reason: "HOME is not set".to_owned(),
    })?;
    Ok(PathBuf::from(home).join(DEFAULT_SECLUSOR_IDENTITY_RELATIVE_PATH))
}

fn default_provenance() -> BTreeMap<String, ConfigSource> {
    BTreeMap::from([
        (
            CONFIG_GATEWAY_CORE_PEER_ID_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            CONFIG_GATEWAY_SOCKET_PATH_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            CONFIG_GATEWAY_CRUCIBLE_SCHEMAS_DIR_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            CONFIG_LLM_CLAUDE_MODEL_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            SECRETS_LLM_CLAUDE_API_KEY_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            CONFIG_LLM_GROK_MODEL_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            SECRETS_LLM_GROK_API_KEY_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            CONFIG_LLM_OPENAI_MODEL_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            CONFIG_LLM_OPENAI_BASE_URL_FIELD.to_owned(),
            ConfigSource::Default,
        ),
        (
            SECRETS_LLM_OPENAI_API_KEY_FIELD.to_owned(),
            ConfigSource::Default,
        ),
    ])
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
    fn apply_file(
        &mut self,
        config: FileGatewayConfig,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(core_peer_id) = config.core_peer_id {
            self.core_peer_id =
                common_env::normalize_nonempty(core_peer_id, CONFIG_GATEWAY_CORE_PEER_ID_FIELD)?;
            provenance.insert(
                CONFIG_GATEWAY_CORE_PEER_ID_FIELD.to_owned(),
                ConfigSource::ConfigFile,
            );
        }
        if let Some(socket_path) = config.socket_path {
            self.socket_path = PathBuf::from(common_env::normalize_nonempty(
                socket_path,
                CONFIG_GATEWAY_SOCKET_PATH_FIELD,
            )?);
            provenance.insert(
                CONFIG_GATEWAY_SOCKET_PATH_FIELD.to_owned(),
                ConfigSource::ConfigFile,
            );
        }
        if let Some(crucible_schemas_dir) = config.crucible_schemas_dir {
            self.crucible_schemas_dir = PathBuf::from(common_env::normalize_nonempty(
                crucible_schemas_dir,
                CONFIG_GATEWAY_CRUCIBLE_SCHEMAS_DIR_FIELD,
            )?);
            provenance.insert(
                CONFIG_GATEWAY_CRUCIBLE_SCHEMAS_DIR_FIELD.to_owned(),
                ConfigSource::ConfigFile,
            );
        }
        Ok(())
    }

    fn apply_env<F>(
        &mut self,
        lookup: &mut F,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()>
    where
        F: FnMut(&'static str) -> Result<Option<String>>,
    {
        if let Some(peer_id) = lookup(LANYTE_CORE_PEER_ID_ENV)? {
            self.core_peer_id = common_env::normalize_nonempty(peer_id, LANYTE_CORE_PEER_ID_ENV)?;
            provenance.insert(
                CONFIG_GATEWAY_CORE_PEER_ID_FIELD.to_owned(),
                ConfigSource::EnvVar(LANYTE_CORE_PEER_ID_ENV.to_owned()),
            );
        }
        if let Some(socket_path) = lookup(LANYTE_GATEWAY_SOCKET_PATH_ENV)? {
            self.socket_path = PathBuf::from(common_env::normalize_nonempty(
                socket_path,
                LANYTE_GATEWAY_SOCKET_PATH_ENV,
            )?);
            provenance.insert(
                CONFIG_GATEWAY_SOCKET_PATH_FIELD.to_owned(),
                ConfigSource::EnvVar(LANYTE_GATEWAY_SOCKET_PATH_ENV.to_owned()),
            );
        }
        if let Some(schemas_dir) = lookup(LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV)? {
            self.crucible_schemas_dir = PathBuf::from(common_env::normalize_nonempty(
                schemas_dir,
                LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV,
            )?);
            provenance.insert(
                CONFIG_GATEWAY_CRUCIBLE_SCHEMAS_DIR_FIELD.to_owned(),
                ConfigSource::EnvVar(LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV.to_owned()),
            );
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
    fn apply_file(
        &mut self,
        config: FileLlmConfig,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(claude) = config.claude {
            self.claude.apply_file(claude, provenance)?;
        }
        if let Some(grok) = config.grok {
            self.grok.apply_file(grok, provenance)?;
        }
        if let Some(openai) = config.openai {
            self.openai.apply_file(openai, provenance)?;
        }
        Ok(())
    }

    fn apply_env<F>(
        &mut self,
        lookup: &mut F,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()>
    where
        F: FnMut(&'static str) -> Result<Option<String>>,
    {
        if let Some(model) = lookup(LLM_CLAUDE_MODEL_ENV)? {
            self.claude.model = common_env::normalize_nonempty(model, LLM_CLAUDE_MODEL_ENV)?;
            provenance.insert(
                CONFIG_LLM_CLAUDE_MODEL_FIELD.to_owned(),
                ConfigSource::EnvVar(LLM_CLAUDE_MODEL_ENV.to_owned()),
            );
        }
        if let Some(api_key) = lookup(LLM_CLAUDE_API_KEY_ENV)? {
            self.claude.api_key = Some(common_env::normalize_nonempty(
                api_key,
                LLM_CLAUDE_API_KEY_ENV,
            )?);
            provenance.insert(
                SECRETS_LLM_CLAUDE_API_KEY_FIELD.to_owned(),
                ConfigSource::EnvVar(LLM_CLAUDE_API_KEY_ENV.to_owned()),
            );
        }
        if let Some(model) = lookup(LLM_GROK_MODEL_ENV)? {
            self.grok.model = common_env::normalize_nonempty(model, LLM_GROK_MODEL_ENV)?;
            provenance.insert(
                CONFIG_LLM_GROK_MODEL_FIELD.to_owned(),
                ConfigSource::EnvVar(LLM_GROK_MODEL_ENV.to_owned()),
            );
        }
        if let Some(api_key) = lookup(LLM_GROK_API_KEY_ENV)? {
            self.grok.api_key = Some(common_env::normalize_nonempty(
                api_key,
                LLM_GROK_API_KEY_ENV,
            )?);
            provenance.insert(
                SECRETS_LLM_GROK_API_KEY_FIELD.to_owned(),
                ConfigSource::EnvVar(LLM_GROK_API_KEY_ENV.to_owned()),
            );
        }
        if let Some(model) = lookup(LLM_OPENAI_MODEL_ENV)? {
            self.openai.model = common_env::normalize_nonempty(model, LLM_OPENAI_MODEL_ENV)?;
            provenance.insert(
                CONFIG_LLM_OPENAI_MODEL_FIELD.to_owned(),
                ConfigSource::EnvVar(LLM_OPENAI_MODEL_ENV.to_owned()),
            );
        }
        if let Some(api_key) = lookup(LLM_OPENAI_API_KEY_ENV)? {
            self.openai.api_key = Some(common_env::normalize_nonempty(
                api_key,
                LLM_OPENAI_API_KEY_ENV,
            )?);
            provenance.insert(
                SECRETS_LLM_OPENAI_API_KEY_FIELD.to_owned(),
                ConfigSource::EnvVar(LLM_OPENAI_API_KEY_ENV.to_owned()),
            );
        }
        if let Some(base_url) = lookup(LLM_OPENAI_BASE_URL_ENV)? {
            self.openai.base_url =
                common_env::normalize_nonempty(base_url, LLM_OPENAI_BASE_URL_ENV)?;
            provenance.insert(
                CONFIG_LLM_OPENAI_BASE_URL_FIELD.to_owned(),
                ConfigSource::EnvVar(LLM_OPENAI_BASE_URL_ENV.to_owned()),
            );
        }
        Ok(())
    }

    fn apply_secrets(
        &mut self,
        config: FileSecretsLlm,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(claude) = config.claude {
            self.claude.apply_secrets(claude, provenance)?;
        }
        if let Some(grok) = config.grok {
            self.grok.apply_secrets(grok, provenance)?;
        }
        if let Some(openai) = config.openai {
            self.openai.apply_secrets(openai, provenance)?;
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
    fn apply_file(
        &mut self,
        config: FileClaudeConfig,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(model) = config.model {
            self.model = common_env::normalize_nonempty(model, CONFIG_LLM_CLAUDE_MODEL_FIELD)?;
            provenance.insert(
                CONFIG_LLM_CLAUDE_MODEL_FIELD.to_owned(),
                ConfigSource::ConfigFile,
            );
        }
        Ok(())
    }

    fn apply_secrets(
        &mut self,
        config: FileSecretsClaudeConfig,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(api_key) = config.api_key {
            self.api_key = Some(common_env::normalize_nonempty(
                api_key,
                SECRETS_LLM_CLAUDE_API_KEY_FIELD,
            )?);
            provenance.insert(
                SECRETS_LLM_CLAUDE_API_KEY_FIELD.to_owned(),
                ConfigSource::SecretsFile,
            );
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
    fn apply_file(
        &mut self,
        config: FileGrokConfig,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(model) = config.model {
            self.model = common_env::normalize_nonempty(model, CONFIG_LLM_GROK_MODEL_FIELD)?;
            provenance.insert(
                CONFIG_LLM_GROK_MODEL_FIELD.to_owned(),
                ConfigSource::ConfigFile,
            );
        }
        Ok(())
    }

    fn apply_secrets(
        &mut self,
        config: FileSecretsGrokConfig,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(api_key) = config.api_key {
            self.api_key = Some(common_env::normalize_nonempty(
                api_key,
                SECRETS_LLM_GROK_API_KEY_FIELD,
            )?);
            provenance.insert(
                SECRETS_LLM_GROK_API_KEY_FIELD.to_owned(),
                ConfigSource::SecretsFile,
            );
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
    fn apply_file(
        &mut self,
        config: FileOpenAiConfig,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(model) = config.model {
            self.model = common_env::normalize_nonempty(model, CONFIG_LLM_OPENAI_MODEL_FIELD)?;
            provenance.insert(
                CONFIG_LLM_OPENAI_MODEL_FIELD.to_owned(),
                ConfigSource::ConfigFile,
            );
        }
        if let Some(base_url) = config.base_url {
            self.base_url =
                common_env::normalize_nonempty(base_url, CONFIG_LLM_OPENAI_BASE_URL_FIELD)?;
            provenance.insert(
                CONFIG_LLM_OPENAI_BASE_URL_FIELD.to_owned(),
                ConfigSource::ConfigFile,
            );
        }
        Ok(())
    }

    fn apply_secrets(
        &mut self,
        config: FileSecretsOpenAiConfig,
        provenance: &mut BTreeMap<String, ConfigSource>,
    ) -> Result<()> {
        if let Some(api_key) = config.api_key {
            self.api_key = Some(common_env::normalize_nonempty(
                api_key,
                SECRETS_LLM_OPENAI_API_KEY_FIELD,
            )?);
            provenance.insert(
                SECRETS_LLM_OPENAI_API_KEY_FIELD.to_owned(),
                ConfigSource::SecretsFile,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "seclusor-secrets")]
    use seclusor_crypto::{encrypt, identity_to_string, Identity};
    use std::collections::HashMap;
    use std::env;
    use std::ffi::OsString;
    use std::fs;
    #[cfg(all(unix, feature = "seclusor-secrets"))]
    use std::fs::OpenOptions;
    #[cfg(all(unix, feature = "seclusor-secrets"))]
    use std::io::Write;
    #[cfg(all(unix, feature = "seclusor-secrets"))]
    use std::os::unix::fs::OpenOptionsExt;
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
            .map(|loaded| loaded.config)
    }

    fn load_config_with_provenance_from_pairs(
        pairs: &[(&str, &str)],
    ) -> Result<ConfigWithProvenance> {
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
    fn secrets_file_populates_api_keys_only() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        let secrets_path = temp_dir.path().join(DEFAULT_SECRETS_TOML_FILENAME);
        fs::write(&config_path, "").expect("config file should be written");
        fs::write(
            &secrets_path,
            r#"
[llm.claude]
api_key = "claude-secret"

[llm.openai]
api_key = "openai-secret"
"#,
        )
        .expect("secrets file should be written");

        let cfg = load_config_from_pairs(&[(
            LANYTE_CONFIG_PATH_ENV,
            config_path.to_str().expect("path should be utf-8"),
        )])
        .expect("config should load");

        assert_eq!(cfg.llm.claude.api_key.as_deref(), Some("claude-secret"));
        assert_eq!(cfg.llm.openai.api_key.as_deref(), Some("openai-secret"));
        assert!(cfg.llm.grok.api_key.is_none());
    }

    #[test]
    fn env_api_keys_override_secrets_files() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        let secrets_path = temp_dir.path().join(DEFAULT_SECRETS_TOML_FILENAME);
        fs::write(&config_path, "").expect("config file should be written");
        fs::write(
            &secrets_path,
            r#"
[llm.claude]
api_key = "file-secret"
"#,
        )
        .expect("secrets file should be written");

        let cfg = load_config_from_pairs(&[
            (
                LANYTE_CONFIG_PATH_ENV,
                config_path.to_str().expect("path should be utf-8"),
            ),
            (LLM_CLAUDE_API_KEY_ENV, "env-secret"),
        ])
        .expect("config should load");

        assert_eq!(cfg.llm.claude.api_key.as_deref(), Some("env-secret"));
    }

    #[test]
    fn load_with_provenance_defaults_all_fields_to_default_source() {
        let loaded = load_config_with_provenance_from_pairs(&[]).expect("config should load");

        assert_eq!(
            loaded.provenance.get(CONFIG_GATEWAY_CORE_PEER_ID_FIELD),
            Some(&ConfigSource::Default)
        );
        assert_eq!(
            loaded.provenance.get(CONFIG_LLM_OPENAI_BASE_URL_FIELD),
            Some(&ConfigSource::Default)
        );
        assert_eq!(
            loaded.provenance.get(SECRETS_LLM_CLAUDE_API_KEY_FIELD),
            Some(&ConfigSource::Default)
        );
    }

    #[test]
    fn load_with_provenance_tracks_layer_winners_per_field() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        let secrets_path = temp_dir.path().join(DEFAULT_SECRETS_TOML_FILENAME);
        fs::write(
            &config_path,
            r#"
[gateway]
core_peer_id = "core-from-file"

[llm.openai]
base_url = "https://file.example/v1"
model = "openai-file"
"#,
        )
        .expect("config file should be written");
        fs::write(
            &secrets_path,
            r#"
[llm.claude]
api_key = "claude-secret"
"#,
        )
        .expect("secrets file should be written");

        let loaded = load_config_with_provenance_from_pairs(&[
            (
                LANYTE_CONFIG_PATH_ENV,
                config_path.to_str().expect("path should be utf-8"),
            ),
            (LLM_OPENAI_MODEL_ENV, "openai-env"),
        ])
        .expect("config should load");

        assert_eq!(
            loaded.provenance.get(CONFIG_GATEWAY_CORE_PEER_ID_FIELD),
            Some(&ConfigSource::ConfigFile)
        );
        assert_eq!(
            loaded.provenance.get(CONFIG_LLM_OPENAI_BASE_URL_FIELD),
            Some(&ConfigSource::ConfigFile)
        );
        assert_eq!(
            loaded.provenance.get(CONFIG_LLM_OPENAI_MODEL_FIELD),
            Some(&ConfigSource::EnvVar(LLM_OPENAI_MODEL_ENV.to_owned()))
        );
        assert_eq!(
            loaded.provenance.get(SECRETS_LLM_CLAUDE_API_KEY_FIELD),
            Some(&ConfigSource::SecretsFile)
        );
    }

    #[test]
    fn secrets_file_rejects_non_secret_fields() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        let secrets_path = temp_dir.path().join(DEFAULT_SECRETS_TOML_FILENAME);
        fs::write(&config_path, "").expect("config file should be written");
        fs::write(
            &secrets_path,
            r#"
[llm.openai]
model = "should-not-be-here"
"#,
        )
        .expect("secrets file should be written");

        let err = load_config_from_pairs(&[(
            LANYTE_CONFIG_PATH_ENV,
            config_path.to_str().expect("path should be utf-8"),
        )])
        .expect_err("unknown fields should fail");

        match err {
            CommonError::SecretsFileParse { reason, .. } => {
                assert!(
                    reason.contains("unknown field`model`")
                        || reason.contains("unknown field `model`")
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[cfg(not(feature = "seclusor-secrets"))]
    #[test]
    fn encrypted_secrets_file_is_ignored_without_feature() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        let secrets_age_path = temp_dir.path().join(DEFAULT_SECRETS_AGE_FILENAME);
        fs::write(&config_path, "").expect("config file should be written");
        fs::write(&secrets_age_path, "not-a-real-ciphertext")
            .expect("encrypted secrets placeholder");

        let cfg = load_config_from_pairs(&[(
            LANYTE_CONFIG_PATH_ENV,
            config_path.to_str().expect("path should be utf-8"),
        )])
        .expect("encrypted secrets should be ignored when feature is disabled");

        assert!(cfg.llm.claude.api_key.is_none());
        assert!(cfg.llm.grok.api_key.is_none());
        assert!(cfg.llm.openai.api_key.is_none());
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

    #[cfg(all(unix, feature = "seclusor-secrets"))]
    fn write_identity_file(path: &Path, contents: &str) {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .expect("identity file should be created");
        file.write_all(contents.as_bytes())
            .expect("identity file should be written");
    }

    #[cfg(feature = "seclusor-secrets")]
    fn encrypted_secret_bytes(api_key: &str, identity: &Identity) -> Vec<u8> {
        let recipients = vec![identity.to_public()];
        encrypt(
            format!(
                r#"
[llm.grok]
api_key = "{api_key}"
"#
            )
            .as_bytes(),
            &recipients,
        )
        .expect("secrets should encrypt")
    }

    #[cfg(feature = "seclusor-secrets")]
    #[test]
    fn encrypted_secrets_override_plaintext_secrets_when_feature_enabled() {
        let temp_dir = TestDir::new();
        let config_path = temp_dir.path().join("config.toml");
        let secrets_path = temp_dir.path().join(DEFAULT_SECRETS_TOML_FILENAME);
        let secrets_age_path = temp_dir.path().join(DEFAULT_SECRETS_AGE_FILENAME);
        let identity_path = temp_dir.path().join("identity.txt");
        fs::write(&config_path, "").expect("config file should be written");
        fs::write(
            &secrets_path,
            r#"
[llm.grok]
api_key = "plaintext-secret"
"#,
        )
        .expect("secrets file should be written");

        let identity = Identity::generate();
        #[cfg(unix)]
        write_identity_file(
            &identity_path,
            &format!("{}\n", identity_to_string(&identity)),
        );
        #[cfg(not(unix))]
        fs::write(
            &identity_path,
            format!("{}\n", identity_to_string(&identity)),
        )
        .expect("identity file should be written");
        fs::write(
            &secrets_age_path,
            encrypted_secret_bytes("encrypted-secret", &identity),
        )
        .expect("encrypted secrets file should be written");

        let mut cfg = LanyteConfig::default();
        let mut provenance = default_provenance();
        cfg.apply_secrets_file(&secrets_path, &mut provenance)
            .expect("plaintext secrets should load");
        cfg.apply_encrypted_secrets_file(&secrets_age_path, &identity_path, &mut provenance)
            .expect("encrypted secrets should load");

        assert_eq!(cfg.llm.grok.api_key.as_deref(), Some("encrypted-secret"));
        assert_eq!(
            provenance.get(SECRETS_LLM_GROK_API_KEY_FIELD),
            Some(&ConfigSource::SecretsFile)
        );
    }
}
