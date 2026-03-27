//! Shared constants and foundational types for the Lanyte core.

mod config;
pub mod env;
mod error;
pub mod naming;

pub use config::{
    ClaudeConfig, GatewayConfig, GrokConfig, LanyteConfig, LlmConfig, OpenAiConfig,
    LANYTE_CONFIG_PATH_ENV, LANYTE_CORE_PEER_ID_ENV, LANYTE_CRUCIBLE_SCHEMAS_DIR_ENV,
    LANYTE_GATEWAY_SOCKET_PATH_ENV, LLM_CLAUDE_API_KEY_ENV, LLM_CLAUDE_MODEL_ENV, LLM_CONFIG,
    LLM_GROK_API_KEY_ENV, LLM_GROK_MODEL_ENV, LLM_OPENAI_API_KEY_ENV, LLM_OPENAI_BASE_URL_ENV,
    LLM_OPENAI_MODEL_ENV,
};
pub use error::{CommonError, Result};
pub use naming::{
    validate_instance_name, validate_role_slug, validate_scope_path, validate_slug, NamingError,
};

pub type ChannelId = u16;

/// IPC channel assignments (matches lanyte-crucible/schemas/ipc/).
pub mod channels {
    use super::ChannelId;

    pub const CONTROL: ChannelId = 0;
    pub const COMMAND: ChannelId = 1;
    pub const TELEMETRY: ChannelId = 3;
    pub const ERROR: ChannelId = 4;
    pub const MAIL: ChannelId = 256;
    pub const PROXY: ChannelId = 257;
    pub const ADMIN: ChannelId = 258;
    pub const SKILL_IO: ChannelId = 259;
}

pub const CONTROL_CHANNELS: [ChannelId; 4] = [
    channels::CONTROL,
    channels::COMMAND,
    channels::TELEMETRY,
    channels::ERROR,
];

pub const PEER_SERVICE_CHANNELS: [ChannelId; 4] = [
    channels::MAIL,
    channels::PROXY,
    channels::ADMIN,
    channels::SKILL_IO,
];

pub const ALL_KNOWN_CHANNELS: [ChannelId; 8] = [
    channels::CONTROL,
    channels::COMMAND,
    channels::TELEMETRY,
    channels::ERROR,
    channels::MAIL,
    channels::PROXY,
    channels::ADMIN,
    channels::SKILL_IO,
];

#[must_use]
pub fn is_known_channel(channel: ChannelId) -> bool {
    ALL_KNOWN_CHANNELS.contains(&channel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_channels_are_unique() {
        let mut channels = ALL_KNOWN_CHANNELS.to_vec();
        channels.sort_unstable();
        channels.dedup();
        assert_eq!(channels.len(), ALL_KNOWN_CHANNELS.len());
    }

    #[test]
    fn peer_service_channels_match_expected_wire_ids() {
        assert_eq!(PEER_SERVICE_CHANNELS, [256, 257, 258, 259]);
    }
}
