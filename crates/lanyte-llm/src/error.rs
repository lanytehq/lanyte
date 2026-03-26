use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM API key is not configured")]
    MissingApiKey,

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("authentication failed")]
    AuthenticationFailed,

    #[error("rate limited")]
    RateLimited { retry_after: Option<Duration> },

    #[error("service unavailable")]
    ServiceUnavailable,

    #[error("invalid model")]
    InvalidModel,

    #[error("upstream returned HTTP {status}: {message}")]
    Upstream { status: u16, message: String },

    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),

    #[error("invalid response: {0}")]
    InvalidResponse(&'static str),

    #[error("unsupported message role: {0}")]
    UnsupportedMessageRole(&'static str),

    #[error("failed to parse JSON response: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, LlmError>;
