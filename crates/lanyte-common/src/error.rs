use thiserror::Error;

/// Shared error type for common configuration and bootstrap failures.
#[derive(Debug, Error)]
pub enum CommonError {
    #[error("configuration value for `{field}` must not be empty")]
    EmptyConfigValue { field: &'static str },

    #[error("invalid value for environment variable `{key}`: {reason}")]
    InvalidEnvironment { key: &'static str, reason: String },
}

pub type Result<T> = std::result::Result<T, CommonError>;
