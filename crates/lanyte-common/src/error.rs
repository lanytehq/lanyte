use std::path::PathBuf;

use thiserror::Error;

/// Shared error type for common configuration and bootstrap failures.
#[derive(Debug, Error)]
pub enum CommonError {
    #[error("configuration value for `{field}` must not be empty")]
    EmptyConfigValue { field: &'static str },

    #[error("failed to resolve configuration file path: {reason}")]
    ConfigPathUnavailable { reason: String },

    #[error("failed to read configuration file `{path}`: {source}")]
    ConfigFileRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid configuration file `{path}`: {reason}")]
    ConfigFileParse { path: PathBuf, reason: String },

    #[error("invalid value for environment variable `{key}`: {reason}")]
    InvalidEnvironment { key: &'static str, reason: String },
}

pub type Result<T> = std::result::Result<T, CommonError>;
