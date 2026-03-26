use std::env;

use crate::error::{CommonError, Result};

/// Read an environment variable as UTF-8.
///
/// - Missing env vars return `Ok(None)`.
/// - Non-UTF-8 values are rejected with `CommonError::InvalidEnvironment`.
pub fn read_env_var_utf8(key: &'static str) -> Result<Option<String>> {
    map_env_var_result(key, env::var(key))
}

/// Map `std::env::var` results to the common error surface.
///
/// Exposed to make unit tests deterministic without mutating the process env.
pub fn map_env_var_result(
    key: &'static str,
    value: std::result::Result<String, env::VarError>,
) -> Result<Option<String>> {
    match value {
        Ok(v) => Ok(Some(v)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(CommonError::InvalidEnvironment {
            key,
            reason: "value must be valid UTF-8".to_owned(),
        }),
    }
}

/// Trim an input string and reject empty/whitespace-only values.
pub fn normalize_nonempty(input: String, key: &'static str) -> Result<String> {
    let trimmed = input.trim().to_owned();
    if trimmed.is_empty() {
        return Err(CommonError::EmptyConfigValue { field: key });
    }
    Ok(trimmed)
}
