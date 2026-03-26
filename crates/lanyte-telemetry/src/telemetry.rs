use std::sync::OnceLock;

use thiserror::Error;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

pub const DEFAULT_LOG_FILTER: &str = "info";
static TRACING_INIT: OnceLock<Result<(), TelemetryError>> = OnceLock::new();

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TelemetryError {
    #[error("failed to initialize tracing subscriber: {0}")]
    SubscriberInit(String),
}

/// Initialize global tracing subscriber with JSON output and env-filter support.
///
/// This function is idempotent when called multiple times by lanyte components.
pub fn init_tracing() -> Result<(), TelemetryError> {
    initialize_once(&TRACING_INIT, || {
        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER));

        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_env_filter(env_filter)
            .with_current_span(true)
            .with_span_events(FmtSpan::CLOSE)
            .with_target(true)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .map_err(|err| TelemetryError::SubscriberInit(err.to_string()))
    })
}

fn initialize_once(
    gate: &OnceLock<Result<(), TelemetryError>>,
    init: impl FnOnce() -> Result<(), TelemetryError>,
) -> Result<(), TelemetryError> {
    match gate.get_or_init(init) {
        Ok(()) => Ok(()),
        Err(err) => Err(err.clone()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    use super::*;

    #[test]
    fn initialize_once_is_concurrency_safe() {
        let gate = Arc::new(OnceLock::new());
        let init_calls = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for _ in 0..48 {
            let gate = Arc::clone(&gate);
            let init_calls = Arc::clone(&init_calls);
            handles.push(thread::spawn(move || {
                initialize_once(&gate, || {
                    init_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }));
        }

        for handle in handles {
            assert!(handle.join().expect("thread panicked").is_ok());
        }

        assert_eq!(init_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn initialize_once_replays_first_error() {
        let gate = OnceLock::new();
        let first_error = TelemetryError::SubscriberInit("boom".to_owned());

        assert_eq!(
            initialize_once(&gate, || Err(first_error.clone())),
            Err(first_error.clone())
        );
        assert_eq!(
            initialize_once(&gate, || Ok(())),
            Err(first_error),
            "once initialized with an error, subsequent calls should observe the same result"
        );
    }
}
