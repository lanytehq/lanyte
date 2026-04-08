use std::collections::BTreeMap;
use std::sync::Arc;
use std::{env, path::PathBuf};

use thiserror::Error;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Error)]
enum MainError {
    #[error(transparent)]
    Telemetry(#[from] lanyte_telemetry::TelemetryError),
    #[error(transparent)]
    Common(#[from] lanyte_common::CommonError),
    #[error(transparent)]
    Llm(#[from] lanyte_llm::LlmError),
    #[error(transparent)]
    Gateway(#[from] lanyte_gateway::GatewayError),
    #[error(transparent)]
    State(#[from] lanyte_state::StateError),
    #[error("failed to resolve local audit state root: {0}")]
    StateBootstrap(String),
    #[error(transparent)]
    Orchestrator(#[from] lanyte_orchestrator::OrchestratorError),
    #[error("failed waiting for Ctrl-C: {0}")]
    CtrlC(#[from] std::io::Error),
    #[error("orchestrator task join failed: {0}")]
    OrchestratorJoin(String),
}

#[tokio::main]
async fn main() -> Result<(), MainError> {
    lanyte_telemetry::init_tracing()?;

    tracing::info!(version = env!("CARGO_PKG_VERSION"), "lanyte starting");

    let loaded_cfg = lanyte_common::LanyteConfig::load_with_provenance()?;
    tracing::debug!(provenance = ?loaded_cfg.provenance, "resolved configuration provenance");
    let cfg = loaded_cfg.config;
    let sock = cfg.gateway.socket_path.clone();
    let schemas = cfg.gateway.crucible_schemas_dir.clone();

    tracing::info!(
        socket_path = %sock.display(),
        crucible_schemas_dir = %schemas.display(),
        "starting gateway"
    );

    let llm = build_llm_backends(&cfg.llm)?;
    let audit_store = Arc::new(std::sync::Mutex::new(open_audit_store()?));
    if let Some(backends) = &llm {
        tracing::info!(
            default_provider = %backends.default_provider(),
            configured_providers = ?backends.configured_providers(),
            "configured orchestrator LLM backends"
        );
    } else {
        tracing::warn!("no LLM backend configured; llm.complete command will return invoke_error");
    }

    let (gateway, events) = lanyte_gateway::spawn(cfg.gateway)?;
    let orchestrator_cancel = CancellationToken::new();
    let orchestrator = lanyte_orchestrator::Orchestrator::new(
        events,
        orchestrator_cancel.clone(),
        gateway.responder(),
        llm,
    )
    .with_audit_store(audit_store);
    let mut gateway = Some(gateway);
    let orchestrator_task = tokio::spawn(orchestrator.run());
    tokio::pin!(orchestrator_task);

    tokio::select! {
        res = tokio::signal::ctrl_c() => {
            res?;
            tracing::info!("Ctrl-C received; shutting down");
            if let Some(handle) = gateway.take() {
                handle.cancel();
                handle.wait().await?;
            }
            orchestrator_cancel.cancel();
        }
        res = &mut orchestrator_task => {
            match res {
                Ok(run_res) => run_res?,
                Err(err) => return Err(MainError::OrchestratorJoin(err.to_string())),
            }
        }
    }

    if !orchestrator_task.is_finished() {
        if let Some(handle) = gateway.take() {
            handle.cancel();
            handle.wait().await?;
        }
    }

    if !orchestrator_task.is_finished() {
        match orchestrator_task.await {
            Ok(run_res) => run_res?,
            Err(err) => return Err(MainError::OrchestratorJoin(err.to_string())),
        }
    }

    Ok(())
}

fn build_llm_backends(
    cfg: &lanyte_common::LlmConfig,
) -> Result<Option<lanyte_orchestrator::ConfiguredBackends>, lanyte_llm::LlmError> {
    let Some(default_provider) = cfg.resolved_default_provider() else {
        return Ok(None);
    };

    let mut backends = BTreeMap::new();
    for provider in cfg.configured_providers() {
        let backend: Arc<dyn lanyte_llm::LlmBackend> = match provider {
            lanyte_common::ProviderKind::Claude => {
                Arc::new(lanyte_llm::ClaudeBackend::from_config(&cfg.claude)?)
            }
            lanyte_common::ProviderKind::OpenAi => {
                Arc::new(lanyte_llm::OpenAiBackend::from_config(&cfg.openai)?)
            }
            lanyte_common::ProviderKind::Grok => {
                Arc::new(lanyte_llm::GrokBackend::from_config(&cfg.grok)?)
            }
        };
        backends.insert(provider, backend);
    }

    Ok(Some(
        lanyte_orchestrator::ConfiguredBackends::new(default_provider, backends)
            .expect("resolved default provider must be configured"),
    ))
}

fn open_audit_store() -> Result<lanyte_state::StateStore, MainError> {
    let explicit_root = env::var_os(lanyte_state::LANYTE_STATE_ROOT_ENV).is_some();
    if explicit_root {
        return Ok(lanyte_state::StateStore::open_default()?);
    }

    match lanyte_state::StateStore::open_default() {
        Ok(store) => Ok(store),
        Err(lanyte_state::StateError::Io(err))
            if err.kind() == std::io::ErrorKind::PermissionDenied =>
        {
            let fallback_root = fallback_audit_state_root(env::var_os("HOME"))?;
            tracing::warn!(
                default_root = lanyte_state::DEFAULT_STATE_ROOT,
                fallback_root = %fallback_root.display(),
                "default audit state root is not writable; using local fallback"
            );
            Ok(lanyte_state::StateStore::open(
                lanyte_state::StatePaths::new(fallback_root),
            )?)
        }
        Err(err) => Err(err.into()),
    }
}

fn fallback_audit_state_root(home: Option<std::ffi::OsString>) -> Result<PathBuf, MainError> {
    let home = home.ok_or_else(|| {
        MainError::StateBootstrap(
            "HOME is not set and default audit state root is unavailable".to_owned(),
        )
    })?;
    Ok(PathBuf::from(home).join(".local/state/lanyte"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn build_llm_backends_uses_compatibility_order_when_default_is_unset() {
        let mut cfg = lanyte_common::LanyteConfig::default();
        cfg.llm.openai.api_key = Some("openai-key".to_owned());
        cfg.llm.grok.api_key = Some("grok-key".to_owned());

        let backends = build_llm_backends(&cfg.llm)
            .expect("backends should build")
            .expect("backends should exist");

        assert_eq!(
            backends.default_provider(),
            lanyte_common::ProviderKind::OpenAi
        );
        assert_eq!(
            backends.configured_providers(),
            vec![
                lanyte_common::ProviderKind::OpenAi,
                lanyte_common::ProviderKind::Grok,
            ]
        );
    }

    #[test]
    fn build_llm_backends_honors_explicit_default_provider() {
        let mut cfg = lanyte_common::LanyteConfig::default();
        cfg.llm.default_provider = Some(lanyte_common::ProviderKind::OpenAi);
        cfg.llm.claude.api_key = Some("claude-key".to_owned());
        cfg.llm.openai.api_key = Some("openai-key".to_owned());

        let backends = build_llm_backends(&cfg.llm)
            .expect("backends should build")
            .expect("backends should exist");

        assert_eq!(
            backends.default_provider(),
            lanyte_common::ProviderKind::OpenAi
        );
        assert_eq!(
            backends.configured_providers(),
            vec![
                lanyte_common::ProviderKind::Claude,
                lanyte_common::ProviderKind::OpenAi,
            ]
        );
    }

    #[test]
    fn fallback_audit_state_root_uses_home_local_state() {
        let path = fallback_audit_state_root(Some(OsString::from("/Users/tester")))
            .expect("fallback path should resolve");
        assert_eq!(path, PathBuf::from("/Users/tester/.local/state/lanyte"));
    }

    #[test]
    fn fallback_audit_state_root_requires_home() {
        let err = fallback_audit_state_root(None).expect_err("missing HOME should fail");
        assert!(err.to_string().contains("HOME is not set"));
    }
}
