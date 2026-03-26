use thiserror::Error;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Error)]
enum MainError {
    #[error(transparent)]
    Telemetry(#[from] lanyte_telemetry::TelemetryError),
    #[error(transparent)]
    Common(#[from] lanyte_common::CommonError),
    #[error(transparent)]
    Gateway(#[from] lanyte_gateway::GatewayError),
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

    let cfg = lanyte_common::LanyteConfig::from_env()?;
    let sock = cfg.gateway.socket_path.clone();
    let schemas = cfg.gateway.crucible_schemas_dir.clone();

    tracing::info!(
        socket_path = %sock.display(),
        crucible_schemas_dir = %schemas.display(),
        "starting gateway"
    );

    let (gateway, events) = lanyte_gateway::spawn(cfg.gateway)?;
    let orchestrator_cancel = CancellationToken::new();
    let orchestrator = lanyte_orchestrator::Orchestrator::new(
        events,
        orchestrator_cancel.clone(),
        gateway.responder(),
    );
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
