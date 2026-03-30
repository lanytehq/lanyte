//! Async IPC gateway for Lanyte core.
//!
//! Responsibilities (CRT-004):
//! - bind/listen for peer connections
//! - load schemas and attach validation at the IPC boundary
//! - route validated frames to downstream components (orchestrator) via an event channel
//!
//! Non-responsibilities:
//! - no business logic, no policy, no stateful decisions beyond hard validation/allowlists

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

use std::collections::HashMap;
use std::future::Future;
use std::io::ErrorKind;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use ipcprims::frame::Frame;
use ipcprims::peer::{AsyncPeer, AsyncPeerListener, AsyncPeerTx, PeerError};
use ipcprims::schema::{RegistryConfig, SchemaRegistry};
use ipcprims::transport::TransportError;
use lanyte_common::{channels, ChannelId, GatewayConfig};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("gateway is only supported on Unix platforms")]
    UnsupportedPlatform,

    #[error("failed to load IPC schemas: {0}")]
    Schema(#[from] ipcprims::schema::SchemaError),

    #[error("peer error: {0}")]
    Peer(#[from] ipcprims::peer::PeerError),

    #[error("gateway task join failed: {0}")]
    Join(String),
}

/// A validated inbound frame, ready for downstream routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayEvent {
    pub peer_id: String,
    pub channel: ChannelId,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerResponse {
    pub peer_id: String,
    pub channel: ChannelId,
    pub payload: Vec<u8>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PeerSendError {
    #[error("unknown peer_id: {0}")]
    UnknownPeer(String),

    #[error("peer disconnected: {0}")]
    PeerDisconnected(String),

    #[error("response channel closed (gateway shutting down)")]
    ChannelClosed,
}

#[derive(Clone)]
pub struct PeerResponder {
    sender: mpsc::Sender<ResponseRequest>,
    peers: Arc<Mutex<HashMap<String, AsyncPeerTx>>>,
}

impl PeerResponder {
    pub async fn send(&self, response: PeerResponse) -> Result<(), PeerSendError> {
        let (result_tx, result_rx) = oneshot::channel();
        self.sender
            .send(ResponseRequest {
                response,
                result_tx,
            })
            .await
            .map_err(|_| PeerSendError::ChannelClosed)?;

        result_rx.await.map_err(|_| PeerSendError::ChannelClosed)?
    }

    #[must_use]
    pub fn peer_count(&self) -> usize {
        self.peers
            .lock()
            .expect("peer registry lock poisoned")
            .len()
    }

    #[cfg(any(test, feature = "test-support"))]
    #[must_use]
    pub fn empty_for_tests() -> Self {
        let (sender, mut receiver) = mpsc::channel::<ResponseRequest>(8);
        tokio::spawn(async move {
            while let Some(request) = receiver.recv().await {
                let _ = request
                    .result_tx
                    .send(Err(PeerSendError::UnknownPeer(request.response.peer_id)));
            }
        });
        Self {
            sender,
            peers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    #[must_use]
    pub fn channel_closed_for_tests() -> Self {
        let (sender, receiver) = mpsc::channel::<ResponseRequest>(1);
        drop(receiver);
        Self {
            sender,
            peers: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

struct ResponseRequest {
    response: PeerResponse,
    result_tx: oneshot::Sender<Result<(), PeerSendError>>,
}

/// Handle for a running gateway.
pub struct GatewayHandle {
    cancel: CancellationToken,
    responder: PeerResponder,
    peers: Arc<Mutex<HashMap<String, AsyncPeerTx>>>,
    join: tokio::task::JoinHandle<Result<(), GatewayError>>,
}

impl GatewayHandle {
    /// Request a structured shutdown.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Wait for the gateway task to exit.
    pub async fn wait(self) -> Result<(), GatewayError> {
        match self.join.await {
            Ok(res) => res,
            Err(err) => Err(GatewayError::Join(err.to_string())),
        }
    }

    #[must_use]
    pub fn responder(&self) -> PeerResponder {
        self.responder.clone()
    }

    #[must_use]
    pub fn peer_count(&self) -> usize {
        self.peers
            .lock()
            .expect("peer registry lock poisoned")
            .len()
    }
}

/// Spawn the gateway accept loop and per-peer reader tasks.
///
/// Must be called from within a Tokio runtime.
pub fn spawn(
    cfg: GatewayConfig,
) -> Result<(GatewayHandle, mpsc::Receiver<GatewayEvent>), GatewayError> {
    #[cfg(not(unix))]
    {
        let _ = cfg;
        return Err(GatewayError::UnsupportedPlatform);
    }

    #[cfg(unix)]
    {
        let registry = Arc::new(load_registry(&cfg)?);
        let peers = Arc::new(Mutex::new(HashMap::<String, AsyncPeerTx>::new()));

        // Authorization boundary for channel negotiation. Do not include CONTROL (0); ipcprims
        // treats it as internal/unnegotiated.
        let allowed_channels: &[u16] = &[
            channels::COMMAND,
            channels::TELEMETRY,
            channels::ERROR,
            channels::MAIL,
            channels::PROXY,
            channels::ADMIN,
            channels::SKILL_IO,
        ];

        let cancel = CancellationToken::new();
        let (responses_tx, responses_rx) = mpsc::channel::<ResponseRequest>(1024);
        let listener = AsyncPeerListener::bind(&cfg.socket_path)?
            .with_channels(allowed_channels)
            .with_schema_registry(registry)
            .with_cancellation_token(cancel.clone());

        let (events_tx, events_rx) = mpsc::channel::<GatewayEvent>(1024);
        let responder = PeerResponder {
            sender: responses_tx,
            peers: Arc::clone(&peers),
        };

        let join = tokio::spawn(run_gateway(
            listener,
            cfg.core_peer_id,
            cancel.clone(),
            events_tx,
            responses_rx,
            Arc::clone(&peers),
        ));
        Ok((
            GatewayHandle {
                cancel,
                responder,
                peers,
                join,
            },
            events_rx,
        ))
    }
}

#[cfg(unix)]
fn load_registry(cfg: &GatewayConfig) -> Result<SchemaRegistry, GatewayError> {
    let config = RegistryConfig {
        strict_mode: true,
        fail_on_missing_schema: true,
        ..RegistryConfig::default()
    };
    Ok(SchemaRegistry::from_directory_with_config(
        &cfg.crucible_schemas_dir,
        config,
    )?)
}

#[cfg(unix)]
trait GatewayListener {
    fn accept_with_id<'a>(
        &'a self,
        peer_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<AsyncPeer, PeerError>> + Send + 'a>>;
}

#[cfg(unix)]
impl GatewayListener for AsyncPeerListener {
    fn accept_with_id<'a>(
        &'a self,
        peer_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<AsyncPeer, PeerError>> + Send + 'a>> {
        Box::pin(async move { AsyncPeerListener::accept_with_id(self, peer_id).await })
    }
}

#[cfg(unix)]
async fn accept_loop(
    listener: AsyncPeerListener,
    core_peer_id: String,
    cancel: CancellationToken,
    events_tx: mpsc::Sender<GatewayEvent>,
    peers: Arc<Mutex<HashMap<String, AsyncPeerTx>>>,
) -> Result<(), GatewayError> {
    accept_loop_with_listener(listener, core_peer_id, cancel, events_tx, peers).await
}

#[cfg(unix)]
async fn accept_loop_with_listener<L: GatewayListener>(
    listener: L,
    core_peer_id: String,
    cancel: CancellationToken,
    events_tx: mpsc::Sender<GatewayEvent>,
    peers: Arc<Mutex<HashMap<String, AsyncPeerTx>>>,
) -> Result<(), GatewayError> {
    let mut next_connection_id: u64 = 1;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::info!("gateway shutting down");
                return Ok(());
            }
            res = listener.accept_with_id(&core_peer_id) => {
                let peer = match res {
                    Ok(peer) => peer,
                    Err(err) if is_transient_accept_error(&err) => {
                        tracing::warn!(error = %err, "transient accept error; continuing");
                        continue;
                    }
                    Err(err) => return Err(err.into()),
                };
                // TODO(CRT-009): switch to opaque connection IDs if these escape gateway internals.
                let connection_id = format!("peer-{next_connection_id}");
                next_connection_id += 1;
                tracing::info!(
                    peer_id = %connection_id,
                    core_peer_id = %core_peer_id,
                    channels = ?peer.channels(),
                    "peer connected"
                );
                tokio::spawn(peer_reader(
                    connection_id,
                    peer,
                    cancel.clone(),
                    events_tx.clone(),
                    Arc::clone(&peers),
                ));
            }
        }
    }
}

#[cfg(unix)]
fn is_transient_accept_error(err: &PeerError) -> bool {
    match err {
        PeerError::Transport(TransportError::Accept(io_err)) => matches!(
            io_err.kind(),
            ErrorKind::ConnectionAborted
                | ErrorKind::ConnectionReset
                | ErrorKind::Interrupted
                | ErrorKind::TimedOut
                | ErrorKind::WouldBlock
        ),
        _ => false,
    }
}

#[cfg(unix)]
async fn run_gateway(
    listener: AsyncPeerListener,
    core_peer_id: String,
    cancel: CancellationToken,
    events_tx: mpsc::Sender<GatewayEvent>,
    responses_rx: mpsc::Receiver<ResponseRequest>,
    peers: Arc<Mutex<HashMap<String, AsyncPeerTx>>>,
) -> Result<(), GatewayError> {
    let response_task = tokio::spawn(response_writer(
        cancel.clone(),
        responses_rx,
        Arc::clone(&peers),
    ));

    let accept_result = accept_loop(listener, core_peer_id, cancel.clone(), events_tx, peers).await;

    cancel.cancel();

    match response_task.await {
        Ok(()) => accept_result,
        Err(err) => Err(GatewayError::Join(err.to_string())),
    }
}

#[cfg(unix)]
async fn peer_reader(
    connection_id: String,
    peer: AsyncPeer,
    cancel: CancellationToken,
    events_tx: mpsc::Sender<GatewayEvent>,
    peers: Arc<Mutex<HashMap<String, AsyncPeerTx>>>,
) {
    // Hold the send half for future bidirectional support (CRT-005/CRT-009).
    let (tx, mut rx) = peer.into_split();
    peers
        .lock()
        .expect("peer registry lock poisoned")
        .insert(connection_id.clone(), tx);

    loop {
        let frame_res = tokio::select! {
            _ = cancel.cancelled() => {
                rx.cancel();
                break;
            }
            res = rx.recv() => res,
        };

        let frame = match frame_res {
            Ok(frame) => frame,
            Err(err) => {
                tracing::info!(peer_id = %connection_id, error = %err, "peer disconnected");
                break;
            }
        };

        if let Err(err) = forward_frame(&connection_id, frame, &events_tx).await {
            tracing::warn!(peer_id = %connection_id, error = %err, "failed to forward gateway event");
            // If downstream is gone, there's no reason to keep accepting traffic.
            cancel.cancel();
            break;
        }
    }

    peers
        .lock()
        .expect("peer registry lock poisoned")
        .remove(&connection_id);
}

#[cfg(unix)]
async fn forward_frame(
    peer_id: &str,
    frame: Frame,
    events_tx: &mpsc::Sender<GatewayEvent>,
) -> Result<(), &'static str> {
    let channel: ChannelId = frame.channel;
    let event = GatewayEvent {
        peer_id: peer_id.to_owned(),
        channel,
        payload: frame.payload.as_ref().to_vec(),
    };
    events_tx
        .send(event)
        .await
        .map_err(|_| "downstream closed")?;
    Ok(())
}

#[cfg(unix)]
async fn response_writer(
    cancel: CancellationToken,
    mut responses_rx: mpsc::Receiver<ResponseRequest>,
    peers: Arc<Mutex<HashMap<String, AsyncPeerTx>>>,
) {
    loop {
        let request = tokio::select! {
            _ = cancel.cancelled() => return,
            request = responses_rx.recv() => match request {
                Some(request) => request,
                None => return,
            },
        };

        let result = send_to_peer(&peers, &request.response).await;
        if let Err(err) = &result {
            match err {
                PeerSendError::PeerDisconnected(peer_id) => {
                    tracing::debug!(peer_id = %peer_id, "peer disconnected during response send");
                }
                PeerSendError::UnknownPeer(peer_id) => {
                    tracing::warn!(peer_id = %peer_id, "attempted to send response to unknown peer");
                }
                PeerSendError::ChannelClosed => {
                    tracing::debug!("response channel closed while sending to peer");
                }
            }
        }
        let _ = request.result_tx.send(result);
    }
}

#[cfg(unix)]
async fn send_to_peer(
    peers: &Arc<Mutex<HashMap<String, AsyncPeerTx>>>,
    response: &PeerResponse,
) -> Result<(), PeerSendError> {
    let tx = {
        peers
            .lock()
            .expect("peer registry lock poisoned")
            .get(&response.peer_id)
            .cloned()
    }
    .ok_or_else(|| PeerSendError::UnknownPeer(response.peer_id.clone()))?;

    match tx.send(response.channel, &response.payload).await {
        Ok(()) => Ok(()),
        Err(_) => {
            peers
                .lock()
                .expect("peer registry lock poisoned")
                .remove(&response.peer_id);
            Err(PeerSendError::PeerDisconnected(response.peer_id.clone()))
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::{
        accept_loop_with_listener, is_transient_accept_error, GatewayEvent, GatewayListener,
    };
    use std::collections::HashMap;
    use std::future::Future;
    use std::io;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use ipcprims::peer::{async_connect, AsyncPeer, AsyncPeerListener, AsyncPeerTx, PeerError};
    use ipcprims::transport::TransportError;
    use lanyte_common::channels;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    use crate::test_support::TempGatewayDir;

    #[test]
    fn accept_connection_aborted_is_treated_as_transient() {
        let err = PeerError::Transport(TransportError::Accept(io::Error::from(
            io::ErrorKind::ConnectionAborted,
        )));

        assert!(is_transient_accept_error(&err));
    }

    #[test]
    fn handshake_failure_is_not_treated_as_transient() {
        let err = PeerError::HandshakeFailed("bad hello".to_owned());

        assert!(!is_transient_accept_error(&err));
    }

    struct TransientThenRealListener {
        inner: AsyncPeerListener,
        call_count: AtomicUsize,
    }

    impl GatewayListener for TransientThenRealListener {
        fn accept_with_id<'a>(
            &'a self,
            peer_id: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<AsyncPeer, PeerError>> + Send + 'a>> {
            Box::pin(async move {
                let call = self.call_count.fetch_add(1, Ordering::Relaxed);
                if call == 0 {
                    Err(PeerError::Transport(TransportError::Accept(
                        io::Error::from(io::ErrorKind::ConnectionAborted),
                    )))
                } else {
                    self.inner.accept_with_id(peer_id).await
                }
            })
        }
    }

    #[tokio::test]
    async fn transient_accept_error_does_not_stop_subsequent_peer_acceptance() {
        let dir = TempGatewayDir::new("transient-accept");
        let cancel = CancellationToken::new();
        let listener = TransientThenRealListener {
            inner: AsyncPeerListener::bind(dir.socket_path())
                .expect("listener should bind")
                .with_channels(&[channels::MAIL]),
            call_count: AtomicUsize::new(0),
        };
        let (events_tx, mut events_rx) = mpsc::channel::<GatewayEvent>(8);
        let peers = Arc::new(Mutex::new(HashMap::<String, AsyncPeerTx>::new()));

        let accept_task = tokio::spawn(accept_loop_with_listener(
            listener,
            "lanyte-core".to_owned(),
            cancel.clone(),
            events_tx,
            Arc::clone(&peers),
        ));

        let client = async_connect(dir.socket_path(), &[channels::MAIL])
            .await
            .expect("client should connect after transient accept failure");
        let (tx, _rx) = client.into_split();
        tx.send_json(channels::MAIL, &serde_json::json!({"after":"transient"}))
            .await
            .expect("frame should send");

        let event = tokio::time::timeout(Duration::from_secs(2), events_rx.recv())
            .await
            .expect("event should arrive")
            .expect("event should exist");

        assert_eq!(event.peer_id, "peer-1");
        assert_eq!(event.channel, channels::MAIL);
        assert_eq!(event.payload, br#"{"after":"transient"}"#.to_vec());

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), accept_task)
            .await
            .expect("accept loop should exit")
            .expect("join should succeed")
            .expect("accept loop should return ok");
    }
}
