//! LLM backend adapters (direct-to-provider) for the Lanyte core.
//!
//! Per ADR-0010, each backend implements `LlmBackend` directly against the
//! provider's native API using `reqwest`.

mod claude;
mod error;
mod grok;
mod openai;
mod types;

use std::pin::Pin;

pub use claude::ClaudeBackend;
pub use error::{LlmError, Result};
pub use grok::GrokBackend;
pub use openai::OpenAiBackend;
pub use types::{
    BackendCapabilities, CompletionRequest, CompletionResponse, HealthStatus, Message, MessageRole,
    StopReason, StreamEvent, ToolDefinition, ToolResult, Usage,
};

use futures_core::Stream;

pub type LlmStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + 'static>>;

/// Backend abstraction surface for the orchestrator.
///
/// - Adapters are transport-only (no policy decisions).
/// - Context window management lives in the orchestrator.
pub trait LlmBackend: Send + Sync {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    fn stream(&self, request: CompletionRequest) -> Result<LlmStream>;

    fn capabilities(&self) -> BackendCapabilities;

    fn health(&self) -> HealthStatus;

    fn name(&self) -> &'static str;
}
