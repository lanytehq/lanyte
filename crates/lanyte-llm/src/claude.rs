use std::collections::HashMap;
use std::fmt;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(test)]
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use reqwest::blocking::Client as BlockingClient;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, RETRY_AFTER};
use reqwest::Client as AsyncClient;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::error::{LlmError, Result};
use crate::types::{
    BackendCapabilities, CompletionRequest, CompletionResponse, HealthStatus, MessageRole,
    StopReason, ToolDefinition, Usage,
};
use crate::{LlmBackend, LlmStream};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";

static JITTER_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub(crate) struct ClaudeBackendOptions {
    pub base_url: String,
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub base_delay_500: Duration,
    pub max_jitter: Duration,
    pub jitter_seed: Option<u64>,
    pub connect_timeout: Duration,
    pub timeout: Duration,
    pub pool_max_idle_per_host: usize,
}

impl Default for ClaudeBackendOptions {
    fn default() -> Self {
        Self {
            base_url: ANTHROPIC_BASE_URL.to_owned(),
            max_attempts: 3,
            base_delay: Duration::from_secs(1),
            base_delay_500: Duration::from_secs(3),
            max_jitter: Duration::from_millis(250),
            jitter_seed: None,
            connect_timeout: Duration::from_secs(10),
            timeout: Duration::from_secs(60),
            pool_max_idle_per_host: 8,
        }
    }
}

#[derive(Clone)]
pub struct ClaudeBackend {
    blocking_client: BlockingClient,
    streaming_client: AsyncClient,
    model: String,
    api_key: String,
    messages_url: String,
    stream_url: String,
    max_attempts: u32,
    base_delay: Duration,
    base_delay_500: Duration,
    max_jitter: Duration,
    jitter_seed: u64,
    #[cfg(test)]
    active_streams: Arc<AtomicUsize>,
}

impl ClaudeBackend {
    pub fn from_config(config: &lanyte_common::ClaudeConfig) -> Result<Self> {
        let api_key = config.api_key.clone().ok_or(LlmError::MissingApiKey)?;
        Self::new(config.model.clone(), api_key)
    }

    pub fn new(model: String, api_key: String) -> Result<Self> {
        Self::new_with_options(model, api_key, ClaudeBackendOptions::default())
    }

    pub(crate) fn new_with_options(
        model: String,
        api_key: String,
        options: ClaudeBackendOptions,
    ) -> Result<Self> {
        if api_key.trim().is_empty() {
            return Err(LlmError::MissingApiKey);
        }

        let base_url = options.base_url.trim_end_matches('/').to_owned();
        if base_url.is_empty() {
            return Err(LlmError::InvalidResponse(
                "Claude base_url must not be empty",
            ));
        }
        let messages_url = format!("{base_url}/v1/messages");
        let stream_url = messages_url.clone();

        let blocking_client = BlockingClient::builder()
            .connect_timeout(options.connect_timeout)
            .timeout(options.timeout)
            .pool_max_idle_per_host(options.pool_max_idle_per_host)
            .build()?;
        let streaming_client = AsyncClient::builder()
            .connect_timeout(options.connect_timeout)
            .timeout(options.timeout)
            .pool_max_idle_per_host(options.pool_max_idle_per_host)
            .build()?;

        let jitter_seed = options.jitter_seed.unwrap_or_else(|| {
            // Jitter must vary across processes/instances to avoid herd behavior during outages.
            // This is not security-critical randomness.
            let pid = std::process::id() as u64;
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| Duration::from_secs(0))
                .as_nanos() as u64;
            let ctr = JITTER_COUNTER.fetch_add(1, Ordering::Relaxed);
            now ^ (pid.rotate_left(17)) ^ (ctr.rotate_left(33))
        });

        Ok(Self {
            blocking_client,
            streaming_client,
            model,
            api_key,
            messages_url,
            stream_url,
            max_attempts: options.max_attempts,
            base_delay: options.base_delay,
            base_delay_500: options.base_delay_500,
            max_jitter: options.max_jitter,
            jitter_seed,
            #[cfg(test)]
            active_streams: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(ANTHROPIC_VERSION)
                .map_err(|_| LlmError::InvalidResponse("invalid anthropic-version header"))?,
        );
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key)
                .map_err(|_| LlmError::InvalidResponse("API key contains invalid header bytes"))?,
        );
        Ok(headers)
    }

    fn do_complete_once(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let headers = self.headers()?;
        let anthropic_req = self.build_request(request)?;

        let resp = self
            .blocking_client
            .post(&self.messages_url)
            .headers(headers)
            .json(&anthropic_req)
            .send()?;

        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        let body = resp.text()?;

        if !status.is_success() {
            return Err(classify_error(status.as_u16(), retry_after, &body));
        }

        let parsed: ClaudeMessagesResponse = serde_json::from_str(&body)?;
        let text = parsed.assistant_text().ok_or(LlmError::InvalidResponse(
            "missing assistant text content blocks",
        ))?;

        Ok(CompletionResponse {
            response_id: None,
            text,
            stop_reason: parsed
                .stop_reason
                .as_deref()
                .map(map_stop_reason)
                .unwrap_or(StopReason::EndTurn),
            usage: parsed.usage.map(|usage| Usage {
                input_tokens: u64::from(usage.input_tokens),
                output_tokens: u64::from(usage.output_tokens),
                reasoning_tokens: None,
            }),
        })
    }

    fn should_retry_status(status: u16) -> bool {
        status == 429 || status == 500 || status == 502 || status == 503 || status == 504
    }

    fn base_delay_for_status(&self, status: u16) -> Duration {
        // 500s can represent partial upstream outage and are often less likely to recover quickly.
        // Use a longer baseline for 500 specifically to reduce load-amplification and increase
        // decorrelation across concurrent agents.
        if status == 500 {
            // Ensure 500 baseline never goes below the normal baseline.
            self.base_delay_500.max(self.base_delay)
        } else {
            self.base_delay
        }
    }

    fn backoff_for(&self, status: u16, attempt: u32) -> Duration {
        // attempt is 1-based: attempt=1 sleeps base, attempt=2 sleeps 2x base, etc.
        let mut d = self.base_delay_for_status(status);
        for _ in 1..attempt {
            d = d.saturating_mul(2);
        }
        d
    }

    fn jitter_for_attempt(&self, attempt: u32) -> Duration {
        if self.max_jitter.is_zero() {
            return Duration::ZERO;
        }

        // Small, dependency-free jitter. This is not security-critical.
        // Seed includes per-process and per-instance entropy (time/pid/counter) to avoid herd retries.
        let mut x = self.jitter_seed ^ (attempt as u64);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let max = self.max_jitter.as_nanos().min(u64::MAX as u128) as u64;
        if max == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(x % (max + 1))
        }
    }

    fn build_request(&self, request: &CompletionRequest) -> Result<ClaudeMessagesRequest> {
        let mut messages = request
            .messages
            .iter()
            .map(ClaudeMessage::from_message)
            .collect::<Result<Vec<_>>>()?;
        if !request.tool_results.is_empty() {
            // Anthropic requires the immediately preceding assistant message to contain the
            // corresponding tool_use blocks before user tool_result blocks are accepted.
            // Reconstruct that assistant turn from ToolResult metadata so follow-up requests
            // remain valid under the shared LlmBackend contract.
            messages.push(ClaudeMessage::tool_uses(&request.tool_results)?);
            messages.push(ClaudeMessage::tool_results(&request.tool_results));
        }

        // Claude accepts temperature directly, but parallel tool-call control is still
        // intentionally disabled until the adapter wires Anthropic's provider-specific knob.
        Ok(ClaudeMessagesRequest {
            model: self.model.clone(),
            max_tokens: request.max_tokens.unwrap_or(1024),
            system: request.system_prompt.clone(),
            thinking: request
                .thinking_budget_tokens
                .map(|budget_tokens| ClaudeThinkingConfig {
                    kind: "enabled",
                    budget_tokens,
                }),
            temperature: request.temperature,
            stream: false,
            tools: request
                .tools
                .iter()
                .map(ClaudeTool::try_from)
                .collect::<Result<Vec<_>>>()?,
            messages,
        })
    }

    #[cfg(test)]
    fn active_stream_count(&self) -> usize {
        self.active_streams.load(Ordering::SeqCst)
    }
}

impl fmt::Debug for ClaudeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClaudeBackend")
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl LlmBackend for ClaudeBackend {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mut attempt: u32 = 0;

        loop {
            attempt += 1;
            match self.do_complete_once(&request) {
                Ok(resp) => return Ok(resp),
                Err(LlmError::RateLimited { retry_after }) => {
                    if attempt >= self.max_attempts {
                        return Err(LlmError::RateLimited { retry_after });
                    }
                    let delay = retry_after.unwrap_or_else(|| self.backoff_for(429, attempt));
                    let sleep_for = delay.saturating_add(self.jitter_for_attempt(attempt));
                    if !sleep_for.is_zero() {
                        thread::sleep(sleep_for);
                    }
                }
                Err(LlmError::ServiceUnavailable) => {
                    if attempt >= self.max_attempts {
                        return Err(LlmError::ServiceUnavailable);
                    }
                    let delay = self.backoff_for(503, attempt);
                    let sleep_for = delay.saturating_add(self.jitter_for_attempt(attempt));
                    if !sleep_for.is_zero() {
                        thread::sleep(sleep_for);
                    }
                }
                Err(LlmError::Upstream { status, message }) => {
                    if attempt >= self.max_attempts || !Self::should_retry_status(status) {
                        return Err(LlmError::Upstream { status, message });
                    }
                    let delay = self.backoff_for(status, attempt);
                    let sleep_for = delay.saturating_add(self.jitter_for_attempt(attempt));
                    if !sleep_for.is_zero() {
                        thread::sleep(sleep_for);
                    }
                }
                Err(other) => return Err(other),
            }
        }
    }

    fn stream(&self, request: CompletionRequest) -> Result<LlmStream> {
        let handle = Handle::try_current()
            .map_err(|_| LlmError::Unsupported("Claude streaming requires tokio runtime"))?;
        let headers = self.headers()?;
        let mut anthropic_req = self.build_request(&request)?;
        anthropic_req.stream = true;
        let client = self.streaming_client.clone();
        let url = self.stream_url.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        #[cfg(test)]
        let active_streams = Arc::clone(&self.active_streams);

        #[cfg(test)]
        active_streams.fetch_add(1, Ordering::SeqCst);

        let task = handle.spawn(async move {
            #[cfg(test)]
            let _guard = StreamTaskGuard(active_streams);
            let _ = stream_sse_response(client, url, headers, anthropic_req, tx).await;
        });

        Ok(Box::pin(ClaudeLlmStream {
            inner: UnboundedReceiverStream::new(rx),
            task: Some(task),
        }))
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_system_prompt: true,
            // Keep this false until the adapter can intentionally control Claude's
            // provider-specific parallel tool-call behavior.
            supports_parallel_tool_calls: false,
            supports_web_search: false,
            supports_image_generation: false,
            max_context_tokens: 200_000,
        }
    }

    fn health(&self) -> HealthStatus {
        if self.api_key.trim().is_empty() {
            return HealthStatus::Unconfigured;
        }
        HealthStatus::Healthy
    }

    fn name(&self) -> &'static str {
        "claude"
    }
}

#[derive(Debug, serde::Serialize)]
struct ClaudeMessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ClaudeThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ClaudeTool>,
    messages: Vec<ClaudeMessage>,
}

#[derive(Debug, serde::Serialize)]
struct ClaudeThinkingConfig {
    #[serde(rename = "type")]
    kind: &'static str,
    budget_tokens: u32,
}

#[derive(Debug, serde::Serialize)]
struct ClaudeMessage {
    role: &'static str,
    content: ClaudeMessageContent,
}

impl ClaudeMessage {
    fn from_message(message: &crate::Message) -> Result<Self> {
        let role = match message.role {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => {
                return Err(LlmError::UnsupportedMessageRole(
                    "system messages must use CompletionRequest.system_prompt",
                ));
            }
            MessageRole::Tool => {
                return Err(LlmError::UnsupportedMessageRole(
                    "tool results must use CompletionRequest.tool_results",
                ));
            }
        };

        Ok(Self {
            role,
            content: ClaudeMessageContent::Text(message.content.clone()),
        })
    }

    fn tool_results(tool_results: &[crate::ToolResult]) -> Self {
        Self {
            role: "user",
            content: ClaudeMessageContent::Blocks(
                tool_results
                    .iter()
                    .map(|result| ClaudeInputBlock::ToolResult {
                        tool_use_id: result.tool_call_id.clone(),
                        content: result.content.clone(),
                    })
                    .collect(),
            ),
        }
    }

    fn tool_uses(tool_results: &[crate::ToolResult]) -> Result<Self> {
        Ok(Self {
            role: "assistant",
            content: ClaudeMessageContent::Blocks(
                tool_results
                    .iter()
                    .map(|result| {
                        Ok(ClaudeInputBlock::ToolUse {
                            id: result.tool_call_id.clone(),
                            name: result.tool_name.clone().ok_or(LlmError::Unsupported(
                                "Claude tool results require tool_name for follow-up requests",
                            ))?,
                            input: serde_json::from_str(result.arguments.as_deref().ok_or(
                                LlmError::Unsupported(
                                    "Claude tool results require arguments for follow-up requests",
                                ),
                            )?)
                            .map_err(|_| {
                                LlmError::InvalidResponse(
                                    "Claude tool result arguments must be valid JSON",
                                )
                            })?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
        })
    }
}

#[derive(Debug, serde::Serialize)]
#[serde(untagged)]
enum ClaudeMessageContent {
    Text(String),
    Blocks(Vec<ClaudeInputBlock>),
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClaudeInputBlock {
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, serde::Serialize)]
struct ClaudeTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

impl TryFrom<&ToolDefinition> for ClaudeTool {
    type Error = LlmError;

    fn try_from(tool: &ToolDefinition) -> Result<Self> {
        match tool {
            ToolDefinition::Function {
                name,
                description,
                parameters,
            } => Ok(Self {
                name: name.clone(),
                description: description.clone(),
                input_schema: parameters.clone(),
            }),
            ToolDefinition::WebSearch { .. } | ToolDefinition::ImageGeneration => Err(
                LlmError::Unsupported("Claude adapter only supports function tools"),
            ),
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct ClaudeMessagesResponse {
    content: Vec<ClaudeContentBlock>,
    #[serde(default)]
    usage: Option<ClaudeUsage>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ClaudeContentBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

impl ClaudeMessagesResponse {
    fn assistant_text(&self) -> Option<String> {
        let mut out = String::new();
        for block in &self.content {
            if block.kind == "text" {
                if let Some(text) = block.text.as_deref() {
                    out.push_str(text);
                }
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeErrorEnvelope {
    error: ClaudeError,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeError {
    message: String,
    #[serde(default)]
    #[serde(rename = "type")]
    kind: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeUsage {
    input_tokens: u32,
    output_tokens: u32,
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

fn classify_error(status: u16, retry_after: Option<Duration>, body: &str) -> LlmError {
    let parsed = serde_json::from_str::<ClaudeErrorEnvelope>(body).ok();
    let message = parsed
        .as_ref()
        .map(|envelope| envelope.error.message.clone())
        .unwrap_or_else(|| body.to_owned());
    let kind = parsed
        .and_then(|envelope| envelope.error.kind)
        .unwrap_or_default();

    match status {
        401 | 403 => LlmError::AuthenticationFailed,
        429 => LlmError::RateLimited { retry_after },
        503 | 529 => LlmError::ServiceUnavailable,
        400 | 404 if is_invalid_model_error(&kind, &message) => LlmError::InvalidModel,
        _ => LlmError::Upstream { status, message },
    }
}

fn is_invalid_model_error(kind: &str, message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    kind == "not_found_error" || (message.contains("model") && message.contains("not found"))
}

fn map_stop_reason(stop_reason: &str) -> StopReason {
    match stop_reason {
        "max_tokens" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

struct ClaudeLlmStream {
    inner: UnboundedReceiverStream<Result<crate::StreamEvent>>,
    task: Option<JoinHandle<()>>,
}

impl futures_core::Stream for ClaudeLlmStream {
    type Item = Result<crate::StreamEvent>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for ClaudeLlmStream {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[cfg(test)]
struct StreamTaskGuard(Arc<AtomicUsize>);

#[cfg(test)]
impl Drop for StreamTaskGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn stream_sse_response(
    client: AsyncClient,
    url: String,
    headers: HeaderMap,
    request: ClaudeMessagesRequest,
    tx: mpsc::UnboundedSender<Result<crate::StreamEvent>>,
) -> Result<()> {
    let response = client
        .post(url)
        .headers(headers)
        .json(&request)
        .send()
        .await?;
    let status = response.status();
    let retry_after = parse_retry_after(response.headers());
    if !status.is_success() {
        let body = response.text().await?;
        let _ = tx.send(Err(classify_error(status.as_u16(), retry_after, &body)));
        return Ok(());
    }

    let mut parser = ClaudeSseParser::default();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        for event in parser.push(&chunk)? {
            if tx.send(Ok(event)).is_err() {
                return Ok(());
            }
        }
    }

    Ok(())
}

#[derive(Default)]
struct ClaudeSseParser {
    buffer: String,
    current_event: Option<String>,
    current_data: String,
    input_tokens: u64,
    tool_blocks: HashMap<u64, ToolBlockState>,
}

#[derive(Clone)]
struct ToolBlockState {
    id: String,
}

impl ClaudeSseParser {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<crate::StreamEvent>> {
        self.buffer.push_str(
            std::str::from_utf8(bytes)
                .map_err(|_| LlmError::InvalidResponse("Claude SSE contained invalid UTF-8"))?,
        );

        let mut out = Vec::new();
        while let Some(pos) = self.buffer.find('\n') {
            let mut line = self.buffer.drain(..=pos).collect::<String>();
            if line.ends_with('\n') {
                line.pop();
            }
            if line.ends_with('\r') {
                line.pop();
            }

            if line.is_empty() {
                out.extend(self.finish_event()?);
                continue;
            }

            if let Some(event) = line.strip_prefix("event:") {
                self.current_event = Some(event.trim().to_owned());
            } else if let Some(data) = line.strip_prefix("data:") {
                if !self.current_data.is_empty() {
                    self.current_data.push('\n');
                }
                self.current_data.push_str(data.trim_start());
            }
        }

        Ok(out)
    }

    fn finish_event(&mut self) -> Result<Vec<crate::StreamEvent>> {
        if self.current_data.is_empty() {
            self.current_event = None;
            return Ok(Vec::new());
        }

        let event_name = self.current_event.take().unwrap_or_default();
        let data = std::mem::take(&mut self.current_data);
        if data == "[DONE]" || event_name == "ping" {
            return Ok(Vec::new());
        }

        let payload: ClaudeStreamEvent = serde_json::from_str(&data)?;
        let mut out = Vec::new();
        match event_name.as_str() {
            "message_start" => {
                if let Some(message) = payload.message {
                    self.input_tokens = u64::from(message.usage.input_tokens);
                }
            }
            "content_block_start" => {
                if let (Some(index), Some(block)) = (payload.index, payload.content_block) {
                    if block.kind == "tool_use" {
                        let id = block
                            .id
                            .ok_or(LlmError::InvalidResponse("tool_use missing id"))?;
                        let name = block
                            .name
                            .ok_or(LlmError::InvalidResponse("tool_use missing name"))?;
                        self.tool_blocks
                            .insert(index, ToolBlockState { id: id.clone() });
                        out.push(crate::StreamEvent::ToolCallStart { id, name });
                    }
                }
            }
            "content_block_delta" => {
                if let (Some(index), Some(delta)) = (payload.index, payload.delta) {
                    match delta.kind.as_deref().unwrap_or_default() {
                        "text_delta" => {
                            if let Some(text) = delta.text {
                                out.push(crate::StreamEvent::TextDelta(text));
                            }
                        }
                        "input_json_delta" => {
                            let tool =
                                self.tool_blocks
                                    .get(&index)
                                    .ok_or(LlmError::InvalidResponse(
                                        "tool delta arrived before tool start",
                                    ))?;
                            out.push(crate::StreamEvent::ToolCallDelta {
                                id: tool.id.clone(),
                                arguments_delta: delta.partial_json.unwrap_or_default(),
                            });
                        }
                        "thinking_delta" => {
                            if let Some(thinking) = delta.thinking {
                                out.push(crate::StreamEvent::ThinkingDelta(thinking));
                            }
                        }
                        _ => {}
                    }
                }
            }
            "content_block_stop" => {
                if let Some(index) = payload.index {
                    if let Some(tool) = self.tool_blocks.remove(&index) {
                        out.push(crate::StreamEvent::ToolCallEnd { id: tool.id });
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = payload.delta {
                    if let Some(stop_reason) = delta.stop_reason {
                        out.push(crate::StreamEvent::StopReason(map_stop_reason(
                            &stop_reason,
                        )));
                    }
                }
                if let Some(usage) = payload.usage {
                    out.push(crate::StreamEvent::Usage(Usage {
                        input_tokens: self.input_tokens,
                        output_tokens: u64::from(usage.output_tokens),
                        reasoning_tokens: None,
                    }));
                }
            }
            "message_stop" => out.push(crate::StreamEvent::Done),
            "error" => {
                if let Some(error) = payload.error {
                    return Err(classify_error(
                        500,
                        None,
                        &serde_json::to_string(&ClaudeErrorEnvelope {
                            error: ClaudeError {
                                message: error.message,
                                kind: Some(error.kind),
                            },
                        })?,
                    ));
                }
            }
            // Anthropic does not currently emit an annotation-style stream event.
            _ => {}
        }

        Ok(out)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeStreamEvent {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    message: Option<ClaudeStreamMessage>,
    #[serde(default)]
    content_block: Option<ClaudeStreamContentBlock>,
    #[serde(default)]
    delta: Option<ClaudeStreamDelta>,
    #[serde(default)]
    usage: Option<ClaudeOutputUsage>,
    #[serde(default)]
    error: Option<ClaudeStreamError>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeStreamMessage {
    usage: ClaudeUsage,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeStreamContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeStreamDelta {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeOutputUsage {
    output_tokens: u32,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ClaudeStreamError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener};
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;

    #[test]
    fn parses_text_content_blocks() {
        let body = r#"{ "content": [ { "type": "text", "text": "hi" }, { "type": "text", "text": " there" } ] }"#;
        let parsed: ClaudeMessagesResponse = serde_json::from_str(body).expect("parse response");
        assert_eq!(parsed.assistant_text().as_deref(), Some("hi there"));
    }

    #[test]
    fn missing_text_blocks_returns_none() {
        let body = r#"{ "content": [ { "type": "tool_use" } ] }"#;
        let parsed: ClaudeMessagesResponse = serde_json::from_str(body).expect("parse response");
        assert!(parsed.assistant_text().is_none());
    }

    #[test]
    fn debug_redacts_api_key() {
        let backend = ClaudeBackend::new("claude-sonnet-4-6".to_owned(), "secret".to_owned())
            .expect("backend init");
        let text = format!("{backend:?}");
        assert!(text.contains("<redacted>"));
        assert!(!text.contains("secret"));
    }

    #[test]
    fn jitter_seed_is_respected_for_deterministic_tests() {
        let options = ClaudeBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            base_delay: Duration::ZERO,
            base_delay_500: Duration::ZERO,
            max_jitter: Duration::from_secs(1),
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend =
            ClaudeBackend::new_with_options("m".to_owned(), "k".to_owned(), options).unwrap();

        // Same attempt => stable output for a given seed.
        let a = backend.jitter_for_attempt(1);
        let b = backend.jitter_for_attempt(1);
        assert_eq!(a, b);

        // Different attempts => (almost certainly) different output.
        assert_ne!(backend.jitter_for_attempt(1), backend.jitter_for_attempt(2));
    }

    #[test]
    fn backoff_uses_longer_baseline_for_500() {
        let options = ClaudeBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            base_delay: Duration::from_secs(1),
            base_delay_500: Duration::from_secs(3),
            max_jitter: Duration::ZERO,
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend =
            ClaudeBackend::new_with_options("m".to_owned(), "k".to_owned(), options).unwrap();

        assert_eq!(backend.backoff_for(503, 1), Duration::from_secs(1));
        assert_eq!(backend.backoff_for(500, 1), Duration::from_secs(3));
        assert_eq!(backend.backoff_for(500, 2), Duration::from_secs(6));
    }

    #[test]
    fn baseline_for_500_is_never_below_normal_baseline() {
        let options = ClaudeBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            base_delay: Duration::from_secs(5),
            base_delay_500: Duration::from_secs(3),
            max_jitter: Duration::ZERO,
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend =
            ClaudeBackend::new_with_options("m".to_owned(), "k".to_owned(), options).unwrap();

        assert_eq!(backend.base_delay_for_status(500), Duration::from_secs(5));
    }

    #[derive(Debug, Clone)]
    struct CapturedRequest {
        method: String,
        path: String,
        headers: BTreeMap<String, String>,
        body: String,
    }

    #[derive(Debug, Clone)]
    struct MockResponse {
        status: u16,
        content_type: &'static str,
        extra_headers: Vec<(String, String)>,
        content_length_override: Option<usize>,
        linger_after_body: Option<Duration>,
        body: String,
    }

    impl MockResponse {
        fn json(status: u16, body: impl Into<String>) -> Self {
            Self {
                status,
                content_type: "application/json",
                extra_headers: Vec::new(),
                content_length_override: None,
                linger_after_body: None,
                body: body.into(),
            }
        }
    }

    struct MockServer {
        addr: SocketAddr,
        requests: Arc<Mutex<Vec<CapturedRequest>>>,
        handle: Option<JoinHandle<()>>,
    }

    impl MockServer {
        fn start(responses: Vec<MockResponse>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("local addr");

            let requests: Arc<Mutex<Vec<CapturedRequest>>> = Arc::new(Mutex::new(Vec::new()));
            let responses: Arc<Mutex<Vec<MockResponse>>> = Arc::new(Mutex::new(responses));

            let requests_thread = Arc::clone(&requests);
            let responses_thread = Arc::clone(&responses);

            let handle = thread::spawn(move || {
                for stream in listener.incoming() {
                    let mut stream = match stream {
                        Ok(s) => s,
                        Err(_) => break,
                    };

                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let captured = match read_http_request(&mut stream) {
                        Ok(c) => c,
                        Err(_) => break,
                    };

                    requests_thread.lock().expect("lock").push(captured);

                    let response = {
                        let mut responses = responses_thread.lock().expect("lock");
                        if responses.is_empty() {
                            break;
                        }
                        responses.remove(0)
                    };
                    let _ = write_http_response(&mut stream, response);
                }
            });

            Self {
                addr,
                requests,
                handle: Some(handle),
            }
        }

        fn base_url(&self) -> String {
            format!("http://{}", self.addr)
        }

        fn take_requests(&self) -> Vec<CapturedRequest> {
            std::mem::take(&mut *self.requests.lock().expect("lock"))
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            // Best-effort: connect once to unblock accept loop.
            let _ = std::net::TcpStream::connect(self.addr);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> std::io::Result<CapturedRequest> {
        let mut buf = Vec::<u8>::new();
        let mut tmp = [0u8; 4096];
        let header_end;
        loop {
            let n = stream.read(&mut tmp)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "eof",
                ));
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                header_end = pos + 4;
                break;
            }
        }

        let headers_bytes = &buf[..header_end];
        let headers_str = std::str::from_utf8(headers_bytes)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "non-utf8"))?;

        let mut lines = headers_str.split("\r\n").filter(|l| !l.is_empty());
        let request_line = lines
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no request"))?;
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("").to_owned();
        let path = parts.next().unwrap_or("").to_owned();

        let mut headers = BTreeMap::new();
        let mut content_length: usize = 0;
        for line in lines {
            if let Some((k, v)) = line.split_once(':') {
                let key = k.trim().to_ascii_lowercase();
                let value = v.trim().to_owned();
                if key == "content-length" {
                    content_length = value.parse::<usize>().unwrap_or(0);
                }
                headers.insert(key, value);
            }
        }

        while buf.len() < header_end + content_length {
            let n = stream.read(&mut tmp)?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
        }

        let body_bytes = &buf[header_end..header_end + content_length];
        let body = String::from_utf8_lossy(body_bytes).to_string();

        Ok(CapturedRequest {
            method,
            path,
            headers,
            body,
        })
    }

    fn reason_phrase(status: u16) -> &'static str {
        match status {
            200 => "OK",
            400 => "Bad Request",
            401 => "Unauthorized",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "OK",
        }
    }

    fn write_http_response(
        stream: &mut std::net::TcpStream,
        response: MockResponse,
    ) -> std::io::Result<()> {
        let body_bytes = response.body.as_bytes();
        let mut extra_headers = String::new();
        for (key, value) in &response.extra_headers {
            use std::fmt::Write as _;

            let _ = write!(extra_headers, "{key}: {value}\r\n");
        }
        let content_length = response.content_length_override.unwrap_or(body_bytes.len());
        let head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n",
            response.status,
            reason_phrase(response.status),
            response.content_type,
            content_length,
            extra_headers
        );
        stream.write_all(head.as_bytes())?;
        stream.write_all(body_bytes)?;
        stream.flush()?;
        if let Some(duration) = response.linger_after_body {
            std::thread::sleep(duration);
        }
        Ok(())
    }

    fn backend_for_server(base_url: String) -> ClaudeBackend {
        let options = ClaudeBackendOptions {
            base_url,
            base_delay: Duration::ZERO,
            base_delay_500: Duration::ZERO,
            max_jitter: Duration::ZERO,
            jitter_seed: Some(1),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        ClaudeBackend::new_with_options(
            "claude-sonnet-4-6".to_owned(),
            "test-key".to_owned(),
            options,
        )
        .expect("backend init")
    }

    fn backend_for_server_with_attempts(base_url: String, max_attempts: u32) -> ClaudeBackend {
        let options = ClaudeBackendOptions {
            base_url,
            max_attempts,
            base_delay: Duration::ZERO,
            base_delay_500: Duration::ZERO,
            max_jitter: Duration::ZERO,
            jitter_seed: Some(1),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        ClaudeBackend::new_with_options(
            "claude-sonnet-4-6".to_owned(),
            "test-key".to_owned(),
            options,
        )
        .expect("backend init")
    }

    #[test]
    fn retries_on_429_then_succeeds() {
        let server = MockServer::start(vec![
            MockResponse::json(429, r#"{ "error": { "message": "rate limit" } }"#),
            MockResponse::json(
                200,
                r#"{ "content": [ { "type": "text", "text": "ok" } ] }"#,
            ),
        ]);

        let backend = backend_for_server(server.base_url());
        let resp = backend
            .complete(CompletionRequest::single_user_message("hello", 16))
            .expect("complete");
        assert_eq!(resp.text, "ok");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);

        let requests = server.take_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].method, "POST");
        assert_eq!(requests[0].path, "/v1/messages");
        assert_eq!(
            requests[0]
                .headers
                .get("anthropic-version")
                .map(String::as_str),
            Some(ANTHROPIC_VERSION)
        );
        assert_eq!(
            requests[0].headers.get("x-api-key").map(String::as_str),
            Some("test-key")
        );

        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["max_tokens"], 16);
        assert!(body.get("system").is_none());
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn retries_on_500_then_succeeds() {
        let server = MockServer::start(vec![
            MockResponse::json(500, r#"{ "error": { "message": "boom" } }"#),
            MockResponse::json(
                200,
                r#"{ "content": [ { "type": "text", "text": "ok" } ] }"#,
            ),
        ]);

        let backend = backend_for_server(server.base_url());
        let resp = backend
            .complete(CompletionRequest::single_user_message("hello", 16))
            .expect("complete");
        assert_eq!(resp.text, "ok");

        let requests = server.take_requests();
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn retries_on_503_up_to_max_attempts() {
        let server = MockServer::start(vec![
            MockResponse::json(503, r#"{ "error": { "message": "nope" } }"#),
            MockResponse::json(503, r#"{ "error": { "message": "nope" } }"#),
            MockResponse::json(503, r#"{ "error": { "message": "nope" } }"#),
        ]);

        let backend = backend_for_server(server.base_url());
        let err = backend
            .complete(CompletionRequest::single_user_message("hello", 16))
            .expect_err("must fail");
        assert!(matches!(err, LlmError::ServiceUnavailable));

        let requests = server.take_requests();
        assert_eq!(requests.len(), 3);
    }

    #[test]
    fn maps_authentication_failure_to_typed_error() {
        let server = MockServer::start(vec![MockResponse::json(
            401,
            r#"{ "error": { "type": "authentication_error", "message": "bad key" } }"#,
        )]);

        let backend = backend_for_server_with_attempts(server.base_url(), 1);
        let err = backend
            .complete(CompletionRequest::single_user_message("hello", 16))
            .expect_err("must fail");
        assert!(matches!(err, LlmError::AuthenticationFailed));
    }

    #[test]
    fn maps_rate_limit_to_typed_error_with_retry_after() {
        let server = MockServer::start(vec![MockResponse {
            status: 429,
            content_type: "application/json",
            extra_headers: vec![("Retry-After".to_owned(), "7".to_owned())],
            content_length_override: None,
            linger_after_body: None,
            body: r#"{ "error": { "type": "rate_limit_error", "message": "slow down" } }"#
                .to_owned(),
        }]);

        let backend = backend_for_server_with_attempts(server.base_url(), 1);
        let err = backend
            .complete(CompletionRequest::single_user_message("hello", 16))
            .expect_err("must fail");
        assert!(matches!(
            err,
            LlmError::RateLimited {
                retry_after: Some(duration)
            } if duration == Duration::from_secs(7)
        ));
    }

    #[test]
    fn maps_invalid_model_to_typed_error() {
        let server = MockServer::start(vec![MockResponse::json(
            404,
            r#"{ "error": { "type": "not_found_error", "message": "model not found" } }"#,
        )]);

        let backend = backend_for_server(server.base_url());
        let err = backend
            .complete(CompletionRequest::single_user_message("hello", 16))
            .expect_err("must fail");
        assert!(matches!(err, LlmError::InvalidModel));
    }

    #[test]
    fn serializes_tools_into_anthropic_request_shape() {
        let server = MockServer::start(vec![MockResponse::json(
            200,
            r#"{ "content": [ { "type": "text", "text": "ok" } ], "stop_reason": "tool_use" }"#,
        )]);

        let backend = backend_for_server(server.base_url());
        let request = CompletionRequest {
            system_prompt: Some("You are helpful".to_owned()),
            previous_response_id: None,
            messages: vec![crate::Message::user("What is the weather in London?")],
            tools: vec![ToolDefinition::function(
                "get_weather",
                "Return the weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                    "additionalProperties": false
                }),
            )],
            tool_results: Vec::new(),
            max_tokens: Some(32),
            thinking_budget_tokens: None,
            temperature: Some(0.25),
            parallel_tool_calls: Some(true),
        };

        let response = backend.complete(request).expect("complete");
        assert_eq!(response.stop_reason, StopReason::ToolUse);

        let requests = server.take_requests();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["system"], "You are helpful");
        assert_eq!(body["temperature"], serde_json::json!(0.25));
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(body["tools"][0]["description"], "Return the weather");
        assert_eq!(
            body["tools"][0]["input_schema"]["required"][0],
            serde_json::Value::String("city".to_owned())
        );
        assert!(body.get("parallel_tool_calls").is_none());
        let capabilities = backend.capabilities();
        assert!(capabilities.supports_streaming);
        assert!(capabilities.supports_tool_use);
        assert!(!capabilities.supports_parallel_tool_calls);
    }

    #[test]
    fn serializes_extended_thinking_request_shape() {
        let server = MockServer::start(vec![MockResponse::json(
            200,
            r#"{ "content": [ { "type": "text", "text": "ok" } ], "stop_reason": "end_turn" }"#,
        )]);

        let backend = backend_for_server(server.base_url());
        let mut request = CompletionRequest::single_user_message("Solve 27 * 453", 64);
        request.thinking_budget_tokens = Some(32);

        let response = backend.complete(request).expect("complete");
        assert_eq!(response.stop_reason, StopReason::EndTurn);

        let requests = server.take_requests();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 32);
    }

    #[test]
    fn rejects_provider_native_tools_in_claude_requests() {
        let backend = backend_for_server("https://example.invalid".to_owned());
        let mut request = CompletionRequest::single_user_message("Search for weather", 32);
        request.tools = vec![ToolDefinition::web_search(
            vec!["example.com".to_owned()],
            Vec::new(),
        )];

        let err = backend
            .complete(request)
            .expect_err("must reject web search tool");
        assert!(
            matches!(err, LlmError::Unsupported(message) if message.contains("function tools"))
        );
    }

    #[test]
    fn sse_parser_parses_anthropic_tool_and_usage_events() {
        let mut parser = ClaudeSseParser::default();
        let events = parser
            .push(
                concat!(
                    "event: message_start\n",
                    "data: {\"message\":{\"usage\":{\"input_tokens\":11,\"output_tokens\":0}}}\n\n",
                    "event: content_block_start\n",
                    "data: {\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"get_weather\"}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"London\\\"}\"}}\n\n",
                    "event: content_block_stop\n",
                    "data: {\"index\":0}\n\n",
                    "event: content_block_start\n",
                    "data: {\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Done\"}}\n\n",
                    "event: message_delta\n",
                    "data: {\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n",
                    "event: message_stop\n",
                    "data: {}\n\n"
                )
                .as_bytes(),
            )
            .expect("parse sse");

        assert_eq!(
            events,
            vec![
                crate::StreamEvent::ToolCallStart {
                    id: "tool-1".to_owned(),
                    name: "get_weather".to_owned()
                },
                crate::StreamEvent::ToolCallDelta {
                    id: "tool-1".to_owned(),
                    arguments_delta: "{\"city\":".to_owned()
                },
                crate::StreamEvent::ToolCallDelta {
                    id: "tool-1".to_owned(),
                    arguments_delta: "\"London\"}".to_owned()
                },
                crate::StreamEvent::ToolCallEnd {
                    id: "tool-1".to_owned()
                },
                crate::StreamEvent::TextDelta("Done".to_owned()),
                crate::StreamEvent::StopReason(StopReason::ToolUse),
                crate::StreamEvent::Usage(Usage {
                    input_tokens: 11,
                    output_tokens: 7,
                    reasoning_tokens: None
                }),
                crate::StreamEvent::Done,
            ]
        );
    }

    #[test]
    fn dropping_stream_releases_adapter_task() {
        let server = MockServer::start(vec![MockResponse {
            status: 200,
            content_type: "text/event-stream",
            extra_headers: Vec::new(),
            content_length_override: Some(10_000),
            linger_after_body: Some(Duration::from_millis(250)),
            body: concat!(
                "event: content_block_start\n",
                "data: {\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n"
            )
            .to_owned(),
        }]);

        let backend = backend_for_server(server.base_url());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let mut stream = backend
                .stream(CompletionRequest::single_user_message("hello", 16))
                .expect("stream");
            assert_eq!(backend.active_stream_count(), 1);
            let first = stream.next().await.expect("first event").expect("ok");
            assert_eq!(first, crate::StreamEvent::TextDelta("partial".to_owned()));
            drop(stream);
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert_eq!(backend.active_stream_count(), 0);
        });
    }
}
