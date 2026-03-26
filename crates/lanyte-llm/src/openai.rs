use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use reqwest::blocking::Client as BlockingClient;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use reqwest::Client as AsyncClient;
#[cfg(test)]
use std::sync::Arc;
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

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
static JITTER_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub(crate) struct OpenAiBackendOptions {
    pub base_url: String,
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub base_delay_500: Duration,
    pub max_jitter: Duration,
    pub jitter_seed: Option<u64>,
    pub connect_timeout: Duration,
    pub timeout: Duration,
    pub stream_idle_timeout: Duration,
    pub max_per_attempt_sleep: Duration,
    pub total_retry_budget: Duration,
    pub pool_max_idle_per_host: usize,
}

impl Default for OpenAiBackendOptions {
    fn default() -> Self {
        Self {
            base_url: OPENAI_BASE_URL.to_owned(),
            max_attempts: 3,
            base_delay: Duration::from_secs(1),
            base_delay_500: Duration::from_secs(3),
            max_jitter: Duration::from_millis(250),
            jitter_seed: None,
            connect_timeout: Duration::from_secs(10),
            timeout: Duration::from_secs(60),
            stream_idle_timeout: Duration::from_secs(30),
            max_per_attempt_sleep: Duration::from_secs(30),
            total_retry_budget: Duration::from_secs(90),
            pool_max_idle_per_host: 8,
        }
    }
}

#[derive(Clone)]
pub struct OpenAiBackend {
    blocking_client: BlockingClient,
    streaming_client: AsyncClient,
    model: String,
    api_key: String,
    chat_url: String,
    max_attempts: u32,
    base_delay: Duration,
    base_delay_500: Duration,
    max_jitter: Duration,
    jitter_seed: u64,
    stream_idle_timeout: Duration,
    max_per_attempt_sleep: Duration,
    total_retry_budget: Duration,
    #[cfg(test)]
    active_streams: Arc<AtomicUsize>,
}

impl OpenAiBackend {
    pub fn from_config(config: &lanyte_common::OpenAiConfig) -> Result<Self> {
        let api_key = config.api_key.clone().ok_or(LlmError::MissingApiKey)?;
        Self::new_with_base_url(config.model.clone(), api_key, config.base_url.clone())
    }

    pub fn new(model: String, api_key: String) -> Result<Self> {
        Self::new_with_options(model, api_key, OpenAiBackendOptions::default())
    }

    pub fn new_with_base_url(model: String, api_key: String, base_url: String) -> Result<Self> {
        Self::new_with_options(
            model,
            api_key,
            OpenAiBackendOptions {
                base_url,
                ..OpenAiBackendOptions::default()
            },
        )
    }

    pub(crate) fn new_with_options(
        model: String,
        api_key: String,
        options: OpenAiBackendOptions,
    ) -> Result<Self> {
        if api_key.trim().is_empty() {
            return Err(LlmError::MissingApiKey);
        }

        let base_url = options.base_url.trim_end_matches('/').to_owned();
        if base_url.is_empty() {
            return Err(LlmError::InvalidResponse(
                "OpenAI base_url must not be empty",
            ));
        }

        let blocking_client = BlockingClient::builder()
            .connect_timeout(options.connect_timeout)
            .timeout(options.timeout)
            .pool_max_idle_per_host(options.pool_max_idle_per_host)
            .build()?;
        let streaming_client = AsyncClient::builder()
            .connect_timeout(options.connect_timeout)
            .pool_max_idle_per_host(options.pool_max_idle_per_host)
            .build()?;

        let jitter_seed = options.jitter_seed.unwrap_or_else(|| {
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
            chat_url: format!("{base_url}/chat/completions"),
            max_attempts: options.max_attempts,
            base_delay: options.base_delay,
            base_delay_500: options.base_delay_500,
            max_jitter: options.max_jitter,
            jitter_seed,
            stream_idle_timeout: options.stream_idle_timeout,
            max_per_attempt_sleep: options.max_per_attempt_sleep,
            total_retry_budget: options.total_retry_budget,
            #[cfg(test)]
            active_streams: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| LlmError::InvalidResponse("API key contains invalid header bytes"))?,
        );
        Ok(headers)
    }

    fn do_complete_once(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let headers = self.headers()?;
        let openai_req = self.build_request(request, false)?;
        let resp = self
            .blocking_client
            .post(&self.chat_url)
            .headers(headers)
            .json(&openai_req)
            .send()?;

        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        let body = resp.text()?;

        if !status.is_success() {
            return Err(classify_error(status.as_u16(), retry_after, &body));
        }

        let parsed: OpenAiChatResponse = serde_json::from_str(&body)?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or(LlmError::InvalidResponse("missing choices"))?;

        Ok(CompletionResponse {
            response_id: parsed.id,
            text: choice.message.content.unwrap_or_default(),
            stop_reason: choice
                .finish_reason
                .as_deref()
                .map(map_finish_reason)
                .unwrap_or(StopReason::EndTurn),
            usage: parsed.usage.map(Into::into),
        })
    }

    fn should_retry_status(status: u16) -> bool {
        matches!(status, 429 | 500 | 502 | 503 | 504)
    }

    fn base_delay_for_status(&self, status: u16) -> Duration {
        if status == 500 {
            self.base_delay_500.max(self.base_delay)
        } else {
            self.base_delay
        }
    }

    fn backoff_for(&self, status: u16, attempt: u32) -> Duration {
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

    fn bounded_sleep_for(
        &self,
        base_delay: Duration,
        attempt: u32,
        slept_so_far: Duration,
    ) -> Option<Duration> {
        let capped = base_delay
            .saturating_add(self.jitter_for_attempt(attempt))
            .min(self.max_per_attempt_sleep);
        let remaining_budget = self.total_retry_budget.saturating_sub(slept_so_far);
        if remaining_budget.is_zero() {
            None
        } else {
            Some(capped.min(remaining_budget))
        }
    }

    fn build_request(&self, request: &CompletionRequest, stream: bool) -> Result<OpenAiRequest> {
        let mut messages = Vec::new();
        if let Some(system_prompt) = request.system_prompt.as_ref() {
            messages.push(OpenAiMessage {
                role: "system",
                content: Some(system_prompt.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }

        messages.extend(
            request
                .messages
                .iter()
                .map(OpenAiMessage::from_message)
                .collect::<Result<Vec<_>>>()?,
        );
        if !request.tool_results.is_empty() {
            messages.push(OpenAiMessage::assistant_tool_calls(&request.tool_results)?);
            messages.extend(
                request
                    .tool_results
                    .iter()
                    .map(OpenAiMessage::from_tool_result),
            );
        }

        Ok(OpenAiRequest {
            model: self.model.clone(),
            messages,
            max_completion_tokens: request.max_tokens,
            temperature: request.temperature,
            stream,
            stream_options: stream.then_some(OpenAiStreamOptions {
                include_usage: true,
            }),
            tools: request
                .tools
                .iter()
                .map(OpenAiTool::try_from)
                .collect::<Result<Vec<_>>>()?,
            parallel_tool_calls: request.parallel_tool_calls,
        })
    }

    #[cfg(test)]
    fn active_stream_count(&self) -> usize {
        self.active_streams.load(Ordering::SeqCst)
    }
}

impl fmt::Debug for OpenAiBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiBackend")
            .field("model", &self.model)
            .field("chat_url", &self.chat_url)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl LlmBackend for OpenAiBackend {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mut attempt: u32 = 0;
        let mut slept_so_far = Duration::ZERO;

        loop {
            attempt += 1;
            match self.do_complete_once(&request) {
                Ok(resp) => return Ok(resp),
                Err(LlmError::RateLimited { retry_after }) => {
                    if attempt >= self.max_attempts {
                        return Err(LlmError::RateLimited { retry_after });
                    }
                    let delay = retry_after.unwrap_or_else(|| self.backoff_for(429, attempt));
                    let Some(sleep_for) = self.bounded_sleep_for(delay, attempt, slept_so_far)
                    else {
                        return Err(LlmError::RateLimited { retry_after });
                    };
                    if !sleep_for.is_zero() {
                        thread::sleep(sleep_for);
                        slept_so_far = slept_so_far.saturating_add(sleep_for);
                    }
                }
                Err(LlmError::ServiceUnavailable) => {
                    if attempt >= self.max_attempts {
                        return Err(LlmError::ServiceUnavailable);
                    }
                    let delay = self.backoff_for(503, attempt);
                    let Some(sleep_for) = self.bounded_sleep_for(delay, attempt, slept_so_far)
                    else {
                        return Err(LlmError::ServiceUnavailable);
                    };
                    if !sleep_for.is_zero() {
                        thread::sleep(sleep_for);
                        slept_so_far = slept_so_far.saturating_add(sleep_for);
                    }
                }
                Err(LlmError::Upstream { status, message }) => {
                    if attempt >= self.max_attempts || !Self::should_retry_status(status) {
                        return Err(LlmError::Upstream { status, message });
                    }
                    let delay = self.backoff_for(status, attempt);
                    let Some(sleep_for) = self.bounded_sleep_for(delay, attempt, slept_so_far)
                    else {
                        return Err(LlmError::Upstream { status, message });
                    };
                    if !sleep_for.is_zero() {
                        thread::sleep(sleep_for);
                        slept_so_far = slept_so_far.saturating_add(sleep_for);
                    }
                }
                Err(other) => return Err(other),
            }
        }
    }

    fn stream(&self, request: CompletionRequest) -> Result<LlmStream> {
        let handle = Handle::try_current()
            .map_err(|_| LlmError::Unsupported("OpenAI streaming requires tokio runtime"))?;
        let headers = self.headers()?;
        let openai_req = self.build_request(&request, true)?;
        let client = self.streaming_client.clone();
        let url = self.chat_url.clone();
        let stream_idle_timeout = self.stream_idle_timeout;
        let (tx, rx) = mpsc::unbounded_channel();
        #[cfg(test)]
        let active_streams = Arc::clone(&self.active_streams);

        #[cfg(test)]
        active_streams.fetch_add(1, Ordering::SeqCst);

        let task = handle.spawn(async move {
            #[cfg(test)]
            let _guard = OpenAiStreamTaskGuard(active_streams);
            let _ = stream_sse_response(client, url, headers, openai_req, stream_idle_timeout, tx)
                .await;
        });

        Ok(Box::pin(OpenAiLlmStream {
            inner: UnboundedReceiverStream::new(rx),
            task: Some(task),
        }))
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_system_prompt: true,
            supports_parallel_tool_calls: true,
            supports_web_search: false,
            supports_image_generation: false,
            max_context_tokens: 128_000,
        }
    }

    fn health(&self) -> HealthStatus {
        if self.api_key.trim().is_empty() {
            return HealthStatus::Unconfigured;
        }
        HealthStatus::Healthy
    }

    fn name(&self) -> &'static str {
        "openai"
    }
}

#[derive(Debug, serde::Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OpenAiStreamOptions>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

#[derive(Debug, serde::Serialize)]
struct OpenAiStreamOptions {
    include_usage: bool,
}

#[derive(Debug, serde::Serialize)]
struct OpenAiMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiAssistantToolCall>>,
}

impl OpenAiMessage {
    fn from_message(message: &crate::Message) -> Result<Self> {
        match message.role {
            MessageRole::System => Err(LlmError::UnsupportedMessageRole(
                "system messages must use CompletionRequest.system_prompt",
            )),
            MessageRole::Tool => Err(LlmError::UnsupportedMessageRole(
                "tool results must use CompletionRequest.tool_results",
            )),
            MessageRole::User => Ok(Self {
                role: "user",
                content: Some(message.content.clone()),
                tool_call_id: None,
                tool_calls: None,
            }),
            MessageRole::Assistant => Ok(Self {
                role: "assistant",
                content: Some(message.content.clone()),
                tool_call_id: None,
                tool_calls: None,
            }),
        }
    }

    fn from_tool_result(result: &crate::ToolResult) -> Self {
        Self {
            role: "tool",
            content: Some(result.content.clone()),
            tool_call_id: Some(result.tool_call_id.clone()),
            tool_calls: None,
        }
    }

    fn assistant_tool_calls(tool_results: &[crate::ToolResult]) -> Result<Self> {
        let mut tool_calls = Vec::with_capacity(tool_results.len());
        for result in tool_results {
            let name = result.tool_name.clone().ok_or(LlmError::Unsupported(
                "OpenAI tool results require tool_name for follow-up requests",
            ))?;
            let arguments = result.arguments.clone().ok_or(LlmError::Unsupported(
                "OpenAI tool results require arguments for follow-up requests",
            ))?;
            tool_calls.push(OpenAiAssistantToolCall {
                id: result.tool_call_id.clone(),
                kind: "function",
                function: OpenAiAssistantFunctionCall { name, arguments },
            });
        }

        Ok(Self {
            role: "assistant",
            content: None,
            tool_call_id: None,
            tool_calls: Some(tool_calls),
        })
    }
}

#[derive(Debug, serde::Serialize)]
struct OpenAiAssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiAssistantFunctionCall,
}

#[derive(Debug, serde::Serialize)]
struct OpenAiAssistantFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, serde::Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionTool,
}

#[derive(Debug, serde::Serialize)]
struct OpenAiFunctionTool {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl TryFrom<&ToolDefinition> for OpenAiTool {
    type Error = LlmError;

    fn try_from(tool: &ToolDefinition) -> Result<Self> {
        match tool {
            ToolDefinition::Function {
                name,
                description,
                parameters,
            } => Ok(Self {
                kind: "function",
                function: OpenAiFunctionTool {
                    name: name.clone(),
                    description: description.clone(),
                    parameters: parameters.clone(),
                },
            }),
            ToolDefinition::WebSearch { .. } | ToolDefinition::ImageGeneration => Err(
                LlmError::Unsupported("OpenAI adapter only supports function tools"),
            ),
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChatResponse {
    #[serde(default)]
    id: Option<String>,
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiResponseMessage {
    // Sync `complete()` currently treats OpenAI tool-use as a stream-first path.
    // If the provider returns tool calls on the unary chat-completions path, the
    // finish_reason is preserved but the tool call payload is not surfaced here.
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiError,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiError {
    message: String,
    #[serde(default, rename = "type")]
    kind: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    completion_tokens_details: Option<OpenAiCompletionTokenDetails>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiCompletionTokenDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

impl From<OpenAiUsage> for Usage {
    fn from(value: OpenAiUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            output_tokens: value.completion_tokens,
            reasoning_tokens: value
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens),
        }
    }
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

fn classify_error(status: u16, retry_after: Option<Duration>, body: &str) -> LlmError {
    let parsed = serde_json::from_str::<OpenAiErrorEnvelope>(body).ok();
    let message = parsed
        .as_ref()
        .map(|envelope| envelope.error.message.clone())
        .unwrap_or_else(|| body.to_owned());
    let kind = parsed.and_then(|envelope| envelope.error.kind);

    match status {
        401 | 403 => LlmError::AuthenticationFailed,
        429 => LlmError::RateLimited { retry_after },
        503 => LlmError::ServiceUnavailable,
        404 if message.to_ascii_lowercase().contains("model") => LlmError::InvalidModel,
        400 if message.to_ascii_lowercase().contains("model") => LlmError::InvalidModel,
        400 if matches!(kind.as_deref(), Some("invalid_request_error"))
            && message.to_ascii_lowercase().contains("does not exist") =>
        {
            LlmError::InvalidModel
        }
        _ => LlmError::Upstream { status, message },
    }
}

fn map_finish_reason(value: &str) -> StopReason {
    match value {
        "length" => StopReason::MaxTokens,
        "tool_calls" => StopReason::ToolUse,
        "content_filter" => StopReason::ContentFiltered,
        _ => StopReason::EndTurn,
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

struct OpenAiLlmStream {
    inner: UnboundedReceiverStream<Result<crate::StreamEvent>>,
    task: Option<JoinHandle<()>>,
}

impl Drop for OpenAiLlmStream {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

impl futures_core::Stream for OpenAiLlmStream {
    type Item = Result<crate::StreamEvent>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

#[cfg(test)]
struct OpenAiStreamTaskGuard(Arc<AtomicUsize>);

#[cfg(test)]
impl Drop for OpenAiStreamTaskGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn stream_sse_response(
    client: AsyncClient,
    url: String,
    headers: HeaderMap,
    request: OpenAiRequest,
    stream_idle_timeout: Duration,
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
        let err = classify_error(status.as_u16(), retry_after, &body);
        let _ = tx.send(Err(err));
        return Ok(());
    }

    let mut parser = OpenAiSseParser::default();
    let mut stream = response.bytes_stream();
    loop {
        let next_chunk = tokio::time::timeout(stream_idle_timeout, stream.next())
            .await
            .map_err(|_| LlmError::Upstream {
                status: 408,
                message: "stream idle timeout exceeded".to_owned(),
            })?;
        let Some(chunk) = next_chunk else {
            break;
        };
        let bytes = chunk?;
        let events = parser.push(&bytes)?;
        for event in events {
            if tx.send(Ok(event)).is_err() {
                return Ok(());
            }
        }
    }

    Ok(())
}

#[derive(Default)]
struct OpenAiSseParser {
    buffer: String,
    tool_calls: BTreeMap<u32, ToolCallState>,
}

#[derive(Default)]
struct ToolCallState {
    id: Option<String>,
    name: Option<String>,
    pending_arguments: String,
    started: bool,
    ended: bool,
}

impl OpenAiSseParser {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<crate::StreamEvent>> {
        let chunk = std::str::from_utf8(bytes)
            .map_err(|_| LlmError::InvalidResponse("OpenAI SSE chunk must be UTF-8"))?;
        self.buffer.push_str(chunk);

        let mut out = Vec::new();
        while let Some(idx) = self.buffer.find("\n\n") {
            let frame = self.buffer[..idx].to_owned();
            self.buffer.drain(..idx + 2);
            self.handle_frame(&frame, &mut out)?;
        }

        Ok(out)
    }

    fn handle_frame(&mut self, frame: &str, out: &mut Vec<crate::StreamEvent>) -> Result<()> {
        let mut data_lines = Vec::new();
        for line in frame.lines() {
            if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim_start());
            }
        }

        if data_lines.is_empty() {
            return Ok(());
        }

        let data = data_lines.join("\n");
        if data == "[DONE]" {
            self.close_open_tool_calls(out);
            out.push(crate::StreamEvent::Done);
            return Ok(());
        }

        let chunk: OpenAiChatChunk = serde_json::from_str(&data)?;
        if let Some(usage) = chunk.usage {
            out.push(crate::StreamEvent::Usage(usage.into()));
        }

        for choice in chunk.choices {
            if let Some(delta) = choice.delta {
                if let Some(content) = delta.content {
                    out.push(crate::StreamEvent::TextDelta(content));
                }
                for tool_call in delta.tool_calls {
                    self.handle_tool_call_delta(tool_call, out);
                }
            }

            if let Some(finish_reason) = choice.finish_reason {
                if finish_reason == "tool_calls" {
                    self.close_open_tool_calls(out);
                }
                out.push(crate::StreamEvent::StopReason(map_finish_reason(
                    &finish_reason,
                )));
            }
        }

        Ok(())
    }

    fn handle_tool_call_delta(
        &mut self,
        tool_call: OpenAiToolCallDelta,
        out: &mut Vec<crate::StreamEvent>,
    ) {
        let state = self.tool_calls.entry(tool_call.index).or_default();
        if let Some(id) = tool_call.id {
            state.id = Some(id);
        }
        if let Some(function) = tool_call.function {
            if let Some(name) = function.name {
                state.name = Some(name);
            }
            if let Some(arguments) = function.arguments {
                state.pending_arguments.push_str(&arguments);
            }
        }

        if !state.started {
            if let (Some(id), Some(name)) = (state.id.clone(), state.name.clone()) {
                out.push(crate::StreamEvent::ToolCallStart {
                    id: id.clone(),
                    name,
                });
                state.started = true;
                if !state.pending_arguments.is_empty() {
                    out.push(crate::StreamEvent::ToolCallDelta {
                        id,
                        arguments_delta: std::mem::take(&mut state.pending_arguments),
                    });
                }
            }
        } else if !state.pending_arguments.is_empty() {
            if let Some(id) = state.id.clone() {
                out.push(crate::StreamEvent::ToolCallDelta {
                    id,
                    arguments_delta: std::mem::take(&mut state.pending_arguments),
                });
            }
        }
    }

    fn close_open_tool_calls(&mut self, out: &mut Vec<crate::StreamEvent>) {
        for state in self.tool_calls.values_mut() {
            if state.started && !state.ended {
                if let Some(id) = state.id.clone() {
                    if !state.pending_arguments.is_empty() {
                        out.push(crate::StreamEvent::ToolCallDelta {
                            id: id.clone(),
                            arguments_delta: std::mem::take(&mut state.pending_arguments),
                        });
                    }
                    out.push(crate::StreamEvent::ToolCallEnd { id });
                    state.ended = true;
                }
            }
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChatChunk {
    #[serde(default)]
    choices: Vec<OpenAiChunkChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChunkChoice {
    #[serde(default)]
    delta: Option<OpenAiChunkDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAiToolCallDelta>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiToolCallDelta {
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiFunctionDelta>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;

    #[test]
    fn jitter_seed_is_respected_for_deterministic_tests() {
        let options = OpenAiBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            base_delay: Duration::ZERO,
            base_delay_500: Duration::ZERO,
            max_jitter: Duration::from_secs(1),
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend =
            OpenAiBackend::new_with_options("m".to_owned(), "k".to_owned(), options).unwrap();
        assert_eq!(backend.jitter_for_attempt(1), backend.jitter_for_attempt(1));
        assert_ne!(backend.jitter_for_attempt(1), backend.jitter_for_attempt(2));
    }

    #[test]
    fn backoff_uses_longer_baseline_for_500() {
        let options = OpenAiBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            base_delay: Duration::from_secs(1),
            base_delay_500: Duration::from_secs(3),
            max_jitter: Duration::ZERO,
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend =
            OpenAiBackend::new_with_options("m".to_owned(), "k".to_owned(), options).unwrap();
        assert_eq!(backend.backoff_for(503, 1), Duration::from_secs(1));
        assert_eq!(backend.backoff_for(500, 1), Duration::from_secs(3));
        assert_eq!(backend.backoff_for(500, 2), Duration::from_secs(6));
    }

    #[test]
    fn bounded_sleep_clamps_each_attempt_to_policy_cap() {
        let options = OpenAiBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            max_jitter: Duration::ZERO,
            max_per_attempt_sleep: Duration::from_secs(30),
            total_retry_budget: Duration::from_secs(90),
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend =
            OpenAiBackend::new_with_options("m".to_owned(), "k".to_owned(), options).unwrap();

        assert_eq!(
            backend.bounded_sleep_for(Duration::from_secs(120), 1, Duration::ZERO),
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn bounded_sleep_respects_remaining_retry_budget() {
        let options = OpenAiBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            max_jitter: Duration::ZERO,
            max_per_attempt_sleep: Duration::from_secs(30),
            total_retry_budget: Duration::from_secs(90),
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend =
            OpenAiBackend::new_with_options("m".to_owned(), "k".to_owned(), options).unwrap();

        assert_eq!(
            backend.bounded_sleep_for(Duration::from_secs(20), 1, Duration::from_secs(85)),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            backend.bounded_sleep_for(Duration::from_secs(20), 1, Duration::from_secs(90)),
            None
        );
    }

    #[test]
    fn default_options_match_adr_0014_stream_idle_timeout() {
        let options = OpenAiBackendOptions::default();
        assert_eq!(options.stream_idle_timeout, Duration::from_secs(30));
    }

    #[test]
    fn serializes_chat_completions_request_shape() {
        let server = MockServer::start(vec![MockResponse::json(
            200,
            r#"{
                "id":"chatcmpl-1",
                "choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":12,"completion_tokens":7}
            }"#,
        )]);

        let backend = backend_for_server(server.base_url());
        let request = CompletionRequest {
            system_prompt: Some("You are helpful".to_owned()),
            previous_response_id: None,
            messages: vec![
                crate::Message::user("Find current weather"),
                crate::Message::assistant("Let me check"),
            ],
            tools: vec![ToolDefinition::function(
                "get_weather",
                "Return the weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
            )],
            tool_results: vec![crate::ToolResult::with_call(
                "call-1",
                "get_weather",
                r#"{"city":"London"}"#,
                r#"{"city":"London","forecast":"rain"}"#,
            )],
            max_tokens: Some(32),
            temperature: Some(0.25),
            parallel_tool_calls: Some(true),
        };

        let response = backend.complete(request).expect("complete");
        assert_eq!(response.text, "ok");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.response_id.as_deref(), Some("chatcmpl-1"));

        let requests = server.take_requests();
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "You are helpful");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][2]["role"], "assistant");
        assert_eq!(body["messages"][3]["role"], "assistant");
        assert_eq!(body["messages"][3]["tool_calls"][0]["id"], "call-1");
        assert_eq!(body["messages"][3]["tool_calls"][0]["type"], "function");
        assert_eq!(
            body["messages"][3]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(body["messages"][4]["role"], "tool");
        assert_eq!(body["messages"][4]["tool_call_id"], "call-1");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(body["max_completion_tokens"], serde_json::json!(32));
        assert_eq!(body["parallel_tool_calls"], serde_json::json!(true));
    }

    #[test]
    fn stream_request_includes_usage_option() {
        let server = MockServer::start(vec![MockResponse::sse(200, "data: [DONE]\n\n", None)]);

        let backend = backend_for_server(server.base_url());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let mut stream = backend
                .stream(CompletionRequest::single_user_message("hello", 16))
                .expect("stream");
            use futures_util::StreamExt;
            while let Some(event) = stream.next().await {
                if matches!(event.expect("stream event"), crate::StreamEvent::Done) {
                    break;
                }
            }
        });

        let requests = server.take_requests();
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(
            body["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn rejects_provider_native_tools_in_openai_requests() {
        let backend = backend_for_server("https://example.invalid/v1".to_owned());
        let mut request = CompletionRequest::single_user_message("Search weather", 16);
        request.tools = vec![ToolDefinition::web_search(Vec::new(), Vec::new())];
        let err = backend
            .complete(request)
            .expect_err("must reject web search tool");
        assert!(
            matches!(err, LlmError::Unsupported(message) if message.contains("function tools"))
        );
    }

    #[test]
    fn rejects_openai_tool_result_without_call_metadata() {
        let backend = backend_for_server("https://example.invalid/v1".to_owned());
        let mut request = CompletionRequest::single_user_message("Use tool result", 16);
        request.tool_results.push(crate::ToolResult::new(
            "call-1",
            r#"{"city":"London","forecast":"rain"}"#,
        ));
        let err = backend
            .complete(request)
            .expect_err("must reject tool result without metadata");
        assert!(matches!(
            err,
            LlmError::Unsupported(message) if message.contains("tool_name")
        ));
    }

    #[test]
    fn maps_openai_errors_to_typed_errors() {
        assert!(matches!(
            classify_error(
                401,
                None,
                r#"{ "error": { "type": "invalid_api_key", "message": "bad key" } }"#
            ),
            LlmError::AuthenticationFailed
        ));

        assert!(matches!(
            classify_error(
                429,
                Some(Duration::from_secs(7)),
                r#"{ "error": { "type": "rate_limit", "message": "slow down" } }"#
            ),
            LlmError::RateLimited {
                retry_after: Some(duration)
            } if duration == Duration::from_secs(7)
        ));

        assert!(matches!(
            classify_error(
                400,
                None,
                r#"{ "error": { "type": "invalid_request_error", "message": "The model `missing` does not exist" } }"#
            ),
            LlmError::InvalidModel
        ));
    }

    #[test]
    fn sse_parser_normalizes_text_tool_calls_usage_and_finish_reason() {
        let mut parser = OpenAiSseParser::default();
        let events = parser
            .push(
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"London\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"completion_tokens_details\":{\"reasoning_tokens\":3}}}\n\n",
                    "data: [DONE]\n\n"
                )
                .as_bytes(),
            )
            .expect("parse events");

        assert_eq!(
            events[0],
            crate::StreamEvent::TextDelta("Hello ".to_owned())
        );
        assert_eq!(events[1], crate::StreamEvent::TextDelta("world".to_owned()));
        assert_eq!(
            events[2],
            crate::StreamEvent::ToolCallStart {
                id: "call-1".to_owned(),
                name: "get_weather".to_owned(),
            }
        );
        assert_eq!(
            events[3],
            crate::StreamEvent::ToolCallDelta {
                id: "call-1".to_owned(),
                arguments_delta: r#"{"city":"#.to_owned(),
            }
        );
        assert_eq!(
            events[4],
            crate::StreamEvent::ToolCallDelta {
                id: "call-1".to_owned(),
                arguments_delta: r#""London"}"#.to_owned(),
            }
        );
        assert_eq!(
            events[5],
            crate::StreamEvent::Usage(Usage {
                input_tokens: 11,
                output_tokens: 7,
                reasoning_tokens: Some(3),
            })
        );
        assert_eq!(
            events[6],
            crate::StreamEvent::ToolCallEnd {
                id: "call-1".to_owned(),
            }
        );
        assert_eq!(
            events[7],
            crate::StreamEvent::StopReason(StopReason::ToolUse)
        );
        assert_eq!(events[8], crate::StreamEvent::Done);
    }

    #[test]
    fn sse_parser_accumulates_parallel_tool_calls_by_index() {
        let mut parser = OpenAiSseParser::default();
        let events = parser
            .push(
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[",
                    "{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}},",
                    "{\"index\":1,\"id\":\"call-2\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{\\\"zone\\\":\"}}",
                    "]},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[",
                    "{\"index\":1,\"function\":{\"arguments\":\"\\\"UTC\\\"}\"}},",
                    "{\"index\":0,\"function\":{\"arguments\":\"\\\"London\\\"}\"}}",
                    "]},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                    "data: [DONE]\n\n"
                )
                .as_bytes(),
            )
            .expect("parse events");

        assert_eq!(
            events[0],
            crate::StreamEvent::ToolCallStart {
                id: "call-1".to_owned(),
                name: "get_weather".to_owned(),
            }
        );
        assert_eq!(
            events[1],
            crate::StreamEvent::ToolCallDelta {
                id: "call-1".to_owned(),
                arguments_delta: r#"{"city":"#.to_owned(),
            }
        );
        assert_eq!(
            events[2],
            crate::StreamEvent::ToolCallStart {
                id: "call-2".to_owned(),
                name: "get_time".to_owned(),
            }
        );
        assert_eq!(
            events[3],
            crate::StreamEvent::ToolCallDelta {
                id: "call-2".to_owned(),
                arguments_delta: r#"{"zone":"#.to_owned(),
            }
        );
        assert_eq!(
            events[4],
            crate::StreamEvent::ToolCallDelta {
                id: "call-2".to_owned(),
                arguments_delta: r#""UTC"}"#.to_owned(),
            }
        );
        assert_eq!(
            events[5],
            crate::StreamEvent::ToolCallDelta {
                id: "call-1".to_owned(),
                arguments_delta: r#""London"}"#.to_owned(),
            }
        );
        assert_eq!(
            events[6],
            crate::StreamEvent::ToolCallEnd {
                id: "call-1".to_owned(),
            }
        );
        assert_eq!(
            events[7],
            crate::StreamEvent::ToolCallEnd {
                id: "call-2".to_owned(),
            }
        );
        assert_eq!(
            events[8],
            crate::StreamEvent::StopReason(StopReason::ToolUse)
        );
        assert_eq!(events[9], crate::StreamEvent::Done);
    }

    #[test]
    fn dropping_stream_releases_adapter_task() {
        let server = MockServer::start(vec![MockResponse::sse(
            200,
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
            Some(Duration::from_millis(100)),
        )]);

        let backend = backend_for_server(server.base_url());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let mut stream = backend
                .stream(CompletionRequest::single_user_message("hello", 16))
                .expect("stream");
            let first = stream.next().await.expect("first event").expect("event ok");
            assert_eq!(first, crate::StreamEvent::TextDelta("Hello".to_owned()));
            drop(stream);
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert_eq!(backend.active_stream_count(), 0);
        });
    }

    fn backend_for_server(base_url: String) -> OpenAiBackend {
        OpenAiBackend::new_with_options(
            "gpt-4o".to_owned(),
            "test-key".to_owned(),
            OpenAiBackendOptions {
                base_url,
                max_attempts: 1,
                base_delay: Duration::ZERO,
                base_delay_500: Duration::ZERO,
                max_jitter: Duration::ZERO,
                jitter_seed: Some(1),
                connect_timeout: Duration::from_secs(2),
                timeout: Duration::from_secs(2),
                stream_idle_timeout: Duration::from_secs(2),
                max_per_attempt_sleep: Duration::from_secs(2),
                total_retry_budget: Duration::from_secs(6),
                pool_max_idle_per_host: 1,
            },
        )
        .expect("backend")
    }

    #[derive(Debug, Clone)]
    struct CapturedRequest {
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
        fn json(status: u16, body: &str) -> Self {
            Self {
                status,
                content_type: "application/json",
                extra_headers: Vec::new(),
                content_length_override: None,
                linger_after_body: None,
                body: body.to_owned(),
            }
        }

        fn sse(status: u16, body: &str, linger_after_body: Option<Duration>) -> Self {
            Self {
                status,
                content_type: "text/event-stream",
                extra_headers: Vec::new(),
                content_length_override: None,
                linger_after_body,
                body: body.to_owned(),
            }
        }
    }

    struct MockServer {
        addr: std::net::SocketAddr,
        captured: Arc<Mutex<Vec<CapturedRequest>>>,
        join: Option<JoinHandle<()>>,
    }

    impl MockServer {
        fn start(responses: Vec<MockResponse>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("addr");
            let captured = Arc::new(Mutex::new(Vec::new()));
            let captured_clone = Arc::clone(&captured);
            let join = std::thread::spawn(move || {
                for response in responses {
                    let (mut stream, _) = listener.accept().expect("accept");
                    let request = read_http_request(&mut stream).expect("request");
                    captured_clone
                        .lock()
                        .expect("captured lock")
                        .push(CapturedRequest { body: request.body });
                    write_http_response(&mut stream, &response).expect("response");
                }
            });

            Self {
                addr,
                captured,
                join: Some(join),
            }
        }

        fn base_url(&self) -> String {
            format!("http://{}", self.addr)
        }

        fn take_requests(&self) -> Vec<CapturedRequest> {
            self.captured.lock().expect("captured lock").clone()
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            let _ = std::net::TcpStream::connect(self.addr);
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
        }
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> std::io::Result<CapturedRequest> {
        let mut buf = Vec::new();
        let mut header_end = None;
        while header_end.is_none() {
            let mut chunk = [0_u8; 1024];
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..read]);
            header_end = find_header_end(&buf);
        }

        let header_end = header_end.expect("http headers");
        let headers_bytes = &buf[..header_end];
        let body_start = header_end + 4;
        let mut body = buf[body_start..].to_vec();
        let headers_text = String::from_utf8_lossy(headers_bytes);
        let headers = parse_headers(&headers_text);
        let content_length = headers
            .get("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);

        while body.len() < content_length {
            let mut chunk = vec![0_u8; content_length - body.len()];
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..read]);
        }

        Ok(CapturedRequest {
            body: String::from_utf8_lossy(&body).into_owned(),
        })
    }

    fn write_http_response(
        stream: &mut std::net::TcpStream,
        response: &MockResponse,
    ) -> std::io::Result<()> {
        let body_bytes = response.body.as_bytes();
        let mut extra_headers = String::new();
        for (key, value) in &response.extra_headers {
            use std::fmt::Write as _;
            let _ = write!(&mut extra_headers, "{key}: {value}\r\n");
        }
        let content_length = response
            .content_length_override
            .unwrap_or(response.body.len());
        let head = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n",
            response.status, response.content_type, content_length, extra_headers
        );
        stream.write_all(head.as_bytes())?;
        stream.write_all(body_bytes)?;
        stream.flush()?;
        if let Some(linger) = response.linger_after_body {
            std::thread::sleep(linger);
        }
        Ok(())
    }

    fn parse_headers(headers_text: &str) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for line in headers_text.lines().skip(1) {
            if let Some((name, value)) = line.split_once(':') {
                out.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
            }
        }
        out
    }

    fn find_header_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|window| window == b"\r\n\r\n")
    }
}
