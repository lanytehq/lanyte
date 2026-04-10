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

const XAI_BASE_URL: &str = "https://api.x.ai";
static JITTER_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub(crate) struct GrokBackendOptions {
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

impl Default for GrokBackendOptions {
    fn default() -> Self {
        Self {
            base_url: XAI_BASE_URL.to_owned(),
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
pub struct GrokBackend {
    blocking_client: BlockingClient,
    streaming_client: AsyncClient,
    model: String,
    api_key: String,
    responses_url: String,
    images_generations_url: String,
    images_edits_url: String,
    max_attempts: u32,
    base_delay: Duration,
    base_delay_500: Duration,
    max_jitter: Duration,
    jitter_seed: u64,
    #[cfg(test)]
    active_streams: Arc<AtomicUsize>,
}

impl GrokBackend {
    pub fn from_config(config: &lanyte_common::GrokConfig) -> Result<Self> {
        let api_key = config.api_key.clone().ok_or(LlmError::MissingApiKey)?;
        Self::new(config.model.clone(), api_key)
    }

    pub fn new(model: String, api_key: String) -> Result<Self> {
        Self::new_with_options(model, api_key, GrokBackendOptions::default())
    }

    pub(crate) fn new_with_options(
        model: String,
        api_key: String,
        options: GrokBackendOptions,
    ) -> Result<Self> {
        if api_key.trim().is_empty() {
            return Err(LlmError::MissingApiKey);
        }

        let base_url = options.base_url.trim_end_matches('/').to_owned();
        if base_url.is_empty() {
            return Err(LlmError::InvalidResponse("Grok base_url must not be empty"));
        }

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
            responses_url: format!("{base_url}/v1/responses"),
            images_generations_url: format!("{base_url}/v1/images/generations"),
            images_edits_url: format!("{base_url}/v1/images/edits"),
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
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| LlmError::InvalidResponse("API key contains invalid header bytes"))?,
        );
        Ok(headers)
    }

    pub fn generate_image(
        &self,
        request: GrokImageGenerationRequest,
    ) -> Result<GrokImageGenerationResponse> {
        let body = GrokImageGenerationBody::from_request(request, self.model.clone());
        self.retry_json_request(&self.images_generations_url, &body)
    }

    pub fn edit_image(&self, request: GrokImageEditRequest) -> Result<GrokImageEditResponse> {
        let body = GrokImageEditBody::from_request(request, self.model.clone())?;
        self.retry_json_request(&self.images_edits_url, &body)
    }

    fn retry_json_request<TRequest, TResponse>(
        &self,
        url: &str,
        request: &TRequest,
    ) -> Result<TResponse>
    where
        TRequest: serde::Serialize,
        TResponse: serde::de::DeserializeOwned,
    {
        let mut attempt: u32 = 0;

        loop {
            attempt += 1;
            match self.do_json_request_once(url, request) {
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

    fn do_json_request_once<TRequest, TResponse>(
        &self,
        url: &str,
        request: &TRequest,
    ) -> Result<TResponse>
    where
        TRequest: serde::Serialize,
        TResponse: serde::de::DeserializeOwned,
    {
        let headers = self.headers()?;
        let resp = self
            .blocking_client
            .post(url)
            .headers(headers)
            .json(request)
            .send()?;

        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        let body = resp.text()?;

        if !status.is_success() {
            return Err(classify_error(status.as_u16(), retry_after, &body));
        }

        Ok(serde_json::from_str(&body)?)
    }

    fn do_complete_once(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let grok_req = self.build_request(request, false)?;
        let parsed: GrokResponse = self.do_json_request_once(&self.responses_url, &grok_req)?;
        let text = parsed.output_text().ok_or(LlmError::InvalidResponse(
            "missing assistant text output items",
        ))?;

        Ok(CompletionResponse {
            response_id: parsed.id,
            text,
            stop_reason: parsed
                .stop_reason
                .as_deref()
                .map(map_stop_reason)
                .unwrap_or(StopReason::EndTurn),
            usage: parsed.usage.map(Into::into),
        })
    }

    fn should_retry_status(status: u16) -> bool {
        status == 429 || status == 500 || status == 502 || status == 503 || status == 504
    }

    fn base_delay_for_status(&self, status: u16) -> Duration {
        if status == 500 {
            self.base_delay_500.max(self.base_delay)
        } else {
            self.base_delay
        }
    }

    fn backoff_for(&self, status: u16, attempt: u32) -> Duration {
        let mut delay = self.base_delay_for_status(status);
        for _ in 1..attempt {
            delay = delay.saturating_mul(2);
        }
        delay
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

    fn build_request(&self, request: &CompletionRequest, stream: bool) -> Result<GrokRequest> {
        let mut input = request
            .messages
            .iter()
            .map(GrokInputItem::from_message)
            .collect::<Result<Vec<_>>>()?;
        input.extend(
            request
                .tool_results
                .iter()
                .map(GrokInputItem::from_tool_result),
        );

        Ok(GrokRequest {
            model: self.model.clone(),
            instructions: request.system_prompt.clone(),
            previous_response_id: request.previous_response_id.clone(),
            input,
            max_output_tokens: request.max_tokens,
            temperature: request.temperature,
            include: Vec::new(),
            stream,
            tools: request.tools.iter().map(GrokTool::from).collect(),
            parallel_tool_calls: request.parallel_tool_calls,
        })
    }

    #[cfg(test)]
    fn active_stream_count(&self) -> usize {
        self.active_streams
            .load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrokImageResponseFormat {
    Url,
    B64Json,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrokImageGenerationRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub n: Option<u32>,
    pub response_format: Option<GrokImageResponseFormat>,
    pub quality: Option<String>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    pub user: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrokImageEditRequest {
    pub prompt: String,
    pub image_url: Option<String>,
    pub image_urls: Vec<String>,
    pub mask_url: Option<String>,
    pub model: Option<String>,
    pub n: Option<u32>,
    pub response_format: Option<GrokImageResponseFormat>,
    pub quality: Option<String>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    pub user: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GrokImageArtifact {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub b64_json: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    pub revised_prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GrokImageUsage {
    #[serde(default)]
    pub cost_in_usd_ticks: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GrokImageGenerationResponse {
    #[serde(default)]
    pub data: Vec<GrokImageArtifact>,
    #[serde(default)]
    pub usage: Option<GrokImageUsage>,
}

pub type GrokImageEditResponse = GrokImageGenerationResponse;

impl fmt::Debug for GrokBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrokBackend")
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl LlmBackend for GrokBackend {
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
            .map_err(|_| LlmError::Unsupported("Grok streaming requires tokio runtime"))?;
        let headers = self.headers()?;
        let grok_req = self.build_request(&request, true)?;
        let client = self.streaming_client.clone();
        let url = self.responses_url.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        #[cfg(test)]
        let active_streams = Arc::clone(&self.active_streams);

        #[cfg(test)]
        active_streams.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let task = handle.spawn(async move {
            #[cfg(test)]
            let _guard = GrokStreamTaskGuard(active_streams);
            let _ = stream_sse_response(client, url, headers, grok_req, tx).await;
        });

        Ok(Box::pin(GrokLlmStream {
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
            supports_web_search: true,
            supports_image_generation: true,
            max_context_tokens: 2_000_000,
        }
    }

    fn health(&self) -> HealthStatus {
        if self.api_key.trim().is_empty() {
            return HealthStatus::Unconfigured;
        }
        HealthStatus::Healthy
    }

    fn name(&self) -> &'static str {
        "grok"
    }
}

#[derive(Debug, serde::Serialize)]
struct GrokRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    input: Vec<GrokInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    include: Vec<String>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GrokTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

#[derive(Debug, serde::Serialize)]
struct GrokImageGenerationBody {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<GrokImageResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

impl GrokImageGenerationBody {
    fn from_request(request: GrokImageGenerationRequest, default_model: String) -> Self {
        Self {
            model: request.model.unwrap_or(default_model),
            prompt: request.prompt,
            n: request.n,
            response_format: request.response_format,
            quality: request.quality,
            resolution: request.resolution,
            aspect_ratio: request.aspect_ratio,
            user: request.user,
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct GrokImageInputRef {
    url: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct GrokImageEditBody {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<GrokImageInputRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<GrokImageInputRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mask: Option<GrokImageInputRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<GrokImageResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

impl GrokImageEditBody {
    fn from_request(request: GrokImageEditRequest, default_model: String) -> Result<Self> {
        let has_single = request.image_url.is_some();
        let has_multiple = !request.image_urls.is_empty();
        if has_single == has_multiple {
            return Err(LlmError::Unsupported(
                "image edit requests must specify exactly one of image_url or image_urls",
            ));
        }

        Ok(Self {
            model: request.model.unwrap_or(default_model),
            prompt: request.prompt,
            image: request.image_url.map(|url| GrokImageInputRef {
                url,
                kind: "image_url",
            }),
            images: request
                .image_urls
                .into_iter()
                .map(|url| GrokImageInputRef {
                    url,
                    kind: "image_url",
                })
                .collect(),
            mask: request.mask_url.map(|url| GrokImageInputRef {
                url,
                kind: "image_url",
            }),
            n: request.n,
            response_format: request.response_format,
            quality: request.quality,
            resolution: request.resolution,
            aspect_ratio: request.aspect_ratio,
            user: request.user,
        })
    }
}

#[derive(Debug, serde::Serialize)]
#[serde(untagged)]
enum GrokInputItem {
    Message {
        role: &'static str,
        content: String,
    },
    FunctionCallOutput {
        #[serde(rename = "type")]
        kind: &'static str,
        call_id: String,
        output: String,
    },
}

impl GrokInputItem {
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

        Ok(Self::Message {
            role,
            content: message.content.clone(),
        })
    }

    fn from_tool_result(result: &crate::ToolResult) -> Self {
        Self::FunctionCallOutput {
            kind: "function_call_output",
            call_id: result.tool_call_id.clone(),
            output: result.content.clone(),
        }
    }
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GrokTool {
    Function {
        name: String,
        description: String,
        parameters: serde_json::Value,
    },
    WebSearch {
        #[serde(skip_serializing_if = "Vec::is_empty")]
        allowed_domains: Vec<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        excluded_domains: Vec<String>,
    },
    ImageGeneration,
}

impl From<&ToolDefinition> for GrokTool {
    fn from(tool: &ToolDefinition) -> Self {
        match tool {
            ToolDefinition::Function {
                name,
                description,
                parameters,
            } => Self::Function {
                name: name.clone(),
                description: description.clone(),
                parameters: parameters.clone(),
            },
            ToolDefinition::WebSearch {
                allowed_domains,
                excluded_domains,
            } => Self::WebSearch {
                allowed_domains: allowed_domains.clone(),
                excluded_domains: excluded_domains.clone(),
            },
            ToolDefinition::ImageGeneration => Self::ImageGeneration,
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct GrokResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    output: Vec<GrokOutputItem>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<GrokUsage>,
}

impl GrokResponse {
    fn output_text(&self) -> Option<String> {
        let mut text = String::new();
        for item in &self.output {
            if let GrokOutputItem::Message { content, .. } = item {
                for block in content {
                    if block.kind == "output_text" {
                        text.push_str(&block.text);
                    }
                }
            }
        }

        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type")]
enum GrokOutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(default)]
        content: Vec<GrokMessageContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(rename = "call_id")]
        _call_id: String,
        #[serde(rename = "name")]
        _name: String,
        #[serde(rename = "arguments")]
        _arguments: String,
    },
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        #[serde(flatten)]
        _extra: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, serde::Deserialize)]
struct GrokMessageContent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens_details: Option<GrokCompletionTokensDetails>,
    #[serde(default)]
    output_tokens_details: Option<GrokCompletionTokensDetails>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokCompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

impl From<GrokUsage> for Usage {
    fn from(usage: GrokUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens.or(usage.prompt_tokens).unwrap_or(0),
            output_tokens: usage.output_tokens.or(usage.completion_tokens).unwrap_or(0),
            reasoning_tokens: usage
                .output_tokens_details
                .or(usage.completion_tokens_details)
                .and_then(|details| details.reasoning_tokens),
        }
    }
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?;
    let value = value.to_str().ok()?;
    let seconds = value.trim().parse::<u64>().ok()?;
    Some(Duration::from_secs(seconds))
}

fn classify_error(status: u16, retry_after: Option<Duration>, body: &str) -> LlmError {
    let parsed = serde_json::from_str::<GrokErrorEnvelope>(body).ok();
    let message = parsed
        .as_ref()
        .map(|env| env.error.message.clone())
        .unwrap_or_else(|| body.to_owned());
    let kind = parsed
        .as_ref()
        .and_then(|env| env.error.kind.as_deref())
        .unwrap_or_default();

    match status {
        401 | 403 => LlmError::AuthenticationFailed,
        404 if message.contains("model") => LlmError::InvalidModel,
        429 => LlmError::RateLimited { retry_after },
        500 | 502 | 503 | 504 | 529 => LlmError::ServiceUnavailable,
        _ if kind == "invalid_request_error" && message.contains("model") => LlmError::InvalidModel,
        _ => LlmError::Upstream { status, message },
    }
}

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "max_output_tokens" | "max_tokens" => StopReason::MaxTokens,
        "function_call" | "tool_calls" | "tool_use" => StopReason::ToolUse,
        "content_filter" => StopReason::ContentFiltered,
        _ => StopReason::EndTurn,
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

struct GrokLlmStream {
    inner: UnboundedReceiverStream<Result<crate::StreamEvent>>,
    task: Option<JoinHandle<()>>,
}

impl futures_core::Stream for GrokLlmStream {
    type Item = Result<crate::StreamEvent>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for GrokLlmStream {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[cfg(test)]
struct GrokStreamTaskGuard(Arc<AtomicUsize>);

#[cfg(test)]
impl Drop for GrokStreamTaskGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

async fn stream_sse_response(
    client: AsyncClient,
    url: String,
    headers: HeaderMap,
    request: GrokRequest,
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

    let mut parser = GrokSseParser::default();
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
struct GrokSseParser {
    buffer: String,
    current_event: Option<String>,
    current_data: String,
}

impl GrokSseParser {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<crate::StreamEvent>> {
        self.buffer.push_str(
            std::str::from_utf8(bytes)
                .map_err(|_| LlmError::InvalidResponse("Grok SSE contained invalid UTF-8"))?,
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

        let payload: GrokStreamEvent = serde_json::from_str(&data)?;
        let mut out = Vec::new();
        match event_name.as_str() {
            "response.output_text.delta" => {
                if let Some(delta) = payload.delta {
                    out.push(crate::StreamEvent::TextDelta(delta));
                }
            }
            "response.output_item.added" | "response.output_item.done" => {
                if let Some(item) = payload.item {
                    out.extend(item.into_events()?);
                }
            }
            "response.completed" => {
                if let Some(response) = payload.response {
                    if let Some(id) = response.id {
                        out.push(crate::StreamEvent::Annotation {
                            kind: "response_id".to_owned(),
                            payload: serde_json::json!({ "id": id }),
                        });
                    }
                    if let Some(stop_reason) = response.stop_reason {
                        out.push(crate::StreamEvent::StopReason(map_stop_reason(
                            &stop_reason,
                        )));
                    }
                    if let Some(usage) = response.usage {
                        out.push(crate::StreamEvent::Usage(usage.into()));
                    }
                    out.push(crate::StreamEvent::Done);
                }
            }
            "error" => {
                if let Some(error) = payload.error {
                    return Err(classify_error(
                        500,
                        None,
                        &serde_json::to_string(&GrokErrorEnvelope {
                            error: GrokError {
                                message: error.message,
                                kind: Some(error.kind),
                            },
                        })?,
                    ));
                }
            }
            _ => {}
        }

        Ok(out)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokStreamEvent {
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    item: Option<GrokStreamItem>,
    #[serde(default)]
    response: Option<GrokStreamResponse>,
    #[serde(default)]
    error: Option<GrokStreamError>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokStreamResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<GrokUsage>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
enum GrokStreamItem {
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "message")]
    Message {
        #[serde(default)]
        content: Vec<GrokStreamContentBlock>,
    },
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        #[serde(flatten)]
        payload: serde_json::Value,
    },
    #[serde(rename = "x_search_call")]
    XSearchCall {
        #[serde(flatten)]
        payload: serde_json::Value,
    },
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall {
        #[serde(flatten)]
        payload: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

impl GrokStreamItem {
    fn into_events(self) -> Result<Vec<crate::StreamEvent>> {
        let mut out = Vec::new();
        match self {
            Self::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                out.push(crate::StreamEvent::ToolCallStart {
                    id: call_id.clone(),
                    name,
                });
                out.push(crate::StreamEvent::ToolCallDelta {
                    id: call_id.clone(),
                    arguments_delta: arguments,
                });
                out.push(crate::StreamEvent::ToolCallEnd { id: call_id });
            }
            Self::Message { content } => {
                for block in content {
                    if block.kind == "output_text" {
                        for annotation in block.annotations {
                            if annotation.kind == "url_citation" {
                                out.push(crate::StreamEvent::Annotation {
                                    kind: "citation".to_owned(),
                                    payload: serde_json::to_value(annotation)?,
                                });
                            }
                        }
                    }
                }
            }
            Self::WebSearchCall { payload } => {
                out.push(crate::StreamEvent::Annotation {
                    kind: "server_tool_call".to_owned(),
                    payload: server_tool_payload("web_search_call", payload),
                });
            }
            Self::XSearchCall { payload } => {
                out.push(crate::StreamEvent::Annotation {
                    kind: "server_tool_call".to_owned(),
                    payload: server_tool_payload("x_search_call", payload),
                });
            }
            Self::CodeInterpreterCall { payload } => {
                out.push(crate::StreamEvent::Annotation {
                    kind: "server_tool_call".to_owned(),
                    payload: server_tool_payload("code_interpreter_call", payload),
                });
            }
            Self::Other => {}
        }
        Ok(out)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokStreamContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    annotations: Vec<GrokCitationAnnotation>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokCitationAnnotation {
    #[serde(rename = "type")]
    kind: String,
    url: String,
    start_index: u64,
    end_index: u64,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokStreamError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

fn server_tool_payload(kind: &str, payload: serde_json::Value) -> serde_json::Value {
    match payload {
        serde_json::Value::Object(mut map) => {
            map.insert(
                "type".to_owned(),
                serde_json::Value::String(kind.to_owned()),
            );
            serde_json::Value::Object(map)
        }
        other => serde_json::json!({
            "type": kind,
            "payload": other,
        }),
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokErrorEnvelope {
    error: GrokError,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GrokError {
    message: String,
    #[serde(default, rename = "type")]
    kind: Option<String>,
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
    fn parses_text_and_reasoning_usage_from_response() {
        let parsed: GrokResponse = serde_json::from_str(
            r#"{
                "output": [
                    {
                        "type": "message",
                        "content": [
                            { "type": "output_text", "text": "Hello " },
                            { "type": "output_text", "text": "world" }
                        ]
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "ignored_for_complete",
                        "arguments": "{\"city\":\"London\"}"
                    }
                ],
                "stop_reason": "stop",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "output_tokens_details": {
                        "reasoning_tokens": 5
                    }
                }
            }"#,
        )
        .expect("parse response");

        assert_eq!(parsed.output_text().as_deref(), Some("Hello world"));
        let usage: Usage = parsed.usage.expect("usage").into();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.reasoning_tokens, Some(5));
    }

    #[test]
    fn jitter_seed_is_respected_for_deterministic_tests() {
        let options = GrokBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            base_delay: Duration::ZERO,
            base_delay_500: Duration::ZERO,
            max_jitter: Duration::from_secs(1),
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend = GrokBackend::new_with_options("m".to_owned(), "k".to_owned(), options)
            .expect("backend");

        let a = backend.jitter_for_attempt(1);
        let b = backend.jitter_for_attempt(1);
        assert_eq!(a, b);
        assert_ne!(backend.jitter_for_attempt(1), backend.jitter_for_attempt(2));
    }

    #[test]
    fn backoff_uses_longer_baseline_for_500() {
        let options = GrokBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            base_delay: Duration::from_secs(1),
            base_delay_500: Duration::from_secs(3),
            max_jitter: Duration::ZERO,
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend = GrokBackend::new_with_options("m".to_owned(), "k".to_owned(), options)
            .expect("backend");

        assert_eq!(backend.backoff_for(503, 1), Duration::from_secs(1));
        assert_eq!(backend.backoff_for(500, 1), Duration::from_secs(3));
        assert_eq!(backend.backoff_for(500, 2), Duration::from_secs(6));
    }

    #[test]
    fn baseline_for_500_is_never_below_normal_baseline() {
        let options = GrokBackendOptions {
            base_url: "http://example.invalid".to_owned(),
            base_delay: Duration::from_secs(5),
            base_delay_500: Duration::from_secs(3),
            max_jitter: Duration::ZERO,
            jitter_seed: Some(123),
            pool_max_idle_per_host: 0,
            ..Default::default()
        };
        let backend = GrokBackend::new_with_options("m".to_owned(), "k".to_owned(), options)
            .expect("backend");

        assert_eq!(backend.base_delay_for_status(500), Duration::from_secs(5));
    }

    #[test]
    fn serializes_responses_api_request_shape() {
        let server = MockServer::start(vec![MockResponse::json(
            200,
            r#"{
                "output": [
                    {
                        "type": "message",
                        "content": [{ "type": "output_text", "text": "ok" }]
                    }
                ],
                "stop_reason": "stop"
            }"#,
        )]);

        let backend = backend_for_server(server.base_url());
        let request = CompletionRequest {
            system_prompt: Some("You are helpful".to_owned()),
            previous_response_id: Some("resp-previous".to_owned()),
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
            tool_results: vec![crate::ToolResult::new(
                "call-1",
                r#"{"city":"London","forecast":"rain"}"#,
            )],
            max_tokens: Some(32),
            thinking_budget_tokens: None,
            temperature: Some(0.25),
            parallel_tool_calls: Some(true),
        };

        let response = backend.complete(request).expect("complete");
        assert_eq!(response.text, "ok");

        let requests = server.take_requests();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["model"], "grok-4.20-beta-latest-reasoning");
        assert_eq!(body["instructions"], "You are helpful");
        assert_eq!(body["previous_response_id"], "resp-previous");
        assert_eq!(body["max_output_tokens"], 32);
        assert_eq!(body["temperature"], serde_json::json!(0.25));
        assert_eq!(body["parallel_tool_calls"], serde_json::json!(true));
        assert!(body.get("include").is_none());
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"], "Find current weather");
        assert_eq!(body["input"][1]["role"], "assistant");
        assert_eq!(body["input"][2]["type"], "function_call_output");
        assert_eq!(body["input"][2]["call_id"], "call-1");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "get_weather");
    }

    #[test]
    fn serializes_builtin_xai_tools_into_request_shape() {
        let server = MockServer::start(vec![MockResponse::json(
            200,
            r#"{
                "output": [
                    {
                        "type": "message",
                        "content": [{ "type": "output_text", "text": "ok" }]
                    }
                ],
                "stop_reason": "stop"
            }"#,
        )]);

        let backend = backend_for_server(server.base_url());
        let mut request = CompletionRequest::single_user_message("Research London weather", 32);
        request.tools = vec![
            ToolDefinition::web_search(
                vec!["example.com".to_owned()],
                vec!["blocked.example".to_owned()],
            ),
            ToolDefinition::image_generation(),
        ];

        let response = backend.complete(request).expect("complete");
        assert_eq!(response.text, "ok");

        let requests = server.take_requests();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert_eq!(
            body["tools"][0]["allowed_domains"][0],
            serde_json::Value::String("example.com".to_owned())
        );
        assert_eq!(
            body["tools"][0]["excluded_domains"][0],
            serde_json::Value::String("blocked.example".to_owned())
        );
        assert_eq!(body["tools"][1]["type"], "image_generation");
    }

    #[test]
    fn serializes_image_generation_request_shape() {
        let server = MockServer::start(vec![MockResponse::json(
            200,
            r#"{
                "data": [
                    {
                        "url": "https://example.com/logo.png",
                        "mime_type": "image/png",
                        "revised_prompt": ""
                    }
                ],
                "usage": { "cost_in_usd_ticks": 42 }
            }"#,
        )]);

        let backend = backend_for_server(server.base_url());
        let response = backend
            .generate_image(GrokImageGenerationRequest {
                prompt: "Create a geometric logo".to_owned(),
                model: Some("grok-imagine-image".to_owned()),
                n: Some(2),
                response_format: Some(GrokImageResponseFormat::Url),
                quality: Some("high".to_owned()),
                resolution: Some("2k".to_owned()),
                aspect_ratio: Some("1:1".to_owned()),
                user: Some("user-123".to_owned()),
            })
            .expect("generate image");

        assert_eq!(response.data.len(), 1);
        assert_eq!(
            response.data[0].url.as_deref(),
            Some("https://example.com/logo.png")
        );
        assert_eq!(
            response.usage.and_then(|usage| usage.cost_in_usd_ticks),
            Some(42)
        );

        let requests = server.take_requests();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["model"], "grok-imagine-image");
        assert_eq!(body["prompt"], "Create a geometric logo");
        assert_eq!(body["n"], 2);
        assert_eq!(body["response_format"], "url");
        assert_eq!(body["quality"], "high");
        assert_eq!(body["resolution"], "2k");
        assert_eq!(body["aspect_ratio"], "1:1");
        assert_eq!(body["user"], "user-123");
    }

    #[test]
    fn serializes_image_edit_request_shape() {
        let server = MockServer::start(vec![MockResponse::json(
            200,
            r#"{
                "data": [
                    {
                        "b64_json": "abcd",
                        "mime_type": "image/png",
                        "revised_prompt": ""
                    }
                ]
            }"#,
        )]);

        let backend = backend_for_server(server.base_url());
        let response = backend
            .edit_image(GrokImageEditRequest {
                prompt: "Make the blue darker".to_owned(),
                image_url: Some("https://example.com/original.png".to_owned()),
                image_urls: Vec::new(),
                mask_url: Some("https://example.com/mask.png".to_owned()),
                model: Some("grok-imagine-image".to_owned()),
                n: Some(1),
                response_format: Some(GrokImageResponseFormat::B64Json),
                quality: Some("medium".to_owned()),
                resolution: Some("1k".to_owned()),
                aspect_ratio: Some("1:1".to_owned()),
                user: None,
            })
            .expect("edit image");

        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("abcd"));

        let requests = server.take_requests();
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["prompt"], "Make the blue darker");
        assert_eq!(body["image"]["url"], "https://example.com/original.png");
        assert_eq!(body["image"]["type"], "image_url");
        assert_eq!(body["mask"]["url"], "https://example.com/mask.png");
        assert_eq!(body["response_format"], "b64_json");
    }

    #[test]
    fn rejects_ambiguous_image_edit_sources() {
        let backend = backend_for_server("http://example.invalid".to_owned());
        let error = backend
            .edit_image(GrokImageEditRequest {
                prompt: "Edit this".to_owned(),
                image_url: Some("https://example.com/one.png".to_owned()),
                image_urls: vec!["https://example.com/two.png".to_owned()],
                mask_url: None,
                model: None,
                n: None,
                response_format: None,
                quality: None,
                resolution: None,
                aspect_ratio: None,
                user: None,
            })
            .expect_err("should reject ambiguous edit source");

        assert!(matches!(error, LlmError::Unsupported(_)));
    }

    #[test]
    fn parses_plural_image_artifacts_from_response() {
        let parsed: GrokImageGenerationResponse = serde_json::from_str(
            r#"{
                "data": [
                    {
                        "url": "https://example.com/one.png",
                        "mime_type": "image/png",
                        "revised_prompt": ""
                    },
                    {
                        "b64_json": "xyz",
                        "mime_type": "image/webp",
                        "revised_prompt": ""
                    }
                ],
                "usage": { "cost_in_usd_ticks": 99 }
            }"#,
        )
        .expect("parse image response");

        assert_eq!(parsed.data.len(), 2);
        assert_eq!(
            parsed.data[0].url.as_deref(),
            Some("https://example.com/one.png")
        );
        assert_eq!(parsed.data[1].b64_json.as_deref(), Some("xyz"));
        assert_eq!(
            parsed.usage.and_then(|usage| usage.cost_in_usd_ticks),
            Some(99)
        );
    }

    #[test]
    fn maps_grok_errors_to_typed_errors() {
        let auth = classify_error(
            401,
            None,
            r#"{ "error": { "type": "authentication_error", "message": "bad key" } }"#,
        );
        assert!(matches!(auth, LlmError::AuthenticationFailed));

        let limited = classify_error(
            429,
            Some(Duration::from_secs(7)),
            r#"{ "error": { "type": "rate_limit_error", "message": "slow down" } }"#,
        );
        assert!(matches!(
            limited,
            LlmError::RateLimited {
                retry_after: Some(duration)
            } if duration == Duration::from_secs(7)
        ));

        let invalid_model = classify_error(
            400,
            None,
            r#"{ "error": { "type": "invalid_request_error", "message": "model does not exist" } }"#,
        );
        assert!(matches!(invalid_model, LlmError::InvalidModel));
    }

    #[test]
    fn capabilities_advertise_grok_specific_features() {
        let backend = GrokBackend::new(
            "grok-4.20-beta-latest-reasoning".to_owned(),
            "test-key".to_owned(),
        )
        .expect("backend");

        let capabilities = backend.capabilities();
        assert!(capabilities.supports_streaming);
        assert!(capabilities.supports_tool_use);
        assert!(capabilities.supports_parallel_tool_calls);
        assert!(capabilities.supports_web_search);
        assert!(capabilities.supports_image_generation);
        assert_eq!(capabilities.max_context_tokens, 2_000_000);
    }

    #[test]
    fn sse_parser_normalizes_tool_calls_citations_and_server_tools() {
        let mut parser = GrokSseParser::default();
        let events = parser
            .push(
                concat!(
                    "event: response.output_text.delta\n",
                    "data: {\"delta\":\"Hello [[1]]\"}\n\n",
                    "event: response.output_item.added\n",
                    "data: {\"item\":{\"type\":\"function_call\",\"call_id\":\"call-1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"London\\\"}\"}}\n\n",
                    "event: response.output_item.done\n",
                    "data: {\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://example.com/weather\",\"start_index\":6,\"end_index\":11,\"title\":\"Weather\"}]}]}}\n\n",
                    "event: response.output_item.added\n",
                    "data: {\"item\":{\"type\":\"web_search_call\",\"id\":\"search-1\",\"query\":\"weather london\"}}\n\n",
                    "event: response.completed\n",
                    "data: {\"response\":{\"id\":\"resp-1\",\"stop_reason\":\"function_call\",\"usage\":{\"input_tokens\":11,\"output_tokens\":7,\"completion_tokens_details\":{\"reasoning_tokens\":3}}}}\n\n"
                )
                .as_bytes(),
            )
            .expect("parse sse");

        assert_eq!(
            events,
            vec![
                crate::StreamEvent::TextDelta("Hello [[1]]".to_owned()),
                crate::StreamEvent::ToolCallStart {
                    id: "call-1".to_owned(),
                    name: "get_weather".to_owned()
                },
                crate::StreamEvent::ToolCallDelta {
                    id: "call-1".to_owned(),
                    arguments_delta: "{\"city\":\"London\"}".to_owned()
                },
                crate::StreamEvent::ToolCallEnd {
                    id: "call-1".to_owned()
                },
                crate::StreamEvent::Annotation {
                    kind: "citation".to_owned(),
                    payload: serde_json::json!({
                        "type": "url_citation",
                        "url": "https://example.com/weather",
                        "start_index": 6,
                        "end_index": 11,
                        "title": "Weather"
                    })
                },
                crate::StreamEvent::Annotation {
                    kind: "server_tool_call".to_owned(),
                    payload: serde_json::json!({
                        "type": "web_search_call",
                        "id": "search-1",
                        "query": "weather london"
                    })
                },
                crate::StreamEvent::Annotation {
                    kind: "response_id".to_owned(),
                    payload: serde_json::json!({
                        "id": "resp-1"
                    })
                },
                crate::StreamEvent::StopReason(StopReason::ToolUse),
                crate::StreamEvent::Usage(Usage {
                    input_tokens: 11,
                    output_tokens: 7,
                    reasoning_tokens: Some(3)
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
            extra_headers: vec![("Cache-Control".to_owned(), "no-cache".to_owned())],
            content_length_override: Some(10_000),
            linger_after_body: Some(Duration::from_millis(250)),
            body: concat!(
                "event: response.output_text.delta\n",
                "data: {\"delta\":\"Hello\"}\n\n"
            )
            .to_owned(),
        }]);

        let backend = backend_for_server(server.base_url());
        assert_eq!(backend.active_stream_count(), 0);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let mut stream = backend
                .stream(CompletionRequest::single_user_message("hello", 16))
                .expect("stream");
            assert_eq!(backend.active_stream_count(), 1);

            let first = stream.next().await.expect("first event").expect("ok event");
            assert_eq!(first, crate::StreamEvent::TextDelta("Hello".to_owned()));

            drop(stream);
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert_eq!(backend.active_stream_count(), 0);
        });
        let requests = server.take_requests();
        let body: serde_json::Value =
            serde_json::from_str(&requests[0].body).expect("request json");
        assert_eq!(body["stream"], serde_json::json!(true));
        assert!(body.get("include").is_none());
    }

    fn backend_for_server(base_url: String) -> GrokBackend {
        GrokBackend::new_with_options(
            "grok-4.20-beta-latest-reasoning".to_owned(),
            "test-key".to_owned(),
            GrokBackendOptions {
                base_url,
                max_attempts: 1,
                base_delay: Duration::ZERO,
                base_delay_500: Duration::ZERO,
                max_jitter: Duration::ZERO,
                jitter_seed: Some(1),
                connect_timeout: Duration::from_secs(2),
                timeout: Duration::from_secs(2),
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
                    captured_clone.lock().expect("captured").push(request);
                    write_http_response(&mut stream, response).expect("response");
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
            std::mem::take(&mut *self.captured.lock().expect("captured mutex"))
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

    fn find_header_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|window| window == b"\r\n\r\n")
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
        let head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n",
            response.status,
            reason_phrase(response.status),
            response.content_type,
            response.content_length_override.unwrap_or(body_bytes.len()),
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
}
