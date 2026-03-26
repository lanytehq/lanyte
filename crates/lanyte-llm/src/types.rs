use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CompletionRequest {
    pub system_prompt: Option<String>,
    pub previous_response_id: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub tool_results: Vec<ToolResult>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub parallel_tool_calls: Option<bool>,
}

impl CompletionRequest {
    #[must_use]
    pub fn single_user_message(content: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            system_prompt: None,
            previous_response_id: None,
            messages: vec![Message::user(content)],
            tools: Vec::new(),
            tool_results: Vec::new(),
            max_tokens: Some(max_tokens),
            temperature: None,
            parallel_tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

impl Message {
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    Function {
        name: String,
        description: String,
        parameters: serde_json::Value,
    },
    WebSearch {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_domains: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        excluded_domains: Vec<String>,
    },
    ImageGeneration,
}

impl ToolDefinition {
    #[must_use]
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self::Function {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    #[must_use]
    pub fn web_search(allowed_domains: Vec<String>, excluded_domains: Vec<String>) -> Self {
        Self::WebSearch {
            allowed_domains,
            excluded_domains,
        }
    }

    #[must_use]
    pub fn image_generation() -> Self {
        Self::ImageGeneration
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    pub content: String,
}

impl ToolResult {
    #[must_use]
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: None,
            arguments: None,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn with_call(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: Some(tool_name.into()),
            arguments: Some(arguments.into()),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub response_id: Option<String>,
    pub text: String,
    pub stop_reason: StopReason,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub supports_streaming: bool,
    pub supports_tool_use: bool,
    pub supports_system_prompt: bool,
    pub supports_parallel_tool_calls: bool,
    pub supports_web_search: bool,
    pub supports_image_generation: bool,
    pub max_context_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Unconfigured,
    Degraded(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamEvent {
    TextDelta(String),
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallDelta {
        id: String,
        arguments_delta: String,
    },
    ToolCallEnd {
        id: String,
    },
    ThinkingDelta(String),
    Usage(Usage),
    StopReason(StopReason),
    // Reserved for providers that surface structured metadata such as citations.
    Annotation {
        kind: String,
        payload: serde_json::Value,
    },
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    ContentFiltered,
}
