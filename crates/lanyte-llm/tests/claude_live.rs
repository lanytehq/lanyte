use lanyte_llm::{
    ClaudeBackend, CompletionRequest, LlmBackend, StopReason, StreamEvent, ToolDefinition,
    ToolResult,
};

const CLAUDE_API_KEY_ENV: &str = "LANYTE_LLM_CLAUDE_API_KEY";
const CLAUDE_MODEL_ENV: &str = "LANYTE_LLM_CLAUDE_MODEL";
const CLAUDE_THINKING_MODEL_ENV: &str = "LANYTE_LLM_CLAUDE_THINKING_MODEL";

fn live_backend() -> Option<ClaudeBackend> {
    let api_key = std::env::var(CLAUDE_API_KEY_ENV).ok()?;
    let model = std::env::var(CLAUDE_MODEL_ENV).unwrap_or_else(|_| "claude-sonnet-4-6".to_owned());
    ClaudeBackend::new(model, api_key).ok()
}

fn live_thinking_backend() -> Option<ClaudeBackend> {
    let api_key = std::env::var(CLAUDE_API_KEY_ENV).ok()?;
    let model = std::env::var(CLAUDE_THINKING_MODEL_ENV).ok()?;
    ClaudeBackend::new(model, api_key).ok()
}

#[test]
fn live_complete_round_trip() {
    let Some(backend) = live_backend() else {
        eprintln!("skipping Claude live test: LANYTE_LLM_CLAUDE_API_KEY not set");
        return;
    };

    let response = backend
        .complete(CompletionRequest::single_user_message(
            "Reply with exactly: live_ok",
            16,
        ))
        .expect("live completion");

    assert_eq!(response.text.trim(), "live_ok");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    let usage = response
        .usage
        .expect("Claude live completion should include usage");
    assert!(usage.input_tokens > 0);
    assert!(usage.output_tokens > 0);
}

#[test]
fn live_stream_round_trip() {
    let Some(backend) = live_backend() else {
        eprintln!("skipping Claude live test: LANYTE_LLM_CLAUDE_API_KEY not set");
        return;
    };

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let mut stream = backend
            .stream(CompletionRequest::single_user_message(
                "Reply with exactly: stream_ok",
                16,
            ))
            .expect("live stream");

        let mut text = String::new();
        let mut saw_done = false;
        let mut saw_usage = false;

        use futures_util::StreamExt;
        while let Some(event) = stream.next().await {
            match event.expect("stream event") {
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::Usage(usage) => {
                    saw_usage = usage.input_tokens > 0 && usage.output_tokens > 0;
                }
                StreamEvent::Done => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }

        assert_eq!(text.trim(), "stream_ok");
        assert!(saw_done);
        assert!(saw_usage);
    });
}

#[test]
fn live_tool_round_trip() {
    let Some(backend) = live_backend() else {
        eprintln!("skipping Claude live test: LANYTE_LLM_CLAUDE_API_KEY not set");
        return;
    };

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (tool_call_id, tool_name, tool_args, saw_tool_end) = runtime.block_on(async {
        let mut request = CompletionRequest::single_user_message(
            "Call the get_weather tool with city set to London. Do not emit any text before the tool call.",
            256,
        );
        request.tools.push(ToolDefinition::function(
            "get_weather",
            "Return a fixed JSON weather payload for a city.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        ));

        let mut stream = backend.stream(request).expect("live tool stream");
        let mut tool_call_id = None;
        let mut tool_name = None;
        let mut tool_args = String::new();
        let mut saw_tool_end = false;

        use futures_util::StreamExt;
        while let Some(event) = stream.next().await {
            match event.expect("stream event") {
                StreamEvent::ToolCallStart { id, name } => {
                    tool_call_id = Some(id);
                    tool_name = Some(name);
                }
                StreamEvent::ToolCallDelta {
                    arguments_delta, ..
                } => tool_args.push_str(&arguments_delta),
                StreamEvent::ToolCallEnd { .. } => saw_tool_end = true,
                StreamEvent::Done => break,
                _ => {}
            }
        }

        (
            tool_call_id.expect("tool call id"),
            tool_name.expect("tool name"),
            tool_args,
            saw_tool_end,
        )
    });

    assert_eq!(tool_name, "get_weather");
    assert!(saw_tool_end);
    assert!(!tool_args.trim().is_empty());
    let parsed: serde_json::Value = serde_json::from_str(&tool_args).expect("tool args json");
    assert_eq!(
        parsed.get("city").and_then(serde_json::Value::as_str),
        Some("London")
    );

    let mut follow_up = CompletionRequest::single_user_message("", 64);
    follow_up.messages.clear();
    follow_up.tool_results.push(ToolResult::with_call(
        tool_call_id,
        tool_name,
        tool_args,
        r#"{"city":"London","forecast":"rain","temp_c":12}"#,
    ));

    let response = backend.complete(follow_up).expect("follow-up completion");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    let text = response.text.to_ascii_lowercase();
    assert!(text.contains("london"));
    assert!(text.contains("rain"));
}

#[test]
fn live_thinking_round_trip() {
    let Some(backend) = live_thinking_backend() else {
        eprintln!("skipping Claude thinking live test: LANYTE_LLM_CLAUDE_THINKING_MODEL not set");
        return;
    };

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let mut request = CompletionRequest::single_user_message(
            "What is 27 * 453? After thinking, answer with exactly: 12231",
            4096,
        );
        request.thinking_budget_tokens = Some(2048);

        let mut stream = backend.stream(request).expect("live thinking stream");
        let mut text = String::new();
        let mut saw_done = false;
        let mut saw_thinking = false;
        let mut saw_usage = false;

        use futures_util::StreamExt;
        while let Some(event) = stream.next().await {
            match event.expect("stream event") {
                StreamEvent::ThinkingDelta(delta) => {
                    if !delta.trim().is_empty() {
                        saw_thinking = true;
                    }
                }
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::Usage(usage) => {
                    saw_usage = usage.input_tokens > 0 && usage.output_tokens > 0;
                }
                StreamEvent::Done => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }

        assert!(saw_thinking);
        assert!(text.contains("12231"));
        assert!(saw_usage);
        assert!(saw_done);
    });
}
