use lanyte_llm::{
    CompletionRequest, LlmBackend, LlmError, OpenAiBackend, StopReason, StreamEvent,
    ToolDefinition, ToolResult,
};

const OPENAI_API_KEY_ENV: &str = "LANYTE_LLM_OPENAI_API_KEY";
const OPENAI_BASE_URL_ENV: &str = "LANYTE_LLM_OPENAI_BASE_URL";
const OPENAI_MODEL_ENV: &str = "LANYTE_LLM_OPENAI_MODEL";

fn configured_base_url() -> String {
    std::env::var(OPENAI_BASE_URL_ENV).unwrap_or_else(|_| "https://api.openai.com/v1".to_owned())
}

fn is_local_base_url(base_url: &str) -> bool {
    base_url.starts_with("http://127.0.0.1") || base_url.starts_with("http://localhost")
}

fn live_backend() -> Option<OpenAiBackend> {
    let base_url = configured_base_url();
    let model = std::env::var(OPENAI_MODEL_ENV).unwrap_or_else(|_| "gpt-5.4".to_owned());
    let api_key = match std::env::var(OPENAI_API_KEY_ENV) {
        Ok(value) => value,
        Err(_) if is_local_base_url(&base_url) => "local-openai-compatible".to_owned(),
        Err(_) => return None,
    };

    OpenAiBackend::new_with_base_url(model, api_key, base_url).ok()
}

// Structured-output live coverage stays deferred until the adapter supports
// response_format: { type: "json_schema" } on the shared request surface.

#[test]
fn live_complete_round_trip() {
    let Some(backend) = live_backend() else {
        eprintln!(
            "skipping OpenAI live test: LANYTE_LLM_OPENAI_API_KEY not set and no local base_url override"
        );
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
}

#[test]
fn live_stream_round_trip() {
    let Some(backend) = live_backend() else {
        eprintln!(
            "skipping OpenAI live test: LANYTE_LLM_OPENAI_API_KEY not set and no local base_url override"
        );
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
        use futures_util::StreamExt;
        while let Some(event) = stream.next().await {
            match event.expect("stream event") {
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::Done => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }

        assert_eq!(text.trim(), "stream_ok");
        assert!(saw_done);
    });
}

#[test]
fn live_tool_round_trip() {
    let Some(backend) = live_backend() else {
        eprintln!(
            "skipping OpenAI live test: LANYTE_LLM_OPENAI_API_KEY not set and no local base_url override"
        );
        return;
    };

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (tool_call_id, tool_name, tool_args) = runtime.block_on(async {
        let mut request = CompletionRequest::single_user_message(
            "Use the get_weather tool for London. Do not answer before the tool result arrives.",
            64,
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
                StreamEvent::Done => break,
                _ => {}
            }
        }

        (
            tool_call_id.expect("tool call id"),
            tool_name.expect("tool name"),
            tool_args,
        )
    });

    assert_eq!(tool_name, "get_weather");
    assert!(tool_args.contains("London"));

    let mut follow_up = CompletionRequest::single_user_message(
        "Use the tool result to answer in one short sentence.",
        64,
    );
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
fn live_error_invalid_model() {
    let base_url = configured_base_url();
    if is_local_base_url(&base_url) {
        eprintln!(
            "skipping OpenAI live invalid-model test: local base URL may not preserve OpenAI error taxonomy"
        );
        return;
    }

    let Ok(api_key) = std::env::var(OPENAI_API_KEY_ENV) else {
        eprintln!("skipping OpenAI live invalid-model test: LANYTE_LLM_OPENAI_API_KEY not set");
        return;
    };

    let backend = OpenAiBackend::new_with_base_url(
        "gpt-invalid-model-for-live-test".to_owned(),
        api_key,
        base_url,
    )
    .expect("backend init");
    let err = backend
        .complete(CompletionRequest::single_user_message(
            "Reply with exactly: live_ok",
            16,
        ))
        .expect_err("invalid model should fail");
    assert!(
        matches!(err, LlmError::InvalidModel),
        "unexpected error: {err:?}"
    );
}
