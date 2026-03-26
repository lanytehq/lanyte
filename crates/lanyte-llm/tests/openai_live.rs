use lanyte_llm::{
    CompletionRequest, LlmBackend, OpenAiBackend, StopReason, StreamEvent, ToolDefinition,
    ToolResult,
};

fn live_backend() -> Option<OpenAiBackend> {
    let base_url = std::env::var("LANYTE_LLM_OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
    let model = std::env::var("LANYTE_LLM_OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.4".to_owned());
    let api_key = match std::env::var("LANYTE_LLM_OPENAI_API_KEY") {
        Ok(value) => value,
        Err(_)
            if base_url.starts_with("http://127.0.0.1")
                || base_url.starts_with("http://localhost") =>
        {
            "local-openai-compatible".to_owned()
        }
        Err(_) => return None,
    };

    OpenAiBackend::new_with_base_url(model, api_key, base_url).ok()
}

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
