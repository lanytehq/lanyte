use lanyte_llm::{
    ClaudeBackend, CompletionRequest, LlmBackend, StopReason, StreamEvent, ToolDefinition,
    ToolResult,
};

fn live_backend() -> Option<ClaudeBackend> {
    let api_key = std::env::var("LANYTE_LLM_CLAUDE_API_KEY").ok()?;
    let model =
        std::env::var("LANYTE_LLM_CLAUDE_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6".to_owned());
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
        eprintln!("skipping Claude live test: LANYTE_LLM_CLAUDE_API_KEY not set");
        return;
    };

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let (tool_call_id, tool_name, tool_args) = runtime.block_on(async {
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
    assert!(!tool_args.trim().is_empty());
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&tool_args) {
        if let Some(city) = parsed.get("city").and_then(serde_json::Value::as_str) {
            assert_eq!(city, "London");
        }
    }

    let mut follow_up = CompletionRequest::single_user_message(
        "Use the tool result to answer in one short sentence.",
        64,
    );
    follow_up.messages.push(lanyte_llm::Message::assistant(
        "Calling get_weather for London.",
    ));
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
