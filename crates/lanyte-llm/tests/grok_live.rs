use lanyte_llm::{
    CompletionRequest, GrokBackend, LlmBackend, LlmError, StopReason, StreamEvent, ToolDefinition,
    ToolResult,
};

const GROK_API_KEY_ENV: &str = "LANYTE_LLM_GROK_API_KEY";
const GROK_MODEL_ENV: &str = "LANYTE_LLM_GROK_MODEL";

fn live_backend() -> Option<GrokBackend> {
    let api_key = std::env::var(GROK_API_KEY_ENV).ok()?;
    let model = std::env::var(GROK_MODEL_ENV)
        .unwrap_or_else(|_| "grok-4.20-beta-latest-reasoning".to_owned());
    GrokBackend::new(model, api_key).ok()
}

// Rate-limit validation stays manual-only. Intentionally triggering provider limits would
// create account-scoped side effects; the conformance harness already covers retry parsing.

#[test]
fn live_complete_round_trip() {
    let Some(backend) = live_backend() else {
        eprintln!("skipping Grok live test: LANYTE_LLM_GROK_API_KEY not set");
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
    assert!(response.response_id.is_some());
}

#[test]
fn live_stream_round_trip() {
    let Some(backend) = live_backend() else {
        eprintln!("skipping Grok live test: LANYTE_LLM_GROK_API_KEY not set");
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
        let mut saw_response_id = false;

        use futures_util::StreamExt;
        while let Some(event) = stream.next().await {
            match event.expect("stream event") {
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::Annotation { kind, .. } if kind == "response_id" => {
                    saw_response_id = true;
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
        assert!(saw_response_id);
    });
}

#[test]
fn live_tool_round_trip_with_previous_response_id() {
    let Some(backend) = live_backend() else {
        eprintln!("skipping Grok live test: LANYTE_LLM_GROK_API_KEY not set");
        return;
    };

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let mut round_trip = None;
    // Grok tool choice is still provider behavior, so retry a few times to reduce flake.
    // Once credentials are present, though, AGI-013 treats failure to emit the tool call as a
    // real validation failure rather than a soft skip.
    for _ in 0..3 {
        let attempt = runtime.block_on(async {
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
            let mut previous_response_id = None;

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
                    StreamEvent::Annotation { kind, payload } if kind == "response_id" => {
                        previous_response_id = payload
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .map(ToOwned::to_owned);
                    }
                    StreamEvent::Done => break,
                    _ => {}
                }
            }

            match (tool_call_id, tool_name, previous_response_id) {
                (Some(tool_call_id), Some(tool_name), Some(previous_response_id)) => Some((
                    tool_call_id,
                    tool_name,
                    tool_args,
                    previous_response_id,
                )),
                _ => None,
            }
        });

        if attempt.is_some() {
            round_trip = attempt;
            break;
        }
    }

    let (tool_call_id, tool_name, tool_args, previous_response_id) =
        round_trip.expect("Grok should emit a tool call after 3 live attempts");

    assert_eq!(tool_name, "get_weather");
    assert!(tool_args.contains("London"));

    let mut follow_up = CompletionRequest::single_user_message("", 64);
    follow_up.messages.clear();
    follow_up.previous_response_id = Some(previous_response_id);
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
    assert!(response.response_id.is_some());
}

#[test]
fn live_error_invalid_model() {
    let Ok(api_key) = std::env::var(GROK_API_KEY_ENV) else {
        eprintln!("skipping Grok live invalid-model test: LANYTE_LLM_GROK_API_KEY not set");
        return;
    };

    let backend = GrokBackend::new("grok-invalid-model-for-live-test".to_owned(), api_key)
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

#[test]
fn live_multi_turn() {
    let Some(backend) = live_backend() else {
        eprintln!("skipping Grok live test: LANYTE_LLM_GROK_API_KEY not set");
        return;
    };

    let first = backend
        .complete(CompletionRequest::single_user_message(
            "Remember this exact code word for the next turn: cerulean-fox. Reply with exactly: remembered",
            32,
        ))
        .expect("first completion");
    assert_eq!(first.text.trim(), "remembered");

    let previous_response_id = first.response_id.expect("previous response id");
    let mut follow_up = CompletionRequest::single_user_message(
        "What code word did I ask you to remember? Reply with exactly the code word.",
        32,
    );
    follow_up.previous_response_id = Some(previous_response_id);

    let second = backend.complete(follow_up).expect("follow-up completion");
    assert_eq!(second.stop_reason, StopReason::EndTurn);
    assert!(second.text.to_ascii_lowercase().contains("cerulean-fox"));
}
