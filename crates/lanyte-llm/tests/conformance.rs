use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt;
use lanyte_llm::{
    BackendCapabilities, CompletionRequest, CompletionResponse, HealthStatus, LlmBackend, LlmError,
    LlmStream, Message, Result, StopReason, StreamEvent, ToolDefinition, ToolResult, Usage,
};

type CompleteHandler =
    dyn Fn(CompletionRequest) -> Result<CompletionResponse> + Send + Sync + 'static;
type StreamHandler =
    dyn Fn(CompletionRequest, Arc<AtomicUsize>) -> Result<LlmStream> + Send + Sync + 'static;

struct TestBackend {
    active_streams: Arc<AtomicUsize>,
    complete_handler: Arc<CompleteHandler>,
    stream_handler: Arc<StreamHandler>,
}

impl TestBackend {
    fn new(
        complete_handler: impl Fn(CompletionRequest) -> Result<CompletionResponse>
            + Send
            + Sync
            + 'static,
        stream_handler: impl Fn(CompletionRequest, Arc<AtomicUsize>) -> Result<LlmStream>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            active_streams: Arc::new(AtomicUsize::new(0)),
            complete_handler: Arc::new(complete_handler),
            stream_handler: Arc::new(stream_handler),
        }
    }

    fn scripted_stream(
        active_streams: Arc<AtomicUsize>,
        events: Vec<Result<StreamEvent>>,
    ) -> LlmStream {
        active_streams.fetch_add(1, Ordering::SeqCst);
        Box::pin(ScriptedStream {
            items: events.into(),
            active_streams,
        })
    }
}

impl LlmBackend for TestBackend {
    fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        (self.complete_handler)(request)
    }

    fn stream(&self, request: CompletionRequest) -> Result<LlmStream> {
        (self.stream_handler)(request, Arc::clone(&self.active_streams))
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_streaming: true,
            supports_tool_use: true,
            supports_system_prompt: true,
            supports_parallel_tool_calls: true,
            supports_web_search: false,
            supports_image_generation: false,
            max_context_tokens: 2_000_000,
        }
    }

    fn health(&self) -> HealthStatus {
        HealthStatus::Healthy
    }

    fn name(&self) -> &'static str {
        "test-backend"
    }
}

struct ScriptedStream {
    items: VecDeque<Result<StreamEvent>>,
    active_streams: Arc<AtomicUsize>,
}

impl Stream for ScriptedStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.items.pop_front())
    }
}

impl Drop for ScriptedStream {
    fn drop(&mut self) {
        self.active_streams.fetch_sub(1, Ordering::SeqCst);
    }
}

fn normalized_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn single_tool() -> ToolDefinition {
    ToolDefinition::function(
        "get_weather",
        "Return the current weather for a city",
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
    )
}

fn request_with_tool_prompt() -> CompletionRequest {
    CompletionRequest {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        previous_response_id: None,
        messages: vec![Message::user("What is the weather in London?")],
        tools: vec![single_tool()],
        tool_results: Vec::new(),
        max_tokens: Some(32),
        thinking_budget_tokens: None,
        temperature: Some(0.0),
        parallel_tool_calls: Some(true),
    }
}

#[tokio::test]
async fn ct1_tool_call_round_trip() {
    let backend = TestBackend::new(
        |request| {
            let tool_result = request
                .tool_results
                .first()
                .expect("follow-up request should include tool result");
            assert_eq!(tool_result.tool_call_id, "tool-1");
            assert!(tool_result.content.contains("rain"));
            Ok(CompletionResponse {
                response_id: None,
                text: "It is rainy in London.".to_owned(),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage {
                    input_tokens: 17,
                    output_tokens: 6,
                    reasoning_tokens: None,
                }),
            })
        },
        |request, active_streams| {
            assert_eq!(request.tools.len(), 1);
            Ok(TestBackend::scripted_stream(
                active_streams,
                vec![
                    Ok(StreamEvent::ToolCallStart {
                        id: "tool-1".to_owned(),
                        name: "get_weather".to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallDelta {
                        id: "tool-1".to_owned(),
                        arguments_delta: r#"{"city":"#.to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallDelta {
                        id: "tool-1".to_owned(),
                        arguments_delta: r#""London"}"#.to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallEnd {
                        id: "tool-1".to_owned(),
                    }),
                    Ok(StreamEvent::StopReason(StopReason::ToolUse)),
                    Ok(StreamEvent::Done),
                ],
            ))
        },
    );

    let mut stream = backend.stream(request_with_tool_prompt()).expect("stream");
    let mut tool_json = String::new();
    while let Some(event) = stream.next().await {
        match event.expect("scripted event") {
            StreamEvent::ToolCallDelta {
                arguments_delta, ..
            } => tool_json.push_str(&arguments_delta),
            StreamEvent::Done => break,
            _ => {}
        }
    }
    assert_eq!(tool_json, r#"{"city":"London"}"#);
    drop(stream);

    let mut follow_up = request_with_tool_prompt();
    follow_up.tools.clear();
    follow_up.tool_results.push(ToolResult::with_call(
        "tool-1",
        "get_weather",
        r#"{"city":"London"}"#,
        r#"{"city":"London","forecast":"rain"}"#,
    ));
    follow_up
        .messages
        .push(Message::assistant("Calling get_weather"));

    let response = backend.complete(follow_up).expect("completion");
    assert_eq!(response.text, "It is rainy in London.");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn ct2_streaming_text_fidelity_uses_normalized_equivalence() {
    let backend = TestBackend::new(
        |_request| {
            Ok(CompletionResponse {
                response_id: None,
                text: "Hello world".to_owned(),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        },
        |_request, active_streams| {
            Ok(TestBackend::scripted_stream(
                active_streams,
                vec![
                    Ok(StreamEvent::TextDelta(" Hello".to_owned())),
                    Ok(StreamEvent::TextDelta("   world ".to_owned())),
                    Ok(StreamEvent::StopReason(StopReason::EndTurn)),
                    Ok(StreamEvent::Done),
                ],
            ))
        },
    );

    let request = CompletionRequest::single_user_message("Say hello", 16);
    let response = backend.complete(request.clone()).expect("complete");
    let mut stream = backend.stream(request).expect("stream");
    let mut streamed = String::new();
    let mut stop_reason = None;
    while let Some(event) = stream.next().await {
        match event.expect("stream event") {
            StreamEvent::TextDelta(delta) => streamed.push_str(&delta),
            StreamEvent::StopReason(reason) => stop_reason = Some(reason),
            StreamEvent::Done => break,
            other => panic!("unexpected event: {other:?}"),
        }
    }

    assert_eq!(normalized_text(&streamed), normalized_text(&response.text));
    assert_eq!(stop_reason, Some(response.stop_reason));
}

#[tokio::test]
async fn ct3_streaming_tool_calls_allow_parallel_sequences() {
    let backend = TestBackend::new(
        |_request| unreachable!("complete is not used for ct3"),
        |_request, active_streams| {
            Ok(TestBackend::scripted_stream(
                active_streams,
                vec![
                    Ok(StreamEvent::ToolCallStart {
                        id: "tool-1".to_owned(),
                        name: "lookup_weather".to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallStart {
                        id: "tool-2".to_owned(),
                        name: "lookup_calendar".to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallDelta {
                        id: "tool-1".to_owned(),
                        arguments_delta: r#"{"city":"#.to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallDelta {
                        id: "tool-2".to_owned(),
                        arguments_delta: r#"{"day":"#.to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallDelta {
                        id: "tool-1".to_owned(),
                        arguments_delta: r#""London"}"#.to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallDelta {
                        id: "tool-2".to_owned(),
                        arguments_delta: r#""Monday"}"#.to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallEnd {
                        id: "tool-1".to_owned(),
                    }),
                    Ok(StreamEvent::ToolCallEnd {
                        id: "tool-2".to_owned(),
                    }),
                    Ok(StreamEvent::Done),
                ],
            ))
        },
    );

    let mut stream = backend
        .stream(CompletionRequest::single_user_message("Plan my day", 32))
        .expect("stream");
    let mut started = HashMap::<String, String>::new();
    let mut argument_buffers = HashMap::<String, String>::new();
    let mut ended = Vec::<String>::new();

    while let Some(event) = stream.next().await {
        match event.expect("stream event") {
            StreamEvent::ToolCallStart { id, name } => {
                assert!(started.insert(id.clone(), name).is_none());
                assert!(argument_buffers.insert(id, String::new()).is_none());
            }
            StreamEvent::ToolCallDelta {
                id,
                arguments_delta,
            } => {
                assert!(started.contains_key(&id));
                let buffer = argument_buffers
                    .get_mut(&id)
                    .expect("tool delta should have a matching start");
                buffer.push_str(&arguments_delta);
            }
            StreamEvent::ToolCallEnd { id } => {
                assert!(started.contains_key(&id));
                ended.push(id);
            }
            StreamEvent::Done => break,
            other => panic!("unexpected event: {other:?}"),
        }
    }

    ended.sort();
    assert_eq!(ended, vec!["tool-1".to_owned(), "tool-2".to_owned()]);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            argument_buffers
                .get("tool-1")
                .expect("tool-1 buffer should exist"),
        )
        .expect("tool-1 assembled JSON"),
        serde_json::json!({ "city": "London" })
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            argument_buffers
                .get("tool-2")
                .expect("tool-2 buffer should exist"),
        )
        .expect("tool-2 assembled JSON"),
        serde_json::json!({ "day": "Monday" })
    );
}

#[test]
fn ct4_stop_reason_classification_is_normalized() {
    let backend = TestBackend::new(
        |request| {
            let stop_reason = if !request.tools.is_empty() {
                StopReason::ToolUse
            } else if request.max_tokens == Some(5) {
                StopReason::MaxTokens
            } else {
                StopReason::EndTurn
            };

            Ok(CompletionResponse {
                response_id: None,
                text: "ok".to_owned(),
                stop_reason,
                usage: None,
            })
        },
        |_request, _active_streams| unreachable!("stream is not used for ct4"),
    );

    let natural = backend
        .complete(CompletionRequest::single_user_message("hello", 32))
        .expect("natural");
    assert_eq!(natural.stop_reason, StopReason::EndTurn);

    let max_tokens = backend
        .complete(CompletionRequest::single_user_message("hello", 5))
        .expect("max_tokens");
    assert_eq!(max_tokens.stop_reason, StopReason::MaxTokens);

    let tool_use = backend
        .complete(request_with_tool_prompt())
        .expect("tool use");
    assert_eq!(tool_use.stop_reason, StopReason::ToolUse);
}

#[tokio::test]
async fn ct5_dropping_stream_cleans_up_adapter_resources() {
    let backend = TestBackend::new(
        |_request| unreachable!("complete is not used for ct5"),
        |_request, active_streams| {
            Ok(TestBackend::scripted_stream(
                active_streams,
                vec![
                    Ok(StreamEvent::TextDelta("partial".to_owned())),
                    Ok(StreamEvent::TextDelta(" response".to_owned())),
                ],
            ))
        },
    );

    let mut stream = backend
        .stream(CompletionRequest::single_user_message(
            "stream something",
            32,
        ))
        .expect("stream");
    match stream.next().await.expect("first item").expect("event") {
        StreamEvent::TextDelta(delta) => assert_eq!(delta, "partial"),
        other => panic!("unexpected first event: {other:?}"),
    }
    drop(stream);

    assert_eq!(backend.active_streams.load(Ordering::SeqCst), 0);
}

#[test]
fn ct6_error_classification_uses_typed_errors() {
    let backend = TestBackend::new(
        |request| match request.messages.first().map(|msg| msg.content.as_str()) {
            Some("auth") => Err(LlmError::AuthenticationFailed),
            Some("rate") => Err(LlmError::RateLimited {
                retry_after: Some(Duration::from_secs(2)),
            }),
            Some("overloaded") => Err(LlmError::ServiceUnavailable),
            Some("model") => Err(LlmError::InvalidModel),
            other => panic!("unexpected request: {other:?}"),
        },
        |_request, _active_streams| unreachable!("stream is not used for ct6"),
    );

    assert!(matches!(
        backend.complete(CompletionRequest::single_user_message("auth", 32)),
        Err(LlmError::AuthenticationFailed)
    ));
    assert!(matches!(
        backend.complete(CompletionRequest::single_user_message("rate", 32)),
        Err(LlmError::RateLimited { retry_after: Some(duration) }) if duration == Duration::from_secs(2)
    ));
    assert!(matches!(
        backend.complete(CompletionRequest::single_user_message("overloaded", 32)),
        Err(LlmError::ServiceUnavailable)
    ));
    assert!(matches!(
        backend.complete(CompletionRequest::single_user_message("model", 32)),
        Err(LlmError::InvalidModel)
    ));
}
