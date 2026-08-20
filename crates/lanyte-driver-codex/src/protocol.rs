use chrono::Utc;
use lanyte_mission::NormalizedHarnessEvent;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum CodexProtocolError {
    #[error("invalid JSON-RPC line: {0}")]
    InvalidLine(#[from] serde_json::Error),
    #[error("remote error: {0}")]
    Remote(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcLine {
    #[serde(default)]
    pub id: Option<Value>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub params: Option<Value>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<Value>,
}

/// Map a server notification (or unsolicited method) to a normalized event.
pub fn map_notification(attempt_id: Uuid, line: &str) -> Option<NormalizedHarnessEvent> {
    let parsed: JsonRpcLine = serde_json::from_str(line).ok()?;
    let method = parsed.method.as_deref()?;
    let now = Utc::now();
    match method {
        "thread/started" | "session/started" => Some(NormalizedHarnessEvent::Started {
            occurred_at: now,
            attempt_id,
            harness_session_id: extract_id(&parsed.params).unwrap_or_else(|| "unknown".to_owned()),
            detail: Some(method.to_owned()),
        }),
        "tool/called" | "command/started" => Some(NormalizedHarnessEvent::ToolProposed {
            occurred_at: now,
            attempt_id,
            tool: extract_tool(&parsed.params).unwrap_or_else(|| method.to_owned()),
            detail: Some(method.to_owned()),
        }),
        "item/started" if is_tool_item(&parsed.params) => {
            Some(NormalizedHarnessEvent::ToolProposed {
                occurred_at: now,
                attempt_id,
                tool: extract_tool(&parsed.params).unwrap_or_else(|| method.to_owned()),
                detail: Some(method.to_owned()),
            })
        }
        "turn/started" => {
            let turn_id = extract_turn(&parsed.params)?;
            Some(NormalizedHarnessEvent::TurnProgress {
                occurred_at: now,
                attempt_id,
                thread_id: extract_id(&parsed.params),
                turn_id,
                status: "started".to_owned(),
            })
        }
        "turn/completed" => {
            let turn_id = extract_turn(&parsed.params)?;
            Some(NormalizedHarnessEvent::TurnProgress {
                occurred_at: now,
                attempt_id,
                thread_id: extract_id(&parsed.params),
                turn_id,
                status: extract_turn_status(&parsed.params)
                    .unwrap_or_else(|| "completed".to_owned()),
            })
        }
        "thread/exited" | "session/completed" => {
            let failed = parsed
                .params
                .as_ref()
                .and_then(|params| params.get("error"))
                .is_some_and(|error| !error.is_null());
            Some(NormalizedHarnessEvent::Exited {
                occurred_at: now,
                attempt_id,
                success: !failed,
                detail: Some(method.to_owned()),
            })
        }
        _ => None,
    }
}

fn extract_id(params: &Option<Value>) -> Option<String> {
    let params = params.as_ref()?;
    params
        .get("threadId")
        .or_else(|| params.get("id"))
        .or_else(|| params.pointer("/thread/id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn extract_turn(params: &Option<Value>) -> Option<String> {
    let params = params.as_ref()?;
    params
        .get("turnId")
        .or_else(|| params.pointer("/turn/id"))
        .or_else(|| params.pointer("/item/id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn extract_turn_status(params: &Option<Value>) -> Option<String> {
    let params = params.as_ref()?;
    params
        .get("status")
        .or_else(|| params.pointer("/turn/status"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn extract_tool(params: &Option<Value>) -> Option<String> {
    let params = params.as_ref()?;
    params
        .get("tool")
        .or_else(|| params.get("name"))
        .or_else(|| params.pointer("/item/command"))
        .or_else(|| params.pointer("/item/type"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn is_tool_item(params: &Option<Value>) -> bool {
    let Some(params) = params.as_ref() else {
        return false;
    };
    let item_type = params
        .pointer("/item/type")
        .or_else(|| params.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    matches!(
        item_type,
        "command_execution" | "tool" | "mcp_tool_call" | "function_call"
    ) || params.get("tool").is_some()
        || params.get("name").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_thread_started() {
        let attempt = Uuid::nil();
        let event = map_notification(
            attempt,
            r#"{"method":"thread/started","params":{"threadId":"thr_1"}}"#,
        )
        .expect("mapped");
        match event {
            NormalizedHarnessEvent::Started {
                harness_session_id, ..
            } => assert_eq!(harness_session_id, "thr_1"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn maps_tool_and_exit() {
        let attempt = Uuid::nil();
        assert!(matches!(
            map_notification(
                attempt,
                r#"{"method":"item/started","params":{"item":{"type":"command_execution"},"name":"shell"}}"#
            ),
            Some(NormalizedHarnessEvent::ToolProposed { tool, .. }) if tool == "shell"
        ));
        assert!(map_notification(
            attempt,
            r#"{"method":"item/completed","params":{"item":{"type":"agent_message"}}}"#
        )
        .is_none());
        assert!(map_notification(attempt, r#"{"method":"turn/completed","params":{}}"#).is_none());
        assert!(matches!(
            map_notification(
                attempt,
                r#"{"method":"thread/exited","params":{"threadId":"thr_1"}}"#
            ),
            Some(NormalizedHarnessEvent::Exited { success: true, .. })
        ));
    }

    #[test]
    fn ignores_unknown_methods() {
        assert!(map_notification(Uuid::nil(), r#"{"method":"window/title"}"#).is_none());
    }
}
