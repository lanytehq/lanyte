#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{json, Value};

const ABI_VERSION: i32 = 1;
const MANIFEST_JSON: &str = r#"{"skill_id":"dev.lanyte.echo","name":"Echo","version":"1.0.0","tier":0,"capabilities":["skill.echo"]}"#;

thread_local! {
    static STATE: RefCell<SkillState> = RefCell::new(SkillState::default());
}

#[derive(Default)]
struct SkillState {
    allocations: BTreeMap<u32, Vec<u8>>,
    last_error: Option<Vec<u8>>,
}

impl SkillState {
    fn allocate(&mut self, bytes: Vec<u8>) -> Option<u32> {
        if bytes.is_empty() {
            return None;
        }

        let mut bytes = bytes;
        let ptr = u32::try_from(bytes.as_mut_ptr() as usize).ok()?;
        self.allocations.insert(ptr, bytes);
        Some(ptr)
    }

    fn free(&mut self, ptr: u32) {
        self.allocations.remove(&ptr);
    }

    fn read_input(&self, ptr: u32, len: u32) -> Result<Vec<u8>, SkillErrorPayload> {
        let buffer = self.allocations.get(&ptr).ok_or_else(|| {
            skill_error("input_invalid", "input buffer pointer was not allocated")
        })?;

        if len as usize > buffer.len() {
            return Err(skill_error(
                "input_invalid",
                "input length exceeds allocated guest buffer",
            ));
        }

        Ok(buffer[..len as usize].to_vec())
    }

    fn set_last_error(&mut self, payload: SkillErrorPayload) {
        let bytes = serde_json::to_vec(&payload).unwrap_or_else(|_| {
            br#"{"code":"execution_failed","message":"failed to serialize skill error","retryable":false}"#.to_vec()
        });
        self.last_error = Some(bytes);
    }

    fn clear_last_error(&mut self) {
        self.last_error = None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct SkillErrorPayload {
    code: &'static str,
    message: String,
    retryable: bool,
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn lanyte_skill_abi_version() -> i32 {
    ABI_VERSION
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn lanyte_skill_alloc(size: i32) -> i32 {
    if size <= 0 {
        return 0;
    }

    STATE.with(|state| {
        state
            .borrow_mut()
            .allocate(vec![0_u8; size as usize])
            .and_then(|ptr| i32::try_from(ptr).ok())
            .unwrap_or(0)
    })
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn lanyte_skill_free(ptr: i32, _len: i32) {
    if ptr <= 0 {
        return;
    }

    STATE.with(|state| state.borrow_mut().free(ptr as u32));
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn lanyte_skill_describe() -> i64 {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.clear_last_error();
        match state.allocate(MANIFEST_JSON.as_bytes().to_vec()) {
            Some(ptr) => pack_ptr_len(ptr, MANIFEST_JSON.len()),
            None => {
                state.set_last_error(skill_error(
                    "execution_failed",
                    "failed to allocate manifest payload",
                ));
                0
            }
        }
    })
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn lanyte_skill_invoke(input_ptr: i32, input_len: i32) -> i64 {
    if input_ptr <= 0 || input_len < 0 {
        store_invoke_error(skill_error(
            "input_invalid",
            "input pointer and length must be non-negative guest values",
        ));
        return 0;
    }

    let input = match STATE.with(|state| {
        state
            .borrow()
            .read_input(input_ptr as u32, input_len as u32)
    }) {
        Ok(input) => input,
        Err(err) => {
            store_invoke_error(err);
            return 0;
        }
    };

    match echo_outcome_bytes(&input) {
        Ok(bytes) => STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.clear_last_error();
            match state.allocate(bytes.clone()) {
                Some(ptr) => pack_ptr_len(ptr, bytes.len()),
                None => {
                    state.set_last_error(skill_error(
                        "execution_failed",
                        "failed to allocate invoke output payload",
                    ));
                    0
                }
            }
        }),
        Err(err) => {
            store_invoke_error(err);
            0
        }
    }
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn lanyte_skill_last_error() -> i64 {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(bytes) = state.last_error.take() else {
            return 0;
        };

        match state.allocate(bytes.clone()) {
            Some(ptr) => pack_ptr_len(ptr, bytes.len()),
            None => {
                state.last_error = Some(bytes);
                0
            }
        }
    })
}

fn store_invoke_error(payload: SkillErrorPayload) {
    STATE.with(|state| state.borrow_mut().set_last_error(payload));
}

fn echo_outcome_bytes(input: &[u8]) -> Result<Vec<u8>, SkillErrorPayload> {
    let value: Value = serde_json::from_slice(input)
        .map_err(|_| skill_error("input_invalid", "input must be valid UTF-8 JSON"))?;
    let echoed_input = value
        .as_object()
        .and_then(|object| object.get("input"))
        .cloned()
        .ok_or_else(|| {
            skill_error(
                "input_invalid",
                "input JSON object must include a top-level `input` field",
            )
        })?;

    serde_json::to_vec(&json!({
        "status": "succeeded",
        "result": echoed_input,
        "error": Value::Null,
    }))
    .map_err(|_| skill_error("execution_failed", "failed to serialize ActionOutcome"))
}

fn skill_error(code: &'static str, message: impl Into<String>) -> SkillErrorPayload {
    SkillErrorPayload {
        code,
        message: message.into(),
        retryable: false,
    }
}

fn pack_ptr_len(ptr: u32, len: usize) -> i64 {
    ((u64::try_from(len).expect("length fits in u64") << 32) | u64::from(ptr)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_matches_mission_note() {
        let manifest: Value = serde_json::from_str(MANIFEST_JSON).expect("manifest should parse");

        assert_eq!(manifest["skill_id"], "dev.lanyte.echo");
        assert_eq!(manifest["name"], "Echo");
        assert_eq!(manifest["version"], "1.0.0");
        assert_eq!(manifest["tier"], 0);
        assert_eq!(manifest["capabilities"], json!(["skill.echo"]));
    }

    #[test]
    fn echo_outcome_returns_action_outcome_shape() {
        let outcome = echo_outcome_bytes(br#"{"input":{"message":"hello"},"unused":true}"#)
            .expect("invoke should succeed");
        let outcome: Value = serde_json::from_slice(&outcome).expect("outcome should parse");

        assert_eq!(outcome["status"], "succeeded");
        assert_eq!(outcome["result"], json!({"message":"hello"}));
        assert_eq!(outcome["error"], Value::Null);
    }

    #[test]
    fn echo_outcome_rejects_missing_input_field() {
        let err =
            echo_outcome_bytes(br#"{"payload":"hello"}"#).expect_err("missing input must fail");

        assert_eq!(err.code, "input_invalid");
        assert!(err.message.contains("top-level `input` field"));
    }

    #[test]
    fn echo_outcome_rejects_invalid_json() {
        let err = echo_outcome_bytes(br#"not-json"#).expect_err("invalid json must fail");

        assert_eq!(err.code, "input_invalid");
    }
}
