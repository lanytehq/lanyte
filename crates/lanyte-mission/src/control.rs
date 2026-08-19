use std::fmt;

use chrono::{DateTime, Utc};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::{Uuid, Version};

use crate::{MissionPhase, MissionRecord, NormalizedHarnessEvent, RecoveryPolicy, Validate};

pub const MISSION_CONTROL_SCHEMA: &str =
    "https://schemas.3leaps.dev/agentic/mission/v0.1/mission-control.schema.json";
pub const MISSION_CONTROL_SCHEMA_V0: &str =
    "https://schemas.3leaps.dev/agentic/mission/v0/mission-control.schema.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MissionCreateBody {
    pub goal: String,
    pub policy_id: String,
    pub deadline_at: Option<DateTime<Utc>>,
    pub recovery_policy: RecoveryPolicy,
}

impl<'de> Deserialize<'de> for MissionCreateBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            goal: String,
            policy_id: String,
            deadline_at: Value,
            recovery_policy: RecoveryPolicy,
        }

        let raw = Raw::deserialize(deserializer)?;
        let deadline_at = match raw.deadline_at {
            Value::Null => None,
            Value::String(value) => Some(parse_contract_timestamp::<D::Error>(&value)?),
            _ => {
                return Err(D::Error::custom(
                    "deadline_at must be null or an RFC 3339 timestamp",
                ));
            }
        };
        Ok(Self {
            goal: raw.goal,
            policy_id: raw.policy_id,
            deadline_at,
            recovery_policy: raw.recovery_policy,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MissionShowBody {
    pub mission_id: Uuid,
}

impl<'de> Deserialize<'de> for MissionShowBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            mission_id: String,
        }

        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            mission_id: parse_canonical_uuid_v4::<D::Error>(&raw.mission_id, "mission_id")?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MissionListBody {
    pub phases: Vec<MissionPhase>,
    pub limit: u16,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MissionLaunchBody {
    pub mission_id: Uuid,
    pub workspace: String,
    pub binary: Option<String>,
}

impl<'de> Deserialize<'de> for MissionLaunchBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            mission_id: String,
            workspace: String,
            binary: Value,
        }

        let raw = Raw::deserialize(deserializer)?;
        if raw.workspace.is_empty() || raw.workspace.chars().count() > 4096 {
            return Err(D::Error::custom(
                "workspace must contain 1..=4096 characters",
            ));
        }
        let binary = match raw.binary {
            Value::Null => None,
            Value::String(value) if !value.is_empty() && value.chars().count() <= 4096 => {
                Some(value)
            }
            _ => {
                return Err(D::Error::custom(
                    "binary must be null or a path of 1..=4096 characters",
                ));
            }
        };
        Ok(Self {
            mission_id: parse_canonical_uuid_v4::<D::Error>(&raw.mission_id, "mission_id")?,
            workspace: raw.workspace,
            binary,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissionControlRequest {
    Create {
        request_id: Uuid,
        idempotency_key: String,
        body: MissionCreateBody,
    },
    Show {
        request_id: Uuid,
        body: MissionShowBody,
    },
    List {
        request_id: Uuid,
        body: MissionListBody,
    },
    Launch {
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        body: MissionLaunchBody,
    },
    Observe {
        request_id: Uuid,
        body: MissionShowBody,
    },
    Close {
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        body: MissionShowBody,
    },
    Cancel {
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        body: MissionShowBody,
    },
}

impl MissionControlRequest {
    pub fn create(
        request_id: Uuid,
        idempotency_key: String,
        body: MissionCreateBody,
    ) -> Result<Self, String> {
        let request = Self::Create {
            request_id,
            idempotency_key,
            body,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn show(request_id: Uuid, mission_id: Uuid) -> Result<Self, String> {
        let request = Self::Show {
            request_id,
            body: MissionShowBody { mission_id },
        };
        request.validate()?;
        Ok(request)
    }

    pub fn list(request_id: Uuid, body: MissionListBody) -> Result<Self, String> {
        let request = Self::List { request_id, body };
        request.validate()?;
        Ok(request)
    }

    pub fn launch(
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        body: MissionLaunchBody,
    ) -> Result<Self, String> {
        let request = Self::Launch {
            request_id,
            idempotency_key,
            expected_revision,
            body,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn observe(request_id: Uuid, mission_id: Uuid) -> Result<Self, String> {
        let request = Self::Observe {
            request_id,
            body: MissionShowBody { mission_id },
        };
        request.validate()?;
        Ok(request)
    }

    pub fn close(
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        mission_id: Uuid,
    ) -> Result<Self, String> {
        let request = Self::Close {
            request_id,
            idempotency_key,
            expected_revision,
            body: MissionShowBody { mission_id },
        };
        request.validate()?;
        Ok(request)
    }

    #[must_use]
    pub const fn request_id(&self) -> Uuid {
        match self {
            Self::Create { request_id, .. }
            | Self::Show { request_id, .. }
            | Self::List { request_id, .. }
            | Self::Launch { request_id, .. }
            | Self::Observe { request_id, .. }
            | Self::Close { request_id, .. }
            | Self::Cancel { request_id, .. } => *request_id,
        }
    }

    #[must_use]
    pub const fn operation(&self) -> &'static str {
        match self {
            Self::Create { .. } => "mission.create",
            Self::Show { .. } => "mission.show",
            Self::List { .. } => "mission.list",
            Self::Launch { .. } => "mission.launch",
            Self::Observe { .. } => "mission.observe",
            Self::Close { .. } => "mission.close",
            Self::Cancel { .. } => "mission.cancel",
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.request_id().get_version() != Some(Version::Random) {
            return Err("request_id must be a UUID v4".to_owned());
        }
        match self {
            Self::Create {
                idempotency_key,
                body,
                ..
            } => {
                validate_idempotency_key(idempotency_key)?;
                validate_create_body(body)
            }
            Self::Show { body, .. } | Self::Observe { body, .. } => {
                if body.mission_id.get_version() != Some(Version::Random) {
                    return Err("mission_id must be a UUID v4".to_owned());
                }
                Ok(())
            }
            Self::List { body, .. } => validate_list_body(body),
            Self::Launch {
                idempotency_key,
                body,
                ..
            } => {
                validate_idempotency_key(idempotency_key)?;
                if body.mission_id.get_version() != Some(Version::Random) {
                    return Err("mission_id must be a UUID v4".to_owned());
                }
                if body.workspace.is_empty() || body.workspace.chars().count() > 4096 {
                    return Err("workspace must contain 1..=4096 characters".to_owned());
                }
                Ok(())
            }
            Self::Close {
                idempotency_key,
                body,
                ..
            }
            | Self::Cancel {
                idempotency_key,
                body,
                ..
            } => {
                validate_idempotency_key(idempotency_key)?;
                if body.mission_id.get_version() != Some(Version::Random) {
                    return Err("mission_id must be a UUID v4".to_owned());
                }
                Ok(())
            }
        }
    }

    pub fn cancel(
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        mission_id: Uuid,
    ) -> Result<Self, String> {
        let request = Self::Cancel {
            request_id,
            idempotency_key,
            expected_revision,
            body: MissionShowBody { mission_id },
        };
        request.validate()?;
        Ok(request)
    }
}

impl Serialize for MissionControlRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut value = match self {
            Self::Create {
                request_id,
                idempotency_key,
                body,
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "expected_revision": null,
                "operation": "mission.create",
                "body": body,
            }),
            Self::Show { request_id, body } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": request_id,
                "idempotency_key": null,
                "expected_revision": null,
                "operation": "mission.show",
                "body": body,
            }),
            Self::List { request_id, body } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": request_id,
                "idempotency_key": null,
                "expected_revision": null,
                "operation": "mission.list",
                "body": body,
            }),
            Self::Launch {
                request_id,
                idempotency_key,
                expected_revision,
                body,
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "expected_revision": expected_revision,
                "operation": "mission.launch",
                "body": body,
            }),
            Self::Observe { request_id, body } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": request_id,
                "idempotency_key": null,
                "expected_revision": null,
                "operation": "mission.observe",
                "body": body,
            }),
            Self::Close {
                request_id,
                idempotency_key,
                expected_revision,
                body,
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "expected_revision": expected_revision,
                "operation": "mission.close",
                "body": body,
            }),
            Self::Cancel {
                request_id,
                idempotency_key,
                expected_revision,
                body,
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "expected_revision": expected_revision,
                "operation": "mission.cancel",
                "body": body,
            }),
        };
        apply_request_control_fields(
            &mut value,
            matches!(
                self,
                Self::Create { .. }
                    | Self::Launch { .. }
                    | Self::Close { .. }
                    | Self::Cancel { .. }
            ),
        );
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MissionControlRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            control_schema: String,
            kind: String,
            request_id: String,
            idempotency_key: Value,
            expected_revision: Value,
            operation: String,
            body: Value,
            #[serde(default)]
            request_fingerprint: Value,
            #[serde(default)]
            original_result_hash: Value,
        }

        let value = Value::deserialize(deserializer)?;
        let schema = value
            .get("control_schema")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let is_v01 = schema == MISSION_CONTROL_SCHEMA;
        if schema == MISSION_CONTROL_SCHEMA_V0 {
            let obj = value
                .as_object()
                .ok_or_else(|| D::Error::custom("request must be an object"))?;
            if obj.contains_key("request_fingerprint") || obj.contains_key("original_result_hash") {
                return Err(D::Error::custom("frozen v0 control hashes must be omitted"));
            }
            if value.get("operation").and_then(Value::as_str) == Some("mission.cancel") {
                return Err(D::Error::custom(
                    "frozen v0 does not include mission.cancel",
                ));
            }
        }
        let raw: Raw = serde_json::from_value(value).map_err(D::Error::custom)?;
        if raw.control_schema != MISSION_CONTROL_SCHEMA
            && raw.control_schema != MISSION_CONTROL_SCHEMA_V0
        {
            return Err(D::Error::custom("unsupported control_schema"));
        }
        if raw.kind != "request" {
            return Err(D::Error::custom("kind must be request"));
        }
        let mutating = matches!(
            raw.operation.as_str(),
            "mission.create" | "mission.launch" | "mission.close" | "mission.cancel"
        );
        verify_raw_control_hashes(HashCheck {
            mutating,
            is_request: true,
            is_v01,
            request_fingerprint: &raw.request_fingerprint,
            original_result_hash: &raw.original_result_hash,
            request_id: raw.request_id.as_str(),
            idempotency_key: raw.idempotency_key.clone(),
            expected_revision: raw.expected_revision.clone(),
            operation: raw.operation.as_str(),
            body: raw.body.clone(),
            control_schema: raw.control_schema.as_str(),
        })
        .map_err(D::Error::custom)?;
        let request_id = parse_canonical_uuid_v4::<D::Error>(&raw.request_id, "request_id")?;
        let request = match raw.operation.as_str() {
            "mission.create" => {
                require_null_revision(&raw.expected_revision)?;
                let idempotency_key = raw
                    .idempotency_key
                    .as_str()
                    .ok_or_else(|| {
                        D::Error::custom("mission.create idempotency_key must be a string")
                    })?
                    .to_owned();
                let body = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::Create {
                    request_id,
                    idempotency_key,
                    body,
                }
            }
            "mission.show" => {
                require_null_revision(&raw.expected_revision)?;
                if !raw.idempotency_key.is_null() {
                    return Err(D::Error::custom(
                        "mission.show idempotency_key must be null",
                    ));
                }
                let body = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::Show { request_id, body }
            }
            "mission.list" => {
                require_null_revision(&raw.expected_revision)?;
                if !raw.idempotency_key.is_null() {
                    return Err(D::Error::custom(
                        "mission.list idempotency_key must be null",
                    ));
                }
                let body = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::List { request_id, body }
            }
            "mission.launch" => {
                let expected_revision = require_revision(&raw.expected_revision)?;
                let idempotency_key = raw
                    .idempotency_key
                    .as_str()
                    .ok_or_else(|| {
                        D::Error::custom("mission.launch idempotency_key must be a string")
                    })?
                    .to_owned();
                let body = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::Launch {
                    request_id,
                    idempotency_key,
                    expected_revision,
                    body,
                }
            }
            "mission.observe" => {
                require_null_revision(&raw.expected_revision)?;
                if !raw.idempotency_key.is_null() {
                    return Err(D::Error::custom(
                        "mission.observe idempotency_key must be null",
                    ));
                }
                let body = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::Observe { request_id, body }
            }
            "mission.close" => {
                let expected_revision = require_revision(&raw.expected_revision)?;
                let idempotency_key = raw
                    .idempotency_key
                    .as_str()
                    .ok_or_else(|| {
                        D::Error::custom("mission.close idempotency_key must be a string")
                    })?
                    .to_owned();
                let body = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::Close {
                    request_id,
                    idempotency_key,
                    expected_revision,
                    body,
                }
            }
            "mission.cancel" => {
                let expected_revision = require_revision(&raw.expected_revision)?;
                let idempotency_key = raw
                    .idempotency_key
                    .as_str()
                    .ok_or_else(|| {
                        D::Error::custom("mission.cancel idempotency_key must be a string")
                    })?
                    .to_owned();
                let body = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::Cancel {
                    request_id,
                    idempotency_key,
                    expected_revision,
                    body,
                }
            }
            _ => return Err(D::Error::custom("unsupported mission operation")),
        };
        request.validate().map_err(D::Error::custom)?;
        Ok(request)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissionControlResult {
    Record {
        request_id: Uuid,
        operation: &'static str,
        idempotency_key: Option<String>,
        expected_revision: Option<u64>,
        record: Box<MissionRecord>,
        request_fingerprint: Option<String>,
    },
    List {
        request_id: Uuid,
        records: Vec<MissionRecord>,
        next_cursor: Option<String>,
    },
    Observe {
        request_id: Uuid,
        mission_id: Uuid,
        attempt_id: Uuid,
        events: Vec<NormalizedHarnessEvent>,
    },
    Close {
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        mission_id: Uuid,
        attempt_id: Uuid,
        request_fingerprint: Option<String>,
    },
    Cancel {
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        record: Box<MissionRecord>,
        progress: CancelProgress,
        request_fingerprint: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolCancelOutcome {
    RequestAccepted,
    Interrupted,
    Unavailable,
    Failed,
    Timeout,
    UnrelatedCompletion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackCancelOutcome {
    KillDispatched,
    Cleared,
    Survivors,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolCancelProgress {
    pub outcome: ProtocolCancelOutcome,
    pub thread_id: Option<String>,
    pub turn_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FallbackCancelProgress {
    pub outcome: FallbackCancelOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancelProgress {
    pub requested: bool,
    pub protocol: Option<ProtocolCancelProgress>,
    pub fallback: Option<FallbackCancelProgress>,
}

impl MissionControlResult {
    pub fn create(
        request_id: Uuid,
        idempotency_key: String,
        record: MissionRecord,
    ) -> Result<Self, String> {
        record.validate().map_err(|err| err.to_string())?;
        validate_idempotency_key(&idempotency_key)?;
        Ok(Self::Record {
            request_id,
            operation: "mission.create",
            idempotency_key: Some(idempotency_key),
            expected_revision: None,
            record: Box::new(record),
            request_fingerprint: None,
        })
    }

    pub fn show(request_id: Uuid, record: MissionRecord) -> Result<Self, String> {
        record.validate().map_err(|err| err.to_string())?;
        Ok(Self::Record {
            request_id,
            operation: "mission.show",
            idempotency_key: None,
            expected_revision: None,
            record: Box::new(record),
            request_fingerprint: None,
        })
    }

    pub fn launch(
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        record: MissionRecord,
    ) -> Result<Self, String> {
        record.validate().map_err(|err| err.to_string())?;
        validate_idempotency_key(&idempotency_key)?;
        Ok(Self::Record {
            request_id,
            operation: "mission.launch",
            idempotency_key: Some(idempotency_key),
            expected_revision: Some(expected_revision),
            record: Box::new(record),
            request_fingerprint: None,
        })
    }

    pub fn observe(
        request_id: Uuid,
        mission_id: Uuid,
        attempt_id: Uuid,
        events: Vec<NormalizedHarnessEvent>,
    ) -> Result<Self, String> {
        if events.len() > 32 {
            return Err("observe result exceeds 32 events".to_owned());
        }
        Ok(Self::Observe {
            request_id,
            mission_id,
            attempt_id,
            events,
        })
    }

    pub fn close(
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        mission_id: Uuid,
        attempt_id: Uuid,
    ) -> Result<Self, String> {
        validate_idempotency_key(&idempotency_key)?;
        Ok(Self::Close {
            request_id,
            idempotency_key,
            expected_revision,
            mission_id,
            attempt_id,
            request_fingerprint: None,
        })
    }

    pub fn list(
        request_id: Uuid,
        records: Vec<MissionRecord>,
        next_cursor: Option<String>,
    ) -> Result<Self, String> {
        for record in &records {
            record.validate().map_err(|err| err.to_string())?;
        }
        if records.len() > 200 {
            return Err("mission list result exceeds 200 records".to_owned());
        }
        validate_cursor(next_cursor.as_deref())?;
        Ok(Self::List {
            request_id,
            records,
            next_cursor,
        })
    }

    #[must_use]
    pub fn bind_request_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        let fingerprint = Some(fingerprint.into());
        match &mut self {
            Self::Record {
                request_fingerprint,
                ..
            }
            | Self::Close {
                request_fingerprint,
                ..
            }
            | Self::Cancel {
                request_fingerprint,
                ..
            } => *request_fingerprint = fingerprint,
            Self::List { .. } | Self::Observe { .. } => {}
        }
        self
    }

    fn stored_request_fingerprint(&self) -> Option<&str> {
        match self {
            Self::Record {
                request_fingerprint,
                ..
            }
            | Self::Close {
                request_fingerprint,
                ..
            }
            | Self::Cancel {
                request_fingerprint,
                ..
            } => request_fingerprint.as_deref(),
            Self::List { .. } | Self::Observe { .. } => None,
        }
    }
}

impl Serialize for MissionControlResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut value = match self {
            Self::Record {
                request_id,
                operation,
                idempotency_key,
                expected_revision,
                record,
                ..
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "result",
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "expected_revision": expected_revision,
                "operation": operation,
                "body": { "record": record },
            }),
            Self::List {
                request_id,
                records,
                next_cursor,
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "result",
                "request_id": request_id,
                "idempotency_key": null,
                "expected_revision": null,
                "operation": "mission.list",
                "body": {
                    "records": records,
                    "next_cursor": next_cursor,
                },
            }),
            Self::Observe {
                request_id,
                mission_id,
                attempt_id,
                events,
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "result",
                "request_id": request_id,
                "idempotency_key": null,
                "expected_revision": null,
                "operation": "mission.observe",
                "body": {
                    "mission_id": mission_id,
                    "attempt_id": attempt_id,
                    "events": events,
                },
            }),
            Self::Close {
                request_id,
                idempotency_key,
                expected_revision,
                mission_id,
                attempt_id,
                ..
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "result",
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "expected_revision": expected_revision,
                "operation": "mission.close",
                "body": {
                    "mission_id": mission_id,
                    "attempt_id": attempt_id,
                    "closed": true,
                },
            }),
            Self::Cancel {
                request_id,
                idempotency_key,
                expected_revision,
                record,
                progress,
                ..
            } => serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "result",
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "expected_revision": expected_revision,
                "operation": "mission.cancel",
                "body": {
                    "record": record,
                    "progress": progress,
                },
            }),
        };
        let mutating = matches!(
            self,
            Self::Record {
                operation: "mission.create" | "mission.launch",
                ..
            } | Self::Close { .. }
                | Self::Cancel { .. }
        );
        apply_result_control_fields(&mut value, mutating, self);
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MissionControlResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            control_schema: String,
            kind: String,
            request_id: String,
            idempotency_key: Value,
            expected_revision: Value,
            operation: String,
            body: Value,
            #[serde(default)]
            request_fingerprint: Value,
            #[serde(default)]
            original_result_hash: Value,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RecordBody {
            record: MissionRecord,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ListBody {
            records: Vec<MissionRecord>,
            #[serde(deserialize_with = "deserialize_required_option")]
            next_cursor: Option<String>,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ObserveBody {
            mission_id: String,
            attempt_id: String,
            events: Vec<NormalizedHarnessEvent>,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct CloseBody {
            mission_id: String,
            attempt_id: String,
            closed: bool,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct CancelBody {
            record: MissionRecord,
            progress: CancelProgress,
        }

        let value = Value::deserialize(deserializer)?;
        let schema = value
            .get("control_schema")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if schema == MISSION_CONTROL_SCHEMA_V0 {
            let obj = value
                .as_object()
                .ok_or_else(|| D::Error::custom("result must be an object"))?;
            if obj.contains_key("request_fingerprint") || obj.contains_key("original_result_hash") {
                return Err(D::Error::custom("frozen v0 control hashes must be omitted"));
            }
            if value.get("operation").and_then(Value::as_str) == Some("mission.cancel") {
                return Err(D::Error::custom(
                    "frozen v0 does not include mission.cancel",
                ));
            }
        }
        let raw: Raw = serde_json::from_value(value).map_err(D::Error::custom)?;
        if raw.control_schema != MISSION_CONTROL_SCHEMA
            && raw.control_schema != MISSION_CONTROL_SCHEMA_V0
        {
            return Err(D::Error::custom("unsupported control_schema"));
        }
        if raw.kind != "result" {
            return Err(D::Error::custom("kind must be result"));
        }
        let mutating = matches!(
            raw.operation.as_str(),
            "mission.create" | "mission.launch" | "mission.close" | "mission.cancel"
        );
        let is_v01 = raw.control_schema == MISSION_CONTROL_SCHEMA;
        verify_raw_control_hashes(HashCheck {
            mutating,
            is_request: false,
            is_v01,
            request_fingerprint: &raw.request_fingerprint,
            original_result_hash: &raw.original_result_hash,
            request_id: raw.request_id.as_str(),
            idempotency_key: raw.idempotency_key.clone(),
            expected_revision: raw.expected_revision.clone(),
            operation: raw.operation.as_str(),
            body: raw.body.clone(),
            control_schema: raw.control_schema.as_str(),
        })
        .map_err(D::Error::custom)?;
        let request_id = parse_canonical_uuid_v4::<D::Error>(&raw.request_id, "request_id")?;
        let result = match raw.operation.as_str() {
            "mission.create" => {
                require_null_revision(&raw.expected_revision)?;
                let idempotency_key = raw
                    .idempotency_key
                    .as_str()
                    .ok_or_else(|| {
                        D::Error::custom("mission.create idempotency_key must be a string")
                    })?
                    .to_owned();
                let body: RecordBody =
                    serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::create(request_id, idempotency_key, body.record).map_err(D::Error::custom)?
            }
            "mission.show" => {
                require_null_revision(&raw.expected_revision)?;
                if !raw.idempotency_key.is_null() {
                    return Err(D::Error::custom(
                        "mission.show idempotency_key must be null",
                    ));
                }
                let body: RecordBody =
                    serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::show(request_id, body.record).map_err(D::Error::custom)?
            }
            "mission.launch" => {
                let expected_revision = require_revision(&raw.expected_revision)?;
                let idempotency_key = raw
                    .idempotency_key
                    .as_str()
                    .ok_or_else(|| {
                        D::Error::custom("mission.launch idempotency_key must be a string")
                    })?
                    .to_owned();
                let body: RecordBody =
                    serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::launch(request_id, idempotency_key, expected_revision, body.record)
                    .map_err(D::Error::custom)?
            }
            "mission.list" => {
                require_null_revision(&raw.expected_revision)?;
                if !raw.idempotency_key.is_null() {
                    return Err(D::Error::custom(
                        "mission.list idempotency_key must be null",
                    ));
                }
                let body: ListBody = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                Self::list(request_id, body.records, body.next_cursor).map_err(D::Error::custom)?
            }
            "mission.observe" => {
                require_null_revision(&raw.expected_revision)?;
                if !raw.idempotency_key.is_null() {
                    return Err(D::Error::custom(
                        "mission.observe idempotency_key must be null",
                    ));
                }
                let body: ObserveBody =
                    serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                let mission_id =
                    parse_canonical_uuid_v4::<D::Error>(&body.mission_id, "mission_id")?;
                let attempt_id =
                    parse_canonical_uuid_v4::<D::Error>(&body.attempt_id, "attempt_id")?;
                Self::observe(request_id, mission_id, attempt_id, body.events)
                    .map_err(D::Error::custom)?
            }
            "mission.close" => {
                let expected_revision = require_revision(&raw.expected_revision)?;
                let idempotency_key = raw
                    .idempotency_key
                    .as_str()
                    .ok_or_else(|| {
                        D::Error::custom("mission.close idempotency_key must be a string")
                    })?
                    .to_owned();
                let body: CloseBody = serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                if !body.closed {
                    return Err(D::Error::custom(
                        "mission.close result must set closed true",
                    ));
                }
                let mission_id =
                    parse_canonical_uuid_v4::<D::Error>(&body.mission_id, "mission_id")?;
                let attempt_id =
                    parse_canonical_uuid_v4::<D::Error>(&body.attempt_id, "attempt_id")?;
                Self::close(
                    request_id,
                    idempotency_key,
                    expected_revision,
                    mission_id,
                    attempt_id,
                )
                .map_err(D::Error::custom)?
            }
            "mission.cancel" => {
                let expected_revision = require_revision(&raw.expected_revision)?;
                let idempotency_key = raw
                    .idempotency_key
                    .as_str()
                    .ok_or_else(|| {
                        D::Error::custom("mission.cancel idempotency_key must be a string")
                    })?
                    .to_owned();
                let body: CancelBody =
                    serde_json::from_value(raw.body).map_err(D::Error::custom)?;
                if !body.progress.requested {
                    return Err(D::Error::custom(
                        "mission.cancel progress.requested must be true",
                    ));
                }
                if let Some(protocol) = &body.progress.protocol {
                    if protocol.outcome == ProtocolCancelOutcome::Interrupted
                        && (protocol
                            .thread_id
                            .as_ref()
                            .is_none_or(|value| value.is_empty())
                            || protocol
                                .turn_id
                                .as_ref()
                                .is_none_or(|value| value.is_empty()))
                    {
                        return Err(D::Error::custom(
                            "interrupted cancel progress requires nonempty thread_id and turn_id",
                        ));
                    }
                }
                body.record
                    .validate()
                    .map_err(|err| D::Error::custom(err.to_string()))?;
                validate_idempotency_key(&idempotency_key).map_err(D::Error::custom)?;
                Self::Cancel {
                    request_id,
                    idempotency_key,
                    expected_revision,
                    record: Box::new(body.record),
                    progress: body.progress,
                    request_fingerprint: None,
                }
            }
            _ => return Err(D::Error::custom("unsupported mission operation")),
        };
        bind_parsed_fingerprint(result, &raw.request_fingerprint)
    }
}

fn bind_parsed_fingerprint<E>(
    result: MissionControlResult,
    fingerprint: &Value,
) -> Result<MissionControlResult, E>
where
    E: serde::de::Error,
{
    match fingerprint {
        Value::Null => Ok(result),
        Value::String(value) => Ok(result.bind_request_fingerprint(value.clone())),
        _ => Err(E::custom("request_fingerprint must be a string or null")),
    }
}

pub fn control_content_hash(value: &Value) -> Option<String> {
    let Value::Object(map) = value else {
        return None;
    };
    let mut filtered = serde_json::Map::new();
    for (key, item) in map {
        if matches!(
            key.as_str(),
            "request_id" | "request_fingerprint" | "original_result_hash"
        ) {
            continue;
        }
        filtered.insert(key.clone(), item.clone());
    }
    let blob = serde_json::to_string(&canonical_control_value(&Value::Object(filtered))).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(blob.as_bytes());
    Some(format!("{:x}", hasher.finalize()))
}

fn canonical_control_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            let mut sorted = serde_json::Map::new();
            for key in keys {
                sorted.insert(key.clone(), canonical_control_value(&map[&key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_control_value).collect()),
        other => other.clone(),
    }
}

fn apply_request_control_fields(value: &mut Value, mutating: bool) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };
    obj.remove("request_fingerprint");
    obj.remove("original_result_hash");
    if !mutating {
        obj.insert("request_fingerprint".to_owned(), Value::Null);
        obj.insert("original_result_hash".to_owned(), Value::Null);
        return;
    }
    let fingerprint = control_content_hash(value).unwrap_or_default();
    if let Some(obj) = value.as_object_mut() {
        obj.insert("request_fingerprint".to_owned(), Value::String(fingerprint));
        obj.insert("original_result_hash".to_owned(), Value::Null);
    }
}

fn apply_result_control_fields(value: &mut Value, mutating: bool, result: &MissionControlResult) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };
    obj.remove("request_fingerprint");
    obj.remove("original_result_hash");
    if !mutating {
        obj.insert("request_fingerprint".to_owned(), Value::Null);
        obj.insert("original_result_hash".to_owned(), Value::Null);
        return;
    }
    let result_hash = control_content_hash(value).unwrap_or_default();
    let request_fingerprint = result
        .stored_request_fingerprint()
        .map(str::to_owned)
        .or_else(|| reconstructed_request_fingerprint(result))
        .unwrap_or_default();
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "request_fingerprint".to_owned(),
            Value::String(request_fingerprint),
        );
        obj.insert(
            "original_result_hash".to_owned(),
            Value::String(result_hash),
        );
    }
}

fn reconstructed_request_fingerprint(result: &MissionControlResult) -> Option<String> {
    let value = match result {
        MissionControlResult::Record {
            request_id,
            operation,
            idempotency_key,
            expected_revision,
            record,
            ..
        } if matches!(*operation, "mission.create" | "mission.launch") => json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "expected_revision": expected_revision,
            "operation": operation,
            "body": if *operation == "mission.create" {
                json!({
                    "goal": record.goal,
                    "policy_id": record.policy_id,
                    "deadline_at": record.deadline_at,
                    "recovery_policy": record.recovery_policy,
                })
            } else {
                json!({
                    "mission_id": record.mission_id,
                    "workspace": record.harness_selection.as_ref().map(|item| item.workspace_ref.clone()).unwrap_or_default(),
                    "binary": Value::Null,
                })
            }
        }),
        MissionControlResult::Close {
            request_id,
            idempotency_key,
            expected_revision,
            mission_id,
            ..
        } => json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "expected_revision": expected_revision,
            "operation": "mission.close",
            "body": { "mission_id": mission_id },
        }),
        MissionControlResult::Cancel {
            request_id,
            idempotency_key,
            expected_revision,
            record,
            ..
        } => json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "expected_revision": expected_revision,
            "operation": "mission.cancel",
            "body": { "mission_id": record.mission_id },
        }),
        _ => return None,
    };
    control_content_hash(&value)
}

struct HashCheck<'a> {
    mutating: bool,
    is_request: bool,
    is_v01: bool,
    request_fingerprint: &'a Value,
    original_result_hash: &'a Value,
    request_id: &'a str,
    idempotency_key: Value,
    expected_revision: Value,
    operation: &'a str,
    body: Value,
    control_schema: &'a str,
}

fn verify_raw_control_hashes(check: HashCheck<'_>) -> Result<(), String> {
    let envelope = json!({
        "control_schema": check.control_schema,
        "kind": if check.is_request { "request" } else { "result" },
        "request_id": check.request_id,
        "idempotency_key": check.idempotency_key,
        "expected_revision": check.expected_revision,
        "operation": check.operation,
        "body": check.body,
    });
    if !check.is_v01 {
        if !check.request_fingerprint.is_null() || !check.original_result_hash.is_null() {
            return Err("frozen v0 control hashes must be omitted".to_owned());
        }
        return Ok(());
    }
    if !check.mutating {
        if !check.request_fingerprint.is_null() || !check.original_result_hash.is_null() {
            return Err("non-mutating control hashes must be null".to_owned());
        }
        return Ok(());
    }
    if check.is_request {
        if !check.original_result_hash.is_null() {
            return Err("mutating request original_result_hash must be null".to_owned());
        }
        let expected = control_content_hash(&envelope)
            .ok_or_else(|| "mutating request fingerprint is not computable".to_owned())?;
        if check.request_fingerprint.as_str() != Some(expected.as_str()) {
            return Err("mutating request_fingerprint does not match canonical bytes".to_owned());
        }
        return Ok(());
    }
    let expected = control_content_hash(&envelope)
        .ok_or_else(|| "mutating result hash is not computable".to_owned())?;
    if check.original_result_hash.as_str() != Some(expected.as_str()) {
        return Err("mutating original_result_hash does not match canonical bytes".to_owned());
    }
    if check
        .request_fingerprint
        .as_str()
        .is_none_or(|value| value.len() != 64)
    {
        return Err("mutating result request_fingerprint must be a SHA-256 hex digest".to_owned());
    }
    Ok(())
}

fn require_null_revision<E>(value: &Value) -> Result<(), E>
where
    E: serde::de::Error,
{
    if value.is_null() {
        Ok(())
    } else {
        Err(E::custom(
            "expected_revision must be null for this operation",
        ))
    }
}

fn require_revision<E>(value: &Value) -> Result<u64, E>
where
    E: serde::de::Error,
{
    value
        .as_u64()
        .ok_or_else(|| E::custom("expected_revision must be an integer >= 0"))
}

fn parse_canonical_uuid_v4<E>(input: &str, field: &str) -> Result<Uuid, E>
where
    E: serde::de::Error,
{
    let parsed = Uuid::parse_str(input)
        .map_err(|_| E::custom(format_args!("{field} must be a canonical UUID v4")))?;
    if parsed.get_version() != Some(Version::Random) || parsed.to_string() != input {
        return Err(E::custom(format_args!(
            "{field} must be a canonical UUID v4"
        )));
    }
    Ok(parsed)
}

fn parse_contract_timestamp<E>(input: &str) -> Result<DateTime<Utc>, E>
where
    E: serde::de::Error,
{
    let bytes = input.as_bytes();
    let fixed_digits = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
    let fixed_shape = bytes.len() >= 20
        && fixed_digits
            .iter()
            .all(|index| bytes.get(*index).is_some_and(u8::is_ascii_digit))
        && bytes.get(4) == Some(&b'-')
        && bytes.get(7) == Some(&b'-')
        && bytes.get(10) == Some(&b'T')
        && bytes.get(13) == Some(&b':')
        && bytes.get(16) == Some(&b':');
    if !fixed_shape {
        return Err(E::custom(
            "deadline_at must use the mission timestamp lexical form",
        ));
    }

    let mut suffix_index = 19;
    if bytes.get(suffix_index) == Some(&b'.') {
        suffix_index += 1;
        let fraction_start = suffix_index;
        while bytes.get(suffix_index).is_some_and(u8::is_ascii_digit) {
            suffix_index += 1;
        }
        if suffix_index == fraction_start || suffix_index - fraction_start > 9 {
            return Err(E::custom(
                "deadline_at fractional seconds must contain 1..=9 digits",
            ));
        }
    }
    let valid_suffix = bytes.get(suffix_index..) == Some(b"Z")
        || (bytes.len() == suffix_index + 6
            && matches!(bytes.get(suffix_index), Some(b'+' | b'-'))
            && bytes
                .get(suffix_index + 1..suffix_index + 3)
                .is_some_and(|part| part.iter().all(u8::is_ascii_digit))
            && bytes.get(suffix_index + 3) == Some(&b':')
            && bytes
                .get(suffix_index + 4..suffix_index + 6)
                .is_some_and(|part| part.iter().all(u8::is_ascii_digit)));
    if !valid_suffix {
        return Err(E::custom(
            "deadline_at must end in Z or a numeric UTC offset",
        ));
    }

    DateTime::parse_from_rfc3339(input)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|_| E::custom("deadline_at is not a valid RFC 3339 timestamp"))
}

fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn validate_create_body(body: &MissionCreateBody) -> Result<(), String> {
    if body.goal.trim().is_empty() || body.goal.chars().count() > 8192 {
        return Err("goal must contain 1..=8192 characters".to_owned());
    }
    if !valid_opaque_id(&body.policy_id) {
        return Err("policy_id is not a valid opaque id".to_owned());
    }
    Ok(())
}

fn validate_list_body(body: &MissionListBody) -> Result<(), String> {
    if body.phases.len() > 10 {
        return Err("phases must contain at most 10 values".to_owned());
    }
    for (index, phase) in body.phases.iter().enumerate() {
        if body.phases[..index].contains(phase) {
            return Err("phases must be unique".to_owned());
        }
    }
    if !(1..=200).contains(&body.limit) {
        return Err("limit must be between 1 and 200".to_owned());
    }
    validate_cursor(body.cursor.as_deref())
}

fn validate_cursor(cursor: Option<&str>) -> Result<(), String> {
    if cursor.is_some_and(|value| value.is_empty() || value.chars().count() > 1024) {
        return Err("cursor must contain 1..=1024 characters".to_owned());
    }
    Ok(())
}

fn validate_idempotency_key(value: &str) -> Result<(), String> {
    if !(16..=256).contains(&value.len())
        || !value.bytes().enumerate().all(|(index, byte)| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' => true,
            b'.' | b'_' | b':' | b'-' => index > 0,
            _ => false,
        })
    {
        return Err("idempotency_key does not match the mission contract".to_owned());
    }
    Ok(())
}

fn valid_opaque_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.bytes().enumerate().all(|(index, byte)| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' => true,
            b'.' | b'_' | b':' | b'/' | b'-' => index > 0,
            _ => false,
        })
}

impl fmt::Display for MissionControlRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.operation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST_ID: &str = "11111111-1111-4111-8111-111111111111";

    fn stamp_request_hashes(mut value: Value, mutating: bool) -> Value {
        if mutating {
            let fingerprint = control_content_hash(&value).expect("hash");
            value["request_fingerprint"] = Value::String(fingerprint);
            value["original_result_hash"] = Value::Null;
        } else {
            value["request_fingerprint"] = Value::Null;
            value["original_result_hash"] = Value::Null;
        }
        value
    }

    #[test]
    fn create_request_requires_explicit_null_revision_and_deadline() {
        let base = serde_json::json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": REQUEST_ID,
            "idempotency_key": "mission-create:11111111-1111-4111-8111-111111111111",
            "expected_revision": null,
            "operation": "mission.create",
            "body": {
                "goal": "persist a mission",
                "policy_id": "code-only",
                "deadline_at": null,
                "recovery_policy": "ask_operator"
            }
        });
        let base = stamp_request_hashes(base, true);
        assert!(serde_json::from_value::<MissionControlRequest>(base.clone()).is_ok());

        let mut missing_revision = base.clone();
        missing_revision
            .as_object_mut()
            .expect("object")
            .remove("expected_revision");
        assert!(serde_json::from_value::<MissionControlRequest>(missing_revision).is_err());

        let mut missing_deadline = base;
        missing_deadline["body"]
            .as_object_mut()
            .expect("body object")
            .remove("deadline_at");
        assert!(serde_json::from_value::<MissionControlRequest>(missing_deadline).is_err());
    }

    #[test]
    fn request_rejects_noncanonical_uuid_and_unknown_body_fields() {
        let noncanonical = serde_json::json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": "{11111111-1111-4111-8111-111111111111}",
            "idempotency_key": null,
            "expected_revision": null,
            "operation": "mission.show",
            "body": { "mission_id": REQUEST_ID }
        });
        let noncanonical = stamp_request_hashes(noncanonical, false);
        assert!(serde_json::from_value::<MissionControlRequest>(noncanonical).is_err());

        let unknown = serde_json::json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": REQUEST_ID,
            "idempotency_key": null,
            "expected_revision": null,
            "operation": "mission.show",
            "body": { "mission_id": REQUEST_ID, "role": "spoofed" }
        });
        let unknown = stamp_request_hashes(unknown, false);
        assert!(serde_json::from_value::<MissionControlRequest>(unknown).is_err());
    }

    #[test]
    fn create_request_enforces_timestamp_lexical_form() {
        let request = |deadline_at: &str| {
            stamp_request_hashes(
                serde_json::json!({
                    "control_schema": MISSION_CONTROL_SCHEMA,
                    "kind": "request",
                    "request_id": REQUEST_ID,
                    "idempotency_key": "mission-create:11111111-1111-4111-8111-111111111111",
                    "expected_revision": null,
                    "operation": "mission.create",
                    "body": {
                        "goal": "persist a mission",
                        "policy_id": "code-only",
                        "deadline_at": deadline_at,
                        "recovery_policy": "ask_operator"
                    }
                }),
                true,
            )
        };

        assert!(serde_json::from_value::<MissionControlRequest>(request(
            "2026-08-18T12:30:45.123456789-04:00"
        ))
        .is_ok());
        for invalid in [
            "2026-08-18t12:30:45Z",
            "2026-08-18 12:30:45Z",
            "2026-08-18T12:30:45z",
            "2026-08-18T12:30:45.1234567890Z",
        ] {
            assert!(
                serde_json::from_value::<MissionControlRequest>(request(invalid)).is_err(),
                "{invalid} must be rejected"
            );
        }
    }

    #[test]
    fn launch_observe_close_requests_round_trip() {
        let launch = serde_json::json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": "55555555-5555-4555-8555-555555555555",
            "idempotency_key": "mission-launch-example-0001",
            "expected_revision": 0,
            "operation": "mission.launch",
            "body": {
                "mission_id": REQUEST_ID,
                "workspace": "/tmp/lanyte-mission-workspace",
                "binary": null
            }
        });
        let launch = stamp_request_hashes(launch, true);
        let parsed = serde_json::from_value::<MissionControlRequest>(launch.clone()).unwrap();
        assert_eq!(parsed.operation(), "mission.launch");
        assert_eq!(serde_json::to_value(&parsed).unwrap(), launch);

        let observe = serde_json::json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": "66666666-6666-4666-8666-666666666666",
            "idempotency_key": null,
            "expected_revision": null,
            "operation": "mission.observe",
            "body": { "mission_id": REQUEST_ID }
        });
        let observe = stamp_request_hashes(observe, false);
        assert_eq!(
            serde_json::from_value::<MissionControlRequest>(observe)
                .unwrap()
                .operation(),
            "mission.observe"
        );

        let close = serde_json::json!({
            "control_schema": MISSION_CONTROL_SCHEMA,
            "kind": "request",
            "request_id": "77777777-7777-4777-8777-777777777777",
            "idempotency_key": "mission-close-example-0001",
            "expected_revision": 1,
            "operation": "mission.close",
            "body": { "mission_id": REQUEST_ID }
        });
        let close = stamp_request_hashes(close, true);
        assert_eq!(
            serde_json::from_value::<MissionControlRequest>(close)
                .unwrap()
                .operation(),
            "mission.close"
        );
    }

    #[test]
    fn cancel_request_and_result_round_trip_hashes() {
        let request = stamp_request_hashes(
            serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": "44444444-4444-4444-8444-444444444444",
                "idempotency_key": "mission-cancel-example-0001",
                "expected_revision": 1,
                "operation": "mission.cancel",
                "body": { "mission_id": REQUEST_ID }
            }),
            true,
        );
        let parsed = serde_json::from_value::<MissionControlRequest>(request.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), request);

        let mut extra = request.clone();
        extra["spoofed_identity"] = Value::String("nope".to_owned());
        extra = stamp_request_hashes(extra, true);
        assert!(serde_json::from_value::<MissionControlRequest>(extra).is_err());

        let mut wrong_schema = request;
        wrong_schema["control_schema"] = Value::String(
            "https://schemas.3leaps.dev/agentic/mission/v0.1/not-control.schema.json".to_owned(),
        );
        wrong_schema = stamp_request_hashes(wrong_schema, true);
        assert!(serde_json::from_value::<MissionControlRequest>(wrong_schema).is_err());
    }

    fn sample_record() -> MissionRecord {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/semantic/conforming/create-only-wave1.json"
        ))
        .unwrap();
        serde_json::from_value(fixture["mission"].clone()).unwrap()
    }

    fn restamp_result(mut value: Value) -> Value {
        if let Some(obj) = value.as_object_mut() {
            obj.remove("original_result_hash");
        }
        let hash = control_content_hash(&value).expect("hash");
        value["original_result_hash"] = Value::String(hash);
        value
    }

    #[test]
    fn launch_result_carries_accepted_request_fingerprint_for_explicit_binary() {
        let request = stamp_request_hashes(
            serde_json::json!({
                "control_schema": MISSION_CONTROL_SCHEMA,
                "kind": "request",
                "request_id": "55555555-5555-4555-8555-555555555555",
                "idempotency_key": "mission-launch-example-0001",
                "expected_revision": 0,
                "operation": "mission.launch",
                "body": {
                    "mission_id": REQUEST_ID,
                    "workspace": "/tmp/lanyte-mission-workspace",
                    "binary": "/tmp/fake-codex"
                }
            }),
            true,
        );
        let fingerprint = request["request_fingerprint"].as_str().unwrap().to_owned();
        let result = MissionControlResult::launch(
            Uuid::parse_str("55555555-5555-4555-8555-555555555555").unwrap(),
            "mission-launch-example-0001".to_owned(),
            0,
            sample_record(),
        )
        .unwrap()
        .bind_request_fingerprint(fingerprint.clone());
        let encoded = serde_json::to_value(&result).unwrap();
        assert_eq!(
            encoded["request_fingerprint"].as_str(),
            Some(fingerprint.as_str())
        );
        let round = serde_json::from_value::<MissionControlResult>(encoded.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(&round).unwrap()["request_fingerprint"],
            encoded["request_fingerprint"]
        );
    }

    #[test]
    fn frozen_v0_envelopes_parse_without_hash_fields() {
        let request = serde_json::json!({
            "control_schema": MISSION_CONTROL_SCHEMA_V0,
            "kind": "request",
            "request_id": REQUEST_ID,
            "idempotency_key": null,
            "expected_revision": null,
            "operation": "mission.show",
            "body": { "mission_id": REQUEST_ID }
        });
        assert!(serde_json::from_value::<MissionControlRequest>(request).is_ok());

        let result = serde_json::json!({
            "control_schema": MISSION_CONTROL_SCHEMA_V0,
            "kind": "result",
            "request_id": REQUEST_ID,
            "idempotency_key": null,
            "expected_revision": null,
            "operation": "mission.show",
            "body": { "record": sample_record() }
        });
        assert!(serde_json::from_value::<MissionControlResult>(result).is_ok());
    }

    #[test]
    fn close_result_rejects_rehashed_invalid_bodies() {
        let result = MissionControlResult::close(
            Uuid::parse_str("77777777-7777-4777-8777-777777777777").unwrap(),
            "mission-close-example-0001".to_owned(),
            1,
            Uuid::parse_str(REQUEST_ID).unwrap(),
            Uuid::parse_str("88888888-8888-4888-8888-888888888888").unwrap(),
        )
        .unwrap()
        .bind_request_fingerprint("a".repeat(64));
        let valid = serde_json::to_value(&result).unwrap();
        assert!(serde_json::from_value::<MissionControlResult>(valid.clone()).is_ok());

        let mut missing = valid.clone();
        missing["body"].as_object_mut().unwrap().remove("closed");
        assert!(serde_json::from_value::<MissionControlResult>(restamp_result(missing)).is_err());

        let mut false_closed = valid.clone();
        false_closed["body"]["closed"] = Value::Bool(false);
        assert!(
            serde_json::from_value::<MissionControlResult>(restamp_result(false_closed)).is_err()
        );

        let mut extra = valid;
        extra["body"]["spoofed"] = Value::Bool(true);
        assert!(serde_json::from_value::<MissionControlResult>(restamp_result(extra)).is_err());
    }
}
