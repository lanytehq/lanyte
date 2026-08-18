use std::fmt;

use chrono::{DateTime, Utc};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
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
        let value = match self {
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
        }

        let raw = Raw::deserialize(deserializer)?;
        if raw.control_schema != MISSION_CONTROL_SCHEMA
            && raw.control_schema != MISSION_CONTROL_SCHEMA_V0
        {
            return Err(D::Error::custom("unsupported control_schema"));
        }
        if raw.kind != "request" {
            return Err(D::Error::custom("kind must be request"));
        }
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
    },
    Cancel {
        request_id: Uuid,
        idempotency_key: String,
        expected_revision: u64,
        record: Box<MissionRecord>,
        progress: CancelProgress,
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
}

impl Serialize for MissionControlResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = match self {
            Self::Record {
                request_id,
                operation,
                idempotency_key,
                expected_revision,
                record,
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
        value.serialize(serializer)
    }
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
        assert!(serde_json::from_value::<MissionControlRequest>(unknown).is_err());
    }

    #[test]
    fn create_request_enforces_timestamp_lexical_form() {
        let request = |deadline_at: &str| {
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
            })
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
        assert_eq!(
            serde_json::from_value::<MissionControlRequest>(close)
                .unwrap()
                .operation(),
            "mission.close"
        );
    }
}
