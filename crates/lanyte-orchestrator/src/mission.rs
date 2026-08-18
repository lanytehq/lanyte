use std::sync::{Arc, Mutex};

use chrono::{SecondsFormat, Utc};
use lanyte_attest::{verify as attest_verify, SessionState};
use lanyte_mission::{
    AttestationRef, BudgetPolicy, EventSource, EventSourceKind, LeasePolicy, LifecycleEvent,
    LifecyclePayload, MissionControlRequest, MissionControlResult, MissionPhase, MissionRecord,
    ObservationLevel, OperatingRole, Principal, PrincipalKind, Validate, LIFECYCLE_EVENT_SCHEMA,
    MISSION_RECORD_SCHEMA,
};
use lanyte_state::{
    MissionCreateIdempotency, MissionListFilter, NewMissionProjectionReceipt, StateError,
    StateStore,
};
use lanyte_telemetry::AuditEnvelopeRef;
use serde::Serialize;
use uuid::{Uuid, Version};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSession {
    pub issuer: String,
    pub subject: String,
    pub session_id: Uuid,
    pub role: String,
    pub scope: String,
    pub jti: Uuid,
    pub context_sha256: String,
    pub token_sha256: String,
    pub verification_policy_sha256: String,
    pub trust_ref: String,
}

pub trait SessionVerifier: Send + Sync {
    fn verify(&self, token: &str) -> Result<VerifiedSession, String>;
}

#[derive(Debug, Default)]
pub struct AttestationSessionVerifier;

impl SessionVerifier for AttestationSessionVerifier {
    fn verify(&self, token: &str) -> Result<VerifiedSession, String> {
        let public_key = attest_verify::load_public_key(None)
            .map_err(|err| format!("failed to load attestation trust root: {err}"))?;
        let policy = attest_verify::policy_from_current_time()
            .map_err(|err| format!("failed to build attestation policy: {err}"))?;
        let claims = attest_verify::validate_token(token, &public_key, &policy)
            .map_err(|err| format!("attestation validation failed: {err}"))?;
        let status = attest_verify::check_revocation_at_time(&claims.jti, None, policy.now)
            .map_err(|err| format!("attestation revocation check failed: {err}"))?;
        if status != SessionState::Active {
            return Err("attestation session is not active".to_owned());
        }

        let session_id = canonical_claim_uuid(&claims.sid, "sid")?;
        let jti = canonical_claim_uuid(&claims.jti, "jti")?;
        let context_sha256 = strip_sha256_prefix(&claims.ctx_hash, "ctx_hash")?;
        let token_sha256 = strip_sha256_prefix(&lanyte_attest::token_hash(token), "token hash")?;
        let policy_material = serde_json::json!({
            "algorithm": "EdDSA",
            "checks": ["signature", "issuer", "time", "revocation"],
            "expected_issuer": policy.expected_issuer,
        });
        let verification_policy_sha256 = hash_json(&policy_material)?;
        let trust_ref = format!("lanyte-attest://{}/sessions/{}", claims.iss, claims.jti);

        Ok(VerifiedSession {
            issuer: claims.iss,
            subject: claims.sub,
            session_id,
            role: claims.role,
            scope: claims.scope,
            jti,
            context_sha256,
            token_sha256,
            verification_policy_sha256,
            trust_ref,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionCommandErrorCode {
    InvalidArgs,
    PermissionDenied,
    InternalError,
}

impl MissionCommandErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgs => "invalid_args",
            Self::PermissionDenied => "permission_denied",
            Self::InternalError => "internal_error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionCommandError {
    pub code: MissionCommandErrorCode,
    pub message: String,
}

impl MissionCommandError {
    fn invalid_args(message: impl Into<String>) -> Self {
        Self {
            code: MissionCommandErrorCode::InvalidArgs,
            message: message.into(),
        }
    }

    fn permission_denied() -> Self {
        Self {
            code: MissionCommandErrorCode::PermissionDenied,
            message: "caller attestation or mission visibility denied".to_owned(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            code: MissionCommandErrorCode::InternalError,
            message: message.into(),
        }
    }
}

#[derive(Clone)]
pub struct MissionService {
    store: Arc<Mutex<StateStore>>,
    verifier: Arc<dyn SessionVerifier>,
}

impl MissionService {
    #[must_use]
    pub fn new(store: Arc<Mutex<StateStore>>, verifier: Arc<dyn SessionVerifier>) -> Self {
        Self { store, verifier }
    }

    #[must_use]
    pub fn production(store: Arc<Mutex<StateStore>>) -> Self {
        Self::new(store, Arc::new(AttestationSessionVerifier))
    }

    pub fn handle(
        &self,
        request: MissionControlRequest,
        token: Option<&str>,
    ) -> Result<MissionControlResult, MissionCommandError> {
        let token = token
            .filter(|value| !value.is_empty())
            .ok_or_else(MissionCommandError::permission_denied)?;
        let caller = self.verifier.verify(token).map_err(|err| {
            tracing::warn!(error = %err, "mission caller attestation denied");
            MissionCommandError::permission_denied()
        })?;

        match request {
            MissionControlRequest::Create {
                request_id,
                idempotency_key,
                body,
            } => {
                let request_fingerprint =
                    create_fingerprint(&caller, &body).map_err(MissionCommandError::internal)?;
                let (mission, receipt) =
                    build_created_mission(&caller, body).map_err(MissionCommandError::internal)?;
                let outcome = self
                    .store
                    .lock()
                    .map_err(|_| MissionCommandError::internal("mission store lock poisoned"))?
                    .create_mission_idempotent(
                        mission,
                        receipt,
                        MissionCreateIdempotency {
                            key: idempotency_key.clone(),
                            request_fingerprint,
                        },
                    )
                    .map_err(map_state_error)?;
                let record = outcome.write.projection.mission;
                MissionControlResult::create(request_id, idempotency_key, record)
                    .map_err(MissionCommandError::internal)
            }
            MissionControlRequest::Show { request_id, body } => {
                let projection = self
                    .store
                    .lock()
                    .map_err(|_| MissionCommandError::internal("mission store lock poisoned"))?
                    .mission(&body.mission_id.to_string())
                    .map_err(map_state_error)?
                    .filter(|projection| visible_to(&projection.mission, &caller))
                    .ok_or_else(MissionCommandError::permission_denied)?;
                MissionControlResult::show(request_id, projection.mission)
                    .map_err(MissionCommandError::internal)
            }
            MissionControlRequest::List { request_id, body } => {
                let page = self
                    .store
                    .lock()
                    .map_err(|_| MissionCommandError::internal("mission store lock poisoned"))?
                    .list_missions(MissionListFilter {
                        phases: body.phases,
                        operating_role: caller.role,
                        operating_scope: caller.scope,
                        limit: body.limit,
                        cursor: body.cursor,
                    })
                    .map_err(map_state_error)?;
                MissionControlResult::list(
                    request_id,
                    page.projections
                        .into_iter()
                        .map(|projection| projection.mission)
                        .collect(),
                    page.next_cursor,
                )
                .map_err(MissionCommandError::internal)
            }
        }
    }
}

fn build_created_mission(
    caller: &VerifiedSession,
    body: lanyte_mission::MissionCreateBody,
) -> Result<(MissionRecord, NewMissionProjectionReceipt), String> {
    let now = Utc::now();
    let mission_id = Uuid::new_v4();
    let principal = Principal {
        kind: PrincipalKind::AttestedSession,
        subject: caller.subject.clone(),
        role: Some(caller.role.clone()),
        scope: Some(caller.scope.clone()),
        attestation: Some(AttestationRef {
            issuer: caller.issuer.clone(),
            session_id: caller.session_id,
            jti: caller.jti,
            context_sha256: caller.context_sha256.clone(),
            token_sha256: caller.token_sha256.clone(),
            verification_policy_sha256: caller.verification_policy_sha256.clone(),
            trust_ref: caller.trust_ref.clone(),
        }),
    };
    let mission = MissionRecord {
        mission_schema: MISSION_RECORD_SCHEMA.to_owned(),
        mission_id,
        revision: 0,
        goal: body.goal,
        policy_id: body.policy_id,
        created_at: now,
        updated_at: now,
        initiator: principal,
        authorizer: None,
        authorization_ref: None,
        supervisor: Principal {
            kind: PrincipalKind::Service,
            subject: "lanyte".to_owned(),
            role: None,
            scope: None,
            attestation: None,
        },
        operating_role: OperatingRole {
            role: caller.role.clone(),
            scope: caller.scope.clone(),
        },
        phase: MissionPhase::Created,
        terminal_reason: None,
        deadline_at: body.deadline_at,
        lease_policy: LeasePolicy {
            enabled: false,
            lease_seconds: None,
            deadman_seconds: None,
        },
        budget_policy: BudgetPolicy {
            wall_clock_seconds: None,
            token_limit: None,
            cost_micros: None,
            action_limit: None,
        },
        harness_selection: None,
        recovery_policy: body.recovery_policy,
        recovery_point_ref: None,
        attempts: Vec::new(),
        current_attempt_id: None,
        evidence_chain_id: mission_id,
        terminal_entry_hash: None,
    };
    mission.validate().map_err(|err| err.to_string())?;

    let event_id = Uuid::new_v4();
    let mut event = LifecycleEvent {
        event_schema: LIFECYCLE_EVENT_SCHEMA.to_owned(),
        event_id,
        mission_id,
        sequence: 1,
        previous_entry_hash: None,
        entry_hash: "0".repeat(64),
        occurred_at: now,
        recorded_at: now,
        event_type: "mission_created".to_owned(),
        source: EventSource {
            kind: EventSourceKind::VerifiedAttestation,
            subject: caller.subject.clone(),
            producer_version: env!("CARGO_PKG_VERSION").to_owned(),
            assurance: ObservationLevel::ResourceAttested,
            evidence_ref: Some(caller.trust_ref.clone()),
        },
        payload: LifecyclePayload::MissionCreated { revision: 0 },
    };
    let mut event_material = serde_json::to_value(&event).map_err(|err| err.to_string())?;
    event_material
        .as_object_mut()
        .ok_or_else(|| "lifecycle event did not serialize as an object".to_owned())?
        .remove("entry_hash");
    event.entry_hash = hash_json(&event_material)?;
    event.validate().map_err(|err| err.to_string())?;

    Ok((
        mission,
        NewMissionProjectionReceipt {
            event,
            envelope: AuditEnvelopeRef {
                action_id: Some(event_id.to_string()),
                correlation_id: Some(mission_id.to_string()),
                trust_ref: Some(caller.trust_ref.clone()),
                ..AuditEnvelopeRef::default()
            },
            verification: Some(serde_json::json!({
                "attestation_issuer": caller.issuer,
                "attestation_session_id": caller.session_id,
                "attestation_jti": caller.jti,
                "token_sha256": caller.token_sha256,
                "verification_policy_sha256": caller.verification_policy_sha256,
                "verified_at": now.to_rfc3339_opts(SecondsFormat::Millis, true),
            })),
        },
    ))
}

fn create_fingerprint(
    caller: &VerifiedSession,
    body: &lanyte_mission::MissionCreateBody,
) -> Result<String, String> {
    hash_json(&serde_json::json!({
        "operation": "mission.create",
        "caller": {
            "issuer": caller.issuer,
            "subject": caller.subject,
            "role": caller.role,
            "scope": caller.scope,
        },
        "body": body,
    }))
}

fn visible_to(mission: &MissionRecord, caller: &VerifiedSession) -> bool {
    mission.operating_role.role == caller.role && mission.operating_role.scope == caller.scope
}

fn map_state_error(error: StateError) -> MissionCommandError {
    match error {
        StateError::MissionIdempotencyConflict { .. } => {
            MissionCommandError::invalid_args("idempotency key conflicts with a prior create")
        }
        other => MissionCommandError::internal(format!("mission state operation failed: {other}")),
    }
}

fn canonical_claim_uuid(input: &str, field: &str) -> Result<Uuid, String> {
    let parsed = Uuid::parse_str(input).map_err(|_| format!("{field} must be a UUID v4"))?;
    if parsed.get_version() != Some(Version::Random) || parsed.to_string() != input {
        return Err(format!("{field} must be a canonical UUID v4"));
    }
    Ok(parsed)
}

fn strip_sha256_prefix(input: &str, field: &str) -> Result<String, String> {
    let digest = input
        .strip_prefix("sha256:")
        .ok_or_else(|| format!("{field} must use sha256 prefix"))?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must contain lowercase SHA-256"));
    }
    Ok(digest.to_owned())
}

fn hash_json(value: &impl Serialize) -> Result<String, String> {
    let encoded = serde_json::to_string(value).map_err(|err| err.to_string())?;
    strip_sha256_prefix(&lanyte_attest::token_hash(&encoded), "JSON hash")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use lanyte_mission::{MissionCreateBody, MissionListBody, RecoveryPolicy};
    use lanyte_state::StatePaths;

    use super::*;

    const VALID_TOKEN: &str = "valid-session-secret";

    struct FakeVerifier;

    impl SessionVerifier for FakeVerifier {
        fn verify(&self, token: &str) -> Result<VerifiedSession, String> {
            match token {
                VALID_TOKEN => Ok(verified_session("operator", "lanytehq", 1)),
                "renewed-session-secret" => Ok(verified_session("operator", "lanytehq", 2)),
                "other-session-secret" => Ok(verified_session("reviewer", "lanytehq", 3)),
                _ => Err("invalid, expired, revoked, or wrong-issuer attestation".to_owned()),
            }
        }
    }

    struct TempStateRoot(PathBuf);

    impl TempStateRoot {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("lanyte-mission-service-{}", Uuid::new_v4())))
        }

        fn paths(&self) -> StatePaths {
            StatePaths::new(&self.0)
        }
    }

    impl Drop for TempStateRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn verified_session(role: &str, scope: &str, suffix: u8) -> VerifiedSession {
        let session_id = Uuid::parse_str(&format!("00000000-0000-4000-8000-{suffix:012x}"))
            .expect("session UUID");
        let jti =
            Uuid::parse_str(&format!("00000000-0000-4000-9000-{suffix:012x}")).expect("jti UUID");
        VerifiedSession {
            issuer: "lanyte-attest".to_owned(),
            subject: format!("{role}-subject"),
            session_id,
            role: role.to_owned(),
            scope: scope.to_owned(),
            jti,
            context_sha256: "1".repeat(64),
            token_sha256: "2".repeat(64),
            verification_policy_sha256: "3".repeat(64),
            trust_ref: format!("lanyte-attest://lanyte-attest/sessions/{jti}"),
        }
    }

    fn test_service(store: Arc<Mutex<StateStore>>) -> MissionService {
        MissionService::new(store, Arc::new(FakeVerifier))
    }

    fn create_request(request_id: Uuid, key: &str) -> MissionControlRequest {
        MissionControlRequest::create(
            request_id,
            key.to_owned(),
            MissionCreateBody {
                goal: "Keep the local daemon record durable".to_owned(),
                policy_id: "policy.local".to_owned(),
                deadline_at: None,
                recovery_policy: RecoveryPolicy::AskOperator,
            },
        )
        .expect("valid create request")
    }

    fn created_record(result: MissionControlResult) -> MissionRecord {
        match result {
            MissionControlResult::Record {
                operation, record, ..
            } => {
                assert_eq!(operation, "mission.create");
                *record
            }
            MissionControlResult::List { .. } => panic!("expected record result"),
        }
    }

    #[test]
    fn missing_or_rejected_attestation_fails_closed_without_persisting() {
        let root = TempStateRoot::new();
        let paths = root.paths();
        let store = Arc::new(Mutex::new(
            StateStore::open(paths.clone()).expect("state store"),
        ));
        let service = test_service(Arc::clone(&store));
        let request = create_request(Uuid::new_v4(), "create:fail-closed");

        for token in [
            None,
            Some("malformed"),
            Some("expired"),
            Some("revoked"),
            Some("wrong-issuer"),
        ] {
            let error = service
                .handle(request.clone(), token)
                .expect_err("unverified caller must be denied");
            assert_eq!(error.code, MissionCommandErrorCode::PermissionDenied);
            assert_eq!(
                error.message,
                "caller attestation or mission visibility denied"
            );
        }

        let page = store
            .lock()
            .expect("store lock")
            .list_missions(MissionListFilter {
                operating_role: "operator".to_owned(),
                operating_scope: "lanytehq".to_owned(),
                phases: Vec::new(),
                limit: 10,
                cursor: None,
            })
            .expect("mission list");
        assert!(page.projections.is_empty());
    }

    #[test]
    fn create_replay_visibility_and_reopen_preserve_the_same_record() {
        let root = TempStateRoot::new();
        let paths = root.paths();
        let store = Arc::new(Mutex::new(
            StateStore::open(paths.clone()).expect("state store"),
        ));
        let service = test_service(Arc::clone(&store));

        let created = created_record(
            service
                .handle(
                    create_request(Uuid::new_v4(), "create:persistent"),
                    Some(VALID_TOKEN),
                )
                .expect("mission create"),
        );
        assert_eq!(created.initiator.role.as_deref(), Some("operator"));
        assert_eq!(created.operating_role.role, "operator");
        assert_eq!(created.supervisor.kind, PrincipalKind::Service);

        let replayed = created_record(
            service
                .handle(
                    create_request(Uuid::new_v4(), "create:persistent"),
                    Some("renewed-session-secret"),
                )
                .expect("idempotent replay after attestation renewal"),
        );
        assert_eq!(replayed.mission_id, created.mission_id);

        let list = service
            .handle(
                MissionControlRequest::list(
                    Uuid::new_v4(),
                    MissionListBody {
                        phases: vec![MissionPhase::Created],
                        limit: 10,
                        cursor: None,
                    },
                )
                .expect("valid list"),
                Some(VALID_TOKEN),
            )
            .expect("caller list");
        match list {
            MissionControlResult::List { records, .. } => {
                assert_eq!(records.len(), 1);
                assert_eq!(records[0].mission_id, created.mission_id);
            }
            MissionControlResult::Record { .. } => panic!("expected list result"),
        }

        let hidden = service
            .handle(
                MissionControlRequest::show(Uuid::new_v4(), created.mission_id)
                    .expect("valid show"),
                Some("other-session-secret"),
            )
            .expect_err("different caller must not see mission");
        assert_eq!(hidden.code, MissionCommandErrorCode::PermissionDenied);

        drop(service);
        drop(store);

        let reopened = Arc::new(Mutex::new(
            StateStore::open(paths.clone()).expect("reopen state store"),
        ));
        let shown = test_service(reopened)
            .handle(
                MissionControlRequest::show(Uuid::new_v4(), created.mission_id)
                    .expect("valid show"),
                Some(VALID_TOKEN),
            )
            .expect("mission survives client/service restart");
        match shown {
            MissionControlResult::Record {
                operation, record, ..
            } => {
                assert_eq!(operation, "mission.show");
                assert_eq!(*record, created);
            }
            MissionControlResult::List { .. } => panic!("expected record result"),
        }

        let database = std::fs::read(paths.hot_db_path()).expect("read hot database");
        assert!(
            !database
                .windows(VALID_TOKEN.len())
                .any(|window| window == VALID_TOKEN.as_bytes()),
            "raw session token must not be persisted"
        );
    }
}
