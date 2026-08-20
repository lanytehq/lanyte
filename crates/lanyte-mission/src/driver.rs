use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::DriverCapabilityReport;

/// Stable identity of a harness driver implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverDescriptor {
    pub driver_id: String,
    pub driver_version: String,
    pub harness_kind: String,
}

/// Normalized observation from a harness session. Provider-native payloads stay
/// in `detail` and never become authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum NormalizedHarnessEvent {
    Started {
        occurred_at: DateTime<Utc>,
        attempt_id: Uuid,
        harness_session_id: String,
        detail: Option<String>,
    },
    ToolProposed {
        occurred_at: DateTime<Utc>,
        attempt_id: Uuid,
        tool: String,
        detail: Option<String>,
    },
    Exited {
        occurred_at: DateTime<Utc>,
        attempt_id: Uuid,
        success: bool,
        detail: Option<String>,
    },
    TurnProgress {
        occurred_at: DateTime<Utc>,
        attempt_id: Uuid,
        thread_id: Option<String>,
        turn_id: String,
        status: String,
    },
}

/// Domain boundary shared by future harness integrations.
///
/// Identity and capability evidence stay on the object-safe surface. Session
/// create/identify/observe/close live on the concrete Wave 2 driver so async
/// and process ownership can be reviewed against Codex App Server.
pub trait HarnessDriver: Send + Sync {
    fn descriptor(&self) -> DriverDescriptor;

    fn capabilities(&self) -> DriverCapabilityReport;
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use crate::{
        CapabilityFidelity, CapabilityName, DriverAvailability, DriverCapability,
        DriverValidityCondition, EnforcementLevel, ObservationLevel, ReplaySupport,
        DRIVER_CAPABILITIES_SCHEMA,
    };

    use super::*;

    struct TestDriver;

    impl HarnessDriver for TestDriver {
        fn descriptor(&self) -> DriverDescriptor {
            DriverDescriptor {
                driver_id: "driver.test".to_owned(),
                driver_version: "0.1.0".to_owned(),
                harness_kind: "test".to_owned(),
            }
        }

        fn capabilities(&self) -> DriverCapabilityReport {
            DriverCapabilityReport {
                capabilities_schema: DRIVER_CAPABILITIES_SCHEMA.to_owned(),
                report_id: crate::invariant::fixtures::uuid(6),
                driver_id: "driver.test".to_owned(),
                driver_version: "0.1.0".to_owned(),
                harness_kind: "test".to_owned(),
                observed_at: Utc.with_ymd_and_hms(2026, 8, 17, 14, 0, 0).unwrap(),
                expires_at: Utc.with_ymd_and_hms(2026, 8, 18, 14, 0, 0).unwrap(),
                availability: DriverAvailability::Available,
                capabilities: vec![DriverCapability {
                    name: CapabilityName::Create,
                    fidelity: CapabilityFidelity::Native,
                    observation: ObservationLevel::KernelObserved,
                    enforcement: EnforcementLevel::ProtocolConfirmed,
                    replay: ReplaySupport::Idempotent,
                    limitation: None,
                    evidence_ref: Some("probes/create".to_owned()),
                }],
                validity_condition: DriverValidityCondition {
                    kind: "executable-version-platform-match".to_owned(),
                    executable_version: "0.1.0".to_owned(),
                    executable_sha256: "e".repeat(64),
                    configuration_sha256: "f".repeat(64),
                    platform: "test-platform".to_owned(),
                    probe_ref: "probes/test-driver".to_owned(),
                },
                evidence_ref: "capability-reports/test-driver".to_owned(),
            }
        }
    }

    #[test]
    fn trait_exposes_identity_and_capability_evidence() {
        let driver: &dyn HarnessDriver = &TestDriver;
        let descriptor = driver.descriptor();
        assert_eq!(descriptor.driver_id, "driver.test");
        driver
            .capabilities()
            .require_usable_at(
                Utc.with_ymd_and_hms(2026, 8, 17, 15, 0, 0).unwrap(),
                &descriptor,
                "test-platform",
                &[CapabilityName::Create],
            )
            .unwrap();
    }
}
