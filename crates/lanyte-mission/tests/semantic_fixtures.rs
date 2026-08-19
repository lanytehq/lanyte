use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use lanyte_mission::{
    semantic_violation_codes_for_fixture, validate_history_with_control, ControlBinding,
    DriverCapabilityReport, LifecycleEvent, MissionRecord,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct HistoryFixture {
    mission: MissionRecord,
    events: Vec<LifecycleEvent>,
    #[serde(default)]
    control_records: Vec<ControlBinding>,
    #[serde(default)]
    driver_capabilities: Vec<DriverCapabilityReport>,
}

#[derive(Deserialize)]
struct Manifest {
    conforming: Vec<String>,
    negative: BTreeMap<String, String>,
}

fn fixtures_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/semantic")
}

fn load_manifest() -> Manifest {
    serde_json::from_str(&fs::read_to_string(fixtures_root().join("manifest.json")).unwrap())
        .expect("manifest")
}

fn json_names(dir: &Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .expect("fixtures dir")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".json"))
        .collect()
}

fn load_history(path: &Path) -> Result<HistoryFixture, String> {
    let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
    serde_json::from_str(&text).map_err(|err| err.to_string())
}

#[test]
fn semantic_manifest_matches_directory_sets() {
    let manifest = load_manifest();
    let conforming: BTreeSet<_> = manifest.conforming.iter().cloned().collect();
    let negative: BTreeSet<_> = manifest.negative.keys().cloned().collect();
    assert_eq!(conforming.len(), 9);
    assert_eq!(negative.len(), 76);
    assert_eq!(json_names(&fixtures_root().join("conforming")), conforming);
    assert_eq!(json_names(&fixtures_root().join("negative")), negative);
}

#[test]
fn conforming_v0_1_histories_validate() {
    let manifest = load_manifest();
    let dir = fixtures_root().join("conforming");
    let mut failures = Vec::new();
    for name in &manifest.conforming {
        match load_history(&dir.join(name)) {
            Ok(fixture) => {
                if let Err(err) = validate_history_with_control(
                    &fixture.mission,
                    &fixture.events,
                    &fixture.control_records,
                ) {
                    failures.push(format!("{name}: {err}"));
                }
                let codes = semantic_violation_codes_for_fixture(
                    &fixture.mission,
                    &fixture.events,
                    &fixture.control_records,
                    &fixture.driver_capabilities,
                );
                if !codes.is_empty() {
                    failures.push(format!("{name}: fixture codes {codes:?}"));
                }
            }
            Err(err) => failures.push(format!("{name}: parse {err}")),
        }
    }
    assert!(
        failures.is_empty(),
        "conforming fixtures failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn negative_v0_1_histories_reject_with_declared_sem() {
    let manifest = load_manifest();
    let dir = fixtures_root().join("negative");
    let mut failures = Vec::new();
    // Incomplete lexical/history shapes that cannot deserialize as
    // MissionRecord/LifecycleEvent. Named allowlist only — not family prefixes.
    const PARSE_EXEMPT: &[&str] = &[
        "attempt-created-ordinal-drift.json",
        "capability-gating.json",
        "forbidden-secret-field.json",
        "illegal-mission-phase-edge.json",
        "noncanonical-mission-id.json",
        "orphan-attempt-lifecycle.json",
        "premature-authorizer.json",
        "recovery-identity-role-mutation.json",
        "sequence-gap.json",
        "stale-generation-fence.json",
        "stale-projection-timestamps.json",
        "successor-without-replaced-predecessor.json",
        "terminal-mission-live-attempt.json",
        "two-live-attempts.json",
        "unauthorized-cancel.json",
    ];
    for (name, expected) in &manifest.negative {
        let text = fs::read_to_string(dir.join(name)).expect("read negative fixture");
        let Ok(_) = serde_json::from_str::<serde_json::Value>(&text) else {
            failures.push(format!("{name}: fixture is not JSON"));
            continue;
        };
        match load_history(&dir.join(name)) {
            Ok(fixture) => {
                let codes = semantic_violation_codes_for_fixture(
                    &fixture.mission,
                    &fixture.events,
                    &fixture.control_records,
                    &fixture.driver_capabilities,
                );
                if codes.is_empty() {
                    failures.push(format!("{name}: accepted (wanted {expected})"));
                } else if !codes.contains(&expected.as_str()) {
                    failures.push(format!("{name}: wanted {expected}, got {codes:?}"));
                }
            }
            Err(err) => {
                if PARSE_EXEMPT.contains(&name.as_str()) {
                    continue;
                }
                failures.push(format!("{name}: parse {err}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "negative fixtures failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn cancel_requested_without_binding_is_sem_a05() {
    let fixture = load_history(&fixtures_root().join("conforming/protocol-confirmed-cancel.json"))
        .expect("conforming cancel fixture");
    let codes = semantic_violation_codes_for_fixture(
        &fixture.mission,
        &fixture.events,
        &[],
        &fixture.driver_capabilities,
    );
    assert!(
        codes.contains(&"SEM-A05"),
        "cancel without binding must be SEM-A05, got {codes:?}"
    );
}

#[test]
fn rehashed_unknown_field_and_wrong_schema_are_sem_a05() {
    let fixture = load_history(&fixtures_root().join("conforming/protocol-confirmed-cancel.json"))
        .expect("conforming cancel fixture");

    let mut unknown = fixture.control_records.clone();
    unknown[0].request["spoofed_identity"] = serde_json::json!("nope");
    if let Some(hash) = lanyte_mission::control_content_hash(&unknown[0].request) {
        unknown[0].request["request_fingerprint"] = serde_json::json!(hash);
        unknown[0].request_fingerprint = hash;
    }
    let unknown_codes =
        semantic_violation_codes_for_fixture(&fixture.mission, &fixture.events, &unknown, &[]);
    assert!(
        unknown_codes.contains(&"SEM-A05"),
        "unknown field must be SEM-A05, got {unknown_codes:?}"
    );

    let mut wrong_schema = fixture.control_records;
    wrong_schema[0].request["control_schema"] = serde_json::json!(
        "https://schemas.3leaps.dev/agentic/mission/v0.1/not-control.schema.json"
    );
    if let Some(hash) = lanyte_mission::control_content_hash(&wrong_schema[0].request) {
        wrong_schema[0].request["request_fingerprint"] = serde_json::json!(hash);
        wrong_schema[0].request_fingerprint = hash;
    }
    let schema_codes =
        semantic_violation_codes_for_fixture(&fixture.mission, &fixture.events, &wrong_schema, &[]);
    assert!(
        schema_codes.contains(&"SEM-A05"),
        "wrong schema pin must be SEM-A05, got {schema_codes:?}"
    );
}
