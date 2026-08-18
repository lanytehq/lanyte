use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use lanyte_mission::{validate_history, LifecycleEvent, MissionRecord};
use serde::Deserialize;

#[derive(Deserialize)]
struct HistoryFixture {
    mission: MissionRecord,
    events: Vec<LifecycleEvent>,
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

fn load_history(path: &Path) -> HistoryFixture {
    serde_json::from_str(&fs::read_to_string(path).expect("read fixture"))
        .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

#[test]
fn semantic_manifest_lists_the_locked_set() {
    let manifest = load_manifest();
    assert_eq!(manifest.conforming.len(), 8);
    assert_eq!(manifest.negative.len(), 57);
}

#[test]
fn conforming_v0_1_histories_validate() {
    let manifest = load_manifest();
    let dir = fixtures_root().join("conforming");
    let mut failures = Vec::new();
    for name in &manifest.conforming {
        let fixture = load_history(&dir.join(name));
        if let Err(err) = validate_history(&fixture.mission, &fixture.events) {
            failures.push(format!("{name}: {err}"));
        }
    }
    assert!(
        failures.is_empty(),
        "conforming fixtures failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn negative_v0_1_histories_reject() {
    let manifest = load_manifest();
    let dir = fixtures_root().join("negative");
    let mut accepted = Vec::new();
    for name in manifest.negative.keys() {
        let path = dir.join(name);
        let Ok(fixture) =
            serde_json::from_str::<HistoryFixture>(&fs::read_to_string(&path).unwrap())
        else {
            continue;
        };
        if validate_history(&fixture.mission, &fixture.events).is_ok() {
            accepted.push(name.clone());
        }
    }
    assert!(
        accepted.is_empty(),
        "negative fixtures were accepted:\n{}",
        accepted.join("\n")
    );
}
