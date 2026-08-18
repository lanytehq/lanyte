use std::collections::{BTreeMap, BTreeSet};
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
    assert_eq!(conforming.len(), 8);
    assert_eq!(negative.len(), 57);
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
                if let Err(err) = validate_history(&fixture.mission, &fixture.events) {
                    failures.push(format!("{name}: {err}"));
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
    for (name, expected) in &manifest.negative {
        let text = fs::read_to_string(dir.join(name)).expect("read negative fixture");
        let Ok(_) = serde_json::from_str::<serde_json::Value>(&text) else {
            failures.push(format!("{name}: fixture is not JSON"));
            continue;
        };
        if let Ok(fixture) = load_history(&dir.join(name)) {
            if validate_history(&fixture.mission, &fixture.events).is_ok() {
                failures.push(format!("{name}: accepted (wanted {expected})"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "negative fixtures failed:\n{}",
        failures.join("\n")
    );
}
