use std::fs;
use std::path::Path;

use lanyte_mission::{validate_history, LifecycleEvent, MissionRecord};
use serde::Deserialize;

#[derive(Deserialize)]
struct HistoryFixture {
    mission: MissionRecord,
    events: Vec<LifecycleEvent>,
}

fn load(path: &Path) -> HistoryFixture {
    serde_json::from_str(&fs::read_to_string(path).expect("read fixture"))
        .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

fn each_json(dir: &Path) -> impl Iterator<Item = std::path::PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(dir)
        .expect("fixtures")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths.into_iter()
}

#[test]
fn conforming_v0_1_histories_validate() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/semantic/conforming");
    let mut failures = Vec::new();
    for path in each_json(&dir) {
        let fixture = load(&path);
        if let Err(err) = validate_history(&fixture.mission, &fixture.events) {
            failures.push(format!(
                "{}: {err}",
                path.file_name().unwrap().to_string_lossy()
            ));
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
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/semantic/negative");
    let mut accepted = Vec::new();
    for path in each_json(&dir) {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Ok(fixture) =
            serde_json::from_str::<HistoryFixture>(&fs::read_to_string(&path).unwrap())
        else {
            continue;
        };
        if validate_history(&fixture.mission, &fixture.events).is_ok() {
            accepted.push(name);
        }
    }
    assert!(
        accepted.is_empty(),
        "negative fixtures were accepted:\n{}",
        accepted.join("\n")
    );
}
