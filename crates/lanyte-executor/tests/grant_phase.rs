use std::fs;
use std::path::PathBuf;

use lanyte_executor::{Executor, ExecutorError};

#[test]
fn grant_records_echo_capability_as_denied() {
    let bytes = read_echo_wasm();
    let mut executor = Executor::new().expect("executor builds");
    let manifest = executor.load(&bytes).expect("echo wasm loads");
    assert_eq!(manifest.capabilities, vec!["skill.echo"]);

    let caps = executor
        .grant(&manifest.skill_id)
        .expect("grant succeeds for echo");

    assert_eq!(caps.declared, vec!["skill.echo"]);
    assert!(caps.granted.is_empty(), "v1 grants nothing");
    assert_eq!(caps.denied, vec!["skill.echo"]);
}

#[test]
fn grant_is_idempotent_and_observable_through_loaded_skill() {
    let bytes = read_echo_wasm();
    let mut executor = Executor::new().expect("executor builds");
    executor.load(&bytes).expect("echo wasm loads");

    let first = executor
        .grant("dev.lanyte.echo")
        .expect("first grant succeeds");
    let second = executor
        .grant("dev.lanyte.echo")
        .expect("second grant returns cached result");
    assert_eq!(first, second);

    let loaded = executor
        .get("dev.lanyte.echo")
        .expect("skill remains registered after grant");
    let cached = loaded
        .granted_capabilities()
        .expect("grant is cached on the LoadedSkill");
    assert_eq!(cached, &first);
}

#[test]
fn grant_returns_skill_not_found_for_unknown_id() {
    let mut executor = Executor::new().expect("executor builds");

    let err = executor
        .grant("dev.lanyte.missing")
        .expect_err("grant on unknown skill must fail");

    assert_eq!(
        err,
        ExecutorError::SkillNotFound("dev.lanyte.missing".to_owned())
    );
}

#[test]
fn load_describe_grant_pipeline_preserves_manifest() {
    let bytes = read_echo_wasm();
    let mut executor = Executor::new().expect("executor builds");
    let manifest_from_load = executor.load(&bytes).expect("echo wasm loads");
    let caps = executor
        .grant(&manifest_from_load.skill_id)
        .expect("grant succeeds");

    let loaded = executor
        .get(&manifest_from_load.skill_id)
        .expect("skill registered");
    assert_eq!(loaded.manifest(), &manifest_from_load);
    assert_eq!(loaded.granted_capabilities(), Some(&caps));
}

fn read_echo_wasm() -> Vec<u8> {
    let path = echo_wasm_path();
    fs::read(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read echo wasm fixture at {}: {err}. Refresh with: cargo build --manifest-path fixtures/skills/echo/Cargo.toml --target wasm32-unknown-unknown --release && cp fixtures/skills/echo/target/wasm32-unknown-unknown/release/lanyte_skill_echo_fixture.wasm fixtures/skills/echo/dev.lanyte.echo.wasm",
            path.display()
        )
    })
}

fn echo_wasm_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("lanyte-executor crate should live under crates/")
        .join("fixtures/skills/echo/dev.lanyte.echo.wasm")
}
