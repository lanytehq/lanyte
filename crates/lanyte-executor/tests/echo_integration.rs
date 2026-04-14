use std::fs;
use std::path::PathBuf;

use lanyte_executor::Executor;

#[test]
fn executor_loads_echo_wasm_and_registers_manifest() {
    let path = echo_wasm_path();
    let bytes = fs::read(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read echo wasm fixture at {}: {err}. Refresh with: cargo build --manifest-path fixtures/skills/echo/Cargo.toml --target wasm32-unknown-unknown --release && cp fixtures/skills/echo/target/wasm32-unknown-unknown/release/lanyte_skill_echo_fixture.wasm fixtures/skills/echo/dev.lanyte.echo.wasm",
            path.display()
        )
    });

    let mut executor = Executor::new().expect("executor should build");
    let manifest = executor
        .load(&bytes)
        .expect("echo wasm should load through executor");

    assert_eq!(manifest.skill_id, "dev.lanyte.echo");
    assert_eq!(manifest.name, "Echo");
    assert_eq!(manifest.version, "1.0.0");
    assert_eq!(manifest.tier, 0);
    assert_eq!(manifest.capabilities, vec!["skill.echo"]);

    let loaded = executor
        .get("dev.lanyte.echo")
        .expect("loaded skill should be available from registry");
    assert_eq!(loaded.manifest(), &manifest);

    let listed = executor.list();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0], &manifest);
}

fn echo_wasm_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("lanyte-executor crate should live under crates/")
        .join("fixtures/skills/echo/dev.lanyte.echo.wasm")
}
