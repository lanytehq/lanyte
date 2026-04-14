# Echo Skill Fixture

This standalone crate is the dogfood Skill ABI v1 fixture introduced in SKL-002.

`dev.lanyte.echo.wasm` is the checked-in artifact used by the SKL-003 host-side
integration test so the current `make pr-final` / CI gate can load a real
Rust-authored skill without adding `wasm32-unknown-unknown` toolchain plumbing to
the workspace test path.

Refresh the checked-in wasm after changing the fixture source:

```bash
cargo build --manifest-path fixtures/skills/echo/Cargo.toml --target wasm32-unknown-unknown --release
cp fixtures/skills/echo/target/wasm32-unknown-unknown/release/lanyte_skill_echo_fixture.wasm fixtures/skills/echo/dev.lanyte.echo.wasm
```
