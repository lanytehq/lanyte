//! SKL-004 grant phase: capability-aware linker with deny stubs.
//!
//! In v1 no real host functions exist. Every import the skill declares is
//! wired to a deny stub that traps with `permission_denied` and identifies
//! the function. `GrantedCapabilities` records what the manifest declared
//! versus what the executor actually granted (v1: always nothing).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use wasmtime::{Caller, Engine, ExternType, FuncType, Linker, Module, Val};

use crate::{ExecutorError, SkillManifest};

/// Capability outcome of the grant phase for a single loaded skill.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantedCapabilities {
    /// Capabilities the manifest declared, in declaration order.
    pub declared: Vec<String>,
    /// Capabilities the executor backed with real host functions. v1: always empty.
    pub granted: Vec<String>,
    /// Capabilities the executor refused to back. v1: equal to `declared`.
    pub denied: Vec<String>,
}

impl GrantedCapabilities {
    fn deny_all(declared: Vec<String>) -> Self {
        let denied = declared.clone();
        Self {
            declared,
            granted: Vec::new(),
            denied,
        }
    }
}

/// Cached grant artifact: the linker and the capability summary.
pub(crate) struct Grant {
    pub(crate) capabilities: GrantedCapabilities,
    // Consumed by the execute phase (SKL-005). Unused by the grant-only v1
    // surface but built and cached here to avoid rebuilding per invocation.
    #[allow(dead_code)]
    pub(crate) linker: Linker<()>,
}

impl std::fmt::Debug for Grant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grant")
            .field("capabilities", &self.capabilities)
            .field("linker", &"<wasmtime::Linker<()>>")
            .finish()
    }
}

/// Build the grant artifact for a loaded skill.
///
/// Every function import the module declares is bound to a deny stub that
/// traps with a `permission_denied` message naming the qualified import.
/// Non-function imports are rejected — v1 has no mechanism to satisfy them.
pub(crate) fn build_grant(
    engine: &Engine,
    module: &Module,
    manifest: &SkillManifest,
) -> Result<Grant, ExecutorError> {
    let mut linker = Linker::<()>::new(engine);
    // WASM allows repeated (module, name) imports. A single deny stub
    // covers every alias iff all aliases share the FuncType — otherwise
    // the linker is not actually linkable and instantiation fails later.
    // Reject mismatched signatures here so grant only returns linkable
    // artifacts.
    let mut defined: HashMap<(String, String), FuncType> = HashMap::new();

    for import in module.imports() {
        let module_name = import.module().to_owned();
        let field_name = import.name().to_owned();
        let qualified = format!("{module_name}::{field_name}");

        match import.ty() {
            ExternType::Func(func_ty) => {
                let key = (module_name.clone(), field_name.clone());
                if let Some(existing) = defined.get(&key) {
                    // Exact structural equality. Discriminant-only matching
                    // is insufficient for reference types, where different
                    // heap types share the `ValType::Ref` variant.
                    if !FuncType::eq(existing, &func_ty) {
                        return Err(ExecutorError::LinkerError(format!(
                            "import `{qualified}` reused with incompatible function signatures"
                        )));
                    }
                    continue;
                }
                define_deny_stub(
                    &mut linker,
                    &module_name,
                    &field_name,
                    func_ty.clone(),
                    qualified,
                )?;
                defined.insert(key, func_ty);
            }
            _ => {
                return Err(ExecutorError::LinkerError(format!(
                    "v1 grant phase rejects non-function import `{qualified}`"
                )));
            }
        }
    }

    Ok(Grant {
        capabilities: GrantedCapabilities::deny_all(manifest.capabilities.clone()),
        linker,
    })
}

fn define_deny_stub(
    linker: &mut Linker<()>,
    module_name: &str,
    field_name: &str,
    func_ty: FuncType,
    qualified: String,
) -> Result<(), ExecutorError> {
    linker
        .func_new(
            module_name,
            field_name,
            func_ty,
            move |_caller: Caller<'_, ()>, _params: &[Val], _results: &mut [Val]| {
                Err(wasmtime::Error::msg(format!(
                    "permission_denied: host function `{qualified}` is not granted to this skill"
                )))
            },
        )
        .map_err(|err| {
            ExecutorError::LinkerError(format!(
                "failed to register deny stub for `{module_name}::{field_name}`: {err}"
            ))
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmtime::{Config, Store};

    fn engine() -> Engine {
        let mut config = Config::new();
        config.consume_fuel(true);
        Engine::new(&config).expect("engine builds")
    }

    fn manifest_with_caps(caps: &[&str]) -> SkillManifest {
        SkillManifest {
            skill_id: "test.skill".to_owned(),
            name: "Test".to_owned(),
            version: "1.0.0".to_owned(),
            tier: 0,
            capabilities: caps.iter().map(|s| (*s).to_owned()).collect(),
            abi_version: Some(1),
            description: None,
            author: None,
            operations: None,
            metadata: None,
        }
    }

    fn compile_wat(engine: &Engine, source: &str) -> Module {
        let bytes = wat::parse_str(source).expect("WAT compiles");
        Module::new(engine, &bytes).expect("module compiles")
    }

    fn module_no_imports(engine: &Engine) -> Module {
        compile_wat(engine, r#"(module (memory (export "memory") 1))"#)
    }

    fn module_with_host_import(engine: &Engine, import_module: &str, import_field: &str) -> Module {
        let wat = format!(
            r#"(module
  (import "{import_module}" "{import_field}" (func $host))
  (memory (export "memory") 1)
  (func (export "call_host") (result i32)
    call $host
    i32.const 0
  )
)"#
        );
        compile_wat(engine, &wat)
    }

    fn module_with_memory_import(engine: &Engine) -> Module {
        compile_wat(engine, r#"(module (import "env" "mem" (memory 1)))"#)
    }

    fn module_with_duplicate_host_imports(engine: &Engine) -> Module {
        compile_wat(
            engine,
            r#"(module
  (import "env" "forbidden_fn" (func $first))
  (import "env" "forbidden_fn" (func $second))
  (memory (export "memory") 1)
  (func (export "call_first") (result i32)
    call $first
    i32.const 0
  )
  (func (export "call_second") (result i32)
    call $second
    i32.const 0
  )
)"#,
        )
    }

    fn module_with_incompatible_duplicate_imports(engine: &Engine) -> Module {
        compile_wat(
            engine,
            r#"(module
  (import "env" "ambiguous" (func (param i32)))
  (import "env" "ambiguous" (func (param i64)))
  (memory (export "memory") 1)
)"#,
        )
    }

    #[test]
    fn deny_all_records_every_declared_capability_as_denied() {
        let caps =
            GrantedCapabilities::deny_all(vec!["skill.echo".to_owned(), "net.http".to_owned()]);
        assert_eq!(caps.declared, vec!["skill.echo", "net.http"]);
        assert!(caps.granted.is_empty());
        assert_eq!(caps.denied, vec!["skill.echo", "net.http"]);
    }

    #[test]
    fn build_grant_succeeds_for_module_with_no_imports() {
        let engine = engine();
        let module = module_no_imports(&engine);
        let manifest = manifest_with_caps(&["skill.echo"]);

        let grant = build_grant(&engine, &module, &manifest).expect("grant builds");

        assert_eq!(grant.capabilities.declared, vec!["skill.echo"]);
        assert!(grant.capabilities.granted.is_empty());
        assert_eq!(grant.capabilities.denied, vec!["skill.echo"]);
    }

    #[test]
    fn build_grant_tolerates_empty_capabilities() {
        let engine = engine();
        let module = module_no_imports(&engine);
        let manifest = manifest_with_caps(&[]);

        let grant = build_grant(&engine, &module, &manifest).expect("grant builds");

        assert!(grant.capabilities.declared.is_empty());
        assert!(grant.capabilities.granted.is_empty());
        assert!(grant.capabilities.denied.is_empty());
    }

    #[test]
    fn build_grant_tolerates_unknown_capability_strings() {
        let engine = engine();
        let module = module_no_imports(&engine);
        let manifest = manifest_with_caps(&["future.unknown.cap", "another.one"]);

        let grant = build_grant(&engine, &module, &manifest).expect("grant builds");

        assert_eq!(
            grant.capabilities.denied,
            vec!["future.unknown.cap", "another.one"]
        );
    }

    #[test]
    fn build_grant_rejects_non_function_imports() {
        let engine = engine();
        let module = module_with_memory_import(&engine);
        let manifest = manifest_with_caps(&[]);

        let err = build_grant(&engine, &module, &manifest)
            .expect_err("non-function import must fail grant");

        match err {
            ExecutorError::LinkerError(message) => {
                assert!(message.contains("non-function import"));
                assert!(message.contains("env::mem"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn deny_stub_traps_with_permission_denied_and_qualified_name() {
        let engine = engine();
        let module = module_with_host_import(&engine, "env", "forbidden_fn");
        let manifest = manifest_with_caps(&["skill.echo"]);

        let grant = build_grant(&engine, &module, &manifest).expect("grant builds");

        let mut store = Store::new(&engine, ());
        store.set_fuel(1_000_000).expect("fuel set");
        let instance = grant
            .linker
            .instantiate(&mut store, &module)
            .expect("module instantiates through deny-stub linker");

        let call_host = instance
            .get_typed_func::<(), i32>(&mut store, "call_host")
            .expect("call_host export present");

        let err = call_host
            .call(&mut store, ())
            .expect_err("calling denied host import must trap");

        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("permission_denied"),
            "trap message missing `permission_denied`: {rendered}"
        );
        assert!(
            rendered.contains("env::forbidden_fn"),
            "trap message must identify the function: {rendered}"
        );
        assert!(
            rendered.contains("is not granted to this skill"),
            "trap message must describe the gate: {rendered}"
        );
    }

    #[test]
    fn build_grant_tolerates_duplicate_host_imports_and_stub_traps_both_aliases() {
        let engine = engine();
        let module = module_with_duplicate_host_imports(&engine);
        let manifest = manifest_with_caps(&[]);

        let grant = build_grant(&engine, &module, &manifest)
            .expect("duplicate imports must not fail grant");

        let mut store = Store::new(&engine, ());
        store.set_fuel(1_000_000).expect("fuel set");
        let instance = grant
            .linker
            .instantiate(&mut store, &module)
            .expect("module instantiates through deduped deny-stub linker");

        for export in ["call_first", "call_second"] {
            let call = instance
                .get_typed_func::<(), i32>(&mut store, export)
                .unwrap_or_else(|err| panic!("{export} export missing: {err}"));
            let err = call
                .call(&mut store, ())
                .expect_err(&format!("calling {export} must trap through the deny stub"));
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains("permission_denied"),
                "{export} trap message missing `permission_denied`: {rendered}"
            );
            assert!(
                rendered.contains("env::forbidden_fn"),
                "{export} trap message must identify the function: {rendered}"
            );
        }
    }

    #[test]
    fn build_grant_rejects_duplicate_host_imports_with_incompatible_signatures() {
        let engine = engine();
        let module = module_with_incompatible_duplicate_imports(&engine);
        let manifest = manifest_with_caps(&[]);

        let err = build_grant(&engine, &module, &manifest)
            .expect_err("duplicate imports with conflicting signatures must hard-fail grant");

        match err {
            ExecutorError::LinkerError(message) => {
                assert!(
                    message.contains("env::ambiguous"),
                    "err must name the conflicting import: {message}"
                );
                assert!(
                    message.contains("incompatible"),
                    "err must flag the signature conflict: {message}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
