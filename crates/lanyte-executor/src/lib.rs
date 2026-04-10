//! WASM skill executor load/describe foundation.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use wasmtime::{
    Config, Engine, ExternType, FuncType, Instance, Linker, Memory, Module, Store, TypedFunc,
    ValType,
};

const SUPPORTED_ABI_VERSION: i32 = 1;
const DESCRIBE_FUEL: u64 = 1_000_000;
const MAX_SKILL_ID_LEN: usize = 256;
const MAX_NAME_LEN: usize = 256;
const MAX_VERSION_LEN: usize = 64;
const MAX_DESCRIPTION_LEN: usize = 1024;
const MAX_AUTHOR_LEN: usize = 256;
const MAX_CAPABILITIES: usize = 20;
const MAX_CAPABILITY_LEN: usize = 64;

/// Loads and validates ABI-compatible skill modules.
#[derive(Debug)]
pub struct Executor {
    engine: Engine,
    skills: BTreeMap<String, LoadedSkill>,
}

impl Executor {
    /// Creates an executor with a shared wasmtime engine.
    pub fn new() -> Result<Self, ExecutorError> {
        Ok(Self {
            engine: build_engine()?,
            skills: BTreeMap::new(),
        })
    }

    /// Loads a skill module from raw wasm bytes and registers it by `skill_id`.
    pub fn load(&mut self, bytes: &[u8]) -> Result<SkillManifest, ExecutorError> {
        let module = compile_module(&self.engine, bytes)?;
        let manifest = describe_module(&self.engine, &module)?;

        if self.skills.contains_key(&manifest.skill_id) {
            return Err(ExecutorError::SkillAlreadyLoaded(manifest.skill_id.clone()));
        }

        self.skills.insert(
            manifest.skill_id.clone(),
            LoadedSkill {
                module,
                manifest: manifest.clone(),
            },
        );

        Ok(manifest)
    }

    /// Returns a loaded skill by id.
    #[must_use]
    pub fn get(&self, skill_id: &str) -> Option<&LoadedSkill> {
        self.skills.get(skill_id)
    }

    /// Lists loaded manifests in skill-id order.
    #[must_use]
    pub fn list(&self) -> Vec<&SkillManifest> {
        self.skills.values().map(|skill| &skill.manifest).collect()
    }

    /// Removes a loaded skill from the registry.
    pub fn unload(&mut self, skill_id: &str) -> bool {
        self.skills.remove(skill_id).is_some()
    }
}

/// A compiled, validated skill and its cached manifest.
#[derive(Debug)]
pub struct LoadedSkill {
    module: Module,
    manifest: SkillManifest,
}

impl LoadedSkill {
    #[must_use]
    pub fn module(&self) -> &Module {
        &self.module
    }

    #[must_use]
    pub fn manifest(&self) -> &SkillManifest {
        &self.manifest
    }
}

/// Parsed skill manifest returned by `lanyte_skill_describe`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillManifest {
    pub skill_id: String,
    pub name: String,
    pub version: String,
    pub tier: u8,
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub abi_version: Option<i32>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub operations: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// Structured skill error payload returned by `lanyte_skill_last_error`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    #[serde(default)]
    pub details: Option<Value>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExecutorError {
    #[error("invalid wasm module: {0}")]
    InvalidModule(String),

    #[error("missing required export `{0}`")]
    MissingExport(String),

    #[error("wrong signature for export `{0}`")]
    WrongExportSignature(String),

    #[error("unsupported skill ABI version `{0}`")]
    UnsupportedAbiVersion(i32),

    #[error("describe failed: {0}")]
    DescribeFailed(String),

    #[error("invalid skill manifest: {0}")]
    ManifestInvalid(String),

    #[error("guest memory payload is out of bounds")]
    MemoryOutOfBounds,

    #[error("guest payload is not valid UTF-8")]
    InvalidUtf8,

    #[error("skill `{0}` is already loaded")]
    SkillAlreadyLoaded(String),

    #[error("skill `{0}` was not found")]
    SkillNotFound(String),

    #[error("wasm runtime error: {0}")]
    Runtime(String),
}

#[derive(Debug, Clone, Copy)]
struct GuestBuffer {
    ptr: u32,
    len: u32,
}

struct LoadedInstance {
    store: Store<()>,
    #[allow(dead_code)]
    instance: Instance,
    memory: Memory,
    abi_version: TypedFunc<(), i32>,
    free: TypedFunc<(i32, i32), ()>,
    describe: TypedFunc<(), i64>,
    last_error: TypedFunc<(), i64>,
}

fn build_engine() -> Result<Engine, ExecutorError> {
    let mut config = Config::new();
    config.consume_fuel(true);
    config.wasm_memory64(false);
    config.wasm_bulk_memory(true);
    config.wasm_multi_value(true);
    Engine::new(&config).map_err(|err| ExecutorError::Runtime(format!("engine init failed: {err}")))
}

fn compile_module(engine: &Engine, bytes: &[u8]) -> Result<Module, ExecutorError> {
    let module =
        Module::new(engine, bytes).map_err(|err| ExecutorError::InvalidModule(err.to_string()))?;
    validate_module_exports(&module)?;
    Ok(module)
}

fn validate_module_exports(module: &Module) -> Result<(), ExecutorError> {
    validate_memory_export(module, "memory")?;
    validate_func_export(module, "lanyte_skill_abi_version", &[], &[ValType::I32])?;
    validate_func_export(
        module,
        "lanyte_skill_alloc",
        &[ValType::I32],
        &[ValType::I32],
    )?;
    validate_func_export(
        module,
        "lanyte_skill_free",
        &[ValType::I32, ValType::I32],
        &[],
    )?;
    validate_func_export(module, "lanyte_skill_describe", &[], &[ValType::I64])?;
    validate_func_export(
        module,
        "lanyte_skill_invoke",
        &[ValType::I32, ValType::I32],
        &[ValType::I64],
    )?;
    validate_func_export(module, "lanyte_skill_last_error", &[], &[ValType::I64])
}

fn validate_memory_export(module: &Module, name: &str) -> Result<(), ExecutorError> {
    match module.get_export(name) {
        Some(ExternType::Memory(_)) => Ok(()),
        Some(_) => Err(ExecutorError::WrongExportSignature(name.to_owned())),
        None => Err(ExecutorError::MissingExport(name.to_owned())),
    }
}

fn validate_func_export(
    module: &Module,
    name: &str,
    params: &[ValType],
    results: &[ValType],
) -> Result<(), ExecutorError> {
    let Some(export) = module.get_export(name) else {
        return Err(ExecutorError::MissingExport(name.to_owned()));
    };
    let Some(func) = export.func() else {
        return Err(ExecutorError::WrongExportSignature(name.to_owned()));
    };
    if !func_type_matches(func, params, results) {
        return Err(ExecutorError::WrongExportSignature(name.to_owned()));
    }
    Ok(())
}

fn func_type_matches(func: &FuncType, params: &[ValType], results: &[ValType]) -> bool {
    let actual_params = func.params().collect::<Vec<_>>();
    let actual_results = func.results().collect::<Vec<_>>();
    val_type_lists_match(&actual_params, params) && val_type_lists_match(&actual_results, results)
}

fn val_type_lists_match(actual: &[ValType], expected: &[ValType]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| val_type_matches(actual, expected))
}

fn val_type_matches(actual: &ValType, expected: &ValType) -> bool {
    std::mem::discriminant(actual) == std::mem::discriminant(expected)
}

fn describe_module(engine: &Engine, module: &Module) -> Result<SkillManifest, ExecutorError> {
    let mut instance = instantiate_module(engine, module)?;
    let abi_version = instance
        .abi_version
        .call(&mut instance.store, ())
        .map_err(|err| runtime_error("calling lanyte_skill_abi_version", err))?;
    if abi_version != SUPPORTED_ABI_VERSION {
        return Err(ExecutorError::UnsupportedAbiVersion(abi_version));
    }
    describe_manifest(&mut instance, abi_version)
}

fn instantiate_module(engine: &Engine, module: &Module) -> Result<LoadedInstance, ExecutorError> {
    let linker = Linker::<()>::new(engine);
    let mut store = Store::new(engine, ());
    store
        .set_fuel(DESCRIBE_FUEL)
        .map_err(|err| runtime_error("setting describe fuel", err))?;
    let instance = linker
        .instantiate(&mut store, module)
        .map_err(|err| runtime_error("instantiating skill module", err))?;
    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| ExecutorError::MissingExport("memory".to_owned()))?;
    let abi_version = instance
        .get_typed_func::<(), i32>(&mut store, "lanyte_skill_abi_version")
        .map_err(|err| runtime_error("loading lanyte_skill_abi_version", err))?;
    let free = instance
        .get_typed_func::<(i32, i32), ()>(&mut store, "lanyte_skill_free")
        .map_err(|err| runtime_error("loading lanyte_skill_free", err))?;
    let describe = instance
        .get_typed_func::<(), i64>(&mut store, "lanyte_skill_describe")
        .map_err(|err| runtime_error("loading lanyte_skill_describe", err))?;
    let last_error = instance
        .get_typed_func::<(), i64>(&mut store, "lanyte_skill_last_error")
        .map_err(|err| runtime_error("loading lanyte_skill_last_error", err))?;

    Ok(LoadedInstance {
        store,
        instance,
        memory,
        abi_version,
        free,
        describe,
        last_error,
    })
}

fn describe_manifest(
    instance: &mut LoadedInstance,
    abi_version: i32,
) -> Result<SkillManifest, ExecutorError> {
    let packed = instance
        .describe
        .call(&mut instance.store, ())
        .map_err(|err| runtime_error("calling lanyte_skill_describe", err))?;

    if unpack_ptr_len(packed).is_none() {
        return Err(describe_failed_from_last_error(instance));
    }

    let bytes = read_owned_guest_bytes(instance, packed)?
        .expect("non-zero packed describe payload must produce bytes");
    parse_manifest(&bytes, abi_version)
}

fn describe_failed_from_last_error(instance: &mut LoadedInstance) -> ExecutorError {
    let packed = match instance.last_error.call(&mut instance.store, ()) {
        Ok(packed) => packed,
        Err(err) => {
            return ExecutorError::DescribeFailed(format!(
                "describe() returned no manifest and lanyte_skill_last_error failed: {err}"
            ))
        }
    };

    let Some(bytes) = (match read_owned_guest_bytes(instance, packed) {
        Ok(bytes) => bytes,
        Err(err) => {
            return ExecutorError::DescribeFailed(format!(
            "describe() returned no manifest and lanyte_skill_last_error could not be read: {err}"
        ))
        }
    }) else {
        return ExecutorError::DescribeFailed(
            "describe() returned no manifest and lanyte_skill_last_error returned no payload"
                .to_owned(),
        );
    };

    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text,
        Err(_) => return ExecutorError::DescribeFailed(
            "describe() returned no manifest and lanyte_skill_last_error returned invalid UTF-8"
                .to_owned(),
        ),
    };

    let value = match serde_json::from_str::<Value>(text) {
        Ok(value) => value,
        Err(err) => {
            return ExecutorError::DescribeFailed(format!(
                "describe() returned no manifest and lanyte_skill_last_error returned invalid JSON: {err}"
            ))
        }
    };

    let skill_error = match serde_json::from_value::<SkillError>(value) {
        Ok(skill_error) => skill_error,
        Err(err) => {
            return ExecutorError::DescribeFailed(format!(
                "describe() returned no manifest and lanyte_skill_last_error payload was invalid: {err}"
            ))
        }
    };

    let rendered = serde_json::to_string(&skill_error)
        .unwrap_or_else(|_| format!("code={} message={}", skill_error.code, skill_error.message));
    ExecutorError::DescribeFailed(format!("describe() failed with skill error {rendered}"))
}

fn read_owned_guest_bytes(
    instance: &mut LoadedInstance,
    packed: i64,
) -> Result<Option<Vec<u8>>, ExecutorError> {
    let Some(buffer) = unpack_ptr_len(packed) else {
        return Ok(None);
    };

    let bytes = read_guest_bytes(&instance.memory, &instance.store, buffer.ptr, buffer.len)?;
    free_guest_buffer(instance, buffer)?;
    Ok(Some(bytes))
}

fn free_guest_buffer(
    instance: &mut LoadedInstance,
    buffer: GuestBuffer,
) -> Result<(), ExecutorError> {
    instance
        .free
        .call(&mut instance.store, (buffer.ptr as i32, buffer.len as i32))
        .map_err(|err| runtime_error("calling lanyte_skill_free", err))
}

fn parse_manifest(bytes: &[u8], abi_version: i32) -> Result<SkillManifest, ExecutorError> {
    let text = std::str::from_utf8(bytes).map_err(|_| ExecutorError::InvalidUtf8)?;
    let value = serde_json::from_str::<Value>(text).map_err(|err| {
        ExecutorError::DescribeFailed(format!("describe() returned invalid JSON manifest: {err}"))
    })?;
    let manifest = serde_json::from_value::<SkillManifest>(value).map_err(|err| {
        ExecutorError::ManifestInvalid(format!("manifest shape is invalid: {err}"))
    })?;
    validate_manifest(&manifest, abi_version)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &SkillManifest, abi_version: i32) -> Result<(), ExecutorError> {
    validate_skill_id(&manifest.skill_id)?;
    validate_nonempty_string("name", &manifest.name, MAX_NAME_LEN)?;
    validate_nonempty_string("version", &manifest.version, MAX_VERSION_LEN)?;
    if manifest.tier > 4 {
        return Err(ExecutorError::ManifestInvalid(format!(
            "tier {} is out of range 0-4",
            manifest.tier
        )));
    }
    if manifest.capabilities.len() > MAX_CAPABILITIES {
        return Err(ExecutorError::ManifestInvalid(format!(
            "capabilities exceeds {} entries",
            MAX_CAPABILITIES
        )));
    }
    for capability in &manifest.capabilities {
        if capability.len() > MAX_CAPABILITY_LEN {
            return Err(ExecutorError::ManifestInvalid(format!(
                "capability `{capability}` exceeds {} characters",
                MAX_CAPABILITY_LEN
            )));
        }
    }
    if let Some(description) = &manifest.description {
        if description.len() > MAX_DESCRIPTION_LEN {
            return Err(ExecutorError::ManifestInvalid(format!(
                "description exceeds {} characters",
                MAX_DESCRIPTION_LEN
            )));
        }
    }
    if let Some(author) = &manifest.author {
        if author.len() > MAX_AUTHOR_LEN {
            return Err(ExecutorError::ManifestInvalid(format!(
                "author exceeds {} characters",
                MAX_AUTHOR_LEN
            )));
        }
    }
    if let Some(manifest_abi_version) = manifest.abi_version {
        if manifest_abi_version != abi_version {
            return Err(ExecutorError::ManifestInvalid(format!(
                "manifest abi_version {manifest_abi_version} does not match exported ABI version {abi_version}"
            )));
        }
    }
    Ok(())
}

fn validate_skill_id(skill_id: &str) -> Result<(), ExecutorError> {
    if skill_id.is_empty() {
        return Err(ExecutorError::ManifestInvalid(
            "skill_id must not be empty".to_owned(),
        ));
    }
    if skill_id.len() > MAX_SKILL_ID_LEN {
        return Err(ExecutorError::ManifestInvalid(format!(
            "skill_id exceeds {} characters",
            MAX_SKILL_ID_LEN
        )));
    }

    let first = skill_id.chars().next().expect("skill_id is non-empty");
    if !first.is_ascii_alphanumeric() {
        return Err(ExecutorError::ManifestInvalid(
            "skill_id must start with an ASCII letter or digit".to_owned(),
        ));
    }

    let last = skill_id.chars().last().expect("skill_id is non-empty");
    if !last.is_ascii_alphanumeric() {
        return Err(ExecutorError::ManifestInvalid(
            "skill_id must end with an ASCII letter or digit".to_owned(),
        ));
    }

    if let Some((index, invalid)) = skill_id
        .char_indices()
        .find(|(_, ch)| !(ch.is_ascii_alphanumeric() || *ch == '.' || *ch == '_' || *ch == '-'))
    {
        return Err(ExecutorError::ManifestInvalid(format!(
            "skill_id contains invalid character `{invalid}` at byte {index}"
        )));
    }

    Ok(())
}

fn validate_nonempty_string(field: &str, value: &str, max_len: usize) -> Result<(), ExecutorError> {
    if value.is_empty() {
        return Err(ExecutorError::ManifestInvalid(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > max_len {
        return Err(ExecutorError::ManifestInvalid(format!(
            "{field} exceeds {max_len} characters"
        )));
    }
    Ok(())
}

fn unpack_ptr_len(packed: i64) -> Option<GuestBuffer> {
    let packed = packed as u64;
    let ptr = (packed & 0xFFFF_FFFF) as u32;
    let len = ((packed >> 32) & 0xFFFF_FFFF) as u32;
    if ptr == 0 && len == 0 {
        None
    } else {
        Some(GuestBuffer { ptr, len })
    }
}

fn read_guest_bytes(
    memory: &Memory,
    store: &Store<()>,
    ptr: u32,
    len: u32,
) -> Result<Vec<u8>, ExecutorError> {
    let data = memory.data(store);
    let start = ptr as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or(ExecutorError::MemoryOutOfBounds)?;
    if end > data.len() {
        return Err(ExecutorError::MemoryOutOfBounds);
    }
    Ok(data[start..end].to_vec())
}

fn runtime_error(context: &str, err: impl std::fmt::Display) -> ExecutorError {
    ExecutorError::Runtime(format!("{context}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST_PTR: u32 = 32;
    const ERROR_PTR: u32 = 512;

    #[test]
    fn executor_loads_valid_module_and_registers_it() {
        let mut executor = Executor::new().expect("executor should build");

        let manifest = executor
            .load(&valid_skill_wasm())
            .expect("valid module should load");

        assert_eq!(manifest.skill_id, "dev.lanyte.echo");
        assert_eq!(executor.list().len(), 1);
        let loaded = executor
            .get("dev.lanyte.echo")
            .expect("loaded skill should be registered");
        assert_eq!(loaded.manifest().skill_id, "dev.lanyte.echo");
        assert!(executor.unload("dev.lanyte.echo"));
        assert!(executor.get("dev.lanyte.echo").is_none());
    }

    #[test]
    fn executor_rejects_missing_export() {
        let mut executor = Executor::new().expect("executor should build");

        let err = executor
            .load(&fixture_wasm(FixtureSpec {
                omit_export: Some("lanyte_skill_last_error"),
                ..FixtureSpec::valid()
            }))
            .expect_err("missing export must fail");

        assert_eq!(
            err,
            ExecutorError::MissingExport("lanyte_skill_last_error".to_owned())
        );
    }

    #[test]
    fn executor_rejects_wrong_export_signature() {
        let mut executor = Executor::new().expect("executor should build");

        let err = executor
            .load(&fixture_wasm(FixtureSpec {
                wrong_signature: Some("lanyte_skill_describe"),
                ..FixtureSpec::valid()
            }))
            .expect_err("wrong signature must fail");

        assert_eq!(
            err,
            ExecutorError::WrongExportSignature("lanyte_skill_describe".to_owned())
        );
    }

    #[test]
    fn executor_rejects_unsupported_abi_version() {
        let mut executor = Executor::new().expect("executor should build");

        let err = executor
            .load(&fixture_wasm(FixtureSpec {
                abi_version: 2,
                ..FixtureSpec::valid()
            }))
            .expect_err("unsupported abi version must fail");

        assert_eq!(err, ExecutorError::UnsupportedAbiVersion(2));
    }

    #[test]
    fn executor_surfaces_last_error_on_describe_failure() {
        let mut executor = Executor::new().expect("executor should build");

        let err = executor
            .load(&fixture_wasm(FixtureSpec {
                describe_packed: 0,
                last_error_bytes: Some(skill_error_bytes(
                    "input_invalid",
                    "manifest generation failed",
                    false,
                )),
                ..FixtureSpec::valid()
            }))
            .expect_err("describe failure must surface last_error");

        match err {
            ExecutorError::DescribeFailed(message) => {
                assert!(message.contains("input_invalid"));
                assert!(message.contains("manifest generation failed"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn executor_rejects_malformed_manifest_json() {
        let mut executor = Executor::new().expect("executor should build");

        let err = executor
            .load(&fixture_wasm(FixtureSpec {
                manifest_bytes: b"{not-json".to_vec(),
                describe_packed: pack_ptr_len(MANIFEST_PTR, b"{not-json".len()),
                ..FixtureSpec::valid()
            }))
            .expect_err("malformed json must fail");

        match err {
            ExecutorError::DescribeFailed(message) => {
                assert!(message.contains("invalid JSON manifest"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn executor_rejects_manifest_with_invalid_shape() {
        let mut executor = Executor::new().expect("executor should build");

        let err = executor
            .load(&fixture_wasm(FixtureSpec {
                manifest_bytes: br#"{"name":"Echo","version":"1.0.0","tier":0,"capabilities":[]}"#
                    .to_vec(),
                describe_packed: pack_ptr_len(
                    MANIFEST_PTR,
                    br#"{"name":"Echo","version":"1.0.0","tier":0,"capabilities":[]}"#.len(),
                ),
                ..FixtureSpec::valid()
            }))
            .expect_err("missing skill_id must fail");

        match err {
            ExecutorError::ManifestInvalid(message) => {
                assert!(message.contains("skill_id"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn executor_rejects_out_of_bounds_manifest_pointer() {
        let mut executor = Executor::new().expect("executor should build");

        let err = executor
            .load(&fixture_wasm(FixtureSpec {
                describe_packed: pack_ptr_len(65_520, 32),
                ..FixtureSpec::valid()
            }))
            .expect_err("oob pointer must fail");

        assert_eq!(err, ExecutorError::MemoryOutOfBounds);
    }

    #[test]
    fn describe_manifest_frees_manifest_once_on_success() {
        let executor = Executor::new().expect("executor should build");
        let module =
            compile_module(&executor.engine, &valid_skill_wasm()).expect("module compiles");
        let mut instance =
            instantiate_module(&executor.engine, &module).expect("instance should build");

        let manifest = describe_manifest(&mut instance, SUPPORTED_ABI_VERSION)
            .expect("describe should succeed");

        assert_eq!(manifest.skill_id, "dev.lanyte.echo");
        assert_eq!(free_count(&mut instance), 1);
    }

    #[test]
    fn describe_manifest_frees_manifest_once_on_invalid_utf8() {
        let executor = Executor::new().expect("executor should build");
        let module = compile_module(
            &executor.engine,
            &fixture_wasm(FixtureSpec {
                manifest_bytes: vec![0xFF, 0xFE, 0xFD],
                describe_packed: pack_ptr_len(MANIFEST_PTR, 3),
                ..FixtureSpec::valid()
            }),
        )
        .expect("module compiles");
        let mut instance =
            instantiate_module(&executor.engine, &module).expect("instance should build");

        let err = describe_manifest(&mut instance, SUPPORTED_ABI_VERSION)
            .expect_err("invalid utf8 must fail");

        assert_eq!(err, ExecutorError::InvalidUtf8);
        assert_eq!(free_count(&mut instance), 1);
    }

    #[test]
    fn describe_manifest_frees_last_error_once_on_describe_failure() {
        let executor = Executor::new().expect("executor should build");
        let module = compile_module(
            &executor.engine,
            &fixture_wasm(FixtureSpec {
                describe_packed: 0,
                last_error_bytes: Some(skill_error_bytes(
                    "input_invalid",
                    "manifest generation failed",
                    false,
                )),
                ..FixtureSpec::valid()
            }),
        )
        .expect("module compiles");
        let mut instance =
            instantiate_module(&executor.engine, &module).expect("instance should build");

        let err = describe_manifest(&mut instance, SUPPORTED_ABI_VERSION)
            .expect_err("describe failure must surface last_error");

        match err {
            ExecutorError::DescribeFailed(message) => {
                assert!(message.contains("input_invalid"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(free_count(&mut instance), 1);
    }

    #[test]
    fn executor_rejects_duplicate_skill_ids() {
        let mut executor = Executor::new().expect("executor should build");
        executor
            .load(&valid_skill_wasm())
            .expect("first load should succeed");

        let err = executor
            .load(&valid_skill_wasm())
            .expect_err("duplicate load must fail");

        assert_eq!(
            err,
            ExecutorError::SkillAlreadyLoaded("dev.lanyte.echo".to_owned())
        );
    }

    #[test]
    fn manifest_rejects_skill_id_that_violates_channel_pattern() {
        let mut executor = Executor::new().expect("executor should build");
        let invalid_manifest = manifest_bytes("dev.lanyte.echo!");

        let err = executor
            .load(&fixture_wasm(FixtureSpec {
                manifest_bytes: invalid_manifest.clone(),
                describe_packed: pack_ptr_len(MANIFEST_PTR, invalid_manifest.len()),
                ..FixtureSpec::valid()
            }))
            .expect_err("invalid skill_id must fail");

        match err {
            ExecutorError::ManifestInvalid(message) => {
                assert!(message.contains("skill_id"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn free_count(instance: &mut LoadedInstance) -> i32 {
        let global = instance
            .instance
            .get_global(&mut instance.store, "free_count")
            .expect("free_count export should exist");
        match global.get(&mut instance.store) {
            wasmtime::Val::I32(value) => value,
            other => panic!("unexpected free_count value: {other:?}"),
        }
    }

    fn valid_skill_wasm() -> Vec<u8> {
        fixture_wasm(FixtureSpec::valid())
    }

    fn manifest_bytes(skill_id: &str) -> Vec<u8> {
        format!(
            "{{\"skill_id\":\"{skill_id}\",\"name\":\"Echo\",\"version\":\"1.0.0\",\"tier\":0,\"capabilities\":[\"skill.echo\"]}}"
        )
        .into_bytes()
    }

    fn skill_error_bytes(code: &str, message: &str, retryable: bool) -> Vec<u8> {
        format!("{{\"code\":\"{code}\",\"message\":\"{message}\",\"retryable\":{retryable}}}")
            .into_bytes()
    }

    fn pack_ptr_len(ptr: u32, len: usize) -> i64 {
        ((len as i64) << 32) | i64::from(ptr)
    }

    #[derive(Clone)]
    struct FixtureSpec {
        abi_version: i32,
        manifest_bytes: Vec<u8>,
        describe_packed: i64,
        last_error_bytes: Option<Vec<u8>>,
        last_error_packed: Option<i64>,
        omit_export: Option<&'static str>,
        wrong_signature: Option<&'static str>,
    }

    impl FixtureSpec {
        fn valid() -> Self {
            let manifest_bytes = manifest_bytes("dev.lanyte.echo");
            Self {
                abi_version: SUPPORTED_ABI_VERSION,
                describe_packed: pack_ptr_len(MANIFEST_PTR, manifest_bytes.len()),
                manifest_bytes,
                last_error_bytes: None,
                last_error_packed: None,
                omit_export: None,
                wrong_signature: None,
            }
        }
    }

    fn fixture_wasm(spec: FixtureSpec) -> Vec<u8> {
        wat::parse_str(fixture_wat(spec)).expect("fixture WAT should compile")
    }

    fn fixture_wat(spec: FixtureSpec) -> String {
        let describe_export = export_name("lanyte_skill_describe", spec.omit_export);
        let last_error_export = export_name("lanyte_skill_last_error", spec.omit_export);
        let abi_version_export = export_name("lanyte_skill_abi_version", spec.omit_export);
        let alloc_export = export_name("lanyte_skill_alloc", spec.omit_export);
        let free_export = export_name("lanyte_skill_free", spec.omit_export);
        let invoke_export = export_name("lanyte_skill_invoke", spec.omit_export);
        let memory_export = if spec.omit_export == Some("memory") {
            "(memory 1)".to_owned()
        } else {
            "(memory (export \"memory\") 1)".to_owned()
        };
        let describe_signature = if spec.wrong_signature == Some("lanyte_skill_describe") {
            "(param i32) (result i64) local.get 0 i64.extend_i32_u".to_owned()
        } else {
            format!("(result i64) i64.const {}", spec.describe_packed)
        };
        let last_error_signature = if spec.wrong_signature == Some("lanyte_skill_last_error") {
            "(param i32) (result i64) local.get 0 i64.extend_i32_u".to_owned()
        } else {
            format!(
                "(result i64) i64.const {}",
                spec.last_error_packed.unwrap_or_else(|| {
                    spec.last_error_bytes
                        .as_ref()
                        .map_or(0, |bytes| pack_ptr_len(ERROR_PTR, bytes.len()))
                })
            )
        };

        let manifest_data = format!(
            "(data (i32.const {MANIFEST_PTR}) \"{}\")",
            wat_bytes(&spec.manifest_bytes)
        );
        let error_data = spec
            .last_error_bytes
            .as_ref()
            .map(|bytes| format!("(data (i32.const {ERROR_PTR}) \"{}\")", wat_bytes(bytes)))
            .unwrap_or_default();

        format!(
            r#"(module
  {memory_export}
  (global $free_count (export "free_count") (mut i32) (i32.const 0))
  (func {abi_version_export} (result i32)
    i32.const {abi_version}
  )
  (func {alloc_export} (param i32) (result i32)
    i32.const 4096
  )
  (func {free_export} (param i32 i32)
    global.get $free_count
    i32.const 1
    i32.add
    global.set $free_count
  )
  (func {describe_export} {describe_signature})
  (func {invoke_export} (param i32 i32) (result i64)
    i64.const 0
  )
  (func {last_error_export} {last_error_signature})
  {manifest_data}
  {error_data}
)"#,
            memory_export = memory_export,
            abi_version_export = abi_version_export,
            abi_version = spec.abi_version,
            alloc_export = alloc_export,
            free_export = free_export,
            describe_export = describe_export,
            describe_signature = describe_signature,
            invoke_export = invoke_export,
            last_error_export = last_error_export,
            last_error_signature = last_error_signature,
            manifest_data = manifest_data,
            error_data = error_data,
        )
    }

    fn export_name(name: &'static str, omitted: Option<&'static str>) -> String {
        if omitted == Some(name) {
            String::new()
        } else {
            format!("(export \"{name}\")")
        }
    }

    fn wat_bytes(bytes: &[u8]) -> String {
        let mut encoded = String::new();
        for byte in bytes {
            encoded.push_str(&format!("\\{:02x}", byte));
        }
        encoded
    }
}
