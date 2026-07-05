mod abi;
mod host_fetch;
mod manifest;

use async_trait::async_trait;
pub use manifest::{FsCapability, Limits, Manifest, ManifestError};
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use wasmtime::{Config, Engine, Instance, Linker, Module, Store};
use wasmtime_wasi::p1::{self, WasiP1Ctx};
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtxBuilder};

#[derive(Debug, thiserror::Error)]
pub enum WasmToolError {
    #[error("failed to read {path}: {source}")]
    ReadWasm {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(
        "plugin {path} has no manifest sidecar {expected} -- a .wasm with no manifest is a hard load error, not an implicit zero-capability load"
    )]
    MissingManifest { path: PathBuf, expected: PathBuf },
    #[error("failed to compile wasm module {path}: {source}")]
    Compile {
        path: PathBuf,
        #[source]
        source: wasmtime::Error,
    },
    #[error("failed to grant filesystem capability {host_dir} -> {guest_dir}: {source}")]
    Capability {
        host_dir: String,
        guest_dir: String,
        #[source]
        source: wasmtime::Error,
    },
    #[error("plugin {path} does not export a `memory`")]
    NoMemoryExport { path: PathBuf },
    #[error("plugin {path} is missing required export `{export}`: {source}")]
    MissingExport {
        path: PathBuf,
        export: &'static str,
        #[source]
        source: wasmtime::Error,
    },
    #[error("plugin {path} failed to instantiate: {source}")]
    Instantiate {
        path: PathBuf,
        #[source]
        source: wasmtime::Error,
    },
    #[error("failed to scan plugin directory {path}: {source}")]
    Scan {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// A tool backed by a sandboxed WASI `.wasm` module, discovered from
/// `.agent/tools/<name>.wasm` + a required sidecar `<name>.toml` capability
/// manifest. Implements the same `Tool` trait as built-in tools, so it slots
/// into the existing `Vec<Box<dyn Tool>>` with no changes to `AgentSession`
/// -- hook coverage via `execute_with_hooks` is automatic.
pub struct WasmTool {
    path: PathBuf,
    engine: Arc<Engine>,
    linker: Arc<Linker<WasiP1Ctx>>,
    module: Module,
    manifest: Manifest,
    name: String,
    description: String,
    parameters_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct RawOutcome {
    content: String,
    #[serde(default)]
    is_error: bool,
    #[serde(default)]
    metadata: serde_json::Value,
}

impl WasmTool {
    pub async fn load(wasm_path: &Path, manifest: Manifest, engine: Arc<Engine>) -> Result<Self, WasmToolError> {
        let bytes = tokio::fs::read(wasm_path).await.map_err(|e| WasmToolError::ReadWasm {
            path: wasm_path.to_path_buf(),
            source: e,
        })?;
        let module = Module::new(&engine, &bytes).map_err(|e| WasmToolError::Compile {
            path: wasm_path.to_path_buf(),
            source: e,
        })?;

        let mut linker = Linker::new(&engine);
        p1::add_to_linker_async(&mut linker, |ctx| ctx).map_err(|e| WasmToolError::Instantiate {
            path: wasm_path.to_path_buf(),
            source: e,
        })?;
        // Only linked in when the manifest grants it -- a plugin that
        // imports `host_web_fetch` without the capability simply fails to
        // instantiate (unresolved import), which is inherently safe.
        if manifest.network {
            host_fetch::link(&mut linker).map_err(|e| WasmToolError::Instantiate {
                path: wasm_path.to_path_buf(),
                source: e,
            })?;
        }

        let mut tool = WasmTool {
            path: wasm_path.to_path_buf(),
            engine,
            linker: Arc::new(linker),
            module,
            manifest,
            name: String::new(),
            description: String::new(),
            parameters_schema: serde_json::Value::Null,
        };

        let (mut store, instance) = tool.instantiate().await?;
        let memory = tool.get_memory(&mut store, &instance)?;
        tool.name = tool
            .call_metadata_export(&mut store, &instance, &memory, "minder_tool_name")
            .await?;
        tool.description = tool
            .call_metadata_export(&mut store, &instance, &memory, "minder_tool_description")
            .await?;
        let schema_raw = tool
            .call_metadata_export(&mut store, &instance, &memory, "minder_tool_parameters_schema")
            .await?;
        tool.parameters_schema = serde_json::from_str(&schema_raw).unwrap_or(serde_json::json!({
            "type": "object"
        }));

        Ok(tool)
    }

    fn build_wasi_ctx(&self) -> Result<WasiP1Ctx, WasmToolError> {
        let mut builder = WasiCtxBuilder::new();
        for cap in &self.manifest.fs {
            let (dir_perms, file_perms) = perms_for(cap);
            builder
                .preopened_dir(&cap.host_dir, &cap.guest_dir, dir_perms, file_perms)
                .map_err(|e| WasmToolError::Capability {
                    host_dir: cap.host_dir.clone(),
                    guest_dir: cap.guest_dir.clone(),
                    source: e,
                })?;
        }
        Ok(builder.build_p1())
    }

    async fn instantiate(&self) -> Result<(Store<WasiP1Ctx>, Instance), WasmToolError> {
        let wasi = self.build_wasi_ctx()?;
        let mut store = Store::new(&self.engine, wasi);
        store
            .set_fuel(self.manifest.limits.fuel)
            .map_err(|e| WasmToolError::Instantiate {
                path: self.path.clone(),
                source: e,
            })?;
        // Lets the fuel-metered async execution actually yield at fuel
        // checkpoints, which is what makes wrapping the call in
        // `tokio::time::timeout` an effective wall-clock cutoff even for a
        // non-cooperative guest loop -- without this, fuel exhaustion still
        // traps eventually, but a `tokio::time::timeout` around a call that
        // never yields would never get a chance to fire first.
        store
            .fuel_async_yield_interval(Some(10_000))
            .map_err(|e| WasmToolError::Instantiate {
                path: self.path.clone(),
                source: e,
            })?;

        let instance = self
            .linker
            .instantiate_async(&mut store, &self.module)
            .await
            .map_err(|e| WasmToolError::Instantiate {
                path: self.path.clone(),
                source: e,
            })?;
        Ok((store, instance))
    }

    fn get_memory(&self, store: &mut Store<WasiP1Ctx>, instance: &Instance) -> Result<wasmtime::Memory, WasmToolError> {
        instance
            .get_memory(&mut *store, "memory")
            .ok_or_else(|| WasmToolError::NoMemoryExport {
                path: self.path.clone(),
            })
    }

    async fn call_metadata_export(
        &self,
        store: &mut Store<WasiP1Ctx>,
        instance: &Instance,
        memory: &wasmtime::Memory,
        export: &'static str,
    ) -> Result<String, WasmToolError> {
        let func =
            instance
                .get_typed_func::<(), i64>(&mut *store, export)
                .map_err(|e| WasmToolError::MissingExport {
                    path: self.path.clone(),
                    export,
                    source: e,
                })?;
        let packed = func
            .call_async(&mut *store, ())
            .await
            .map_err(|e| WasmToolError::MissingExport {
                path: self.path.clone(),
                export,
                source: e,
            })?;
        Ok(abi::read_packed_string(store, memory, packed))
    }
}

fn perms_for(cap: &FsCapability) -> (DirPerms, FilePerms) {
    if cap.read_only {
        (DirPerms::READ, FilePerms::READ)
    } else {
        (DirPerms::all(), FilePerms::all())
    }
}

#[async_trait]
impl Tool for WasmTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.parameters_schema.clone()
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        match self.run(arguments, ctx).await {
            Ok(outcome) => outcome,
            Err(e) => error_outcome(e.to_string()),
        }
    }
}

impl WasmTool {
    async fn run(&self, arguments: serde_json::Value, ctx: &ToolContext) -> Result<ToolExecOutcome, WasmToolError> {
        let (mut store, instance) = self.instantiate().await?;
        let memory = self.get_memory(&mut store, &instance)?;

        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, "minder_alloc")
            .map_err(|e| WasmToolError::MissingExport {
                path: self.path.clone(),
                export: "minder_alloc",
                source: e,
            })?;
        let dealloc = instance
            .get_typed_func::<(i32, i32), ()>(&mut store, "minder_dealloc")
            .map_err(|e| WasmToolError::MissingExport {
                path: self.path.clone(),
                export: "minder_dealloc",
                source: e,
            })?;
        let execute_fn = instance
            .get_typed_func::<(i32, i32), i64>(&mut store, "minder_tool_execute")
            .map_err(|e| WasmToolError::MissingExport {
                path: self.path.clone(),
                export: "minder_tool_execute",
                source: e,
            })?;

        let args_bytes = serde_json::to_vec(&arguments).unwrap_or_default();
        let args_ptr = alloc
            .call_async(&mut store, args_bytes.len() as i32)
            .await
            .map_err(|e| WasmToolError::MissingExport {
                path: self.path.clone(),
                export: "minder_alloc",
                source: e,
            })?;
        if memory.write(&mut store, args_ptr as usize, &args_bytes).is_err() {
            return Ok(error_outcome(
                "plugin allocated a buffer too small for the arguments".to_string(),
            ));
        }

        let timeout = Duration::from_secs(self.manifest.limits.timeout_secs);
        let call = execute_fn.call_async(&mut store, (args_ptr, args_bytes.len() as i32));

        let packed = tokio::select! {
            result = tokio::time::timeout(timeout, call) => {
                match result {
                    Ok(Ok(packed)) => packed,
                    Ok(Err(trap)) => return Ok(error_outcome(format!("plugin trapped: {trap}"))),
                    Err(_) => return Ok(error_outcome(format!(
                        "plugin timed out after {}s", timeout.as_secs()
                    ))),
                }
            }
            _ = ctx.cancel.cancelled() => {
                return Ok(error_outcome("plugin execution cancelled".to_string()));
            }
        };

        let _ = dealloc
            .call_async(&mut store, (args_ptr, args_bytes.len() as i32))
            .await;

        let (result_ptr, result_len) = abi::unpack(packed);
        let result_bytes = abi::read_bytes(&store, &memory, result_ptr, result_len);
        let _ = dealloc
            .call_async(&mut store, (result_ptr as i32, result_len as i32))
            .await;

        let raw: RawOutcome = match serde_json::from_slice(&result_bytes) {
            Ok(r) => r,
            Err(e) => {
                return Ok(error_outcome(format!("plugin returned invalid result JSON: {e}")));
            }
        };

        Ok(ToolExecOutcome {
            content: raw.content,
            is_error: raw.is_error,
            metadata: raw.metadata,
        })
    }
}

fn error_outcome(message: String) -> ToolExecOutcome {
    ToolExecOutcome {
        content: message,
        is_error: true,
        metadata: serde_json::Value::Null,
    }
}

/// Builds the shared, process-wide `Engine` used for every loaded plugin.
/// Fuel accounting and async support are enabled here once, at the engine
/// level, rather than per-plugin.
fn build_engine() -> Result<Engine, wasmtime::Error> {
    let mut config = Config::new();
    config.consume_fuel(true);
    Engine::new(&config)
}

/// Discovers and loads every `.agent/tools/*.wasm` plugin (each requiring a
/// sibling `.toml` manifest) under `agent_dir`. Mirrors
/// `HookEngine::load(&working_dir.join(".agent"))`'s optionality: a missing
/// `tools/` directory is not an error, `Ok(vec![])`. Any plugin that fails
/// to load (missing manifest, bad wasm, missing exports) is a hard error --
/// same as a hook load failure.
pub async fn load_plugins(agent_dir: &Path) -> Result<Vec<Box<dyn Tool>>, WasmToolError> {
    let tools_dir = agent_dir.join("tools");
    if !tools_dir.exists() {
        return Ok(Vec::new());
    }

    let engine = Arc::new(build_engine().map_err(|e| WasmToolError::Instantiate {
        path: tools_dir.clone(),
        source: e,
    })?);

    let mut entries = tokio::fs::read_dir(&tools_dir).await.map_err(|e| WasmToolError::Scan {
        path: tools_dir.clone(),
        source: e,
    })?;

    let mut wasm_paths = Vec::new();
    while let Some(entry) = entries.next_entry().await.map_err(|e| WasmToolError::Scan {
        path: tools_dir.clone(),
        source: e,
    })? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("wasm") {
            wasm_paths.push(path);
        }
    }
    wasm_paths.sort();

    let mut tools: Vec<Box<dyn Tool>> = Vec::new();
    for wasm_path in wasm_paths {
        let manifest_path = wasm_path.with_extension("toml");
        if !manifest_path.exists() {
            return Err(WasmToolError::MissingManifest {
                path: wasm_path,
                expected: manifest_path,
            });
        }
        let manifest = Manifest::load(&manifest_path)?;
        let tool = WasmTool::load(&wasm_path, manifest, engine.clone()).await?;
        tools.push(Box::new(tool));
    }

    Ok(tools)
}
