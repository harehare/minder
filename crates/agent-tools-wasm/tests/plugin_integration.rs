//! Exercises `WasmTool`/`load_plugins` against the prebuilt fixture
//! binaries in `tests/fixtures/*.wasm` (see `tests/fixtures/regenerate.sh`
//! for how to rebuild them from source). Mirrors the cross-crate
//! integration-test precedent in `crates/agent-hooks/tests/session_integration.rs`.

use agent_core::{Tool, ToolContext};
use agent_tools_wasm::{FsCapability, Limits, Manifest, WasmTool};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wasmtime::{Config, Engine};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn engine() -> Arc<Engine> {
    let mut config = Config::new();
    config.consume_fuel(true);
    Arc::new(Engine::new(&config).unwrap())
}

fn ctx() -> ToolContext {
    ToolContext {
        working_dir: std::env::temp_dir(),
        session_id: "test".to_string(),
        cancel: tokio_util::sync::CancellationToken::new(),
    }
}

#[tokio::test]
async fn echo_plugin_round_trips_through_the_full_abi() {
    let tool = WasmTool::load(&fixture("echo_plugin.wasm"), Manifest::default(), engine())
        .await
        .unwrap();
    assert_eq!(tool.name(), "echo");
    assert!(!tool.description().is_empty());

    let outcome = tool
        .execute(serde_json::json!({"hello": "world"}), &ctx())
        .await;
    assert!(!outcome.is_error);
    assert!(outcome.content.contains("hello"));
    assert!(outcome.content.contains("world"));
}

#[tokio::test]
async fn panicking_plugin_traps_are_converted_to_tool_errors_not_host_panics() {
    let tool = WasmTool::load(
        &fixture("panicking_plugin.wasm"),
        Manifest::default(),
        engine(),
    )
    .await
    .unwrap();

    let outcome = tool.execute(serde_json::json!({}), &ctx()).await;
    assert!(outcome.is_error);
    assert!(outcome.content.contains("trapped") || outcome.content.contains("panic"));
}

#[tokio::test]
async fn slow_loop_plugin_is_aborted_by_fuel_exhaustion() {
    let manifest = Manifest {
        limits: Limits {
            fuel: 200_000,
            ..Default::default()
        },
        ..Default::default()
    };
    let tool = WasmTool::load(&fixture("slow_loop_plugin.wasm"), manifest, engine())
        .await
        .unwrap();

    let started = std::time::Instant::now();
    let outcome = tool.execute(serde_json::json!({}), &ctx()).await;
    assert!(outcome.is_error);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "fuel exhaustion should abort the loop quickly, took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn fs_probe_plugin_is_denied_access_it_was_not_granted() {
    // Zero `fs` capabilities -- the plugin must see the WASI-level denial
    // itself, not have the host special-case the attempt.
    let tool = WasmTool::load(
        &fixture("fs_probe_plugin.wasm"),
        Manifest::default(),
        engine(),
    )
    .await
    .unwrap();

    let outcome = tool.execute(serde_json::json!({}), &ctx()).await;
    assert!(outcome.is_error);
    assert!(outcome.content.contains("denied"));
}

#[tokio::test]
async fn fs_probe_plugin_with_unrelated_grant_is_still_denied() {
    // Granting access to *some* directory must not leak access to a path
    // outside that grant.
    let dir = std::env::temp_dir().join(format!(
        "minder-wasm-fs-grant-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = Manifest {
        fs: vec![FsCapability {
            host_dir: dir.to_string_lossy().into_owned(),
            guest_dir: "/granted".to_string(),
            read_only: true,
        }],
        ..Default::default()
    };
    let tool = WasmTool::load(&fixture("fs_probe_plugin.wasm"), manifest, engine())
        .await
        .unwrap();

    let outcome = tool.execute(serde_json::json!({}), &ctx()).await;
    assert!(outcome.is_error);
    assert!(outcome.content.contains("denied"));
}

#[tokio::test]
async fn load_plugins_returns_empty_when_tools_dir_is_absent() {
    let dir = std::env::temp_dir().join(format!("minder-wasm-notools-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let tools = agent_tools_wasm::load_plugins(&dir).await.unwrap();
    assert!(tools.is_empty());
}

#[tokio::test]
async fn load_plugins_errors_on_missing_manifest_sidecar() {
    let dir = std::env::temp_dir().join(format!("minder-wasm-missing-manifest-{}", uuid::Uuid::new_v4()));
    let tools_dir = dir.join("tools");
    std::fs::create_dir_all(&tools_dir).unwrap();
    std::fs::copy(fixture("echo_plugin.wasm"), tools_dir.join("echo_plugin.wasm")).unwrap();
    // deliberately no echo_plugin.toml sidecar

    let result = agent_tools_wasm::load_plugins(&dir).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn net_probe_plugin_host_fetch_enforces_the_same_ssrf_guard_as_web_fetch_tool() {
    // `host_web_fetch` calls the exact same `agent_tools::fetch` the
    // built-in `web_fetch` tool uses, so it applies the same guard --
    // proven here by a granted-network plugin still being refused a
    // loopback address. (This also means a real `wiremock` mock server,
    // which binds to loopback, can't be used to test the success path
    // from this crate; see the `#[ignore]`d live test below instead.)
    let manifest = Manifest {
        network: true,
        ..Default::default()
    };
    let tool = WasmTool::load(&fixture("net_probe_plugin.wasm"), manifest, engine())
        .await
        .unwrap();

    let outcome = tool
        .execute(serde_json::json!({"url": "http://127.0.0.1/secret"}), &ctx())
        .await;
    assert!(outcome.is_error);
    assert!(outcome.content.contains("loopback"));
}

#[tokio::test]
#[ignore]
async fn net_probe_plugin_with_network_capability_reaches_the_host_fetch_live() {
    let manifest = Manifest {
        network: true,
        ..Default::default()
    };
    let tool = WasmTool::load(&fixture("net_probe_plugin.wasm"), manifest, engine())
        .await
        .unwrap();

    let outcome = tool
        .execute(serde_json::json!({"url": "https://example.com"}), &ctx())
        .await;
    assert!(!outcome.is_error);
    assert!(outcome.content.contains("\"status\":200"));
}

#[tokio::test]
async fn net_probe_plugin_without_network_capability_fails_to_instantiate() {
    // `network = false` (the default) means `host_web_fetch` is never
    // linked in, so a plugin importing it fails at instantiation time
    // (unresolved import) -- proven here by `WasmTool::load` itself
    // failing, since load instantiates once to read metadata exports.
    let result = WasmTool::load(
        &fixture("net_probe_plugin.wasm"),
        Manifest::default(),
        engine(),
    )
    .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn load_plugins_discovers_and_loads_a_valid_plugin() {
    let dir = std::env::temp_dir().join(format!("minder-wasm-discover-{}", uuid::Uuid::new_v4()));
    let tools_dir = dir.join("tools");
    std::fs::create_dir_all(&tools_dir).unwrap();
    std::fs::copy(fixture("echo_plugin.wasm"), tools_dir.join("echo_plugin.wasm")).unwrap();
    std::fs::write(tools_dir.join("echo_plugin.toml"), "").unwrap();

    let tools = agent_tools_wasm::load_plugins(&dir).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "echo");
}
