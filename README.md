<div align="center">
  <img src="assets/logo.svg" style="width: 128px; height: 128px;"/>

<h1>minder</h1>

**A coding-agent harness in Rust.**

</div>

Multi-provider (Anthropic, OpenAI, Gemini, Ollama), with policy/observability hooks written in [`mq`](https://github.com/harehare/mq)'s embeddable query language instead of a general-purpose scripting language.

`mq-lang` has no builtin for file writes, network requests, or process execution, so hook scripts under `.agent/hooks/*.mq` can observe, block, or transform what the agent does without being able to do anything unsafe themselves — the host mediates all real side effects.

The agent loop itself is a standard ReAct-style tool-calling loop: the LLM's own response (does it emit tool calls or not) drives whether the loop continues, not the hooks. Hooks only answer narrow policy questions at five fixed interception points.

> [!IMPORTANT]
> This project is under active development and has not been thoroughly tested end to end yet. Providers, tools, and hooks work individually in unit tests, but the full agent loop hasn't seen broad real-world verification — expect rough edges.

See `crates/agent-core`, `crates/agent-providers`, `crates/agent-tools`, `crates/agent-tools-wasm`, `crates/agent-hooks`, `crates/agent-cli`.

## Usage

```sh
# Anthropic (default)
ANTHROPIC_API_KEY=... cargo run -p agent-cli -- "run the tests and summarize failures"

# OpenAI
MINDER_PROVIDER=openai OPENAI_API_KEY=... cargo run -p agent-cli -- "..."

# Gemini
MINDER_PROVIDER=gemini GEMINI_API_KEY=... cargo run -p agent-cli -- "..."

# Ollama (local, no key needed)
MINDER_PROVIDER=ollama MINDER_MODEL=llama3.2 cargo run -p agent-cli -- "..."
```

`MINDER_MODEL` overrides the model for any provider. Tools available: `read_file`, `write_file`, `edit_file`, `bash`, `glob`, `grep`, `ls`, `git_diff`, `git_log`, `git_status`, `git_commit`, `web_fetch`, and `web_search` (only registered when `TAVILY_API_KEY` is set).

## Hooks

```
.agent/hooks/security.mq
```

```mq
def on_tool_call(call):
  if (call["name"] == "bash" && contains(call["arguments"]["command"], "rm -rf")):
    {"action": "block", "reason": "destructive bash command blocked by policy"}
  else:
    {"action": "allow", "value": call};
```

Every hook returns `{"action": "allow", "value": ...}` or `{"action": "block", "reason": "..."}` (the
gate-only hook `before_compact` returns `{"action": "allow"}` with no `value`). Hooks are
optional — if a hook function isn't defined, the corresponding interception point is a no-op. A
buggy `on_tool_call` fails **closed** (blocks the action); every other hook point fails **open**
(the buggy transform is skipped).

| Hook point | mq function | Fires |
|---|---|---|
| `before_agent_start` | `before_agent_start(prompt)` | Once, before the first LLM call |
| `on_context` | `on_context(messages)` | Before every LLM call |
| `on_tool_call` | `on_tool_call(call)` | Before a tool executes (fails closed) |
| `on_tool_result` | `on_tool_result(result)` | Before a tool's result re-enters history |
| `before_compact` | `before_compact(messages)` | Before history is truncated under context pressure |

See `hooks/security.mq` for a runnable example (copy it to `.agent/hooks/` in a project to try it).

## Tool plugins (WASM)

Tools can also be provided by sandboxed WASI plugins, discovered from `.agent/tools/`:

```
.agent/tools/weather.wasm
.agent/tools/weather.toml
```

Every `.wasm` requires a sidecar `.toml` manifest of the same name declaring its capabilities — a
plugin with no manifest fails to load, it does not silently run with zero capabilities:

```toml
network = false          # grants the one host-mediated fetch primitive (see below)

[[fs]]
host_dir = "./data"      # resolved relative to the working directory
guest_dir = "/data"
read_only = true

[limits]
timeout_secs = 30
max_memory_pages = 256
fuel = 5_000_000
```

Plugins are plain `wasm32-wasip1` modules (no component model) exporting `minder_tool_name`,
`minder_tool_description`, `minder_tool_parameters_schema`, `minder_tool_execute`, plus
`minder_alloc`/`minder_dealloc` for passing JSON across linear memory — see
`crates/agent-tools-wasm/tests/fixtures/echo_plugin` for a minimal example and
`crates/agent-tools-wasm/tests/fixtures/regenerate.sh` for the build command. Filesystem access is
granted per-plugin via WASI preopens (no grant, no access); network access is not a raw socket —
plugins with `network = true` get a single `host_web_fetch` import that reuses the same
SSRF-guarded path as the built-in `web_fetch` tool. Execution is metered with wasmtime fuel, so a
runaway plugin traps instead of hanging.

## Development

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace
```

Each provider's live round-trip test needs real credentials and is `#[ignore]`d by default:

```sh
ANTHROPIC_API_KEY=... cargo test -p agent-providers -- --ignored anthropic
OPENAI_API_KEY=... cargo test -p agent-providers -- --ignored openai
GEMINI_API_KEY=... cargo test -p agent-providers -- --ignored gemini
cargo test -p agent-providers -- --ignored ollama  # needs `ollama serve` running locally
```

## License

MIT
