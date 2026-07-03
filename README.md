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

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Providers](#providers)
- [Tools](#tools)
- [Hooks](#hooks)
- [Tool plugins (WASM)](#tool-plugins-wasm)
- [Project layout](#project-layout)
- [Development](#development)

## Install

Requires a recent stable Rust toolchain (`rustup` recommended).

```sh
git clone https://github.com/harehare/minder.git
cd minder
cargo build --workspace --release
```

Run the CLI in place with `cargo run -p agent-cli --`, or install the `minder` binary onto your
`PATH`:

```sh
cargo install --path crates/agent-cli
minder "..."
```

The rest of this README uses `minder "..."` for brevity — substitute `cargo run -p agent-cli --
"..."` if you'd rather not install the binary.

## Quick start

minder takes a single task string as its only argument and runs it to completion, printing the
final assistant message to stdout. There's no interactive chat mode yet — one process, one task.

```sh
$ export ANTHROPIC_API_KEY=sk-ant-...
$ cd path/to/some/project
$ minder "list the top-level files and summarize what this project does"
loaded hooks from .agent/            # only printed if .agent/hooks/*.mq exist
The project is a Rust workspace with six crates under crates/... (etc.)
```

Behind the scenes each turn runs a standard tool-calling loop: the model reads your prompt,
decides whether it needs a tool (`read_file`, `bash`, `grep`, ...), the CLI executes it in your
current working directory, the result is fed back to the model, and this repeats until the model
replies without requesting another tool call. Everything the agent touches — files read/written,
commands run — is scoped to the directory `minder` was launched from.

A few real tasks to try:

```sh
minder "run the tests and summarize any failures"
minder "find all TODO comments under src/ and turn them into a checklist"
minder "explain what crates/agent-hooks/src/lib.rs does"
minder "check git status and stage+commit the pending changes with a sensible message"
```

Because `bash` and `write_file`/`edit_file` are unrestricted by default, drop a hook (see
[Hooks](#hooks)) into `.agent/hooks/` for any project where you want a policy layer between the
model and your filesystem/shell before pointing it at real work.

## Providers

Selected via `MINDER_PROVIDER`; `MINDER_MODEL` overrides the default model for whichever provider
is active.

| `MINDER_PROVIDER` | Required env | Default model | Notes |
|---|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5-20250929` | |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` | |
| `ollama` | none | `llama3.2` | needs a local `ollama serve`; override the endpoint with `OLLAMA_BASE_URL` |

```sh
# Anthropic (default)
ANTHROPIC_API_KEY=... minder "run the tests and summarize failures"

# OpenAI
MINDER_PROVIDER=openai OPENAI_API_KEY=... MINDER_MODEL=gpt-4o minder "..."

# Gemini
MINDER_PROVIDER=gemini GEMINI_API_KEY=... minder "..."

# Ollama (local, no key needed)
MINDER_PROVIDER=ollama MINDER_MODEL=llama3.2 minder "..."
OLLAMA_BASE_URL=http://localhost:11434 MINDER_PROVIDER=ollama minder "..."
```

## Tools

Always registered:

| Tool | Does |
|---|---|
| `read_file` | Reads a file, optionally restricted to a 1-indexed inclusive line range |
| `write_file` | Creates or overwrites a file, creating parent directories as needed |
| `edit_file` | Replaces `old_string` with `new_string` in a file (must match exactly once unless `replace_all`) |
| `bash` | Runs a shell command and returns combined stdout/stderr (default 120s timeout) |
| `glob` | Finds files matching a glob pattern, e.g. `**/*.rs` |
| `grep` | Searches file contents by regex, honoring `.gitignore`, up to 200 matches |
| `ls` | Lists a directory, honoring `.gitignore`; `recursive` for a tree view, up to 500 entries |
| `git_diff` | Shows `git diff` output, against a ref, staged or unstaged, optionally path-scoped |
| `git_log` | Shows commit history |
| `git_status` | Shows `git status` |
| `git_commit` | Creates a commit |
| `web_fetch` | Fetches an http(s) URL as text; rejects non-http(s) schemes and literal loopback/private-network hosts (partial SSRF guard, not a complete one — use a hook for stronger guarantees) |

Registered only when configured:

| Tool | Enabled by |
|---|---|
| `web_search` | `TAVILY_API_KEY` set — omitted entirely otherwise, so the model never sees a tool it can't use |

Additional tools can be supplied per-project as WASM plugins — see [Tool plugins (WASM)](#tool-plugins-wasm).

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

## Project layout

| Crate | Responsibility |
|---|---|
| `agent-core` | Session/turn loop, tool-calling protocol, hook port trait |
| `agent-providers` | Anthropic/OpenAI/Gemini/Ollama client implementations |
| `agent-tools` | Built-in tools (file, shell, git, web) |
| `agent-tools-wasm` | WASI plugin loader and sandboxed host runtime |
| `agent-hooks` | `mq`-based hook engine (`.agent/hooks/*.mq`) |
| `agent-cli` | The `minder` binary — wires providers/tools/hooks together |

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
