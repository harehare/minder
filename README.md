<div align="center">
  <img src="assets/logo.svg" style="width: 128px; height: 128px;"/>

<h1>minder</h1>

**A coding-agent harness in Rust.**

[![CI](https://github.com/harehare/minder/actions/workflows/ci.yml/badge.svg)](https://github.com/harehare/minder/actions/workflows/ci.yml)
[![Security audit](https://github.com/harehare/minder/actions/workflows/audit.yml/badge.svg)](https://github.com/harehare/minder/actions/workflows/audit.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Website](https://harehare.github.io/minder/)

</div>

Multi-provider (Anthropic, OpenAI, Gemini, Ollama) coding-agent harness with policy/observability
hooks written in [`mq`](https://github.com/harehare/mq)'s embeddable query language rather than a
general-purpose scripting language — `mq-lang` has no builtin for file writes, network requests, or
process execution, so hooks can observe, block, or transform agent behavior without being able to
cause side effects themselves.

The agent loop is a standard ReAct-style tool-calling loop; hooks only answer narrow policy
questions at five fixed interception points and never drive the loop itself. Every tool call and
result streams live to the terminal — see [Live execution display](#live-execution-display).
`mq-lang` also powers the harness's [autonomous loop mode](#autonomous-loop-mode): `minder loop
TODO.md` re-queries a Markdown checklist after each turn and keeps working through unchecked items,
with no user in the loop and no external `mq` binary required.

> [!IMPORTANT]
> This project is under active development and has not been thoroughly tested end to end yet. Providers, tools, and hooks work individually in unit tests, but the full agent loop hasn't seen broad real-world verification — expect rough edges.

See `crates/minder-core`, `crates/minder-providers`, `crates/minder-tools`, `crates/minder-tools-wasm`, `crates/minder-tools-mcp`, `crates/minder-hooks`, `crates/minder-cli`.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Providers](#providers)
- [Tools](#tools)
- [Skills](#skills)
- [Hooks](#hooks)
- [Tool plugins (WASM)](#tool-plugins-wasm)
- [MCP servers (optional)](#mcp-servers-optional)
- [Autonomous loop mode](#autonomous-loop-mode)
- [Project layout](#project-layout)
- [Development](#development)

## Install

Requires a recent stable Rust toolchain (`rustup` recommended).

```sh
cargo install minder-cli
minder "..."
```

Or build from a clone:

```sh
git clone https://github.com/harehare/minder.git
cd minder
cargo build --workspace --release
```

Run the CLI in place with `cargo run -p minder-cli --`, or install the `minder` binary onto your
`PATH`:

```sh
cargo install --path crates/minder-cli
minder "..."
```

Prebuilt binaries for Linux/macOS/Windows are also attached to each [GitHub
Release](https://github.com/harehare/minder/releases).

The rest of this README uses `minder "..."` for brevity — substitute `cargo run -p minder-cli --
"..."` if you'd rather not install the binary.

## Quick start

minder takes a single task string as its only argument and runs it to completion. There's no
interactive chat mode yet — one process, one task.

```sh
$ export ANTHROPIC_API_KEY=sk-ant-...
$ cd path/to/some/project
$ minder "list the top-level files and summarize what this project does"
loaded hooks from .agent/            # only printed if .agent/hooks/*.mq exist
→ ls recursive=false
✓ ls: Cargo.toml  README.md  crates/  ...
The project is a Rust workspace with six crates under crates/... (etc.)
```

Each turn: the model reads the prompt, optionally calls a tool (`read_file`, `bash`, `grep`, ...),
the CLI runs it in the current working directory, and the result feeds back — repeating until the
model replies without requesting another tool call. Everything the agent touches — files
read/written, commands run — is scoped to the directory `minder` was launched from.

### Live execution display

Every tool call streams to the terminal as it happens, not just the final answer — useful for
watching the agent work and for debugging a stuck turn. Output is split across two streams so
piping `minder`'s answer elsewhere stays clean:

- **stdout** — the conversation itself: any assistant text, including commentary the model emits
  on turns where it also calls a tool (previously dropped silently, now shown live).
- **stderr** — the execution trace: `● tool_name(key=value)` before each call, then either a diff
  stat line (`+N -N`) followed by a colorized, indented unified diff (for `write_file`/`edit_file`
  — capped at 40 lines with a `… N more line(s)` trailer so one big rewrite can't flood the
  terminal) or a `✓`/`✗` one-line result summary.

```sh
$ minder "fix the off-by-one in the pagination helper"
● grep(pattern=page_size)
  ✓ src/pagination.rs:42:    let end = start + page_size;
● edit_file(path=src/pagination.rs)
  ✓ +1 -1
  --- a/src/pagination.rs
  +++ b/src/pagination.rs
  @@ -40,2 +40,2 @@
  - let end = start + page_size;
  + let end = start + page_size - 1;
Fixed the off-by-one: `end` was one past the last valid index.
```

Colors turn off automatically when stderr isn't a terminal (e.g. redirected to a file) or when
`NO_COLOR` is set. Every line here is also overridable per-project from `.agent/hooks/*.mq` — see
[Customizing the display](#customizing-the-display) under Hooks.

A few real tasks to try:

```sh
minder "run the tests and summarize any failures"
minder "find all TODO comments under src/ and turn them into a checklist"
minder "explain what crates/minder-hooks/src/lib.rs does"
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
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-5` | |
| `openai` | `OPENAI_API_KEY` | `gpt-5.4-mini` | |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.5-flash` | |
| `ollama` | none | `llama3.2` | needs a local `ollama serve`; override the endpoint with `OLLAMA_BASE_URL` |

```sh
# Anthropic (default)
ANTHROPIC_API_KEY=... minder "run the tests and summarize failures"

# OpenAI
MINDER_PROVIDER=openai OPENAI_API_KEY=... MINDER_MODEL=gpt-5.4 minder "..."

# Gemini
MINDER_PROVIDER=gemini GEMINI_API_KEY=... minder "..."

# Ollama (local, no key needed)
MINDER_PROVIDER=ollama MINDER_MODEL=llama3.2 minder "..."
OLLAMA_BASE_URL=http://localhost:11434 MINDER_PROVIDER=ollama minder "..."
```

### Running with gpt-oss

[gpt-oss](https://openai.com/index/introducing-gpt-oss/) (OpenAI's open-weight models,
`gpt-oss-20b`/`gpt-oss-120b`) runs through the existing `ollama` provider — no minder changes
needed, since Ollama handles the gpt-oss-specific translation over its generic `/api/chat`
endpoint.

1. Install Ollama v0.11.4+ (needed for correct gpt-oss tool-calling support):
   <https://ollama.com/download>, or:

   ```sh
   # macOS
   brew install ollama
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Start the server (skip this if your install already runs it as a background service):

   ```sh
   ollama serve
   ```
3. Pull a gpt-oss model (`20b` needs ~16GB RAM/VRAM; `120b` needs ~65GB+, multi-GPU/datacenter-class
   hardware) — start with `20b` unless you know you have the headroom:

   ```sh
   ollama pull gpt-oss:20b
   # or, if your hardware can take it:
   ollama pull gpt-oss:120b
   ```
4. Point minder at it:

   ```sh
   MINDER_PROVIDER=ollama MINDER_MODEL=gpt-oss:20b minder "..."

   # remote/non-default Ollama host:
   OLLAMA_BASE_URL=http://your-ollama-host:11434 MINDER_PROVIDER=ollama MINDER_MODEL=gpt-oss:20b minder "..."
   ```

gpt-oss's reasoning effort (low/medium/high) isn't configurable through minder today — it runs at
Ollama's default for the model. `minder loop` (see below) works the same way with `gpt-oss` as
with any other provider, since it drives `AgentSession::run_turn` generically.

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
| `skill` | one or more `.agent/skills/*/SKILL.md` files present — see [Skills](#skills) |

Additional tools can be supplied per-project as WASM plugins — see [Tool plugins (WASM)](#tool-plugins-wasm)
— or from MCP servers, behind an opt-in feature — see [MCP servers (optional)](#mcp-servers-optional).

## Skills

```
.agent/skills/commit-messages/SKILL.md
```

```markdown
---
name: commit-messages
description: Writes commit messages in this repo's conventional-commit style
---
# Commit messages

Use Conventional Commits: `<type>(<scope>): <summary>`, imperative mood...
```

Each skill is a directory with a `SKILL.md`: `---`-delimited frontmatter (`name`, `description`)
followed by instructions as the body. minder discovers every `.agent/skills/*/SKILL.md` at startup
and registers a single `skill` tool listing each skill's name/description — cheap to keep in
context every turn. The model calls `skill` with a `name` to pull that skill's full body into the
conversation only when it's actually relevant.

Skill names must be unique, and startup fails if a `SKILL.md` is missing frontmatter or the
`name`/`description` fields. See `skills/commit-messages/SKILL.md` for a runnable example (copy
`skills/` to `.agent/skills/` in a project to try it).

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

Every hook returns `{"action": "allow", "value": ...}` or `{"action": "block", "reason": "..."}`
(the gate-only hook `before_compact` returns `{"action": "allow"}` with no `value`). Hooks are
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

### The `agent` module

A small set of convenience functions is always loaded before any hook file, bare-callable with no
`import` needed (an `agent_` prefix keeps them out of your own functions' way — redefine one
yourself and your version simply shadows it):

| Function | Returns |
|---|---|
| `agent_content_blocks(messages)` | Every message's `content` array, flattened into one array of blocks |
| `agent_tool_calls(messages)` | All `tool_use` blocks so far, unwrapped to `{id, name, arguments}` |
| `agent_tool_results(messages)` | All `tool_result` blocks so far, unwrapped to `{tool_call_id, content, is_error}` |
| `agent_assistant_texts(messages)` | Every assistant-authored text string, in order |
| `agent_tool_names(messages)` | Distinct tool names called so far |
| `agent_error_count(messages)` | Count of tool results with `is_error: true` |
| `agent_consecutive_errors(results)` | Trailing streak of `is_error: true` results (feed it `agent_tool_results(messages)`) |
| `agent_last_n(items, n)` | The last `n` items of any array |

```mq
# .agent/hooks/circuit_breaker.mq -- stop the turn once 3 tool calls in a row have failed.
# on_context sees the full history (on_tool_call only sees the one call about to run), so
# that's where a check like this belongs.
def on_context(messages):
  if (agent_consecutive_errors(agent_tool_results(messages)) >= 3):
    {"action": "block", "reason": "3 consecutive tool failures -- pausing for a human"}
  else:
    {"action": "allow", "value": messages};
```

### Overriding a tool's result

`on_tool_call` can also `override`: `{"action": "override", "value": {"content": "...",
"is_error": false, "metadata": null}}` supplies the tool's result directly. The real tool never
runs, but the result still flows through `on_tool_result` afterward like any other, so
post-processing hooks stay uniform either way. Useful for mocking a tool in tests, or for
short-circuiting it once some condition (like `agent_consecutive_errors` above) is met without
just erroring out:

```mq
def on_tool_call(call):
  if (call["name"] == "web_fetch"):
    {"action": "override", "value": {"content": "(network disabled in this environment)", "is_error": false, "metadata": None}}
  else:
    {"action": "allow", "value": call};
```

### Customizing the display

The [live execution display](#live-execution-display) is driven by two more optional hook
functions, checked before minder's own built-in formatting — same files, same loading, nothing
extra to set up. Both fail **open**: a broken or undefined render function falls back to the
built-in look, since a display bug should never be able to affect what the agent actually does.

| Function | Called with | Controls |
|---|---|---|
| `render_tool_call(call)` | the upcoming `ToolCall` | how the `● name(...)` header line prints |
| `render_tool_result(arg)` | `{"call": ToolCall, "outcome": ToolExecOutcome}` | how the result/diff line(s) print |

Each returns `{"action": "default"}` (use the built-in formatting), `{"action": "hide"}` (print
nothing), or `{"action": "text", "value": "...", "style": "..."}` (print this instead — `style` is
one of `green`/`red`/`yellow`/`cyan`/`dim`/`bold`, or omitted/anything else for no styling):

```mq
# .agent/hooks/display.mq -- quiet git_status noise, and prefix bash calls with a shell-style `$`
def render_tool_call(call):
  if (call["name"] == "git_status"):
    {"action": "hide"}
  else:
    {"action": "default"};

def render_tool_result(arg):
  if (arg["call"]["name"] == "bash"):
    {"action": "text", "value": "$ " + arg["call"]["arguments"]["command"], "style": "cyan"}
  else:
    {"action": "default"};
```

Both `render_tool_call` and (via `arg["call"]`) `render_tool_result` only see the one call/outcome
in front of them, not the conversation — if a display decision needs history, reach for the
[`agent` module](#the-agent-module) inside `on_context`/`before_compact` (which do see `messages`)
and have that hook stash whatever's needed back onto the call/result some other way (e.g. blocking
before it ever reaches the display layer).

## Tool plugins (WASM)

Tools can also be provided by sandboxed WASI plugins, discovered from `.agent/tools/`:

```
.agent/tools/weather.wasm
.agent/tools/weather.toml
```

Every `.wasm` needs a sidecar `.toml` manifest of the same name declaring its capabilities — a
plugin with no manifest fails to load rather than silently running with zero capabilities:

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
`crates/minder-tools-wasm/tests/fixtures/echo_plugin` for a minimal example (`regenerate.sh`
alongside it has the build command). Filesystem access is granted per-plugin via WASI preopens
(none by default); network isn't a raw socket — `network = true` grants a single `host_web_fetch`
import reusing the built-in `web_fetch`'s SSRF guard. Execution is metered with wasmtime fuel, so a
runaway plugin traps instead of hanging.

## MCP servers (optional)

MCP is a client/server protocol built around subprocesses and long-lived JSON-RPC sessions, which
doesn't fit the WASI sandbox above (no arbitrary process execution by design), so it's wired in on
the host side instead, behind an opt-in `mcp` Cargo feature so the `rmcp` dependency and its
subprocess-spawning code aren't part of the binary unless you ask for them:

```sh
cargo install --path crates/minder-cli --features mcp
```

With the feature enabled, minder discovers `.agent/mcp.toml`, launches each configured server as a
child process over the stdio transport, and registers every tool it advertises as an
`agent_core::Tool` named `mcp__<server>__<tool>`:

```toml
# .agent/mcp.toml
[[server]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]

[[server]]
name = "github"
command = "docker"
args = ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "..." }
```

Built without `--features mcp`, minder ignores `.agent/mcp.toml` entirely (the `mcp` tool /
`minder-tools-mcp` crate is compiled out, not just disabled at runtime). A configured server that
fails to start, initialize, or list its tools is a hard error at startup, same as a broken wasm
plugin or hook file. Remote tool calls are opaque to `on_tool_call`/`on_tool_result` hooks in the
same way built-in and wasm tool calls are — nothing MCP-specific bypasses the hook layer.

## Autonomous loop mode

```sh
minder loop TODO.md
minder loop TODO.md "ship the v2 pagination rewrite"   # optional overall-goal hint
```

`minder loop <file> ["<goal>"]` drives the same `AgentSession` turn after turn against a Markdown
checklist, with no user in the loop, and keeps watching the file for new work once it's clear:

1. Query `<file>` for GFM checklist lines that are still unchecked, entirely inside `mq-lang` —
   embedded the same way as the [hooks](#hooks) engine, just pointed at a file on disk instead of
   the conversation:

   ```
   read_file(path) | split(., "\n") | filter(., fn(line): is_regex_match(line, "^\\s*[-*+]\\s+\\[ \\]") end)
   ```

   This matches lines with a regex rather than walking markdown's list/checkbox AST, since
   mq-lang's only builtin that parses markdown into that AST (`collection`) works over an entire
   directory tree — overkill for re-checking one file every few seconds.
2. If nothing comes back, the file is done: minder logs that it's idle and polls `<file>` on an
   interval for new items.
3. Otherwise the remaining items are folded into a prompt ("pick the first unfinished item,
   implement it, then check it off in `<file>`") and handed to `run_turn`.
4. Repeat — the item just finished no longer shows up as unchecked, so the next prompt derives
   naturally from the file's current state, not a stale plan.

```markdown
<!-- TODO.md -->
## Backend
- [x] Set up database schema
- [ ] Add user authentication endpoint
- [ ] Write tests for auth endpoint
```

```sh
$ minder loop TODO.md
[loop 1/50] 2 item(s) remaining in TODO.md
→ write_file path=src/auth.rs
...
[loop 2/50] 1 item(s) remaining in TODO.md
→ edit_file path=tests/auth_test.rs
...
[loop] TODO.md has no unchecked items -- polling every 5s for new work (Ctrl-C to stop)
```

At that point the process just keeps running: add a new `- [ ] ...` line to `TODO.md` (by hand, or
have something else append to it) and the next poll picks it up automatically, no restart needed.
Stop it with Ctrl-C when you're done.

Safety limits keep a stuck agent from spinning forever, all overridable via env vars:

| Env var | Default | Guards against |
|---|---|---|
| `MINDER_LOOP_MAX_ITERATIONS` | 50 | Runaway spend — a lifetime cap on actual working turns (idle polling doesn't count against it) |
| `MINDER_LOOP_POLL_INTERVAL_SECS` | 5 | How often to re-check the file while idle |
| `MINDER_LOOP_QUERY` | `read_file(path) \| split(., "\n") \| filter(., fn(line): is_regex_match(...) end)` (see above) | Lets you point at a differently-structured file (a custom "done" marker, a different bullet convention, ...) |

If the unchecked count doesn't drop for two consecutive working iterations, the loop stops with an
error rather than burning turns on a task the model isn't making progress on.

## Project layout

| Crate | Responsibility |
|---|---|
| `minder-core` | Session/turn loop, tool-calling protocol, hook port trait |
| `minder-providers` | Anthropic/OpenAI/Gemini/Ollama client implementations |
| `minder-tools` | Built-in tools (file, shell, git, web) |
| `minder-tools-wasm` | WASI plugin loader and sandboxed host runtime |
| `minder-tools-mcp` | MCP client — spawns configured servers, exposes their tools (opt-in `mcp` feature on `minder-cli`) |
| `minder-hooks` | `mq`-based hook engine (`.agent/hooks/*.mq`) |
| `minder-cli` | The `minder` binary — wires providers/tools/hooks together |

## Development

Requires [`just`](https://github.com/casey/just). The same recipes run in CI ([`ci.yml`](.github/workflows/ci.yml)):

```sh
just test-all   # fmt --check, clippy -D warnings, doc tests, nextest
just fmt         # cargo fmt --all -- --check
just lint        # cargo clippy --all-targets --all-features --workspace -- -D clippy::all
just test        # cargo nextest run --workspace --all-features
just deps        # cargo machete (unused dependencies)
just audit       # cargo deny check (licenses/bans/sources/advisories)
```

Each provider's live round-trip test needs real credentials and is `#[ignore]`d by default:

```sh
ANTHROPIC_API_KEY=... cargo test -p minder-providers -- --ignored anthropic
OPENAI_API_KEY=... cargo test -p minder-providers -- --ignored openai
GEMINI_API_KEY=... cargo test -p minder-providers -- --ignored gemini
cargo test -p minder-providers -- --ignored ollama  # needs `ollama serve` running locally
```

### CI and releases

Every push/PR to `main` runs tests (Linux; the full Linux/macOS/Windows matrix is available via
`workflow_dispatch`), `rustfmt`, `clippy`, and `cargo-deny`. Separate scheduled/PR-triggered
workflows cover `cargo audit`, CodeQL, spell-checking (`typos`), unused-dependency detection
(`cargo-machete`), and Actions-workflow security linting (`zizmor`).

Pushing a `vX.Y.Z` tag builds the `minder` binary for Linux (gnu/musl, x86_64/aarch64), macOS
(aarch64), and Windows (x86_64) and attaches them — with checksums — to a draft GitHub Release
(review and publish it manually). See [`release.yml`](.github/workflows/release.yml).

All crates publish to crates.io under the `minder-*` prefix (`cargo install minder-cli`), and a
`vX.Y.Z` tag push also runs [`cargo-publish.yml`](.github/workflows/cargo-publish.yml) (needs a
`CARGO_REGISTRY_TOKEN` secret). Prefer a GitHub Release binary if you don't want to build from
source (see [Install](#install)).

## License

MIT
</content>
