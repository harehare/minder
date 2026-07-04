<div align="center">
  <img src="assets/logo.svg" style="width: 128px; height: 128px;"/>

<h1>minder</h1>

**A coding-agent harness in Rust.**

</div>

Multi-provider (Anthropic, OpenAI, Gemini, Ollama), with policy/observability hooks written in [`mq`](https://github.com/harehare/mq)'s embeddable query language instead of a general-purpose scripting language.

`mq-lang` has no builtin for file writes, network requests, or process execution, so hook scripts under `.agent/hooks/*.mq` can observe, block, or transform what the agent does without being able to do anything unsafe themselves — the host mediates all real side effects.

The agent loop itself is a standard ReAct-style tool-calling loop: the LLM's own response (does it emit tool calls or not) drives whether the loop continues, not the hooks. Hooks only answer narrow policy questions at five fixed interception points.

Every tool call, its result, and (for file edits) a colorized diff stream live to the terminal as the loop runs — see [Live execution display](#live-execution-display). `mq-lang` shows up a second time, embedded the same way as the hooks, as the harness's own [autonomous loop mode](#autonomous-loop-mode): `minder loop TODO.md` re-queries a Markdown checklist after every turn and keeps handing the model whatever's still unchecked, then keeps polling for more once the file is clear — no user in the loop, no external `mq` binary required.

> [!IMPORTANT]
> This project is under active development and has not been thoroughly tested end to end yet. Providers, tools, and hooks work individually in unit tests, but the full agent loop hasn't seen broad real-world verification — expect rough edges.

See `crates/agent-core`, `crates/agent-providers`, `crates/agent-tools`, `crates/agent-tools-wasm`, `crates/agent-hooks`, `crates/agent-cli`.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Providers](#providers)
- [Tools](#tools)
- [Skills](#skills)
- [Hooks](#hooks)
- [Tool plugins (WASM)](#tool-plugins-wasm)
- [Autonomous loop mode](#autonomous-loop-mode)
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

Behind the scenes each turn runs a standard tool-calling loop: the model reads your prompt,
decides whether it needs a tool (`read_file`, `bash`, `grep`, ...), the CLI executes it in your
current working directory, the result is fed back to the model, and this repeats until the model
replies without requesting another tool call. Everything the agent touches — files read/written,
commands run — is scoped to the directory `minder` was launched from.

### Live execution display

Every tool call is streamed to the terminal as it happens instead of only surfacing the final
answer once the whole loop has finished — useful both for watching what the agent is doing and
for debugging a stuck turn. The two output streams stay deliberately separate so piping `minder`'s
answer elsewhere stays clean:

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

### Running with gpt-oss

[gpt-oss](https://openai.com/index/introducing-gpt-oss/) (OpenAI's open-weight model family, `gpt-oss-20b`/`gpt-oss-120b`) runs through the existing `ollama` provider above — no minder code changes needed, since minder talks to Ollama's generic `/api/chat` endpoint and Ollama does the gpt-oss-specific translation.

1. Install Ollama (v0.11.4+; gpt-oss needs recent Ollama for correct tool-calling support): <https://ollama.com/download>, or:

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
3. Pull a gpt-oss model. `20b` needs ~16GB RAM/VRAM; `120b` needs ~65GB+ and is meant for
   multi-GPU/datacenter-class hardware — start with `20b` unless you know you have the headroom:

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

gpt-oss's reasoning effort (low/medium/high) isn't separately configurable through minder today — it runs at Ollama's default for the model. `minder loop` (see below) works the same way with `gpt-oss` as with any other provider, since it drives `AgentSession::run_turn` generically.

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

Additional tools can be supplied per-project as WASM plugins — see [Tool plugins (WASM)](#tool-plugins-wasm).

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

Each skill is a directory containing a `SKILL.md` with `---`-delimited frontmatter
(`name`, `description`) followed by the skill's instructions as the file body. minder
discovers every `.agent/skills/*/SKILL.md` at startup and, if any exist, registers a single
`skill` tool whose description lists each skill's name and short description — cheap enough to
keep in context on every turn. The model calls `skill` with a `name` to pull that skill's full
body into the conversation only when it's actually relevant, rather than paying for every
skill's full instructions on every turn.

Skill names must be unique across all discovered skills, and startup fails if a `SKILL.md` is
missing its frontmatter or the `name`/`description` fields. See `skills/commit-messages/SKILL.md`
in this repo for a runnable example (copy the `skills/` directory to `.agent/skills/` in a
project to try it).

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

`on_tool_call` can go a step further than allow/block: `{"action": "override", "value": {"content":
"...", "is_error": false, "metadata": null}}` supplies the tool's result directly. The real tool
never runs, but the outcome still flows through `on_tool_result` afterward like any other, so
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
extra to set up. Both fail **open**: a broken or undefined render function just falls back to the
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
in front of them, not the conversation — reach for the [`agent` module](#the-agent-module) inside
`on_context`/`before_compact` (which do see `messages`) if a display decision needs history, and
have that hook stash whatever's needed back onto the call/result some other way (e.g. blocking
before it ever reaches the display layer).

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

## Autonomous loop mode

```sh
minder loop TODO.md
minder loop TODO.md "ship the v2 pagination rewrite"   # optional overall-goal hint
```

`minder loop <file> ["<goal>"]` drives the same `AgentSession` turn after turn against a Markdown
checklist, with no user in the loop between iterations, and doesn't stop once the checklist is
clear — it keeps watching the file for new work indefinitely:

1. Query `<file>` for GFM checklist lines that are still unchecked. This runs entirely inside
   `mq-lang` — embedded directly (`DefaultEngine` + `file-io` feature), the same embedding the
   [hooks](#hooks) layer's `HookEngine` uses, just pointed at a file on disk instead of the
   conversation. minder's own Rust code never touches the filesystem for this: the query's
   `read_file(path)` does the reading, with `path` bound in as a variable the same way
   `HookEngine` binds `__hook_arg` — no `mq` subprocess, no external `mq` binary, no `std::fs` call
   on minder's side at all:

   ```
   read_file(path) | split(., "\n") | filter(., fn(line): is_regex_match(line, "^\\s*[-*+]\\s+\\[ \\]") end)
   ```

   (the doubled backslashes are mq's own string-escaping — a literal `\s` inside an mq string
   literal is written `\\s`, same as JSON)

   This works line-by-line with a regex rather than through markdown's list/checkbox AST
   (`is_list()`/`attr("checked")`): those selectors only see nodes the *host* already parsed and
   handed in as input, and mq-lang has no script-level builtin that turns a `read_file`d string
   into that same node form for a single file. (The one builtin that does, `collection`, parses
   every markdown file in an entire directory tree — overkill, and slow, for re-checking one file
   every few seconds.) The query's output is already the literal `- [ ] ...` lines, so there's no
   extra parsing on minder's side either way.
2. If nothing comes back, the file is done: `minder loop` logs that it's idle and starts polling
   `<file>` on an interval, waiting for someone (or something) to add a new checklist item.
3. Otherwise the remaining items are folded into a prompt ("pick the first unfinished item,
   implement it, then check it off in `<file>`") and handed to `run_turn`.
4. Repeat from step 1 — the item the model just finished no longer shows up as unchecked, so the
   next prompt is naturally derived from the file's current state, not from a stale plan.

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

If the unchecked count doesn't drop for more than two consecutive working iterations in a row, the
loop stops with an error rather than burning turns on a task the model isn't making progress on.

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
