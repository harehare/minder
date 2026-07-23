mod config;
mod file_reporter;
mod loop_mode;
mod markdown;
mod provider_select;
mod reporter;
mod session_store;

use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use minder_core::{AgentError, AgentSession, HookPort, LlmProvider, Message, Reporter, Tool, ToolContext};
use minder_hooks::HookEngine;
use minder_tools::{
    AgentTool, BashTool, Checkpoint, CheckpointedTool, DeleteFileTool, EditFileTool, GitCommitTool, GitDiffTool,
    GitLogTool, GitStatusTool, GlobTool, GrepTool, LsTool, ReadFileTool, SkillTool, TodoWriteTool, WebFetchTool,
    WebSearchTool, WorktreeAddTool, WorktreeListTool, WorktreeRemoveTool, WriteFileTool, builtin_subagents,
    discover_skills, discover_subagents, format_checklist,
};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};

use file_reporter::{CompositeReporter, FileReporter};
use provider_select::select_provider;
use reporter::{BOLD, CYAN, DIM, RESET, TerminalReporter, YELLOW};
use session_store::SessionRecord;

/// Tools offered to a `/plan` turn: investigation only, nothing that can
/// change the working directory or run arbitrary commands, so a plan can
/// never turn into an unreviewed action -- enforced by omitting the tools
/// entirely rather than relying on the model to behave.
const PLAN_READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "grep",
    "glob",
    "ls",
    "git_diff",
    "git_log",
    "git_status",
    "web_fetch",
    "web_search",
];

const PLAN_SYSTEM_PROMPT: &str = "\
You are minder in planning mode. Investigate the repository using only the read-only tools \
available to you -- file writes, shell, git mutations, and delegation aren't offered right now, \
on purpose. Reply with a concise, numbered implementation plan for the task below. Do not \
attempt to implement anything yourself; a human reviews the plan and decides whether to proceed.";

const SYSTEM_PROMPT: &str = "\
You are minder, a coding agent working in a git repository via tool calls. Investigate with \
`read_file`/`grep`/`glob`/`git_log`/`git_diff` before answering or editing -- read a file before \
editing it, prefer `edit_file` over `write_file` for existing files, and verify a change with \
`git_diff`/tests before calling it done.

Delegate self-contained work to `agent`, and check `skill` for a matching project skill before \
improvising. Only commit, push, or run other state-changing git/bash commands when asked. Use \
`delete_file` (not `bash rm`) to remove a file -- it's recoverable, `rm` isn't.

Use `todo_write` to plan and track progress on any task with several non-trivial steps -- keep at \
most one item `in_progress` at a time and mark items `completed` as soon as they're actually done. \
Skip it for a single quick action.

Keep replies short and grounded in what the tools actually returned.";

/// Multi-provider coding-agent CLI.
///
/// Run with no arguments to start an interactive session ('exit'/'quit' or
/// Ctrl-D to leave). Pass a task string to run it to completion (the session
/// is saved for --continue). With no <task>, --continue/--resume drop into
/// an interactive session too.
///
/// Piped stdin (e.g. `cat log.txt | minder "summarize the errors"`) is
/// folded into the task as extra input -- lets `minder` act as a general
/// Unix-pipeline filter, not just an interactive coding assistant. Only
/// applies to a one-shot task (plain, `--continue`, or `--resume` with a
/// task); has no effect on interactive `chat`/`loop`, since those read their
/// own input from stdin.
#[derive(Parser)]
#[command(name = "minder", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,

    /// Resume the most recent session in this project
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    r#continue: bool,

    /// Resume a specific session by id (or unambiguous prefix)
    #[arg(short = 'r', long = "resume", value_name = "ID")]
    resume: Option<String>,

    /// Output format for a one-shot task's final answer (plain, --continue,
    /// or --resume with a task); ignored by interactive `chat` and `loop`,
    /// which always print live text as they go
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    output: OutputFormat,

    /// Task to run to completion; with --continue/--resume, the task fed
    /// into the resumed session
    task: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    /// Assistant text streams live, matching interactive/default behavior
    Text,
    /// A single JSON object on stdout after the turn completes -- no live
    /// text, so a script/pipeline gets exactly one parseable value
    Json,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Same as running with no arguments
    Chat,
    /// Work through the file's unchecked checklist items, then keep polling
    /// it for new ones (mq-lang embedded, see README) -- runs until stopped
    /// (Ctrl-C) or a safety limit is hit
    Loop {
        /// Markdown checklist file to work through
        file: PathBuf,
        /// Optional task hint guiding the first pass over the checklist
        task_hint: Option<String>,
    },
    /// Prints a shell completion script to stdout
    Completion {
        /// Shell to generate the script for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

/// Everything `build_session` assembles, kept around (not just handed to
/// `AgentSession::new` and dropped) so the REPL's `/plan` command can build
/// a second, read-only-tooled `AgentSession` sharing the same provider/hooks
/// -- the same sharing `AgentTool` already does for subagents.
struct BuiltSession {
    session: AgentSession,
    provider: Arc<dyn LlmProvider>,
    /// Every tool the main session has, including `agent` -- `/plan` filters
    /// this down to `PLAN_READ_ONLY_TOOLS` itself.
    tools: Vec<Arc<dyn Tool>>,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
    reporter: Arc<dyn Reporter>,
    tool_ctx: ToolContext,
    /// Shared with `TerminalReporter`'s `on_thinking` gate -- toggling this
    /// (via `/thinking`) takes effect on the next turn without rebuilding
    /// the session. See `config::ProjectConfig::thinking_budget` for the
    /// initial value.
    show_thinking: Arc<AtomicBool>,
    /// Also present in `tools` as a type-erased `Arc<dyn Tool>` (that's what
    /// the model actually calls); kept here too, concretely typed, so
    /// `/todo` can read the current list without downcasting.
    todo: Arc<TodoWriteTool>,
    /// Shared with the `write_file`/`edit_file`/`delete_file` tools'
    /// `CheckpointedTool` wrapper (see above) -- `run_turn_interruptible`
    /// starts a fresh generation before each turn, and `/undo` reverts the
    /// most recent one.
    checkpoint: Arc<Checkpoint>,
}

/// Loads `.agent/config.toml`, exiting like every other `.agent/` loader
/// (hooks, skills, subagents) on a malformed file.
fn load_project_config(agent_dir: &Path) -> config::ProjectConfig {
    match config::load(agent_dir) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("failed to load .agent/config.toml: {e}");
            std::process::exit(1);
        }
    }
}

/// Builds the tool/hook/skill/plugin-wired session shared by both the
/// one-shot and `loop` entry points -- only the prompt(s) fed into
/// `run_turn` differ between the two modes. `output` decides whether
/// assistant text streams live to stdout (`Text`, the default and the only
/// sensible choice for `chat`/`loop`) or is held back so a caller can print
/// one JSON object instead (`Json`, one-shot only -- see `run_one_shot`).
async fn build_session(output: OutputFormat) -> BuiltSession {
    let working_dir = std::env::current_dir().expect("cwd");
    let agent_dir = working_dir.join(".agent");
    let cfg = load_project_config(&agent_dir);
    let provider = select_provider(&cfg);
    let tool_ctx = ToolContext {
        working_dir: working_dir.clone(),
        session_id: "cli".to_string(),
        cancel: tokio_util::sync::CancellationToken::new(),
    };

    let has_project_hooks = agent_dir.join("hooks").is_dir() || agent_dir.join("hooks.mq").is_file();
    let hooks = match HookEngine::load(&agent_dir) {
        Ok(engine) => {
            if has_project_hooks {
                eprintln!("loaded hooks from .agent/");
            }
            let boxed: Box<dyn HookPort> = Box::new(engine);
            Some(Arc::new(tokio::sync::Mutex::new(boxed)))
        }
        Err(e) => {
            eprintln!("failed to load hooks: {e}");
            std::process::exit(1);
        }
    };

    // Wraps write_file/edit_file/delete_file (not bash -- see `Checkpoint`'s
    // own docs on why arbitrary shell side effects aren't covered) so
    // `/undo` can revert whatever the most recently completed turn changed.
    let checkpoint = Arc::new(Checkpoint::new());
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool),
        Arc::new(CheckpointedTool::new(Arc::new(WriteFileTool), checkpoint.clone())),
        Arc::new(CheckpointedTool::new(Arc::new(EditFileTool), checkpoint.clone())),
        Arc::new(CheckpointedTool::new(Arc::new(DeleteFileTool), checkpoint.clone())),
        Arc::new(BashTool),
        Arc::new(GlobTool),
        Arc::new(GrepTool),
        Arc::new(LsTool),
        Arc::new(GitDiffTool),
        Arc::new(GitLogTool),
        Arc::new(GitStatusTool),
        Arc::new(GitCommitTool),
        Arc::new(WorktreeAddTool),
        Arc::new(WorktreeListTool),
        Arc::new(WorktreeRemoveTool),
        Arc::new(WebFetchTool::new()),
    ];
    // Omitted entirely (not registered with a doomed-to-fail key) when unset,
    // so the LLM never sees a tool in its list that it can't actually use.
    if let Ok(key) = std::env::var("TAVILY_API_KEY") {
        tools.push(Arc::new(WebSearchTool::new(key)));
    }

    match discover_skills(&working_dir.join(".agent")) {
        Ok(skills) => {
            if !skills.is_empty() {
                eprintln!("loaded {} skill(s) from .agent/skills/", skills.len());
                tools.push(Arc::new(SkillTool::new(skills)));
            }
        }
        Err(e) => {
            eprintln!("failed to load skills: {e}");
            std::process::exit(1);
        }
    }

    match minder_tools_wasm::load_plugins(&working_dir.join(".agent")).await {
        Ok(plugins) => {
            if !plugins.is_empty() {
                eprintln!("loaded {} wasm plugin tool(s) from .agent/tools/", plugins.len());
            }
            tools.extend(plugins.into_iter().map(Arc::from));
        }
        Err(e) => {
            eprintln!("failed to load wasm plugins: {e}");
            std::process::exit(1);
        }
    }

    #[cfg(feature = "mcp")]
    match minder_tools_mcp::load_mcp_tools(&working_dir.join(".agent")).await {
        Ok(mcp_tools) => {
            if !mcp_tools.is_empty() {
                eprintln!("loaded {} mcp tool(s) from .agent/mcp.toml", mcp_tools.len());
            }
            tools.extend(mcp_tools.into_iter().map(Arc::from));
        }
        Err(e) => {
            eprintln!("failed to load mcp servers: {e}");
            std::process::exit(1);
        }
    }

    let show_thinking = Arc::new(AtomicBool::new(cfg.thinking_budget.is_some()));
    let mut terminal_reporter_impl = TerminalReporter::new(hooks.clone(), show_thinking.clone());
    if output == OutputFormat::Json {
        terminal_reporter_impl = terminal_reporter_impl.silence_stdout();
    }
    let terminal_reporter: Arc<dyn Reporter> = Arc::new(terminal_reporter_impl);
    let reporter: Arc<dyn Reporter> = match std::env::var("MINDER_LOG_FILE") {
        Ok(path) => match FileReporter::new(Path::new(&path)) {
            Ok(file_reporter) => {
                eprintln!("logging to {path}");
                Arc::new(CompositeReporter::new(vec![terminal_reporter, Arc::new(file_reporter)]))
            }
            Err(e) => {
                eprintln!("failed to open log file {path}: {e}");
                terminal_reporter
            }
        },
        Err(_) => terminal_reporter,
    };

    // Builtins first; user-defined agents override by name.
    let mut subagents = builtin_subagents();
    match discover_subagents(&working_dir.join(".agent")) {
        Ok(discovered) => {
            if !discovered.is_empty() {
                eprintln!("loaded {} subagent(s) from .agent/agents/", discovered.len());
            }
            for subagent in discovered {
                match subagents.iter_mut().find(|s| s.name == subagent.name) {
                    Some(existing) => *existing = subagent,
                    None => subagents.push(subagent),
                }
            }
        }
        Err(e) => {
            eprintln!("failed to load subagents: {e}");
            std::process::exit(1);
        }
    }
    tools.push(Arc::new(AgentTool::new(
        subagents,
        provider.clone(),
        tools.clone(),
        hooks.clone(),
        reporter.clone(),
    )));

    // Added after `agent` is built (not before) so subagents -- which
    // inherit a snapshot of `tools` taken above -- don't get a shared todo
    // list of their own; todo tracking is for the top-level conversation.
    let todo = Arc::new(TodoWriteTool::new());
    tools.push(todo.clone() as Arc<dyn Tool>);

    let session = AgentSession::new(
        provider.clone(),
        tools.clone(),
        hooks.clone(),
        SYSTEM_PROMPT,
        tool_ctx.clone(),
    )
    .with_reporter(reporter.clone());

    BuiltSession {
        session,
        provider,
        tools,
        hooks,
        reporter,
        tool_ctx,
        show_thinking,
        todo,
        checkpoint,
    }
}

enum Command {
    OneShot {
        task: String,
        output: OutputFormat,
    },
    Continue {
        task: Option<String>,
        output: OutputFormat,
    },
    Resume {
        id: String,
        task: Option<String>,
        output: OutputFormat,
    },
    Chat,
    Loop {
        file: PathBuf,
        task_hint: Option<String>,
    },
    Completion {
        shell: clap_complete::Shell,
    },
}

impl From<Cli> for Command {
    fn from(cli: Cli) -> Self {
        match cli.command {
            Some(CliCommand::Chat) => return Command::Chat,
            Some(CliCommand::Loop { file, task_hint }) => return Command::Loop { file, task_hint },
            Some(CliCommand::Completion { shell }) => return Command::Completion { shell },
            None => {}
        }
        let output = cli.output;
        if let Some(id) = cli.resume {
            return Command::Resume {
                id,
                task: cli.task,
                output,
            };
        }
        if cli.r#continue {
            return Command::Continue { task: cli.task, output };
        }
        match cli.task {
            Some(task) => Command::OneShot { task, output },
            None => Command::Chat,
        }
    }
}

/// Prints a completion script for `shell` to stdout, e.g.:
/// `minder completion zsh >> ~/.zshrc` (or wherever your shell sources
/// completions from -- see its docs for the right file/directory).
fn print_completion(shell: clap_complete::Shell) {
    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
}

/// Bytes read from piped stdin beyond this point are dropped (with a note
/// appended) rather than folded whole into the task -- keeps one enormous
/// pipe (e.g. `cat huge.log | minder ...`) from single-handedly blowing the
/// context budget before the model even gets a turn.
const MAX_STDIN_CHARS: usize = 200_000;

/// Folds piped stdin (if any) into `task` as extra input, the same
/// separation `some_command | jq '...'` has between piped data and the
/// query itself. A no-op when stdin is a terminal (nothing piped) or empty
/// -- callers only invoke this for a one-shot task, never before the
/// interactive REPL takes over stdin for its own input.
fn with_piped_stdin(task: String) -> String {
    if std::io::stdin().is_terminal() {
        return task;
    }
    let mut piped = String::new();
    if std::io::stdin().read_to_string(&mut piped).is_err() {
        return task;
    }
    combine_task_with_piped_input(task, &piped)
}

/// The pure formatting half of `with_piped_stdin`, split out so it's
/// testable without a real (or faked) stdin handle.
fn combine_task_with_piped_input(task: String, piped: &str) -> String {
    let piped = piped.trim();
    if piped.is_empty() {
        return task;
    }

    let char_count = piped.chars().count();
    if char_count <= MAX_STDIN_CHARS {
        format!("{task}\n\n---\nPiped input:\n{piped}")
    } else {
        let truncated: String = piped.chars().take(MAX_STDIN_CHARS).collect();
        format!(
            "{task}\n\n---\nPiped input (truncated to the first {MAX_STDIN_CHARS} of {char_count} characters):\n{truncated}"
        )
    }
}

#[tokio::main]
async fn main() {
    match Command::from(Cli::parse()) {
        Command::OneShot { task, output } => run_one_shot(&with_piped_stdin(task), output).await,
        Command::Continue { task, output } => run_resume(None, task.map(with_piped_stdin), output).await,
        Command::Resume { id, task, output } => run_resume(Some(id), task.map(with_piped_stdin), output).await,
        Command::Chat => run_chat().await,
        Command::Loop { file, task_hint } => run_loop_mode(&file, task_hint.as_deref()).await,
        Command::Completion { shell } => print_completion(shell),
    }
}

fn working_dir() -> PathBuf {
    std::env::current_dir().expect("cwd")
}

/// How long an interrupted turn gets to wind down gracefully after a Ctrl-C
/// (e.g. `bash` killing its child process, see `bash.rs`) before it's force-
/// aborted regardless -- see `run_turn_interruptible`.
const INTERRUPT_GRACE_PERIOD: Duration = Duration::from_millis(1500);

/// Runs `session.run_turn(input)` the same as calling it directly, except a
/// Ctrl-C during the turn interrupts it instead of killing the whole
/// process. First Ctrl-C cancels the turn's tools (`AgentSession::reset_cancel_token`)
/// and waits up to `INTERRUPT_GRACE_PERIOD` for the turn to notice and wind
/// down on its own -- e.g. `bash` killing its child process rather than
/// leaving it orphaned. A second Ctrl-C during that window (or the grace
/// period simply elapsing) force-aborts immediately regardless.
///
/// Either way, an interrupted turn's partial messages are discarded (see
/// `AgentSession::discard_interrupted_turn`) so the next turn starts from a
/// clean, correctly-alternating transcript instead of one a provider might
/// reject. The interrupted task itself isn't lost, though: the REPL just
/// loops back to the prompt with full history intact, ready for a follow-up
/// instruction -- the same "stop and redirect" flow other coding agents
/// offer via Esc, just bound to Ctrl-C here since minder's REPL already
/// treats Ctrl-C-during-input as "cancel this line" (see `run_repl`).
async fn run_turn_interruptible(session: &mut AgentSession, input: &str) -> Result<Message, AgentError> {
    let pre_turn_len = session.messages().len();
    let cancel = session.reset_cancel_token();

    // Scoped so `turn` (and the mutable borrow of `session` it holds) ends
    // with this block, before `discard_interrupted_turn` below needs its
    // own borrow -- `tokio::pin!`'s storage otherwise outlives the binding
    // itself, into the rest of the function.
    let result = {
        let turn = session.run_turn(input);
        tokio::pin!(turn);

        tokio::select! {
            result = &mut turn => return result,
            _ = tokio::signal::ctrl_c() => {}
        }

        cancel.cancel();
        tokio::select! {
            result = &mut turn => result,
            _ = tokio::signal::ctrl_c() => Err(AgentError::Interrupted),
            _ = tokio::time::sleep(INTERRUPT_GRACE_PERIOD) => Err(AgentError::Interrupted),
        }
    };

    if matches!(result, Err(AgentError::Interrupted)) {
        session.discard_interrupted_turn(pre_turn_len);
    }
    result
}

/// Prints a turn's error the way it deserves: an interrupt is an expected,
/// user-initiated outcome (plain notice, no "error:" framing), everything
/// else is a real failure.
fn print_turn_error(err: &AgentError) {
    if matches!(err, AgentError::Interrupted) {
        println!("Interrupted.");
    } else {
        eprintln!("error: {err}");
    }
}

/// Refreshes a session record from the live session's transcript and saves
/// it, so the process can be resumed later via `--continue`/`--resume`.
/// Best-effort: a save failure is a warning, never fatal to the turn itself.
fn persist(dir: &Path, record: &mut SessionRecord, session: &AgentSession) {
    record.system_prompt = session.system_prompt().to_string();
    record.messages = session.messages().to_vec();
    if let Err(e) = session_store::save(dir, record) {
        eprintln!("warning: failed to save session: {e}");
    }
}

/// Prints the one JSON object `--output json` promises: the turn's answer
/// (or `error`, mutually exclusive) plus which provider/model produced it.
/// Called instead of letting assistant text stream to stdout live -- see
/// `TerminalReporter::silence_stdout`.
fn print_json_result(session: &AgentSession, result: &Result<Message, AgentError>) {
    let payload = json_result_payload(session.provider_id(), session.model(), result);
    println!("{payload}");
}

/// The pure half of `print_json_result`, split out so the payload shape is
/// testable without spinning up a real `AgentSession`.
fn json_result_payload(provider_id: &str, model: &str, result: &Result<Message, AgentError>) -> serde_json::Value {
    let (answer, error) = match result {
        Ok(message) => (Some(message.text()), None),
        Err(e) => (None, Some(e.to_string())),
    };
    serde_json::json!({
        "provider": provider_id,
        "model": model,
        "answer": answer,
        "error": error,
    })
}

async fn run_one_shot(task: &str, output: OutputFormat) {
    let dir = working_dir();
    let mut built = build_session(output).await;
    let mut record = SessionRecord::new();

    let result = built.session.run_turn(task).await;
    persist(&dir, &mut record, &built.session);

    if output == OutputFormat::Json {
        print_json_result(&built.session, &result);
    }

    if let Err(e) = result {
        if output == OutputFormat::Text {
            eprintln!("error: {e}");
        }
        std::process::exit(1);
    }
}

async fn run_chat() {
    let dir = working_dir();
    let mut built = build_session(OutputFormat::Text).await;
    let mut record = SessionRecord::new();
    run_repl(&mut built, &dir, &mut record).await;
}

/// Resumes a saved session (latest if `id` is `None`) and either runs one
/// more task, or drops into an interactive session when no task is given.
/// `output` only takes effect in the "one more task" branch -- the
/// interactive branch always streams live text, so it's forced back to
/// `Text` rather than risk silencing the REPL if `--output json` was passed
/// alongside a bare `--continue`/`--resume`.
async fn run_resume(id: Option<String>, task: Option<String>, output: OutputFormat) {
    let dir = working_dir();
    let loaded = match &id {
        Some(id) => session_store::load_by_id(&dir, id),
        None => session_store::load_latest(&dir),
    };
    let mut record = match loaded {
        Ok(Some(record)) => record,
        Ok(None) => {
            eprintln!("no session found to resume in this project");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("error: failed to load session: {e}");
            std::process::exit(1);
        }
    };

    let effective_output = if task.is_some() { output } else { OutputFormat::Text };
    let mut built = build_session(effective_output).await;
    built
        .session
        .restore(record.system_prompt.clone(), record.messages.clone());

    match task {
        Some(task) => {
            let result = built.session.run_turn(&task).await;
            persist(&dir, &mut record, &built.session);

            if effective_output == OutputFormat::Json {
                print_json_result(&built.session, &result);
            }

            if let Err(e) = result {
                if effective_output == OutputFormat::Text {
                    eprintln!("error: {e}");
                }
                std::process::exit(1);
            }
        }
        None => run_repl(&mut built, &dir, &mut record).await,
    }
}

/// Width used for the rule line when the terminal's own width can't be
/// determined (not a tty, e.g. output piped to a file).
const FALLBACK_RULE_WIDTH: usize = 64;

/// Floor on the rule width so a slim terminal (or a bogus 0-width report)
/// doesn't collapse it into something unreadable.
const MIN_RULE_WIDTH: usize = 20;

/// Width of the rule line drawn above/below the input, matched to the
/// terminal's current column count so it spans (almost) the full width.
/// Kept one column short of the terminal's actual width -- filling every
/// column triggers auto-wrap on many terminals, which pushed the closing
/// rule off screen instead of onto its own line. Queried fresh on every
/// draw rather than cached, so a mid-session terminal resize is picked up
/// on the next turn.
fn rule_width() -> usize {
    terminal_size::terminal_size()
        .map(|(terminal_size::Width(w), _)| (w as usize).saturating_sub(1).max(MIN_RULE_WIDTH))
        .unwrap_or(FALLBACK_RULE_WIDTH)
}

/// Both the prompt and its surrounding rules print to stdout (rustyline
/// renders the prompt there, not stderr), so they need the same color
/// decision or the rule and the `❯` would disagree about `NO_COLOR`.
fn color_enabled(stream_is_tty: bool) -> bool {
    stream_is_tty && std::env::var_os("NO_COLOR").is_none()
}

/// A plain dashed rule marking the input area's top/bottom edge. Uses
/// ASCII `-` rather than box-drawing characters: those render double-width
/// under some CJK-locale terminal/font combinations, which threw off the
/// column count and wrapped the line early, leaving only a stray corner
/// glyph visible.
fn rule_line(color: bool) -> String {
    let rule = "-".repeat(rule_width());
    if color { format!("{DIM}{rule}{RESET}") } else { rule }
}

/// Builds the REPL's input prompt: `❯ ` marks a turn boundary without a
/// full box around it -- see `run_repl` for the rule lines drawn above and
/// below it.
fn repl_prompt(color: bool) -> String {
    if color {
        format!("{BOLD}{CYAN}❯{RESET} ")
    } else {
        "> ".to_string()
    }
}

/// The line above each turn's input area: provider/model and working
/// directory, so that context stays visible without repeating the full
/// startup banner every turn.
fn status_line(session: &AgentSession, dir: &Path, color: bool) -> String {
    let text = format!("{} ({}) · {}", session.provider_id(), session.model(), dir.display());
    if color { format!("{DIM}{text}{RESET}") } else { text }
}

/// The line below each turn's input area: the same keyboard shortcuts every
/// time, so they're always one glance away instead of scrolled off after
/// the first turn.
fn hint_line(color: bool) -> String {
    let text = "Ctrl-C cancel input/turn · Ctrl-D or 'exit'/'quit' to leave · /help for commands";
    if color {
        format!("{DIM}{text}{RESET}")
    } else {
        text.to_string()
    }
}

const SLASH_HELP: &str = "\
Available commands:
  /help          Show this list
  /model         Show the active provider and model
  /clear         Clear the conversation history (keeps the session file, starts fresh)
  /plan <task>   Investigate read-only and propose a plan for <task> before touching anything
  /thinking      Toggle showing the model's extended-thinking output (Anthropic only)
  /todo          Show the model's current todo list
  /undo          Revert the file changes from the most recently completed turn
  exit, quit     Leave (Ctrl-D also works)";

/// Names accepted by `handle_slash_command`, kept in sync with `SLASH_HELP`
/// above; also drives completion/hinting in `SlashCommandHelper`.
const SLASH_COMMANDS: &[&str] = &["help", "model", "clear", "plan", "thinking", "todo", "undo"];

/// Commands still matching what's typed after `/`, or `None` if `line`/`pos`
/// isn't in "typing a command name" position at all (no leading `/`, cursor
/// not at the end, or a space already reached -- i.e. past the command name
/// into its arguments).
fn matching_slash_commands(line: &str, pos: usize) -> Option<Vec<&'static str>> {
    if pos != line.len() {
        return None;
    }
    let typed = line.strip_prefix('/')?;
    if typed.contains(' ') {
        return None;
    }
    Some(
        SLASH_COMMANDS
            .iter()
            .copied()
            .filter(|cmd| cmd.starts_with(typed))
            .collect(),
    )
}

/// A hint shown after the cursor while typing a `/`-command: either the rest
/// of the single remaining command name (so `Right`/`End` accepts it like a
/// shell autosuggestion), or, while still ambiguous, a plain listing of every
/// command still in the running -- so typing bare `/` immediately shows what
/// can be selected instead of waiting for a Tab press.
struct SlashCommandHint {
    display: String,
    completion_len: usize,
}

impl rustyline::hint::Hint for SlashCommandHint {
    fn display(&self) -> &str {
        &self.display
    }

    fn completion(&self) -> Option<&str> {
        (self.completion_len > 0).then(|| &self.display[..self.completion_len])
    }
}

/// Line-editor helper wired into `ReplEditor`: `Completer` handles Tab, and
/// `Hinter` shows a live suggestion after every keystroke without needing
/// Tab at all. `color` mirrors the REPL's own `NO_COLOR`/tty decision so the
/// hint text is dimmed consistently with everything else `run_repl` prints.
struct SlashCommandHelper {
    color: bool,
}

impl Completer for SlashCommandHelper {
    type Candidate = Pair;

    fn complete(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> rustyline::Result<(usize, Vec<Pair>)> {
        let Some(matches) = matching_slash_commands(line, pos) else {
            return Ok((pos, Vec::new()));
        };
        let candidates = matches
            .into_iter()
            .map(|cmd| Pair {
                display: format!("/{cmd}"),
                replacement: format!("/{cmd} "),
            })
            .collect();
        Ok((0, candidates))
    }
}

impl Hinter for SlashCommandHelper {
    type Hint = SlashCommandHint;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<SlashCommandHint> {
        let matches = matching_slash_commands(line, pos)?;
        let typed_len = pos - 1; // chars typed after the leading '/'
        match matches.as_slice() {
            [] => None,
            [only] => {
                let suffix = &only[typed_len..];
                (!suffix.is_empty()).then(|| SlashCommandHint {
                    display: suffix.to_string(),
                    completion_len: suffix.len(),
                })
            }
            many => {
                let list = many.iter().map(|cmd| format!("/{cmd}")).collect::<Vec<_>>().join("  ");
                Some(SlashCommandHint {
                    display: format!("  {list}"),
                    completion_len: 0,
                })
            }
        }
    }
}

impl Highlighter for SlashCommandHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> std::borrow::Cow<'h, str> {
        if self.color {
            std::borrow::Cow::Owned(format!("{DIM}{hint}{RESET}"))
        } else {
            std::borrow::Cow::Borrowed(hint)
        }
    }
}
impl Validator for SlashCommandHelper {}
impl Helper for SlashCommandHelper {}

/// The concrete editor type used by the REPL: `rustyline`'s `DefaultEditor`
/// with `SlashCommandHelper` swapped in for `()` so Tab-completion and live
/// hints work.
type ReplEditor = Editor<SlashCommandHelper, FileHistory>;

/// Runs a `/`-prefixed REPL command. Returns `false` only for a command that
/// should end the REPL (none do today, but keeping the return type leaves
/// room for e.g. a future `/exit` alias without changing every call site).
async fn handle_slash_command(
    input: &str,
    built: &mut BuiltSession,
    dir: &Path,
    record: &mut SessionRecord,
    editor: &mut ReplEditor,
) {
    let (cmd, rest) = input.split_once(' ').unwrap_or((input, ""));
    let rest = rest.trim();

    match cmd {
        "help" => println!("{SLASH_HELP}"),
        "model" => println!("{} · {}", built.session.provider_id(), built.session.model()),
        "clear" => {
            let system_prompt = built.session.system_prompt().to_string();
            built.session.restore(system_prompt, Vec::new());
            persist(dir, record, &built.session);
            println!("Conversation cleared.");
        }
        "plan" if !rest.is_empty() => run_plan_command(rest, built, dir, record, editor).await,
        "plan" => println!("Usage: /plan <task>"),
        "thinking" => {
            let shown = !built.show_thinking.fetch_xor(true, Ordering::Relaxed);
            println!("Extended-thinking display is now {}.", if shown { "on" } else { "off" });
        }
        "todo" => println!("{}", format_checklist(&built.todo.items())),
        "undo" => run_undo_command(built, dir).await,
        other => println!("Unknown command '/{other}'. Type /help for a list."),
    }
}

/// `/plan <task>`: runs `task` through a throwaway `AgentSession` that
/// shares the real session's provider/hooks (same sharing `AgentTool` uses
/// for subagents) but only has read-only tools, so it can investigate and
/// propose without being able to change anything. The plan is shown to the
/// user, who then decides whether to actually run `task` on the real,
/// fully-tooled session.
async fn run_plan_command(
    task: &str,
    built: &mut BuiltSession,
    dir: &Path,
    record: &mut SessionRecord,
    editor: &mut ReplEditor,
) {
    println!("Planning (read-only investigation, no changes will be made)...");
    println!();

    let read_only_tools: Vec<Arc<dyn Tool>> = built
        .tools
        .iter()
        .filter(|t| PLAN_READ_ONLY_TOOLS.contains(&t.name()))
        .cloned()
        .collect();
    let mut plan_session = AgentSession::new(
        built.provider.clone(),
        read_only_tools,
        built.hooks.clone(),
        PLAN_SYSTEM_PROMPT,
        built.tool_ctx.clone(),
    )
    .with_reporter(built.reporter.clone());

    let plan_text = match run_turn_interruptible(&mut plan_session, task).await {
        Ok(message) => message.text(),
        Err(AgentError::Interrupted) => {
            println!("Planning interrupted.");
            return;
        }
        Err(e) => {
            eprintln!("error: planning turn failed: {e}");
            return;
        }
    };

    println!("{plan_text}");
    println!();

    let confirmed = match editor.readline("Proceed with this plan? [y/N] ") {
        Ok(answer) => matches!(answer.trim().to_lowercase().as_str(), "y" | "yes"),
        Err(_) => false,
    };

    if !confirmed {
        println!("Plan discarded.");
        return;
    }

    let follow_up = format!("Implement the following plan:\n\n{plan_text}\n\nOriginal task: {task}");
    built.checkpoint.start_turn();
    if let Err(e) = run_turn_interruptible(&mut built.session, &follow_up).await {
        print_turn_error(&e);
    }
    persist(dir, record, &built.session);
}

/// `/undo`: reverts every file the most recently completed turn's
/// `write_file`/`edit_file` calls touched, back to how it looked just
/// before that turn started (see `Checkpoint`). Only ever reaches back one
/// turn -- running it again right after has nothing left to revert, and a
/// `bash` command's own file changes were never tracked in the first place.
async fn run_undo_command(built: &BuiltSession, dir: &Path) {
    let restored = built.checkpoint.undo().await;
    if restored.is_empty() {
        println!("Nothing to undo.");
        return;
    }
    println!("Reverted {} file(s):", restored.len());
    for path in restored {
        println!("  {}", path.strip_prefix(dir).unwrap_or(&path).display());
    }
}

/// Short "it's alive" banner shown once when a REPL starts, so launching
/// `minder` feels intentional rather than dropping silently into a bare
/// prompt. The mark is a plain glyph, not a figlet wordmark -- it needs to
/// render identically whether or not the terminal's font covers box-drawing
/// or emoji beyond basic Unicode. Colored only when stderr is a tty and
/// `NO_COLOR` isn't set, matching `TerminalReporter`'s own rule.
fn print_banner(session: &AgentSession, record: &SessionRecord) {
    let color = color_enabled(std::io::stderr().is_terminal());
    let paint = |code: &str, text: &str| {
        if color {
            format!("{code}{text}{RESET}")
        } else {
            text.to_string()
        }
    };

    let version = env!("CARGO_PKG_VERSION");
    let status = if record.messages.is_empty() {
        "new session".to_string()
    } else {
        format!("resumed, {} prior message(s)", record.messages.len())
    };

    // Three ◆ arranged as a diamond outline (top/left-right/bottom points)
    // rather than one lone glyph -- a single character can't get visually
    // bigger in a terminal (font size is fixed), so this fakes scale by
    // spreading it across a few rows instead, the same trick figlet-style
    // banners use.
    let accent = format!("{YELLOW}{BOLD}");
    eprintln!();
    eprintln!("  {}   {}", paint(&accent, " ◆ "), paint(BOLD, &format!("v{version}")));
    eprintln!(
        "  {}   {}",
        paint(&accent, "◆ ◆"),
        paint(DIM, &format!("{} · {status}", session.provider_id()))
    );
    eprintln!(
        "  {}   {}",
        paint(&accent, " ◆ "),
        paint(DIM, &working_dir().display().to_string())
    );
    eprintln!();
}

/// Reads tasks from a line editor in a loop, feeding each into the same
/// session so context carries over between them; saves after every turn so
/// Ctrl-C or a crash mid-conversation loses at most the in-flight turn.
///
/// Uses `rustyline` rather than a raw `stdin().read_line()` so editing (and
/// especially IME-driven Japanese input) redraws correctly: rustyline tracks
/// display width itself instead of assuming the terminal's canonical line
/// discipline gets multi-byte/wide characters right, which is what caused
/// visible cursor drift when composing non-ASCII input. History persists
/// per-project across sessions (see `session_store::history_path`) so an
/// up-arrow recalls prior turns too.
///
/// Exits on EOF (Ctrl-D), or an "exit"/"quit" line. Ctrl-C cancels the
/// in-progress input line and re-prompts rather than killing the process.
/// A line starting with `/` is a REPL command (`/help`, `/model`, `/clear`,
/// `/plan <task>`) handled by `handle_slash_command` instead of being sent
/// to the model.
async fn run_repl(built: &mut BuiltSession, dir: &Path, record: &mut SessionRecord) {
    print_banner(&built.session, record);

    // Decided once (a terminal doesn't change tty-ness mid-session) and
    // shared by every rule/status/hint line drawn below -- and the slash
    // command hinter -- so they never disagree with the prompt itself about
    // `NO_COLOR`.
    let color = color_enabled(std::io::stdout().is_terminal());
    let prompt = repl_prompt(color);

    let history = session_store::history_path(dir).ok();
    let mut editor: ReplEditor = Editor::new().expect("failed to initialize line editor");
    editor.set_helper(Some(SlashCommandHelper { color }));
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }

    loop {
        println!("{}", status_line(&built.session, dir, color));
        println!("{}", rule_line(color));

        let line = match editor.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                println!("{}", rule_line(color));
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("error: {e}");
                break;
            }
        };
        println!("{}", rule_line(color));
        println!("{}", hint_line(color));
        println!();

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }

        let _ = editor.add_history_entry(line);
        if let Some(path) = &history {
            let _ = editor.save_history(path);
        }

        if let Some(command) = line.strip_prefix('/') {
            handle_slash_command(command, built, dir, record, &mut editor).await;
            println!();
            continue;
        }

        built.checkpoint.start_turn();
        if let Err(e) = run_turn_interruptible(&mut built.session, line).await {
            print_turn_error(&e);
        }
        persist(dir, record, &built.session);
        println!();
    }
}

/// `minder loop <file>` is keyed by `file`'s canonical path (not a random
/// id) so re-running the same command after a crash, Ctrl-C, or a container
/// restart resumes the same conversation automatically -- see
/// `session_store::key_for_path`.
async fn run_loop_mode(file: &Path, task_hint: Option<&str>) {
    let dir = working_dir();
    let mut built = build_session(OutputFormat::Text).await;

    let key = session_store::key_for_path(file);
    let mut record = match session_store::load_by_id(&dir, &key) {
        Ok(Some(record)) => {
            built
                .session
                .restore(record.system_prompt.clone(), record.messages.clone());
            eprintln!(
                "resuming loop session for {} ({} prior message(s))",
                file.display(),
                record.messages.len()
            );
            record
        }
        Ok(None) => SessionRecord::with_id(key),
        Err(e) => {
            eprintln!("warning: failed to load prior loop session: {e}");
            SessionRecord::with_id(key)
        }
    };

    let result = loop_mode::run(
        &mut built.session,
        file,
        task_hint,
        loop_mode::LoopOptions::default(),
        |session| {
            persist(&dir, &mut record, session);
        },
    )
    .await;
    persist(&dir, &mut record, &built.session);

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::{ProviderError, ProviderResponse, Role, StopReason, Tool, ToolContext, Usage};

    /// Always answers with fixed `text` -- enough to prove
    /// `run_turn_interruptible` behaves like a plain `run_turn` when nothing
    /// ever interrupts it (the interrupt path itself needs a real SIGINT, so
    /// isn't exercised by this in-process test suite).
    struct FixedTextProvider(&'static str);

    #[async_trait::async_trait]
    impl LlmProvider for FixedTextProvider {
        fn id(&self) -> &'static str {
            "fixed"
        }
        fn model(&self) -> &str {
            "fixed-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[minder_core::ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(ProviderResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![minder_core::ContentBlock::Text(self.0.to_string())],
                    metadata: serde_json::Value::Null,
                },
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    }

    fn test_tool_ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn run_turn_interruptible_matches_plain_run_turn_when_uninterrupted() {
        let mut session = AgentSession::new(
            Arc::new(FixedTextProvider("all done")),
            Vec::<Arc<dyn Tool>>::new(),
            None,
            "test agent",
            test_tool_ctx(),
        );

        let result = run_turn_interruptible(&mut session, "do something").await.unwrap();

        assert_eq!(result.text(), "all done");
        // user input + assistant reply, nothing rolled back since nothing interrupted it
        assert_eq!(session.messages().len(), 2);
    }

    /// Real, end-to-end verification of the thing `run_turn_interruptible`
    /// exists for: a live SIGINT to this test's own process should cancel a
    /// running `bash` command (killing its child rather than waiting out
    /// its full duration) and let the turn wind down gracefully within
    /// `INTERRUPT_GRACE_PERIOD`. Sends a real signal to the current
    /// process, so it's opt-in (`--ignored`) rather than part of the
    /// default suite -- same convention as the providers' `live_round_trip`
    /// tests that need real external state.
    #[tokio::test]
    #[ignore = "sends a real SIGINT to this test's own process -- run explicitly with --ignored"]
    async fn ctrl_c_interrupts_a_running_bash_command_instead_of_waiting_out_its_full_duration() {
        struct ScriptedProvider(std::sync::Mutex<std::collections::VecDeque<ProviderResponse>>);

        #[async_trait::async_trait]
        impl LlmProvider for ScriptedProvider {
            fn id(&self) -> &'static str {
                "scripted"
            }
            fn model(&self) -> &str {
                "scripted-model"
            }
            async fn complete(
                &self,
                _messages: &[Message],
                _tools: &[minder_core::ToolSpec],
                _system_prompt: Option<&str>,
            ) -> Result<ProviderResponse, ProviderError> {
                Ok(self.0.lock().unwrap().pop_front().expect("script exhausted"))
            }
        }

        let tool_use = ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![minder_core::ContentBlock::ToolUse(minder_core::ToolCall {
                    id: "call_1".to_string(),
                    name: "bash".to_string(),
                    arguments: serde_json::json!({"command": "sleep 30", "timeout_secs": 60}),
                })],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        };
        let final_text = ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![minder_core::ContentBlock::Text("acknowledged".to_string())],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        };

        let provider = Arc::new(ScriptedProvider(std::sync::Mutex::new(
            [tool_use, final_text].into_iter().collect(),
        )));
        let mut session = AgentSession::new(
            provider,
            vec![Arc::new(minder_tools::BashTool) as Arc<dyn Tool>],
            None,
            "test agent",
            test_tool_ctx(),
        );

        let pid = std::process::id();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let _ = tokio::process::Command::new("kill")
                .arg("-INT")
                .arg(pid.to_string())
                .status()
                .await;
        });

        let start = std::time::Instant::now();
        let result = run_turn_interruptible(&mut session, "run something slow").await;
        let elapsed = start.elapsed();

        assert!(
            result.is_ok(),
            "expected a graceful completion after the tool call was cancelled, got {result:?}"
        );
        assert_eq!(result.unwrap().text(), "acknowledged");
        assert!(
            elapsed < Duration::from_secs(10),
            "took {elapsed:?} -- the cancelled `sleep 30` should have been killed almost instantly"
        );
    }

    #[test]
    fn empty_piped_input_leaves_the_task_untouched() {
        assert_eq!(
            combine_task_with_piped_input("do the thing".to_string(), ""),
            "do the thing"
        );
        assert_eq!(
            combine_task_with_piped_input("do the thing".to_string(), "   \n  "),
            "do the thing"
        );
    }

    #[test]
    fn piped_input_is_appended_after_the_task() {
        let combined = combine_task_with_piped_input("summarize this".to_string(), "line one\nline two");
        assert!(combined.starts_with("summarize this"));
        assert!(combined.contains("line one\nline two"));
    }

    #[test]
    fn oversized_piped_input_is_truncated_with_a_note() {
        let huge = "x".repeat(MAX_STDIN_CHARS + 500);
        let combined = combine_task_with_piped_input("task".to_string(), &huge);
        assert!(combined.contains("truncated to the first"));
        assert!(!combined.contains(&"x".repeat(MAX_STDIN_CHARS + 1)));
    }

    #[test]
    fn json_payload_carries_the_answer_on_success() {
        let ok: Result<Message, AgentError> = Ok(Message {
            role: minder_core::Role::Assistant,
            content: vec![minder_core::ContentBlock::Text("42".to_string())],
            metadata: serde_json::Value::Null,
        });
        let payload = json_result_payload("anthropic", "claude-sonnet-5", &ok);
        assert_eq!(payload["provider"], "anthropic");
        assert_eq!(payload["model"], "claude-sonnet-5");
        assert_eq!(payload["answer"], "42");
        assert!(payload["error"].is_null());
    }

    #[test]
    fn json_payload_carries_the_error_on_failure() {
        let err: Result<Message, AgentError> = Err(AgentError::HookBlocked("blocked by policy".to_string()));
        let payload = json_result_payload("anthropic", "claude-sonnet-5", &err);
        assert!(payload["answer"].is_null());
        assert!(payload["error"].as_str().unwrap().contains("blocked by policy"));
    }

    fn complete_at_cursor(line: &str) -> (usize, Vec<Pair>) {
        let history = rustyline::history::MemHistory::new();
        let ctx = Context::new(&history);
        SlashCommandHelper { color: false }
            .complete(line, line.len(), &ctx)
            .unwrap()
    }

    fn hint_at_cursor(line: &str) -> Option<String> {
        let history = rustyline::history::MemHistory::new();
        let ctx = Context::new(&history);
        SlashCommandHelper { color: false }
            .hint(line, line.len(), &ctx)
            .map(|h| rustyline::hint::Hint::display(&h).to_string())
    }

    #[test]
    fn slash_completion_lists_all_commands_for_bare_slash() {
        let (start, candidates) = complete_at_cursor("/");
        assert_eq!(start, 0);
        let names: Vec<&str> = candidates.iter().map(|p| p.display.as_str()).collect();
        assert_eq!(names.len(), SLASH_COMMANDS.len());
        assert!(names.contains(&"/plan"));
    }

    #[test]
    fn slash_completion_narrows_by_prefix() {
        let (_, candidates) = complete_at_cursor("/th");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].display, "/thinking");
        assert_eq!(candidates[0].replacement, "/thinking ");
    }

    #[test]
    fn slash_completion_stops_after_the_command_name() {
        let (_, candidates) = complete_at_cursor("/plan fix the ");
        assert!(candidates.is_empty());
    }

    #[test]
    fn slash_completion_empty_for_plain_text() {
        let (_, candidates) = complete_at_cursor("hello");
        assert!(candidates.is_empty());
    }

    #[test]
    fn slash_hint_lists_every_command_for_bare_slash() {
        let hint = hint_at_cursor("/").expect("bare '/' should hint the full command list");
        for cmd in SLASH_COMMANDS {
            assert!(hint.contains(&format!("/{cmd}")), "hint {hint:?} missing /{cmd}");
        }
    }

    #[test]
    fn slash_hint_completes_the_rest_of_a_single_match() {
        let hint = hint_at_cursor("/th").expect("'/th' uniquely matches '/thinking'");
        assert_eq!(hint, "inking");
    }

    #[test]
    fn slash_hint_absent_once_a_command_is_fully_typed() {
        assert_eq!(hint_at_cursor("/thinking"), None);
    }

    #[test]
    fn slash_hint_absent_past_the_command_name() {
        assert_eq!(hint_at_cursor("/plan fix the "), None);
    }

    #[test]
    fn slash_hint_absent_for_plain_text() {
        assert_eq!(hint_at_cursor("hello"), None);
    }
}
