mod config;
mod file_reporter;
mod input_watcher;
mod loop_mode;
mod markdown;
mod mentions;
mod provider_select;
mod reporter;
mod schedule_mode;
mod session_store;
mod status_reporter;
mod tui;

use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use minder_core::{AgentError, AgentSession, HookPort, LlmProvider, Message, Reporter, Tool, ToolContext};
use minder_hooks::HookEngine;
use minder_tools::{
    AgentOutputTool, AgentRegistry, AgentStopTool, AgentTool, BashTool, Checkpoint, CheckpointedTool, DeleteFileTool,
    EditFileTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool, GlobTool, GrepTool, ListAgentsTool, LsTool,
    ProviderFactory, ReadFileTool, SkillTool, TodoWriteTool, WebFetchTool, WebSearchTool, WriteFileTool,
    builtin_subagents, discover_all_skills, discover_plugins, discover_subagents, format_checklist,
};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};

use file_reporter::{CompositeReporter, FileReporter, LogFormat};
use input_watcher::InputWatcher;
use provider_select::select_provider;
use reporter::{BOLD, CYAN, DIM, RESET, TerminalReporter, YELLOW};
use session_store::SessionRecord;
use status_reporter::StatusReporter;

const SYSTEM_PROMPT: &str = "\
You are minder, a coding agent working in a git repository via tool calls. Investigate with \
`read_file`/`grep`/`glob`/`git_log`/`git_diff` before answering or editing -- read a file before \
editing it, prefer `edit_file` over `write_file` for existing files, and verify a change with \
`git_diff`/tests before calling it done.

Delegate self-contained work to `agent`, and check `skill` for a matching project skill before \
improvising. Pass `background: true` to `agent` for a long-running or parallelizable piece of \
work you don't need to wait on -- check on it later with `list_agents`/`agent_output`, or cancel \
it with `agent_stop`. Only commit, push, or run other state-changing git/bash commands when \
asked. Use `delete_file` (not `bash rm`) to remove a file -- it's recoverable, `rm` isn't.

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
    /// into the resumed session. An `@path` word (file or directory,
    /// relative to the cwd or absolute) attaches that path's contents --
    /// same convention the interactive REPL uses (see `mentions.rs`).
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
    /// Reruns the same task on a fixed interval, forever (or until
    /// --max-runs is hit) -- for a recurring job with no checklist to poll
    /// (e.g. periodic status checks), instead of an external cron/systemd
    /// timer. Re-running the same task string later resumes its history.
    Schedule {
        /// The task to run every interval
        task: String,
        /// Seconds between runs (the first run happens immediately)
        #[arg(long, default_value_t = 3600)]
        every_secs: u64,
        /// Stop after this many runs instead of running forever
        #[arg(long)]
        max_runs: Option<usize>,
    },
    /// Prints a shell completion script to stdout
    Completion {
        /// Shell to generate the script for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

/// Everything `build_session` assembles, kept around (not just handed to
/// `AgentSession::new` and dropped) so `tui.rs`'s `spawn_side_question` can
/// build its own ephemeral `AgentSession` sharing the same provider/hooks --
/// the same sharing `AgentTool` already does for subagents.
struct BuiltSession {
    session: AgentSession,
    provider: Arc<dyn LlmProvider>,
    cfg: config::ProjectConfig,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
    reporter: Arc<dyn Reporter>,
    tool_ctx: ToolContext,
    show_thinking: Arc<AtomicBool>,
    show_status: Arc<AtomicBool>,
    todo: Arc<TodoWriteTool>,
    checkpoint: Arc<Checkpoint>,
}

fn load_project_config(agent_dir: &Path) -> config::ProjectConfig {
    match config::load(agent_dir) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("failed to load .agent/config.toml: {e}");
            std::process::exit(1);
        }
    }
}

async fn build_session(output: OutputFormat) -> BuiltSession {
    match build_session_with_sink(output, Arc::new(tui::DirectPrintSink)).await {
        Ok(built) => built,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

async fn build_session_with_sink(output: OutputFormat, sink: Arc<dyn tui::OutputSink>) -> Result<BuiltSession, String> {
    let working_dir = std::env::current_dir().expect("cwd");
    let agent_dir = working_dir.join(".agent");
    let cfg = load_project_config(&agent_dir);
    let provider = select_provider(&cfg);
    let tool_ctx = ToolContext {
        working_dir: working_dir.clone(),
        session_id: "cli".to_string(),
        cancel: tokio_util::sync::CancellationToken::new(),
        mailbox: None,
    };

    let plugins = match discover_plugins(&agent_dir) {
        Ok(plugins) => {
            if !plugins.is_empty() {
                let names: Vec<&str> = plugins.iter().map(|p| p.manifest.name.as_str()).collect();
                eprintln!(
                    "loaded {} plugin(s) from .agent/plugins/: {}",
                    plugins.len(),
                    names.join(", ")
                );
            }
            plugins
        }
        Err(e) => return Err(format!("failed to load plugins: {e}")),
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
        Err(e) => return Err(format!("failed to load hooks: {e}")),
    };
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
        Arc::new(WebFetchTool::new()),
    ];
    // Omitted entirely (not registered with a doomed-to-fail key) when unset,
    // so the LLM never sees a tool in its list that it can't actually use.
    if let Ok(key) = std::env::var("TAVILY_API_KEY") {
        tools.push(Arc::new(WebSearchTool::new(key)));
    }

    let skill_roots = std::iter::once(agent_dir.as_path()).chain(plugins.iter().map(|p| p.root.as_path()));
    match discover_all_skills(skill_roots) {
        Ok(skills) => {
            if !skills.is_empty() {
                eprintln!("loaded {} skill(s) from .agent/skills/ and plugins/", skills.len());
                tools.push(Arc::new(SkillTool::new(skills)));
            }
        }
        Err(e) => return Err(format!("failed to load skills: {e}")),
    }

    #[cfg(feature = "wasm")]
    match minder_tools_wasm::load_plugins(&working_dir.join(".agent")).await {
        Ok(plugins) => {
            if !plugins.is_empty() {
                eprintln!("loaded {} wasm plugin tool(s) from .agent/tools/", plugins.len());
            }
            tools.extend(plugins.into_iter().map(Arc::from));
        }
        Err(e) => return Err(format!("failed to load wasm plugins: {e}")),
    }

    #[cfg(feature = "mcp")]
    match minder_tools_mcp::load_mcp_tools(&working_dir.join(".agent")).await {
        Ok(mcp_tools) => {
            if !mcp_tools.is_empty() {
                eprintln!("loaded {} mcp tool(s) from .agent/mcp.toml", mcp_tools.len());
            }
            tools.extend(mcp_tools.into_iter().map(Arc::from));
        }
        Err(e) => return Err(format!("failed to load mcp servers: {e}")),
    }

    #[cfg(feature = "mcp")]
    for plugin in &plugins {
        match minder_tools_mcp::load_plugin_mcp_tools(&plugin.root).await {
            Ok(mcp_tools) => {
                if !mcp_tools.is_empty() {
                    eprintln!(
                        "loaded {} mcp tool(s) from plugin '{}'",
                        mcp_tools.len(),
                        plugin.manifest.name
                    );
                }
                tools.extend(mcp_tools.into_iter().map(Arc::from));
            }
            Err(e) => {
                return Err(format!(
                    "failed to load mcp servers for plugin '{}': {e}",
                    plugin.manifest.name
                ));
            }
        }
    }

    let show_thinking = Arc::new(AtomicBool::new(true));
    let show_status = Arc::new(AtomicBool::new(
        std::env::var("MINDER_SHOW_STATUS_BAR")
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .or(cfg.show_status_bar)
            .unwrap_or(true),
    ));
    let mut terminal_reporter_impl =
        TerminalReporter::with_sink(hooks.clone(), show_thinking.clone(), show_status.clone(), sink);
    if output == OutputFormat::Json {
        terminal_reporter_impl = terminal_reporter_impl.silence_stdout();
    }
    let terminal_reporter: Arc<dyn Reporter> = Arc::new(terminal_reporter_impl);
    let mut reporters: Vec<Arc<dyn Reporter>> = vec![terminal_reporter.clone()];
    if let Ok(path) = std::env::var("MINDER_LOG_FILE") {
        match FileReporter::new(Path::new(&path), LogFormat::from_env()) {
            Ok(file_reporter) => {
                eprintln!("logging to {path}");
                reporters.push(Arc::new(file_reporter));
            }
            Err(e) => eprintln!("failed to open log file {path}: {e}"),
        }
    }
    if let Ok(path) = std::env::var("MINDER_STATUS_FILE") {
        eprintln!("writing status to {path}");
        reporters.push(Arc::new(StatusReporter::new(PathBuf::from(path))));
    }
    let reporter: Arc<dyn Reporter> = if reporters.len() == 1 {
        terminal_reporter
    } else {
        Arc::new(CompositeReporter::new(reporters))
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
        Err(e) => return Err(format!("failed to load subagents: {e}")),
    }
    let provider_factory: Arc<ProviderFactory> = {
        let cfg = cfg.clone();
        Arc::new(move |provider: &str, model: &str| {
            provider_select::build_provider(provider, Some(model.to_string()), &cfg)
        })
    };
    let agent_registry = Arc::new(AgentRegistry::new());
    tools.push(Arc::new(AgentTool::new(
        subagents,
        provider.clone(),
        tools.clone(),
        hooks.clone(),
        reporter.clone(),
        Some(provider_factory),
        agent_registry.clone(),
    )));

    let todo = Arc::new(TodoWriteTool::new());
    tools.push(todo.clone() as Arc<dyn Tool>);
    tools.push(Arc::new(ListAgentsTool::new(agent_registry.clone())));
    tools.push(Arc::new(AgentOutputTool::new(agent_registry.clone())));
    tools.push(Arc::new(AgentStopTool::new(agent_registry)));

    let session = AgentSession::new(
        provider.clone(),
        tools.clone(),
        hooks.clone(),
        SYSTEM_PROMPT,
        tool_ctx.clone(),
    )
    .with_reporter(reporter.clone());

    reporter.on_provider_changed(provider.id(), provider.model()).await;
    if let Err(e) = provider.ensure_model_available(reporter.as_ref()).await {
        return Err(format!("error: {e}"));
    }

    Ok(BuiltSession {
        session,
        provider,
        cfg,
        tools,
        hooks,
        reporter,
        tool_ctx,
        show_thinking,
        show_status,
        todo,
        checkpoint,
    })
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
    Schedule {
        task: String,
        every_secs: u64,
        max_runs: Option<usize>,
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
            Some(CliCommand::Schedule {
                task,
                every_secs,
                max_runs,
            }) => {
                return Command::Schedule {
                    task,
                    every_secs,
                    max_runs,
                };
            }
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

fn print_completion(shell: clap_complete::Shell) {
    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
}

const MAX_STDIN_CHARS: usize = 200_000;

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

/// Same `@path` expansion `run_repl` applies to each typed line.
fn expand_task_mentions(task: String) -> String {
    mentions::expand_mentions(&task, &working_dir())
}

#[tokio::main]
async fn main() {
    match Command::from(Cli::parse()) {
        Command::OneShot { task, output } => run_one_shot(&with_piped_stdin(expand_task_mentions(task)), output).await,
        Command::Continue { task, output } => {
            run_resume(None, task.map(expand_task_mentions).map(with_piped_stdin), output).await
        }
        Command::Resume { id, task, output } => {
            run_resume(Some(id), task.map(expand_task_mentions).map(with_piped_stdin), output).await
        }
        Command::Chat => run_chat().await,
        Command::Loop { file, task_hint } => run_loop_mode(&file, task_hint.as_deref()).await,
        Command::Schedule {
            task,
            every_secs,
            max_runs,
        } => run_schedule_mode(&expand_task_mentions(task), every_secs, max_runs).await,
        Command::Completion { shell } => print_completion(shell),
    }
}

fn working_dir() -> PathBuf {
    std::env::current_dir().expect("cwd")
}

const INTERRUPT_GRACE_PERIOD: Duration = Duration::from_millis(1500);

const IDLE_CTRL_C_QUIT_WINDOW: Duration = Duration::from_secs(2);

async fn run_turn_interruptible(session: &mut AgentSession, input: &str) -> Result<Message, AgentError> {
    let pre_turn_len = session.messages().len();
    let cancel = session.reset_cancel_token();
    let steering_tx = session.enable_steering();
    let mut watcher = InputWatcher::spawn(cancel, steering_tx);

    let result = 'turn: {
        let turn = session.run_turn(input);
        tokio::pin!(turn);

        tokio::select! {
            result = &mut turn => break 'turn result,
            _ = watcher.next_cancel() => {}
        }

        tokio::select! {
            result = &mut turn => result,
            _ = watcher.next_cancel() => Err(AgentError::Interrupted),
            _ = tokio::time::sleep(INTERRUPT_GRACE_PERIOD) => Err(AgentError::Interrupted),
        }
    };

    watcher.stop().await;

    if matches!(result, Err(AgentError::Interrupted)) {
        session.discard_interrupted_turn(pre_turn_len);
    }
    result
}

fn print_turn_error(err: &AgentError, checkpoint: &Checkpoint) {
    if matches!(err, AgentError::Interrupted) {
        println!("Interrupted.");
        if !checkpoint.is_empty() {
            println!(
                "note: this turn already edited file(s) on disk before being interrupted -- run /undo to revert them."
            );
        }
    } else {
        eprintln!("error: {err}");
    }
}

/// `persist` only ever runs once a turn has concluded (successfully, or
/// after `discard_interrupted_turn` already rolled back an in-memory
/// Ctrl-C), so it doubles as "this session is safely at rest" -- clears
/// `interrupted` (see `mark_turn_started`) accordingly.
fn persist(dir: &Path, record: &mut SessionRecord, session: &AgentSession) {
    record.system_prompt = session.system_prompt().to_string();
    record.messages = session.messages().to_vec();
    record.interrupted = false;
    if let Err(e) = session_store::save(dir, record) {
        eprintln!("warning: failed to save session: {e}");
    }
}

/// Flags `record` as mid-turn *before* running one, so a crash (panic,
/// SIGKILL, power loss) between this call and the matching `persist` leaves
/// `interrupted: true` on disk -- what `run_chat` checks on its next
/// invocation to offer resuming instead of starting fresh. Called only from
/// the two interactive REPL loops (`run_repl_fallback`, `tui::run_tui_repl`)
/// that `run_chat` actually leads into -- one-shot paths and `minder loop`/
/// `minder schedule` already resume unconditionally (explicit
/// `--continue`/`--resume`) or via their own deterministic per-iteration
/// persistence, so they don't need this.
fn mark_turn_started(dir: &Path, record: &mut SessionRecord) {
    record.interrupted = true;
    if let Err(e) = session_store::save(dir, record) {
        eprintln!("warning: failed to save session: {e}");
    }
}

fn print_json_result(session: &AgentSession, result: &Result<Message, AgentError>) {
    let payload = json_result_payload(session.provider_id(), session.model(), result);
    println!("{payload}");
}

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

enum ReplBackend {
    Tui(tui::PinnedHandles),
    Fallback,
}

async fn build_repl_session(output: OutputFormat) -> (BuiltSession, ReplBackend) {
    if input_watcher::supports_key_watching()
        && let Ok(handles) = tui::init()
    {
        let color = color_enabled(std::io::stdout().is_terminal());
        let sink: Arc<dyn tui::OutputSink> = Arc::new(tui::FullscreenSink::new(handles.clone(), color));
        // No event loop reads keystrokes until `run_tui_repl` starts -- show
        // the box as disabled rather than blank until then.
        handles.input.lock().unwrap().disabled_message = Some("please wait, setting up the session…".to_string());
        let built = match build_session_with_sink(output, sink).await {
            Ok(built) => built,
            Err(e) => {
                tui::restore_terminal();
                eprintln!("{e}");
                std::process::exit(1);
            }
        };
        return (built, ReplBackend::Tui(handles));
    }
    (build_session(output).await, ReplBackend::Fallback)
}

/// Starts fresh, unless the most recently updated session for this
/// directory was left `interrupted` (see `mark_turn_started`/`persist`) --
/// evidence the last `minder chat` here ended mid-turn rather than cleanly
/// (`exit`/`quit`), in which case it's restored automatically instead of
/// silently discarding it.
async fn run_chat() {
    let dir = working_dir();
    let (mut built, backend) = build_repl_session(OutputFormat::Text).await;
    let mut record = match session_store::load_latest(&dir) {
        Ok(Some(prior)) if prior.interrupted => {
            eprintln!(
                "resuming a session that didn't finish cleanly last time ({} prior message(s))",
                prior.messages.len()
            );
            built
                .session
                .restore(prior.system_prompt.clone(), prior.messages.clone());
            prior
        }
        _ => SessionRecord::new(),
    };
    run_repl(&mut built, &dir, &mut record, backend).await;
}

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

    match task {
        Some(task) => {
            let mut built = build_session(output).await;
            built
                .session
                .restore(record.system_prompt.clone(), record.messages.clone());

            let result = built.session.run_turn(&task).await;
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
        None => {
            let (mut built, backend) = build_repl_session(OutputFormat::Text).await;
            built
                .session
                .restore(record.system_prompt.clone(), record.messages.clone());
            run_repl(&mut built, &dir, &mut record, backend).await;
        }
    }
}

const FALLBACK_RULE_WIDTH: usize = 64;

const MIN_RULE_WIDTH: usize = 20;

fn rule_width() -> usize {
    terminal_size::terminal_size()
        .map(|(terminal_size::Width(w), _)| (w as usize).saturating_sub(1).max(MIN_RULE_WIDTH))
        .unwrap_or(FALLBACK_RULE_WIDTH)
}

fn color_enabled(stream_is_tty: bool) -> bool {
    stream_is_tty && std::env::var_os("NO_COLOR").is_none()
}

fn rule_line(color: bool) -> String {
    let rule = "-".repeat(rule_width());
    if color { format!("{DIM}{rule}{RESET}") } else { rule }
}

fn repl_prompt(color: bool) -> String {
    if color {
        format!("{BOLD}{CYAN}❯{RESET} ")
    } else {
        "> ".to_string()
    }
}

fn status_line(session: &AgentSession, dir: &Path, color: bool) -> String {
    let text = format!("{} ({}) · {}", session.provider_id(), session.model(), dir.display());
    if color { format!("{DIM}{text}{RESET}") } else { text }
}

fn hint_line(color: bool, key_watching_supported: bool) -> String {
    let text = if key_watching_supported {
        "Esc/Ctrl-C cancel input/turn, type + Enter to steer a running turn · Ctrl-D, 'exit'/'quit', or Ctrl-C twice to leave \
         · /help for commands · @path attaches a file/dir"
    } else {
        "Ctrl-C cancels a running turn (press twice at the prompt to quit) · Ctrl-D or 'exit'/'quit' to leave \
         · /help for commands · @path attaches a file/dir"
    };
    if color {
        format!("{DIM}{text}{RESET}")
    } else {
        text.to_string()
    }
}

const SLASH_HELP: &str = "\
Available commands:
  /help                      Show this list
  /model                     Show the active provider and model
  /model <provider> [model]  Switch the active provider/model mid-session (keeps history)
  /models                    List locally pulled models, marking the active one
  /clear                     Clear the conversation history (keeps the session file, starts fresh)
  /status                    Toggle showing the active provider/model in the spinner while a turn runs
  /thinking                  Toggle showing the model's extended-thinking output
  /todo                      Show the model's current todo list
  /undo                      Revert the file changes from the most recently completed turn
  exit, quit                 Leave (Ctrl-D also works)

Type @path (Tab to complete) to attach a file or directory's contents, e.g. @src/main.rs or @src/.";

/// Names accepted by `handle_slash_command`, kept in sync with `SLASH_HELP`
/// above; also drives completion/hinting in `SlashCommandHelper`.
const SLASH_COMMANDS: &[&str] = &["help", "model", "clear", "status", "thinking", "todo", "undo"];

/// Commands still matching what's typed after `/`, or `None` if `line`/`pos`
/// isn't in "typing a command name" position at all (no leading `/`, cursor
/// not at the end, or a space already reached -- i.e. past the command name
/// into its arguments).
pub(crate) fn matching_slash_commands(line: &str, pos: usize) -> Option<Vec<&'static str>> {
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
/// Tab at all. Covers both `/`-commands and `@path` mentions (see `mentions.rs`).
struct SlashCommandHelper {
    color: bool,
    working_dir: PathBuf,
    /// Snapshot of locally pulled Ollama models, fetched once at REPL
    /// startup -- drives `/model`'s argument completion. Doesn't pick up a
    /// model pulled mid-session; `/models` always re-fetches fresh.
    known_models: Vec<String>,
}

/// Completion candidates for `/model`'s `<provider> [model]` arguments:
/// "ollama" (today's only provider), then a name from `known_models`.
pub(crate) fn matching_model_args(line: &str, pos: usize, known_models: &[String]) -> Option<(usize, Vec<String>)> {
    if pos != line.len() {
        return None;
    }
    let rest = line.strip_prefix("/model ")?;
    if let Some((provider, model_prefix)) = rest.split_once(' ') {
        if provider != "ollama" {
            return None;
        }
        let matches: Vec<String> = known_models
            .iter()
            .filter(|m| m.starts_with(model_prefix))
            .cloned()
            .collect();
        return (!matches.is_empty()).then(|| (line.len() - model_prefix.len(), matches));
    }
    let matches: Vec<String> = ["ollama"]
        .into_iter()
        .filter(|p| p.starts_with(rest))
        .map(String::from)
        .collect();
    (!matches.is_empty()).then(|| (line.len() - rest.len(), matches))
}

impl Completer for SlashCommandHelper {
    type Candidate = Pair;

    fn complete(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> rustyline::Result<(usize, Vec<Pair>)> {
        if let Some(matches) = matching_slash_commands(line, pos) {
            let candidates = matches
                .into_iter()
                .map(|cmd| Pair {
                    display: format!("/{cmd}"),
                    replacement: format!("/{cmd} "),
                })
                .collect();
            return Ok((0, candidates));
        }
        if let Some((start, matches)) = matching_model_args(line, pos, &self.known_models) {
            let candidates = matches
                .into_iter()
                .map(|m| Pair {
                    display: m.clone(),
                    replacement: format!("{m} "),
                })
                .collect();
            return Ok((start, candidates));
        }
        if let Some((start, prefix)) = mentions::at_mention_token(line, pos) {
            return Ok((start, mentions::complete_at_mention(prefix, &self.working_dir)));
        }
        Ok((pos, Vec::new()))
    }
}

impl Hinter for SlashCommandHelper {
    type Hint = SlashCommandHint;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<SlashCommandHint> {
        if let Some(matches) = matching_slash_commands(line, pos) {
            let typed_len = pos - 1; // chars typed after the leading '/'
            return match matches.as_slice() {
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
            };
        }
        if let Some((start, matches)) = matching_model_args(line, pos, &self.known_models) {
            let typed_len = pos - start;
            return match matches.as_slice() {
                [] => None,
                [only] => {
                    let suffix = &only[typed_len..];
                    (!suffix.is_empty()).then(|| SlashCommandHint {
                        display: suffix.to_string(),
                        completion_len: suffix.len(),
                    })
                }
                many => Some(SlashCommandHint {
                    display: format!("  {}", many.join("  ")),
                    completion_len: 0,
                }),
            };
        }
        // Ghost-text hints only make sense at end-of-line.
        if pos == line.len()
            && let Some((_, prefix)) = mentions::at_mention_token(line, pos)
            && let Some((display, completion_len)) = mentions::at_mention_hint(prefix, &self.working_dir)
        {
            return Some(SlashCommandHint {
                display,
                completion_len,
            });
        }
        None
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
async fn handle_slash_command(input: &str, built: &mut BuiltSession, dir: &Path, record: &mut SessionRecord) {
    let (cmd, rest) = input.split_once(' ').unwrap_or((input, ""));
    let rest = rest.trim();

    match cmd {
        "help" => println!("{SLASH_HELP}"),
        "model" if !rest.is_empty() => run_model_command(rest, built).await,
        "model" => println!("{} · {}", built.session.provider_id(), built.session.model()),
        "clear" => {
            let system_prompt = built.session.system_prompt().to_string();
            built.session.restore(system_prompt, Vec::new());
            persist(dir, record, &built.session);
            println!("Conversation cleared.");
        }
        "status" => {
            let shown = !built.show_status.fetch_xor(true, Ordering::Relaxed);
            println!("Spinner status bar is now {}.", if shown { "on" } else { "off" });
        }
        "thinking" => {
            let shown = !built.show_thinking.fetch_xor(true, Ordering::Relaxed);
            println!("Extended-thinking display is now {}.", if shown { "on" } else { "off" });
        }
        "todo" => println!("{}", format_checklist(&built.todo.items())),
        "undo" => run_undo_command(built, dir).await,
        "models" => run_models_command(built).await,
        other => println!("Unknown command '/{other}'. Type /help for a list."),
    }
}

/// `/models`: lists locally pulled models, marking the active one.
async fn run_models_command(built: &BuiltSession) {
    match built.provider.list_models().await {
        Ok(models) if models.is_empty() => {
            println!("No local models found -- try `ollama pull <model>`.")
        }
        Ok(models) => {
            for m in models {
                let marker = if m == built.session.model() { "  (active)" } else { "" };
                println!("  {m}{marker}");
            }
        }
        Err(e) => println!("failed to list models: {e}"),
    }
}

/// `/model <provider> [model]`: swaps the live provider, keeping history.
/// Doesn't affect the `agent` tool's own default provider for subagents.
async fn run_model_command(rest: &str, built: &mut BuiltSession) {
    let (provider_name, model) = rest.split_once(' ').unwrap_or((rest, ""));
    let model = (!model.trim().is_empty()).then(|| model.trim().to_string());

    match provider_select::build_provider(provider_name, model, &built.cfg) {
        Ok(provider) => {
            if let Err(e) = provider.ensure_model_available(built.reporter.as_ref()).await {
                println!("error: {e}");
                return;
            }
            built.session.set_provider(provider.clone()).await;
            built.provider = provider;
            println!(
                "Switched to {} · {}",
                built.session.provider_id(),
                built.session.model()
            );
        }
        Err(e) => println!("error: {e}"),
    }
}

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

/// Short "it's alive" banner shown once when a REPL starts. Routed through
/// `reporter.on_notice` rather than raw `eprintln!` -- `tui::init()` has
/// already taken over cursor placement by the time this runs, so writing
/// straight to stderr would scramble the banner into the pinned box.
async fn print_banner(session: &AgentSession, record: &SessionRecord, reporter: &Arc<dyn Reporter>) {
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

    let accent = format!("{YELLOW}{BOLD}");
    reporter.on_notice("").await;
    reporter
        .on_notice(&format!(
            "  {}   {}",
            paint(&accent, " ◆ "),
            paint(BOLD, &format!("v{version}"))
        ))
        .await;
    reporter
        .on_notice(&format!(
            "  {}   {}",
            paint(&accent, "◆ ◆"),
            paint(DIM, &format!("{} · {status}", session.provider_id()))
        ))
        .await;
    reporter
        .on_notice(&format!(
            "  {}   {}",
            paint(&accent, " ◆ "),
            paint(DIM, &working_dir().display().to_string())
        ))
        .await;
    reporter.on_notice("").await;
}

/// Dispatches to whichever REPL implementation `backend` selected (see
/// `build_repl_session`): the `ratatui` pinned-input-box loop, or today's
/// `rustyline` + per-turn `InputWatcher` split, completely unchanged.
async fn run_repl(built: &mut BuiltSession, dir: &Path, record: &mut SessionRecord, backend: ReplBackend) {
    print_banner(&built.session, record, &built.reporter).await;
    match backend {
        ReplBackend::Tui(handles) => tui::run_tui_repl(built, dir, record, handles).await,
        ReplBackend::Fallback => run_repl_fallback(built, dir, record).await,
    }
}

async fn run_repl_fallback(built: &mut BuiltSession, dir: &Path, record: &mut SessionRecord) {
    let color = color_enabled(std::io::stdout().is_terminal());
    let prompt = repl_prompt(color);

    let key_watching_supported = input_watcher::supports_key_watching();
    if !key_watching_supported {
        eprintln!(
            "warning: this terminal doesn't support raw input -- only Ctrl-C (not Esc) cancels a running turn, and mid-turn steering is unavailable"
        );
    }

    let history = session_store::history_path(dir).ok();
    let known_models = built.provider.list_models().await.unwrap_or_default();
    let mut editor: ReplEditor = Editor::new().expect("failed to initialize line editor");
    editor.set_helper(Some(SlashCommandHelper {
        color,
        working_dir: dir.to_path_buf(),
        known_models,
    }));
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }

    let mut last_idle_interrupt: Option<Instant> = None;

    loop {
        println!("{}", status_line(&built.session, dir, color));
        println!("{}", rule_line(color));

        let line = match editor.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                if last_idle_interrupt.is_some_and(|t| t.elapsed() < IDLE_CTRL_C_QUIT_WINDOW) {
                    break;
                }
                last_idle_interrupt = Some(Instant::now());
                println!("(Ctrl-C again to exit)");
                println!("{}", rule_line(color));
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("error: {e}");
                break;
            }
        };
        last_idle_interrupt = None;
        println!("{}", rule_line(color));
        println!("{}", hint_line(color, key_watching_supported));
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
            handle_slash_command(command, built, dir, record).await;
            println!();
            continue;
        }

        let expanded = mentions::expand_mentions(line, dir);
        built.checkpoint.start_turn();
        mark_turn_started(dir, record);
        if let Err(e) = run_turn_interruptible(&mut built.session, &expanded).await {
            print_turn_error(&e, &built.checkpoint);
        }
        persist(dir, record, &built.session);
        println!();
    }
}

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

/// Reruns `task` every `every_secs`, forever unless `max_runs` caps it --
/// see `CliCommand::Schedule`. Resumes via `session_store::key_for_task`
/// exactly like `run_loop_mode` resumes via `key_for_path`: re-running the
/// same task string later picks the same session back up.
async fn run_schedule_mode(task: &str, every_secs: u64, max_runs: Option<usize>) {
    let dir = working_dir();
    let mut built = build_session(OutputFormat::Text).await;

    let key = session_store::key_for_task(task);
    let mut record = match session_store::load_by_id(&dir, &key) {
        Ok(Some(record)) => {
            built
                .session
                .restore(record.system_prompt.clone(), record.messages.clone());
            eprintln!(
                "resuming schedule session for '{task}' ({} prior message(s))",
                record.messages.len()
            );
            record
        }
        Ok(None) => SessionRecord::with_id(key),
        Err(e) => {
            eprintln!("warning: failed to load prior schedule session: {e}");
            SessionRecord::with_id(key)
        }
    };

    eprintln!("[schedule] running '{task}' every {every_secs}s (Ctrl-C to stop)");
    let opts = schedule_mode::ScheduleOptions {
        interval: Duration::from_secs(every_secs),
        max_runs,
    };
    let result = schedule_mode::run(&mut built.session, task, opts, |session| {
        persist(&dir, &mut record, session);
    })
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
            mailbox: None,
        }
    }

    fn test_built_session() -> BuiltSession {
        let provider: Arc<dyn LlmProvider> = Arc::new(FixedTextProvider("all done"));
        let session = AgentSession::new(provider.clone(), Vec::new(), None, "test agent", test_tool_ctx())
            .with_reporter(Arc::new(minder_core::NoopReporter));
        BuiltSession {
            session,
            provider,
            cfg: config::ProjectConfig::default(),
            tools: Vec::new(),
            hooks: None,
            reporter: Arc::new(minder_core::NoopReporter),
            tool_ctx: test_tool_ctx(),
            show_thinking: Arc::new(AtomicBool::new(false)),
            show_status: Arc::new(AtomicBool::new(true)),
            todo: Arc::new(TodoWriteTool::new()),
            checkpoint: Arc::new(Checkpoint::new()),
        }
    }

    #[tokio::test]
    async fn model_command_switches_the_live_session_to_a_key_free_provider() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{"name": "llama3.2"}],
            })))
            .mount(&server)
            .await;

        let mut built = test_built_session();
        built.cfg.ollama_base_url = Some(server.uri());
        assert_eq!(built.session.provider_id(), "fixed");

        run_model_command("ollama llama3.2", &mut built).await;

        assert_eq!(built.session.provider_id(), "ollama");
        assert_eq!(built.session.model(), "llama3.2");
    }

    #[tokio::test]
    async fn model_command_leaves_the_session_untouched_on_an_unknown_provider() {
        let mut built = test_built_session();

        run_model_command("not-a-real-provider", &mut built).await;

        assert_eq!(built.session.provider_id(), "fixed");
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
        let payload = json_result_payload("ollama", "llama3.2", &ok);
        assert_eq!(payload["provider"], "ollama");
        assert_eq!(payload["model"], "llama3.2");
        assert_eq!(payload["answer"], "42");
        assert!(payload["error"].is_null());
    }

    #[test]
    fn json_payload_carries_the_error_on_failure() {
        let err: Result<Message, AgentError> = Err(AgentError::HookBlocked("blocked by policy".to_string()));
        let payload = json_result_payload("ollama", "llama3.2", &err);
        assert!(payload["answer"].is_null());
        assert!(payload["error"].as_str().unwrap().contains("blocked by policy"));
    }

    fn test_helper() -> SlashCommandHelper {
        SlashCommandHelper {
            color: false,
            working_dir: std::env::temp_dir(),
            known_models: Vec::new(),
        }
    }

    fn test_helper_with_models(models: &[&str]) -> SlashCommandHelper {
        SlashCommandHelper {
            known_models: models.iter().map(|m| m.to_string()).collect(),
            ..test_helper()
        }
    }

    fn complete_at_cursor(line: &str) -> (usize, Vec<Pair>) {
        let history = rustyline::history::MemHistory::new();
        let ctx = Context::new(&history);
        test_helper().complete(line, line.len(), &ctx).unwrap()
    }

    fn complete_at_cursor_with(helper: &SlashCommandHelper, line: &str) -> (usize, Vec<Pair>) {
        let history = rustyline::history::MemHistory::new();
        let ctx = Context::new(&history);
        helper.complete(line, line.len(), &ctx).unwrap()
    }

    fn hint_at_cursor_with(helper: &SlashCommandHelper, line: &str) -> Option<String> {
        let history = rustyline::history::MemHistory::new();
        let ctx = Context::new(&history);
        helper
            .hint(line, line.len(), &ctx)
            .map(|h| rustyline::hint::Hint::display(&h).to_string())
    }

    fn hint_at_cursor(line: &str) -> Option<String> {
        let history = rustyline::history::MemHistory::new();
        let ctx = Context::new(&history);
        test_helper()
            .hint(line, line.len(), &ctx)
            .map(|h| rustyline::hint::Hint::display(&h).to_string())
    }

    #[test]
    fn slash_completion_lists_all_commands_for_bare_slash() {
        let (start, candidates) = complete_at_cursor("/");
        assert_eq!(start, 0);
        let names: Vec<&str> = candidates.iter().map(|p| p.display.as_str()).collect();
        assert_eq!(names.len(), SLASH_COMMANDS.len());
        assert!(names.contains(&"/model"));
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
        let (_, candidates) = complete_at_cursor("/model fix the ");
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
        assert_eq!(hint_at_cursor("/model fix the "), None);
    }

    #[test]
    fn slash_hint_absent_for_plain_text() {
        assert_eq!(hint_at_cursor("hello"), None);
    }

    #[test]
    fn model_completion_suggests_ollama_as_the_provider() {
        let helper = test_helper_with_models(&["qwen2.5-coder:14b"]);
        let (start, candidates) = complete_at_cursor_with(&helper, "/model oll");
        assert_eq!(start, "/model ".len());
        assert_eq!(candidates[0].display, "ollama");
        assert_eq!(candidates[0].replacement, "ollama ");
    }

    #[test]
    fn model_completion_suggests_pulled_model_names() {
        let helper = test_helper_with_models(&["qwen2.5-coder:14b", "llama3.2"]);
        let (start, candidates) = complete_at_cursor_with(&helper, "/model ollama qwen");
        assert_eq!(start, "/model ollama ".len());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].display, "qwen2.5-coder:14b");
    }

    #[test]
    fn model_completion_is_empty_for_an_unknown_provider() {
        let helper = test_helper_with_models(&["qwen2.5-coder:14b"]);
        let (_, candidates) = complete_at_cursor_with(&helper, "/model anthropic ");
        assert!(candidates.is_empty());
    }

    #[test]
    fn model_hint_completes_the_rest_of_a_single_matching_model() {
        let helper = test_helper_with_models(&["qwen2.5-coder:14b", "llama3.2"]);
        let hint = hint_at_cursor_with(&helper, "/model ollama qwen").unwrap();
        assert_eq!(hint, "2.5-coder:14b");
    }
}
