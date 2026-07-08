mod file_reporter;
mod loop_mode;
mod markdown;
mod provider_select;
mod reporter;
mod session_store;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use minder_core::{AgentSession, HookPort, Reporter, Tool, ToolContext};
use minder_hooks::HookEngine;
use minder_tools::{
    AgentTool, BashTool, EditFileTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool, GlobTool, GrepTool,
    LsTool, ReadFileTool, SkillTool, WebFetchTool, WebSearchTool, WorktreeAddTool, WorktreeListTool,
    WorktreeRemoveTool, WriteFileTool, builtin_subagents, discover_skills, discover_subagents,
};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use file_reporter::{CompositeReporter, FileReporter};
use provider_select::select_provider;
use reporter::{BOLD, CYAN, DIM, RESET, TerminalReporter, YELLOW};
use session_store::SessionRecord;

const SYSTEM_PROMPT: &str = "\
You are minder, a coding agent working in a git repository via tool calls. Investigate with \
`read_file`/`grep`/`glob`/`git_log`/`git_diff` before answering or editing -- read a file before \
editing it, prefer `edit_file` over `write_file` for existing files, and verify a change with \
`git_diff`/tests before calling it done.

Delegate self-contained work to `agent`, and check `skill` for a matching project skill before \
improvising. Only commit, push, or run other state-changing git/bash commands when asked.

Keep replies short and grounded in what the tools actually returned.";

#[derive(Parser)]
#[command(version, about = "Multi-provider coding-agent harness CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Run a single task to completion
    task: Option<String>,

    /// Resume the most recent session in this project
    #[arg(short, long, conflicts_with = "resume")]
    continue_session: bool,

    /// Resume a specific session by id (or unambiguous prefix)
    #[arg(short, long)]
    resume: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive session
    Chat,
    /// Work through the file's unchecked checklist items, then keep polling it for new ones
    Loop {
        /// The checklist file
        file: PathBuf,
        /// Optional task hint
        task: Option<String>,
    },
}

/// Builds the tool/hook/skill/plugin-wired session shared by both the
/// one-shot and `loop` entry points -- only the prompt(s) fed into
/// `run_turn` differ between the two modes.
async fn build_session() -> AgentSession {
    let provider = select_provider();
    let working_dir = std::env::current_dir().expect("cwd");
    let tool_ctx = ToolContext {
        working_dir: working_dir.clone(),
        session_id: "cli".to_string(),
        cancel: tokio_util::sync::CancellationToken::new(),
    };

    let agent_dir = working_dir.join(".agent");
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

    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool),
        Arc::new(WriteFileTool),
        Arc::new(EditFileTool),
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

    let terminal_reporter: Arc<dyn Reporter> = Arc::new(TerminalReporter::new(hooks.clone()));
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

    AgentSession::new(provider, tools, hooks, SYSTEM_PROMPT, tool_ctx).with_reporter(reporter)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if let Some(command) = cli.command {
        match command {
            Commands::Chat => run_chat().await,
            Commands::Loop { file, task } => run_loop_mode(&file, task.as_deref()).await,
        }
    } else if cli.continue_session {
        run_resume(None, cli.task).await;
    } else if let Some(id) = cli.resume {
        run_resume(Some(id), cli.task).await;
    } else if let Some(task) = cli.task {
        run_one_shot(&task).await;
    } else {
        run_chat().await;
    }
}

fn working_dir() -> PathBuf {
    std::env::current_dir().expect("cwd")
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

async fn run_one_shot(task: &str) {
    let dir = working_dir();
    let mut session = build_session().await;
    let mut record = SessionRecord::new();

    let result = session.run_turn(task).await;
    persist(&dir, &mut record, &session);

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run_chat() {
    let dir = working_dir();
    let mut session = build_session().await;
    let mut record = SessionRecord::new();
    run_repl(&mut session, &dir, &mut record).await;
}

/// Resumes a saved session (latest if `id` is `None`) and either runs one
/// more task, or drops into an interactive session when no task is given.
async fn run_resume(id: Option<String>, task: Option<String>) {
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

    let mut session = build_session().await;
    session.restore(record.system_prompt.clone(), record.messages.clone());

    match task {
        Some(task) => {
            let result = session.run_turn(&task).await;
            persist(&dir, &mut record, &session);
            if let Err(e) = result {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        None => run_repl(&mut session, &dir, &mut record).await,
    }
}

/// Interior width of the input box (including the border characters).
fn box_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(80)
}

/// Both the prompt and its surrounding box print to stdout (rustyline
/// renders the prompt there, not stderr), so they need the same color
/// decision or the border and the `❯` would disagree about `NO_COLOR`.
fn color_enabled(stream_is_tty: bool) -> bool {
    stream_is_tty && std::env::var_os("NO_COLOR").is_none()
}

fn box_border(left: char, right: char, color: bool, width: usize) -> String {
    let rule = "─".repeat(width.saturating_sub(2));
    if color {
        format!("{DIM}{left}{rule}{right}{RESET}")
    } else {
        format!("{left}{rule}{right}")
    }
}

/// Builds the REPL's input prompt: a boxed `❯ ` so a turn boundary reads as
/// its own input area rather than a bare `> `. Only the left border lives in
/// the prompt string itself -- rustyline owns everything after it, so the
/// box can't close on the right without redrawing on every keystroke; see
/// `run_repl` for the top/bottom rules drawn around it.
///
/// We wrap ANSI escape sequences in \x01/\x02 so rustyline knows they are
/// zero-width, preventing cursor drift and incorrect line wrapping.
fn repl_prompt(color: bool) -> String {
    if color {
        format!("\x01{DIM}\x02│\x01{RESET}\x02 \x01{BOLD}{CYAN}\x02❯\x01{RESET}\x02 ")
    } else {
        "| > ".to_string()
    }
}

/// The line above each turn's input box: provider/model and working
/// directory, so that context stays visible without repeating the full
/// startup banner every turn.
fn status_line(session: &AgentSession, dir: &Path, color: bool) -> String {
    let text = format!("{} · {}", session.provider_id(), dir.display());
    if color { format!("{DIM}{text}{RESET}") } else { text }
}

/// The line below each turn's input box: the same keyboard shortcuts every
/// time, so they're always one glance away instead of scrolled off after
/// the first turn.
fn hint_line(color: bool) -> String {
    let text = "Ctrl-C cancel input · Ctrl-D or 'exit'/'quit' to leave";
    if color {
        format!("{DIM}{text}{RESET}")
    } else {
        text.to_string()
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
async fn run_repl(session: &mut AgentSession, dir: &Path, record: &mut SessionRecord) {
    print_banner(session, record);

    let history = session_store::history_path(dir).ok();
    let mut editor = DefaultEditor::new().expect("failed to initialize line editor");
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }

    // Decided once (a terminal doesn't change tty-ness mid-session) and
    // shared by every box/status/hint line drawn below, so they never
    // disagree with the prompt itself about `NO_COLOR`.
    let color = color_enabled(std::io::stdout().is_terminal());
    let prompt = repl_prompt(color);

    loop {
        let width = box_width();
        println!("{}", status_line(session, dir, color));
        println!("{}", box_border('╭', '╮', color, width));

        let line = match editor.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                println!("{}", box_border('╰', '╯', color, width));
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("error: {e}");
                break;
            }
        };
        println!("{}", box_border('╰', '╯', color, width));
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

        if let Err(e) = session.run_turn(line).await {
            eprintln!("error: {e}");
        }
        persist(dir, record, session);
        println!();
    }
}

/// `minder loop <file>` is keyed by `file`'s canonical path (not a random
/// id) so re-running the same command after a crash, Ctrl-C, or a container
/// restart resumes the same conversation automatically -- see
/// `session_store::key_for_path`.
async fn run_loop_mode(file: &Path, task_hint: Option<&str>) {
    let dir = working_dir();
    let mut session = build_session().await;

    let key = session_store::key_for_path(file);
    let mut record = match session_store::load_by_id(&dir, &key) {
        Ok(Some(record)) => {
            session.restore(record.system_prompt.clone(), record.messages.clone());
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
        &mut session,
        file,
        task_hint,
        loop_mode::LoopOptions::default(),
        |session| {
            persist(&dir, &mut record, session);
        },
    )
    .await;
    persist(&dir, &mut record, &session);

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
