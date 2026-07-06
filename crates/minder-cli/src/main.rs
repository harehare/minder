mod loop_mode;
mod markdown;
mod provider_select;
mod reporter;
mod session_store;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use minder_core::{AgentSession, HookPort, Tool, ToolContext};
use minder_hooks::HookEngine;
use minder_tools::{
    AgentTool, BashTool, EditFileTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool, GlobTool, GrepTool,
    LsTool, ReadFileTool, SkillTool, WebFetchTool, WebSearchTool, WorktreeAddTool, WorktreeListTool,
    WorktreeRemoveTool, WriteFileTool, builtin_subagents, discover_skills, discover_subagents,
};

use provider_select::select_provider;
use reporter::TerminalReporter;
use session_store::SessionRecord;

const SYSTEM_PROMPT: &str =
    "You are a careful coding assistant. Use the available tools to inspect and run code before answering.";

const USAGE: &str = "usage:\n  \
    minder \"<task>\"                  run a single task to completion (session saved for --continue)\n  \
    minder --continue|-c [\"<task>\"]  resume the most recent session in this project\n  \
    minder --resume|-r <id> [\"<task>\"] resume a specific session by id (or unambiguous prefix)\n  \
    minder chat                     start an interactive session (empty input or Ctrl-D to exit)\n  \
    minder loop <file.md> [\"<task>\"] work through the file's unchecked checklist items, then \
    keep polling it for new ones (mq-lang embedded, see README) -- runs until stopped (Ctrl-C) \
    or a safety limit is hit\n  \
    (with no <task>, --continue/--resume drop into an interactive session too)";

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

    let reporter = Arc::new(TerminalReporter::new(hooks.clone()));

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

enum Command {
    OneShot { task: String },
    Continue { task: Option<String> },
    Resume { id: String, task: Option<String> },
    Chat,
    Loop { file: PathBuf, task_hint: Option<String> },
    Usage,
}

fn parse_args(args: &[String]) -> Command {
    match args.first().map(String::as_str) {
        Some("loop") => match args.get(1) {
            Some(file) => Command::Loop {
                file: PathBuf::from(file),
                task_hint: args.get(2).cloned(),
            },
            None => Command::Usage,
        },
        Some("chat") => Command::Chat,
        Some("--continue") | Some("-c") => Command::Continue { task: args.get(1).cloned() },
        Some("--resume") | Some("-r") => match args.get(1) {
            Some(id) => Command::Resume {
                id: id.clone(),
                task: args.get(2).cloned(),
            },
            None => Command::Usage,
        },
        Some(task) => Command::OneShot { task: task.to_string() },
        None => Command::Usage,
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match parse_args(&args) {
        Command::OneShot { task } => run_one_shot(&task).await,
        Command::Continue { task } => run_resume(None, task).await,
        Command::Resume { id, task } => run_resume(Some(id), task).await,
        Command::Chat => run_chat().await,
        Command::Loop { file, task_hint } => run_loop_mode(&file, task_hint.as_deref()).await,
        Command::Usage => {
            eprintln!("{USAGE}");
            std::process::exit(1);
        }
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

/// Reads tasks from stdin in a loop, feeding each into the same session so
/// context carries over between them; saves after every turn so Ctrl-C or a
/// crash mid-conversation loses at most the in-flight turn. Exits on EOF
/// (Ctrl-D), or an "exit"/"quit" line.
async fn run_repl(session: &mut AgentSession, dir: &Path, record: &mut SessionRecord) {
    eprintln!("entering interactive session -- 'exit'/'quit' or Ctrl-D to leave");
    loop {
        eprint!("> ");
        let _ = std::io::stderr().flush();

        let mut line = String::new();
        let read = std::io::stdin().read_line(&mut line).unwrap_or(0);
        if read == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }

        if let Err(e) = session.run_turn(line).await {
            eprintln!("error: {e}");
        }
        persist(dir, record, session);
    }
}

async fn run_loop_mode(file: &Path, task_hint: Option<&str>) {
    let mut session = build_session().await;
    if let Err(e) = loop_mode::run(&mut session, file, task_hint, loop_mode::LoopOptions::default()).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
