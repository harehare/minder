mod loop_mode;
mod provider_select;
mod reporter;

use std::path::Path;
use std::sync::Arc;

use agent_core::{AgentSession, HookPort, Tool, ToolContext};
use agent_hooks::HookEngine;
use agent_tools::{
    BashTool, EditFileTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool, GlobTool,
    GrepTool, LsTool, ReadFileTool, SkillTool, WebFetchTool, WebSearchTool, WriteFileTool,
    discover_skills,
};

use provider_select::select_provider;
use reporter::TerminalReporter;

const SYSTEM_PROMPT: &str = "You are a careful coding assistant. Use the available tools to inspect and run code before answering.";

const USAGE: &str = "usage:\n  \
    minder \"<task>\"                 run a single task to completion\n  \
    minder loop <file.md> [\"<task>\"] work through the file's unchecked checklist items, then \
    keep polling it for new ones (mq-lang embedded, see README) -- runs until stopped (Ctrl-C) \
    or a safety limit is hit";

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

    let hooks = match HookEngine::load(&working_dir.join(".agent")) {
        Ok(Some(engine)) => {
            eprintln!("loaded hooks from .agent/");
            let boxed: Box<dyn HookPort> = Box::new(engine);
            Some(Arc::new(tokio::sync::Mutex::new(boxed)))
        }
        Ok(None) => None,
        Err(e) => {
            eprintln!("failed to load hooks: {e}");
            std::process::exit(1);
        }
    };

    let mut tools: Vec<Box<dyn Tool>> = vec![
        Box::new(ReadFileTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(BashTool),
        Box::new(GlobTool),
        Box::new(GrepTool),
        Box::new(LsTool),
        Box::new(GitDiffTool),
        Box::new(GitLogTool),
        Box::new(GitStatusTool),
        Box::new(GitCommitTool),
        Box::new(WebFetchTool::new()),
    ];
    // Omitted entirely (not registered with a doomed-to-fail key) when unset,
    // so the LLM never sees a tool in its list that it can't actually use.
    if let Ok(key) = std::env::var("TAVILY_API_KEY") {
        tools.push(Box::new(WebSearchTool::new(key)));
    }

    match discover_skills(&working_dir.join(".agent")) {
        Ok(skills) => {
            if !skills.is_empty() {
                eprintln!("loaded {} skill(s) from .agent/skills/", skills.len());
                tools.push(Box::new(SkillTool::new(skills)));
            }
        }
        Err(e) => {
            eprintln!("failed to load skills: {e}");
            std::process::exit(1);
        }
    }

    match agent_tools_wasm::load_plugins(&working_dir.join(".agent")).await {
        Ok(mut plugins) => {
            if !plugins.is_empty() {
                eprintln!(
                    "loaded {} wasm plugin tool(s) from .agent/tools/",
                    plugins.len()
                );
            }
            tools.append(&mut plugins);
        }
        Err(e) => {
            eprintln!("failed to load wasm plugins: {e}");
            std::process::exit(1);
        }
    }

    AgentSession::new(provider, tools, hooks, SYSTEM_PROMPT, tool_ctx)
        .with_reporter(Arc::new(TerminalReporter::new()))
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("loop") => run_loop_mode(&args[2..]).await,
        Some(task) => run_one_shot(task).await,
        None => {
            eprintln!("{USAGE}");
            std::process::exit(1);
        }
    }
}

async fn run_one_shot(task: &str) {
    let mut session = build_session().await;
    if let Err(e) = session.run_turn(task).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run_loop_mode(args: &[String]) {
    let Some(file) = args.first() else {
        eprintln!("{USAGE}");
        std::process::exit(1);
    };
    let task_hint = args.get(1).map(String::as_str);

    let mut session = build_session().await;
    if let Err(e) = loop_mode::run(
        &mut session,
        Path::new(file),
        task_hint,
        loop_mode::LoopOptions::default(),
    )
    .await
    {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
