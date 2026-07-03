use std::sync::Arc;

use agent_core::{AgentSession, ContentBlock, HookPort, LlmProvider, Tool, ToolContext};
use agent_hooks::HookEngine;
use agent_providers::{AnthropicProvider, GeminiProvider, OllamaProvider, OpenAiProvider};
use agent_tools::{
    BashTool, EditFileTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool, GlobTool,
    GrepTool, LsTool, ReadFileTool, WebFetchTool, WebSearchTool, WriteFileTool,
};

const SYSTEM_PROMPT: &str = "You are a careful coding assistant. Use the available tools to inspect and run code before answering.";

/// Selects a provider via `MINDER_PROVIDER` (`anthropic` [default], `openai`,
/// `gemini`, `ollama`), with the model overridable via `MINDER_MODEL` and
/// each provider's own API key env var (`ANTHROPIC_API_KEY`,
/// `OPENAI_API_KEY`, `GEMINI_API_KEY`; Ollama needs no key, just a local
/// server -- see `OLLAMA_BASE_URL`).
fn select_provider() -> Box<dyn LlmProvider> {
    let provider = std::env::var("MINDER_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    match provider.as_str() {
        "anthropic" => {
            let key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
            let model = std::env::var("MINDER_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-5-20250929".to_string());
            Box::new(AnthropicProvider::new(key, model))
        }
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
            let model = std::env::var("MINDER_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
            Box::new(OpenAiProvider::new(key, model))
        }
        "gemini" => {
            let key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY");
            let model =
                std::env::var("MINDER_MODEL").unwrap_or_else(|_| "gemini-2.0-flash".to_string());
            Box::new(GeminiProvider::new(key, model))
        }
        "ollama" => {
            let model = std::env::var("MINDER_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
            let mut provider = OllamaProvider::new(model);
            if let Ok(base_url) = std::env::var("OLLAMA_BASE_URL") {
                provider = provider.with_base_url(base_url);
            }
            Box::new(provider)
        }
        other => panic!(
            "unknown MINDER_PROVIDER '{other}' (expected anthropic, openai, gemini, or ollama)"
        ),
    }
}

#[tokio::main]
async fn main() {
    let task = std::env::args().nth(1).expect("usage: minder \"<task>\"");
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

    match agent_tools_wasm::load_plugins(&working_dir.join(".agent")).await {
        Ok(mut plugins) => {
            if !plugins.is_empty() {
                eprintln!("loaded {} wasm plugin tool(s) from .agent/tools/", plugins.len());
            }
            tools.append(&mut plugins);
        }
        Err(e) => {
            eprintln!("failed to load wasm plugins: {e}");
            std::process::exit(1);
        }
    }

    let mut session = AgentSession::new(provider, tools, hooks, SYSTEM_PROMPT, tool_ctx);

    match session.run_turn(&task).await {
        Ok(message) => {
            for block in &message.content {
                if let ContentBlock::Text(text) = block {
                    println!("{text}");
                }
            }
        }
        Err(e) => eprintln!("error: {e}"),
    }
}
