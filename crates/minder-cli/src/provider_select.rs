use std::sync::Arc;

use minder_core::LlmProvider;
use minder_providers::{AnthropicProvider, GeminiProvider, OllamaProvider, OpenAiProvider};

use crate::config::ProjectConfig;

/// Builds a provider by name (`anthropic`, `openai`, `gemini`, `ollama`),
/// with `model_override` winning over `cfg`'s/the built-in default model.
/// Returns `Err` instead of panicking on a missing API key or unknown name
/// -- this also runs lazily from a subagent's model-override tool argument
/// (see `AgentTool`), where a panic would kill the whole session.
pub fn build_provider(
    provider: &str,
    model_override: Option<String>,
    cfg: &ProjectConfig,
) -> Result<Arc<dyn LlmProvider>, String> {
    match provider {
        "anthropic" => {
            let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| "set ANTHROPIC_API_KEY".to_string())?;
            let model = model_override.unwrap_or_else(|| "claude-sonnet-5".to_string());
            let mut provider = AnthropicProvider::new(key, model);
            let thinking_budget = std::env::var("MINDER_THINKING_BUDGET")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .or(cfg.thinking_budget);
            if let Some(budget) = thinking_budget {
                provider = provider.with_thinking_budget(budget);
            }
            Ok(Arc::new(provider))
        }
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").map_err(|_| "set OPENAI_API_KEY".to_string())?;
            let model = model_override.unwrap_or_else(|| "gpt-5.4-mini".to_string());
            Ok(Arc::new(OpenAiProvider::new(key, model)))
        }
        "gemini" => {
            let key = std::env::var("GEMINI_API_KEY").map_err(|_| "set GEMINI_API_KEY".to_string())?;
            let model = model_override.unwrap_or_else(|| "gemini-3.5-flash".to_string());
            Ok(Arc::new(GeminiProvider::new(key, model)))
        }
        "ollama" => {
            let model = model_override.unwrap_or_else(|| "llama3.2".to_string());
            let mut provider = OllamaProvider::new(model);
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .ok()
                .or_else(|| cfg.ollama_base_url.clone());
            if let Some(base_url) = base_url {
                provider = provider.with_base_url(base_url);
            }
            Ok(Arc::new(provider))
        }
        other => Err(format!(
            "unknown provider '{other}' (expected anthropic, openai, gemini, or ollama)"
        )),
    }
}

/// Selects the main session's provider via `MINDER_PROVIDER` (`anthropic`
/// [default], `openai`, `gemini`, `ollama`) and `MINDER_MODEL`, falling back
/// to `cfg` (`.agent/config.toml`) then the built-in default. Returned as
/// `Arc` (not `Box`) so the same client can be reused by subagent sessions
/// without reconnecting -- see `AgentTool`. Exits the process on failure
/// (unknown provider, missing API key) -- see `build_provider`.
pub fn select_provider(cfg: &ProjectConfig) -> Arc<dyn LlmProvider> {
    let provider = std::env::var("MINDER_PROVIDER")
        .ok()
        .or_else(|| cfg.provider.clone())
        .unwrap_or_else(|| "anthropic".to_string());
    let model_override = std::env::var("MINDER_MODEL").ok().or_else(|| cfg.model.clone());

    build_provider(&provider, model_override, cfg).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    })
}
