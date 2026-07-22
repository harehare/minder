use std::sync::Arc;

use minder_core::LlmProvider;
use minder_providers::{AnthropicProvider, GeminiProvider, OllamaProvider, OpenAiProvider};

use crate::config::ProjectConfig;

/// Selects a provider via `MINDER_PROVIDER` (`anthropic` [default], `openai`,
/// `gemini`, `ollama`), with the model overridable via `MINDER_MODEL` and
/// each provider's own API key env var (`ANTHROPIC_API_KEY`,
/// `OPENAI_API_KEY`, `GEMINI_API_KEY`; Ollama needs no key, just a local
/// server -- see `OLLAMA_BASE_URL`). Returned as `Arc` (not `Box`) so the
/// same client can be reused by subagent sessions without reconnecting --
/// see `AgentTool`.
///
/// Precedence for `provider`/`model`/`ollama_base_url`: the env var wins if
/// set, otherwise `cfg` (loaded from `.agent/config.toml`, see
/// `crate::config`), otherwise the built-in default below.
pub fn select_provider(cfg: &ProjectConfig) -> Arc<dyn LlmProvider> {
    let provider = std::env::var("MINDER_PROVIDER")
        .ok()
        .or_else(|| cfg.provider.clone())
        .unwrap_or_else(|| "anthropic".to_string());
    let model_override = || std::env::var("MINDER_MODEL").ok().or_else(|| cfg.model.clone());

    match provider.as_str() {
        "anthropic" => {
            let key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
            let model = model_override().unwrap_or_else(|| "claude-sonnet-5".to_string());
            let mut provider = AnthropicProvider::new(key, model);
            let thinking_budget = std::env::var("MINDER_THINKING_BUDGET")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .or(cfg.thinking_budget);
            if let Some(budget) = thinking_budget {
                provider = provider.with_thinking_budget(budget);
            }
            Arc::new(provider)
        }
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
            let model = model_override().unwrap_or_else(|| "gpt-5.4-mini".to_string());
            Arc::new(OpenAiProvider::new(key, model))
        }
        "gemini" => {
            let key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY");
            let model = model_override().unwrap_or_else(|| "gemini-3.5-flash".to_string());
            Arc::new(GeminiProvider::new(key, model))
        }
        "ollama" => {
            let model = model_override().unwrap_or_else(|| "llama3.2".to_string());
            let mut provider = OllamaProvider::new(model);
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .ok()
                .or_else(|| cfg.ollama_base_url.clone());
            if let Some(base_url) = base_url {
                provider = provider.with_base_url(base_url);
            }
            Arc::new(provider)
        }
        other => panic!("unknown MINDER_PROVIDER '{other}' (expected anthropic, openai, gemini, or ollama)"),
    }
}
