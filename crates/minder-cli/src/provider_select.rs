use minder_core::LlmProvider;
use minder_providers::{AnthropicProvider, GeminiProvider, OllamaProvider, OpenAiProvider};

/// Selects a provider via `MINDER_PROVIDER` (`anthropic` [default], `openai`,
/// `gemini`, `ollama`), with the model overridable via `MINDER_MODEL` and
/// each provider's own API key env var (`ANTHROPIC_API_KEY`,
/// `OPENAI_API_KEY`, `GEMINI_API_KEY`; Ollama needs no key, just a local
/// server -- see `OLLAMA_BASE_URL`).
pub fn select_provider() -> Box<dyn LlmProvider> {
    let provider = std::env::var("MINDER_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    match provider.as_str() {
        "anthropic" => {
            let key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
            let model = std::env::var("MINDER_MODEL").unwrap_or_else(|_| "claude-sonnet-5".to_string());
            Box::new(AnthropicProvider::new(key, model))
        }
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
            let model = std::env::var("MINDER_MODEL").unwrap_or_else(|_| "gpt-5.4-mini".to_string());
            Box::new(OpenAiProvider::new(key, model))
        }
        "gemini" => {
            let key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY");
            let model = std::env::var("MINDER_MODEL").unwrap_or_else(|_| "gemini-3.5-flash".to_string());
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
        other => panic!("unknown MINDER_PROVIDER '{other}' (expected anthropic, openai, gemini, or ollama)"),
    }
}
