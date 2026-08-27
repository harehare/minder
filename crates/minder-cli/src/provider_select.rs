use std::sync::Arc;

use minder_core::LlmProvider;
use minder_providers::{OllamaProvider, OpenAiCompatProvider};

use crate::config::ProjectConfig;

/// Builds a provider by name (`ollama` or `openai-compat`). Returns `Err` on
/// an unknown name instead of panicking, since this also runs lazily from a
/// subagent's model-override argument.
pub fn build_provider(
    provider: &str,
    model_override: Option<String>,
    cfg: &ProjectConfig,
) -> Result<Arc<dyn LlmProvider>, String> {
    let request_timeout_secs = std::env::var("MINDER_REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .or(cfg.request_timeout_secs);
    let num_ctx = std::env::var("MINDER_NUM_CTX")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .or(cfg.num_ctx);

    match provider {
        "ollama" => {
            let model = model_override.unwrap_or_else(|| "qwen2.5-coder:14b".to_string());
            let mut provider = OllamaProvider::new(model);
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .ok()
                .or_else(|| cfg.ollama_base_url.clone());
            if let Some(base_url) = base_url {
                provider = provider.with_base_url(base_url);
            }
            if let Some(secs) = request_timeout_secs {
                provider = provider.with_request_timeout_secs(secs);
            }
            if let Some(num_ctx) = num_ctx {
                provider = provider.with_num_ctx(num_ctx);
            }
            Ok(Arc::new(provider))
        }
        "openai-compat" => {
            let base_url = std::env::var("MINDER_OPENAI_COMPAT_BASE_URL")
                .ok()
                .or_else(|| cfg.openai_compat_base_url.clone())
                .ok_or_else(|| {
                    "set MINDER_OPENAI_COMPAT_BASE_URL (e.g. http://localhost:8080/v1 for llama-server, \
                     http://localhost:1234/v1 for LM Studio)"
                        .to_string()
                })?;
            let model = model_override
                .ok_or_else(|| "set MINDER_MODEL -- openai-compat has no built-in default".to_string())?;
            let mut provider = OpenAiCompatProvider::new(base_url, model);
            if let Ok(key) = std::env::var("MINDER_OPENAI_COMPAT_API_KEY") {
                provider = provider.with_api_key(key);
            }
            if let Some(secs) = request_timeout_secs {
                provider = provider.with_request_timeout_secs(secs);
            }
            Ok(Arc::new(provider))
        }
        other => Err(format!("unknown provider '{other}' (expected ollama or openai-compat)")),
    }
}

/// Selects the main session's provider via `MINDER_PROVIDER`/`MINDER_MODEL`,
/// falling back to `cfg` then the built-in default. Exits the process on
/// failure -- see `build_provider`.
pub fn select_provider(cfg: &ProjectConfig) -> Arc<dyn LlmProvider> {
    let provider = std::env::var("MINDER_PROVIDER")
        .ok()
        .or_else(|| cfg.provider.clone())
        .unwrap_or_else(|| "ollama".to_string());
    let model_override = std::env::var("MINDER_MODEL").ok().or_else(|| cfg.model.clone());

    build_provider(&provider, model_override, cfg).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_compat_without_a_base_url_is_a_clear_error() {
        let err = build_provider(
            "openai-compat",
            Some("some-model".to_string()),
            &ProjectConfig::default(),
        )
        .map(|_| ())
        .unwrap_err();
        assert!(err.contains("MINDER_OPENAI_COMPAT_BASE_URL"), "{err}");
    }

    #[test]
    fn openai_compat_without_a_model_is_a_clear_error() {
        let cfg = ProjectConfig {
            openai_compat_base_url: Some("http://localhost:8080/v1".to_string()),
            ..Default::default()
        };
        let err = build_provider("openai-compat", None, &cfg).map(|_| ()).unwrap_err();
        assert!(err.contains("MINDER_MODEL"), "{err}");
    }

    #[test]
    fn openai_compat_builds_when_base_url_and_model_are_set() {
        let cfg = ProjectConfig {
            openai_compat_base_url: Some("http://localhost:8080/v1".to_string()),
            ..Default::default()
        };
        let provider = build_provider("openai-compat", Some("some-model".to_string()), &cfg).unwrap();
        assert_eq!(provider.id(), "openai-compat");
        assert_eq!(provider.model(), "some-model");
    }

    #[test]
    fn unknown_provider_names_both_supported_providers() {
        let err = build_provider("not-a-provider", None, &ProjectConfig::default())
            .map(|_| ())
            .unwrap_err();
        assert!(err.contains("ollama") && err.contains("openai-compat"), "{err}");
    }
}
