use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Project-level defaults read from `.agent/config.toml`. Every field is
/// optional -- a missing `.agent/config.toml` resolves to all-`None`.
///
/// Precedence when both are set: the matching env var (`MINDER_PROVIDER`,
/// `MINDER_MODEL`, `OLLAMA_BASE_URL`, `MINDER_REQUEST_TIMEOUT_SECS`,
/// `MINDER_NUM_CTX`, `MINDER_SHOW_STATUS_BAR`) always wins over this file --
/// see `provider_select::select_provider`.
#[derive(Debug, Default, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProjectConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub ollama_base_url: Option<String>,
    /// Base URL for `provider = "openai-compat"` (llama-server, LM Studio,
    /// vLLM, ...). No default -- must be set for that provider to build.
    pub openai_compat_base_url: Option<String>,
    /// Overrides the default request timeout (900s).
    pub request_timeout_secs: Option<u64>,
    /// Overrides Ollama's context window in tokens (minder's default: 8192).
    pub num_ctx: Option<u32>,
    /// Whether the spinner shows the active provider/model while a turn
    /// runs. Defaults to `true`; runtime-toggleable via `/status`.
    pub show_status_bar: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("failed to parse {0}: {1}")]
    Parse(PathBuf, toml::de::Error),
}

/// Loads `<agent_dir>/config.toml`. Returns the default (all-`None`) config
/// if the file doesn't exist.
pub fn load(agent_dir: &Path) -> Result<ProjectConfig, ConfigError> {
    let path = agent_dir.join("config.toml");
    if !path.is_file() {
        return Ok(ProjectConfig::default());
    }
    let raw = std::fs::read_to_string(&path).map_err(|e| ConfigError::Io(path.clone(), e))?;
    toml::from_str(&raw).map_err(|e| ConfigError::Parse(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir() -> PathBuf {
        std::env::temp_dir().join(format!("minder-config-test-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn missing_file_resolves_to_all_none() {
        let dir = scratch_dir();
        let cfg = load(&dir).unwrap();
        assert_eq!(cfg, ProjectConfig::default());
    }

    #[test]
    fn parses_every_field() {
        let dir = scratch_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.toml"),
            "provider = \"ollama\"\nmodel = \"llama3.2\"\nollama_base_url = \"http://localhost:11434\"\nopenai_compat_base_url = \"http://localhost:8080/v1\"\nrequest_timeout_secs = 1800\nnum_ctx = 16384\nshow_status_bar = false\n",
        )
        .unwrap();

        let cfg = load(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(cfg.provider.as_deref(), Some("ollama"));
        assert_eq!(cfg.model.as_deref(), Some("llama3.2"));
        assert_eq!(cfg.ollama_base_url.as_deref(), Some("http://localhost:11434"));
        assert_eq!(cfg.openai_compat_base_url.as_deref(), Some("http://localhost:8080/v1"));
        assert_eq!(cfg.request_timeout_secs, Some(1800));
        assert_eq!(cfg.num_ctx, Some(16384));
        assert_eq!(cfg.show_status_bar, Some(false));
    }

    #[test]
    fn unknown_field_is_a_parse_error() {
        let dir = scratch_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.toml"), "nonsense = true\n").unwrap();

        let err = load(&dir).unwrap_err();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(matches!(err, ConfigError::Parse(..)));
    }
}
