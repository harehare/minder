use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Project-level defaults read from `.agent/config.toml`. Every field is
/// optional and fully optional as a file -- a missing `.agent/config.toml`
/// resolves to all-`None`, same as every other `.agent/` input.
///
/// Precedence when both are set: the matching env var (`MINDER_PROVIDER`,
/// `MINDER_MODEL`, `OLLAMA_BASE_URL`, `MINDER_THINKING_BUDGET`) always wins
/// over this file, so a one-off override never requires editing the project
/// config back and forth -- see `provider_select::select_provider`.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProjectConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub ollama_base_url: Option<String>,
    /// Anthropic-only: requests extended thinking with this token budget
    /// (unset by default -- no thinking requested, no cost/latency change).
    /// Whether the resulting `Thinking` blocks are actually shown is a
    /// separate, runtime-toggleable question -- see `/thinking` in the REPL.
    pub thinking_budget: Option<u32>,
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
            "provider = \"openai\"\nmodel = \"gpt-5.4\"\nollama_base_url = \"http://localhost:11434\"\nthinking_budget = 4000\n",
        )
        .unwrap();

        let cfg = load(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(cfg.provider.as_deref(), Some("openai"));
        assert_eq!(cfg.model.as_deref(), Some("gpt-5.4"));
        assert_eq!(cfg.ollama_base_url.as_deref(), Some("http://localhost:11434"));
        assert_eq!(cfg.thinking_budget, Some(4000));
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
