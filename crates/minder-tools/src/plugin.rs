use serde::Deserialize;
use std::path::{Path, PathBuf};

/// The `author` object in a `plugin.json` manifest.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct PluginAuthor {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

/// A plugin's `plugin.json` manifest, per the Agent Plugins spec
/// (<https://agent-plugins.org>): the vendor-neutral format for packaging
/// Agent Skills and MCP servers into one distributable plugin. Only `name`
/// is required by minder beyond `$schema` -- every other field is metadata
/// clients may show but don't need to act on.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub author: Option<PluginAuthor>,
    #[serde(default)]
    pub homepage: Option<String>,
    #[serde(default)]
    pub repository: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Client-specific manifest data, keyed by reverse-domain namespace --
    /// opaque to minder, passed through unparsed.
    #[serde(default)]
    pub extensions: serde_json::Value,
}

/// A discovered plugin: its parsed `plugin.json` plus the directory it
/// lives in, so callers can point skill/MCP loaders at `root.join("skills")`
/// / `root.join("mcp.json")` the same way they already point at
/// `.agent/skills` / `.agent/mcp.toml` for project-local config.
#[derive(Debug, Clone)]
pub struct Plugin {
    pub manifest: PluginManifest,
    pub root: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum PluginLoadError {
    #[error("failed to read plugins directory {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("failed to read plugin manifest {0}: {1}")]
    ReadManifest(PathBuf, std::io::Error),
    #[error("failed to parse plugin manifest {0}: {1}")]
    ParseManifest(PathBuf, serde_json::Error),
    #[error(
        "plugin manifest {0} has an invalid 'name' ({1:?}) -- must be 1-64 lowercase \
         alphanumeric/'.'/'-' characters, not start/end with '.'/'-', and not contain '--' or '..'"
    )]
    InvalidName(PathBuf, String),
    #[error("duplicate plugin name '{name}' in {first} and {second} -- plugin names must be unique")]
    DuplicateName {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },
}

/// Discovers plugins from `agent_dir/plugins/*/plugin.json`, one directory
/// per plugin -- "every compatible client checks for `plugin.json` at the
/// plugin root" per the Agent Plugins spec. Returns an empty vec if the
/// plugins directory doesn't exist; plugins are fully optional, like skills,
/// subagents, and hooks. A directory under `plugins/` with no `plugin.json`
/// is silently skipped rather than an error, so partially-populated or
/// in-progress plugin checkouts don't break startup.
pub fn discover_plugins(agent_dir: &Path) -> Result<Vec<Plugin>, PluginLoadError> {
    let plugins_dir = agent_dir.join("plugins");
    if !plugins_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&plugins_dir)
        .map_err(|e| PluginLoadError::Io(plugins_dir.clone(), e))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir())
        .collect();
    entries.sort();

    let mut plugins: Vec<Plugin> = Vec::new();
    let mut sources: Vec<PathBuf> = Vec::new(); // parallel to `plugins`, for error messages
    for dir in entries {
        let manifest_path = dir.join("plugin.json");
        if !manifest_path.is_file() {
            continue;
        }

        let raw = std::fs::read_to_string(&manifest_path)
            .map_err(|e| PluginLoadError::ReadManifest(manifest_path.clone(), e))?;
        let manifest: PluginManifest =
            serde_json::from_str(&raw).map_err(|e| PluginLoadError::ParseManifest(manifest_path.clone(), e))?;

        if !is_valid_plugin_name(&manifest.name) {
            return Err(PluginLoadError::InvalidName(manifest_path, manifest.name));
        }

        if let Some(idx) = plugins.iter().position(|p| p.manifest.name == manifest.name) {
            return Err(PluginLoadError::DuplicateName {
                name: manifest.name,
                first: sources[idx].clone(),
                second: manifest_path,
            });
        }

        sources.push(manifest_path);
        plugins.push(Plugin { manifest, root: dir });
    }

    Ok(plugins)
}

/// Mirrors the Agent Plugins `plugin.schema.json` `name` pattern
/// (`^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$`, 1-64 chars) by
/// hand rather than via `regex`, since that crate doesn't support the
/// negative lookahead the pattern uses for rejecting `--`/`..`.
fn is_valid_plugin_name(name: &str) -> bool {
    if name.is_empty() || name.chars().count() > 64 {
        return false;
    }
    let is_body_char = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '-';
    if !name.chars().all(is_body_char) {
        return false;
    }
    let is_alnum = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit();
    let first = name.chars().next().expect("checked non-empty above");
    let last = name.chars().last().expect("checked non-empty above");
    is_alnum(first) && is_alnum(last) && !name.contains("--") && !name.contains("..")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir() -> PathBuf {
        std::env::temp_dir().join(format!("minder-plugin-test-{}", uuid::Uuid::new_v4()))
    }

    fn write_plugin(agent_dir: &Path, dir_name: &str, plugin_json: &str) {
        let plugin_dir = agent_dir.join("plugins").join(dir_name);
        std::fs::create_dir_all(&plugin_dir).unwrap();
        std::fs::write(plugin_dir.join("plugin.json"), plugin_json).unwrap();
    }

    #[test]
    fn discovers_no_plugins_when_plugins_dir_is_absent() {
        let agent_dir = scratch_dir();
        assert!(discover_plugins(&agent_dir).unwrap().is_empty());
    }

    #[test]
    fn discovers_and_parses_a_plugin() {
        let agent_dir = scratch_dir();
        write_plugin(
            &agent_dir,
            "my-plugin",
            r#"{
                "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
                "name": "my-plugin",
                "version": "1.0.0",
                "description": "An example plugin",
                "author": {"name": "Someone", "url": "https://example.com"},
                "keywords": ["example"]
            }"#,
        );

        let plugins = discover_plugins(&agent_dir).unwrap();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].manifest.name, "my-plugin");
        assert_eq!(plugins[0].manifest.version.as_deref(), Some("1.0.0"));
        assert_eq!(
            plugins[0].manifest.author.as_ref().unwrap().name.as_deref(),
            Some("Someone")
        );
        assert_eq!(plugins[0].root, agent_dir.join("plugins").join("my-plugin"));
    }

    #[test]
    fn minimal_manifest_only_needs_schema_and_name() {
        let agent_dir = scratch_dir();
        write_plugin(
            &agent_dir,
            "minimal",
            r#"{"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "minimal"}"#,
        );
        let plugins = discover_plugins(&agent_dir).unwrap();
        assert_eq!(plugins[0].manifest.name, "minimal");
        assert!(plugins[0].manifest.version.is_none());
    }

    #[test]
    fn directory_without_plugin_json_is_skipped() {
        let agent_dir = scratch_dir();
        std::fs::create_dir_all(agent_dir.join("plugins").join("not-a-plugin")).unwrap();
        assert!(discover_plugins(&agent_dir).unwrap().is_empty());
    }

    #[test]
    fn missing_name_is_a_parse_error() {
        let agent_dir = scratch_dir();
        write_plugin(
            &agent_dir,
            "bad",
            r#"{"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"}"#,
        );
        let err = discover_plugins(&agent_dir).unwrap_err();
        assert!(matches!(err, PluginLoadError::ParseManifest(_, _)));
    }

    #[test]
    fn invalid_name_is_an_error() {
        let agent_dir = scratch_dir();
        write_plugin(
            &agent_dir,
            "bad",
            r#"{"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "Not_Valid--Name"}"#,
        );
        let err = discover_plugins(&agent_dir).unwrap_err();
        assert!(matches!(err, PluginLoadError::InvalidName(_, name) if name == "Not_Valid--Name"));
    }

    #[test]
    fn duplicate_plugin_names_are_an_error() {
        let agent_dir = scratch_dir();
        write_plugin(
            &agent_dir,
            "a",
            r#"{"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "dup"}"#,
        );
        write_plugin(
            &agent_dir,
            "b",
            r#"{"$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "dup"}"#,
        );
        let err = discover_plugins(&agent_dir).unwrap_err();
        assert!(matches!(err, PluginLoadError::DuplicateName { name, .. } if name == "dup"));
    }

    #[test]
    fn valid_plugin_names() {
        for name in ["a", "my-plugin", "plugin.v1", "a1", "9-lives"] {
            assert!(is_valid_plugin_name(name), "expected {name} to be valid");
        }
    }

    #[test]
    fn invalid_plugin_names() {
        for name in [
            "",
            "-leading",
            "trailing-",
            ".leading",
            "trailing.",
            "has--double",
            "has..dots",
            "Has_Upper",
            "",
        ] {
            assert!(!is_valid_plugin_name(name), "expected {name:?} to be invalid");
        }
    }
}
