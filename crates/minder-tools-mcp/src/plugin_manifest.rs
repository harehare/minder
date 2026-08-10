use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One entry in a plugin's `mcpServers` map (Agent Plugins `mcp.json`).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum PluginServerConfig {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
        #[serde(default)]
        cwd: Option<String>,
    },
    StreamableHttp {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
    Sse {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

#[derive(Debug, Default, Deserialize)]
pub struct PluginMcpManifest {
    #[serde(default, rename = "mcpServers")]
    pub servers: HashMap<String, PluginServerConfig>,
}

#[derive(Debug, thiserror::Error)]
pub enum PluginMcpManifestError {
    #[error("failed to read mcp manifest {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse mcp manifest {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

impl PluginMcpManifest {
    pub async fn load(path: &Path) -> Result<Self, PluginMcpManifestError> {
        let raw = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| PluginMcpManifestError::Read {
                path: path.to_path_buf(),
                source: e,
            })?;
        serde_json::from_str(&raw).map_err(|e| PluginMcpManifestError::Parse {
            path: path.to_path_buf(),
            source: e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_three_transport_types() {
        let raw = r#"{
            "mcpServers": {
                "filesystem": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                },
                "remote": {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer secret"}
                },
                "legacy": {
                    "type": "sse",
                    "url": "https://example.com/sse"
                }
            }
        }"#;
        let manifest: PluginMcpManifest = serde_json::from_str(raw).unwrap();
        assert_eq!(manifest.servers.len(), 3);
        assert!(matches!(
            manifest.servers["filesystem"],
            PluginServerConfig::Stdio { .. }
        ));
        assert!(matches!(
            manifest.servers["remote"],
            PluginServerConfig::StreamableHttp { .. }
        ));
        assert!(matches!(manifest.servers["legacy"], PluginServerConfig::Sse { .. }));
    }

    #[test]
    fn defaults_are_applied_when_absent() {
        let manifest: PluginMcpManifest = serde_json::from_str("{}").unwrap();
        assert!(manifest.servers.is_empty());
    }

    #[test]
    fn unknown_type_is_a_parse_error() {
        let raw = r#"{"mcpServers": {"bad": {"type": "carrier-pigeon", "url": "x"}}}"#;
        assert!(serde_json::from_str::<PluginMcpManifest>(raw).is_err());
    }

    #[tokio::test]
    async fn missing_file_is_an_error() {
        let path = std::env::temp_dir().join(format!("minder-plugin-mcp-manifest-missing-{}", uuid::Uuid::new_v4()));
        assert!(matches!(
            PluginMcpManifest::load(&path).await,
            Err(PluginMcpManifestError::Read { .. })
        ));
    }
}
