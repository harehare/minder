use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One `[[server]]` entry in `.agent/mcp.toml`: an MCP server to launch as a
/// child process talking the stdio transport.
#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Manifest {
    #[serde(default, rename = "server")]
    pub servers: Vec<ServerConfig>,
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("failed to read manifest {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse manifest {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

impl Manifest {
    pub async fn load(path: &Path) -> Result<Self, ManifestError> {
        let raw = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ManifestError::Read {
                path: path.to_path_buf(),
                source: e,
            })?;
        toml::from_str(&raw).map_err(|e| ManifestError::Parse {
            path: path.to_path_buf(),
            source: e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiple_servers() {
        let raw = r#"
            [[server]]
            name = "filesystem"
            command = "npx"
            args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

            [[server]]
            name = "github"
            command = "docker"
            args = ["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"]
            env = { GITHUB_PERSONAL_ACCESS_TOKEN = "secret" }
        "#;
        let manifest: Manifest = toml::from_str(raw).unwrap();
        assert_eq!(manifest.servers.len(), 2);
        assert_eq!(manifest.servers[0].name, "filesystem");
        assert_eq!(manifest.servers[0].command, "npx");
        assert_eq!(manifest.servers[1].env["GITHUB_PERSONAL_ACCESS_TOKEN"], "secret");
    }

    #[test]
    fn defaults_are_applied_when_absent() {
        let manifest: Manifest = toml::from_str("").unwrap();
        assert!(manifest.servers.is_empty());
    }

    #[tokio::test]
    async fn malformed_toml_is_an_error() {
        let dir = std::env::temp_dir().join(format!("minder-mcp-manifest-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.toml");
        std::fs::write(&path, "this is not valid = = toml").unwrap();
        assert!(matches!(
            Manifest::load(&path).await,
            Err(ManifestError::Parse { .. })
        ));
    }

    #[tokio::test]
    async fn missing_file_is_an_error() {
        let path = std::env::temp_dir().join(format!("minder-mcp-manifest-missing-{}", uuid::Uuid::new_v4()));
        assert!(matches!(Manifest::load(&path).await, Err(ManifestError::Read { .. })));
    }
}
