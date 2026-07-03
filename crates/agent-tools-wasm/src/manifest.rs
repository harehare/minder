use serde::Deserialize;
use std::path::{Path, PathBuf};

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_MEMORY_PAGES: u32 = 256; // 256 * 64KiB = 16 MiB
const DEFAULT_FUEL: u64 = 5_000_000;

/// Capability grants for one `.wasm` plugin, loaded from a required sidecar
/// `<name>.toml` next to `<name>.wasm`. The manifest is host-authored and
/// trusted at the same tier as `.agent/hooks/*.mq` -- it is the wasm binary
/// that is sandboxed, not the manifest, so `fs[].host_dir` is not restricted
/// to live under the project's working directory.
#[derive(Debug, Default, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub network: bool,
    #[serde(default)]
    pub fs: Vec<FsCapability>,
    #[serde(default)]
    pub limits: Limits,
}

#[derive(Debug, Deserialize)]
pub struct FsCapability {
    pub host_dir: String,
    pub guest_dir: String,
    #[serde(default)]
    pub read_only: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Limits {
    pub timeout_secs: u64,
    pub max_memory_pages: u32,
    pub fuel: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_memory_pages: DEFAULT_MAX_MEMORY_PAGES,
            fuel: DEFAULT_FUEL,
        }
    }
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
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let raw = std::fs::read_to_string(path).map_err(|e| ManifestError::Read {
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
    fn parses_full_manifest() {
        let raw = r#"
            network = true

            [[fs]]
            host_dir = "./data"
            guest_dir = "/data"
            read_only = true

            [limits]
            timeout_secs = 10
            max_memory_pages = 64
            fuel = 1000
        "#;
        let manifest: Manifest = toml::from_str(raw).unwrap();
        assert!(manifest.network);
        assert_eq!(manifest.fs.len(), 1);
        assert_eq!(manifest.fs[0].host_dir, "./data");
        assert!(manifest.fs[0].read_only);
        assert_eq!(manifest.limits.timeout_secs, 10);
        assert_eq!(manifest.limits.fuel, 1000);
    }

    #[test]
    fn defaults_are_applied_when_sections_are_omitted() {
        let manifest: Manifest = toml::from_str("").unwrap();
        assert!(!manifest.network);
        assert!(manifest.fs.is_empty());
        assert_eq!(manifest.limits.timeout_secs, DEFAULT_TIMEOUT_SECS);
        assert_eq!(manifest.limits.max_memory_pages, DEFAULT_MAX_MEMORY_PAGES);
        assert_eq!(manifest.limits.fuel, DEFAULT_FUEL);
    }

    #[test]
    fn malformed_toml_is_an_error() {
        let dir = std::env::temp_dir().join(format!("minder-manifest-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.toml");
        std::fs::write(&path, "this is not valid = = toml").unwrap();
        assert!(matches!(
            Manifest::load(&path),
            Err(ManifestError::Parse { .. })
        ));
    }

    #[test]
    fn missing_file_is_an_error() {
        let path = std::env::temp_dir().join(format!("minder-manifest-missing-{}", uuid::Uuid::new_v4()));
        assert!(matches!(Manifest::load(&path), Err(ManifestError::Read { .. })));
    }
}
