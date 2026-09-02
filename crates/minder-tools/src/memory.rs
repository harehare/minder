use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub struct MemoryTool {
    memory_dir: PathBuf,
    description: String,
}

impl MemoryTool {
    pub fn new(agent_dir: &Path) -> Self {
        let memory_dir = agent_dir.join("memory");
        let description = match index(&memory_dir) {
            entries if entries.is_empty() => {
                "Reads and writes persistent notes under .agent/memory/ that survive across \
                 sessions. Actions: list, read, write, append. Call `list` for the current index."
                    .to_string()
            }
            entries => {
                let list = entries
                    .iter()
                    .map(|(name, first_line)| format!("- {name}: {first_line}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "Reads and writes persistent notes under .agent/memory/ that survive across \
                     sessions. Actions: list, read, write, append. Call `list` for the current \
                     index (this snapshot may be stale).\n\nKnown entries as of session start:\n{list}"
                )
            }
        };
        Self {
            memory_dir,
            description,
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum Args {
    List,
    Read { name: String },
    Write { name: String, content: String },
    Append { name: String, content: String },
}

#[async_trait]
impl Tool for MemoryTool {
    fn name(&self) -> &str {
        "memory"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["list", "read", "write", "append"] },
                "name": { "type": "string", "description": "Memory entry name, e.g. 'user-preferences' (becomes <name>.md)" },
                "content": { "type": "string", "description": "Full content for 'write', or text to add for 'append'" }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        match args {
            Args::List => {
                let entries = index(&self.memory_dir);
                if entries.is_empty() {
                    ok("no memory entries yet".to_string())
                } else {
                    let list = entries
                        .iter()
                        .map(|(name, first_line)| format!("- {name}: {first_line}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    ok(list)
                }
            }
            Args::Read { name } => {
                let path = match entry_path(&self.memory_dir, &name) {
                    Ok(p) => p,
                    Err(e) => return error(e),
                };
                match tokio::fs::read_to_string(&path).await {
                    Ok(content) => ok(content),
                    Err(_) => error(format!("no memory entry named '{name}'")),
                }
            }
            Args::Write { name, content } => {
                let path = match entry_path(&self.memory_dir, &name) {
                    Ok(p) => p,
                    Err(e) => return error(e),
                };
                if let Err(e) = tokio::fs::create_dir_all(&self.memory_dir).await {
                    return error(format!("failed to create {}: {e}", self.memory_dir.display()));
                }
                match tokio::fs::write(&path, &content).await {
                    Ok(()) => ok(format!("wrote memory entry '{name}' ({} bytes)", content.len())),
                    Err(e) => error(format!("failed to write {}: {e}", path.display())),
                }
            }
            Args::Append { name, content } => {
                let path = match entry_path(&self.memory_dir, &name) {
                    Ok(p) => p,
                    Err(e) => return error(e),
                };
                if let Err(e) = tokio::fs::create_dir_all(&self.memory_dir).await {
                    return error(format!("failed to create {}: {e}", self.memory_dir.display()));
                }
                let existing = tokio::fs::read_to_string(&path).await.unwrap_or_default();
                let updated = if existing.is_empty() || existing.ends_with('\n') {
                    format!("{existing}{content}\n")
                } else {
                    format!("{existing}\n{content}\n")
                };
                match tokio::fs::write(&path, &updated).await {
                    Ok(()) => ok(format!("appended to memory entry '{name}'")),
                    Err(e) => error(format!("failed to write {}: {e}", path.display())),
                }
            }
        }
    }
}

/// Rejects path separators so a memory entry can't escape `memory_dir`.
fn entry_path(memory_dir: &Path, name: &str) -> Result<PathBuf, String> {
    if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
        return Err(format!("invalid memory entry name '{name}'"));
    }
    Ok(memory_dir.join(format!("{name}.md")))
}

fn index(memory_dir: &Path) -> Vec<(String, String)> {
    let Ok(read_dir) = std::fs::read_dir(memory_dir) else {
        return Vec::new();
    };
    let mut entries: Vec<(String, String)> = read_dir
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("md"))
        .filter_map(|e| {
            let name = e.path().file_stem()?.to_str()?.to_string();
            let content = std::fs::read_to_string(e.path()).ok()?;
            let first_line = content.lines().find(|l| !l.trim().is_empty()).unwrap_or("").to_string();
            Some((name, first_line))
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

fn ok(content: String) -> ToolExecOutcome {
    ToolExecOutcome {
        content,
        is_error: false,
        metadata: serde_json::Value::Null,
    }
}

fn error(message: String) -> ToolExecOutcome {
    ToolExecOutcome {
        content: message,
        is_error: true,
        metadata: serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::AskChannel;

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
            mailbox: None,
            ask: AskChannel::unavailable(),
        }
    }

    fn scratch_agent_dir() -> PathBuf {
        std::env::temp_dir().join(format!("minder-memory-test-{}", uuid::Uuid::new_v4()))
    }

    #[tokio::test]
    async fn list_is_empty_before_anything_is_written() {
        let tool = MemoryTool::new(&scratch_agent_dir());
        let outcome = tool.execute(serde_json::json!({"action": "list"}), &ctx()).await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("no memory entries"));
    }

    #[tokio::test]
    async fn write_then_read_round_trips() {
        let agent_dir = scratch_agent_dir();
        let tool = MemoryTool::new(&agent_dir);

        let write = tool
            .execute(
                serde_json::json!({"action": "write", "name": "user-prefs", "content": "likes tabs"}),
                &ctx(),
            )
            .await;
        assert!(!write.is_error, "{}", write.content);

        let read = tool
            .execute(serde_json::json!({"action": "read", "name": "user-prefs"}), &ctx())
            .await;
        assert!(!read.is_error);
        assert_eq!(read.content, "likes tabs");

        std::fs::remove_dir_all(&agent_dir).ok();
    }

    #[tokio::test]
    async fn append_adds_a_new_line_to_an_existing_entry() {
        let agent_dir = scratch_agent_dir();
        let tool = MemoryTool::new(&agent_dir);

        tool.execute(
            serde_json::json!({"action": "write", "name": "notes", "content": "first"}),
            &ctx(),
        )
        .await;
        tool.execute(
            serde_json::json!({"action": "append", "name": "notes", "content": "second"}),
            &ctx(),
        )
        .await;

        let read = tool
            .execute(serde_json::json!({"action": "read", "name": "notes"}), &ctx())
            .await;
        assert_eq!(read.content, "first\nsecond\n");

        std::fs::remove_dir_all(&agent_dir).ok();
    }

    #[tokio::test]
    async fn read_of_unknown_entry_is_an_error() {
        let tool = MemoryTool::new(&scratch_agent_dir());
        let outcome = tool
            .execute(serde_json::json!({"action": "read", "name": "nope"}), &ctx())
            .await;
        assert!(outcome.is_error);
    }

    #[tokio::test]
    async fn name_with_path_separators_is_rejected() {
        let agent_dir = scratch_agent_dir();
        let tool = MemoryTool::new(&agent_dir);
        let outcome = tool
            .execute(
                serde_json::json!({"action": "write", "name": "../escape", "content": "x"}),
                &ctx(),
            )
            .await;
        assert!(outcome.is_error);
        assert!(!agent_dir.parent().unwrap().join("escape.md").exists());
    }

    #[tokio::test]
    async fn description_lists_entries_present_at_construction() {
        let agent_dir = scratch_agent_dir();
        std::fs::create_dir_all(agent_dir.join("memory")).unwrap();
        std::fs::write(agent_dir.join("memory/user-prefs.md"), "likes tabs\nmore detail").unwrap();

        let tool = MemoryTool::new(&agent_dir);
        assert!(tool.description().contains("user-prefs: likes tabs"));

        std::fs::remove_dir_all(&agent_dir).ok();
    }
}
