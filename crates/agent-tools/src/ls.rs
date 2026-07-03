use agent_core::{Tool, ToolContext, ToolExecOutcome};
use async_trait::async_trait;
use serde::Deserialize;
use std::path::Path;

const MAX_ENTRIES: usize = 500;
const DEFAULT_MAX_DEPTH: u32 = 3;

pub struct LsTool;

#[derive(Deserialize)]
struct Args {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    recursive: bool,
    #[serde(default)]
    max_depth: Option<u32>,
    #[serde(default)]
    show_hidden: bool,
}

struct Entry {
    relative: std::path::PathBuf,
    depth: usize,
    is_dir: bool,
}

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "Lists files and directories, honoring .gitignore. Non-recursive by default; set \
         \"recursive\" for a tree view. Returns up to 500 entries."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to list, relative to the working directory (default: \".\")" },
                "recursive": { "type": "boolean", "description": "List subdirectories recursively, tree-style (default false)" },
                "max_depth": { "type": "integer", "description": "Max recursion depth when recursive is true (default 3)" },
                "show_hidden": { "type": "boolean", "description": "Include dotfiles/dotdirs (default false)" }
            }
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let list_root = ctx.working_dir.join(args.path.as_deref().unwrap_or("."));
        let working_dir = ctx.working_dir.clone();
        let max_depth = if args.recursive {
            args.max_depth.unwrap_or(DEFAULT_MAX_DEPTH)
        } else {
            1
        };
        let show_hidden = args.show_hidden;

        let result = tokio::task::spawn_blocking(move || {
            list(&list_root, &working_dir, max_depth, show_hidden)
        })
        .await
        .unwrap_or_else(|e| Err(format!("list task panicked: {e}")));

        match result {
            Ok(entries) => {
                let count = entries.len();
                let truncated = count >= MAX_ENTRIES;
                let content = if entries.is_empty() {
                    "(empty)".to_string()
                } else {
                    entries
                        .iter()
                        .map(|e| {
                            let indent = "  ".repeat(e.depth);
                            let suffix = if e.is_dir { "/" } else { "" };
                            format!("{indent}{}{suffix}", e.relative.display())
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                ToolExecOutcome {
                    content,
                    is_error: false,
                    metadata: serde_json::json!({ "count": count, "truncated": truncated }),
                }
            }
            Err(e) => error(e),
        }
    }
}

fn list(
    root: &Path,
    working_dir: &Path,
    max_depth: u32,
    show_hidden: bool,
) -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    // `.gitignore` is honored regardless of whether `root` is inside an
    // actual git repository -- same rationale as grep.rs's WalkBuilder use.
    let walker = ignore::WalkBuilder::new(root)
        .max_depth(Some(max_depth as usize))
        .hidden(!show_hidden)
        .require_git(false)
        .build();

    let mut walked = Vec::new();
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.path() == root {
            continue; // don't list the root itself
        }
        let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
        let relative = entry
            .path()
            .strip_prefix(working_dir)
            .unwrap_or(entry.path())
            .to_path_buf();
        let depth = entry.depth().saturating_sub(1);
        walked.push(Entry {
            relative,
            depth,
            is_dir,
        });
        if walked.len() >= MAX_ENTRIES {
            break;
        }
    }

    walked.sort_by(|a, b| match (a.is_dir, b.is_dir) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.relative.cmp(&b.relative),
    });
    entries.extend(walked);
    Ok(entries)
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

    async fn ctx_with_files(files: &[(&str, &str)]) -> ToolContext {
        let dir = std::env::temp_dir().join(format!("minder-ls-test-{}", uuid::Uuid::new_v4()));
        for (name, content) in files {
            let path = dir.join(name);
            tokio::fs::create_dir_all(path.parent().unwrap())
                .await
                .unwrap();
            tokio::fs::write(path, content).await.unwrap();
        }
        ToolContext {
            working_dir: dir,
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn non_recursive_lists_only_top_level() {
        let ctx = ctx_with_files(&[("a.txt", ""), ("sub/b.txt", "")]).await;
        let outcome = LsTool.execute(serde_json::json!({}), &ctx).await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("a.txt"));
        assert!(outcome.content.contains("sub/"));
        assert!(!outcome.content.contains("b.txt"));
    }

    #[tokio::test]
    async fn recursive_respects_max_depth() {
        let ctx = ctx_with_files(&[("a/b/c/d.txt", "")]).await;
        let outcome = LsTool
            .execute(
                serde_json::json!({"recursive": true, "max_depth": 2}),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("a/"));
        assert!(outcome.content.contains("a/b/"));
        assert!(!outcome.content.contains("d.txt"));
    }

    #[tokio::test]
    async fn hidden_entries_excluded_by_default() {
        let ctx = ctx_with_files(&[(".secret", ""), ("visible.txt", "")]).await;
        let outcome = LsTool.execute(serde_json::json!({}), &ctx).await;
        assert!(!outcome.is_error);
        assert!(!outcome.content.contains(".secret"));
        assert!(outcome.content.contains("visible.txt"));
    }

    #[tokio::test]
    async fn show_hidden_includes_dotfiles() {
        let ctx = ctx_with_files(&[(".secret", ""), ("visible.txt", "")]).await;
        let outcome = LsTool
            .execute(serde_json::json!({"show_hidden": true}), &ctx)
            .await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains(".secret"));
    }

    #[tokio::test]
    async fn respects_gitignore() {
        let ctx = ctx_with_files(&[
            (".gitignore", "ignored.txt\n"),
            ("ignored.txt", ""),
            ("kept.txt", ""),
        ])
        .await;
        let outcome = LsTool.execute(serde_json::json!({}), &ctx).await;
        assert!(!outcome.is_error);
        assert!(!outcome.content.contains("ignored.txt"));
        assert!(outcome.content.contains("kept.txt"));
    }
}
