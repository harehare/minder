use crate::diff::diff_files;
use agent_core::{Tool, ToolContext, ToolExecOutcome};
use async_trait::async_trait;
use serde::Deserialize;

pub struct EditFileTool;

#[derive(Deserialize)]
struct Args {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replaces `old_string` with `new_string` in a file. `old_string` must match exactly \
         once unless `replace_all` is true."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path, absolute or relative to the working directory" },
                "old_string": { "type": "string", "description": "Exact text to replace" },
                "new_string": { "type": "string", "description": "Replacement text" },
                "replace_all": { "type": "boolean", "description": "Replace every occurrence instead of requiring exactly one (default false)" }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let path = ctx.working_dir.join(&args.path);
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return error(format!("failed to read {}: {e}", path.display())),
        };

        let occurrences = content.matches(args.old_string.as_str()).count();
        if occurrences == 0 {
            return error(format!("old_string not found in {}", path.display()));
        }
        if occurrences > 1 && !args.replace_all {
            return error(format!(
                "old_string matches {occurrences} times in {} -- pass replace_all: true, or narrow old_string to a unique match",
                path.display()
            ));
        }

        let new_content = if args.replace_all {
            content.replace(&args.old_string, &args.new_string)
        } else {
            content.replacen(&args.old_string, &args.new_string, 1)
        };

        match tokio::fs::write(&path, &new_content).await {
            Ok(()) => {
                let diff = diff_files(&args.path, &content, &new_content);
                ToolExecOutcome {
                    content: format!("replaced {occurrences} occurrence(s) in {}", path.display()),
                    is_error: false,
                    metadata: serde_json::json!({
                        "occurrences": occurrences,
                        "diff": diff.unified,
                        "additions": diff.additions,
                        "deletions": diff.deletions,
                    }),
                }
            }
            Err(e) => error(format!("failed to write {}: {e}", path.display())),
        }
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

    async fn ctx_with_file(content: &str) -> ToolContext {
        let dir =
            std::env::temp_dir().join(format!("minder-edit-file-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("a.txt"), content).await.unwrap();
        ToolContext {
            working_dir: dir,
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn replaces_a_unique_match() {
        let ctx = ctx_with_file("hello world").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "world", "new_string": "there"}),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error);
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt"))
                .await
                .unwrap(),
            "hello there"
        );
    }

    #[tokio::test]
    async fn rejects_ambiguous_match_without_replace_all() {
        let ctx = ctx_with_file("foo foo foo").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "foo", "new_string": "bar"}),
                &ctx,
            )
            .await;
        assert!(outcome.is_error);
        assert!(outcome.content.contains("3 times"));
    }

    #[tokio::test]
    async fn replace_all_replaces_every_occurrence() {
        let ctx = ctx_with_file("foo foo foo").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "foo", "new_string": "bar", "replace_all": true}),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error);
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt"))
                .await
                .unwrap(),
            "bar bar bar"
        );
    }

    #[tokio::test]
    async fn missing_old_string_is_an_error() {
        let ctx = ctx_with_file("hello world").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "xyz", "new_string": "abc"}),
                &ctx,
            )
            .await;
        assert!(outcome.is_error);
    }
}
