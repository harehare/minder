use agent_core::{Tool, ToolContext, ToolExecOutcome};
use async_trait::async_trait;
use serde::Deserialize;

pub struct ReadFileTool;

#[derive(Deserialize)]
struct Args {
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads a file's contents, optionally restricted to a 1-indexed inclusive line range."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path, absolute or relative to the working directory" },
                "start_line": { "type": "integer", "description": "1-indexed first line to read (optional)" },
                "end_line": { "type": "integer", "description": "1-indexed last line to read, inclusive (optional)" }
            },
            "required": ["path"]
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

        let content = match (args.start_line, args.end_line) {
            (None, None) => content,
            (start, end) => {
                let start = start.unwrap_or(1).max(1);
                let lines: Vec<&str> = content.lines().collect();
                let end = end.unwrap_or(lines.len()).min(lines.len());
                if start > end {
                    String::new()
                } else {
                    lines[start - 1..end].join("\n")
                }
            }
        };

        ToolExecOutcome {
            content,
            is_error: false,
            metadata: serde_json::Value::Null,
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

    async fn ctx_with_file(name: &str, content: &str) -> ToolContext {
        let dir =
            std::env::temp_dir().join(format!("minder-read-file-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join(name), content).await.unwrap();
        ToolContext {
            working_dir: dir,
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn reads_whole_file() {
        let ctx = ctx_with_file("a.txt", "line1\nline2\nline3").await;
        let outcome = ReadFileTool
            .execute(serde_json::json!({"path": "a.txt"}), &ctx)
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "line1\nline2\nline3");
    }

    #[tokio::test]
    async fn reads_a_line_range() {
        let ctx = ctx_with_file("a.txt", "line1\nline2\nline3\nline4").await;
        let outcome = ReadFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "start_line": 2, "end_line": 3}),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "line2\nline3");
    }

    #[tokio::test]
    async fn missing_file_is_an_error() {
        let ctx = ctx_with_file("a.txt", "x").await;
        let outcome = ReadFileTool
            .execute(serde_json::json!({"path": "missing.txt"}), &ctx)
            .await;
        assert!(outcome.is_error);
    }
}
