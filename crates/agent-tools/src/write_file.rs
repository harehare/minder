use agent_core::{Tool, ToolContext, ToolExecOutcome};
use async_trait::async_trait;
use serde::Deserialize;

pub struct WriteFileTool;

#[derive(Deserialize)]
struct Args {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Creates or overwrites a file with the given content, creating parent directories as needed."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path, absolute or relative to the working directory" },
                "content": { "type": "string", "description": "Full file content to write" }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let path = ctx.working_dir.join(&args.path);
        if let Some(parent) = path.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            return error(format!(
                "failed to create parent directories for {}: {e}",
                path.display()
            ));
        }

        match tokio::fs::write(&path, &args.content).await {
            Ok(()) => ToolExecOutcome {
                content: format!("wrote {} bytes to {}", args.content.len(), path.display()),
                is_error: false,
                metadata: serde_json::json!({ "bytes_written": args.content.len() }),
            },
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

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir()
                .join(format!("minder-write-file-test-{}", uuid::Uuid::new_v4())),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn writes_a_new_file_creating_parent_dirs() {
        let ctx = ctx();
        let outcome = WriteFileTool
            .execute(
                serde_json::json!({"path": "nested/dir/a.txt", "content": "hello"}),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error);
        let written = tokio::fs::read_to_string(ctx.working_dir.join("nested/dir/a.txt"))
            .await
            .unwrap();
        assert_eq!(written, "hello");
    }

    #[tokio::test]
    async fn overwrites_an_existing_file() {
        let ctx = ctx();
        tokio::fs::create_dir_all(&ctx.working_dir).await.unwrap();
        tokio::fs::write(ctx.working_dir.join("a.txt"), "old")
            .await
            .unwrap();

        WriteFileTool
            .execute(serde_json::json!({"path": "a.txt", "content": "new"}), &ctx)
            .await;
        let written = tokio::fs::read_to_string(ctx.working_dir.join("a.txt"))
            .await
            .unwrap();
        assert_eq!(written, "new");
    }
}
