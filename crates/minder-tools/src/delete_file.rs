use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Project-local trash; gitignored on first use, same as `.agent/sessions/`.
const TRASH_DIR: &str = ".agent/trash";

pub struct DeleteFileTool;

#[derive(Deserialize)]
struct Args {
    path: String,
}

#[async_trait]
impl Tool for DeleteFileTool {
    fn name(&self) -> &str {
        "delete_file"
    }

    fn description(&self) -> &str {
        "Moves a file to a project-local trash directory (.agent/trash/) instead of deleting it \
         outright, so it stays recoverable. Files only -- refuses directories."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path, absolute or relative to the working directory" }
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
        let metadata = match tokio::fs::metadata(&path).await {
            Ok(m) => m,
            Err(e) => return error(format!("{}: {e}", path.display())),
        };
        if metadata.is_dir() {
            return error(format!(
                "{} is a directory -- delete_file only removes individual files",
                path.display()
            ));
        }

        let trash_dir = match ensure_trash_dir(&ctx.working_dir).await {
            Ok(dir) => dir,
            Err(e) => return error(format!("failed to prepare trash directory: {e}")),
        };
        let trash_path = trash_dir.join(trash_name(&args.path));

        if let Err(e) = move_to_trash(&path, &trash_path).await {
            return error(format!("failed to move {} to trash: {e}", path.display()));
        }

        let shown_trash_path = trash_path.strip_prefix(&ctx.working_dir).unwrap_or(&trash_path);
        ToolExecOutcome {
            content: format!(
                "moved {} to {} (recoverable there, or via /undo this turn)",
                args.path,
                shown_trash_path.display()
            ),
            is_error: false,
            metadata: serde_json::json!({ "trashed_to": shown_trash_path.display().to_string() }),
        }
    }
}

/// Same pattern as `session_store::ensure_sessions_dir`.
async fn ensure_trash_dir(working_dir: &Path) -> std::io::Result<PathBuf> {
    let dir = working_dir.join(TRASH_DIR);
    tokio::fs::create_dir_all(&dir).await?;
    let gitignore = dir.join(".gitignore");
    if tokio::fs::metadata(&gitignore).await.is_err() {
        tokio::fs::write(&gitignore, "*\n!.gitignore\n").await?;
    }
    Ok(dir)
}

/// Flattened + nanosecond-prefixed so nested paths need no subdirectories and repeats never collide.
fn trash_name(relative_path: &str) -> String {
    let flattened = relative_path.replace(['/', '\\'], "_");
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{ts}_{flattened}")
}

/// Falls back to copy+remove only if `rename` fails (e.g. cross-filesystem).
async fn move_to_trash(src: &Path, dest: &Path) -> std::io::Result<()> {
    if tokio::fs::rename(src, dest).await.is_ok() {
        return Ok(());
    }
    tokio::fs::copy(src, dest).await?;
    tokio::fs::remove_file(src).await
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
            working_dir: std::env::temp_dir().join(format!("minder-delete-file-test-{}", uuid::Uuid::new_v4())),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn moves_a_file_to_the_trash_directory_instead_of_deleting_it() {
        let ctx = ctx();
        tokio::fs::create_dir_all(&ctx.working_dir).await.unwrap();
        tokio::fs::write(ctx.working_dir.join("a.txt"), "keep me")
            .await
            .unwrap();

        let outcome = DeleteFileTool.execute(serde_json::json!({"path": "a.txt"}), &ctx).await;

        assert!(!outcome.is_error);
        assert!(!ctx.working_dir.join("a.txt").exists());

        let trash_dir = ctx.working_dir.join(".agent/trash");
        let mut entries = tokio::fs::read_dir(&trash_dir).await.unwrap();
        let mut trashed_content = None;
        while let Some(entry) = entries.next_entry().await.unwrap() {
            if entry.file_name().to_string_lossy().ends_with("a.txt") {
                trashed_content = Some(tokio::fs::read_to_string(entry.path()).await.unwrap());
            }
        }
        assert_eq!(trashed_content.as_deref(), Some("keep me"));
    }

    #[tokio::test]
    async fn drops_a_gitignore_in_the_trash_directory_on_first_use() {
        let ctx = ctx();
        tokio::fs::create_dir_all(&ctx.working_dir).await.unwrap();
        tokio::fs::write(ctx.working_dir.join("a.txt"), "x").await.unwrap();

        DeleteFileTool.execute(serde_json::json!({"path": "a.txt"}), &ctx).await;

        let gitignore = tokio::fs::read_to_string(ctx.working_dir.join(".agent/trash/.gitignore"))
            .await
            .unwrap();
        assert_eq!(gitignore, "*\n!.gitignore\n");
    }

    #[tokio::test]
    async fn missing_file_is_an_error() {
        let ctx = ctx();
        tokio::fs::create_dir_all(&ctx.working_dir).await.unwrap();

        let outcome = DeleteFileTool
            .execute(serde_json::json!({"path": "missing.txt"}), &ctx)
            .await;
        assert!(outcome.is_error);
    }

    #[tokio::test]
    async fn refuses_to_move_a_directory() {
        let ctx = ctx();
        tokio::fs::create_dir_all(ctx.working_dir.join("subdir")).await.unwrap();

        let outcome = DeleteFileTool
            .execute(serde_json::json!({"path": "subdir"}), &ctx)
            .await;
        assert!(outcome.is_error);
        assert!(ctx.working_dir.join("subdir").exists());
    }
}
