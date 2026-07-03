use agent_core::{Tool, ToolContext, ToolExecOutcome};
use async_trait::async_trait;
use serde::Deserialize;

pub struct GlobTool;

#[derive(Deserialize)]
struct Args {
    pattern: String,
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Finds files matching a glob pattern (e.g. \"**/*.rs\"), relative to the working directory."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern, e.g. \"src/**/*.rs\"" }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let full_pattern = ctx.working_dir.join(&args.pattern);
        let full_pattern = match full_pattern.to_str() {
            Some(s) => s.to_string(),
            None => return error("working directory path is not valid UTF-8".to_string()),
        };

        let paths = match glob::glob(&full_pattern) {
            Ok(p) => p,
            Err(e) => return error(format!("invalid glob pattern: {e}")),
        };

        let mut matches: Vec<String> = Vec::new();
        for entry in paths {
            match entry {
                Ok(path) => {
                    let relative = path.strip_prefix(&ctx.working_dir).unwrap_or(&path);
                    matches.push(relative.display().to_string());
                }
                Err(e) => return error(format!("glob walk error: {e}")),
            }
        }
        matches.sort();

        ToolExecOutcome {
            content: if matches.is_empty() {
                "no matches".to_string()
            } else {
                matches.join("\n")
            },
            is_error: false,
            metadata: serde_json::json!({ "count": matches.len() }),
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

    async fn ctx_with_files(files: &[&str]) -> ToolContext {
        let dir = std::env::temp_dir().join(format!("minder-glob-test-{}", uuid::Uuid::new_v4()));
        for f in files {
            let path = dir.join(f);
            tokio::fs::create_dir_all(path.parent().unwrap())
                .await
                .unwrap();
            tokio::fs::write(path, "x").await.unwrap();
        }
        ToolContext {
            working_dir: dir,
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn finds_matching_files_recursively() {
        let ctx = ctx_with_files(&[
            "src/main.rs",
            "src/lib.rs",
            "src/nested/util.rs",
            "README.md",
        ])
        .await;
        let outcome = GlobTool
            .execute(serde_json::json!({"pattern": "src/**/*.rs"}), &ctx)
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.metadata["count"], 3);
        assert!(outcome.content.contains("main.rs"));
        assert!(!outcome.content.contains("README"));
    }

    #[tokio::test]
    async fn no_matches_reports_cleanly() {
        let ctx = ctx_with_files(&["README.md"]).await;
        let outcome = GlobTool
            .execute(serde_json::json!({"pattern": "**/*.rs"}), &ctx)
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "no matches");
    }
}
