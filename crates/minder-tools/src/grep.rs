use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;

const MAX_MATCHES: usize = 200;

pub struct GrepTool;

#[derive(Deserialize)]
struct Args {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Searches file contents for a regex pattern, honoring .gitignore. Returns up to \
         200 matches as \"path:line: text\"."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regular expression to search for" },
                "path": { "type": "string", "description": "Directory or file to search, relative to the working directory (default: \".\")" },
                "case_insensitive": { "type": "boolean", "description": "Case-insensitive match (default false)" }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let search_root = ctx.working_dir.join(args.path.as_deref().unwrap_or("."));
        let working_dir = ctx.working_dir.clone();

        let result = tokio::task::spawn_blocking(move || {
            search(&args.pattern, args.case_insensitive, &search_root, &working_dir)
        })
        .await
        .unwrap_or_else(|e| Err(format!("search task panicked: {e}")));

        match result {
            Ok(matches) => {
                let count = matches.len();
                ToolExecOutcome {
                    content: if matches.is_empty() {
                        "no matches".to_string()
                    } else {
                        matches.join("\n")
                    },
                    is_error: false,
                    metadata: serde_json::json!({ "count": count }),
                }
            }
            Err(e) => error(e),
        }
    }
}

fn search(
    pattern: &str,
    case_insensitive: bool,
    search_root: &std::path::Path,
    working_dir: &std::path::Path,
) -> Result<Vec<String>, String> {
    let regex = regex::RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .map_err(|e| format!("invalid regex: {e}"))?;

    let mut matches = Vec::new();
    // `.gitignore` is honored regardless of whether `search_root` is inside
    // an actual git repository -- without this, ignore::WalkBuilder only
    // applies gitignore rules when a real `.git` directory is present.
    for entry in ignore::WalkBuilder::new(search_root).require_git(false).build() {
        if matches.len() >= MAX_MATCHES {
            break;
        }
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue; // binary or unreadable file
        };
        let relative = entry.path().strip_prefix(working_dir).unwrap_or(entry.path());
        for (i, line) in content.lines().enumerate() {
            if regex.is_match(line) {
                matches.push(format!("{}:{}: {}", relative.display(), i + 1, line));
                if matches.len() >= MAX_MATCHES {
                    break;
                }
            }
        }
    }
    Ok(matches)
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
        let dir = std::env::temp_dir().join(format!("minder-grep-test-{}", uuid::Uuid::new_v4()));
        for (name, content) in files {
            let path = dir.join(name);
            tokio::fs::create_dir_all(path.parent().unwrap()).await.unwrap();
            tokio::fs::write(path, content).await.unwrap();
        }
        ToolContext {
            working_dir: dir,
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn finds_matching_lines() {
        let ctx = ctx_with_files(&[("a.txt", "hello\nworld\nfoobar"), ("b.txt", "nothing here")]).await;
        let outcome = GrepTool.execute(serde_json::json!({"pattern": "wor"}), &ctx).await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.metadata["count"], 1);
        assert!(outcome.content.contains("a.txt:2: world"));
    }

    #[tokio::test]
    async fn case_insensitive_flag_works() {
        let ctx = ctx_with_files(&[("a.txt", "Hello World")]).await;
        let outcome = GrepTool
            .execute(serde_json::json!({"pattern": "hello", "case_insensitive": true}), &ctx)
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.metadata["count"], 1);
    }

    #[tokio::test]
    async fn respects_gitignore() {
        let ctx = ctx_with_files(&[
            (".gitignore", "ignored.txt\n"),
            ("ignored.txt", "secret findme"),
            ("kept.txt", "findme too"),
        ])
        .await;
        let outcome = GrepTool.execute(serde_json::json!({"pattern": "findme"}), &ctx).await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.metadata["count"], 1);
        assert!(outcome.content.contains("kept.txt"));
    }

    #[tokio::test]
    async fn invalid_regex_is_an_error() {
        let ctx = ctx_with_files(&[("a.txt", "x")]).await;
        let outcome = GrepTool.execute(serde_json::json!({"pattern": "("}), &ctx).await;
        assert!(outcome.is_error);
    }
}
