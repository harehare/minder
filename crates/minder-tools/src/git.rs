use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::time::Duration;

const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Shared spawn/timeout/cancel plumbing for the git_* tools. Args are passed
/// directly to `Command::args`, never through a shell, so callers can't
/// smuggle in shell metacharacters the way a generic `bash "git ..."` call
/// could -- a commit message like `"; rm -rf /tmp"` is just a literal argv
/// entry here.
async fn run_git(args: &[&str], ctx: &ToolContext, timeout: Duration) -> ToolExecOutcome {
    let child = match tokio::process::Command::new("git")
        .args(args)
        .current_dir(&ctx.working_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return error(format!("failed to spawn git: {e}")),
    };
    let child_id = child.id();

    tokio::select! {
        result = tokio::time::timeout(timeout, child.wait_with_output()) => {
            match result {
                Ok(Ok(output)) => {
                    let mut content = String::from_utf8_lossy(&output.stdout).into_owned();
                    content.push_str(&String::from_utf8_lossy(&output.stderr));
                    ToolExecOutcome {
                        content,
                        is_error: !output.status.success(),
                        metadata: serde_json::json!({ "exit_code": output.status.code() }),
                    }
                }
                Ok(Err(e)) => error(format!("git command failed: {e}")),
                Err(_) => error(format!("git command timed out after {}s", timeout.as_secs())),
            }
        }
        _ = ctx.cancel.cancelled() => {
            if let Some(pid) = child_id {
                let _ = tokio::process::Command::new("kill").arg("-9").arg(pid.to_string()).status().await;
            }
            error("git command cancelled".to_string())
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

fn timeout() -> Duration {
    Duration::from_secs(DEFAULT_TIMEOUT_SECS)
}

// ---- git_diff ----

pub struct GitDiffTool;

#[derive(Deserialize)]
struct DiffArgs {
    #[serde(rename = "ref")]
    #[serde(default)]
    git_ref: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    staged: bool,
}

#[async_trait]
impl Tool for GitDiffTool {
    fn name(&self) -> &str {
        "git_diff"
    }

    fn description(&self) -> &str {
        "Shows changes between commits, the working tree, or the index (git diff)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "ref": { "type": "string", "description": "Commit or ref to diff against (default: working tree vs index/HEAD)" },
                "path": { "type": "string", "description": "Limit the diff to this path" },
                "staged": { "type": "boolean", "description": "Show staged (index) changes instead of the working tree (default false)" }
            }
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: DiffArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let mut git_args = vec!["diff".to_string()];
        if args.staged {
            git_args.push("--staged".to_string());
        }
        if let Some(r) = &args.git_ref {
            git_args.push(r.clone());
        }
        if let Some(p) = &args.path {
            git_args.push("--".to_string());
            git_args.push(p.clone());
        }

        let refs: Vec<&str> = git_args.iter().map(String::as_str).collect();
        run_git(&refs, ctx, timeout()).await
    }
}

// ---- git_log ----

pub struct GitLogTool;

#[derive(Deserialize)]
struct LogArgs {
    #[serde(default)]
    max_count: Option<u32>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default = "default_true")]
    oneline: bool,
}

fn default_true() -> bool {
    true
}

#[async_trait]
impl Tool for GitLogTool {
    fn name(&self) -> &str {
        "git_log"
    }

    fn description(&self) -> &str {
        "Shows commit history (git log)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "max_count": { "type": "integer", "description": "Max number of commits to show (default 20)" },
                "path": { "type": "string", "description": "Limit history to this path" },
                "oneline": { "type": "boolean", "description": "Use compact one-line-per-commit format (default true)" }
            }
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: LogArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let max_count = args.max_count.unwrap_or(20).to_string();
        let mut git_args = vec!["log".to_string(), format!("-n{max_count}")];
        if args.oneline {
            git_args.push("--oneline".to_string());
        }
        if let Some(p) = &args.path {
            git_args.push("--".to_string());
            git_args.push(p.clone());
        }

        let refs: Vec<&str> = git_args.iter().map(String::as_str).collect();
        run_git(&refs, ctx, timeout()).await
    }
}

// ---- git_status ----

pub struct GitStatusTool;

#[derive(Deserialize)]
struct StatusArgs {
    #[serde(default)]
    path: Option<String>,
}

#[async_trait]
impl Tool for GitStatusTool {
    fn name(&self) -> &str {
        "git_status"
    }

    fn description(&self) -> &str {
        "Shows the working tree status in machine-parseable form (git status --porcelain=v1 -b)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Limit status to this path" }
            }
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: StatusArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let mut git_args = vec!["status".to_string(), "--porcelain=v1".to_string(), "-b".to_string()];
        if let Some(p) = &args.path {
            git_args.push("--".to_string());
            git_args.push(p.clone());
        }

        let refs: Vec<&str> = git_args.iter().map(String::as_str).collect();
        run_git(&refs, ctx, timeout()).await
    }
}

// ---- git_commit ----

pub struct GitCommitTool;

#[derive(Deserialize)]
struct CommitArgs {
    message: String,
    #[serde(default)]
    all: bool,
}

#[async_trait]
impl Tool for GitCommitTool {
    fn name(&self) -> &str {
        "git_commit"
    }

    fn description(&self) -> &str {
        "Records staged changes as a new commit (git commit -m)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": { "type": "string", "description": "Commit message" },
                "all": { "type": "boolean", "description": "Automatically stage all tracked, modified files first (git commit -a) (default false)" }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: CommitArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let mut git_args = vec!["commit".to_string()];
        if args.all {
            git_args.push("-a".to_string());
        }
        git_args.push("-m".to_string());
        git_args.push(args.message.clone());

        let refs: Vec<&str> = git_args.iter().map(String::as_str).collect();
        let outcome = run_git(&refs, ctx, timeout()).await;
        if outcome.is_error {
            return outcome;
        }

        let hash_outcome = run_git(&["rev-parse", "HEAD"], ctx, timeout()).await;
        let commit_hash = hash_outcome.content.trim().to_string();
        ToolExecOutcome {
            content: outcome.content,
            is_error: false,
            metadata: serde_json::json!({ "commit_hash": commit_hash }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn ctx_with_git_repo() -> ToolContext {
        let dir = std::env::temp_dir().join(format!("minder-git-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let ctx = ToolContext {
            working_dir: dir,
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        run_git(&["init"], &ctx, timeout()).await;
        run_git(&["config", "user.email", "test@example.com"], &ctx, timeout()).await;
        run_git(&["config", "user.name", "Test"], &ctx, timeout()).await;
        ctx
    }

    #[tokio::test]
    async fn status_reports_untracked_file() {
        let ctx = ctx_with_git_repo().await;
        tokio::fs::write(ctx.working_dir.join("a.txt"), "hi").await.unwrap();
        let outcome = GitStatusTool.execute(serde_json::json!({}), &ctx).await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("a.txt"));
    }

    #[tokio::test]
    async fn diff_shows_change_after_commit() {
        let ctx = ctx_with_git_repo().await;
        tokio::fs::write(ctx.working_dir.join("a.txt"), "hi").await.unwrap();
        run_git(&["add", "."], &ctx, timeout()).await;
        GitCommitTool
            .execute(serde_json::json!({"message": "initial"}), &ctx)
            .await;
        tokio::fs::write(ctx.working_dir.join("a.txt"), "hi there")
            .await
            .unwrap();
        let outcome = GitDiffTool.execute(serde_json::json!({}), &ctx).await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("hi there"));
    }

    #[tokio::test]
    async fn commit_all_creates_a_commit_and_returns_hash() {
        let ctx = ctx_with_git_repo().await;
        tokio::fs::write(ctx.working_dir.join("a.txt"), "hi").await.unwrap();
        run_git(&["add", "."], &ctx, timeout()).await;
        GitCommitTool
            .execute(serde_json::json!({"message": "initial"}), &ctx)
            .await;
        tokio::fs::write(ctx.working_dir.join("a.txt"), "changed")
            .await
            .unwrap();
        let outcome = GitCommitTool
            .execute(serde_json::json!({"message": "second", "all": true}), &ctx)
            .await;
        assert!(!outcome.is_error);
        let hash = outcome.metadata["commit_hash"].as_str().unwrap();
        assert_eq!(hash.len(), 40);
    }

    #[tokio::test]
    async fn log_returns_entries_for_each_commit() {
        let ctx = ctx_with_git_repo().await;
        for i in 0..2 {
            tokio::fs::write(ctx.working_dir.join("a.txt"), format!("v{i}"))
                .await
                .unwrap();
            run_git(&["add", "."], &ctx, timeout()).await;
            GitCommitTool
                .execute(serde_json::json!({"message": format!("commit {i}")}), &ctx)
                .await;
        }
        let outcome = GitLogTool.execute(serde_json::json!({}), &ctx).await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content.lines().count(), 2);
    }

    #[tokio::test]
    async fn commit_message_with_shell_metacharacters_is_stored_verbatim() {
        let ctx = ctx_with_git_repo().await;
        tokio::fs::write(ctx.working_dir.join("a.txt"), "hi").await.unwrap();
        run_git(&["add", "."], &ctx, timeout()).await;
        let malicious = "hello; rm -rf /tmp/pwned && echo pwned";
        let outcome = GitCommitTool
            .execute(serde_json::json!({"message": malicious}), &ctx)
            .await;
        assert!(!outcome.is_error);

        let log = run_git(&["log", "-1", "--format=%s"], &ctx, timeout()).await;
        assert_eq!(log.content.trim(), malicious);
        assert!(!std::path::Path::new("/tmp/pwned").exists());
    }
}
