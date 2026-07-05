use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;

use crate::git::{error, run_git, timeout};

// ---- worktree_add ----

/// Creates a new git worktree, optionally checking out an existing branch or
/// creating a new one -- lets the agent work on a second branch in a
/// separate directory without disturbing the current checkout, e.g. to run
/// tests against `main` while mid-edit on a feature branch.
pub struct WorktreeAddTool;

#[derive(Deserialize)]
struct AddArgs {
    path: String,
    #[serde(default)]
    branch: Option<String>,
    #[serde(default)]
    create_branch: bool,
    #[serde(default)]
    start_point: Option<String>,
}

#[async_trait]
impl Tool for WorktreeAddTool {
    fn name(&self) -> &str {
        "worktree_add"
    }

    fn description(&self) -> &str {
        "Creates a new git worktree at `path` (git worktree add) so a separate branch can be \
         worked on in its own directory alongside the current checkout."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to create the worktree in, relative to the repo (must not already exist)" },
                "branch": { "type": "string", "description": "Branch to check out in the new worktree. With create_branch, the name of the new branch to create; without it, an existing branch to check out (default: git creates a new branch named after `path`'s basename)" },
                "create_branch": { "type": "boolean", "description": "Create `branch` as a new branch instead of checking out an existing one (git worktree add -b) (default false)" },
                "start_point": { "type": "string", "description": "Commit/branch the new branch starts from -- only meaningful with create_branch (default HEAD)" }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: AddArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        if args.start_point.is_some() && !args.create_branch {
            return error("start_point only applies when create_branch is true".to_string());
        }

        let mut git_args = vec!["worktree".to_string(), "add".to_string()];
        match (&args.branch, args.create_branch) {
            (Some(branch), true) => {
                git_args.push("-b".to_string());
                git_args.push(branch.clone());
                git_args.push(args.path.clone());
                if let Some(start_point) = &args.start_point {
                    git_args.push(start_point.clone());
                }
            }
            (None, true) => return error("create_branch requires a branch name".to_string()),
            (Some(branch), false) => {
                git_args.push(args.path.clone());
                git_args.push(branch.clone());
            }
            (None, false) => git_args.push(args.path.clone()),
        }

        let refs: Vec<&str> = git_args.iter().map(String::as_str).collect();
        run_git(&refs, ctx, timeout()).await
    }
}

// ---- worktree_list ----

/// Lists every worktree linked to the current repo -- the main checkout plus
/// any created via `worktree_add`, each with its path, current HEAD, and
/// checked-out branch.
pub struct WorktreeListTool;

#[async_trait]
impl Tool for WorktreeListTool {
    fn name(&self) -> &str {
        "worktree_list"
    }

    fn description(&self) -> &str {
        "Lists every worktree linked to this repo, one per line: path, HEAD commit, and branch \
         (git worktree list)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, _arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        run_git(&["worktree", "list"], ctx, timeout()).await
    }
}

// ---- worktree_remove ----

/// Removes a worktree created by `worktree_add`, deleting its working
/// directory. Does not delete the branch it had checked out.
pub struct WorktreeRemoveTool;

#[derive(Deserialize)]
struct RemoveArgs {
    path: String,
    #[serde(default)]
    force: bool,
}

#[async_trait]
impl Tool for WorktreeRemoveTool {
    fn name(&self) -> &str {
        "worktree_remove"
    }

    fn description(&self) -> &str {
        "Removes a worktree and its working directory (git worktree remove). The branch it had \
         checked out is left intact -- only the worktree's directory is deleted."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path of the worktree to remove, as shown by worktree_list" },
                "force": { "type": "boolean", "description": "Remove even if the worktree has uncommitted changes (git worktree remove --force) (default false)" }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: RemoveArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let mut git_args = vec!["worktree".to_string(), "remove".to_string()];
        if args.force {
            git_args.push("--force".to_string());
        }
        git_args.push(args.path.clone());

        let refs: Vec<&str> = git_args.iter().map(String::as_str).collect();
        run_git(&refs, ctx, timeout()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn ctx_with_git_repo() -> ToolContext {
        let dir = std::env::temp_dir().join(format!("minder-worktree-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let ctx = ToolContext {
            working_dir: dir,
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        run_git(&["init"], &ctx, timeout()).await;
        run_git(&["config", "user.email", "test@example.com"], &ctx, timeout()).await;
        run_git(&["config", "user.name", "Test"], &ctx, timeout()).await;
        tokio::fs::write(ctx.working_dir.join("a.txt"), "hi").await.unwrap();
        run_git(&["add", "."], &ctx, timeout()).await;
        run_git(&["commit", "-m", "initial"], &ctx, timeout()).await;
        ctx
    }

    #[tokio::test]
    async fn adds_a_worktree_with_a_new_branch() {
        let ctx = ctx_with_git_repo().await;
        let outcome = WorktreeAddTool
            .execute(
                serde_json::json!({"path": "../wt1", "branch": "feature-x", "create_branch": true}),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error, "worktree_add failed: {}", outcome.content);

        let wt_path = ctx.working_dir.parent().unwrap().join("wt1");
        assert!(wt_path.join("a.txt").exists());

        let list = WorktreeListTool.execute(serde_json::json!({}), &ctx).await;
        assert!(!list.is_error);
        assert!(list.content.contains("feature-x"));
    }

    #[tokio::test]
    async fn create_branch_without_a_branch_name_is_an_error() {
        let ctx = ctx_with_git_repo().await;
        let outcome = WorktreeAddTool
            .execute(serde_json::json!({"path": "../wt2", "create_branch": true}), &ctx)
            .await;
        assert!(outcome.is_error);
    }

    #[tokio::test]
    async fn removes_a_worktree() {
        let ctx = ctx_with_git_repo().await;
        WorktreeAddTool
            .execute(
                serde_json::json!({"path": "../wt3", "branch": "feature-y", "create_branch": true}),
                &ctx,
            )
            .await;

        let outcome = WorktreeRemoveTool
            .execute(serde_json::json!({"path": "../wt3"}), &ctx)
            .await;
        assert!(!outcome.is_error, "worktree_remove failed: {}", outcome.content);

        let wt_path = ctx.working_dir.parent().unwrap().join("wt3");
        assert!(!wt_path.exists());
    }
}
