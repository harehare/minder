use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Tracks the pre-edit content of every file a wrapped tool (see
/// `CheckpointedTool`) touches during one "generation" (in practice, one
/// turn -- see `Checkpoint::start_turn`), so `/undo` can restore them.
///
/// Deliberately not git-based: no stash, no commit, no worktree -- just the
/// plain file content from just before the first edit this generation, kept
/// in memory. That keeps it simple and safe (nothing here can surprise a
/// user's own git state) at the cost of only covering `write_file`/
/// `edit_file`; a `bash` command that edits or deletes a file bypasses it
/// entirely, since undoing arbitrary shell side effects in general would
/// need something much heavier (e.g. a real git snapshot) for a much
/// smaller benefit.
#[derive(Default)]
pub struct Checkpoint {
    /// Absolute path -> content from just before this generation's first
    /// touch, or `None` if the file didn't exist yet (so `undo` knows to
    /// delete it rather than write back empty content).
    snapshots: Mutex<HashMap<PathBuf, Option<String>>>,
}

impl Checkpoint {
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a fresh generation, discarding whatever the previous one
    /// tracked -- called once per turn (see `main.rs::run_turn_interruptible`
    /// callers), so `/undo` only ever reaches back to the turn just
    /// completed, never further.
    pub fn start_turn(&self) {
        self.snapshots.lock().unwrap().clear();
    }

    /// Records `path`'s current on-disk content, but only the first time
    /// it's touched this generation -- a second edit to the same file
    /// within one turn must not overwrite the snapshot with the first
    /// edit's *output*, or undoing would only reverse the last edit instead
    /// of the whole turn's cumulative effect on that file.
    async fn record(&self, path: &Path) {
        if self.snapshots.lock().unwrap().contains_key(path) {
            return;
        }
        let prior = tokio::fs::read_to_string(path).await.ok();
        self.snapshots
            .lock()
            .unwrap()
            .entry(path.to_path_buf())
            .or_insert(prior);
    }

    /// Restores every tracked file to its pre-generation content (deleting
    /// it if it didn't exist before), then clears the generation so a
    /// second `/undo` doesn't try to redo the same restore. Returns the
    /// paths actually restored, in no particular order.
    pub async fn undo(&self) -> Vec<PathBuf> {
        let snapshots = std::mem::take(&mut *self.snapshots.lock().unwrap());
        let mut restored = Vec::with_capacity(snapshots.len());
        for (path, prior) in snapshots {
            let result = match &prior {
                Some(content) => tokio::fs::write(&path, content).await,
                None => match tokio::fs::remove_file(&path).await {
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    other => other,
                },
            };
            if result.is_ok() {
                restored.push(path);
            }
        }
        restored
    }

    /// Whether there's anything for `/undo` to act on right now.
    pub fn is_empty(&self) -> bool {
        self.snapshots.lock().unwrap().is_empty()
    }
}

/// Wraps a file-editing tool (`write_file`/`edit_file`) so every call is
/// snapshotted by `checkpoint` before being delegated to `inner` -- the tool
/// itself is otherwise untouched (same name/description/schema/behavior).
pub struct CheckpointedTool {
    inner: Arc<dyn Tool>,
    checkpoint: Arc<Checkpoint>,
}

impl CheckpointedTool {
    pub fn new(inner: Arc<dyn Tool>, checkpoint: Arc<Checkpoint>) -> Self {
        Self { inner, checkpoint }
    }
}

#[async_trait]
impl Tool for CheckpointedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.inner.parameters_schema()
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        if let Some(path) = arguments.get("path").and_then(|v| v.as_str()) {
            self.checkpoint.record(&ctx.working_dir.join(path)).await;
        }
        self.inner.execute(arguments, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WriteFileTool;

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir().join(format!("minder-checkpoint-test-{}", uuid::Uuid::new_v4())),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn undo_restores_a_modified_file_to_its_prior_content() {
        let ctx = ctx();
        tokio::fs::create_dir_all(&ctx.working_dir).await.unwrap();
        tokio::fs::write(ctx.working_dir.join("a.txt"), "original")
            .await
            .unwrap();

        let checkpoint = Arc::new(Checkpoint::new());
        let tool = CheckpointedTool::new(Arc::new(WriteFileTool), checkpoint.clone());
        checkpoint.start_turn();
        tool.execute(serde_json::json!({"path": "a.txt", "content": "changed"}), &ctx)
            .await;
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt")).await.unwrap(),
            "changed"
        );

        let restored = checkpoint.undo().await;
        assert_eq!(restored, vec![ctx.working_dir.join("a.txt")]);
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt")).await.unwrap(),
            "original"
        );
    }

    #[tokio::test]
    async fn undo_deletes_a_file_that_did_not_exist_before_this_turn() {
        let ctx = ctx();
        tokio::fs::create_dir_all(&ctx.working_dir).await.unwrap();

        let checkpoint = Arc::new(Checkpoint::new());
        let tool = CheckpointedTool::new(Arc::new(WriteFileTool), checkpoint.clone());
        checkpoint.start_turn();
        tool.execute(serde_json::json!({"path": "new.txt", "content": "brand new"}), &ctx)
            .await;

        checkpoint.undo().await;
        assert!(!ctx.working_dir.join("new.txt").exists());
    }

    #[tokio::test]
    async fn a_second_edit_to_the_same_file_does_not_overwrite_the_original_snapshot() {
        let ctx = ctx();
        tokio::fs::create_dir_all(&ctx.working_dir).await.unwrap();
        tokio::fs::write(ctx.working_dir.join("a.txt"), "v0").await.unwrap();

        let checkpoint = Arc::new(Checkpoint::new());
        let tool = CheckpointedTool::new(Arc::new(WriteFileTool), checkpoint.clone());
        checkpoint.start_turn();
        tool.execute(serde_json::json!({"path": "a.txt", "content": "v1"}), &ctx)
            .await;
        tool.execute(serde_json::json!({"path": "a.txt", "content": "v2"}), &ctx)
            .await;

        checkpoint.undo().await;
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt")).await.unwrap(),
            "v0"
        );
    }

    #[tokio::test]
    async fn start_turn_discards_the_previous_generation() {
        let ctx = ctx();
        tokio::fs::create_dir_all(&ctx.working_dir).await.unwrap();
        tokio::fs::write(ctx.working_dir.join("a.txt"), "original")
            .await
            .unwrap();

        let checkpoint = Arc::new(Checkpoint::new());
        let tool = CheckpointedTool::new(Arc::new(WriteFileTool), checkpoint.clone());
        checkpoint.start_turn();
        tool.execute(serde_json::json!({"path": "a.txt", "content": "turn one"}), &ctx)
            .await;

        // A new turn starts before /undo is ever called for the first one.
        checkpoint.start_turn();
        assert!(checkpoint.is_empty());

        let restored = checkpoint.undo().await;
        assert!(restored.is_empty());
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt")).await.unwrap(),
            "turn one"
        );
    }

    #[tokio::test]
    async fn undo_with_nothing_tracked_is_a_noop() {
        let checkpoint = Checkpoint::new();
        assert!(checkpoint.is_empty());
        assert!(checkpoint.undo().await.is_empty());
    }
}
