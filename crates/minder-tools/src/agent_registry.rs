use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRunStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Read-only view of one tracked run, safe to hand out of the registry (no
/// `CancellationToken`, `Instant` already resolved to elapsed seconds).
#[derive(Debug, Clone, Serialize)]
pub struct AgentRunSnapshot {
    pub id: String,
    pub name: String,
    pub task: String,
    pub status: AgentRunStatus,
    pub elapsed_secs: u64,
    pub result: Option<String>,
}

struct AgentRun {
    name: String,
    task: String,
    status: AgentRunStatus,
    started_at: Instant,
    finished_at: Option<Instant>,
    result: Option<String>,
    cancel: CancellationToken,
}

impl AgentRun {
    fn snapshot(&self, id: &str) -> AgentRunSnapshot {
        let elapsed = self.finished_at.unwrap_or_else(Instant::now) - self.started_at;
        AgentRunSnapshot {
            id: id.to_string(),
            name: self.name.clone(),
            task: self.task.clone(),
            status: self.status,
            elapsed_secs: elapsed.as_secs(),
            result: self.result.clone(),
        }
    }
}

/// Tracks subagents started via `agent`'s `background: true` option so
/// `list_agents`/`agent_output`/`agent_stop` can inspect or cancel them after
/// the tool call that started them has already returned -- the same
/// background-task-plus-registry shape recent coding-agent harnesses (e.g.
/// Claude Code's Task/TaskList/TaskOutput/TaskStop) expose for long-running
/// delegated work.
#[derive(Default)]
pub struct AgentRegistry {
    next_id: AtomicU64,
    runs: Mutex<HashMap<String, AgentRun>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new run as `Running`, storing `cancel` so `cancel_run`
    /// can stop it later -- typically a child of the parent turn's own
    /// cancellation token, so an interrupt of the whole session takes the
    /// background run down with it, while `cancel_run` alone doesn't affect
    /// any of the run's siblings.
    pub fn start(&self, name: &str, task: &str, cancel: CancellationToken) -> String {
        let id = format!("agent-{}", self.next_id.fetch_add(1, Ordering::Relaxed) + 1);
        let run = AgentRun {
            name: name.to_string(),
            task: task.to_string(),
            status: AgentRunStatus::Running,
            started_at: Instant::now(),
            finished_at: None,
            result: None,
            cancel,
        };
        self.runs.lock().unwrap().insert(id.clone(), run);
        id
    }

    /// Records the final outcome -- a no-op if `cancel_run` already marked
    /// this id `Cancelled`, so a run that finishes just after being
    /// cancelled doesn't overwrite that status with a stale result.
    pub fn finish(&self, id: &str, result: String, is_error: bool) {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.get_mut(id)
            && run.status == AgentRunStatus::Running
        {
            run.status = if is_error {
                AgentRunStatus::Failed
            } else {
                AgentRunStatus::Completed
            };
            run.result = Some(result);
            run.finished_at = Some(Instant::now());
        }
    }

    /// Best-effort cancel: flips the token the running session's tool calls
    /// check between steps, same as any other interrupt -- it can't abort
    /// work already in flight (e.g. a provider call), only stop the next
    /// step from starting.
    pub fn cancel_run(&self, id: &str) -> Result<(), String> {
        let mut runs = self.runs.lock().unwrap();
        let Some(run) = runs.get_mut(id) else {
            return Err(format!("unknown agent id '{id}'"));
        };
        if run.status != AgentRunStatus::Running {
            return Err(format!("agent '{id}' is already {:?}, not running", run.status));
        }
        run.cancel.cancel();
        run.status = AgentRunStatus::Cancelled;
        run.finished_at = Some(Instant::now());
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<AgentRunSnapshot> {
        self.runs.lock().unwrap().get(id).map(|r| r.snapshot(id))
    }

    /// Most recently started first, so `list_agents` reads newest-on-top.
    pub fn list(&self) -> Vec<AgentRunSnapshot> {
        let runs = self.runs.lock().unwrap();
        let mut snapshots: Vec<AgentRunSnapshot> = runs.iter().map(|(id, r)| r.snapshot(id)).collect();
        snapshots.sort_by(|a, b| b.id.cmp(&a.id));
        snapshots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_registers_a_running_entry_with_a_fresh_id() {
        let registry = AgentRegistry::new();
        let id = registry.start("reviewer", "review this", CancellationToken::new());
        let snapshot = registry.get(&id).unwrap();
        assert_eq!(snapshot.status, AgentRunStatus::Running);
        assert_eq!(snapshot.name, "reviewer");
        assert!(snapshot.result.is_none());
    }

    #[test]
    fn finish_marks_success_or_failure() {
        let registry = AgentRegistry::new();
        let ok_id = registry.start("a", "task a", CancellationToken::new());
        let fail_id = registry.start("b", "task b", CancellationToken::new());

        registry.finish(&ok_id, "done".to_string(), false);
        registry.finish(&fail_id, "boom".to_string(), true);

        assert_eq!(registry.get(&ok_id).unwrap().status, AgentRunStatus::Completed);
        assert_eq!(registry.get(&fail_id).unwrap().status, AgentRunStatus::Failed);
        assert_eq!(registry.get(&ok_id).unwrap().result, Some("done".to_string()));
    }

    #[test]
    fn cancel_run_flips_status_and_the_token() {
        let registry = AgentRegistry::new();
        let token = CancellationToken::new();
        let id = registry.start("a", "task", token.clone());

        registry.cancel_run(&id).unwrap();

        assert!(token.is_cancelled());
        assert_eq!(registry.get(&id).unwrap().status, AgentRunStatus::Cancelled);
    }

    #[test]
    fn cancel_run_on_an_unknown_id_is_an_error() {
        let registry = AgentRegistry::new();
        assert!(registry.cancel_run("nope").is_err());
    }

    #[test]
    fn cancel_run_on_an_already_finished_run_is_an_error() {
        let registry = AgentRegistry::new();
        let id = registry.start("a", "task", CancellationToken::new());
        registry.finish(&id, "done".to_string(), false);
        assert!(registry.cancel_run(&id).is_err());
    }

    #[test]
    fn a_finish_after_cancel_does_not_overwrite_the_cancelled_status() {
        let registry = AgentRegistry::new();
        let id = registry.start("a", "task", CancellationToken::new());
        registry.cancel_run(&id).unwrap();
        registry.finish(&id, "late result".to_string(), false);

        let snapshot = registry.get(&id).unwrap();
        assert_eq!(snapshot.status, AgentRunStatus::Cancelled);
        assert!(snapshot.result.is_none());
    }

    #[test]
    fn list_returns_newest_first() {
        let registry = AgentRegistry::new();
        let first = registry.start("a", "task a", CancellationToken::new());
        let second = registry.start("b", "task b", CancellationToken::new());

        let ids: Vec<String> = registry.list().into_iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![second, first]);
    }

    #[test]
    fn get_of_unknown_id_is_none() {
        let registry = AgentRegistry::new();
        assert!(registry.get("nope").is_none());
    }
}
