use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use minder_core::{Reporter, ToolCall, ToolExecOutcome};
use serde_json::json;

/// Writes a JSON snapshot of "what is this agent doing right now" to `path`
/// on every state-changing event -- see `MINDER_STATUS_FILE` in `main.rs`.
///
/// Running/idle is a depth counter (`on_turn_start`/`on_tool_call` bump it,
/// `on_turn_end`/`on_tool_result` unbump it): `on_turn_end` always fires even
/// on a provider error, so this self-heals to idle without a dedicated
/// "turn fully done" hook. Gap: a tool interrupted mid-flight (Ctrl-C) may
/// never reach `on_tool_result`, leaving `running` stuck until the next turn.
pub struct StatusReporter {
    path: PathBuf,
    state: Mutex<StatusState>,
}

struct StatusState {
    depth: u32,
    turn_started_at: Option<u64>,
    current_action: Option<String>,
    provider_id: String,
    model: String,
}

impl StatusReporter {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            state: Mutex::new(StatusState {
                depth: 0,
                turn_started_at: None,
                current_action: None,
                provider_id: String::new(),
                model: String::new(),
            }),
        }
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Bumps `depth`, sets `current_action`, and writes the snapshot.
    fn enter(&self, action: &str) {
        let mut state = self.state.lock().unwrap();
        if state.depth == 0 {
            state.turn_started_at = Some(Self::now_secs());
        }
        state.depth += 1;
        state.current_action = Some(action.to_string());
        drop(state);
        self.write_snapshot();
    }

    /// Unbumps `depth`, clearing state back to idle at zero, and writes the snapshot.
    fn leave(&self) {
        let mut state = self.state.lock().unwrap();
        state.depth = state.depth.saturating_sub(1);
        if state.depth == 0 {
            state.turn_started_at = None;
            state.current_action = None;
        }
        drop(state);
        self.write_snapshot();
    }

    /// Writes via temp-file-then-rename so a concurrent reader never sees a half-written file.
    fn write_snapshot(&self) {
        let value = {
            let state = self.state.lock().unwrap();
            json!({
                "state": if state.depth > 0 { "running" } else { "idle" },
                "current_action": state.current_action,
                "turn_started_at": state.turn_started_at,
                "provider": state.provider_id,
                "model": state.model,
                "pid": std::process::id(),
                "updated_at": Self::now_secs(),
            })
        };
        let Ok(rendered) = serde_json::to_string_pretty(&value) else {
            return;
        };
        let tmp_path = PathBuf::from(format!("{}.tmp", self.path.display()));
        if std::fs::write(&tmp_path, rendered).is_ok() {
            let _ = std::fs::rename(&tmp_path, &self.path);
        }
    }
}

#[async_trait]
impl Reporter for StatusReporter {
    async fn on_turn_start(&self) {
        self.enter("Waiting on model");
    }

    async fn on_turn_end(&self) {
        self.leave();
    }

    async fn on_tool_call(&self, call: &ToolCall) {
        self.enter(&format!("Running {}", call.name));
    }

    async fn on_tool_result(&self, _call: &ToolCall, _outcome: &ToolExecOutcome) {
        self.leave();
    }

    async fn on_provider_changed(&self, provider_id: &str, model: &str) {
        {
            let mut state = self.state.lock().unwrap();
            state.provider_id = provider_id.to_string();
            state.model = model.to_string();
        }
        self.write_snapshot();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("minder-status-reporter-test-{}-{name}", uuid::Uuid::new_v4()))
    }

    fn read_snapshot(path: &PathBuf) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[tokio::test]
    async fn idle_before_any_event() {
        let path = scratch_path("idle.json");
        let reporter = StatusReporter::new(path.clone());
        reporter.write_snapshot();
        let snapshot = read_snapshot(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(snapshot["state"], "idle");
        assert!(snapshot["current_action"].is_null());
    }

    #[tokio::test]
    async fn turn_start_reports_running_and_turn_end_reports_idle() {
        let path = scratch_path("turn.json");
        let reporter = StatusReporter::new(path.clone());

        reporter.on_turn_start().await;
        let running = read_snapshot(&path);
        assert_eq!(running["state"], "running");
        assert_eq!(running["current_action"], "Waiting on model");
        assert!(!running["turn_started_at"].is_null());

        reporter.on_turn_end().await;
        let idle = read_snapshot(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(idle["state"], "idle");
        assert!(idle["turn_started_at"].is_null());
    }

    #[tokio::test]
    async fn a_provider_error_still_clears_running_via_the_paired_turn_end() {
        // Mirrors `AgentSession::run_turn_inner`: `on_turn_end` always fires
        // before the provider `Result` is unwrapped, even on error.
        let path = scratch_path("error.json");
        let reporter = StatusReporter::new(path.clone());

        reporter.on_turn_start().await;
        reporter.on_turn_end().await; // still called on the error path
        let snapshot = read_snapshot(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(snapshot["state"], "idle");
    }

    #[tokio::test]
    async fn a_tool_call_nested_inside_a_turn_stays_running_until_both_close() {
        let path = scratch_path("nested.json");
        let reporter = StatusReporter::new(path.clone());
        let call = ToolCall {
            id: "1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({}),
        };
        let outcome = ToolExecOutcome {
            content: String::new(),
            is_error: false,
            metadata: serde_json::Value::Null,
        };

        reporter.on_turn_start().await;
        reporter.on_turn_end().await;
        reporter.on_tool_call(&call).await;
        let mid = read_snapshot(&path);
        assert_eq!(mid["state"], "running");
        assert_eq!(mid["current_action"], "Running bash");

        reporter.on_tool_result(&call, &outcome).await;
        let after_tool = read_snapshot(&path);
        assert_eq!(after_tool["state"], "idle");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn provider_changed_updates_the_snapshot_without_touching_running_state() {
        let path = scratch_path("provider.json");
        let reporter = StatusReporter::new(path.clone());
        reporter.on_provider_changed("anthropic", "claude-sonnet-5").await;
        let snapshot = read_snapshot(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(snapshot["provider"], "anthropic");
        assert_eq!(snapshot["model"], "claude-sonnet-5");
        assert_eq!(snapshot["state"], "idle");
    }
}
