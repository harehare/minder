use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;

use crate::agent_registry::{AgentRegistry, AgentRunSnapshot, AgentRunStatus};

/// Poll interval for `agent_output`'s `wait_secs` -- short enough that a
/// caller waiting on a quick subagent doesn't feel it, long enough not to
/// spin the registry's mutex.
const POLL_INTERVAL: Duration = Duration::from_millis(200);

pub struct ListAgentsTool {
    registry: Arc<AgentRegistry>,
}

impl ListAgentsTool {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for ListAgentsTool {
    fn name(&self) -> &str {
        "list_agents"
    }

    fn description(&self) -> &str {
        "Lists subagents started in the background via `agent`'s `background: true` option, \
         newest first, with id/status/elapsed time. Use `agent_output` to fetch a finished \
         run's result, or `agent_stop` to cancel one still running."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, _arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let runs = self.registry.list();
        let content = if runs.is_empty() {
            "(no background subagents)".to_string()
        } else {
            runs.iter().map(format_run_line).collect::<Vec<_>>().join("\n")
        };
        ToolExecOutcome {
            content,
            is_error: false,
            metadata: serde_json::json!({ "agents": runs }),
        }
    }
}

fn format_run_line(run: &AgentRunSnapshot) -> String {
    format!(
        "{} [{}] {} ({}s) -- {}",
        run.id,
        status_label(run.status),
        run.name,
        run.elapsed_secs,
        truncate(&run.task, 80)
    )
}

fn status_label(status: AgentRunStatus) -> &'static str {
    match status {
        AgentRunStatus::Running => "running",
        AgentRunStatus::Completed => "completed",
        AgentRunStatus::Failed => "failed",
        AgentRunStatus::Cancelled => "cancelled",
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut truncated: String = s.chars().take(max_chars).collect();
    truncated.push('\u{2026}');
    truncated
}

#[derive(Deserialize)]
struct OutputArgs {
    id: String,
    /// How long to wait for a still-running agent to finish before
    /// returning a "still running" result -- omit to check once and return
    /// immediately either way.
    #[serde(default)]
    wait_secs: Option<u64>,
}

pub struct AgentOutputTool {
    registry: Arc<AgentRegistry>,
}

impl AgentOutputTool {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for AgentOutputTool {
    fn name(&self) -> &str {
        "agent_output"
    }

    fn description(&self) -> &str {
        "Fetches a background subagent's status and, once it's finished, its result. Pass \
         `wait_secs` to block until it finishes (or that many seconds pass) instead of checking \
         once -- useful right after starting a quick background task."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "Agent id returned by `agent` (background: true) or listed by `list_agents`" },
                "wait_secs": { "type": "integer", "description": "Wait up to this many seconds for completion instead of returning immediately" }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: OutputArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let deadline = args
            .wait_secs
            .map(|s| tokio::time::Instant::now() + Duration::from_secs(s));

        loop {
            let Some(run) = self.registry.get(&args.id) else {
                return error(format!("unknown agent id '{}'", args.id));
            };

            if run.status != AgentRunStatus::Running {
                return ToolExecOutcome {
                    content: run
                        .result
                        .clone()
                        .unwrap_or_else(|| status_label(run.status).to_string()),
                    is_error: run.status == AgentRunStatus::Failed,
                    metadata: run_metadata(&run),
                };
            }

            let Some(deadline) = deadline else {
                return ToolExecOutcome {
                    content: format!("still running ({}s elapsed)", run.elapsed_secs),
                    is_error: false,
                    metadata: run_metadata(&run),
                };
            };
            if tokio::time::Instant::now() >= deadline {
                return ToolExecOutcome {
                    content: format!("still running after waiting ({}s elapsed)", run.elapsed_secs),
                    is_error: false,
                    metadata: run_metadata(&run),
                };
            }

            tokio::select! {
                _ = tokio::time::sleep(POLL_INTERVAL) => {}
                _ = ctx.cancel.cancelled() => return error("agent_output wait cancelled".to_string()),
            }
        }
    }
}

fn run_metadata(run: &AgentRunSnapshot) -> serde_json::Value {
    serde_json::json!({ "id": run.id, "status": run.status, "elapsed_secs": run.elapsed_secs })
}

#[derive(Deserialize)]
struct StopArgs {
    id: String,
}

pub struct AgentStopTool {
    registry: Arc<AgentRegistry>,
}

impl AgentStopTool {
    pub fn new(registry: Arc<AgentRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for AgentStopTool {
    fn name(&self) -> &str {
        "agent_stop"
    }

    fn description(&self) -> &str {
        "Cancels a background subagent by id. Best-effort: it stops the run before its next \
         step rather than aborting work already in flight."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "Agent id returned by `agent` (background: true) or listed by `list_agents`" }
            },
            "required": ["id"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let args: StopArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        match self.registry.cancel_run(&args.id) {
            Ok(()) => ToolExecOutcome {
                content: format!("cancelled {}", args.id),
                is_error: false,
                metadata: serde_json::json!({ "id": args.id }),
            },
            Err(e) => error(e),
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
    use tokio_util::sync::CancellationToken;

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: CancellationToken::new(),
            mailbox: None,
        }
    }

    #[tokio::test]
    async fn list_agents_reports_no_agents_when_empty() {
        let tool = ListAgentsTool::new(Arc::new(AgentRegistry::new()));
        let outcome = tool.execute(serde_json::json!({}), &ctx()).await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "(no background subagents)");
        assert_eq!(outcome.metadata["agents"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn list_agents_shows_a_running_entry() {
        let registry = Arc::new(AgentRegistry::new());
        let id = registry.start("reviewer", "review the diff", CancellationToken::new());
        let tool = ListAgentsTool::new(registry);

        let outcome = tool.execute(serde_json::json!({}), &ctx()).await;
        assert!(outcome.content.contains(&id));
        assert!(outcome.content.contains("reviewer"));
        assert!(outcome.content.contains("running"));
    }

    #[tokio::test]
    async fn agent_output_returns_the_result_once_finished() {
        let registry = Arc::new(AgentRegistry::new());
        let id = registry.start("a", "task", CancellationToken::new());
        registry.finish(&id, "all done".to_string(), false);
        let tool = AgentOutputTool::new(registry);

        let outcome = tool.execute(serde_json::json!({"id": id}), &ctx()).await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "all done");
    }

    #[tokio::test]
    async fn agent_output_reports_failure_as_a_tool_error() {
        let registry = Arc::new(AgentRegistry::new());
        let id = registry.start("a", "task", CancellationToken::new());
        registry.finish(&id, "boom".to_string(), true);
        let tool = AgentOutputTool::new(registry);

        let outcome = tool.execute(serde_json::json!({"id": id}), &ctx()).await;
        assert!(outcome.is_error);
        assert_eq!(outcome.content, "boom");
    }

    #[tokio::test]
    async fn agent_output_with_no_wait_secs_reports_still_running_immediately() {
        let registry = Arc::new(AgentRegistry::new());
        let id = registry.start("a", "task", CancellationToken::new());
        let tool = AgentOutputTool::new(registry);

        let outcome = tool.execute(serde_json::json!({"id": id}), &ctx()).await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("still running"));
    }

    #[tokio::test]
    async fn agent_output_unknown_id_is_an_error() {
        let tool = AgentOutputTool::new(Arc::new(AgentRegistry::new()));
        let outcome = tool.execute(serde_json::json!({"id": "nope"}), &ctx()).await;
        assert!(outcome.is_error);
    }

    #[tokio::test(start_paused = true)]
    async fn agent_output_wait_secs_returns_as_soon_as_the_run_finishes() {
        let registry = Arc::new(AgentRegistry::new());
        let id = registry.start("a", "task", CancellationToken::new());
        let tool = AgentOutputTool::new(registry.clone());

        let waiting = tokio::spawn(async move {
            tool.execute(serde_json::json!({"id": id, "wait_secs": 30}), &ctx())
                .await
        });

        tokio::time::sleep(Duration::from_millis(500)).await;
        registry.finish(&registry.list()[0].id.clone(), "finished".to_string(), false);

        let outcome = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("did not return promptly after finishing")
            .unwrap();
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "finished");
    }

    #[tokio::test(start_paused = true)]
    async fn agent_output_wait_secs_times_out_if_still_running() {
        let registry = Arc::new(AgentRegistry::new());
        let id = registry.start("a", "task", CancellationToken::new());
        let tool = AgentOutputTool::new(registry);

        let outcome = tool
            .execute(serde_json::json!({"id": id, "wait_secs": 1}), &ctx())
            .await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("still running"));
    }

    #[tokio::test]
    async fn agent_stop_cancels_a_running_agent() {
        let registry = Arc::new(AgentRegistry::new());
        let id = registry.start("a", "task", CancellationToken::new());
        let tool = AgentStopTool::new(registry.clone());

        let outcome = tool.execute(serde_json::json!({"id": id}), &ctx()).await;
        assert!(!outcome.is_error);
        assert_eq!(
            registry.get(outcome.metadata["id"].as_str().unwrap()).unwrap().status,
            AgentRunStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn agent_stop_unknown_id_is_an_error() {
        let tool = AgentStopTool::new(Arc::new(AgentRegistry::new()));
        let outcome = tool.execute(serde_json::json!({"id": "nope"}), &ctx()).await;
        assert!(outcome.is_error);
    }
}
