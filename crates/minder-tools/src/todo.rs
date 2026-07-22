use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Deserialize)]
struct Args {
    todos: Vec<TodoItem>,
}

/// Lets the model track its own progress through a multi-step task as an
/// explicit checklist, the same role Claude Code's TodoWrite plays: every
/// call replaces the *entire* list rather than patching one item, so the
/// model always states the full current state instead of the tool having to
/// infer a diff.
///
/// Holds the current list in memory for the lifetime of this tool instance
/// (one per session, see `main.rs::build_session`) purely for display --
/// nothing here is persisted across process restarts or `/clear`.
pub struct TodoWriteTool {
    items: Mutex<Vec<TodoItem>>,
}

impl TodoWriteTool {
    pub fn new() -> Self {
        Self {
            items: Mutex::new(Vec::new()),
        }
    }

    /// Current list, for a caller (e.g. the REPL's `/todo`) to render on
    /// demand rather than only right after the model itself calls the tool.
    pub fn items(&self) -> Vec<TodoItem> {
        self.items.lock().unwrap().clone()
    }
}

impl Default for TodoWriteTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "todo_write"
    }

    fn description(&self) -> &str {
        "Replaces the current todo list with a full, updated list -- use it to plan and track \
         progress on a multi-step task. Always pass the *entire* list, not just the items that \
         changed. Keep at most one item `in_progress` at a time, and mark an item `completed` as \
         soon as it's actually done rather than preemptively."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "Short description of the step" },
                            "status": { "type": "string", "enum": ["pending", "in_progress", "completed"] }
                        },
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["todos"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => {
                return ToolExecOutcome {
                    content: format!("invalid arguments: {e}"),
                    is_error: true,
                    metadata: serde_json::Value::Null,
                };
            }
        };

        *self.items.lock().unwrap() = args.todos.clone();

        let counts = Counts::tally(&args.todos);
        ToolExecOutcome {
            content: format_checklist(&args.todos),
            is_error: false,
            metadata: serde_json::json!({
                "todos": args.todos,
                "pending": counts.pending,
                "in_progress": counts.in_progress,
                "completed": counts.completed,
            }),
        }
    }
}

struct Counts {
    pending: usize,
    in_progress: usize,
    completed: usize,
}

impl Counts {
    fn tally(todos: &[TodoItem]) -> Self {
        Self {
            pending: todos.iter().filter(|t| t.status == TodoStatus::Pending).count(),
            in_progress: todos.iter().filter(|t| t.status == TodoStatus::InProgress).count(),
            completed: todos.iter().filter(|t| t.status == TodoStatus::Completed).count(),
        }
    }
}

/// Plain-text checklist -- used as `content` (so a plain reporter, file log,
/// or `--output json` shows something readable instead of raw JSON) and
/// reused as-is by the REPL's `/todo` command.
pub fn format_checklist(todos: &[TodoItem]) -> String {
    if todos.is_empty() {
        return "(empty)".to_string();
    }
    todos
        .iter()
        .map(|t| {
            let mark = match t.status {
                TodoStatus::Pending => "☐",
                TodoStatus::InProgress => "◐",
                TodoStatus::Completed => "☑",
            };
            format!("{mark} {}", t.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn replaces_the_whole_list_and_reports_counts() {
        let tool = TodoWriteTool::new();
        let outcome = tool
            .execute(
                serde_json::json!({"todos": [
                    {"content": "write tests", "status": "completed"},
                    {"content": "implement feature", "status": "in_progress"},
                    {"content": "update docs", "status": "pending"},
                ]}),
                &ctx(),
            )
            .await;

        assert!(!outcome.is_error);
        assert_eq!(outcome.metadata["pending"], 1);
        assert_eq!(outcome.metadata["in_progress"], 1);
        assert_eq!(outcome.metadata["completed"], 1);
        assert_eq!(tool.items().len(), 3);
    }

    #[tokio::test]
    async fn a_later_call_fully_replaces_the_earlier_list() {
        let tool = TodoWriteTool::new();
        tool.execute(
            serde_json::json!({"todos": [{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]}),
            &ctx(),
        )
        .await;

        tool.execute(
            serde_json::json!({"todos": [{"content": "a", "status": "completed"}]}),
            &ctx(),
        )
        .await;

        let items = tool.items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "a");
        assert_eq!(items[0].status, TodoStatus::Completed);
    }

    #[tokio::test]
    async fn invalid_arguments_is_an_error_and_leaves_the_list_untouched() {
        let tool = TodoWriteTool::new();
        tool.execute(
            serde_json::json!({"todos": [{"content": "a", "status": "pending"}]}),
            &ctx(),
        )
        .await;

        let outcome = tool.execute(serde_json::json!({"todos": "not a list"}), &ctx()).await;
        assert!(outcome.is_error);
        assert_eq!(tool.items().len(), 1);
    }

    #[test]
    fn format_checklist_uses_a_distinct_mark_per_status() {
        let todos = vec![
            TodoItem {
                content: "a".to_string(),
                status: TodoStatus::Pending,
            },
            TodoItem {
                content: "b".to_string(),
                status: TodoStatus::InProgress,
            },
            TodoItem {
                content: "c".to_string(),
                status: TodoStatus::Completed,
            },
        ];
        assert_eq!(format_checklist(&todos), "☐ a\n◐ b\n☑ c");
    }

    #[test]
    fn format_checklist_of_an_empty_list_says_so() {
        assert_eq!(format_checklist(&[]), "(empty)");
    }
}
