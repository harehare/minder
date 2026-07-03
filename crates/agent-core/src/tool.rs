use std::path::PathBuf;

/// Trait lives here (not in `agent-tools`) so `AgentSession` can hold
/// `Vec<Box<dyn Tool>>` without agent-core depending on agent-tools.
/// Concrete tool implementations (read_file, bash, ...) live in agent-tools.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for the `arguments` object the LLM must produce.
    fn parameters_schema(&self) -> serde_json::Value;

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome;
}

pub struct ToolContext {
    pub working_dir: PathBuf,
    pub session_id: String,
    pub cancel: tokio_util::sync::CancellationToken,
}

pub struct ToolExecOutcome {
    pub content: String,
    pub is_error: bool,
    pub metadata: serde_json::Value,
}

pub fn spec(tool: &dyn Tool) -> crate::message::ToolSpec {
    crate::message::ToolSpec {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        parameters: tool.parameters_schema(),
    }
}
