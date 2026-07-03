use crate::message::{Message, ToolCall};
use serde::Serialize;

/// Trait lives here (not in agent-hooks) so `AgentSession` can hold a
/// `dyn HookPort` without agent-core depending on agent-hooks (which itself
/// depends on agent-core for these same types). `agent_hooks::HookEngine`
/// is the concrete implementation.
#[async_trait::async_trait]
pub trait HookPort: Send + Sync {
    async fn before_agent_start(&mut self, system_prompt: &str) -> HookDecision<String>;
    async fn on_context(&mut self, messages: &[Message]) -> HookDecision<Vec<Message>>;
    async fn on_tool_call(&mut self, call: &ToolCall) -> HookDecision<ToolCall>;
    async fn on_tool_result(&mut self, result: &ToolResultInfo) -> HookDecision<String>;
    async fn before_compact(&mut self, messages: &[Message]) -> HookDecision<()>;
}

#[derive(Debug, Clone)]
pub enum HookDecision<T> {
    Allow(T),
    Block(String),
}

/// Passed to `on_tool_result` -- the tool call that produced this result plus
/// the outcome, so a hook can make decisions based on both (e.g. "only
/// redact bash output," not read/write output).
#[derive(Debug, Clone, Serialize)]
pub struct ToolResultInfo {
    pub tool_name: String,
    pub content: String,
    pub is_error: bool,
}
