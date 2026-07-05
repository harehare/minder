use crate::message::{Message, ToolCall};
use crate::tool::ToolExecOutcome;
use serde::Serialize;

/// Trait lives here (not in minder-hooks) so `AgentSession` can hold a
/// `dyn HookPort` without minder-core depending on minder-hooks (which itself
/// depends on minder-core for these same types). `minder_hooks::HookEngine`
/// is the concrete implementation.
#[async_trait::async_trait]
pub trait HookPort: Send + Sync {
    async fn before_agent_start(&mut self, system_prompt: &str) -> HookDecision<String>;
    async fn on_context(&mut self, messages: &[Message]) -> HookDecision<Vec<Message>>;
    async fn on_tool_call(&mut self, call: &ToolCall) -> ToolCallDecision;
    async fn on_tool_result(&mut self, result: &ToolResultInfo) -> HookDecision<String>;
    async fn before_compact(&mut self, messages: &[Message]) -> HookDecision<()>;

    /// Display-only hooks: how a tool call/result should be *printed*, not
    /// whether it runs. Default-bodied so only `HookEngine` needs to
    /// override them -- any other `HookPort` implementor keeps compiling
    /// unchanged. Unlike `on_tool_call`, these always fail open: a broken
    /// render script falls back to the harness's built-in formatting rather
    /// than affecting execution at all.
    async fn render_tool_call(&mut self, _call: &ToolCall) -> RenderDecision {
        RenderDecision::Default
    }
    async fn render_tool_result(&mut self, _call: &ToolCall, _outcome: &ToolExecOutcome) -> RenderDecision {
        RenderDecision::Default
    }
}

/// What to print for a tool call/result -- returned by `render_tool_call`/
/// `render_tool_result`. `style` is a plain name (`"green"`, `"red"`,
/// `"yellow"`, `"cyan"`, `"dim"`, `"bold"`, or anything else/`None` for no
/// styling) rather than a shared color type, matching how every other value
/// crossing the Rust/mq boundary is plain JSON -- the terminal reporter owns
/// the actual ANSI mapping.
#[derive(Debug, Clone)]
pub enum RenderDecision {
    /// No override -- use the harness's built-in formatting.
    Default,
    /// Replace the built-in formatting with this line(s).
    Text { value: String, style: Option<String> },
    /// Print nothing for this event.
    Hide,
}

#[derive(Debug, Clone)]
pub enum HookDecision<T> {
    Allow(T),
    Block(String),
}

/// `on_tool_call`'s decision is its own type rather than `HookDecision<ToolCall>`
/// because it has a third option `HookDecision` doesn't model: a hook can
/// supply the tool's result outright (`Override`) instead of only
/// allowing/rewriting the call or blocking it -- real middleware, not just
/// an observer. `AgentSession::execute_with_hooks` skips calling the real
/// tool on `Override`, but still runs the outcome through `on_tool_result`
/// same as a real one would.
#[derive(Debug, Clone)]
pub enum ToolCallDecision {
    Allow(ToolCall),
    Block(String),
    Override(ToolExecOutcome),
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
