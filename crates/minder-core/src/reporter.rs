use std::time::Duration;

use crate::message::ToolCall;
use crate::tool::ToolExecOutcome;

/// Live progress callbacks fired while a turn runs, so a CLI/TUI can render
/// tool calls and their results as they happen instead of only seeing the
/// final assistant message once the whole tool-calling loop has finished.
///
/// All methods have no-op default bodies -- implementors only override the
/// events they care about. Fully optional: a session with no reporter set
/// behaves exactly as before (see `NoopReporter`).
#[async_trait::async_trait]
pub trait Reporter: Send + Sync {
    /// Fired just before the provider is asked to complete a turn.
    async fn on_turn_start(&self) {}
    /// Fired as soon as the provider responds, before other events fire.
    async fn on_turn_end(&self) {}
    /// Assistant text seen on any turn, including turns that also request a
    /// tool call (previously dropped silently -- see `AgentSession::run_turn`).
    async fn on_assistant_text(&self, _text: &str) {}
    /// Extended-thinking content seen on a turn (only fired when the
    /// provider actually returns a `Thinking` block, e.g. Anthropic with a
    /// thinking budget configured -- see `AnthropicProvider::with_thinking_budget`).
    /// Whether/how to display it is entirely up to the reporter.
    async fn on_thinking(&self, _text: &str) {}
    /// Fired just before a tool call is executed (post-hook-transform).
    async fn on_tool_call(&self, _call: &ToolCall) {}
    /// Fired just after a tool call finishes (post-hook-transform).
    async fn on_tool_result(&self, _call: &ToolCall, _outcome: &ToolExecOutcome) {}
    /// Fired before sleeping to retry a transient provider error.
    async fn on_retry(&self, _attempt: usize, _max_attempts: usize, _delay: Duration, _reason: &str) {}
}

/// Default reporter: observes nothing, prints nothing.
pub struct NoopReporter;

#[async_trait::async_trait]
impl Reporter for NoopReporter {}
