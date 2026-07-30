use std::time::Duration;

use crate::message::{ToolCall, Usage};
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
    /// A chunk of assistant text as the provider streams it in, fired zero
    /// or more times before the same text's single `on_assistant_text` call
    /// -- lets a live reporter print as the response is generated instead of
    /// only once it's complete. See `LlmProvider::complete_streaming`.
    async fn on_assistant_text_delta(&self, _delta: &str) {}
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
    /// Fired when text queued via `AgentSession::enable_steering` (typed
    /// while a turn was already running) gets spliced into the transcript.
    async fn on_steering_message(&self, _text: &str) {}
    /// Fired once `run_turn` has its final answer, with tokens summed across
    /// every provider round-trip the turn took (each round-trip resends the
    /// whole conversation, so this is the turn's real token cost, not just
    /// the last call's).
    async fn on_usage(&self, _usage: &Usage) {}
    /// Fired once the active provider is set or changed (initial setup, or
    /// `AgentSession::set_provider` mid-session), so a live display can show
    /// which provider is actually about to answer.
    async fn on_provider_changed(&self, _provider_id: &str, _model: &str) {}
}

/// Default reporter: observes nothing, prints nothing.
pub struct NoopReporter;

#[async_trait::async_trait]
impl Reporter for NoopReporter {}
