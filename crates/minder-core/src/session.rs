use std::sync::Arc;
use std::time::Duration;

use crate::hooks::{HookDecision, HookPort, ToolCallDecision, ToolResultInfo};
use crate::mailbox::Mailbox;
use crate::message::{
    ContentBlock, Message, ProviderResponse, Role, StopReason, ToolCall, ToolResult, ToolResultContent, ToolSpec, Usage,
};
use crate::provider::{LlmProvider, ProviderError};
use crate::reporter::{NoopReporter, Reporter};
use crate::tool::{Tool, ToolContext, ToolExecOutcome, spec};

const COMPACT_THRESHOLD: usize = 60;
const KEEP_RECENT: usize = 40;

/// Proactive compaction trigger based on the last response's real token usage,
/// not just message count (a few big tool results can blow the window early).
const TOKEN_COMPACT_THRESHOLD: u32 = 100_000;

/// Harder fallback used once the provider itself rejects a request as too big.
const EMERGENCY_KEEP_RECENT: usize = 20;

/// How many times a transient provider error (rate limit, 5xx, transport) is
/// retried before giving up -- unattended runs shouldn't die on one blip.
const MAX_TRANSIENT_RETRIES: usize = 5;
const BASE_BACKOFF: Duration = Duration::from_secs(2);
const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Calls to this tool run concurrently with each other (subagent
/// delegations share no state); every other tool stays sequential.
const CONCURRENT_TOOL_NAME: &str = "agent";

pub struct AgentSession {
    provider: Arc<dyn LlmProvider>,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
    reporter: Arc<dyn Reporter>,
    messages: Vec<Message>,
    system_prompt: String,
    tool_ctx: ToolContext,
    started: bool,
    /// Input tokens from the last response; drives proactive compaction.
    last_input_tokens: Option<u32>,
    /// Usage summed across every provider round-trip this session, fed to
    /// `on_budget`.
    total_usage: Usage,
    turn_count: usize,
    /// Set by `enable_steering` -- lets a caller (e.g. the REPL, while a
    /// user types over a running turn) queue text that gets spliced into
    /// the transcript at the next safe point instead of waiting for this
    /// turn to end. `None` means steering isn't wired up, the common case
    /// (subagents, tests, non-interactive runs).
    steering_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    /// Structured facts salvaged from messages before `truncate_to` drops
    /// them -- survives compaction unlike the raw transcript. See
    /// `DecisionLedger`.
    decision_ledger: DecisionLedger,
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),
    #[error("blocked by hook: {0}")]
    HookBlocked(String),
    /// The turn was cancelled mid-flight (e.g. Ctrl-C in the REPL) rather
    /// than failing on its own -- see `AgentSession::reset_cancel_token` and
    /// `AgentSession::discard_interrupted_turn`.
    #[error("interrupted")]
    Interrupted,
}

impl AgentSession {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        tools: Vec<Arc<dyn Tool>>,
        hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
        system_prompt: impl Into<String>,
        tool_ctx: ToolContext,
    ) -> Self {
        Self {
            provider,
            tools,
            hooks,
            reporter: Arc::new(NoopReporter),
            messages: Vec::new(),
            system_prompt: system_prompt.into(),
            tool_ctx,
            started: false,
            last_input_tokens: None,
            total_usage: Usage::default(),
            turn_count: 0,
            steering_rx: None,
            decision_ledger: DecisionLedger::default(),
        }
    }

    /// Sets the reporter used to observe live progress (assistant text, tool
    /// calls, tool results) as a turn runs. Defaults to `NoopReporter`.
    pub fn with_reporter(mut self, reporter: Arc<dyn Reporter>) -> Self {
        self.reporter = reporter;
        self
    }

    /// Opts this session into mid-turn steering: returns a sender a caller
    /// can use to queue a user message while a turn is running. Queued text
    /// isn't spliced in immediately (there's no safe mid-provider-call
    /// injection point) -- it's picked up the next time this turn reaches a
    /// tool-results message (or, failing that, the start of the *next*
    /// `run_turn` call) and appended there, see `drain_steering`.
    pub fn enable_steering(&mut self) -> tokio::sync::mpsc::UnboundedSender<String> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.steering_rx = Some(rx);
        tx
    }

    /// Runs one user turn to completion (looping on tool calls as needed)
    /// and returns the final assistant message. On error, rolls the
    /// transcript back to before this call -- otherwise a failed turn leaves
    /// a dangling user message that breaks role alternation on the next call.
    pub async fn run_turn(&mut self, user_input: &str) -> Result<Message, AgentError> {
        let pre_turn_len = self.messages.len();
        let result = self.run_turn_inner(user_input).await;
        if result.is_err() {
            self.messages.truncate(pre_turn_len);
        }
        result
    }

    async fn run_turn_inner(&mut self, user_input: &str) -> Result<Message, AgentError> {
        if !self.started {
            self.system_prompt = self.run_before_agent_start().await?;
            self.started = true;
        }

        self.messages.push(Message::user_text(user_input));
        let injected_at = self.messages.len() - 1;
        self.drain_steering(injected_at).await;
        self.turn_count += 1;

        let mut turn_usage = Usage::default();
        loop {
            self.maybe_compact().await?;

            let outgoing = self.run_context_hook().await?;
            let tool_specs: Vec<ToolSpec> = self.tools.iter().map(|t| spec(t.as_ref())).collect();
            self.reporter.on_turn_start().await;
            let mut result = self.complete_with_retries(&outgoing, &tool_specs).await;

            // Provider rejected the request as too large: compact harder and retry once.
            if let Err(err) = &result
                && is_context_length_error(err)
                && self.messages.len() > EMERGENCY_KEEP_RECENT
            {
                self.force_compact().await?;
                let retry_outgoing = self.run_context_hook().await?;
                result = self.complete_with_retries(&retry_outgoing, &tool_specs).await;
            }
            self.reporter.on_turn_end().await;
            let response = result?;
            self.last_input_tokens = Some(response.usage.input_tokens);
            turn_usage.input_tokens += response.usage.input_tokens;
            turn_usage.output_tokens += response.usage.output_tokens;
            self.total_usage.input_tokens += response.usage.input_tokens;
            self.total_usage.output_tokens += response.usage.output_tokens;
            self.run_budget_hook(turn_usage).await?;
            self.messages.push(response.message.clone());

            for block in &response.message.content {
                match block {
                    ContentBlock::Text(text) => self.reporter.on_assistant_text(text).await,
                    ContentBlock::Thinking { text, .. } => self.reporter.on_thinking(text).await,
                    _ => {}
                }
            }

            let tool_calls: Vec<ToolCall> = response.message.tool_calls().cloned().collect();
            if tool_calls.is_empty() || response.stop_reason != StopReason::ToolUse {
                // A text-only response never reaches the tool-results drain
                // point below, so without this, text typed while this turn
                // was running would sit unconsumed in `steering_rx` until
                // the *next* `run_turn` call silently tacked it onto that
                // unrelated message instead. Treat it as the next turn now.
                if self.drain_steering_into_new_turn().await {
                    continue;
                }
                self.reporter.on_usage(&turn_usage).await;
                return Ok(response.message);
            }

            let mut results: Vec<Option<ToolResult>> = vec![None; tool_calls.len()];
            let mut concurrent_indices = Vec::new();

            for (i, call) in tool_calls.iter().enumerate() {
                if call.name == CONCURRENT_TOOL_NAME {
                    concurrent_indices.push(i);
                    continue;
                }
                self.reporter.on_tool_call(call).await;
                let outcome = self.execute_with_hooks(call.clone(), &self.tool_ctx).await?;
                self.reporter.on_tool_result(call, &outcome).await;
                results[i] = Some(ToolResult {
                    tool_call_id: call.id.clone(),
                    content: ToolResultContent::Text(outcome.content),
                    is_error: outcome.is_error,
                });
            }

            if !concurrent_indices.is_empty() {
                for &i in &concurrent_indices {
                    self.reporter.on_tool_call(&tool_calls[i]).await;
                }
                // One shared mailbox per batch, so these siblings can coordinate via
                // send_message/check_messages -- see `ToolContext::mailbox`.
                let batch_ctx = ToolContext {
                    mailbox: Some(Mailbox::new()),
                    ..self.tool_ctx.clone()
                };
                // Shared reborrow so these futures can run concurrently.
                let session = &*self;
                let futures = concurrent_indices.iter().map(|&i| {
                    let call = tool_calls[i].clone();
                    let batch_ctx = &batch_ctx;
                    async move {
                        let outcome = session.execute_with_hooks(call.clone(), batch_ctx).await?;
                        session.reporter.on_tool_result(&call, &outcome).await;
                        Ok::<(usize, ToolResult), AgentError>((
                            i,
                            ToolResult {
                                tool_call_id: call.id,
                                content: ToolResultContent::Text(outcome.content),
                                is_error: outcome.is_error,
                            },
                        ))
                    }
                });
                for (i, result) in futures_util::future::try_join_all(futures).await? {
                    results[i] = Some(result);
                }
            }

            let results: Vec<ToolResult> = results
                .into_iter()
                .map(|r| r.expect("every tool_calls index is filled by one of the two loops above"))
                .collect();
            self.messages.push(Message::tool_results(results));
            let tool_results_at = self.messages.len() - 1;
            self.drain_steering(tool_results_at).await;
        }
    }

    /// Calls the provider, retrying transient failures (rate limit, 5xx,
    /// transport) with backoff instead of surfacing them immediately --
    /// an unattended run shouldn't die on one blip.
    async fn complete_with_retries(
        &self,
        messages: &[Message],
        tool_specs: &[ToolSpec],
    ) -> Result<ProviderResponse, ProviderError> {
        let system_prompt = match self.decision_ledger.render() {
            Some(ledger) => format!("{}\n\n{ledger}", self.system_prompt),
            None => self.system_prompt.clone(),
        };
        let mut attempt = 0usize;
        loop {
            let result = self
                .provider
                .complete_streaming(messages, tool_specs, Some(&system_prompt), self.reporter.as_ref())
                .await;
            match &result {
                Err(err) if is_transient_error(err) && attempt < MAX_TRANSIENT_RETRIES => {
                    let delay = backoff_delay(attempt, err);
                    self.reporter
                        .on_retry(attempt + 1, MAX_TRANSIENT_RETRIES, delay, &err.to_string())
                        .await;
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                _ => return result,
            }
        }
    }

    async fn run_before_agent_start(&mut self) -> Result<String, AgentError> {
        let Some(hooks) = &self.hooks else {
            return Ok(self.system_prompt.clone());
        };
        match hooks.lock().await.before_agent_start(&self.system_prompt).await {
            HookDecision::Allow(prompt) => Ok(prompt),
            HookDecision::Block(reason) => Err(AgentError::HookBlocked(reason)),
        }
    }

    async fn run_context_hook(&self) -> Result<Vec<Message>, AgentError> {
        let Some(hooks) = &self.hooks else {
            return Ok(self.messages.clone());
        };
        match hooks.lock().await.on_context(&self.messages).await {
            HookDecision::Allow(msgs) => Ok(msgs),
            HookDecision::Block(reason) => Err(AgentError::HookBlocked(reason)),
        }
    }

    async fn maybe_compact(&mut self) -> Result<(), AgentError> {
        // Tier 1: free, runs every turn regardless of pressure.
        self.dedup_stale_reads();

        let over_message_count = self.messages.len() > COMPACT_THRESHOLD;
        let over_token_budget = self
            .last_input_tokens
            .is_some_and(|t| t > self.token_compact_threshold());
        if !over_message_count && !over_token_budget {
            return Ok(());
        }
        self.run_before_compact_hook().await?;
        self.truncate_to(KEEP_RECENT).await;
        Ok(())
    }

    /// 75% of the provider's context window if known, else the fallback constant.
    fn token_compact_threshold(&self) -> u32 {
        match self.provider.context_window() {
            Some(window) => (window as f64 * 0.75) as u32,
            None => TOKEN_COMPACT_THRESHOLD,
        }
    }

    /// Marks a collapsed result, so a later pass can skip it.
    const STALE_READ_MARKER: &str = "(superseded by a later read_file of ";

    /// Collapses every `read_file` result but the most recent one per path
    /// into a short pointer -- the model already has the newer copy.
    fn dedup_stale_reads(&mut self) {
        let mut read_paths: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for msg in &self.messages {
            for block in &msg.content {
                if let ContentBlock::ToolUse(call) = block
                    && call.name == "read_file"
                    && let Some(path) = call.arguments.get("path").and_then(|v| v.as_str())
                {
                    read_paths.insert(call.id.clone(), path.to_string());
                }
            }
        }
        if read_paths.is_empty() {
            return;
        }

        // Last (message index, block index) touching each path.
        let mut last_seen: std::collections::HashMap<&str, (usize, usize)> = std::collections::HashMap::new();
        for (mi, msg) in self.messages.iter().enumerate() {
            for (bi, block) in msg.content.iter().enumerate() {
                if let ContentBlock::ToolResult(result) = block
                    && !result.is_error
                    && let Some(path) = read_paths.get(&result.tool_call_id)
                {
                    last_seen.insert(path.as_str(), (mi, bi));
                }
            }
        }

        for (mi, msg) in self.messages.iter_mut().enumerate() {
            for (bi, block) in msg.content.iter_mut().enumerate() {
                if let ContentBlock::ToolResult(result) = block
                    && !result.is_error
                    && let Some(path) = read_paths.get(&result.tool_call_id)
                    && last_seen.get(path.as_str()) != Some(&(mi, bi))
                    && let ToolResultContent::Text(text) = &result.content
                    && !text.starts_with(Self::STALE_READ_MARKER)
                {
                    result.content = ToolResultContent::Text(format!("{}{path})", Self::STALE_READ_MARKER));
                }
            }
        }
    }

    /// Emergency compaction after the provider itself rejects a request as too large.
    async fn force_compact(&mut self) -> Result<(), AgentError> {
        self.run_before_compact_hook().await?;
        self.truncate_to(EMERGENCY_KEEP_RECENT).await;
        Ok(())
    }

    async fn run_before_compact_hook(&self) -> Result<(), AgentError> {
        let Some(hooks) = &self.hooks else {
            return Ok(());
        };
        match hooks.lock().await.before_compact(&self.messages).await {
            HookDecision::Block(reason) => Err(AgentError::HookBlocked(reason)),
            HookDecision::Allow(()) => Ok(()),
        }
    }

    async fn run_budget_hook(&self, turn: Usage) -> Result<(), AgentError> {
        let Some(hooks) = &self.hooks else {
            return Ok(());
        };
        let info = crate::hooks::BudgetInfo {
            turn,
            session: self.total_usage,
            turn_count: self.turn_count,
        };
        match hooks.lock().await.on_budget(&info).await {
            HookDecision::Block(reason) => Err(AgentError::HookBlocked(reason)),
            HookDecision::Allow(()) => Ok(()),
        }
    }

    // Keeps only the most recent `keep` messages, salvaging what the dropped
    // ones are worth into `decision_ledger` first. Real free-text
    // summarization (Tier 3) is still a v2 concern.
    async fn truncate_to(&mut self, keep: usize) {
        if self.messages.len() <= keep {
            return;
        }
        let drop_count = self.messages.len() - keep;
        let dropped: Vec<Message> = self.messages.drain(0..drop_count).collect();
        self.decision_ledger.record(&dropped);
        // Tier 3, best-effort: a flaky/weak summarizer just means the ledger
        // (Tiers 1-2) is all that survives -- never fail compaction over it.
        if let Some(summary) = self.summarize_dropped(&dropped).await {
            self.decision_ledger.push_summary(summary);
        }
    }

    /// Asks the session's own provider to recap `dropped` in a few bullet
    /// points, via a standalone call that never touches `self.messages`.
    async fn summarize_dropped(&self, dropped: &[Message]) -> Option<String> {
        let transcript = plain_text_transcript(dropped);
        if transcript.trim().is_empty() {
            return None;
        }
        let prompt = format!("{SUMMARIZE_PROMPT_PREFIX}{transcript}");
        let resp = self
            .provider
            .complete(&[Message::user_text(prompt)], &[], None)
            .await
            .ok()?;
        let text = resp.message.text();
        (!text.trim().is_empty()).then_some(text)
    }

    async fn execute_with_hooks(&self, call: ToolCall, ctx: &ToolContext) -> Result<ToolExecOutcome, AgentError> {
        let decision = if let Some(hooks) = &self.hooks {
            hooks.lock().await.on_tool_call(&call).await
        } else {
            ToolCallDecision::Allow(call.clone())
        };

        match decision {
            ToolCallDecision::Allow(effective_call) => {
                let outcome = self.execute_tool(&effective_call, ctx).await;
                self.run_tool_result_hook(&effective_call.name, outcome).await
            }
            ToolCallDecision::Block(reason) => Ok(ToolExecOutcome {
                content: format!("Blocked by policy: {reason}"),
                is_error: true,
                metadata: serde_json::Value::Null,
            }),
            // A hook supplied the result outright -- the real tool never
            // runs, but the outcome still flows through `on_tool_result`
            // like any other, so post-processing stays uniform either way.
            ToolCallDecision::Override(outcome) => self.run_tool_result_hook(&call.name, outcome).await,
        }
    }

    /// Unknown tool name -> error result with a suggestion, not a hard failure.
    async fn execute_tool(&self, call: &ToolCall, ctx: &ToolContext) -> ToolExecOutcome {
        match self.tools.iter().find(|t| t.name() == call.name) {
            Some(tool) => tool.execute(call.arguments.clone(), ctx).await,
            None => ToolExecOutcome {
                content: unknown_tool_message(&call.name, &self.tools),
                is_error: true,
                metadata: serde_json::Value::Null,
            },
        }
    }

    async fn run_tool_result_hook(
        &self,
        tool_name: &str,
        outcome: ToolExecOutcome,
    ) -> Result<ToolExecOutcome, AgentError> {
        let Some(hooks) = &self.hooks else {
            return Ok(outcome);
        };
        let info = ToolResultInfo {
            tool_name: tool_name.to_string(),
            content: outcome.content.clone(),
            is_error: outcome.is_error,
        };
        match hooks.lock().await.on_tool_result(&info).await {
            HookDecision::Allow(content) => Ok(ToolExecOutcome { content, ..outcome }),
            HookDecision::Block(reason) => Ok(ToolExecOutcome {
                content: format!("Blocked by policy: {reason}"),
                is_error: true,
                metadata: outcome.metadata,
            }),
        }
    }

    /// Transcript so far, for a caller to persist across process restarts.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// System prompt after any `before_agent_start` hook transform.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// The active provider's id (e.g. `"ollama"`), for display purposes
    /// (banner, status lines) -- not used for any routing decision.
    pub fn provider_id(&self) -> &'static str {
        self.provider.id()
    }

    /// The active provider's model name (e.g. `"llama3.2"`), for
    /// display purposes only -- not used for any routing decision.
    pub fn model(&self) -> &str {
        self.provider.model()
    }

    /// Swaps the active provider in place; the transcript is untouched.
    pub async fn set_provider(&mut self, provider: Arc<dyn LlmProvider>) {
        let (id, model) = (provider.id(), provider.model().to_string());
        self.provider = provider;
        self.reporter.on_provider_changed(id, &model).await;
    }

    /// Loads a saved transcript and marks the session started, so
    /// `before_agent_start` won't re-run. Used to resume a prior session.
    pub fn restore(&mut self, system_prompt: String, messages: Vec<Message>) {
        self.system_prompt = system_prompt;
        self.messages = messages;
        self.started = true;
        self.decision_ledger = DecisionLedger::default();
    }

    /// Swaps in a fresh, un-cancelled `CancellationToken` for this turn and
    /// returns a clone of it, so a caller (e.g. the REPL's Ctrl-C handling)
    /// can cancel just this turn's in-flight tool calls without permanently
    /// poisoning every later turn -- `CancellationToken` never un-cancels
    /// itself once fired, so reusing one across turns would mean the first
    /// interrupt silently cancels every tool call from then on.
    pub fn reset_cancel_token(&mut self) -> tokio_util::sync::CancellationToken {
        let token = tokio_util::sync::CancellationToken::new();
        self.tool_ctx.cancel = token.clone();
        token
    }

    /// Rolls the transcript back to `pre_turn_len` (the length just before
    /// the interrupted turn started), discarding whatever partial
    /// user/assistant/tool-result messages it left behind. Without this, an
    /// interrupted turn can leave a trailing message whose role doesn't
    /// correctly alternate with the next turn's, which providers reject.
    pub fn discard_interrupted_turn(&mut self, pre_turn_len: usize) {
        self.messages.truncate(pre_turn_len);
    }

    /// Appends any steering text queued since the last drain onto
    /// `self.messages[at]` as extra content blocks, rather than pushing a
    /// new message, to keep role alternation intact for providers that
    /// reject consecutive same-role turns.
    async fn drain_steering(&mut self, at: usize) {
        let Some(rx) = &mut self.steering_rx else { return };
        let mut drained = Vec::new();
        while let Ok(text) = rx.try_recv() {
            drained.push(text);
        }
        for text in drained {
            self.reporter.on_steering_message(&text).await;
            self.messages[at].content.push(ContentBlock::Text(format!(
                "[User, while you were working on this]: {text}"
            )));
        }
    }

    /// Like `drain_steering`, but for a turn whose response had no tool
    /// calls -- there's no tool-results message at that point to append
    /// onto, and the message just before it is the assistant's own reply,
    /// so appending there would put steered text in the assistant's mouth.
    /// Pushes a plain new user message instead: the last message pushed is
    /// always the assistant's (never `Role::Tool`), so this can't create
    /// the consecutive-"user"-turn problem `drain_steering`'s doc warns
    /// about. Returns `true` (and the caller should loop for one more
    /// round-trip) only if there was anything queued.
    async fn drain_steering_into_new_turn(&mut self) -> bool {
        let Some(rx) = &mut self.steering_rx else { return false };
        let mut drained = Vec::new();
        while let Ok(text) = rx.try_recv() {
            drained.push(text);
        }
        if drained.is_empty() {
            return false;
        }
        for text in &drained {
            self.reporter.on_steering_message(text).await;
        }
        self.messages.push(Message {
            role: Role::User,
            content: drained.into_iter().map(ContentBlock::Text).collect(),
            metadata: serde_json::Value::Null,
        });
        true
    }
}

/// Suggests the closest registered tool name (Levenshtein distance) for a typo'd call.
fn unknown_tool_message(name: &str, tools: &[Arc<dyn Tool>]) -> String {
    let available: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    let suggestion = available
        .iter()
        .min_by_key(|candidate| levenshtein(name, candidate))
        .filter(|candidate| levenshtein(name, candidate) <= (name.len().max(3) / 2));

    match suggestion {
        Some(candidate) => {
            format!(
                "Unknown tool '{name}'. Did you mean '{candidate}'? Available tools: {}",
                available.join(", ")
            )
        }
        None => format!("Unknown tool '{name}'. Available tools: {}", available.join(", ")),
    }
}

/// Edit distance between two short strings.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0; b.len() + 1];

    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

const DECISION_LEDGER_COMMIT_CAP: usize = 12;
const DECISION_LEDGER_FILE_CAP: usize = 30;
const DECISION_LEDGER_SUMMARY_CAP: usize = 5;

/// Prompt for Tier 3's summarization call -- kept short and demanding
/// brevity, since the summarizer may itself be a small local model.
const SUMMARIZE_PROMPT_PREFIX: &str = "Summarize the key facts, decisions, and open threads from this part of a coding session in 3-5 short bullet points. Be concise.\n\n";

/// Structured facts mechanically extracted from tool calls about to be
/// dropped by truncation, so a compacted session doesn't lose track of what
/// it already did. Deliberately not an LLM-written summary -- see
/// `AgentSession::complete_with_retries`, which folds `render()`'s output
/// into the system prompt.
#[derive(Default)]
struct DecisionLedger {
    touched_files: Vec<String>,
    latest_todo: Option<String>,
    commits: Vec<String>,
    /// Tier 3: short LLM-written recaps of prose (non-tool) content that Tiers
    /// 1-2 can't capture, oldest first, capped at `DECISION_LEDGER_SUMMARY_CAP`.
    summaries: Vec<String>,
}

impl DecisionLedger {
    fn push_summary(&mut self, summary: String) {
        self.summaries.push(summary);
        if self.summaries.len() > DECISION_LEDGER_SUMMARY_CAP {
            self.summaries.remove(0);
        }
    }

    fn record(&mut self, dropped: &[Message]) {
        for msg in dropped {
            for block in &msg.content {
                let ContentBlock::ToolUse(call) = block else { continue };
                match call.name.as_str() {
                    "git_commit" => {
                        if let Some(m) = call.arguments.get("message").and_then(|v| v.as_str()) {
                            self.commits.push(m.to_string());
                            if self.commits.len() > DECISION_LEDGER_COMMIT_CAP {
                                self.commits.remove(0);
                            }
                        }
                    }
                    "todo_write" => self.latest_todo = summarize_todo_write(&call.arguments),
                    "write_file" | "edit_file" | "delete_file" => {
                        if let Some(path) = call.arguments.get("path").and_then(|v| v.as_str())
                            && !self.touched_files.iter().any(|p| p == path)
                        {
                            self.touched_files.push(path.to_string());
                            if self.touched_files.len() > DECISION_LEDGER_FILE_CAP {
                                self.touched_files.remove(0);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn render(&self) -> Option<String> {
        if self.touched_files.is_empty()
            && self.latest_todo.is_none()
            && self.commits.is_empty()
            && self.summaries.is_empty()
        {
            return None;
        }
        let mut out = String::from("Earlier context was compacted; these facts from it still apply:\n");
        if let Some(todo) = &self.latest_todo {
            out.push_str(&format!("Current todo list:\n{todo}\n"));
        }
        if !self.touched_files.is_empty() {
            out.push_str(&format!("Files touched so far: {}\n", self.touched_files.join(", ")));
        }
        if !self.commits.is_empty() {
            out.push_str("Commits made:\n");
            for c in &self.commits {
                out.push_str(&format!("  - {c}\n"));
            }
        }
        if !self.summaries.is_empty() {
            out.push_str("Summary of earlier discussion:\n");
            for s in &self.summaries {
                out.push_str(&format!("  - {s}\n"));
            }
        }
        Some(out)
    }
}

/// A `todo_write` call's `todos` array as one checkmark-per-line string, or
/// `None` if the arguments don't parse (malformed calls just aren't logged).
fn summarize_todo_write(args: &serde_json::Value) -> Option<String> {
    let todos = args.get("todos")?.as_array()?;
    let lines: Vec<String> = todos
        .iter()
        .filter_map(|t| {
            let content = t.get("content")?.as_str()?;
            let mark = match t.get("status").and_then(|s| s.as_str()) {
                Some("completed") => "x",
                Some("in_progress") => "~",
                _ => " ",
            };
            Some(format!("  [{mark}] {content}"))
        })
        .collect();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// User/Assistant text content, one line per message, for Tier 3's
/// summarization prompt -- tool calls/results are structured data already
/// captured by Tiers 1-2, so they're left out here.
fn plain_text_transcript(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|m| matches!(m.role, Role::User | Role::Assistant))
        .filter_map(|m| {
            let text = m.text();
            (!text.trim().is_empty()).then(|| format!("{:?}: {text}", m.role))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// True if a provider error looks like "request too large for context window".
fn is_context_length_error(err: &ProviderError) -> bool {
    let ProviderError::Api { status, body } = err else {
        return false;
    };
    if *status == 413 {
        return true;
    }
    const NEEDLES: [&str; 6] = [
        "context length",
        "context_length",
        "context window",
        "too many tokens",
        "maximum context",
        "prompt is too long",
    ];
    let body = body.to_lowercase();
    NEEDLES.iter().any(|needle| body.contains(needle))
}

/// True for provider errors worth retrying: rate limits, 5xx, and transport
/// (network) failures. Anything else (bad request, malformed response) is a
/// permanent failure that retrying can't fix.
fn is_transient_error(err: &ProviderError) -> bool {
    match err {
        ProviderError::RateLimited { .. } | ProviderError::Transport(_) => true,
        ProviderError::Api { status, .. } => *status >= 500,
        ProviderError::Deserialize(_) => false,
    }
}

/// Exponential backoff (base 2s, capped at 60s), except a rate limit with an
/// explicit `retry_after_secs` is honored as-is.
fn backoff_delay(attempt: usize, err: &ProviderError) -> Duration {
    if let ProviderError::RateLimited {
        retry_after_secs: Some(secs),
    } = err
    {
        return Duration::from_secs(*secs);
    }
    let secs = BASE_BACKOFF
        .as_secs()
        .saturating_mul(1u64 << attempt.min(5))
        .min(MAX_BACKOFF.as_secs());
    Duration::from_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, ProviderResponse, Role, Usage};
    use std::sync::Mutex as StdMutex;

    /// Returns a fixed queue of responses, one per `complete()` call --
    /// enough to drive the loop through a scripted tool-call sequence
    /// without a network call.
    struct ScriptedProvider(StdMutex<std::collections::VecDeque<ProviderResponse>>);

    impl ScriptedProvider {
        fn new(responses: Vec<ProviderResponse>) -> Self {
            Self(StdMutex::new(responses.into()))
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for ScriptedProvider {
        fn id(&self) -> &'static str {
            "scripted"
        }
        fn model(&self) -> &str {
            "scripted-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(self.0.lock().unwrap().pop_front().expect("script exhausted"))
        }
    }

    /// Distinct id/model from `ScriptedProvider`, for `set_provider` tests.
    struct AltProvider(StdMutex<std::collections::VecDeque<ProviderResponse>>);

    impl AltProvider {
        fn new(responses: Vec<ProviderResponse>) -> Self {
            Self(StdMutex::new(responses.into()))
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for AltProvider {
        fn id(&self) -> &'static str {
            "alt"
        }
        fn model(&self) -> &str {
            "alt-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(self.0.lock().unwrap().pop_front().expect("script exhausted"))
        }
    }

    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes its `text` argument"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}})
        }
        async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
            ToolExecOutcome {
                content: arguments["text"].as_str().unwrap_or_default().to_string(),
                is_error: false,
                metadata: serde_json::Value::Null,
            }
        }
    }

    fn tool_use_response(call_id: &str, tool: &str, args: serde_json::Value) -> ProviderResponse {
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse(ToolCall {
                    id: call_id.to_string(),
                    name: tool.to_string(),
                    arguments: args,
                })],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        }
    }

    fn text_response(text: &str) -> ProviderResponse {
        text_response_with_usage(text, 0)
    }

    fn text_response_with_usage(text: &str, input_tokens: u32) -> ProviderResponse {
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(text.to_string())],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens,
                output_tokens: 0,
            },
        }
    }

    /// Records `on_thinking`/`on_assistant_text` calls in order, so tests can
    /// assert a `Thinking` block reaches the reporter distinctly from `Text`.
    #[derive(Default)]
    struct SpyReporter(StdMutex<Vec<String>>);

    #[async_trait::async_trait]
    impl Reporter for SpyReporter {
        async fn on_thinking(&self, text: &str) {
            self.0.lock().unwrap().push(format!("thinking:{text}"));
        }
        async fn on_assistant_text(&self, text: &str) {
            self.0.lock().unwrap().push(format!("text:{text}"));
        }
        async fn on_steering_message(&self, text: &str) {
            self.0.lock().unwrap().push(format!("steering:{text}"));
        }
        async fn on_usage(&self, usage: &Usage) {
            self.0
                .lock()
                .unwrap()
                .push(format!("usage:{}/{}", usage.input_tokens, usage.output_tokens));
        }
        async fn on_provider_changed(&self, provider_id: &str, model: &str) {
            self.0
                .lock()
                .unwrap()
                .push(format!("provider_changed:{provider_id}/{model}"));
        }
    }

    fn thinking_then_text_response(thinking: &str, text: &str) -> ProviderResponse {
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        text: thinking.to_string(),
                        signature: None,
                    },
                    ContentBlock::Text(text.to_string()),
                ],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    #[tokio::test]
    async fn thinking_block_reaches_the_reporter_ahead_of_the_final_text() {
        let provider = ScriptedProvider::new(vec![thinking_then_text_response(
            "working through the problem",
            "here's the answer",
        )]);
        let spy = Arc::new(SpyReporter::default());
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "you are a test agent", test_ctx())
            .with_reporter(spy.clone());

        session.run_turn("solve it").await.unwrap();

        assert_eq!(
            spy.0.lock().unwrap().as_slice(),
            [
                "thinking:working through the problem".to_string(),
                "text:here's the answer".to_string(),
                "usage:0/0".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn on_usage_sums_tokens_across_every_round_trip_in_the_turn() {
        let provider = ScriptedProvider::new(vec![
            ProviderResponse {
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 10,
                },
                ..tool_use_response("call_1", "echo", serde_json::json!({"text": "hi"}))
            },
            ProviderResponse {
                usage: Usage {
                    input_tokens: 150,
                    output_tokens: 20,
                },
                ..text_response("done")
            },
        ]);
        let spy = Arc::new(SpyReporter::default());
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(EchoTool)],
            None,
            "you are a test agent",
            test_ctx(),
        )
        .with_reporter(spy.clone());

        session.run_turn("do it").await.unwrap();

        // 100+150 input, 10+20 output -- summed across both round-trips, not
        // just the final one.
        assert!(
            spy.0.lock().unwrap().iter().any(|line| line == "usage:250/30"),
            "usage not summed correctly: {:?}",
            spy.0.lock().unwrap()
        );
    }

    fn test_ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
            mailbox: None,
            ask: crate::ask::AskChannel::unavailable(),
        }
    }

    #[tokio::test]
    async fn loop_executes_tool_call_then_terminates_on_text_response() {
        let provider = ScriptedProvider::new(vec![
            tool_use_response("call_1", "echo", serde_json::json!({"text": "hi from tool"})),
            text_response("the tool said: hi from tool"),
        ]);
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(EchoTool)],
            None,
            "you are a test agent",
            test_ctx(),
        );

        let final_message = session.run_turn("please echo something").await.unwrap();

        match &final_message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "the tool said: hi from tool"),
            other => panic!("expected final Text response, got {other:?}"),
        }
        // user input, assistant tool_use, tool results, assistant final text
        assert_eq!(session.messages.len(), 4);
    }

    #[tokio::test]
    async fn set_provider_swaps_the_provider_used_by_the_next_turn() {
        let spy = Arc::new(SpyReporter::default());
        let mut session = AgentSession::new(
            Arc::new(ScriptedProvider::new(vec![text_response("from scripted")])),
            vec![],
            None,
            "you are a test agent",
            test_ctx(),
        )
        .with_reporter(spy.clone());

        let first = session.run_turn("hi").await.unwrap();
        assert_eq!(first.text(), "from scripted");
        assert_eq!(session.provider_id(), "scripted");
        assert_eq!(session.model(), "scripted-model");

        session
            .set_provider(Arc::new(AltProvider::new(vec![text_response("from alt")])))
            .await;
        assert_eq!(session.provider_id(), "alt");
        assert_eq!(session.model(), "alt-model");
        assert!(
            spy.0
                .lock()
                .unwrap()
                .contains(&"provider_changed:alt/alt-model".to_string())
        );

        // History survives the swap.
        let second = session.run_turn("hi again").await.unwrap();
        assert_eq!(second.text(), "from alt");
        assert_eq!(session.messages.len(), 4);
    }

    /// Tracks the peak number of overlapping `execute` calls.
    struct ConcurrencyProbeTool {
        active: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Tool for ConcurrencyProbeTool {
        fn name(&self) -> &str {
            "agent"
        }
        fn description(&self) -> &str {
            "probes concurrency"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
            use std::sync::atomic::Ordering;
            let current = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(current, Ordering::SeqCst);
            tokio::task::yield_now().await; // let the other call get polled too
            self.active.fetch_sub(1, Ordering::SeqCst);
            ToolExecOutcome {
                content: "done".to_string(),
                is_error: false,
                metadata: serde_json::Value::Null,
            }
        }
    }

    fn two_agent_tool_calls_response() -> ProviderResponse {
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolUse(ToolCall {
                        id: "call_1".to_string(),
                        name: "agent".to_string(),
                        arguments: serde_json::json!({}),
                    }),
                    ContentBlock::ToolUse(ToolCall {
                        id: "call_2".to_string(),
                        name: "agent".to_string(),
                        arguments: serde_json::json!({}),
                    }),
                ],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        }
    }

    #[tokio::test]
    async fn multiple_agent_tool_calls_in_one_turn_run_concurrently() {
        let provider = ScriptedProvider::new(vec![two_agent_tool_calls_response(), text_response("both done")]);
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tool = ConcurrencyProbeTool {
            active: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            peak: peak.clone(),
        };
        let mut session = AgentSession::new(Arc::new(provider), vec![Arc::new(tool)], None, "test", test_ctx());

        session.run_turn("delegate two things at once").await.unwrap();

        assert_eq!(
            peak.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "both `agent` calls in the same turn should have been in flight at once"
        );
    }

    #[tokio::test]
    async fn loop_terminates_immediately_with_no_tool_calls() {
        let provider = ScriptedProvider::new(vec![text_response("no tools needed")]);
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "you are a test agent", test_ctx());

        let final_message = session.run_turn("hello").await.unwrap();
        match &final_message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "no tools needed"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(session.messages.len(), 2); // user input, assistant text
    }

    #[tokio::test]
    async fn unknown_tool_call_is_reported_back_instead_of_aborting_the_turn() {
        let provider = ScriptedProvider::new(vec![
            tool_use_response("call_1", "grpe", serde_json::json!({})),
            text_response("retried with the right tool"),
        ]);
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(EchoTool)], // named "echo", close enough to "grpe" to never match
            None,
            "you are a test agent",
            test_ctx(),
        );

        let final_message = session.run_turn("do something").await.unwrap();
        match &final_message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "retried with the right tool"),
            other => panic!("expected Text, got {other:?}"),
        }

        let tool_result_msg = &session.messages[2];
        match &tool_result_msg.content[0] {
            ContentBlock::ToolResult(r) => {
                assert!(r.is_error);
                match &r.content {
                    ToolResultContent::Text(t) => assert!(t.contains("Unknown tool 'grpe'"), "got: {t}"),
                    other => panic!("expected Text, got {other:?}"),
                }
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn levenshtein_finds_close_names_but_not_distant_ones() {
        assert_eq!(levenshtein("grep", "grep"), 0);
        assert_eq!(levenshtein("grpe", "grep"), 2);
        assert!(levenshtein("bash", "web_fetch") > 3);
    }

    #[test]
    fn unknown_tool_message_suggests_the_closest_registered_name() {
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        assert!(unknown_tool_message("ecko", &tools).contains("Did you mean 'echo'?"));
        assert!(!unknown_tool_message("completely_unrelated_xyz", &tools).contains("Did you mean"));
    }

    #[tokio::test]
    async fn restore_replaces_history_and_skips_before_agent_start_again() {
        let provider = ScriptedProvider::new(vec![text_response("continuing")]);
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "original prompt", test_ctx());

        session.restore("restored prompt".to_string(), vec![Message::user_text("earlier turn")]);
        assert_eq!(session.system_prompt(), "restored prompt");
        assert_eq!(session.messages().len(), 1);

        session.run_turn("follow up").await.unwrap();
        assert_eq!(session.messages().len(), 3);
    }

    #[tokio::test]
    async fn reset_cancel_token_returns_a_fresh_uncancelled_token_each_time() {
        let provider = ScriptedProvider::new(vec![]);
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test agent", test_ctx());

        let first = session.reset_cancel_token();
        first.cancel();
        assert!(first.is_cancelled());

        let second = session.reset_cancel_token();
        assert!(
            !second.is_cancelled(),
            "a later turn's token must not inherit an earlier turn's cancellation"
        );
    }

    #[tokio::test]
    async fn discard_interrupted_turn_rolls_the_transcript_back() {
        let provider = ScriptedProvider::new(vec![
            tool_use_response("call_1", "echo", serde_json::json!({"text": "hi"})),
            text_response("done"),
        ]);
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(EchoTool)],
            None,
            "test agent",
            test_ctx(),
        );
        session.restore("test agent".to_string(), vec![Message::user_text("earlier turn")]);
        let pre_turn_len = session.messages().len();

        // A real interrupt drops the turn mid-flight; here the turn just
        // runs to completion, but `discard_interrupted_turn` only cares
        // about rolling `messages` back to a prior length, so exercising it
        // against a normal turn's (larger) transcript is equivalent.
        session.run_turn("do something").await.unwrap();
        assert!(session.messages().len() > pre_turn_len);

        session.discard_interrupted_turn(pre_turn_len);
        assert_eq!(session.messages().len(), pre_turn_len);
    }

    #[tokio::test]
    async fn drain_steering_appends_onto_the_existing_message_at_that_index() {
        // Exercises `drain_steering` directly rather than through a real
        // `run_turn`: sending before the turn even starts would just get
        // picked up by *that* turn's own initial drain (there's only one
        // queue), so this is the deterministic way to pin down what a
        // mid-turn arrival (a real race between the terminal and the
        // running turn, in practice) does to an already-pushed message.
        let provider = ScriptedProvider::new(vec![
            tool_use_response("call_1", "echo", serde_json::json!({"text": "hi"})),
            text_response("done"),
        ]);
        let reporter = Arc::new(SpyReporter::default());
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(EchoTool)],
            None,
            "test agent",
            test_ctx(),
        )
        .with_reporter(reporter.clone());
        session.run_turn("do something").await.unwrap();
        let tool_results_idx = session
            .messages()
            .iter()
            .position(|m| m.role == Role::Tool)
            .expect("a tool-results message was pushed");

        let steering_tx = session.enable_steering();
        steering_tx.send("also check the licensing".to_string()).unwrap();
        session.drain_steering(tool_results_idx).await;

        let has_steering_text = session.messages()[tool_results_idx]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text(t) if t.contains("also check the licensing")));
        assert!(has_steering_text, "got: {:?}", session.messages()[tool_results_idx]);
        assert!(
            reporter
                .0
                .lock()
                .unwrap()
                .contains(&"steering:also check the licensing".to_string())
        );
    }

    #[tokio::test]
    async fn steering_text_arriving_during_a_toolless_turn_becomes_a_follow_up_turn() {
        // Exercises `drain_steering_into_new_turn` directly for the same
        // reason `drain_steering_appends_onto_the_existing_message_at_that_index`
        // does: a genuine mid-turn arrival is a race between the terminal
        // and the running turn, so this pins down deterministically what a
        // text-only turn does with it instead of losing it until the next
        // `run_turn` call (the bug this covers).
        let provider = ScriptedProvider::new(vec![text_response("first reply")]);
        let reporter = Arc::new(SpyReporter::default());
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test agent", test_ctx())
            .with_reporter(reporter.clone());
        session.run_turn("first").await.unwrap();

        let steering_tx = session.enable_steering();
        steering_tx.send("also check the changelog".to_string()).unwrap();
        let queued = session.drain_steering_into_new_turn().await;
        assert!(queued, "expected queued steering text to start a follow-up turn");

        let last_message = session.messages().last().expect("a message was pushed");
        assert_eq!(last_message.role, Role::User);
        let has_steering_text = last_message
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text(t) if t.contains("also check the changelog")));
        assert!(has_steering_text, "got: {last_message:?}");

        assert!(
            !session.drain_steering_into_new_turn().await,
            "an empty queue must not start another turn"
        );
    }

    #[tokio::test]
    async fn steering_text_left_unconsumed_by_one_turn_carries_over_to_the_next() {
        // Only a text response -- the turn ends after the very first
        // provider call, before the loop ever reaches a tool-results drain
        // point, so anything queued after `run_turn` returns has nowhere to
        // land until the *next* call's initial drain.
        let provider = ScriptedProvider::new(vec![text_response("immediate reply"), text_response("second reply")]);
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test agent", test_ctx());
        let steering_tx = session.enable_steering();

        session.run_turn("first").await.unwrap();
        steering_tx.send("don't forget the changelog".to_string()).unwrap();
        session.run_turn("second").await.unwrap();

        let messages = session.messages();
        let second_user_message = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .expect("the second turn's user message");
        let has_steering_text = second_user_message
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text(t) if t.contains("don't forget the changelog")));
        assert!(has_steering_text, "got: {second_user_message:?}");
    }

    #[tokio::test]
    async fn no_enable_steering_call_means_drain_is_a_harmless_noop() {
        let provider = ScriptedProvider::new(vec![text_response("reply")]);
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test agent", test_ctx());
        // enable_steering was never called -- steering_rx stays None.
        session.run_turn("hi").await.unwrap();
    }

    struct FlakyThenOkProvider {
        calls: StdMutex<usize>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for FlakyThenOkProvider {
        fn id(&self) -> &'static str {
            "flaky"
        }
        fn model(&self) -> &str {
            "flaky-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            if *calls == 1 {
                Err(ProviderError::Api {
                    status: 400,
                    body: "maximum context length exceeded".to_string(),
                })
            } else {
                Ok(text_response("recovered"))
            }
        }
    }

    #[tokio::test]
    async fn context_length_error_triggers_compaction_and_retry() {
        let provider = FlakyThenOkProvider {
            calls: StdMutex::new(0),
        };
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test", test_ctx());
        let seed: Vec<Message> = (0..25).map(|i| Message::user_text(format!("msg {i}"))).collect();
        session.restore("test".to_string(), seed);

        let final_message = session.run_turn("trigger").await.unwrap();
        match &final_message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "recovered"),
            other => panic!("expected Text, got {other:?}"),
        }
        // 25 seeded + 1 user = 26, compacted to EMERGENCY_KEEP_RECENT (20), + 1 assistant reply.
        assert_eq!(session.messages().len(), EMERGENCY_KEEP_RECENT + 1);
    }

    #[test]
    fn transient_errors_are_classified_correctly() {
        assert!(is_transient_error(&ProviderError::RateLimited {
            retry_after_secs: None
        }));
        assert!(is_transient_error(&ProviderError::Transport("boom".to_string())));
        assert!(is_transient_error(&ProviderError::Api {
            status: 503,
            body: String::new()
        }));
        assert!(!is_transient_error(&ProviderError::Api {
            status: 400,
            body: "bad request".to_string()
        }));
        assert!(!is_transient_error(&ProviderError::Deserialize("oops".to_string())));
    }

    #[test]
    fn backoff_honors_retry_after_and_otherwise_grows_exponentially_up_to_a_cap() {
        let rate_limited = ProviderError::RateLimited {
            retry_after_secs: Some(7),
        };
        assert_eq!(backoff_delay(0, &rate_limited), Duration::from_secs(7));

        let transport = ProviderError::Transport("x".to_string());
        assert_eq!(backoff_delay(0, &transport), Duration::from_secs(2));
        assert_eq!(backoff_delay(1, &transport), Duration::from_secs(4));
        assert_eq!(backoff_delay(10, &transport), Duration::from_secs(60));
    }

    struct TransientThenOkProvider {
        calls: StdMutex<usize>,
        fail_times: usize,
    }

    #[async_trait::async_trait]
    impl LlmProvider for TransientThenOkProvider {
        fn id(&self) -> &'static str {
            "transient"
        }
        fn model(&self) -> &str {
            "transient-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            if *calls <= self.fail_times {
                Err(ProviderError::RateLimited {
                    retry_after_secs: Some(1),
                })
            } else {
                Ok(text_response("recovered"))
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn transient_errors_are_retried_until_success() {
        let provider = TransientThenOkProvider {
            calls: StdMutex::new(0),
            fail_times: 3,
        };
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test", test_ctx());

        let final_message = session.run_turn("go").await.unwrap();
        match &final_message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "recovered"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn exhausting_transient_retries_surfaces_the_error() {
        let provider = TransientThenOkProvider {
            calls: StdMutex::new(0),
            fail_times: MAX_TRANSIENT_RETRIES + 1,
        };
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test", test_ctx());

        let err = session.run_turn("go").await.unwrap_err();
        assert!(matches!(err, AgentError::Provider(ProviderError::RateLimited { .. })));
    }

    struct AlwaysFailingProvider;

    #[async_trait::async_trait]
    impl LlmProvider for AlwaysFailingProvider {
        fn id(&self) -> &'static str {
            "always-failing"
        }
        fn model(&self) -> &str {
            "always-failing-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            Err(ProviderError::Api {
                status: 400,
                body: "bad request".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn a_non_transient_error_rolls_the_turn_back_instead_of_leaving_a_dangling_message() {
        let mut session = AgentSession::new(Arc::new(AlwaysFailingProvider), vec![], None, "test", test_ctx());
        session.restore(
            "test".to_string(),
            vec![
                Message::user_text("earlier turn"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text("earlier reply".to_string())],
                    metadata: serde_json::Value::Null,
                },
            ],
        );
        let pre_turn_len = session.messages().len();

        let err = session.run_turn("this will fail").await.unwrap_err();
        assert!(matches!(
            err,
            AgentError::Provider(ProviderError::Api { status: 400, .. })
        ));
        assert_eq!(
            session.messages().len(),
            pre_turn_len,
            "a failed turn must not leave a dangling, unanswered user message behind"
        );
    }

    struct FailOnceThenOkProvider {
        calls: StdMutex<usize>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for FailOnceThenOkProvider {
        fn id(&self) -> &'static str {
            "fail-once"
        }
        fn model(&self) -> &str {
            "fail-once-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            if *calls == 1 {
                Err(ProviderError::Api {
                    status: 400,
                    body: "bad request".to_string(),
                })
            } else {
                Ok(text_response("recovered"))
            }
        }
    }

    #[tokio::test]
    async fn a_failed_turn_does_not_prevent_the_next_turn_from_succeeding() {
        let provider = FailOnceThenOkProvider {
            calls: StdMutex::new(0),
        };
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test", test_ctx());

        session.run_turn("first, will fail").await.unwrap_err();

        let final_message = session.run_turn("second, should work").await.unwrap();
        match &final_message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "recovered"),
            other => panic!("expected Text, got {other:?}"),
        }
        // user input, assistant reply -- no leftover dangling message from the failed first turn.
        assert_eq!(session.messages().len(), 2);
    }

    #[tokio::test]
    async fn high_token_usage_triggers_proactive_compaction_under_message_threshold() {
        let provider = ScriptedProvider::new(vec![
            text_response_with_usage("first", TOKEN_COMPACT_THRESHOLD + 1),
            text_response("summary"), // consumed by Tier 3's summarization call during compaction
            text_response("second"),
        ]);
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "test", test_ctx());
        session.run_turn("prime usage").await.unwrap();

        let seed: Vec<Message> = (0..44).map(|i| Message::user_text(format!("msg {i}"))).collect();
        session.restore("test".to_string(), seed);

        session.run_turn("go").await.unwrap();
        // 44 seeded + 1 user = 45, under COMPACT_THRESHOLD (60), but the primed
        // usage was over TOKEN_COMPACT_THRESHOLD so it compacts anyway.
        assert_eq!(session.messages().len(), KEEP_RECENT + 1);
    }

    struct FakeReadFileTool(&'static str);

    #[async_trait::async_trait]
    impl Tool for FakeReadFileTool {
        fn name(&self) -> &str {
            "read_file"
        }
        fn description(&self) -> &str {
            "reads a file"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
        }
        async fn execute(&self, _arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
            ToolExecOutcome {
                content: self.0.to_string(),
                is_error: false,
                metadata: serde_json::Value::Null,
            }
        }
    }

    fn tool_result_content<'a>(messages: &'a [Message], call_id: &str) -> &'a ToolResultContent {
        messages
            .iter()
            .flat_map(|m| &m.content)
            .find_map(|b| match b {
                ContentBlock::ToolResult(r) if r.tool_call_id == call_id => Some(&r.content),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no tool result for {call_id}"))
    }

    #[tokio::test]
    async fn a_stale_read_file_result_is_collapsed_once_the_file_is_read_again() {
        let provider = ScriptedProvider::new(vec![
            tool_use_response("call1", "read_file", serde_json::json!({"path": "a.txt"})),
            text_response("ok"),
            tool_use_response("call2", "read_file", serde_json::json!({"path": "a.txt"})),
            text_response("ok again"),
        ]);
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(FakeReadFileTool("the real file content"))],
            None,
            "test",
            test_ctx(),
        );

        session.run_turn("read it").await.unwrap();
        session.run_turn("read it again").await.unwrap();

        let ToolResultContent::Text(first) = tool_result_content(session.messages(), "call1") else {
            panic!("expected Text")
        };
        assert!(first.contains("superseded") && first.contains("a.txt"), "got: {first}");

        let ToolResultContent::Text(second) = tool_result_content(session.messages(), "call2") else {
            panic!("expected Text")
        };
        assert_eq!(second, "the real file content");
    }

    #[tokio::test]
    async fn reads_of_different_paths_are_left_alone() {
        let provider = ScriptedProvider::new(vec![
            tool_use_response("call1", "read_file", serde_json::json!({"path": "a.txt"})),
            text_response("ok"),
            tool_use_response("call2", "read_file", serde_json::json!({"path": "b.txt"})),
            text_response("ok again"),
        ]);
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(FakeReadFileTool("content"))],
            None,
            "test",
            test_ctx(),
        );

        session.run_turn("read a").await.unwrap();
        session.run_turn("read b").await.unwrap();

        for id in ["call1", "call2"] {
            let ToolResultContent::Text(text) = tool_result_content(session.messages(), id) else {
                panic!("expected Text")
            };
            assert_eq!(text, "content", "{id} should not have been collapsed");
        }
    }

    struct SmallContextProvider;

    #[async_trait::async_trait]
    impl LlmProvider for SmallContextProvider {
        fn id(&self) -> &'static str {
            "small"
        }
        fn model(&self) -> &str {
            "small-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            unreachable!("not exercised by this test")
        }
        fn context_window(&self) -> Option<u32> {
            Some(8192)
        }
    }

    #[test]
    fn token_compact_threshold_scales_with_the_providers_context_window() {
        let session = AgentSession::new(Arc::new(SmallContextProvider), vec![], None, "test", test_ctx());
        assert_eq!(session.token_compact_threshold(), 6144); // 75% of 8192
    }

    #[test]
    fn token_compact_threshold_falls_back_to_the_constant_when_unknown() {
        let session = AgentSession::new(
            Arc::new(ScriptedProvider::new(vec![])),
            vec![],
            None,
            "test",
            test_ctx(),
        );
        assert_eq!(session.token_compact_threshold(), TOKEN_COMPACT_THRESHOLD);
    }

    fn tool_use_message(id: &str, tool: &str, args: serde_json::Value) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse(ToolCall {
                id: id.to_string(),
                name: tool.to_string(),
                arguments: args,
            })],
            metadata: serde_json::Value::Null,
        }
    }

    #[test]
    fn decision_ledger_captures_touched_files_commits_and_the_todo_list() {
        let mut ledger = DecisionLedger::default();
        ledger.record(&[
            tool_use_message("1", "write_file", serde_json::json!({"path": "src/a.rs"})),
            tool_use_message("2", "git_commit", serde_json::json!({"message": "add a.rs"})),
            tool_use_message(
                "3",
                "todo_write",
                serde_json::json!({"todos": [
                    {"content": "write a.rs", "status": "completed"},
                    {"content": "write tests", "status": "in_progress"},
                ]}),
            ),
        ]);

        let rendered = ledger.render().unwrap();
        assert!(rendered.contains("src/a.rs"), "{rendered}");
        assert!(rendered.contains("add a.rs"), "{rendered}");
        assert!(rendered.contains("[x] write a.rs"), "{rendered}");
        assert!(rendered.contains("[~] write tests"), "{rendered}");
    }

    #[test]
    fn decision_ledger_renders_nothing_when_empty() {
        assert!(DecisionLedger::default().render().is_none());
    }

    #[test]
    fn decision_ledger_keeps_only_the_latest_todo_write() {
        let mut ledger = DecisionLedger::default();
        ledger.record(&[tool_use_message(
            "1",
            "todo_write",
            serde_json::json!({"todos": [{"content": "first plan", "status": "pending"}]}),
        )]);
        ledger.record(&[tool_use_message(
            "2",
            "todo_write",
            serde_json::json!({"todos": [{"content": "revised plan", "status": "pending"}]}),
        )]);

        let rendered = ledger.render().unwrap();
        assert!(rendered.contains("revised plan"));
        assert!(!rendered.contains("first plan"));
    }

    struct CapturingProvider {
        responses: StdMutex<std::collections::VecDeque<ProviderResponse>>,
        captured_system_prompts: StdMutex<Vec<Option<String>>>,
    }

    impl CapturingProvider {
        fn new(responses: Vec<ProviderResponse>) -> Self {
            Self {
                responses: StdMutex::new(responses.into()),
                captured_system_prompts: StdMutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for CapturingProvider {
        fn id(&self) -> &'static str {
            "capturing"
        }
        fn model(&self) -> &str {
            "capturing-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            self.captured_system_prompts
                .lock()
                .unwrap()
                .push(system_prompt.map(String::from));
            Ok(self.responses.lock().unwrap().pop_front().expect("script exhausted"))
        }
    }

    #[tokio::test]
    async fn compacted_facts_reach_the_system_prompt_on_the_next_call() {
        let provider = Arc::new(CapturingProvider::new(vec![
            text_response("summary"), // consumed by Tier 3's summarization call during compaction
            text_response("done"),
        ]));
        let mut session = AgentSession::new(provider.clone(), vec![], None, "base prompt", test_ctx());

        let mut seed = vec![tool_use_message(
            "1",
            "write_file",
            serde_json::json!({"path": "src/a.rs"}),
        )];
        seed.extend((0..COMPACT_THRESHOLD).map(|i| Message::user_text(format!("msg {i}"))));
        session.restore("base prompt".to_string(), seed);

        session.run_turn("go").await.unwrap();

        let captured = provider.captured_system_prompts.lock().unwrap();
        let last = captured
            .last()
            .expect("one call")
            .as_ref()
            .expect("system prompt was set");
        assert!(last.contains("src/a.rs"), "got: {last}");
        assert!(last.starts_with("base prompt"), "got: {last}");
    }

    #[test]
    fn decision_ledger_caps_summaries_dropping_the_oldest() {
        let mut ledger = DecisionLedger::default();
        for i in 0..DECISION_LEDGER_SUMMARY_CAP + 2 {
            ledger.push_summary(format!("summary {i}"));
        }
        let rendered = ledger.render().unwrap();
        assert!(
            !rendered.contains("summary 0"),
            "oldest summary should have been dropped"
        );
        assert!(rendered.contains(&format!("summary {}", DECISION_LEDGER_SUMMARY_CAP + 1)));
    }

    struct PanicIfCalledProvider;

    #[async_trait::async_trait]
    impl LlmProvider for PanicIfCalledProvider {
        fn id(&self) -> &'static str {
            "panic-if-called"
        }
        fn model(&self) -> &str {
            "panic-if-called-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            panic!("should never be called -- nothing in the transcript to summarize");
        }
    }

    #[tokio::test]
    async fn summarize_dropped_skips_the_call_when_theres_no_prose() {
        let session = AgentSession::new(Arc::new(PanicIfCalledProvider), vec![], None, "test", test_ctx());
        let dropped = vec![tool_use_message("1", "write_file", serde_json::json!({"path": "a.rs"}))];
        assert_eq!(session.summarize_dropped(&dropped).await, None);
    }

    #[tokio::test]
    async fn summarize_dropped_is_best_effort_on_a_flaky_provider() {
        let session = AgentSession::new(Arc::new(AlwaysFailingProvider), vec![], None, "test", test_ctx());
        let dropped = vec![Message::user_text("please remember this")];
        assert_eq!(session.summarize_dropped(&dropped).await, None);
    }

    /// Counts real invocations so a test can assert an `Override`d call
    /// never reached it.
    struct CountingTool(Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait::async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "counts calls, echoes its `text` argument"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}})
        }
        async fn execute(&self, _arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ToolExecOutcome {
                content: "real tool output".to_string(),
                is_error: false,
                metadata: serde_json::Value::Null,
            }
        }
    }

    /// A `HookPort` that always overrides tool calls with a canned outcome,
    /// and otherwise allows everything through unmodified.
    struct OverrideHooks;

    #[async_trait::async_trait]
    impl HookPort for OverrideHooks {
        async fn before_agent_start(&mut self, system_prompt: &str) -> HookDecision<String> {
            HookDecision::Allow(system_prompt.to_string())
        }
        async fn on_context(&mut self, messages: &[Message]) -> HookDecision<Vec<Message>> {
            HookDecision::Allow(messages.to_vec())
        }
        async fn on_tool_call(&mut self, _call: &ToolCall) -> ToolCallDecision {
            ToolCallDecision::Override(ToolExecOutcome {
                content: "mocked by hook".to_string(),
                is_error: false,
                metadata: serde_json::Value::Null,
            })
        }
        async fn on_tool_result(&mut self, result: &ToolResultInfo) -> HookDecision<String> {
            HookDecision::Allow(result.content.clone())
        }
        async fn before_compact(&mut self, _messages: &[Message]) -> HookDecision<()> {
            HookDecision::Allow(())
        }
    }

    #[tokio::test]
    async fn tool_call_override_skips_the_real_tool_but_still_runs_on_tool_result() {
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let provider = ScriptedProvider::new(vec![
            tool_use_response("call_1", "echo", serde_json::json!({"text": "hi"})),
            text_response("done"),
        ]);
        let hooks: Box<dyn HookPort> = Box::new(OverrideHooks);
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(CountingTool(call_count.clone()))],
            Some(Arc::new(tokio::sync::Mutex::new(hooks))),
            "you are a test agent",
            test_ctx(),
        );

        let final_message = session.run_turn("do something").await.unwrap();
        match &final_message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "done"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the real tool must never run when a hook overrides the call"
        );

        // tool result message should carry the hook's mocked content, not
        // anything from the (never-run) real tool.
        let tool_result_msg = &session.messages[2];
        match &tool_result_msg.content[0] {
            ContentBlock::ToolResult(r) => match &r.content {
                ToolResultContent::Text(t) => assert_eq!(t, "mocked by hook"),
                other => panic!("expected Text, got {other:?}"),
            },
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    struct BudgetBlockingHooks;

    #[async_trait::async_trait]
    impl HookPort for BudgetBlockingHooks {
        async fn before_agent_start(&mut self, system_prompt: &str) -> HookDecision<String> {
            HookDecision::Allow(system_prompt.to_string())
        }
        async fn on_context(&mut self, messages: &[Message]) -> HookDecision<Vec<Message>> {
            HookDecision::Allow(messages.to_vec())
        }
        async fn on_tool_call(&mut self, call: &ToolCall) -> ToolCallDecision {
            ToolCallDecision::Allow(call.clone())
        }
        async fn on_tool_result(&mut self, result: &ToolResultInfo) -> HookDecision<String> {
            HookDecision::Allow(result.content.clone())
        }
        async fn before_compact(&mut self, _messages: &[Message]) -> HookDecision<()> {
            HookDecision::Allow(())
        }
        async fn on_budget(&mut self, _info: &crate::hooks::BudgetInfo) -> HookDecision<()> {
            HookDecision::Block("budget exceeded".to_string())
        }
    }

    #[tokio::test]
    async fn on_budget_block_stops_the_turn() {
        let provider = ScriptedProvider::new(vec![text_response("should never be returned")]);
        let hooks: Box<dyn HookPort> = Box::new(BudgetBlockingHooks);
        let mut session = AgentSession::new(
            Arc::new(provider),
            vec![],
            Some(Arc::new(tokio::sync::Mutex::new(hooks))),
            "test",
            test_ctx(),
        );

        let err = session.run_turn("do something").await.unwrap_err();
        assert!(matches!(err, AgentError::HookBlocked(reason) if reason == "budget exceeded"));
    }
}
