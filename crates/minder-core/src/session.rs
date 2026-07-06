use std::sync::Arc;
use std::time::Duration;

use crate::hooks::{HookDecision, HookPort, ToolCallDecision, ToolResultInfo};
use crate::message::{
    ContentBlock, Message, ProviderResponse, StopReason, ToolCall, ToolResult, ToolResultContent, ToolSpec,
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
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),
    #[error("blocked by hook: {0}")]
    HookBlocked(String),
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
        }
    }

    /// Sets the reporter used to observe live progress (assistant text, tool
    /// calls, tool results) as a turn runs. Defaults to `NoopReporter`.
    pub fn with_reporter(mut self, reporter: Arc<dyn Reporter>) -> Self {
        self.reporter = reporter;
        self
    }

    /// Runs one user turn to completion (looping on tool calls as needed)
    /// and returns the final assistant message.
    pub async fn run_turn(&mut self, user_input: &str) -> Result<Message, AgentError> {
        if !self.started {
            self.system_prompt = self.run_before_agent_start().await?;
            self.started = true;
        }

        self.messages.push(Message::user_text(user_input));

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
            self.messages.push(response.message.clone());

            for block in &response.message.content {
                if let ContentBlock::Text(text) = block {
                    self.reporter.on_assistant_text(text).await;
                }
            }

            let tool_calls: Vec<ToolCall> = response.message.tool_calls().cloned().collect();
            if tool_calls.is_empty() || response.stop_reason != StopReason::ToolUse {
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
                let outcome = self.execute_with_hooks(call.clone()).await?;
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
                // Shared reborrow so these futures can run concurrently.
                let session = &*self;
                let futures = concurrent_indices.iter().map(|&i| {
                    let call = tool_calls[i].clone();
                    async move {
                        let outcome = session.execute_with_hooks(call.clone()).await?;
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
        let mut attempt = 0usize;
        loop {
            let result = self
                .provider
                .complete(messages, tool_specs, Some(&self.system_prompt))
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
        let over_message_count = self.messages.len() > COMPACT_THRESHOLD;
        let over_token_budget = self.last_input_tokens.is_some_and(|t| t > TOKEN_COMPACT_THRESHOLD);
        if !over_message_count && !over_token_budget {
            return Ok(());
        }
        self.run_before_compact_hook().await?;
        self.truncate_to(KEEP_RECENT);
        Ok(())
    }

    /// Emergency compaction after the provider itself rejects a request as too large.
    async fn force_compact(&mut self) -> Result<(), AgentError> {
        self.run_before_compact_hook().await?;
        self.truncate_to(EMERGENCY_KEEP_RECENT);
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

    // Truncation-based compaction: keep only the most recent `keep` messages.
    // Real summarization is a v2 concern (see plan's Compaction hook
    // semantics open question).
    fn truncate_to(&mut self, keep: usize) {
        if self.messages.len() <= keep {
            return;
        }
        let drop_count = self.messages.len() - keep;
        self.messages.drain(0..drop_count);
    }

    async fn execute_with_hooks(&self, call: ToolCall) -> Result<ToolExecOutcome, AgentError> {
        let decision = if let Some(hooks) = &self.hooks {
            hooks.lock().await.on_tool_call(&call).await
        } else {
            ToolCallDecision::Allow(call.clone())
        };

        match decision {
            ToolCallDecision::Allow(effective_call) => {
                let outcome = self.execute_tool(&effective_call).await;
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
    async fn execute_tool(&self, call: &ToolCall) -> ToolExecOutcome {
        match self.tools.iter().find(|t| t.name() == call.name) {
            Some(tool) => tool.execute(call.arguments.clone(), &self.tool_ctx).await,
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

    /// The active provider's id (e.g. `"anthropic"`), for display purposes
    /// (banner, status lines) -- not used for any routing decision.
    pub fn provider_id(&self) -> &'static str {
        self.provider.id()
    }

    /// Loads a saved transcript and marks the session started, so
    /// `before_agent_start` won't re-run. Used to resume a prior session.
    pub fn restore(&mut self, system_prompt: String, messages: Vec<Message>) {
        self.system_prompt = system_prompt;
        self.messages = messages;
        self.started = true;
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

    fn test_ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
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

    struct FlakyThenOkProvider {
        calls: StdMutex<usize>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for FlakyThenOkProvider {
        fn id(&self) -> &'static str {
            "flaky"
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

    #[tokio::test]
    async fn high_token_usage_triggers_proactive_compaction_under_message_threshold() {
        let provider = ScriptedProvider::new(vec![
            text_response_with_usage("first", TOKEN_COMPACT_THRESHOLD + 1),
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
}
