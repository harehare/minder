use std::sync::Arc;

use crate::hooks::{HookDecision, HookPort, ToolCallDecision, ToolResultInfo};
use crate::message::{ContentBlock, Message, StopReason, ToolCall, ToolResult, ToolResultContent, ToolSpec};
use crate::provider::{LlmProvider, ProviderError};
use crate::reporter::{NoopReporter, Reporter};
use crate::tool::{Tool, ToolContext, ToolExecOutcome, spec};

const COMPACT_THRESHOLD: usize = 60;
const KEEP_RECENT: usize = 40;

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
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),
    #[error("blocked by hook: {0}")]
    HookBlocked(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
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
            let response = self
                .provider
                .complete(&outgoing, &tool_specs, Some(&self.system_prompt))
                .await?;
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
        if self.messages.len() <= COMPACT_THRESHOLD {
            return Ok(());
        }
        if let Some(hooks) = &self.hooks {
            match hooks.lock().await.before_compact(&self.messages).await {
                HookDecision::Block(reason) => return Err(AgentError::HookBlocked(reason)),
                HookDecision::Allow(()) => {}
            }
        }
        // Truncation-based compaction: keep only the most recent messages.
        // Real summarization is a v2 concern (see plan's Compaction hook
        // semantics open question).
        let drop_count = self.messages.len() - KEEP_RECENT;
        self.messages.drain(0..drop_count);
        Ok(())
    }

    async fn execute_with_hooks(&self, call: ToolCall) -> Result<ToolExecOutcome, AgentError> {
        let decision = if let Some(hooks) = &self.hooks {
            hooks.lock().await.on_tool_call(&call).await
        } else {
            ToolCallDecision::Allow(call.clone())
        };

        match decision {
            ToolCallDecision::Allow(effective_call) => {
                let outcome = self.execute_tool(&effective_call).await?;
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

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolExecOutcome, AgentError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == call.name)
            .ok_or_else(|| AgentError::UnknownTool(call.name.clone()))?;
        Ok(tool.execute(call.arguments.clone(), &self.tool_ctx).await)
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
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(text.to_string())],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
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
    async fn unknown_tool_call_is_an_error() {
        let provider = ScriptedProvider::new(vec![tool_use_response(
            "call_1",
            "does_not_exist",
            serde_json::json!({}),
        )]);
        let mut session = AgentSession::new(Arc::new(provider), vec![], None, "you are a test agent", test_ctx());

        let err = session.run_turn("do something").await.unwrap_err();
        assert!(matches!(err, AgentError::UnknownTool(name) if name == "does_not_exist"));
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
