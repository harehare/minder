use std::sync::Arc;

use crate::hooks::{HookDecision, HookPort, ToolResultInfo};
use crate::message::{Message, StopReason, ToolCall, ToolResult, ToolResultContent, ToolSpec};
use crate::provider::{LlmProvider, ProviderError};
use crate::tool::{Tool, ToolContext, ToolExecOutcome, spec};

const COMPACT_THRESHOLD: usize = 60;
const KEEP_RECENT: usize = 40;

pub struct AgentSession {
    provider: Box<dyn LlmProvider>,
    tools: Vec<Box<dyn Tool>>,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
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
        provider: Box<dyn LlmProvider>,
        tools: Vec<Box<dyn Tool>>,
        hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
        system_prompt: impl Into<String>,
        tool_ctx: ToolContext,
    ) -> Self {
        Self {
            provider,
            tools,
            hooks,
            messages: Vec::new(),
            system_prompt: system_prompt.into(),
            tool_ctx,
            started: false,
        }
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

            let tool_calls: Vec<ToolCall> = response.message.tool_calls().cloned().collect();
            if tool_calls.is_empty() || response.stop_reason != StopReason::ToolUse {
                return Ok(response.message);
            }

            let mut results = Vec::with_capacity(tool_calls.len());
            for call in tool_calls {
                let outcome = self.execute_with_hooks(call.clone()).await?;
                results.push(ToolResult {
                    tool_call_id: call.id,
                    content: ToolResultContent::Text(outcome.content),
                    is_error: outcome.is_error,
                });
            }
            self.messages.push(Message::tool_results(results));
        }
    }

    async fn run_before_agent_start(&mut self) -> Result<String, AgentError> {
        let Some(hooks) = &self.hooks else {
            return Ok(self.system_prompt.clone());
        };
        match hooks
            .lock()
            .await
            .before_agent_start(&self.system_prompt)
            .await
        {
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

    async fn execute_with_hooks(&mut self, call: ToolCall) -> Result<ToolExecOutcome, AgentError> {
        let effective_call = if let Some(hooks) = &self.hooks {
            match hooks.lock().await.on_tool_call(&call).await {
                HookDecision::Allow(c) => c,
                HookDecision::Block(reason) => {
                    return Ok(ToolExecOutcome {
                        content: format!("Blocked by policy: {reason}"),
                        is_error: true,
                        metadata: serde_json::Value::Null,
                    });
                }
            }
        } else {
            call
        };

        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == effective_call.name)
            .ok_or_else(|| AgentError::UnknownTool(effective_call.name.clone()))?;
        let outcome = tool
            .execute(effective_call.arguments.clone(), &self.tool_ctx)
            .await;

        let Some(hooks) = &self.hooks else {
            return Ok(outcome);
        };
        let info = ToolResultInfo {
            tool_name: effective_call.name.clone(),
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
            Ok(self
                .0
                .lock()
                .unwrap()
                .pop_front()
                .expect("script exhausted"))
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
        async fn execute(
            &self,
            arguments: serde_json::Value,
            _ctx: &ToolContext,
        ) -> ToolExecOutcome {
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
            tool_use_response(
                "call_1",
                "echo",
                serde_json::json!({"text": "hi from tool"}),
            ),
            text_response("the tool said: hi from tool"),
        ]);
        let mut session = AgentSession::new(
            Box::new(provider),
            vec![Box::new(EchoTool)],
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
    async fn loop_terminates_immediately_with_no_tool_calls() {
        let provider = ScriptedProvider::new(vec![text_response("no tools needed")]);
        let mut session = AgentSession::new(
            Box::new(provider),
            vec![],
            None,
            "you are a test agent",
            test_ctx(),
        );

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
        let mut session = AgentSession::new(
            Box::new(provider),
            vec![],
            None,
            "you are a test agent",
            test_ctx(),
        );

        let err = session.run_turn("do something").await.unwrap_err();
        assert!(matches!(err, AgentError::UnknownTool(name) if name == "does_not_exist"));
    }
}
