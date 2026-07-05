//! Proves the MVP claim end to end: a real `.mq` hook script, loaded by
//! `HookEngine`, actually blocks a tool call driven through the real
//! `AgentSession` loop -- not just `HookEngine` in isolation.

use async_trait::async_trait;
use minder_core::{
    AgentSession, ContentBlock, HookPort, LlmProvider, Message, ProviderError, ProviderResponse, Role, StopReason,
    Tool, ToolCall, ToolContext, ToolExecOutcome, ToolSpec, Usage,
};
use minder_hooks::HookEngine;
use std::sync::{Arc, Mutex as StdMutex};

const SECURITY_HOOK: &str = r#"
def on_tool_call(call):
  if (call["name"] == "bash" && contains(call["arguments"]["command"], "rm -rf")):
    {"action": "block", "reason": "destructive bash command blocked by policy"}
  else:
    {"action": "allow", "value": call};
"#;

struct ScriptedProvider(StdMutex<std::collections::VecDeque<ProviderResponse>>);

#[async_trait]
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

struct FakeBashTool;

#[async_trait]
impl Tool for FakeBashTool {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> &str {
        "fake bash for testing -- never actually runs commands"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}})
    }
    async fn execute(&self, _arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        ToolExecOutcome {
            content: "SHOULD NOT HAVE RUN".to_string(),
            is_error: false,
            metadata: serde_json::Value::Null,
        }
    }
}

fn tool_use_response(id: &str, command: &str) -> ProviderResponse {
    ProviderResponse {
        message: Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse(ToolCall {
                id: id.to_string(),
                name: "bash".to_string(),
                arguments: serde_json::json!({"command": command}),
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

#[tokio::test]
async fn hook_script_blocks_destructive_command_through_the_real_agent_loop() {
    let agent_dir = std::env::temp_dir().join(format!("minder-session-integration-{}", uuid::Uuid::new_v4()));
    let hooks_dir = agent_dir.join("hooks");
    std::fs::create_dir_all(&hooks_dir).unwrap();
    std::fs::write(hooks_dir.join("security.mq"), SECURITY_HOOK).unwrap();

    let hook_engine = HookEngine::load(&agent_dir).unwrap().expect("hooks should load");
    let hooks: Box<dyn HookPort> = Box::new(hook_engine);

    let provider = ScriptedProvider(StdMutex::new(
        vec![
            tool_use_response("call_1", "rm -rf /important/data"),
            text_response("I wasn't able to run that command."),
        ]
        .into(),
    ));

    let tool_ctx = ToolContext {
        working_dir: std::env::temp_dir(),
        session_id: "test".to_string(),
        cancel: tokio_util::sync::CancellationToken::new(),
    };

    let mut session = AgentSession::new(
        Box::new(provider),
        vec![Box::new(FakeBashTool)],
        Some(Arc::new(tokio::sync::Mutex::new(hooks))),
        "you are a test agent",
        tool_ctx,
    );

    let final_message = session.run_turn("delete everything").await.unwrap();

    // the tool must never have actually run
    match &final_message.content[0] {
        ContentBlock::Text(t) => assert_eq!(t, "I wasn't able to run that command."),
        other => panic!("expected final Text response, got {other:?}"),
    }
}

#[tokio::test]
async fn hook_script_allows_safe_command_through_the_real_agent_loop() {
    let agent_dir = std::env::temp_dir().join(format!("minder-session-integration-{}", uuid::Uuid::new_v4()));
    let hooks_dir = agent_dir.join("hooks");
    std::fs::create_dir_all(&hooks_dir).unwrap();
    std::fs::write(hooks_dir.join("security.mq"), SECURITY_HOOK).unwrap();

    let hook_engine = HookEngine::load(&agent_dir).unwrap().expect("hooks should load");
    let hooks: Box<dyn HookPort> = Box::new(hook_engine);

    let provider = ScriptedProvider(StdMutex::new(
        vec![
            tool_use_response("call_1", "ls -la"),
            text_response("Here are the files."),
        ]
        .into(),
    ));

    let tool_ctx = ToolContext {
        working_dir: std::env::temp_dir(),
        session_id: "test".to_string(),
        cancel: tokio_util::sync::CancellationToken::new(),
    };

    let mut session = AgentSession::new(
        Box::new(provider),
        vec![Box::new(FakeBashTool)],
        Some(Arc::new(tokio::sync::Mutex::new(hooks))),
        "you are a test agent",
        tool_ctx,
    );

    let final_message = session.run_turn("list files").await.unwrap();
    match &final_message.content[0] {
        ContentBlock::Text(t) => assert_eq!(t, "Here are the files."),
        other => panic!("expected final Text response, got {other:?}"),
    }
}
