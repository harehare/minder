use agent_core::{
    ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Role, StopReason,
    ToolCall, ToolResult, ToolResultContent, ToolSpec, Usage,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8192;

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
    ) -> Result<ProviderResponse, ProviderError> {
        let body = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: DEFAULT_MAX_TOKENS,
            system: system_prompt.map(str::to_string),
            messages: to_anthropic_messages(messages),
            tools: to_anthropic_tools(tools),
        };

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ProviderError::RateLimited {
                retry_after_secs: None,
            });
        }
        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: AnthropicResponse =
            serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;

        Ok(from_anthropic_response(parsed))
    }
}

// -- wire format --

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// -- mapping --

/// Anthropic has no dedicated "tool" role: tool_result blocks travel inside
/// a "user" message replying to the prior assistant turn.
fn to_anthropic_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    messages
        .iter()
        .filter(|m| m.role != Role::System) // system prompt goes in the top-level `system` field
        .map(|m| AnthropicMessage {
            role: match m.role {
                Role::User | Role::Tool => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
                Role::System => unreachable!("filtered above"),
            },
            content: m.content.iter().map(to_anthropic_block).collect(),
        })
        .collect()
}

fn to_anthropic_block(block: &ContentBlock) -> AnthropicContentBlock {
    match block {
        ContentBlock::Text(text) => AnthropicContentBlock::Text { text: text.clone() },
        ContentBlock::ToolUse(call) => AnthropicContentBlock::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.arguments.clone(),
        },
        ContentBlock::ToolResult(result) => AnthropicContentBlock::ToolResult {
            tool_use_id: result.tool_call_id.clone(),
            content: match &result.content {
                ToolResultContent::Text(t) => t.clone(),
                ToolResultContent::Blocks(b) => serde_json::to_string(b).unwrap_or_default(),
            },
            is_error: result.is_error.then_some(true),
        },
        ContentBlock::Thinking { text, signature } => AnthropicContentBlock::Thinking {
            thinking: text.clone(),
            signature: signature.clone(),
        },
    }
}

fn to_anthropic_tools(tools: &[ToolSpec]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|t| AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.parameters.clone(),
        })
        .collect()
}

fn from_anthropic_response(resp: AnthropicResponse) -> ProviderResponse {
    let has_tool_use = resp
        .content
        .iter()
        .any(|b| matches!(b, AnthropicContentBlock::ToolUse { .. }));

    let content = resp
        .content
        .into_iter()
        .map(|b| match b {
            AnthropicContentBlock::Text { text } => ContentBlock::Text(text),
            AnthropicContentBlock::ToolUse { id, name, input } => ContentBlock::ToolUse(ToolCall {
                id,
                name,
                arguments: input,
            }),
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => ContentBlock::ToolResult(ToolResult {
                tool_call_id: tool_use_id,
                content: ToolResultContent::Text(content),
                is_error: is_error.unwrap_or(false),
            }),
            AnthropicContentBlock::Thinking {
                thinking,
                signature,
            } => ContentBlock::Thinking {
                text: thinking,
                signature,
            },
        })
        .collect();

    let stop_reason = match (resp.stop_reason.as_deref(), has_tool_use) {
        (_, true) => StopReason::ToolUse,
        (Some("end_turn") | Some("stop_sequence"), _) => StopReason::EndTurn,
        (Some("max_tokens"), _) => StopReason::MaxTokens,
        _ => StopReason::Other,
    };

    ProviderResponse {
        message: Message {
            role: Role::Assistant,
            content,
            metadata: serde_json::Value::Null,
        },
        stop_reason,
        usage: Usage {
            input_tokens: resp.usage.input_tokens,
            output_tokens: resp.usage.output_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real Anthropic Messages API response shapes, used as fixtures so the
    // mapping logic is verified without hitting the network.

    #[test]
    fn parses_text_only_response() {
        let raw = r#"{
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "hello there friend"}],
            "model": "claude-sonnet-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4}
        }"#;
        let parsed: AnthropicResponse = serde_json::from_str(raw).unwrap();
        let resp = from_anthropic_response(parsed);

        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 4);
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "hello there friend"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn parses_tool_use_response_as_stop_reason_tool_use() {
        let raw = r#"{
            "id": "msg_2", "type": "message", "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check that."},
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}}
            ],
            "model": "claude-sonnet-5",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 8}
        }"#;
        let parsed: AnthropicResponse = serde_json::from_str(raw).unwrap();
        let resp = from_anthropic_response(parsed);

        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn tool_role_message_becomes_user_role_with_tool_result_block() {
        let messages = vec![Message::tool_results(vec![ToolResult {
            tool_call_id: "toolu_1".into(),
            content: ToolResultContent::Text("file1\nfile2".into()),
            is_error: false,
        }])];
        let mapped = to_anthropic_messages(&messages);

        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].role, "user");
        match &mapped[0].content[0] {
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "toolu_1");
                assert_eq!(content, "file1\nfile2");
                assert_eq!(*is_error, None);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn system_role_message_is_excluded_from_messages_array() {
        let messages = vec![
            Message {
                role: Role::System,
                content: vec![ContentBlock::Text("ignored".into())],
                metadata: serde_json::Value::Null,
            },
            Message::user_text("hi"),
        ];
        let mapped = to_anthropic_messages(&messages);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].role, "user");
    }

    /// Live smoke test against the real API. Requires ANTHROPIC_API_KEY;
    /// ignored by default so CI doesn't need a key.
    #[tokio::test]
    #[ignore]
    async fn live_round_trip() {
        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY to run this test");
        let provider = AnthropicProvider::new(api_key, "claude-sonnet-5");
        let messages = vec![Message::user_text("Say hello in exactly three words.")];
        let resp = provider
            .complete(&messages, &[], None)
            .await
            .expect("live API call failed");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert!(!resp.message.content.is_empty());
    }
}
