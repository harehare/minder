use agent_core::{
    ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Role, StopReason,
    ToolCall, ToolResultContent, ToolSpec, Usage,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";
const DEFAULT_MAX_TOKENS: u32 = 8192;

pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiProvider {
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
impl LlmProvider for OpenAiProvider {
    fn id(&self) -> &'static str {
        "openai"
    }

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
    ) -> Result<ProviderResponse, ProviderError> {
        let mut oa_messages = Vec::new();
        if let Some(sp) = system_prompt {
            oa_messages.push(OaMessage {
                role: "system".to_string(),
                content: Some(sp.to_string()),
                tool_calls: vec![],
                tool_call_id: None,
            });
        }
        oa_messages.extend(to_openai_messages(messages));

        let body = OaRequest {
            model: self.model.clone(),
            max_completion_tokens: DEFAULT_MAX_TOKENS,
            messages: oa_messages,
            tools: to_openai_tools(tools),
        };

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
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

        let parsed: OaResponse =
            serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        from_openai_response(parsed)
    }
}

// -- wire format --

#[derive(Serialize)]
struct OaRequest {
    model: String,
    max_completion_tokens: u32,
    messages: Vec<OaMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OaTool>,
}

#[derive(Serialize, Debug)]
struct OaMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OaToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OaToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: OaFunctionCall,
}

#[derive(Serialize, Deserialize, Debug)]
struct OaFunctionCall {
    name: String,
    arguments: String, // JSON-encoded string, not a nested object
}

#[derive(Serialize)]
struct OaTool {
    #[serde(rename = "type")]
    kind: String,
    function: OaFunctionSpec,
}

#[derive(Serialize)]
struct OaFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct OaResponse {
    choices: Vec<OaChoice>,
    usage: Option<OaUsage>,
}

#[derive(Deserialize)]
struct OaChoice {
    message: OaResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OaResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OaToolCall>,
}

#[derive(Deserialize)]
struct OaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// -- mapping --

fn to_openai_messages(messages: &[Message]) -> Vec<OaMessage> {
    messages.iter().flat_map(to_openai_message_group).collect()
}

/// One `agent_core::Message` can expand to multiple OpenAI messages: a
/// `Role::Tool` message (one or more `ToolResult` blocks) becomes one
/// OpenAI `role: "tool"` message *per result*, since OpenAI has no way to
/// carry multiple tool results in a single message.
fn to_openai_message_group(m: &Message) -> Vec<OaMessage> {
    match m.role {
        Role::System => vec![], // handled separately via the top-level system message
        Role::User | Role::Assistant => {
            let text = m
                .content
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::Text(t) = b {
                        Some(t.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let tool_calls: Vec<OaToolCall> = m
                .tool_calls()
                .map(|c| OaToolCall {
                    id: c.id.clone(),
                    kind: "function".to_string(),
                    function: OaFunctionCall {
                        name: c.name.clone(),
                        arguments: c.arguments.to_string(),
                    },
                })
                .collect();
            vec![OaMessage {
                role: if m.role == Role::User {
                    "user"
                } else {
                    "assistant"
                }
                .to_string(),
                content: if text.is_empty() { None } else { Some(text) },
                tool_calls,
                tool_call_id: None,
            }]
        }
        Role::Tool => m
            .content
            .iter()
            .filter_map(|b| {
                if let ContentBlock::ToolResult(r) = b {
                    Some(r)
                } else {
                    None
                }
            })
            .map(|r| OaMessage {
                role: "tool".to_string(),
                content: Some(match &r.content {
                    ToolResultContent::Text(t) => t.clone(),
                    ToolResultContent::Blocks(b) => serde_json::to_string(b).unwrap_or_default(),
                }),
                tool_calls: vec![],
                tool_call_id: Some(r.tool_call_id.clone()),
            })
            .collect(),
    }
}

fn to_openai_tools(tools: &[ToolSpec]) -> Vec<OaTool> {
    tools
        .iter()
        .map(|t| OaTool {
            kind: "function".to_string(),
            function: OaFunctionSpec {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect()
}

fn from_openai_response(resp: OaResponse) -> Result<ProviderResponse, ProviderError> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ProviderError::Deserialize("no choices in response".to_string()))?;

    let mut content = Vec::new();
    if let Some(text) = choice.message.content {
        content.push(ContentBlock::Text(text));
    }
    let has_tool_calls = !choice.message.tool_calls.is_empty();
    for tc in choice.message.tool_calls {
        let arguments =
            serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
        content.push(ContentBlock::ToolUse(ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments,
        }));
    }

    let stop_reason = match (choice.finish_reason.as_deref(), has_tool_calls) {
        (_, true) => StopReason::ToolUse,
        (Some("stop"), _) => StopReason::EndTurn,
        (Some("length"), _) => StopReason::MaxTokens,
        _ => StopReason::Other,
    };

    Ok(ProviderResponse {
        message: Message {
            role: Role::Assistant,
            content,
            metadata: serde_json::Value::Null,
        },
        stop_reason,
        usage: resp
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            })
            .unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::ToolResult;

    #[test]
    fn parses_text_only_response() {
        let raw = r#"{
            "choices": [{"message": {"content": "hello there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}
        }"#;
        let parsed: OaResponse = serde_json::from_str(raw).unwrap();
        let resp = from_openai_response(parsed).unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "hello there"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn parses_tool_call_response() {
        let raw = r#"{
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8}
        }"#;
        let parsed: OaResponse = serde_json::from_str(raw).unwrap();
        let resp = from_openai_response(parsed).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn tool_result_message_becomes_tool_role_with_call_id() {
        let messages = vec![Message::tool_results(vec![ToolResult {
            tool_call_id: "call_1".into(),
            content: ToolResultContent::Text("output".into()),
            is_error: false,
        }])];
        let mapped = to_openai_messages(&messages);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].role, "tool");
        assert_eq!(mapped[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(mapped[0].content.as_deref(), Some("output"));
    }

    #[test]
    fn assistant_tool_use_becomes_tool_calls_with_stringified_arguments() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse(ToolCall {
                id: "call_1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "ls"}),
            })],
            metadata: serde_json::Value::Null,
        }];
        let mapped = to_openai_messages(&messages);
        assert_eq!(
            mapped[0].tool_calls[0].function.arguments,
            r#"{"command":"ls"}"#
        );
    }

    #[tokio::test]
    #[ignore]
    async fn live_round_trip() {
        let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY to run this test");
        let provider = OpenAiProvider::new(api_key, "gpt-4o-mini");
        let messages = vec![Message::user_text("Say hello in exactly three words.")];
        let resp = provider
            .complete(&messages, &[], None)
            .await
            .expect("live API call failed");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }
}
