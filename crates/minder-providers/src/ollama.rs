use async_trait::async_trait;
use minder_core::{
    ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Role, StopReason, ToolCall, ToolResultContent,
    ToolSpec, Usage,
};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";
pub struct OllamaProvider {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    fn id(&self) -> &'static str {
        "ollama"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
    ) -> Result<ProviderResponse, ProviderError> {
        let mut ol_messages = Vec::new();
        if let Some(sp) = system_prompt {
            ol_messages.push(OlMessage {
                role: "system".to_string(),
                content: sp.to_string(),
                tool_calls: vec![],
            });
        }
        ol_messages.extend(to_ollama_messages(messages));

        let body = OlRequest {
            model: self.model.clone(),
            messages: ol_messages,
            tools: to_ollama_tools(tools),
            stream: false,
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| ProviderError::Transport(e.to_string()))?;

        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: OlResponse = serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        Ok(from_ollama_response(parsed))
    }
}

// -- wire format --

#[derive(Serialize)]
struct OlRequest {
    model: String,
    messages: Vec<OlMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OlTool>,
    stream: bool,
}

#[derive(Serialize, Debug)]
struct OlMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OlToolCall>,
}

#[derive(Serialize, Deserialize, Debug)]
struct OlToolCall {
    function: OlFunctionCall,
}

#[derive(Serialize, Deserialize, Debug)]
struct OlFunctionCall {
    name: String,
    /// Unlike OpenAI, Ollama's native API sends/expects a JSON object here,
    /// not a stringified JSON blob.
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct OlTool {
    #[serde(rename = "type")]
    kind: String,
    function: OlFunctionSpec,
}

#[derive(Serialize)]
struct OlFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct OlResponse {
    message: OlResponseMessage,
    #[serde(default)]
    done: bool,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OlResponseMessage {
    content: String,
    #[serde(default)]
    tool_calls: Vec<OlToolCall>,
}

// -- mapping --

/// Ollama's native API has no `tool_call_id` field on "tool" messages --
/// results are matched to calls positionally, not by id. So `Role::Tool`
/// messages just become `role: "tool"` messages carrying content, and any
/// `tool_call_id` on the `ToolResult` is simply dropped on the way out.
fn to_ollama_messages(messages: &[Message]) -> Vec<OlMessage> {
    messages
        .iter()
        .filter(|m| m.role != Role::System)
        .flat_map(|m| match m.role {
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
                let tool_calls: Vec<OlToolCall> = m
                    .tool_calls()
                    .map(|c| OlToolCall {
                        function: OlFunctionCall {
                            name: c.name.clone(),
                            arguments: c.arguments.clone(),
                        },
                    })
                    .collect();
                vec![OlMessage {
                    role: if m.role == Role::User { "user" } else { "assistant" }.to_string(),
                    content: text,
                    tool_calls,
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
                .map(|r| OlMessage {
                    role: "tool".to_string(),
                    content: match &r.content {
                        ToolResultContent::Text(t) => t.clone(),
                        ToolResultContent::Blocks(b) => serde_json::to_string(b).unwrap_or_default(),
                    },
                    tool_calls: vec![],
                })
                .collect(),
            Role::System => vec![],
        })
        .collect()
}

fn to_ollama_tools(tools: &[ToolSpec]) -> Vec<OlTool> {
    tools
        .iter()
        .map(|t| OlTool {
            kind: "function".to_string(),
            function: OlFunctionSpec {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect()
}

fn from_ollama_response(resp: OlResponse) -> ProviderResponse {
    let mut content = Vec::new();
    if !resp.message.content.is_empty() {
        content.push(ContentBlock::Text(resp.message.content));
    }
    let has_tool_calls = !resp.message.tool_calls.is_empty();
    for tc in resp.message.tool_calls {
        // Ollama doesn't send an id at all; synthesize one.
        content.push(ContentBlock::ToolUse(ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4()),
            name: tc.function.name,
            arguments: tc.function.arguments,
        }));
    }

    let stop_reason = if has_tool_calls {
        StopReason::ToolUse
    } else if resp.done {
        StopReason::EndTurn
    } else {
        StopReason::Other
    };

    ProviderResponse {
        message: Message {
            role: Role::Assistant,
            content,
            metadata: serde_json::Value::Null,
        },
        stop_reason,
        usage: Usage {
            input_tokens: resp.prompt_eval_count.unwrap_or(0),
            output_tokens: resp.eval_count.unwrap_or(0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::ToolResult;

    #[test]
    fn parses_text_only_response() {
        let raw = r#"{"message": {"role": "assistant", "content": "hello there"}, "done": true, "prompt_eval_count": 10, "eval_count": 3}"#;
        let parsed: OlResponse = serde_json::from_str(raw).unwrap();
        let resp = from_ollama_response(parsed);
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "hello there"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn parses_tool_call_and_synthesizes_id() {
        let raw = r#"{
            "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}]},
            "done": false
        }"#;
        let parsed: OlResponse = serde_json::from_str(raw).unwrap();
        let resp = from_ollama_response(parsed);
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls[0].name, "bash");
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn tool_result_becomes_tool_role_message() {
        let messages = vec![Message::tool_results(vec![ToolResult {
            tool_call_id: "whatever".into(),
            content: ToolResultContent::Text("output".into()),
            is_error: false,
        }])];
        let mapped = to_ollama_messages(&messages);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].role, "tool");
        assert_eq!(mapped[0].content, "output");
    }

    #[tokio::test]
    #[ignore]
    async fn live_round_trip() {
        let provider = OllamaProvider::new("llama3.2");
        let messages = vec![Message::user_text("Say hello in exactly three words.")];
        let resp = provider
            .complete(&messages, &[], None)
            .await
            .expect("live call to local Ollama server failed -- is `ollama serve` running?");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }
}
