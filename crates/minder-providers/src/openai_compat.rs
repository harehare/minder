use std::time::Duration;

use async_trait::async_trait;
use minder_core::{
    ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Role, StopReason, ToolCall,
    ToolResultContent, ToolSpec, Usage,
};
use serde::{Deserialize, Serialize};

/// Generic client for any server speaking the OpenAI chat-completions wire
/// format -- llama.cpp's `llama-server`, LM Studio, vLLM, LocalAI, and
/// anything else that treats it as the local-LLM lingua franca. Unlike
/// `OllamaProvider`, this has no native equivalent to lean on: no `/api/tags`
/// (`list_models` stays the trait default), no context-window introspection,
/// no pull endpoint.
pub struct OpenAiCompatProvider {
    base_url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl OpenAiCompatProvider {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            client: Self::build_client(None),
        }
    }

    /// Most local servers need no key; set one for a remote/gated deployment.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Overrides the default request timeout -- see `crate::http`.
    pub fn with_request_timeout_secs(mut self, secs: u64) -> Self {
        self.client = Self::build_client(Some(secs));
        self
    }

    fn build_client(request_timeout_secs: Option<u64>) -> reqwest::Client {
        crate::http::client_builder(request_timeout_secs)
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .expect("reqwest client config is static and valid")
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatProvider {
    fn id(&self) -> &'static str {
        "openai-compat"
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
        let mut oc_messages = Vec::new();
        if let Some(sp) = system_prompt {
            oc_messages.push(OcMessage {
                role: "system".to_string(),
                content: Some(sp.to_string()),
                tool_calls: vec![],
                tool_call_id: None,
            });
        }
        oc_messages.extend(to_oc_messages(messages));

        let body = OcRequest {
            model: self.model.clone(),
            messages: oc_messages,
            tools: to_oc_tools(tools),
            stream: false,
        };

        let mut req = self.client.post(format!("{}/chat/completions", self.base_url)).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| crate::http::describe_transport_error(e, &self.base_url))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| crate::http::describe_transport_error(e, &self.base_url))?;

        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: OcResponse = serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        from_oc_response(parsed)
    }
}

// -- wire format --
// OpenAI's chat-completions shape: `function.arguments` is a JSON-encoded
// *string*, not a raw object like Ollama's native API -- see `to_oc_messages`
// and `from_oc_response`.

#[derive(Serialize)]
struct OcRequest {
    model: String,
    messages: Vec<OcMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OcTool>,
    stream: bool,
}

#[derive(Serialize)]
struct OcMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OcToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct OcToolCall {
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default = "function_type")]
    kind: String,
    function: OcFunctionCall,
}

fn function_type() -> String {
    "function".to_string()
}

#[derive(Serialize, Deserialize)]
struct OcFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OcTool {
    #[serde(rename = "type")]
    kind: String,
    function: OcFunctionSpec,
}

#[derive(Serialize)]
struct OcFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct OcResponse {
    choices: Vec<OcChoice>,
    #[serde(default)]
    usage: Option<OcUsage>,
}

#[derive(Deserialize)]
struct OcChoice {
    message: OcResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OcResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OcToolCall>,
}

#[derive(Deserialize)]
struct OcUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

fn to_oc_messages(messages: &[Message]) -> Vec<OcMessage> {
    messages
        .iter()
        .flat_map(|m| match m.role {
            Role::User | Role::Assistant => {
                let text = m
                    .content
                    .iter()
                    .filter_map(|b| if let ContentBlock::Text(t) = b { Some(t.as_str()) } else { None })
                    .collect::<Vec<_>>()
                    .join("\n");
                let tool_calls: Vec<OcToolCall> = m
                    .tool_calls()
                    .map(|c| OcToolCall {
                        id: c.id.clone(),
                        kind: "function".to_string(),
                        function: OcFunctionCall {
                            name: c.name.clone(),
                            arguments: c.arguments.to_string(),
                        },
                    })
                    .collect();
                vec![OcMessage {
                    role: if m.role == Role::User { "user" } else { "assistant" }.to_string(),
                    content: (!text.is_empty()).then_some(text),
                    tool_calls,
                    tool_call_id: None,
                }]
            }
            Role::Tool => m
                .content
                .iter()
                .filter_map(|b| if let ContentBlock::ToolResult(r) = b { Some(r) } else { None })
                .map(|r| OcMessage {
                    role: "tool".to_string(),
                    content: Some(match &r.content {
                        ToolResultContent::Text(t) => t.clone(),
                        ToolResultContent::Blocks(b) => serde_json::to_string(b).unwrap_or_default(),
                    }),
                    tool_calls: vec![],
                    tool_call_id: Some(r.tool_call_id.clone()),
                })
                .collect(),
            Role::System => vec![],
        })
        .collect()
}

fn to_oc_tools(tools: &[ToolSpec]) -> Vec<OcTool> {
    tools
        .iter()
        .map(|t| OcTool {
            kind: "function".to_string(),
            function: OcFunctionSpec {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect()
}

fn from_oc_response(resp: OcResponse) -> Result<ProviderResponse, ProviderError> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ProviderError::Deserialize("response had no choices".to_string()))?;

    let mut content = Vec::new();
    if let Some(text) = choice.message.content
        && !text.is_empty()
    {
        content.push(ContentBlock::Text(text));
    }
    let has_tool_calls = !choice.message.tool_calls.is_empty();
    for tc in choice.message.tool_calls {
        // A malformed arguments string becomes `null` rather than failing
        // the whole turn -- the model gets a chance to correct itself.
        let arguments = serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
        content.push(ContentBlock::ToolUse(ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments,
        }));
    }

    let stop_reason = if has_tool_calls {
        StopReason::ToolUse
    } else if choice.finish_reason.as_deref() == Some("length") {
        StopReason::MaxTokens
    } else {
        StopReason::EndTurn
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
    use minder_core::ToolSpec;

    #[tokio::test]
    async fn complete_sends_openai_shaped_tools_and_parses_a_tool_call_response() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "tools": [{"type": "function", "function": {"name": "bash", "description": "runs a command", "parameters": {"type": "object"}}}],
            })))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(server.uri(), "test-model");
        let tools = vec![ToolSpec {
            name: "bash".to_string(),
            description: "runs a command".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let resp = provider.complete(&[Message::user_text("list files")], &tools, None).await.unwrap();

        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[tokio::test]
    async fn complete_parses_a_plain_text_response() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "42"},
                    "finish_reason": "stop",
                }],
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(server.uri(), "test-model");
        let resp = provider.complete(&[Message::user_text("what's 6*7")], &[], None).await.unwrap();

        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.message.text(), "42");
    }

    #[tokio::test]
    async fn complete_uses_the_bearer_token_when_an_api_key_is_set() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .and(wiremock::matchers::header("authorization", "Bearer secret-key"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(server.uri(), "test-model").with_api_key("secret-key");
        provider.complete(&[Message::user_text("hi")], &[], None).await.unwrap();
    }

    #[tokio::test]
    async fn a_non_success_status_becomes_an_api_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(wiremock::ResponseTemplate::new(404).set_body_string("model not found"))
            .mount(&server)
            .await;

        let provider = OpenAiCompatProvider::new(server.uri(), "test-model");
        let err = provider.complete(&[Message::user_text("hi")], &[], None).await.unwrap_err();
        assert!(matches!(err, ProviderError::Api { status: 404, .. }), "{err}");
    }
}
