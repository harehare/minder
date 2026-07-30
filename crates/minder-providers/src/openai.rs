use async_trait::async_trait;
use futures_util::StreamExt;
use minder_core::{
    ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Reporter, Role, StopReason, ToolCall,
    ToolResultContent, ToolSpec, Usage,
};
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
            client: crate::http::client_builder(None)
                .build()
                .expect("reqwest client config is static and valid"),
        }
    }

    /// Overrides the default request timeout -- see `crate::http`.
    pub fn with_request_timeout_secs(mut self, secs: u64) -> Self {
        self.client = crate::http::client_builder(Some(secs))
            .build()
            .expect("reqwest client config is static and valid");
        self
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn id(&self) -> &'static str {
        "openai"
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
            stream: false,
            stream_options: None,
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
        let text = resp.text().await.map_err(|e| ProviderError::Transport(e.to_string()))?;

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ProviderError::RateLimited { retry_after_secs: None });
        }
        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: OaResponse = serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        from_openai_response(parsed)
    }

    async fn complete_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
        reporter: &dyn Reporter,
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
            stream: true,
            stream_options: Some(OaStreamOptions { include_usage: true }),
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
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ProviderError::RateLimited { retry_after_secs: None });
        }
        if !status.is_success() {
            let text = resp.text().await.map_err(|e| ProviderError::Transport(e.to_string()))?;
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let mut stream = resp.bytes_stream();
        let mut line_buf: Vec<u8> = Vec::new();
        let mut accum = OaStreamAccumulator::default();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ProviderError::Transport(e.to_string()))?;
            line_buf.extend_from_slice(&chunk);

            while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = line_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = line.trim_end().strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    return from_openai_response(accum.into_response());
                }
                accum.handle_chunk(data, reporter).await?;
            }
        }

        from_openai_response(accum.into_response())
    }
}

/// Mirrors `chat.completions`' SSE chunk shape -- separate from `OaResponse`
/// since streaming has no top-level `choices[].message`, only a per-chunk
/// `delta`, and usage arrives in its own final chunk (via `stream_options`).
#[derive(Deserialize)]
struct OaChunk {
    #[serde(default)]
    choices: Vec<OaChunkChoice>,
    usage: Option<OaUsage>,
}

#[derive(Deserialize)]
struct OaChunkChoice {
    delta: OaDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct OaDelta {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OaToolCallDelta>,
}

#[derive(Deserialize)]
struct OaToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<OaFunctionDelta>,
}

#[derive(Deserialize)]
struct OaFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

/// Assembles chunks into the same shape `from_openai_response` maps from a
/// single non-streaming response, so both paths share that logic. Tool
/// calls are keyed by their `index` within `delta.tool_calls` (which model
/// slot they belong to, not the SSE event's position) -- growing a `Vec`
/// indexed directly by it keeps out-of-order delivery from mixing calls up.
#[derive(Default)]
struct OaStreamAccumulator {
    text: String,
    saw_text: bool,
    tool_calls: Vec<Option<PendingToolCall>>,
    finish_reason: Option<String>,
    usage: Option<OaUsage>,
}

#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl OaStreamAccumulator {
    async fn handle_chunk(&mut self, data: &str, reporter: &dyn Reporter) -> Result<(), ProviderError> {
        let chunk: OaChunk = serde_json::from_str(data).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        let Some(choice) = chunk.choices.into_iter().next() else {
            return Ok(());
        };
        if let Some(reason) = choice.finish_reason {
            self.finish_reason = Some(reason);
        }
        if let Some(text) = choice.delta.content
            && !text.is_empty()
        {
            self.saw_text = true;
            self.text.push_str(&text);
            reporter.on_assistant_text_delta(&text).await;
        }
        for tc in choice.delta.tool_calls {
            if self.tool_calls.len() <= tc.index {
                self.tool_calls.resize_with(tc.index + 1, || None);
            }
            let entry = self.tool_calls[tc.index].get_or_insert_with(PendingToolCall::default);
            if let Some(id) = tc.id {
                entry.id = id;
            }
            if let Some(function) = tc.function {
                if let Some(name) = function.name {
                    entry.name = name;
                }
                if let Some(args) = function.arguments {
                    entry.arguments.push_str(&args);
                }
            }
        }
        Ok(())
    }

    fn into_response(self) -> OaResponse {
        let tool_calls: Vec<OaToolCall> = self
            .tool_calls
            .into_iter()
            .flatten()
            .map(|t| OaToolCall {
                id: t.id,
                kind: "function".to_string(),
                function: OaFunctionCall {
                    name: t.name,
                    arguments: t.arguments,
                },
            })
            .collect();

        OaResponse {
            choices: vec![OaChoice {
                message: OaResponseMessage {
                    content: self.saw_text.then_some(self.text),
                    tool_calls,
                },
                finish_reason: self.finish_reason,
            }],
            usage: self.usage,
        }
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
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OaStreamOptions>,
}

#[derive(Serialize)]
struct OaStreamOptions {
    include_usage: bool,
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

/// One `minder_core::Message` can expand to multiple OpenAI messages: a
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
                role: if m.role == Role::User { "user" } else { "assistant" }.to_string(),
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
        let arguments = serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
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
    use minder_core::ToolResult;

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
        assert_eq!(mapped[0].tool_calls[0].function.arguments, r#"{"command":"ls"}"#);
    }

    #[derive(Default)]
    struct RecordingReporter {
        deltas: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Reporter for RecordingReporter {
        async fn on_assistant_text_delta(&self, delta: &str) {
            self.deltas.lock().unwrap().push(delta.to_string());
        }
    }

    async fn accumulate(sse_lines: &[&str], reporter: &dyn Reporter) -> OaResponse {
        let mut accum = OaStreamAccumulator::default();
        for line in sse_lines {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            accum.handle_chunk(data, reporter).await.unwrap();
        }
        accum.into_response()
    }

    #[tokio::test]
    async fn streaming_text_deltas_are_reported_live_and_assembled_in_order() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"data: {"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
                r#"data: {"choices":[{"delta":{"content":"Hello, "},"finish_reason":null}]}"#,
                r#"data: {"choices":[{"delta":{"content":"world!"},"finish_reason":null}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
                r#"data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}"#,
                "data: [DONE]",
            ],
            &reporter,
        )
        .await;

        assert_eq!(*reporter.deltas.lock().unwrap(), vec!["Hello, ", "world!"]);
        let resp = from_openai_response(resp).unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 3);
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "Hello, world!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_tool_calls_accumulate_by_index_across_deltas() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]},"finish_reason":null}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]},"finish_reason":null}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
                "data: [DONE]",
            ],
            &reporter,
        )
        .await;

        assert!(reporter.deltas.lock().unwrap().is_empty());
        let resp = from_openai_response(resp).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[tokio::test]
    async fn complete_streaming_parses_a_real_sse_response_over_http() {
        let server = wiremock::MockServer::start().await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n",
        );

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/chat/completions"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;

        let provider = OpenAiProvider {
            api_key: "test".to_string(),
            base_url: server.uri(),
            model: "gpt-5.4-mini".to_string(),
            client: reqwest::Client::new(),
        };

        let reporter = RecordingReporter::default();
        let resp = provider
            .complete_streaming(&[Message::user_text("hi")], &[], None, &reporter)
            .await
            .unwrap();

        assert_eq!(*reporter.deltas.lock().unwrap(), vec!["hi"]);
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    #[ignore]
    async fn live_round_trip() {
        let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY to run this test");
        let provider = OpenAiProvider::new(api_key, "gpt-5.4-mini");
        let messages = vec![Message::user_text("Say hello in exactly three words.")];
        let resp = provider
            .complete(&messages, &[], None)
            .await
            .expect("live API call failed");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }
}
