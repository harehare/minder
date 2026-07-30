use async_trait::async_trait;
use futures_util::StreamExt;
use minder_core::{
    ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Reporter, Role, StopReason, ToolCall,
    ToolResult, ToolResultContent, ToolSpec, Usage,
};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8192;

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
    /// `Some(n)` requests extended thinking with an `n`-token budget (the
    /// resulting `Thinking` blocks flow through `Message`/`Reporter` like any
    /// other content block -- see `Reporter::on_thinking`). `None` (the
    /// default) omits the `thinking` field entirely, so behavior/cost is
    /// unchanged unless a caller opts in.
    thinking_budget: Option<u32>,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            client: crate::http::client_builder(None)
                .build()
                .expect("reqwest client config is static and valid"),
            thinking_budget: None,
        }
    }

    pub fn with_thinking_budget(mut self, budget_tokens: u32) -> Self {
        self.thinking_budget = Some(budget_tokens);
        self
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
impl LlmProvider for AnthropicProvider {
    fn id(&self) -> &'static str {
        "anthropic"
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
        let body = AnthropicRequest {
            model: self.model.clone(),
            // Anthropic requires max_tokens > thinking.budget_tokens, so the
            // budget is added on top of the normal output budget rather than
            // eating into it.
            max_tokens: DEFAULT_MAX_TOKENS + self.thinking_budget.unwrap_or(0),
            system: system_prompt.map(str::to_string),
            messages: to_anthropic_messages(messages),
            tools: to_anthropic_tools(tools),
            thinking: self.thinking_budget.map(|budget_tokens| AnthropicThinkingConfig {
                thinking_type: "enabled",
                budget_tokens,
            }),
            stream: false,
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

        let parsed: AnthropicResponse =
            serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;

        Ok(from_anthropic_response(parsed))
    }

    async fn complete_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
        reporter: &dyn Reporter,
    ) -> Result<ProviderResponse, ProviderError> {
        let body = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: DEFAULT_MAX_TOKENS + self.thinking_budget.unwrap_or(0),
            system: system_prompt.map(str::to_string),
            messages: to_anthropic_messages(messages),
            tools: to_anthropic_tools(tools),
            thinking: self.thinking_budget.map(|budget_tokens| AnthropicThinkingConfig {
                thinking_type: "enabled",
                budget_tokens,
            }),
            stream: true,
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
        let mut accum = StreamAccumulator::default();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ProviderError::Transport(e.to_string()))?;
            line_buf.extend_from_slice(&chunk);

            while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = line_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = line.trim_end().strip_prefix("data: ") else {
                    continue; // blank line, `event: ...` line, or a comment
                };
                if accum.handle_event(data, reporter).await? {
                    return Ok(from_anthropic_response(accum.into_response()));
                }
            }
        }

        Ok(from_anthropic_response(accum.into_response()))
    }
}

/// Assembles Anthropic's SSE stream (`content_block_start`/`_delta`/`_stop`,
/// `message_start`/`_delta`/`_stop`) into the same shape `complete`'s single
/// JSON response parses into, so both paths share `from_anthropic_response`.
/// Blocks arrive strictly in index order, so a plain `Vec` (pushed on each
/// `content_block_start`) tracks them without needing an index map.
#[derive(Default)]
struct StreamAccumulator {
    blocks: Vec<PendingBlock>,
    stop_reason: Option<String>,
    input_tokens: u32,
    output_tokens: u32,
}

enum PendingBlock {
    Text(String),
    Thinking {
        text: String,
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        partial_json: String,
    },
}

impl StreamAccumulator {
    /// Returns `Ok(true)` once `message_stop` closes the stream.
    async fn handle_event(&mut self, data: &str, reporter: &dyn Reporter) -> Result<bool, ProviderError> {
        let event: serde_json::Value =
            serde_json::from_str(data).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        match event.get("type").and_then(|v| v.as_str()) {
            Some("message_start") => {
                if let Some(tokens) = event["message"]["usage"]["input_tokens"].as_u64() {
                    self.input_tokens = tokens as u32;
                }
            }
            Some("content_block_start") => {
                let block = &event["content_block"];
                let pending = match block["type"].as_str() {
                    Some("tool_use") => PendingBlock::ToolUse {
                        id: block["id"].as_str().unwrap_or_default().to_string(),
                        name: block["name"].as_str().unwrap_or_default().to_string(),
                        partial_json: String::new(),
                    },
                    Some("thinking") => PendingBlock::Thinking {
                        text: String::new(),
                        signature: None,
                    },
                    _ => PendingBlock::Text(String::new()),
                };
                self.blocks.push(pending);
            }
            Some("content_block_delta") => {
                let Some(current) = self.blocks.last_mut() else {
                    return Ok(false);
                };
                let delta = &event["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => {
                        let text = delta["text"].as_str().unwrap_or_default();
                        if let PendingBlock::Text(buf) = current {
                            buf.push_str(text);
                        }
                        reporter.on_assistant_text_delta(text).await;
                    }
                    Some("thinking_delta") => {
                        if let PendingBlock::Thinking { text, .. } = current {
                            text.push_str(delta["thinking"].as_str().unwrap_or_default());
                        }
                    }
                    Some("signature_delta") => {
                        if let PendingBlock::Thinking { signature, .. } = current {
                            *signature = Some(delta["signature"].as_str().unwrap_or_default().to_string());
                        }
                    }
                    Some("input_json_delta") => {
                        if let PendingBlock::ToolUse { partial_json, .. } = current {
                            partial_json.push_str(delta["partial_json"].as_str().unwrap_or_default());
                        }
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(tokens) = event["usage"]["output_tokens"].as_u64() {
                    self.output_tokens = tokens as u32;
                }
            }
            Some("message_stop") => return Ok(true),
            Some("error") => {
                let message = event["error"]["message"].as_str().unwrap_or("unknown stream error");
                return Err(ProviderError::Transport(message.to_string()));
            }
            _ => {} // ping, content_block_stop -- nothing to do
        }
        Ok(false)
    }

    fn into_response(self) -> AnthropicResponse {
        let content = self
            .blocks
            .into_iter()
            .map(|b| match b {
                PendingBlock::Text(text) => AnthropicContentBlock::Text { text },
                PendingBlock::Thinking { text, signature } => AnthropicContentBlock::Thinking {
                    thinking: text,
                    signature,
                },
                PendingBlock::ToolUse { id, name, partial_json } => AnthropicContentBlock::ToolUse {
                    id,
                    name,
                    input: if partial_json.is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&partial_json).unwrap_or(serde_json::json!({}))
                    },
                },
            })
            .collect();

        AnthropicResponse {
            content,
            stop_reason: self.stop_reason,
            usage: AnthropicUsage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
            },
        }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize)]
struct AnthropicThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: &'static str,
    budget_tokens: u32,
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
            AnthropicContentBlock::Thinking { thinking, signature } => ContentBlock::Thinking {
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
    fn parses_thinking_block_in_response() {
        let raw = r#"{
            "id": "msg_3", "type": "message", "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me work through this", "signature": "sig123"},
                {"type": "text", "text": "here's the answer"}
            ],
            "model": "claude-sonnet-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 12}
        }"#;
        let parsed: AnthropicResponse = serde_json::from_str(raw).unwrap();
        let resp = from_anthropic_response(parsed);

        match &resp.message.content[0] {
            ContentBlock::Thinking { text, signature } => {
                assert_eq!(text, "let me work through this");
                assert_eq!(signature.as_deref(), Some("sig123"));
            }
            other => panic!("expected Thinking, got {other:?}"),
        }
    }

    #[test]
    fn thinking_config_is_omitted_by_default_and_present_when_requested() {
        let without_thinking = AnthropicRequest {
            model: "claude-sonnet-5".to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            system: None,
            messages: vec![],
            tools: vec![],
            thinking: None,
            stream: false,
        };
        let json = serde_json::to_value(&without_thinking).unwrap();
        assert!(json.get("thinking").is_none());

        let with_thinking = AnthropicRequest {
            thinking: Some(AnthropicThinkingConfig {
                thinking_type: "enabled",
                budget_tokens: 4000,
            }),
            ..without_thinking
        };
        let json = serde_json::to_value(&with_thinking).unwrap();
        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["thinking"]["budget_tokens"], 4000);
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

    /// Records every `on_assistant_text_delta` call, so streaming tests can
    /// assert deltas arrived live rather than only checking the final
    /// accumulated response.
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

    /// Feeds each `data: ...` line (skipping blanks and `event:` lines, same
    /// as `complete_streaming`'s real parsing) through a fresh accumulator.
    async fn accumulate(sse_lines: &[&str], reporter: &dyn Reporter) -> AnthropicResponse {
        let mut accum = StreamAccumulator::default();
        for line in sse_lines {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if accum.handle_event(data, reporter).await.unwrap() {
                break;
            }
        }
        accum.into_response()
    }

    #[tokio::test]
    async fn streaming_text_deltas_are_reported_live_and_assembled_in_order() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"data: {"type":"message_start","message":{"usage":{"input_tokens":7,"output_tokens":0}}}"#,
                r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, "}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world!"}}"#,
                r#"data: {"type":"content_block_stop","index":0}"#,
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
                r#"data: {"type":"message_stop"}"#,
            ],
            &reporter,
        )
        .await;

        assert_eq!(*reporter.deltas.lock().unwrap(), vec!["Hello, ", "world!"]);
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(resp.usage.input_tokens, 7);
        assert_eq!(resp.usage.output_tokens, 5);
        match &resp.content[0] {
            AnthropicContentBlock::Text { text } => assert_eq!(text, "Hello, world!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_tool_use_accumulates_partial_json_across_deltas() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash"}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}"#,
                r#"data: {"type":"content_block_stop","index":0}"#,
                r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}"#,
                r#"data: {"type":"message_stop"}"#,
            ],
            &reporter,
        )
        .await;

        assert!(
            reporter.deltas.lock().unwrap().is_empty(),
            "tool args aren't streamed as text"
        );
        match &resp.content[0] {
            AnthropicContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "bash");
                assert_eq!(input["command"], "ls");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_thinking_block_accumulates_text_and_signature() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig123"}}"#,
                r#"data: {"type":"content_block_stop","index":0}"#,
                r#"data: {"type":"message_stop"}"#,
            ],
            &reporter,
        )
        .await;

        match &resp.content[0] {
            AnthropicContentBlock::Thinking { thinking, signature } => {
                assert_eq!(thinking, "let me think");
                assert_eq!(signature.as_deref(), Some("sig123"));
            }
            other => panic!("expected Thinking, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_error_event_surfaces_as_a_provider_error() {
        let reporter = RecordingReporter::default();
        let mut accum = StreamAccumulator::default();
        let err = accum
            .handle_event(
                r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}"#,
                &reporter,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Transport(msg) if msg == "overloaded"));
    }

    /// Unlike `accumulate` above (which feeds synthetic lines straight into
    /// the accumulator), this goes over a real socket via `wiremock` -- the
    /// thing actually worth distrusting is `reqwest`'s `bytes_stream` plus
    /// our own line-buffering, since TCP can fragment the body at any byte
    /// offset regardless of where the SSE line breaks fall.
    #[tokio::test]
    async fn complete_streaming_parses_a_real_sse_response_over_http() {
        let server = wiremock::MockServer::start().await;
        let body = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;

        let provider = AnthropicProvider {
            api_key: "test".to_string(),
            base_url: server.uri(),
            model: "claude-sonnet-5".to_string(),
            client: reqwest::Client::new(),
            thinking_budget: None,
        };

        let reporter = RecordingReporter::default();
        let resp = provider
            .complete_streaming(&[Message::user_text("hi")], &[], None, &reporter)
            .await
            .unwrap();

        assert_eq!(*reporter.deltas.lock().unwrap(), vec!["hi"]);
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "hi"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }

    /// Response is delayed past the override, so this exercises the actual
    /// timeout rather than just checking the builder stores the value.
    #[tokio::test]
    async fn request_timeout_override_fails_a_response_that_never_arrives_in_time() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(2))
                    .set_body_raw("data: {\"type\":\"message_stop\"}\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let mut provider = AnthropicProvider::new("test", "claude-sonnet-5").with_request_timeout_secs(1);
        provider.base_url = server.uri();

        let reporter = RecordingReporter::default();
        let err = provider
            .complete_streaming(&[Message::user_text("hi")], &[], None, &reporter)
            .await
            .unwrap_err();

        assert!(matches!(err, ProviderError::Transport(_)));
    }

    /// Live smoke test against the real API. Requires ANTHROPIC_API_KEY;
    /// ignored by default so CI doesn't need a key.
    #[tokio::test]
    #[ignore]
    async fn live_round_trip() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY to run this test");
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
