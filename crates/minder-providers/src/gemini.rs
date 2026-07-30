use async_trait::async_trait;
use futures_util::StreamExt;
use minder_core::{
    ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Reporter, Role, StopReason, ToolCall,
    ToolResult, ToolResultContent, ToolSpec, Usage,
};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_MAX_TOKENS: u32 = 8192;
/// Separator between a synthesized tool-call id's function name and its
/// disambiguating index (see `to_gemini_contents` / `from_gemini_response`).
const ID_SEPARATOR: &str = "::";

pub struct GeminiProvider {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl GeminiProvider {
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
impl LlmProvider for GeminiProvider {
    fn id(&self) -> &'static str {
        "gemini"
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
        let body = GmRequest {
            contents: to_gemini_contents(messages),
            system_instruction: system_prompt.map(|sp| GmContent {
                role: None,
                parts: vec![GmPart::Text { text: sp.to_string() }],
            }),
            tools: to_gemini_tools(tools),
            generation_config: GmGenerationConfig {
                max_output_tokens: DEFAULT_MAX_TOKENS,
            },
        };

        let url = format!("{}/v1beta/models/{}:generateContent", self.base_url, self.model);
        let resp = self
            .client
            .post(&url)
            .query(&[("key", &self.api_key)])
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

        let parsed: GmResponse = serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        from_gemini_response(parsed)
    }

    async fn complete_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
        reporter: &dyn Reporter,
    ) -> Result<ProviderResponse, ProviderError> {
        let body = GmRequest {
            contents: to_gemini_contents(messages),
            system_instruction: system_prompt.map(|sp| GmContent {
                role: None,
                parts: vec![GmPart::Text { text: sp.to_string() }],
            }),
            tools: to_gemini_tools(tools),
            generation_config: GmGenerationConfig {
                max_output_tokens: DEFAULT_MAX_TOKENS,
            },
        };

        // `alt=sse` switches the wire format from Gemini's default (a single
        // JSON array whose elements trickle in, awkward to parse
        // incrementally) to standard `data: ...` SSE lines, each a complete
        // `GenerateContentResponse` for that increment.
        let url = format!("{}/v1beta/models/{}:streamGenerateContent", self.base_url, self.model);
        let resp = self
            .client
            .post(&url)
            .query(&[("key", &self.api_key), ("alt", &"sse".to_string())])
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
        let mut accum = GmStreamAccumulator::default();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ProviderError::Transport(e.to_string()))?;
            line_buf.extend_from_slice(&chunk);

            while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = line_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = line.trim_end().strip_prefix("data: ") else {
                    continue;
                };
                accum.handle_event(data, reporter).await?;
            }
        }

        from_gemini_response(accum.into_response())
    }
}

/// Each SSE event is a complete `GenerateContentResponse` for that
/// increment (unlike Anthropic/OpenAI's dedicated delta shapes), so events
/// parse straight into `GmResponse`. Consecutive `Text` parts are merged as
/// they arrive so the final shape matches a non-streaming response instead
/// of one part per chunk; function calls arrive whole in a single event, so
/// they just get pushed through unmodified.
#[derive(Default)]
struct GmStreamAccumulator {
    parts: Vec<GmPart>,
    finish_reason: Option<String>,
    usage: Option<GmUsage>,
}

impl GmStreamAccumulator {
    async fn handle_event(&mut self, data: &str, reporter: &dyn Reporter) -> Result<(), ProviderError> {
        let chunk: GmResponse = serde_json::from_str(data).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        if let Some(usage) = chunk.usage_metadata {
            self.usage = Some(usage);
        }
        let Some(candidate) = chunk.candidates.into_iter().next() else {
            return Ok(());
        };
        if let Some(reason) = candidate.finish_reason {
            self.finish_reason = Some(reason);
        }
        for part in candidate.content.map(|c| c.parts).unwrap_or_default() {
            match part {
                GmPart::Text { text } => {
                    reporter.on_assistant_text_delta(&text).await;
                    match self.parts.last_mut() {
                        Some(GmPart::Text { text: buf }) => buf.push_str(&text),
                        _ => self.parts.push(GmPart::Text { text }),
                    }
                }
                other => self.parts.push(other),
            }
        }
        Ok(())
    }

    fn into_response(self) -> GmResponse {
        GmResponse {
            candidates: vec![GmCandidate {
                content: Some(GmContent {
                    role: Some("model".to_string()),
                    parts: self.parts,
                }),
                finish_reason: self.finish_reason,
            }],
            usage_metadata: self.usage,
        }
    }
}

// -- wire format --

#[derive(Serialize)]
struct GmRequest {
    contents: Vec<GmContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GmContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GmToolDecl>,
    generation_config: GmGenerationConfig,
}

#[derive(Serialize)]
struct GmGenerationConfig {
    max_output_tokens: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct GmContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GmPart>,
}

// Untagged: Gemini distinguishes part kinds structurally (which field is
// present), not with an explicit type tag.
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
enum GmPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GmFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GmFunctionResponse,
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct GmFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct GmFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Serialize)]
struct GmToolDecl {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GmFunctionSpec>,
}

#[derive(Serialize)]
struct GmFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct GmResponse {
    candidates: Vec<GmCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GmUsage>,
}

#[derive(Deserialize)]
struct GmCandidate {
    content: Option<GmContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GmUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
}

// -- mapping --

/// Gemini only has "user"/"model" roles and keys function calls/responses by
/// *name*, not id -- so `Role::Tool` results get merged into a "user" turn
/// as `functionResponse` parts, and the id synthesized in
/// `from_gemini_response` is `"{name}::{index}"`, letting the name be
/// recovered here without a side table.
fn to_gemini_contents(messages: &[Message]) -> Vec<GmContent> {
    messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|m| {
            let role = if m.role == Role::Assistant { "model" } else { "user" };
            let parts = m
                .content
                .iter()
                .map(|b| match b {
                    ContentBlock::Text(t) => GmPart::Text { text: t.clone() },
                    ContentBlock::ToolUse(c) => GmPart::FunctionCall {
                        function_call: GmFunctionCall {
                            name: c.name.clone(),
                            args: c.arguments.clone(),
                        },
                    },
                    ContentBlock::ToolResult(r) => {
                        let name = r
                            .tool_call_id
                            .split_once(ID_SEPARATOR)
                            .map(|(n, _)| n)
                            .unwrap_or(&r.tool_call_id);
                        let text = match &r.content {
                            ToolResultContent::Text(t) => t.clone(),
                            ToolResultContent::Blocks(b) => serde_json::to_string(b).unwrap_or_default(),
                        };
                        GmPart::FunctionResponse {
                            function_response: GmFunctionResponse {
                                name: name.to_string(),
                                response: serde_json::json!({ "result": text }),
                            },
                        }
                    }
                    ContentBlock::Thinking { text, .. } => GmPart::Text { text: text.clone() },
                })
                .collect();
            GmContent {
                role: Some(role.to_string()),
                parts,
            }
        })
        .collect()
}

fn to_gemini_tools(tools: &[ToolSpec]) -> Vec<GmToolDecl> {
    if tools.is_empty() {
        return vec![];
    }
    vec![GmToolDecl {
        function_declarations: tools
            .iter()
            .map(|t| GmFunctionSpec {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            })
            .collect(),
    }]
}

fn from_gemini_response(resp: GmResponse) -> Result<ProviderResponse, ProviderError> {
    let candidate = resp
        .candidates
        .into_iter()
        .next()
        .ok_or_else(|| ProviderError::Deserialize("no candidates in response".to_string()))?;
    let parts = candidate.content.map(|c| c.parts).unwrap_or_default();

    let has_function_call = parts.iter().any(|p| matches!(p, GmPart::FunctionCall { .. }));
    let content = parts
        .into_iter()
        .enumerate()
        .map(|(i, p)| match p {
            GmPart::Text { text } => ContentBlock::Text(text),
            GmPart::FunctionCall { function_call } => ContentBlock::ToolUse(ToolCall {
                id: format!("{}{ID_SEPARATOR}{i}", function_call.name),
                name: function_call.name,
                arguments: function_call.args,
            }),
            GmPart::FunctionResponse { function_response } => ContentBlock::ToolResult(ToolResult {
                tool_call_id: function_response.name,
                content: ToolResultContent::Text(function_response.response.to_string()),
                is_error: false,
            }),
        })
        .collect();

    let stop_reason = match (candidate.finish_reason.as_deref(), has_function_call) {
        (_, true) => StopReason::ToolUse,
        (Some("STOP"), _) => StopReason::EndTurn,
        (Some("MAX_TOKENS"), _) => StopReason::MaxTokens,
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
            .usage_metadata
            .map(|u| Usage {
                input_tokens: u.prompt_token_count,
                output_tokens: u.candidates_token_count,
            })
            .unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_only_response() {
        let raw = r#"{
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hello there"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}
        }"#;
        let parsed: GmResponse = serde_json::from_str(raw).unwrap();
        let resp = from_gemini_response(parsed).unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "hello there"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn parses_function_call_and_synthesizes_id() {
        let raw = r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {"command": "ls"}}}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8}
        }"#;
        let parsed: GmResponse = serde_json::from_str(raw).unwrap();
        let resp = from_gemini_response(parsed).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].id, "bash::0");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn tool_result_becomes_function_response_keyed_by_recovered_name() {
        let messages = vec![Message::tool_results(vec![ToolResult {
            tool_call_id: "bash::0".into(),
            content: ToolResultContent::Text("file listing".into()),
            is_error: false,
        }])];
        let mapped = to_gemini_contents(&messages);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].role.as_deref(), Some("user"));
        match &mapped[0].parts[0] {
            GmPart::FunctionResponse { function_response } => {
                assert_eq!(function_response.name, "bash");
                assert_eq!(function_response.response["result"], "file listing");
            }
            other => panic!("expected FunctionResponse, got {other:?}"),
        }
    }

    #[test]
    fn system_role_message_is_excluded_from_contents() {
        let messages = vec![
            Message {
                role: Role::System,
                content: vec![ContentBlock::Text("ignored".into())],
                metadata: serde_json::Value::Null,
            },
            Message::user_text("hi"),
        ];
        let mapped = to_gemini_contents(&messages);
        assert_eq!(mapped.len(), 1);
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

    async fn accumulate(sse_lines: &[&str], reporter: &dyn Reporter) -> GmResponse {
        let mut accum = GmStreamAccumulator::default();
        for line in sse_lines {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            accum.handle_event(data, reporter).await.unwrap();
        }
        accum.into_response()
    }

    #[tokio::test]
    async fn streaming_text_deltas_are_reported_live_and_merged_into_one_block() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello, "}]}}]}"#,
                r#"data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "world!"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}}"#,
            ],
            &reporter,
        )
        .await;

        assert_eq!(*reporter.deltas.lock().unwrap(), vec!["Hello, ", "world!"]);
        let resp = from_gemini_response(resp).unwrap();
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 3);
        assert_eq!(
            resp.message.content.len(),
            1,
            "consecutive text parts should merge into one block"
        );
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "Hello, world!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_function_call_arrives_whole_in_one_event() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {"command": "ls"}}}]}, "finishReason": "STOP"}]}"#,
            ],
            &reporter,
        )
        .await;

        assert!(reporter.deltas.lock().unwrap().is_empty());
        let resp = from_gemini_response(resp).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[tokio::test]
    async fn complete_streaming_parses_a_real_sse_response_over_http() {
        let server = wiremock::MockServer::start().await;
        let body = "data: {\"candidates\": [{\"content\": {\"role\": \"model\", \"parts\": [{\"text\": \"hi\"}]}, \"finishReason\": \"STOP\"}], \"usageMetadata\": {\"promptTokenCount\": 2, \"candidatesTokenCount\": 1}}\n\n";

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path_regex(
                r"^/v1beta/models/.*:streamGenerateContent$",
            ))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;

        let provider = GeminiProvider {
            api_key: "test".to_string(),
            base_url: server.uri(),
            model: "gemini-3.5-flash".to_string(),
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
        let api_key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY to run this test");
        let provider = GeminiProvider::new(api_key, "gemini-3.5-flash");
        let messages = vec![Message::user_text("Say hello in exactly three words.")];
        let resp = provider
            .complete(&messages, &[], None)
            .await
            .expect("live API call failed");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }
}
