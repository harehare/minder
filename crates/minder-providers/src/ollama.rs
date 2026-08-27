use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use minder_core::{
    ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Reporter, Role, StopReason, ToolCall,
    ToolResultContent, ToolSpec, Usage,
};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Ollama's own default (2048-4096) silently truncates context instead of
/// erroring -- a common cause of broken tool-calling. Sent on every request.
const DEFAULT_NUM_CTX: u32 = 8192;

/// A refused connection almost always means `ollama serve` isn't running --
/// surface that guess instead of reqwest's raw "connection refused" text.
fn describe_transport_error(e: reqwest::Error, base_url: &str) -> ProviderError {
    if e.is_connect() {
        ProviderError::Transport(format!(
            "could not connect to Ollama at {base_url} -- is it running? Start it with `ollama serve`, or install it from https://ollama.com/download"
        ))
    } else {
        ProviderError::Transport(e.to_string())
    }
}

pub struct OllamaProvider {
    base_url: String,
    model: String,
    client: reqwest::Client,
    num_ctx: u32,
}

impl OllamaProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            client: Self::build_client(None),
            num_ctx: DEFAULT_NUM_CTX,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Overrides the default request timeout -- see `crate::http`.
    pub fn with_request_timeout_secs(mut self, secs: u64) -> Self {
        self.client = Self::build_client(Some(secs));
        self
    }

    /// Overrides `DEFAULT_NUM_CTX` -- see its doc comment.
    pub fn with_num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = num_ctx;
        self
    }

    fn build_client(request_timeout_secs: Option<u64>) -> reqwest::Client {
        crate::http::client_builder(request_timeout_secs)
            // Detects a connection that's silently gone dead (common through
            // proxies/tunnels) instead of waiting the full request timeout to
            // notice.
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .expect("reqwest client config is static and valid")
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
            options: OlOptions { num_ctx: self.num_ctx },
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| describe_transport_error(e, &self.base_url))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| describe_transport_error(e, &self.base_url))?;

        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: OlResponse = serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        Ok(from_ollama_response(parsed))
    }

    async fn complete_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
        reporter: &dyn Reporter,
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
            stream: true,
            options: OlOptions { num_ctx: self.num_ctx },
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| describe_transport_error(e, &self.base_url))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(|e| describe_transport_error(e, &self.base_url))?;
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        // Ollama's native API streams newline-delimited JSON objects, not
        // SSE -- no `data: ` prefix, just one complete object per line, each
        // carrying the newly generated fragment of `content`/`thinking`.
        let mut stream = resp.bytes_stream();
        let mut line_buf: Vec<u8> = Vec::new();
        let mut accum = OlStreamAccumulator::default();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| describe_transport_error(e, &self.base_url))?;
            line_buf.extend_from_slice(&chunk);

            while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = line_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                accum.handle_line(line, reporter).await?;
            }
        }

        Ok(from_ollama_response(accum.into_response()))
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let resp = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map_err(|e| describe_transport_error(e, &self.base_url))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| describe_transport_error(e, &self.base_url))?;
        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: OlTagsResponse = serde_json::from_str(&text).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        Ok(parsed.models.into_iter().map(|m| m.name).collect())
    }

    fn context_window(&self) -> Option<u32> {
        Some(self.num_ctx)
    }

    async fn ensure_model_available(&self, reporter: &dyn Reporter) -> Result<(), ProviderError> {
        if self.list_models().await?.iter().any(|m| m == &self.model) {
            return Ok(());
        }
        reporter
            .on_notice(&format!("model '{}' isn't pulled yet -- downloading...", self.model))
            .await;

        let resp = self
            .client
            .post(format!("{}/api/pull", self.base_url))
            .json(&serde_json::json!({"model": self.model, "stream": true}))
            .send()
            .await
            .map_err(|e| describe_transport_error(e, &self.base_url))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: text,
            });
        }

        let mut stream = resp.bytes_stream();
        let mut line_buf: Vec<u8> = Vec::new();
        // Reports at most once per 2s -- pull progress lines arrive fast
        // enough to otherwise spam the display.
        let mut last_reported = std::time::Instant::now() - Duration::from_secs(2);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| describe_transport_error(e, &self.base_url))?;
            line_buf.extend_from_slice(&chunk);
            while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = line_buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let progress: OlPullProgress =
                    serde_json::from_str(line).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
                if let Some(err) = progress.error {
                    return Err(ProviderError::Api { status: 0, body: err });
                }
                if progress.status == "success" {
                    reporter.on_notice(&format!("model '{}' pulled.", self.model)).await;
                } else if let (Some(completed), Some(total)) = (progress.completed, progress.total)
                    && total > 0
                    && last_reported.elapsed() > Duration::from_secs(2)
                {
                    reporter
                        .on_notice(&format!("  {} ({}%)", progress.status, completed * 100 / total))
                        .await;
                    last_reported = std::time::Instant::now();
                }
            }
        }
        Ok(())
    }
}

/// Assembles Ollama's NDJSON stream into the same shape `from_ollama_response`
/// maps from a single non-streaming response. Tool calls are sent whole
/// (Ollama doesn't stream partial arguments the way Anthropic/OpenAI do), so
/// they're just collected as they arrive rather than accumulated piecewise.
#[derive(Default)]
struct OlStreamAccumulator {
    thinking: String,
    text: String,
    tool_calls: Vec<OlToolCall>,
    done: bool,
    done_reason: Option<String>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

impl OlStreamAccumulator {
    async fn handle_line(&mut self, line: &str, reporter: &dyn Reporter) -> Result<(), ProviderError> {
        let chunk: OlResponse = serde_json::from_str(line).map_err(|e| ProviderError::Deserialize(e.to_string()))?;
        if !chunk.message.content.is_empty() {
            reporter.on_assistant_text_delta(&chunk.message.content).await;
            self.text.push_str(&chunk.message.content);
        }
        if !chunk.message.thinking.is_empty() {
            self.thinking.push_str(&chunk.message.thinking);
        }
        self.tool_calls.extend(chunk.message.tool_calls);
        self.done = chunk.done;
        if chunk.done_reason.is_some() {
            self.done_reason = chunk.done_reason;
        }
        if chunk.prompt_eval_count.is_some() {
            self.prompt_eval_count = chunk.prompt_eval_count;
        }
        if chunk.eval_count.is_some() {
            self.eval_count = chunk.eval_count;
        }
        Ok(())
    }

    fn into_response(self) -> OlResponse {
        OlResponse {
            message: OlResponseMessage {
                content: self.text,
                thinking: self.thinking,
                tool_calls: self.tool_calls,
            },
            done: self.done,
            done_reason: self.done_reason,
            prompt_eval_count: self.prompt_eval_count,
            eval_count: self.eval_count,
        }
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
    options: OlOptions,
}

#[derive(Serialize)]
struct OlOptions {
    num_ctx: u32,
}

#[derive(Deserialize)]
struct OlTagsResponse {
    models: Vec<OlTagEntry>,
}

#[derive(Deserialize)]
struct OlTagEntry {
    name: String,
}

#[derive(Deserialize)]
struct OlPullProgress {
    #[serde(default)]
    status: String,
    #[serde(default)]
    completed: Option<u64>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    error: Option<String>,
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
    /// Set when `done` is true; e.g. "stop" or "length" (context/predict
    /// budget exhausted). Reasoning models can burn the whole budget on
    /// hidden `thinking` and never reach a final answer -- without this,
    /// that case was indistinguishable from a normal, complete "stop".
    done_reason: Option<String>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OlResponseMessage {
    content: String,
    /// Chain-of-thought from reasoning models (gpt-oss, deepseek-r1, ...).
    /// Ollama returns this separately from `content`; dropping it silently
    /// meant a response that was all thinking and no final answer showed up
    /// as an empty assistant message with nothing reported to the user.
    #[serde(default)]
    thinking: String,
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
    if !resp.message.thinking.is_empty() {
        content.push(ContentBlock::Thinking {
            text: resp.message.thinking,
            signature: None,
        });
    }
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
    } else if resp.done_reason.as_deref() == Some("length") {
        StopReason::MaxTokens
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
    fn surfaces_thinking_as_a_content_block() {
        let raw = r#"{
            "message": {"role": "assistant", "content": "42", "thinking": "let me reason..."},
            "done": true
        }"#;
        let parsed: OlResponse = serde_json::from_str(raw).unwrap();
        let resp = from_ollama_response(parsed);
        match &resp.message.content[0] {
            ContentBlock::Thinking { text, .. } => assert_eq!(text, "let me reason..."),
            other => panic!("expected Thinking, got {other:?}"),
        }
        match &resp.message.content[1] {
            ContentBlock::Text(t) => assert_eq!(t, "42"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn truncated_by_length_is_reported_as_max_tokens_not_a_silent_end_turn() {
        // A reasoning model can burn its whole budget on hidden `thinking`
        // and never reach a final answer -- `content` stays empty, but this
        // must not look like a normal, complete turn.
        let raw = r#"{
            "message": {"role": "assistant", "content": "", "thinking": "still reasoning when cut off"},
            "done": true,
            "done_reason": "length"
        }"#;
        let parsed: OlResponse = serde_json::from_str(raw).unwrap();
        let resp = from_ollama_response(parsed);
        assert_eq!(resp.stop_reason, StopReason::MaxTokens);
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

    #[derive(Default)]
    struct RecordingReporter {
        deltas: std::sync::Mutex<Vec<String>>,
        notices: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Reporter for RecordingReporter {
        async fn on_assistant_text_delta(&self, delta: &str) {
            self.deltas.lock().unwrap().push(delta.to_string());
        }
        async fn on_notice(&self, text: &str) {
            self.notices.lock().unwrap().push(text.to_string());
        }
    }

    async fn accumulate(lines: &[&str], reporter: &dyn Reporter) -> OlResponse {
        let mut accum = OlStreamAccumulator::default();
        for line in lines {
            accum.handle_line(line, reporter).await.unwrap();
        }
        accum.into_response()
    }

    #[tokio::test]
    async fn streaming_text_deltas_are_reported_live_and_assembled_in_order() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"{"message": {"role": "assistant", "content": "Hello, "}, "done": false}"#,
                r#"{"message": {"role": "assistant", "content": "world!"}, "done": false}"#,
                r#"{"message": {"role": "assistant", "content": ""}, "done": true, "prompt_eval_count": 10, "eval_count": 3}"#,
            ],
            &reporter,
        )
        .await;

        assert_eq!(*reporter.deltas.lock().unwrap(), vec!["Hello, ", "world!"]);
        let resp = from_ollama_response(resp);
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 3);
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "Hello, world!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_tool_call_and_truncation_are_carried_through() {
        let reporter = RecordingReporter::default();
        let resp = accumulate(
            &[
                r#"{"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}]}, "done": false}"#,
                r#"{"message": {"role": "assistant", "content": ""}, "done": true, "done_reason": "length"}"#,
            ],
            &reporter,
        )
        .await;

        assert!(reporter.deltas.lock().unwrap().is_empty());
        let resp = from_ollama_response(resp);
        // Ollama's own quirk (see `from_ollama_response`): a tool call, if
        // present, always wins the stop reason over a truncation reason.
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[tokio::test]
    async fn complete_streaming_parses_a_real_ndjson_response_over_http() {
        let server = wiremock::MockServer::start().await;
        let body = concat!(
            "{\"message\": {\"role\": \"assistant\", \"content\": \"hi\"}, \"done\": false}\n",
            "{\"message\": {\"role\": \"assistant\", \"content\": \"\"}, \"done\": true, \"prompt_eval_count\": 2, \"eval_count\": 1}\n",
        );

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(body, "application/x-ndjson"))
            .mount(&server)
            .await;

        let provider = OllamaProvider::new("llama3.2").with_base_url(server.uri());

        let reporter = RecordingReporter::default();
        let resp = provider
            .complete_streaming(&[Message::user_text("hi")], &[], None, &reporter)
            .await
            .unwrap();

        assert_eq!(*reporter.deltas.lock().unwrap(), vec!["hi"]);
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn complete_sends_num_ctx_defaulting_to_8192() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"options": {"num_ctx": 8192}}),
            ))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": {"role": "assistant", "content": "hi"},
                "done": true,
            })))
            .mount(&server)
            .await;

        let provider = OllamaProvider::new("llama3.2").with_base_url(server.uri());
        provider.complete(&[Message::user_text("hi")], &[], None).await.unwrap();
    }

    #[tokio::test]
    async fn with_num_ctx_overrides_the_default() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/chat"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"options": {"num_ctx": 32768}}),
            ))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": {"role": "assistant", "content": "hi"},
                "done": true,
            })))
            .mount(&server)
            .await;

        let provider = OllamaProvider::new("llama3.2")
            .with_base_url(server.uri())
            .with_num_ctx(32768);
        provider.complete(&[Message::user_text("hi")], &[], None).await.unwrap();
    }

    #[tokio::test]
    async fn list_models_returns_the_pulled_model_names() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{"name": "qwen2.5-coder:14b"}, {"name": "llama3.2"}],
            })))
            .mount(&server)
            .await;

        let provider = OllamaProvider::new("llama3.2").with_base_url(server.uri());
        let models = provider.list_models().await.unwrap();
        assert_eq!(models, vec!["qwen2.5-coder:14b".to_string(), "llama3.2".to_string()]);
    }

    #[tokio::test]
    async fn ensure_model_available_is_a_noop_when_already_pulled() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{"name": "llama3.2"}],
            })))
            .mount(&server)
            .await;
        // No /api/pull mock -- if ensure_model_available tried to pull anyway,
        // wiremock's default 404-on-unmatched-request would surface as an error.

        let provider = OllamaProvider::new("llama3.2").with_base_url(server.uri());
        let reporter = RecordingReporter::default();
        provider.ensure_model_available(&reporter).await.unwrap();
        assert!(reporter.notices.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ensure_model_available_pulls_a_missing_model_and_reports_progress() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({"models": []})))
            .mount(&server)
            .await;
        let pull_body = concat!(
            "{\"status\": \"pulling manifest\"}\n",
            "{\"status\": \"success\"}\n",
        );
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/pull"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(pull_body, "application/x-ndjson"))
            .mount(&server)
            .await;

        let provider = OllamaProvider::new("qwen2.5-coder:14b").with_base_url(server.uri());
        let reporter = RecordingReporter::default();
        provider.ensure_model_available(&reporter).await.unwrap();

        let notices = reporter.notices.lock().unwrap();
        assert!(notices.iter().any(|n| n.contains("downloading")), "{notices:?}");
        assert!(notices.iter().any(|n| n.contains("pulled")), "{notices:?}");
    }

    #[tokio::test]
    async fn ensure_model_available_surfaces_a_pull_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/tags"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({"models": []})))
            .mount(&server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/pull"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
                "{\"error\": \"model 'nonexistent' not found\"}\n",
                "application/x-ndjson",
            ))
            .mount(&server)
            .await;

        let provider = OllamaProvider::new("nonexistent").with_base_url(server.uri());
        let reporter = RecordingReporter::default();
        let err = provider.ensure_model_available(&reporter).await.unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }

    #[tokio::test]
    async fn a_refused_connection_hints_at_starting_ollama() {
        // Bind a std listener then drop it synchronously -- unlike wiremock's
        // async shutdown, this guarantees nothing answers on the port after.
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let base_url = format!("http://127.0.0.1:{port}");

        let provider = OllamaProvider::new("llama3.2").with_base_url(base_url.clone());
        let err = provider
            .complete(&[Message::user_text("hi")], &[], None)
            .await
            .unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("ollama serve"), "expected an `ollama serve` hint, got: {msg}");
        assert!(msg.contains(&base_url), "expected the base url in the message, got: {msg}");
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
