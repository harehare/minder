use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_RESULTS: u32 = 5;
const MAX_RESULTS_CAP: u32 = 10;
const TAVILY_BASE_URL: &str = "https://api.tavily.com";

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub content: String,
}

#[async_trait]
trait SearchBackend: Send + Sync {
    async fn search(&self, query: &str, max_results: u32) -> Result<Vec<SearchHit>, String>;
}

pub struct WebSearchTool {
    backend: Box<dyn SearchBackend>,
}

impl WebSearchTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            backend: Box::new(TavilyBackend::new(api_key)),
        }
    }
}

#[derive(Deserialize)]
struct Args {
    query: String,
    #[serde(default)]
    max_results: Option<u32>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Searches the web and returns matching results (title, url, snippet)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "max_results": { "type": "integer", "description": "Max results to return (default 5, capped at 10)" }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let max_results = args.max_results.unwrap_or(DEFAULT_MAX_RESULTS).min(MAX_RESULTS_CAP);

        match self.backend.search(&args.query, max_results).await {
            Ok(hits) => {
                let content = if hits.is_empty() {
                    "no results".to_string()
                } else {
                    hits.iter()
                        .map(|h| format!("{}\n{}\n{}", h.title, h.url, h.content))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                };
                ToolExecOutcome {
                    content,
                    is_error: false,
                    metadata: serde_json::json!({ "count": hits.len() }),
                }
            }
            Err(e) => error(e),
        }
    }
}

fn error(message: String) -> ToolExecOutcome {
    ToolExecOutcome {
        content: message,
        is_error: true,
        metadata: serde_json::Value::Null,
    }
}

// -- Tavily backend --

struct TavilyBackend {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl TavilyBackend {
    fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: TAVILY_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        }
    }

    #[cfg(test)]
    fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[derive(Serialize)]
struct TavilyRequest {
    api_key: String,
    query: String,
    max_results: u32,
}

#[derive(Deserialize)]
struct TavilyResponse {
    results: Vec<TavilyResult>,
}

#[derive(Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
}

#[async_trait]
impl SearchBackend for TavilyBackend {
    async fn search(&self, query: &str, max_results: u32) -> Result<Vec<SearchHit>, String> {
        let body = TavilyRequest {
            api_key: self.api_key.clone(),
            query: query.to_string(),
            max_results,
        };

        let resp = self
            .client
            .post(format!("{}/search", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| format!("failed to read response body: {e}"))?;

        if !status.is_success() {
            return Err(format!("search API returned {status}: {text}"));
        }

        let parsed: TavilyResponse =
            serde_json::from_str(&text).map_err(|e| format!("failed to parse response: {e}"))?;

        Ok(parsed
            .results
            .into_iter()
            .map(|r| SearchHit {
                title: r.title,
                url: r.url,
                content: r.content,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn parses_results_from_mock_server() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::path("/search"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [
                    {"title": "Rust", "url": "https://rust-lang.org", "content": "A language"}
                ]
            })))
            .mount(&server)
            .await;

        let tool = WebSearchTool {
            backend: Box::new(TavilyBackend::new("test-key").with_base_url(server.uri())),
        };
        let outcome = tool.execute(serde_json::json!({"query": "rust"}), &ctx()).await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.metadata["count"], 1);
        assert!(outcome.content.contains("rust-lang.org"));
    }

    #[tokio::test]
    async fn api_error_is_a_tool_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::path("/search"))
            .respond_with(wiremock::ResponseTemplate::new(401).set_body_string("unauthorized"))
            .mount(&server)
            .await;

        let tool = WebSearchTool {
            backend: Box::new(TavilyBackend::new("bad-key").with_base_url(server.uri())),
        };
        let outcome = tool.execute(serde_json::json!({"query": "rust"}), &ctx()).await;
        assert!(outcome.is_error);
    }

    #[tokio::test]
    #[ignore]
    async fn live_round_trip() {
        let key = std::env::var("TAVILY_API_KEY").expect("set TAVILY_API_KEY");
        let tool = WebSearchTool::new(key);
        let outcome = tool
            .execute(serde_json::json!({"query": "rust programming language"}), &ctx())
            .await;
        assert!(!outcome.is_error);
    }
}
