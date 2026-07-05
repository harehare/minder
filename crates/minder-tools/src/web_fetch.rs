use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::net::IpAddr;
use std::time::Duration;

const DEFAULT_MAX_BYTES: usize = 1_000_000;
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Result of a guarded fetch, shared by [`WebFetchTool`] and the wasm
/// plugin sandbox's `host_web_fetch` host function -- both go through the
/// same [`fetch`], not a reimplementation of the SSRF guard.
pub struct FetchResult {
    pub status: u16,
    pub body: String,
    pub truncated: bool,
}

/// Fetches `url_str`, rejecting non-http(s) schemes and literal
/// loopback/private-network hosts. See [`WebFetchTool::description`] for
/// the caveats (this is a partial SSRF guard, not a complete one).
pub async fn fetch(
    client: &reqwest::Client,
    url_str: &str,
    max_bytes: usize,
    timeout: Duration,
) -> Result<FetchResult, String> {
    fetch_inner(client, url_str, max_bytes, timeout, false).await
}

async fn fetch_inner(
    client: &reqwest::Client,
    url_str: &str,
    max_bytes: usize,
    timeout: Duration,
    allow_loopback: bool,
) -> Result<FetchResult, String> {
    let url = reqwest::Url::parse(url_str).map_err(|e| format!("invalid url: {e}"))?;
    guard_url(&url, allow_loopback)?;

    let response = client
        .get(url)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    let status = response.status().as_u16();
    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("failed to read response body: {e}"))?;

    let truncated = bytes.len() > max_bytes;
    let slice = &bytes[..bytes.len().min(max_bytes)];
    let body = String::from_utf8_lossy(slice).into_owned();

    Ok(FetchResult {
        status,
        body,
        truncated,
    })
}

pub struct WebFetchTool {
    client: reqwest::Client,
    allow_loopback: bool,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            allow_loopback: false,
        }
    }

    /// Only for tests that need to hit a local `wiremock` server, which
    /// necessarily binds to a loopback address -- the guard the tool
    /// applies to real traffic would otherwise reject the test server too.
    #[cfg(test)]
    fn new_allowing_loopback() -> Self {
        Self {
            client: reqwest::Client::new(),
            allow_loopback: true,
        }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct Args {
    url: String,
    #[serde(default)]
    max_bytes: Option<usize>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetches the content of an http(s) URL and returns its body as text, truncated to \
         max_bytes. Rejects non-http(s) schemes and literal loopback/private-network hosts; \
         this is a partial safeguard, not a complete SSRF defense -- prefer a hook policy on \
         on_tool_call for stronger guarantees."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "http(s) URL to fetch" },
                "max_bytes": { "type": "integer", "description": "Max response bytes to return (default 1,000,000)" },
                "timeout_secs": { "type": "integer", "description": "Timeout in seconds (default 30)" }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let max_bytes = args.max_bytes.unwrap_or(DEFAULT_MAX_BYTES);
        let timeout = Duration::from_secs(args.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS));

        match fetch_inner(&self.client, &args.url, max_bytes, timeout, self.allow_loopback).await {
            Ok(result) => ToolExecOutcome {
                content: result.body,
                is_error: false,
                metadata: serde_json::json!({
                    "status": result.status,
                    "truncated": result.truncated,
                }),
            },
            Err(e) => error(e),
        }
    }
}

/// Rejects non-http(s) schemes and hosts that are literal loopback/private/
/// link-local addresses. This is a string/IP-literal check only -- it does
/// NOT defend against DNS rebinding (a hostname resolving to a private IP
/// only at connect time). See the tool description.
fn guard_url(url: &reqwest::Url, allow_loopback: bool) -> Result<(), String> {
    if url.scheme() != "http" && url.scheme() != "https" {
        return Err(format!("unsupported scheme: {}", url.scheme()));
    }
    if allow_loopback {
        return Ok(());
    }
    let Some(host) = url.host_str() else {
        return Err("url has no host".to_string());
    };
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_disallowed_ip(&ip) {
            return Err(format!("refusing to fetch loopback/private address: {ip}"));
        }
    } else if host.eq_ignore_ascii_case("localhost") {
        return Err("refusing to fetch localhost".to_string());
    }
    Ok(())
}

fn is_disallowed_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified(),
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.is_unspecified() || (v6.segments()[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
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
    async fn successful_fetch_returns_body() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::path("/hello"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("hello world"))
            .mount(&server)
            .await;

        let outcome = WebFetchTool::new_allowing_loopback()
            .execute(serde_json::json!({"url": format!("{}/hello", server.uri())}), &ctx())
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "hello world");
        assert_eq!(outcome.metadata["status"], 200);
    }

    #[tokio::test]
    async fn http_error_status_is_not_a_tool_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::path("/missing"))
            .respond_with(wiremock::ResponseTemplate::new(404).set_body_string("not found"))
            .mount(&server)
            .await;

        let outcome = WebFetchTool::new_allowing_loopback()
            .execute(serde_json::json!({"url": format!("{}/missing", server.uri())}), &ctx())
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.metadata["status"], 404);
    }

    #[tokio::test]
    async fn transport_failure_is_an_error() {
        let outcome = WebFetchTool::new_allowing_loopback()
            .execute(
                serde_json::json!({"url": "http://127.0.0.1:1", "timeout_secs": 2}),
                &ctx(),
            )
            .await;
        assert!(outcome.is_error);
    }

    #[tokio::test]
    async fn max_bytes_truncates_response() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::path("/big"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("0123456789"))
            .mount(&server)
            .await;

        let outcome = WebFetchTool::new_allowing_loopback()
            .execute(
                serde_json::json!({"url": format!("{}/big", server.uri()), "max_bytes": 4}),
                &ctx(),
            )
            .await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "0123");
        assert_eq!(outcome.metadata["truncated"], true);
    }

    #[tokio::test]
    async fn rejects_file_scheme() {
        let outcome = WebFetchTool::new()
            .execute(serde_json::json!({"url": "file:///etc/passwd"}), &ctx())
            .await;
        assert!(outcome.is_error);
    }

    #[tokio::test]
    async fn rejects_loopback_ip() {
        let outcome = WebFetchTool::new()
            .execute(serde_json::json!({"url": "http://127.0.0.1/secret"}), &ctx())
            .await;
        assert!(outcome.is_error);
        assert!(outcome.content.contains("loopback"));
    }
}
