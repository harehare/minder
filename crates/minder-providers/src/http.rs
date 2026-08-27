use std::time::Duration;

use minder_core::ProviderError;

/// Generous enough for a reasoning model's silent thinking time, but still
/// bounded so a stalled connection eventually fails instead of hanging.
pub(crate) const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 900;
pub(crate) const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 15;

/// Returns a builder, not a built `Client`, so `OllamaProvider` can chain its
/// own `tcp_keepalive` on top before `.build()`.
pub(crate) fn client_builder(request_timeout_secs: Option<u64>) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(
            request_timeout_secs.unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS),
        ))
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
}

/// A refused connection almost always means the local server isn't running
/// -- surface that guess instead of reqwest's raw "connection refused" text.
pub(crate) fn describe_transport_error(e: reqwest::Error, base_url: &str) -> ProviderError {
    if e.is_connect() {
        ProviderError::Transport(format!("could not connect to {base_url} -- is the server running?"))
    } else {
        ProviderError::Transport(e.to_string())
    }
}
