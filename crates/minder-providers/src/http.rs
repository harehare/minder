use std::time::Duration;

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
