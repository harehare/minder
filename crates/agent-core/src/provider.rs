use crate::message::{Message, ProviderResponse, ToolSpec};

#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    fn id(&self) -> &'static str;

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
    ) -> Result<ProviderResponse, ProviderError>;
}

// No dependency on reqwest here by design -- agent-core stays HTTP-agnostic.
// Provider adapters convert their transport errors to `Transport(String)`.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("provider returned error status {status}: {body}")]
    Api { status: u16, body: String },
    #[error("failed to parse provider response: {0}")]
    Deserialize(String),
    #[error("rate-limited, retry after {retry_after_secs:?}s")]
    RateLimited { retry_after_secs: Option<u64> },
}
