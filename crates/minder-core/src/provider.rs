use crate::message::{ContentBlock, Message, ProviderResponse, ToolSpec};
use crate::reporter::Reporter;

#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    fn id(&self) -> &'static str;

    /// The model name in use (e.g. `"claude-sonnet-5"`), for display purposes
    /// only -- not used for any routing decision.
    fn model(&self) -> &str;

    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
    ) -> Result<ProviderResponse, ProviderError>;

    /// Same as `complete`, but calls `reporter.on_assistant_text_delta` with
    /// each chunk of assistant text as it's generated, so a live caller can
    /// render it as it arrives instead of only once the whole turn is done.
    /// Default just runs `complete` and reports the whole text as one chunk
    /// -- providers that haven't implemented real streaming still work,
    /// they just don't feel live.
    async fn complete_streaming(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        system_prompt: Option<&str>,
        reporter: &dyn Reporter,
    ) -> Result<ProviderResponse, ProviderError> {
        let response = self.complete(messages, tools, system_prompt).await?;
        for block in &response.message.content {
            if let ContentBlock::Text(text) = block {
                reporter.on_assistant_text_delta(text).await;
            }
        }
        Ok(response)
    }
}

// No dependency on reqwest here by design -- minder-core stays HTTP-agnostic.
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
