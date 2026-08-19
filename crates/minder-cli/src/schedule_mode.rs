use std::time::Duration;

use minder_core::{AgentError, AgentSession};

pub struct ScheduleOptions {
    pub interval: Duration,
    /// `None` runs forever; `Some(n)` stops cleanly after `n` runs.
    pub max_runs: Option<usize>,
}

/// Runs `task` as a fresh turn every `opts.interval`, forever unless
/// `opts.max_runs` caps it -- the "run this exact thing on a timer" sibling
/// of `loop_mode::run`'s "poll a checklist for new work" model. Runs once
/// immediately, then sleeps, matching the "run now, then every N" cadence
/// most schedulers use.
///
/// `on_turn` fires after every completed turn (not just at the end), same
/// contract as `loop_mode::run`, so a caller can persist incrementally -- a
/// Ctrl-C or crash between runs then loses at most the in-flight run, not
/// the whole schedule's history.
pub async fn run(
    session: &mut AgentSession,
    task: &str,
    opts: ScheduleOptions,
    mut on_turn: impl FnMut(&AgentSession),
) -> Result<(), AgentError> {
    let mut runs = 0usize;
    loop {
        session.run_turn(task).await?;
        on_turn(session);
        runs += 1;
        if opts.max_runs.is_some_and(|max| runs >= max) {
            return Ok(());
        }
        tokio::time::sleep(opts.interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::{
        ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Role, StopReason, ToolContext, ToolSpec,
        Usage,
    };
    use std::sync::{Arc, Mutex as StdMutex};

    /// Scripted provider: always the same plain-text reply, so every call to
    /// `run_turn` succeeds immediately with no tool calls involved.
    struct FixedReplyProvider;

    #[async_trait::async_trait]
    impl LlmProvider for FixedReplyProvider {
        fn id(&self) -> &'static str {
            "scripted"
        }
        fn model(&self) -> &str {
            "scripted-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(ProviderResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text("done".to_string())],
                    metadata: serde_json::Value::Null,
                },
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        }
    }

    fn test_session() -> AgentSession {
        let tool_ctx = ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
            mailbox: None,
        };
        AgentSession::new(
            Arc::new(FixedReplyProvider),
            Vec::new(),
            None,
            "you are a test agent",
            tool_ctx,
        )
    }

    #[tokio::test(start_paused = true)]
    async fn on_turn_fires_once_per_run_and_honors_max_runs() {
        let mut session = test_session();
        let opts = ScheduleOptions {
            interval: Duration::from_secs(60),
            max_runs: Some(3),
        };

        let runs = Arc::new(StdMutex::new(0usize));
        let result = run(&mut session, "say hello", opts, {
            let runs = runs.clone();
            move |_session| *runs.lock().unwrap() += 1
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(*runs.lock().unwrap(), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn runs_again_only_after_the_interval_elapses() {
        let mut session = test_session();
        let opts = ScheduleOptions {
            interval: Duration::from_secs(60),
            max_runs: None,
        };

        let runs = Arc::new(StdMutex::new(0usize));
        let handle = tokio::spawn({
            let runs = runs.clone();
            async move {
                let _ = run(&mut session, "say hello", opts, move |_session| {
                    *runs.lock().unwrap() += 1;
                })
                .await;
            }
        });

        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!(*runs.lock().unwrap(), 1, "should run immediately, before any sleep");

        tokio::time::advance(Duration::from_secs(60)).await;
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!(*runs.lock().unwrap(), 2, "second run only after the interval elapses");

        handle.abort();
    }
}
