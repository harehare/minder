use std::path::Path;
use std::time::Duration;

use minder_core::AgentSession;

/// Reads `path` (via mq-lang's own `read_file` builtin, gated behind the
/// `file-io` feature) and picks out unchecked GFM checklist lines with a
/// regex, entirely inside the query -- the harness never touches the
/// filesystem itself for this. `path` is bound as a plain variable (see
/// `remaining_todos`) rather than string-interpolated into the query text,
/// the same way `minder-hooks::engine::HookEngine` binds `__hook_arg`.
///
/// This works line-by-line rather than through markdown's list/checkbox AST
/// (`is_list()`/`attr("checked")`) on purpose: those selectors only see
/// nodes the *host* already parsed and handed in as input, and mq-lang has
/// no script-level builtin that turns a `read_file`d string into that same
/// node form for a single file (the only builtin that does, `collection`,
/// parses every markdown file in an entire directory tree -- overkill, and
/// slow, for polling one checklist file every few seconds).
const DEFAULT_QUERY: &str =
    r#"read_file(path) | split(., "\n") | filter(., fn(line): is_regex_match(line, "^\\s*[-*+]\\s+\\[ \\]") end)"#;
const DEFAULT_MAX_ITERATIONS: usize = 50;
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;
const MAX_CONSECUTIVE_STALLS: usize = 2;

#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    #[error("mq query failed: {0}")]
    Query(String),
    #[error("agent turn failed: {0}")]
    Agent(#[from] minder_core::AgentError),
    #[error(
        "no progress after {0} consecutive iteration(s) (item count didn't decrease) -- \
         stopping to avoid spinning forever"
    )]
    Stalled(usize),
    #[error(
        "reached the maximum of {0} working iteration(s) -- stopping (raise \
         MINDER_LOOP_MAX_ITERATIONS to allow more)"
    )]
    MaxIterationsReached(usize),
}

pub struct LoopOptions {
    pub max_iterations: usize,
    pub query: String,
    pub poll_interval: Duration,
}

impl Default for LoopOptions {
    fn default() -> Self {
        Self {
            max_iterations: env_parsed("MINDER_LOOP_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS),
            query: std::env::var("MINDER_LOOP_QUERY").unwrap_or_else(|_| DEFAULT_QUERY.to_string()),
            poll_interval: Duration::from_secs(env_parsed(
                "MINDER_LOOP_POLL_INTERVAL_SECS",
                DEFAULT_POLL_INTERVAL_SECS,
            )),
        }
    }
}

fn env_parsed<T: std::str::FromStr>(var: &str, default: T) -> T {
    std::env::var(var).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Runs `query` with `path` bound to the `path` variable, entirely through
/// mq-lang -- no `std::fs` call here at all, the query's own `read_file`
/// does the reading. Mirrors `minder-hooks::engine::HookEngine`'s embedding
/// (`DefaultEngine` + `define_value` + `null_input`), just with `file-io`
/// enabled so the query can reach the filesystem on its own.
fn remaining_todos(path: &Path, query: &str) -> Result<Vec<String>, LoopError> {
    let mut engine = mq_lang::DefaultEngine::default();
    engine.load_builtin_module();
    engine.define_value("path", path.to_string_lossy().into_owned().into());

    let results = engine
        .eval(query, mq_lang::null_input().into_iter())
        .map_err(|e| LoopError::Query(e.to_string()))?;

    Ok(match results.values().first() {
        Some(mq_lang::RuntimeValue::Array(items)) => items.iter().map(|v| v.to_string()).collect(),
        _ => Vec::new(),
    })
}

fn build_prompt(path: &Path, task_hint: Option<&str>, remaining: &[String]) -> String {
    let hint = task_hint.map(|h| format!("Overall goal: {h}\n\n")).unwrap_or_default();
    let items = remaining.join("\n");
    format!(
        "{hint}Work through the checklist in `{}`. Remaining unchecked items:\n\n{items}\n\n\
         Pick the first unfinished item, implement it completely, then edit `{}` to check it off \
         (`- [ ]` -> `- [x]`) for that item before ending your turn. Only check off an item once \
         it is genuinely done -- if you run out of turn budget partway through, leave it unchecked.",
        path.display(),
        path.display(),
    )
}

/// Drives `session` turn after turn against the checklist in `path`,
/// re-deriving the next prompt from whatever's still unchecked after each
/// turn -- no user involvement between iterations. Once nothing is left
/// unchecked it doesn't exit: it polls `path` on `poll_interval` so newly
/// added checklist items (added by a human, or by another process) get
/// picked up automatically. The only ways out are a hard error (missing
/// file, a bad query, a provider error), a stall (item count stops
/// decreasing for too many consecutive working iterations), or hitting
/// `max_iterations` worth of actual (non-idle) turns -- polling while idle
/// doesn't count against that budget.
///
/// `on_turn` fires after every completed turn (not just at the end), so a
/// caller can persist the session incrementally -- a Ctrl-C or crash mid-loop
/// then loses at most the in-flight turn, not the whole run's history.
pub async fn run(
    session: &mut AgentSession,
    path: &Path,
    task_hint: Option<&str>,
    opts: LoopOptions,
    mut on_turn: impl FnMut(&AgentSession),
) -> Result<(), LoopError> {
    let mut previous_count: Option<usize> = None;
    let mut consecutive_stalls = 0;
    let mut working_iterations = 0usize;
    let mut announced_idle = false;

    loop {
        let remaining = remaining_todos(path, &opts.query)?;
        let count = remaining.len();

        if count == 0 {
            if !announced_idle {
                eprintln!(
                    "[loop] {} has no unchecked items -- polling every {}s for new work \
                     (Ctrl-C to stop)",
                    path.display(),
                    opts.poll_interval.as_secs()
                );
                announced_idle = true;
            }
            previous_count = None;
            tokio::time::sleep(opts.poll_interval).await;
            continue;
        }
        announced_idle = false;

        working_iterations += 1;
        if working_iterations > opts.max_iterations {
            return Err(LoopError::MaxIterationsReached(opts.max_iterations));
        }

        eprintln!(
            "[loop {working_iterations}/{}] {count} item(s) remaining in {}",
            opts.max_iterations,
            path.display()
        );

        if previous_count.is_some_and(|prev| count >= prev) {
            consecutive_stalls += 1;
            if consecutive_stalls > MAX_CONSECUTIVE_STALLS {
                return Err(LoopError::Stalled(consecutive_stalls));
            }
        } else {
            consecutive_stalls = 0;
        }
        previous_count = Some(count);

        let prompt = build_prompt(path, task_hint, &remaining);
        session.run_turn(&prompt).await?;
        on_turn(session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_md(content: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        // Nanosecond timestamps alone can collide under parallel test
        // execution on platforms with coarser clock resolution -- an
        // in-process counter guarantees uniqueness regardless.
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("minder-loop-test-{}-{n}.md", std::process::id()));
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn remaining_todos_lists_only_unchecked_checklist_items() {
        let path =
            temp_md("# T\n\n- [x] done\n- [ ] not done\n- plain bullet, no checkbox\n  - [ ] nested unchecked\n");
        let items = remaining_todos(&path, DEFAULT_QUERY).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(items.len(), 2);
        assert!(items[0].contains("not done"));
        assert!(items[1].contains("nested unchecked"));
    }

    #[test]
    fn remaining_todos_is_empty_once_everything_is_checked() {
        let path = temp_md("- [x] a\n- [x] b\n");
        let items = remaining_todos(&path, DEFAULT_QUERY).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert!(items.is_empty());
    }

    #[test]
    fn remaining_todos_reports_a_clear_error_for_a_missing_file() {
        let err = remaining_todos(Path::new("/nonexistent/does-not-exist.md"), DEFAULT_QUERY).unwrap_err();
        assert!(matches!(err, LoopError::Query(_)));
    }

    #[test]
    fn prompt_includes_remaining_items_and_instructs_checkoff() {
        let remaining = vec!["- [ ] a".to_string()];
        let prompt = build_prompt(Path::new("TODO.md"), Some("ship the feature"), &remaining);
        assert!(prompt.contains("ship the feature"));
        assert!(prompt.contains("- [ ] a"));
        assert!(prompt.contains("TODO.md"));
        assert!(prompt.contains("- [x]"));
    }

    use minder_core::{
        ContentBlock, LlmProvider, Message, ProviderError, ProviderResponse, Role, StopReason, Tool, ToolCall,
        ToolContext, ToolExecOutcome, ToolSpec, Usage,
    };
    use std::sync::{Arc, Mutex as StdMutex};

    /// Scripted provider: one tool_use call that "finishes" the checklist
    /// item, then a plain text reply -- exactly one working `run_turn`.
    struct OneShotFinishProvider(StdMutex<std::collections::VecDeque<ProviderResponse>>);

    #[async_trait::async_trait]
    impl LlmProvider for OneShotFinishProvider {
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
            Ok(self.0.lock().unwrap().pop_front().expect("script exhausted"))
        }
    }

    fn tool_use_response(path: &Path) -> ProviderResponse {
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse(ToolCall {
                    id: "1".to_string(),
                    name: "finish".to_string(),
                    arguments: serde_json::json!({"path": path.to_string_lossy()}),
                })],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        }
    }

    fn text_response(text: &str) -> ProviderResponse {
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(text.to_string())],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    /// Rewrites the checklist file to all-checked, simulating what a real
    /// `edit_file` tool call would do once the model finishes the item.
    struct FinishTool;

    #[async_trait::async_trait]
    impl Tool for FinishTool {
        fn name(&self) -> &str {
            "finish"
        }
        fn description(&self) -> &str {
            "test-only: checks off every item in the given checklist file"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
        }
        async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
            let path = arguments["path"].as_str().unwrap();
            std::fs::write(path, "- [x] task one\n").unwrap();
            ToolExecOutcome {
                content: "checked off".to_string(),
                is_error: false,
                metadata: serde_json::Value::Null,
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn on_turn_fires_once_per_completed_turn_then_the_loop_idles() {
        let path = temp_md("- [ ] task one\n");
        let provider = OneShotFinishProvider(StdMutex::new(
            vec![tool_use_response(&path), text_response("done")].into(),
        ));
        let tool_ctx = ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        let mut session = minder_core::AgentSession::new(
            Arc::new(provider),
            vec![Arc::new(FinishTool)],
            None,
            "you are a test agent",
            tool_ctx,
        );

        let turn_count = Arc::new(StdMutex::new(0usize));
        let opts = LoopOptions {
            max_iterations: 5,
            query: DEFAULT_QUERY.to_string(),
            poll_interval: Duration::from_millis(10),
        };

        let outcome = tokio::time::timeout(Duration::from_millis(500), {
            let turn_count = turn_count.clone();
            run(&mut session, &path, None, opts, move |_session| {
                *turn_count.lock().unwrap() += 1;
            })
        })
        .await;

        std::fs::remove_file(&path).unwrap();
        assert!(
            outcome.is_err(),
            "expected the idle-poll branch to still be running at the timeout"
        );
        assert_eq!(*turn_count.lock().unwrap(), 1);
    }
}
