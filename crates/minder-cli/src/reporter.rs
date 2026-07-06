use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use minder_core::{HookPort, RenderDecision, Reporter, ToolCall, ToolExecOutcome};
use tokio::sync::Mutex;

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";

const RESULT_PREVIEW_CHARS: usize = 300;
const MAX_DIFF_LINES: usize = 40;

const SPINNER_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_INTERVAL: Duration = Duration::from_millis(90);
const TURN_LABEL_KEY: &str = "__turn__";

/// Active spinner labels keyed by an id (tool call id, or a fixed key for
/// "waiting on the model") -- shared between the ticker task and whichever
/// reporter method starts/stops a label, and also used as a lock to
/// serialize real output against spinner redraws.
struct SpinnerState {
    labels: HashMap<String, (String, Instant)>,
    frame: usize,
}

/// Prints live progress to the terminal as a turn runs: assistant text goes
/// to stdout (it's the actual conversation), tool calls/results/diffs go to
/// stderr (execution trace), matching the existing convention of `eprintln!`
/// for setup/diagnostic lines in `main.rs`. Colors itself off automatically
/// when stderr isn't a tty or `NO_COLOR` is set.
///
/// When stderr is a tty, a spinner also runs on stderr while waiting on the
/// model or a tool call, so a long-running step never looks stalled -- see
/// `start_spinner`/`stop_spinner`/`with_terminal_lock`.
///
/// Before falling back to its own formatting, each event is offered to
/// `hooks` (the *same* `HookPort` handle `AgentSession` uses for policy --
/// see `main.rs::build_session`) via `render_tool_call`/`render_tool_result`,
/// so a `.agent/hooks/*.mq` script can hide, retext, or restyle any line
/// without minder-side code changes.
pub struct TerminalReporter {
    color: bool,
    interactive: bool,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
    spinner: Arc<Mutex<SpinnerState>>,
}

impl TerminalReporter {
    pub fn new(hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>) -> Self {
        let color = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        let interactive = std::io::stderr().is_terminal();
        let spinner = Arc::new(Mutex::new(SpinnerState {
            labels: HashMap::new(),
            frame: 0,
        }));

        if interactive {
            let spinner = spinner.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(SPINNER_INTERVAL).await;
                    let mut state = spinner.lock().await;
                    if state.labels.is_empty() {
                        continue;
                    }
                    state.frame = (state.frame + 1) % SPINNER_FRAMES.len();
                    let frame = SPINNER_FRAMES[state.frame];
                    let elapsed = state
                        .labels
                        .values()
                        .map(|(_, t)| t.elapsed())
                        .max()
                        .unwrap_or_default();
                    let mut names: Vec<&str> = state.labels.values().map(|(l, _)| l.as_str()).collect();
                    names.sort_unstable();
                    eprint!(
                        "\r\x1b[2K{DIM}{frame} {} ({:.1}s){RESET}",
                        names.join(", "),
                        elapsed.as_secs_f32()
                    );
                    let _ = std::io::stderr().flush();
                }
            });
        }

        Self {
            color,
            interactive,
            hooks,
            spinner,
        }
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color {
            format!("{code}{text}{RESET}")
        } else {
            text.to_string()
        }
    }

    /// Maps a hook-supplied style name to this reporter's own ANSI codes --
    /// `style` is deliberately just a string across the Rust/mq boundary
    /// (see `RenderDecision`), so an unrecognized or absent name is simply
    /// unstyled rather than an error.
    fn painted_by_name(&self, style: Option<&str>, text: &str) -> String {
        let code = match style {
            Some("green") => GREEN,
            Some("red") => RED,
            Some("yellow") => YELLOW,
            Some("cyan") => CYAN,
            Some("dim") => DIM,
            Some("bold") => BOLD,
            _ => return text.to_string(),
        };
        self.paint(code, text)
    }

    fn color_diff_line(&self, line: &str) -> String {
        if line.starts_with("+++") || line.starts_with("---") {
            self.paint(BOLD, line)
        } else if line.starts_with("@@") {
            self.paint(CYAN, line)
        } else if line.starts_with('+') {
            self.paint(GREEN, line)
        } else if line.starts_with('-') {
            self.paint(RED, line)
        } else {
            self.paint(DIM, line)
        }
    }

    /// Colors, indents (so it visually nests under the result's stat line),
    /// and caps a unified diff at `MAX_DIFF_LINES` so one big rewrite can't
    /// flood the terminal.
    fn render_diff(&self, unified: &str) -> String {
        let lines: Vec<&str> = unified.lines().collect();
        let shown = &lines[..lines.len().min(MAX_DIFF_LINES)];
        let mut rendered: Vec<String> = shown
            .iter()
            .map(|line| format!("  {}", self.color_diff_line(line)))
            .collect();
        if lines.len() > MAX_DIFF_LINES {
            rendered.push(format!(
                "  {}",
                self.paint(DIM, &format!("… {} more line(s)", lines.len() - MAX_DIFF_LINES))
            ));
        }
        rendered.join("\n")
    }

    async fn render_call_decision(&self, call: &ToolCall) -> RenderDecision {
        let Some(hooks) = &self.hooks else {
            return RenderDecision::Default;
        };
        hooks.lock().await.render_tool_call(call).await
    }

    async fn render_result_decision(&self, call: &ToolCall, outcome: &ToolExecOutcome) -> RenderDecision {
        let Some(hooks) = &self.hooks else {
            return RenderDecision::Default;
        };
        hooks.lock().await.render_tool_result(call, outcome).await
    }

    fn format_default_call(&self, call: &ToolCall) -> String {
        let summary = summarize_args(&call.arguments);
        let header = if summary.is_empty() {
            call.name.clone()
        } else {
            format!("{}({summary})", call.name)
        };
        self.paint(BOLD, &format!("● {header}"))
    }

    fn format_default_result(&self, outcome: &ToolExecOutcome, elapsed: Option<Duration>) -> String {
        let mark = if outcome.is_error {
            self.paint(RED, "✗")
        } else {
            self.paint(GREEN, "✓")
        };
        let suffix = elapsed
            .filter(|d| d.as_secs_f32() >= 1.0)
            .map(|d| format!(" {}", self.paint(DIM, &format!("({:.1}s)", d.as_secs_f32()))))
            .unwrap_or_default();

        if let Some(diff) = outcome.metadata.get("diff").and_then(|v| v.as_str())
            && !diff.is_empty()
        {
            let additions = outcome.metadata.get("additions").and_then(|v| v.as_u64());
            let deletions = outcome.metadata.get("deletions").and_then(|v| v.as_u64());
            format!(
                "  {mark} {} {}{suffix}\n{}",
                self.paint(GREEN, &format!("+{}", additions.unwrap_or(0))),
                self.paint(RED, &format!("-{}", deletions.unwrap_or(0))),
                self.render_diff(diff),
            )
        } else {
            format!(
                "  {mark} {}{suffix}",
                self.paint(DIM, &truncate(&outcome.content, RESULT_PREVIEW_CHARS))
            )
        }
    }

    /// Registers a live spinner label; no-op when stderr isn't a tty.
    async fn start_spinner(&self, key: &str, label: String) {
        if !self.interactive {
            return;
        }
        self.spinner
            .lock()
            .await
            .labels
            .insert(key.to_string(), (label, Instant::now()));
    }

    /// Clears a spinner label, returning how long it was active.
    async fn stop_spinner(&self, key: &str) -> Option<Duration> {
        if !self.interactive {
            return None;
        }
        let mut state = self.spinner.lock().await;
        let elapsed = state.labels.remove(key).map(|(_, started)| started.elapsed());
        if state.labels.is_empty() {
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
        }
        elapsed
    }

    /// Prints a line while holding the spinner lock, so the ticker task
    /// can't redraw mid-print and the spinner's current line gets cleared
    /// first (stdout and stderr share one terminal row when both are ttys).
    async fn print_guarded(&self, f: impl FnOnce()) {
        if self.interactive {
            let _guard = self.spinner.lock().await;
            eprint!("\r\x1b[2K");
            f();
        } else {
            f();
        }
    }
}

#[async_trait]
impl Reporter for TerminalReporter {
    async fn on_turn_start(&self) {
        self.start_spinner(TURN_LABEL_KEY, "Thinking".to_string()).await;
    }

    async fn on_turn_end(&self) {
        self.stop_spinner(TURN_LABEL_KEY).await;
    }

    async fn on_assistant_text(&self, text: &str) {
        let rendered = crate::markdown::render(text, self.color);
        self.print_guarded(|| println!("{rendered}")).await;
    }

    async fn on_tool_call(&self, call: &ToolCall) {
        match self.render_call_decision(call).await {
            RenderDecision::Hide => {}
            RenderDecision::Text { value, style } => {
                let line = self.painted_by_name(style.as_deref(), &value);
                self.print_guarded(|| eprintln!("{line}")).await;
            }
            RenderDecision::Default => {
                let line = self.format_default_call(call);
                self.print_guarded(|| eprintln!("{line}")).await;
            }
        }
        let label = format!("Running {}", call.name);
        self.start_spinner(&call.id, label).await;
    }

    async fn on_tool_result(&self, call: &ToolCall, outcome: &ToolExecOutcome) {
        let elapsed = self.stop_spinner(&call.id).await;
        match self.render_result_decision(call, outcome).await {
            RenderDecision::Hide => {}
            RenderDecision::Text { value, style } => {
                let line = self.painted_by_name(style.as_deref(), &value);
                self.print_guarded(|| eprintln!("{line}")).await;
            }
            RenderDecision::Default => {
                let line = self.format_default_result(outcome, elapsed);
                self.print_guarded(|| eprintln!("{line}")).await;
            }
        }
    }

    async fn on_retry(&self, attempt: usize, max_attempts: usize, delay: Duration, reason: &str) {
        let line = self.paint(
            YELLOW,
            &format!(
                "retrying in {:.0}s (attempt {attempt}/{max_attempts}): {reason}",
                delay.as_secs_f32()
            ),
        );
        self.print_guarded(|| eprintln!("{line}")).await;
    }
}

/// Picks out the argument most useful for a one-line "what is this tool call
/// doing" summary, falling back to a truncated compact JSON dump.
fn summarize_args(args: &serde_json::Value) -> String {
    if let Some(obj) = args.as_object() {
        for key in ["path", "command", "pattern", "url", "name", "query"] {
            if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                return format!("{key}={v}");
            }
        }
    }
    truncate(&args.to_string(), 100)
}

pub(crate) fn truncate(s: &str, max_chars: usize) -> String {
    let mut truncated: String = s.chars().take(max_chars).collect();
    if s.chars().count() > max_chars {
        truncated.push_str("...");
    }
    truncated.replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::{HookDecision, Message, ToolCallDecision, ToolResultInfo};

    #[test]
    fn summarizes_path_argument() {
        let args = serde_json::json!({"path": "src/main.rs", "content": "..."});
        assert_eq!(summarize_args(&args), "path=src/main.rs");
    }

    #[test]
    fn falls_back_to_truncated_json() {
        let args = serde_json::json!({"foo": "bar"});
        assert_eq!(summarize_args(&args), r#"{"foo":"bar"}"#);
    }

    #[test]
    fn truncate_collapses_newlines_and_caps_length() {
        let s = "a\nb\nc".repeat(50);
        let out = truncate(&s, 10);
        assert!(!out.contains('\n'));
        assert!(out.ends_with("..."));
    }

    fn no_color_reporter(hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>) -> TerminalReporter {
        // color/interactive are decided once in `new()` from the tty check,
        // which is meaningless (and flaky) under `cargo test` -- tests only
        // care about the plain-text `value`, so force both off directly.
        TerminalReporter {
            color: false,
            interactive: false,
            hooks,
            spinner: Arc::new(Mutex::new(SpinnerState {
                labels: HashMap::new(),
                frame: 0,
            })),
        }
    }

    struct StubHooks {
        call_decision: RenderDecision,
        result_decision: RenderDecision,
    }

    #[async_trait]
    impl HookPort for StubHooks {
        async fn before_agent_start(&mut self, system_prompt: &str) -> HookDecision<String> {
            HookDecision::Allow(system_prompt.to_string())
        }
        async fn on_context(&mut self, messages: &[Message]) -> HookDecision<Vec<Message>> {
            HookDecision::Allow(messages.to_vec())
        }
        async fn on_tool_call(&mut self, call: &ToolCall) -> ToolCallDecision {
            ToolCallDecision::Allow(call.clone())
        }
        async fn on_tool_result(&mut self, result: &ToolResultInfo) -> HookDecision<String> {
            HookDecision::Allow(result.content.clone())
        }
        async fn before_compact(&mut self, _messages: &[Message]) -> HookDecision<()> {
            HookDecision::Allow(())
        }
        async fn render_tool_call(&mut self, _call: &ToolCall) -> RenderDecision {
            self.call_decision.clone()
        }
        async fn render_tool_result(&mut self, _call: &ToolCall, _outcome: &ToolExecOutcome) -> RenderDecision {
            self.result_decision.clone()
        }
    }

    fn call() -> ToolCall {
        ToolCall {
            id: "1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        }
    }

    fn outcome() -> ToolExecOutcome {
        ToolExecOutcome {
            content: "a.txt".to_string(),
            is_error: false,
            metadata: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn no_hooks_configured_always_defaults() {
        let reporter = no_color_reporter(None);
        assert!(matches!(
            reporter.render_call_decision(&call()).await,
            RenderDecision::Default
        ));
        assert!(matches!(
            reporter.render_result_decision(&call(), &outcome()).await,
            RenderDecision::Default
        ));
    }

    #[tokio::test]
    async fn hook_hide_decision_is_honored() {
        let hooks: Box<dyn HookPort> = Box::new(StubHooks {
            call_decision: RenderDecision::Hide,
            result_decision: RenderDecision::Default,
        });
        let reporter = no_color_reporter(Some(Arc::new(tokio::sync::Mutex::new(hooks))));
        assert!(matches!(
            reporter.render_call_decision(&call()).await,
            RenderDecision::Hide
        ));
    }

    #[tokio::test]
    async fn hook_text_decision_is_honored() {
        let hooks: Box<dyn HookPort> = Box::new(StubHooks {
            call_decision: RenderDecision::Default,
            result_decision: RenderDecision::Text {
                value: "custom".to_string(),
                style: Some("green".to_string()),
            },
        });
        let reporter = no_color_reporter(Some(Arc::new(tokio::sync::Mutex::new(hooks))));
        match reporter.render_result_decision(&call(), &outcome()).await {
            RenderDecision::Text { value, style } => {
                assert_eq!(value, "custom");
                assert_eq!(style.as_deref(), Some("green"));
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn painted_by_name_falls_back_to_plain_for_unknown_style() {
        let reporter = no_color_reporter(None);
        assert_eq!(reporter.painted_by_name(Some("not-a-color"), "x"), "x");
        assert_eq!(reporter.painted_by_name(None, "x"), "x");
    }

    #[test]
    fn render_diff_truncates_long_diffs() {
        let reporter = no_color_reporter(None);
        let long_diff = (0..100).map(|i| format!(" line {i}")).collect::<Vec<_>>().join("\n");
        let rendered = reporter.render_diff(&long_diff);
        assert!(rendered.contains("more line(s)"));
        assert_eq!(rendered.lines().count(), MAX_DIFF_LINES + 1);
    }

    #[tokio::test]
    async fn spinner_is_a_noop_when_not_interactive() {
        let reporter = no_color_reporter(None);
        reporter.start_spinner("k", "Running".to_string()).await;
        assert!(reporter.spinner.lock().await.labels.is_empty());
        assert!(reporter.stop_spinner("k").await.is_none());
    }

    #[tokio::test]
    async fn elapsed_under_a_second_is_not_shown_in_result_line() {
        let reporter = no_color_reporter(None);
        let line = reporter.format_default_result(&outcome(), Some(Duration::from_millis(200)));
        assert!(!line.contains('('), "unexpected elapsed suffix in: {line}");
    }

    #[test]
    fn elapsed_over_a_second_is_shown_in_result_line() {
        let reporter = no_color_reporter(None);
        let line = reporter.format_default_result(&outcome(), Some(Duration::from_secs(3)));
        assert!(line.contains("(3.0s)"), "missing elapsed suffix in: {line}");
    }
}
