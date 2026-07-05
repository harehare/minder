use std::io::IsTerminal;
use std::sync::Arc;

use async_trait::async_trait;
use minder_core::{HookPort, RenderDecision, Reporter, ToolCall, ToolExecOutcome};

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";

const RESULT_PREVIEW_CHARS: usize = 300;
const MAX_DIFF_LINES: usize = 40;

/// Prints live progress to the terminal as a turn runs: assistant text goes
/// to stdout (it's the actual conversation), tool calls/results/diffs go to
/// stderr (execution trace), matching the existing convention of `eprintln!`
/// for setup/diagnostic lines in `main.rs`. Colors itself off automatically
/// when stderr isn't a tty or `NO_COLOR` is set.
///
/// Before falling back to its own formatting, each event is offered to
/// `hooks` (the *same* `HookPort` handle `AgentSession` uses for policy --
/// see `main.rs::build_session`) via `render_tool_call`/`render_tool_result`,
/// so a `.agent/hooks/*.mq` script can hide, retext, or restyle any line
/// without minder-side code changes.
pub struct TerminalReporter {
    color: bool,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
}

impl TerminalReporter {
    pub fn new(hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>) -> Self {
        let color = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Self { color, hooks }
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

    fn print_default_call(&self, call: &ToolCall) {
        let summary = summarize_args(&call.arguments);
        let header = if summary.is_empty() {
            call.name.clone()
        } else {
            format!("{}({summary})", call.name)
        };
        eprintln!("{}", self.paint(BOLD, &format!("● {header}")));
    }

    fn print_default_result(&self, outcome: &ToolExecOutcome) {
        let mark = if outcome.is_error {
            self.paint(RED, "✗")
        } else {
            self.paint(GREEN, "✓")
        };

        if let Some(diff) = outcome.metadata.get("diff").and_then(|v| v.as_str())
            && !diff.is_empty()
        {
            let additions = outcome.metadata.get("additions").and_then(|v| v.as_u64());
            let deletions = outcome.metadata.get("deletions").and_then(|v| v.as_u64());
            eprintln!(
                "  {mark} {} {}",
                self.paint(GREEN, &format!("+{}", additions.unwrap_or(0))),
                self.paint(RED, &format!("-{}", deletions.unwrap_or(0))),
            );
            eprintln!("{}", self.render_diff(diff));
            return;
        }

        eprintln!(
            "  {mark} {}",
            self.paint(DIM, &truncate(&outcome.content, RESULT_PREVIEW_CHARS))
        );
    }
}

#[async_trait]
impl Reporter for TerminalReporter {
    async fn on_assistant_text(&self, text: &str) {
        println!("{}", crate::markdown::render(text, self.color));
    }

    async fn on_tool_call(&self, call: &ToolCall) {
        match self.render_call_decision(call).await {
            RenderDecision::Hide => {}
            RenderDecision::Text { value, style } => {
                eprintln!("{}", self.painted_by_name(style.as_deref(), &value));
            }
            RenderDecision::Default => self.print_default_call(call),
        }
    }

    async fn on_tool_result(&self, call: &ToolCall, outcome: &ToolExecOutcome) {
        match self.render_result_decision(call, outcome).await {
            RenderDecision::Hide => {}
            RenderDecision::Text { value, style } => {
                eprintln!("{}", self.painted_by_name(style.as_deref(), &value));
            }
            RenderDecision::Default => self.print_default_result(outcome),
        }
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

fn truncate(s: &str, max_chars: usize) -> String {
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
        // color is decided once in `new()` from the tty/NO_COLOR check, which
        // is meaningless (and flaky) under `cargo test` -- tests only care
        // about the plain-text `value`, so force it off directly.
        TerminalReporter { color: false, hooks }
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
}
