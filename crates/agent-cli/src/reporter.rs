use std::io::IsTerminal;

use agent_core::{Reporter, ToolCall, ToolExecOutcome};
use async_trait::async_trait;

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";

const RESULT_PREVIEW_CHARS: usize = 300;

/// Prints live progress to the terminal as a turn runs: assistant text goes
/// to stdout (it's the actual conversation), tool calls/results/diffs go to
/// stderr (execution trace), matching the existing convention of `eprintln!`
/// for setup/diagnostic lines in `main.rs`. Colors itself off automatically
/// when stderr isn't a tty or `NO_COLOR` is set.
pub struct TerminalReporter {
    color: bool,
}

impl TerminalReporter {
    pub fn new() -> Self {
        let color = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Self { color }
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color {
            format!("{code}{text}{RESET}")
        } else {
            text.to_string()
        }
    }

    fn render_diff(&self, unified: &str) -> String {
        unified
            .lines()
            .map(|line| {
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
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for TerminalReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Reporter for TerminalReporter {
    async fn on_assistant_text(&self, text: &str) {
        println!("{text}");
    }

    async fn on_tool_call(&self, call: &ToolCall) {
        let summary = summarize_args(&call.arguments);
        let line = if summary.is_empty() {
            call.name.clone()
        } else {
            format!("{} {summary}", call.name)
        };
        eprintln!("{}", self.paint(DIM, &format!("→ {line}")));
    }

    async fn on_tool_result(&self, call: &ToolCall, outcome: &ToolExecOutcome) {
        if let Some(diff) = outcome.metadata.get("diff").and_then(|v| v.as_str())
            && !diff.is_empty()
        {
            eprintln!("{}", self.render_diff(diff));
            return;
        }

        let mark = if outcome.is_error {
            self.paint(RED, "✗")
        } else {
            self.paint(GREEN, "✓")
        };
        eprintln!(
            "{mark} {}: {}",
            call.name,
            self.paint(DIM, &truncate(&outcome.content, RESULT_PREVIEW_CHARS))
        );
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
}
