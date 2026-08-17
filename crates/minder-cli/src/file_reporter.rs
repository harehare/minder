use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use minder_core::{Reporter, ToolCall, ToolExecOutcome, Usage};
use serde_json::json;

use crate::reporter::truncate;

const RESULT_LOG_CHARS: usize = 2000;

/// How `FileReporter` serializes each event.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Default: one human-readable line per event.
    Text,
    /// One JSON object per line (JSON Lines), for scripts to parse.
    Json,
}

impl LogFormat {
    /// `MINDER_LOG_FORMAT=json` opts in; anything else keeps `Text`.
    pub fn from_env() -> Self {
        match std::env::var("MINDER_LOG_FORMAT") {
            Ok(v) if v.eq_ignore_ascii_case("json") => Self::Json,
            _ => Self::Text,
        }
    }
}

/// Appends every reporter event to a log file, independent of the terminal
/// -- keeps a detached run (`minder loop` under `nohup`/systemd/tmux) reviewable after the fact.
pub struct FileReporter {
    file: Mutex<File>,
    format: LogFormat,
}

impl FileReporter {
    pub fn new(path: &Path, format: LogFormat) -> io::Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Mutex::new(file),
            format,
        })
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Writes one event as `[ts] line` (`Text`) or `{"ts", "event", ...fields}` (`Json`).
    fn write_event(&self, name: &str, line: &str, fields: serde_json::Value) {
        let ts = Self::now_secs();
        let rendered = match self.format {
            LogFormat::Text => format!("[{ts}] {line}"),
            LogFormat::Json => {
                let mut obj = json!({"ts": ts, "event": name});
                if let (Some(obj), Some(fields)) = (obj.as_object_mut(), fields.as_object()) {
                    obj.extend(fields.clone());
                }
                obj.to_string()
            }
        };
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{rendered}");
        }
    }
}

#[async_trait]
impl Reporter for FileReporter {
    async fn on_turn_start(&self) {
        self.write_event("turn_start", "turn: waiting on model", json!({}));
    }

    async fn on_assistant_text(&self, text: &str) {
        self.write_event(
            "assistant_text",
            &format!("assistant: {text}"),
            json!({"text": text}),
        );
    }

    async fn on_thinking(&self, text: &str) {
        self.write_event("thinking", &format!("thinking: {text}"), json!({"text": text}));
    }

    async fn on_tool_call(&self, call: &ToolCall) {
        self.write_event(
            "tool_call",
            &format!("tool_call: {}({})", call.name, call.arguments),
            json!({"id": call.id, "name": call.name, "arguments": call.arguments}),
        );
    }

    async fn on_tool_result(&self, call: &ToolCall, outcome: &ToolExecOutcome) {
        let status = if outcome.is_error { "error" } else { "ok" };
        self.write_event(
            "tool_result",
            &format!(
                "tool_result: {} [{status}] {}",
                call.name,
                truncate(&outcome.content, RESULT_LOG_CHARS)
            ),
            json!({
                "id": call.id,
                "name": call.name,
                "is_error": outcome.is_error,
                "content": truncate(&outcome.content, RESULT_LOG_CHARS),
            }),
        );
    }

    async fn on_retry(&self, attempt: usize, max_attempts: usize, delay: Duration, reason: &str) {
        self.write_event(
            "retry",
            &format!("retry {attempt}/{max_attempts} in {delay:?}: {reason}"),
            json!({
                "attempt": attempt,
                "max_attempts": max_attempts,
                "delay_ms": delay.as_millis() as u64,
                "reason": reason,
            }),
        );
    }

    async fn on_usage(&self, usage: &Usage) {
        self.write_event(
            "usage",
            &format!(
                "usage: input={} output={}",
                usage.input_tokens, usage.output_tokens
            ),
            json!({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}),
        );
    }
}

/// Fans every event out to all inner reporters, in order -- used to run the
/// terminal display and file logging side by side without either knowing
/// about the other.
pub struct CompositeReporter(Vec<Arc<dyn Reporter>>);

impl CompositeReporter {
    pub fn new(reporters: Vec<Arc<dyn Reporter>>) -> Self {
        Self(reporters)
    }
}

#[async_trait]
impl Reporter for CompositeReporter {
    async fn on_turn_start(&self) {
        for r in &self.0 {
            r.on_turn_start().await;
        }
    }

    async fn on_turn_end(&self) {
        for r in &self.0 {
            r.on_turn_end().await;
        }
    }

    async fn on_assistant_text(&self, text: &str) {
        for r in &self.0 {
            r.on_assistant_text(text).await;
        }
    }

    async fn on_assistant_text_delta(&self, delta: &str) {
        for r in &self.0 {
            r.on_assistant_text_delta(delta).await;
        }
    }

    async fn on_thinking(&self, text: &str) {
        for r in &self.0 {
            r.on_thinking(text).await;
        }
    }

    async fn on_tool_call(&self, call: &ToolCall) {
        for r in &self.0 {
            r.on_tool_call(call).await;
        }
    }

    async fn on_tool_result(&self, call: &ToolCall, outcome: &ToolExecOutcome) {
        for r in &self.0 {
            r.on_tool_result(call, outcome).await;
        }
    }

    async fn on_retry(&self, attempt: usize, max_attempts: usize, delay: Duration, reason: &str) {
        for r in &self.0 {
            r.on_retry(attempt, max_attempts, delay, reason).await;
        }
    }

    async fn on_usage(&self, usage: &Usage) {
        for r in &self.0 {
            r.on_usage(usage).await;
        }
    }

    async fn on_steering_message(&self, text: &str) {
        for r in &self.0 {
            r.on_steering_message(text).await;
        }
    }

    async fn on_provider_changed(&self, provider_id: &str, model: &str) {
        for r in &self.0 {
            r.on_provider_changed(provider_id, model).await;
        }
    }

    async fn on_notice(&self, text: &str) {
        for r in &self.0 {
            r.on_notice(text).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("minder-file-reporter-test-{}-{name}", uuid::Uuid::new_v4()))
    }

    #[tokio::test]
    async fn writes_assistant_text_and_tool_events_as_lines() {
        let path = scratch_path("log.txt");
        let reporter = FileReporter::new(&path, LogFormat::Text).unwrap();

        reporter.on_turn_start().await;
        reporter.on_assistant_text("hello there").await;
        reporter.on_thinking("mulling it over").await;
        reporter
            .on_tool_call(&ToolCall {
                id: "1".to_string(),
                name: "bash".to_string(),
                arguments: serde_json::json!({"command": "ls"}),
            })
            .await;
        reporter
            .on_tool_result(
                &ToolCall {
                    id: "1".to_string(),
                    name: "bash".to_string(),
                    arguments: serde_json::json!({}),
                },
                &ToolExecOutcome {
                    content: "a.txt".to_string(),
                    is_error: false,
                    metadata: serde_json::Value::Null,
                },
            )
            .await;
        reporter
            .on_usage(&Usage {
                input_tokens: 100,
                output_tokens: 20,
            })
            .await;

        let contents = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert!(contents.contains("turn: waiting on model"));
        assert!(contents.contains("assistant: hello there"));
        assert!(contents.contains("thinking: mulling it over"));
        assert!(contents.contains("tool_call: bash"));
        assert!(contents.contains("tool_result: bash [ok] a.txt"));
        assert!(contents.contains("usage: input=100 output=20"));
    }

    #[tokio::test]
    async fn appends_across_multiple_instances_instead_of_truncating() {
        let path = scratch_path("append.txt");
        FileReporter::new(&path, LogFormat::Text)
            .unwrap()
            .on_assistant_text("first")
            .await;
        FileReporter::new(&path, LogFormat::Text)
            .unwrap()
            .on_assistant_text("second")
            .await;

        let contents = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert!(contents.contains("first"));
        assert!(contents.contains("second"));
    }

    #[tokio::test]
    async fn json_format_emits_one_parseable_object_per_line() {
        let path = scratch_path("log.jsonl");
        let reporter = FileReporter::new(&path, LogFormat::Json).unwrap();

        reporter.on_turn_start().await;
        reporter
            .on_tool_call(&ToolCall {
                id: "1".to_string(),
                name: "bash".to_string(),
                arguments: serde_json::json!({"command": "ls"}),
            })
            .await;
        reporter
            .on_usage(&Usage {
                input_tokens: 100,
                output_tokens: 20,
            })
            .await;

        let contents = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let lines: Vec<serde_json::Value> = contents
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(lines[0]["event"], "turn_start");
        assert_eq!(lines[1]["event"], "tool_call");
        assert_eq!(lines[1]["name"], "bash");
        assert_eq!(lines[1]["arguments"]["command"], "ls");
        assert_eq!(lines[2]["event"], "usage");
        assert_eq!(lines[2]["input_tokens"], 100);
    }

    struct RecordingReporter(Mutex<Vec<String>>);

    #[async_trait]
    impl Reporter for RecordingReporter {
        async fn on_assistant_text(&self, text: &str) {
            self.0.lock().unwrap().push(format!("text:{text}"));
        }
        async fn on_thinking(&self, text: &str) {
            self.0.lock().unwrap().push(format!("thinking:{text}"));
        }
        async fn on_usage(&self, usage: &Usage) {
            self.0
                .lock()
                .unwrap()
                .push(format!("usage:{}/{}", usage.input_tokens, usage.output_tokens));
        }
        async fn on_assistant_text_delta(&self, delta: &str) {
            self.0.lock().unwrap().push(format!("delta:{delta}"));
        }
        async fn on_steering_message(&self, text: &str) {
            self.0.lock().unwrap().push(format!("steering:{text}"));
        }
        async fn on_provider_changed(&self, provider_id: &str, model: &str) {
            self.0.lock().unwrap().push(format!("provider:{provider_id}/{model}"));
        }
    }

    #[tokio::test]
    async fn composite_reporter_fans_out_to_every_inner_reporter() {
        let a = Arc::new(RecordingReporter(Mutex::new(Vec::new())));
        let b = Arc::new(RecordingReporter(Mutex::new(Vec::new())));
        let composite = CompositeReporter::new(vec![a.clone(), b.clone()]);

        composite.on_assistant_text("hi").await;
        composite.on_assistant_text_delta("h").await;
        composite.on_thinking("hmm").await;
        composite.on_steering_message("wait").await;
        composite.on_provider_changed("anthropic", "claude").await;
        composite
            .on_usage(&Usage {
                input_tokens: 5,
                output_tokens: 1,
            })
            .await;

        let expected = [
            "text:hi",
            "delta:h",
            "thinking:hmm",
            "steering:wait",
            "provider:anthropic/claude",
            "usage:5/1",
        ];
        assert_eq!(a.0.lock().unwrap().as_slice(), expected);
        assert_eq!(b.0.lock().unwrap().as_slice(), expected);
    }
}
