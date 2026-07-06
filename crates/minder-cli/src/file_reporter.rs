use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use minder_core::{Reporter, ToolCall, ToolExecOutcome};

use crate::reporter::truncate;

const RESULT_LOG_CHARS: usize = 2000;

/// Appends every reporter event as a plain-text line to a log file,
/// independent of the terminal -- so a detached/unattended run (`minder
/// loop` under `nohup`/systemd/tmux) stays reviewable after the fact. No
/// color, no spinner, no hook-driven customization: always the raw default
/// view, since a log should show ground truth regardless of display hooks.
pub struct FileReporter {
    file: Mutex<File>,
}

impl FileReporter {
    pub fn new(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file: Mutex::new(file) })
    }

    fn write_line(&self, line: &str) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "[{ts}] {line}");
        }
    }
}

#[async_trait]
impl Reporter for FileReporter {
    async fn on_turn_start(&self) {
        self.write_line("turn: waiting on model");
    }

    async fn on_assistant_text(&self, text: &str) {
        self.write_line(&format!("assistant: {text}"));
    }

    async fn on_tool_call(&self, call: &ToolCall) {
        self.write_line(&format!("tool_call: {}({})", call.name, call.arguments));
    }

    async fn on_tool_result(&self, call: &ToolCall, outcome: &ToolExecOutcome) {
        let status = if outcome.is_error { "error" } else { "ok" };
        self.write_line(&format!(
            "tool_result: {} [{status}] {}",
            call.name,
            truncate(&outcome.content, RESULT_LOG_CHARS)
        ));
    }

    async fn on_retry(&self, attempt: usize, max_attempts: usize, delay: Duration, reason: &str) {
        self.write_line(&format!("retry {attempt}/{max_attempts} in {delay:?}: {reason}"));
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
        let reporter = FileReporter::new(&path).unwrap();

        reporter.on_turn_start().await;
        reporter.on_assistant_text("hello there").await;
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

        let contents = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert!(contents.contains("turn: waiting on model"));
        assert!(contents.contains("assistant: hello there"));
        assert!(contents.contains("tool_call: bash"));
        assert!(contents.contains("tool_result: bash [ok] a.txt"));
    }

    #[tokio::test]
    async fn appends_across_multiple_instances_instead_of_truncating() {
        let path = scratch_path("append.txt");
        FileReporter::new(&path).unwrap().on_assistant_text("first").await;
        FileReporter::new(&path).unwrap().on_assistant_text("second").await;

        let contents = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert!(contents.contains("first"));
        assert!(contents.contains("second"));
    }

    struct RecordingReporter(Mutex<Vec<String>>);

    #[async_trait]
    impl Reporter for RecordingReporter {
        async fn on_assistant_text(&self, text: &str) {
            self.0.lock().unwrap().push(text.to_string());
        }
    }

    #[tokio::test]
    async fn composite_reporter_fans_out_to_every_inner_reporter() {
        let a = Arc::new(RecordingReporter(Mutex::new(Vec::new())));
        let b = Arc::new(RecordingReporter(Mutex::new(Vec::new())));
        let composite = CompositeReporter::new(vec![a.clone(), b.clone()]);

        composite.on_assistant_text("hi").await;

        assert_eq!(a.0.lock().unwrap().as_slice(), ["hi"]);
        assert_eq!(b.0.lock().unwrap().as_slice(), ["hi"]);
    }
}
