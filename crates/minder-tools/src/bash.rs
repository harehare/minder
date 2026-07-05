use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::time::Duration;

const DEFAULT_TIMEOUT_SECS: u64 = 120;

pub struct BashTool;

#[derive(Deserialize)]
struct Args {
    command: String,
    timeout_secs: Option<u64>,
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Runs a shell command and returns its combined stdout/stderr."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to run" },
                "timeout_secs": { "type": "integer", "description": "Timeout in seconds (default 120)" }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };
        let timeout = Duration::from_secs(args.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS));

        let child = match tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&args.command)
            .current_dir(&ctx.working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return error(format!("failed to spawn command: {e}")),
        };
        // captured before the select below moves `child` into wait_with_output()
        let child_id = child.id();

        tokio::select! {
            result = tokio::time::timeout(timeout, child.wait_with_output()) => {
                match result {
                    Ok(Ok(output)) => {
                        let mut content = String::from_utf8_lossy(&output.stdout).into_owned();
                        content.push_str(&String::from_utf8_lossy(&output.stderr));
                        ToolExecOutcome {
                            content,
                            is_error: !output.status.success(),
                            metadata: serde_json::json!({ "exit_code": output.status.code() }),
                        }
                    }
                    Ok(Err(e)) => error(format!("command failed: {e}")),
                    Err(_) => error(format!("command timed out after {}s", timeout.as_secs())),
                }
            }
            _ = ctx.cancel.cancelled() => {
                if let Some(pid) = child_id {
                    let _ = tokio::process::Command::new("kill").arg("-9").arg(pid.to_string()).status().await;
                }
                error("command cancelled".to_string())
            }
        }
    }
}

fn error(message: String) -> ToolExecOutcome {
    ToolExecOutcome {
        content: message,
        is_error: true,
        metadata: serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn runs_a_real_command_and_captures_stdout() {
        let outcome = BashTool
            .execute(serde_json::json!({"command": "echo hello_from_bash"}), &ctx())
            .await;
        assert!(!outcome.is_error);
        assert!(outcome.content.contains("hello_from_bash"));
        assert_eq!(outcome.metadata["exit_code"], 0);
    }

    #[tokio::test]
    async fn nonzero_exit_is_reported_as_error() {
        let outcome = BashTool.execute(serde_json::json!({"command": "exit 3"}), &ctx()).await;
        assert!(outcome.is_error);
        assert_eq!(outcome.metadata["exit_code"], 3);
    }

    #[tokio::test]
    async fn timeout_is_enforced() {
        let outcome = BashTool
            .execute(serde_json::json!({"command": "sleep 5", "timeout_secs": 1}), &ctx())
            .await;
        assert!(outcome.is_error);
        assert!(outcome.content.contains("timed out"));
    }
}
