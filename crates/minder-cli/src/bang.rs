use std::path::Path;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_OUTPUT_CHARS: usize = 20_000;

pub struct BangResult {
    pub output: String,
    pub success: bool,
}

pub async fn run(command: &str, dir: &Path) -> BangResult {
    let child = match tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return BangResult {
                output: format!("failed to spawn command: {e}"),
                success: false,
            };
        }
    };

    match tokio::time::timeout(TIMEOUT, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
            combined.push_str(&String::from_utf8_lossy(&output.stderr));
            let truncated = combined.chars().count() > MAX_OUTPUT_CHARS;
            let combined: String = combined.chars().take(MAX_OUTPUT_CHARS).collect();
            BangResult {
                output: if truncated {
                    format!("{combined}\n... (truncated to the first {MAX_OUTPUT_CHARS} characters)")
                } else {
                    combined
                },
                success: output.status.success(),
            }
        }
        Ok(Err(e)) => BangResult {
            output: format!("command failed: {e}"),
            success: false,
        },
        Err(_) => BangResult {
            output: format!("command timed out after {}s", TIMEOUT.as_secs()),
            success: false,
        },
    }
}

pub fn format_for_agent(command: &str, result: &BangResult) -> String {
    let status = if result.success { "" } else { " (exit status: failed)" };
    format!(
        "I ran `{command}`{status}, here's the output:\n\n```\n{}\n```",
        result.output
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn captures_combined_stdout_and_stderr() {
        let result = run("echo out; echo err 1>&2", &std::env::temp_dir()).await;
        assert!(result.success);
        assert!(result.output.contains("out"));
        assert!(result.output.contains("err"));
    }

    #[tokio::test]
    async fn success_is_false_on_a_nonzero_exit() {
        let result = run("exit 1", &std::env::temp_dir()).await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn long_output_is_truncated() {
        let result = run("yes x | head -c 30000", &std::env::temp_dir()).await;
        assert!(result.output.contains("truncated to the first"));
        assert!(result.output.chars().count() < 30_000);
    }

    #[test]
    fn format_for_agent_notes_a_failed_exit() {
        let result = BangResult {
            output: "boom".to_string(),
            success: false,
        };
        let formatted = format_for_agent("false", &result);
        assert!(formatted.contains("failed"));
        assert!(formatted.contains("boom"));
    }
}
