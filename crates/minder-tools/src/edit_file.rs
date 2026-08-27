use crate::diff::diff_files;
use async_trait::async_trait;
use minder_core::{Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;

pub struct EditFileTool;

#[derive(Deserialize)]
struct Args {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replaces `old_string` with `new_string` in a file. `old_string` must match exactly \
         once unless `replace_all` is true."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path, absolute or relative to the working directory" },
                "old_string": { "type": "string", "description": "Exact text to replace" },
                "new_string": { "type": "string", "description": "Replacement text" },
                "replace_all": { "type": "boolean", "description": "Replace every occurrence instead of requiring exactly one (default false)" }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let path = ctx.working_dir.join(&args.path);
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return error(format!("failed to read {}: {e}", path.display())),
        };

        let (ranges, fuzzy) = find_occurrences(&content, &args.old_string);
        let occurrences = ranges.len();
        if occurrences == 0 {
            return error(format!("old_string not found in {}", path.display()));
        }
        if occurrences > 1 && !args.replace_all {
            return error(format!(
                "old_string matches {occurrences} times in {} -- pass replace_all: true, or narrow old_string to a unique match",
                path.display()
            ));
        }

        let to_replace = if args.replace_all { &ranges[..] } else { &ranges[..1] };
        let mut new_content = String::with_capacity(content.len());
        let mut last_end = 0;
        for range in to_replace {
            new_content.push_str(&content[last_end..range.start]);
            new_content.push_str(&args.new_string);
            last_end = range.end;
        }
        new_content.push_str(&content[last_end..]);

        match tokio::fs::write(&path, &new_content).await {
            Ok(()) => {
                let diff = diff_files(&args.path, &content, &new_content);
                let suffix = if fuzzy { " (matched ignoring trailing whitespace)" } else { "" };
                ToolExecOutcome {
                    content: format!("replaced {occurrences} occurrence(s) in {}{suffix}", path.display()),
                    is_error: false,
                    metadata: serde_json::json!({
                        "occurrences": occurrences,
                        "diff": diff.unified,
                        "additions": diff.additions,
                        "deletions": diff.deletions,
                    }),
                }
            }
            Err(e) => error(format!("failed to write {}: {e}", path.display())),
        }
    }
}

/// Byte ranges of every occurrence of `needle`, preferring an exact match;
/// if none exist, falls back to matching line-by-line while ignoring each
/// line's trailing whitespace/newline style -- the most common near-miss
/// from a smaller model reproducing a block with slightly different
/// trailing spaces. Never touches leading whitespace (indentation).
/// Returns `(ranges, used_fallback)`.
fn find_occurrences(haystack: &str, needle: &str) -> (Vec<std::ops::Range<usize>>, bool) {
    let exact: Vec<_> = haystack.match_indices(needle).map(|(i, m)| i..i + m.len()).collect();
    if !exact.is_empty() {
        return (exact, false);
    }
    (find_occurrences_trailing_ws_tolerant(haystack, needle), true)
}

fn find_occurrences_trailing_ws_tolerant(haystack: &str, needle: &str) -> Vec<std::ops::Range<usize>> {
    let h_lines: Vec<&str> = haystack.split_inclusive('\n').collect();
    let n_lines: Vec<&str> = needle.split_inclusive('\n').collect();
    if n_lines.is_empty() || h_lines.len() < n_lines.len() {
        return Vec::new();
    }

    // `starts[i]` is line `i`'s byte offset; the sentinel at the end lets
    // `starts[i + n_lines.len()]` address "one past the window" uniformly.
    let mut starts = Vec::with_capacity(h_lines.len() + 1);
    let mut offset = 0;
    for line in &h_lines {
        starts.push(offset);
        offset += line.len();
    }
    starts.push(offset);

    (0..=h_lines.len() - n_lines.len())
        .filter(|&i| {
            h_lines[i..i + n_lines.len()]
                .iter()
                .zip(&n_lines)
                .all(|(h, n)| h.trim_end() == n.trim_end())
        })
        .map(|i| starts[i]..starts[i + n_lines.len()])
        .collect()
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

    async fn ctx_with_file(content: &str) -> ToolContext {
        let dir = std::env::temp_dir().join(format!("minder-edit-file-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("a.txt"), content).await.unwrap();
        ToolContext {
            working_dir: dir,
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
            mailbox: None,
        }
    }

    #[tokio::test]
    async fn replaces_a_unique_match() {
        let ctx = ctx_with_file("hello world").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "world", "new_string": "there"}),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error);
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt")).await.unwrap(),
            "hello there"
        );
    }

    #[tokio::test]
    async fn rejects_ambiguous_match_without_replace_all() {
        let ctx = ctx_with_file("foo foo foo").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "foo", "new_string": "bar"}),
                &ctx,
            )
            .await;
        assert!(outcome.is_error);
        assert!(outcome.content.contains("3 times"));
    }

    #[tokio::test]
    async fn replace_all_replaces_every_occurrence() {
        let ctx = ctx_with_file("foo foo foo").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "foo", "new_string": "bar", "replace_all": true}),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error);
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt")).await.unwrap(),
            "bar bar bar"
        );
    }

    #[tokio::test]
    async fn missing_old_string_is_an_error() {
        let ctx = ctx_with_file("hello world").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "xyz", "new_string": "abc"}),
                &ctx,
            )
            .await;
        assert!(outcome.is_error);
    }

    #[tokio::test]
    async fn a_trailing_whitespace_mismatch_still_matches_via_fallback() {
        let ctx = ctx_with_file("fn foo() {\n    let x = 1;  \n    let y = 2;\n}\n").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({
                    "path": "a.txt",
                    "old_string": "    let x = 1;\n    let y = 2;\n",
                    "new_string": "    let x = 1;\n    let z = 3;\n",
                }),
                &ctx,
            )
            .await;
        assert!(!outcome.is_error, "{}", outcome.content);
        assert!(outcome.content.contains("ignoring trailing whitespace"));
        assert_eq!(
            tokio::fs::read_to_string(ctx.working_dir.join("a.txt")).await.unwrap(),
            "fn foo() {\n    let x = 1;\n    let z = 3;\n}\n"
        );
    }

    #[tokio::test]
    async fn a_leading_whitespace_mismatch_is_not_tolerated() {
        // 2-space indent in the file, 4-space indent in `old_string` -- not a
        // literal substring either way, so this must fail even with the
        // trailing-whitespace-tolerant fallback (which never touches leading
        // whitespace).
        let ctx = ctx_with_file("  let x = 1;\n").await;
        let outcome = EditFileTool
            .execute(
                serde_json::json!({"path": "a.txt", "old_string": "    let x = 1;\n", "new_string": "    let y = 2;\n"}),
                &ctx,
            )
            .await;
        assert!(outcome.is_error);
    }
}
