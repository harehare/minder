//! `@path` mentions: attaches a file/directory's contents to the prompt,
//! shared by the REPL and one-shot/`--continue`/`--resume` task strings.
//! A mention only expands if it resolves to a real path on disk.

use std::path::{Path, PathBuf};

use rustyline::completion::Pair;

/// Per-file truncation limit, same rationale as `main.rs`'s `MAX_STDIN_CHARS`.
const MAX_MENTION_FILE_CHARS: usize = 100_000;

/// Max entries listed for a directory mention (one level deep).
const MAX_DIR_ENTRIES: usize = 200;

/// Appends an attachment for every resolvable `@path` mention in `text`;
/// returns `text` unchanged if none resolve.
pub fn expand_mentions(text: &str, working_dir: &Path) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut attachments = Vec::new();

    for token in mention_tokens(text) {
        let Some(resolved) = resolve_mention_path(token, working_dir) else {
            continue;
        };
        if !seen.insert(resolved.clone()) {
            continue;
        }
        if let Some(block) = render_attachment(&resolved, working_dir) {
            attachments.push(block);
        }
    }

    if attachments.is_empty() {
        return text.to_string();
    }

    format!("{text}\n\n---\nAttached via @mention:\n\n{}", attachments.join("\n\n"))
}

/// `@`-prefixed words with the `@` and trailing sentence punctuation stripped.
fn mention_tokens(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .filter_map(|word| word.strip_prefix('@'))
        .map(|raw| raw.trim_end_matches([',', '.', ';', ':', '!', '?', ')', ']', '}', '\'', '"']))
        .filter(|s| !s.is_empty())
        .collect()
}

/// Resolves a token to an absolute path, `None` if nothing exists there.
fn resolve_mention_path(token: &str, working_dir: &Path) -> Option<PathBuf> {
    let path = if let Some(rest) = token.strip_prefix("~/") {
        PathBuf::from(std::env::var("HOME").ok()?).join(rest)
    } else if token == "~" {
        PathBuf::from(std::env::var("HOME").ok()?)
    } else {
        let candidate = Path::new(token);
        if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            working_dir.join(candidate)
        }
    };
    path.exists().then_some(path)
}

/// Renders one resolved mention: a file's content in a fenced code block,
/// or a directory's shallow listing.
fn render_attachment(path: &Path, working_dir: &Path) -> Option<String> {
    let label = path.strip_prefix(working_dir).unwrap_or(path).display().to_string();
    if path.is_dir() {
        Some(format!(
            "### {label}/ (directory listing)\n{}",
            list_dir_shallow(path, working_dir)
        ))
    } else if path.is_file() {
        let content = std::fs::read_to_string(path).ok()?; // binary/non-UTF8 -- skip rather than fail the turn
        let char_count = content.chars().count();
        let body = if char_count > MAX_MENTION_FILE_CHARS {
            let truncated: String = content.chars().take(MAX_MENTION_FILE_CHARS).collect();
            format!("{truncated}\n... (truncated to the first {MAX_MENTION_FILE_CHARS} of {char_count} characters)")
        } else {
            content
        };
        Some(format!("### {label}\n```\n{body}\n```"))
    } else {
        None
    }
}

/// One level of `dir`'s entries, dirs first then alphabetical.
fn list_dir_shallow(dir: &Path, working_dir: &Path) -> String {
    let mut entries: Vec<(bool, String)> = ignore::WalkBuilder::new(dir)
        .max_depth(Some(1))
        .hidden(true)
        .require_git(false)
        .build()
        .filter_map(|e| e.ok())
        .filter(|e| e.depth() != 0)
        .map(|e| {
            let is_dir = e.file_type().is_some_and(|t| t.is_dir());
            let relative = e.path().strip_prefix(working_dir).unwrap_or(e.path());
            (
                is_dir,
                format!("{}{}", relative.display(), if is_dir { "/" } else { "" }),
            )
        })
        .collect();
    entries.sort_by(|a, b| match (a.0, b.0) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.1.cmp(&b.1),
    });
    entries.truncate(MAX_DIR_ENTRIES);

    if entries.is_empty() {
        "(empty)".to_string()
    } else {
        entries.into_iter().map(|(_, s)| s).collect::<Vec<_>>().join("\n")
    }
}

/// If the cursor sits in an `@`-prefixed word, its start offset and text after the `@`.
pub fn at_mention_token(line: &str, pos: usize) -> Option<(usize, &str)> {
    if pos > line.len() || !line.is_char_boundary(pos) {
        return None;
    }
    let before = &line[..pos];
    let start = before.rfind(char::is_whitespace).map(|i| i + 1).unwrap_or(0);
    let token = &before[start..];
    let rest = token.strip_prefix('@')?;
    Some((start, rest))
}

/// Filesystem entries completing `prefix` (typed after `@`), for Tab-completion.
pub fn complete_at_mention(prefix: &str, working_dir: &Path) -> Vec<Pair> {
    let (dir_part, name_prefix) = match prefix.rfind('/') {
        Some(i) => (&prefix[..=i], &prefix[i + 1..]),
        None => ("", prefix),
    };
    let list_dir = if dir_part.is_empty() {
        working_dir.to_path_buf()
    } else {
        working_dir.join(dir_part)
    };
    let show_hidden = name_prefix.starts_with('.');

    let mut entries: Vec<(bool, String)> = ignore::WalkBuilder::new(&list_dir)
        .max_depth(Some(1))
        .hidden(!show_hidden)
        .require_git(false)
        .build()
        .filter_map(|e| e.ok())
        .filter(|e| e.depth() != 0)
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            if !name.starts_with(name_prefix) {
                return None;
            }
            let is_dir = e.file_type().is_some_and(|t| t.is_dir());
            Some((is_dir, name))
        })
        .collect();
    entries.sort_by(|a, b| match (a.0, b.0) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.1.cmp(&b.1),
    });
    entries.truncate(100);

    entries
        .into_iter()
        .map(|(is_dir, name)| {
            let suffix = if is_dir { "/" } else { "" };
            Pair {
                display: format!("{name}{suffix}"),
                replacement: format!("@{dir_part}{name}{suffix}"),
            }
        })
        .collect()
}

/// Live-hint half of `complete_at_mention`: single match's suffix, or a listing of all matches.
pub fn at_mention_hint(prefix: &str, working_dir: &Path) -> Option<(String, usize)> {
    let candidates = complete_at_mention(prefix, working_dir);
    match candidates.as_slice() {
        [] => None,
        [only] => {
            let suffix = only.replacement.strip_prefix('@')?.strip_prefix(prefix)?;
            (!suffix.is_empty()).then(|| (suffix.to_string(), suffix.len()))
        }
        many => {
            let list = many.iter().map(|p| p.display.clone()).collect::<Vec<_>>().join("  ");
            Some((format!("  {list}"), 0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("minder-mentions-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn no_mentions_leaves_text_untouched() {
        let dir = scratch_dir();
        assert_eq!(expand_mentions("just a plain task", &dir), "just a plain task");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unresolvable_mention_is_left_alone() {
        let dir = scratch_dir();
        let text = "thanks @someone for the report";
        assert_eq!(expand_mentions(text, &dir), text);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn file_mention_attaches_its_content() {
        let dir = scratch_dir();
        std::fs::write(dir.join("notes.md"), "hello world").unwrap();

        let expanded = expand_mentions("summarize @notes.md please", &dir);
        assert!(expanded.starts_with("summarize @notes.md please"));
        assert!(expanded.contains("### notes.md"));
        assert!(expanded.contains("hello world"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn trailing_punctuation_is_stripped_before_resolving() {
        let dir = scratch_dir();
        std::fs::write(dir.join("README.md"), "content").unwrap();

        let expanded = expand_mentions("see @README.md.", &dir);
        assert!(expanded.contains("### README.md"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn directory_mention_lists_its_entries() {
        let dir = scratch_dir();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/main.rs"), "fn main() {}").unwrap();

        let expanded = expand_mentions("look at @src", &dir);
        assert!(expanded.contains("### src/ (directory listing)"));
        assert!(expanded.contains("main.rs"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn duplicate_mentions_are_only_attached_once() {
        let dir = scratch_dir();
        std::fs::write(dir.join("a.txt"), "content").unwrap();

        let expanded = expand_mentions("@a.txt and again @a.txt", &dir);
        assert_eq!(expanded.matches("### a.txt").count(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oversized_file_mention_is_truncated_with_a_note() {
        let dir = scratch_dir();
        std::fs::write(dir.join("big.txt"), "x".repeat(MAX_MENTION_FILE_CHARS + 500)).unwrap();

        let expanded = expand_mentions("@big.txt", &dir);
        assert!(expanded.contains("truncated to the first"));
        assert!(!expanded.contains(&"x".repeat(MAX_MENTION_FILE_CHARS + 1)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn at_mention_token_finds_the_word_under_the_cursor() {
        let line = "explain @src/mai";
        let (start, prefix) = at_mention_token(line, line.len()).unwrap();
        assert_eq!(start, 8);
        assert_eq!(prefix, "src/mai");
    }

    #[test]
    fn at_mention_token_absent_for_plain_word() {
        assert_eq!(at_mention_token("explain this", 12), None);
    }

    #[test]
    fn complete_at_mention_matches_top_level_entries_by_prefix() {
        let dir = scratch_dir();
        std::fs::write(dir.join("main.rs"), "").unwrap();
        std::fs::write(dir.join("mod.rs"), "").unwrap();
        std::fs::create_dir_all(dir.join("markdown")).unwrap();

        let mut candidates = complete_at_mention("ma", &dir);
        candidates.sort_by(|a, b| a.display.cmp(&b.display));
        let displays: Vec<&str> = candidates.iter().map(|p| p.display.as_str()).collect();
        assert_eq!(displays, vec!["main.rs", "markdown/"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn complete_at_mention_descends_into_a_subdirectory() {
        let dir = scratch_dir();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/main.rs"), "").unwrap();

        let candidates = complete_at_mention("src/ma", &dir);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].replacement, "@src/main.rs");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
