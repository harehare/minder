use similar::{ChangeTag, TextDiff};

/// A unified-style diff plus the raw added/removed line counts, computed
/// once here so both the tool's own JSON metadata and any terminal renderer
/// (see `agent-cli`'s reporter) work from the same numbers.
pub struct FileDiff {
    pub unified: String,
    pub additions: usize,
    pub deletions: usize,
}

/// Builds a unified diff of `old` -> `new` content, labeled with `path` in
/// the `---`/`+++` header lines the way `git diff`/`diff -u` would.
pub fn diff_files(path: &str, old: &str, new: &str) -> FileDiff {
    let diff = TextDiff::from_lines(old, new);
    let unified = diff
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();

    let (mut additions, mut deletions) = (0, 0);
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => additions += 1,
            ChangeTag::Delete => deletions += 1,
            ChangeTag::Equal => {}
        }
    }

    FileDiff {
        unified,
        additions,
        deletions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_additions_and_deletions() {
        let diff = diff_files(
            "a.txt",
            "one\ntwo\nthree\n",
            "one\ntwo changed\nthree\nfour\n",
        );
        assert_eq!(diff.additions, 2);
        assert_eq!(diff.deletions, 1);
        assert!(diff.unified.contains("a/a.txt"));
        assert!(diff.unified.contains("b/a.txt"));
        assert!(diff.unified.contains("-two"));
        assert!(diff.unified.contains("+two changed"));
    }

    #[test]
    fn new_file_is_all_additions() {
        let diff = diff_files("new.txt", "", "hello\nworld\n");
        assert_eq!(diff.additions, 2);
        assert_eq!(diff.deletions, 0);
    }
}
