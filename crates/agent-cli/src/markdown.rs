use std::borrow::Cow;
use std::str::FromStr;
use std::sync::LazyLock;

use mq_markdown::{ColorTheme, Markdown, Node};
use two_face::re_exports::syntect::easy::HighlightLines;
use two_face::re_exports::syntect::parsing::SyntaxSet;
use two_face::re_exports::syntect::util::as_24_bit_terminal_escaped;
use two_face::theme::EmbeddedThemeName;

const ANSI_RESET: &str = "\x1b[0m";

static SYNTAX_SET: LazyLock<SyntaxSet> = LazyLock::new(two_face::syntax::extra_newlines);
static CODE_THEME: LazyLock<two_face::re_exports::syntect::highlighting::Theme> =
    LazyLock::new(|| two_face::theme::extra().get(EmbeddedThemeName::MonokaiExtended).clone());

/// Renders assistant markdown for the terminal.
///
/// Structural elements (headings, lists, emphasis, links, blockquotes, ...)
/// are colored via `mq-markdown`'s theme. Fenced code blocks get real
/// per-language syntax highlighting through `syntect` (via `two-face`'s
/// bundled syntax/theme set, which covers languages syntect's own defaults
/// miss, like TypeScript, TOML and Dockerfiles) rather than `mq-markdown`'s
/// flat single-color treatment. Falls back to the raw text untouched when
/// color is off or the input doesn't parse as markdown.
pub fn render(text: &str, color: bool) -> String {
    if !color {
        return text.to_string();
    }

    let Ok(md) = Markdown::from_str(text) else {
        return text.to_string();
    };

    let prepared = highlight_code_blocks(text, &md.nodes);

    let Ok(md) = Markdown::from_str(&prepared) else {
        return text.to_string();
    };

    let mut theme = ColorTheme::COLORED;
    // Code blocks are already colored (or intentionally left plain when the
    // language is unknown) by `highlight_code_blocks`, so mq-markdown's own
    // flat code color would just fight with it.
    theme.code = (Cow::Borrowed(""), Cow::Borrowed(""));
    md.to_colored_string_with_theme(&theme)
}

/// Replaces the body of every fenced code block with a syntax-highlighted
/// version, using each block's parsed `Position` to locate its exact source
/// lines. Fence lines themselves are left untouched, so the line count (and
/// thus every other node's position) is unchanged and safe to re-parse.
fn highlight_code_blocks(text: &str, nodes: &[Node]) -> String {
    let mut lines: Vec<Cow<str>> = text.lines().map(Cow::Borrowed).collect();

    for node in nodes {
        let Node::Code(code) = node else { continue };
        if !code.fence {
            continue;
        }
        let Some(lang) = code.lang.as_deref() else {
            continue;
        };
        let Some(syntax) = SYNTAX_SET.find_syntax_by_token(lang) else {
            continue;
        };
        let Some(position) = &code.position else {
            continue;
        };

        // `position` spans the opening fence line through the closing fence
        // line (1-indexed); the body is strictly between them.
        let body_start = position.start.line; // 0-indexed first body line
        let body_end = position.end.line.saturating_sub(1); // exclusive
        if body_start >= body_end || body_end > lines.len() {
            continue;
        }

        let mut highlighter = HighlightLines::new(syntax, &CODE_THEME);
        for line in &mut lines[body_start..body_end] {
            let mut with_newline = String::with_capacity(line.len() + 1);
            with_newline.push_str(line);
            with_newline.push('\n');
            let Ok(ranges) = highlighter.highlight_line(&with_newline, &SYNTAX_SET) else {
                continue;
            };
            // The trailing "\n" fed in above is only there so syntect tracks
            // newline-sensitive state correctly; strip it back out since
            // `lines.join("\n")` supplies the real line break.
            let mut highlighted = as_24_bit_terminal_escaped(&ranges, false);
            if highlighted.ends_with('\n') {
                highlighted.pop();
            }
            highlighted.push_str(ANSI_RESET);
            *line = Cow::Owned(highlighted);
        }
    }

    let mut out = lines.join("\n");
    if text.ends_with('\n') {
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passes_through_uncolored() {
        assert_eq!(render("hello world", false), "hello world");
    }

    #[test]
    fn heading_gets_colored_when_color_enabled() {
        let out = render("# Title", true);
        assert!(out.contains("\x1b["));
        assert!(out.contains("Title"));
    }

    #[test]
    fn invalid_utf8_boundary_like_input_falls_back_to_plain() {
        // Not actual invalid UTF-8 (Rust strings guarantee valid UTF-8) --
        // this just checks a pathological/empty input doesn't panic.
        assert_eq!(render("", true), "");
    }

    #[test]
    fn recognized_language_code_block_is_syntax_highlighted() {
        let text = "```rust\nfn main() {}\n```\n";
        let out = render(text, true);
        // `fn` and `main` land in separate, differently colored token spans,
        // so the literal substring "fn main" doesn't survive -- check the
        // tokens and coloring independently instead.
        assert!(out.contains("\x1b[38;2;"), "expected 24-bit color codes: {out}");
        assert!(out.contains("fn"));
        assert!(out.contains("main"));
    }

    #[test]
    fn unknown_language_code_block_is_left_plain() {
        let text = "```not-a-real-language\nsome text\n```\n";
        let out = render(text, true);
        assert_eq!(out.trim_end(), "```not-a-real-language\nsome text\n```".trim_end());
    }

    #[test]
    fn fence_lines_are_never_recolored() {
        let text = "```python\nprint('hi')\n```\n";
        let out = render(text, true);
        assert!(out.starts_with("```python\n"));
    }

    #[test]
    fn multiline_code_block_keeps_one_output_line_per_source_line() {
        let text = "```rust\nfn a() {}\nfn b() {}\nfn c() {}\n```\n";
        let out = render(text, true);
        let body_lines = out
            .lines()
            .skip(1) // opening fence
            .take_while(|line| !line.starts_with("```"))
            .count();
        assert_eq!(body_lines, 3, "got:\n{out}");
    }
}
