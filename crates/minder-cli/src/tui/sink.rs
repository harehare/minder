use std::io::{Stdout, Write};
use std::sync::Mutex;

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Widget, Wrap};

/// Where a `TerminalReporter`'s formatted output actually goes -- lets it
/// share all its formatting/color logic between the plain-print fallback
/// REPL and the ratatui inline-viewport REPL, which need very different
/// final delivery mechanisms for the same lines.
pub(crate) trait OutputSink: Send + Sync {
    /// Assistant text / conversational content.
    fn print_stdout(&self, text: &str);
    /// Tool trace / diagnostic content.
    fn print_stderr(&self, text: &str);
    /// Redraws the transient spinner/status line in place.
    fn redraw_status(&self, text: &str);
}

/// Today's exact behavior: raw `print!`/`eprint!` straight to the real
/// terminal. Used by every non-interactive path (`--output json`, `loop`
/// mode, one-shot runs) and by the REPL's non-TTY fallback, so none of them
/// change behavior because this type exists.
pub(crate) struct DirectPrintSink;

impl OutputSink for DirectPrintSink {
    fn print_stdout(&self, text: &str) {
        print!("{text}");
        let _ = std::io::stdout().flush();
    }

    fn print_stderr(&self, text: &str) {
        eprint!("{text}");
        let _ = std::io::stderr().flush();
    }

    fn redraw_status(&self, text: &str) {
        eprint!("\r\x1b[2K{text}");
        let _ = std::io::stderr().flush();
    }
}

pub(crate) type InlineTerminal = Terminal<CrosstermBackend<Stdout>>;

/// Pushes formatted lines permanently into the terminal's real scrollback
/// (above the pinned input box) via `Terminal::insert_before`, instead of
/// printing directly -- see `tui::run_tui_repl`. Status/spinner text is
/// captured into `status` instead, for the input box's own render pass to
/// pick up on its next redraw rather than clobbering a terminal row itself.
pub(crate) struct InlineViewportSink {
    terminal: std::sync::Arc<Mutex<InlineTerminal>>,
    status: std::sync::Arc<Mutex<String>>,
}

impl InlineViewportSink {
    pub(crate) fn new(terminal: std::sync::Arc<Mutex<InlineTerminal>>, status: std::sync::Arc<Mutex<String>>) -> Self {
        Self { terminal, status }
    }
}

impl OutputSink for InlineViewportSink {
    fn print_stdout(&self, text: &str) {
        insert_text(&self.terminal, text);
    }

    fn print_stderr(&self, text: &str) {
        insert_text(&self.terminal, text);
    }

    fn redraw_status(&self, text: &str) {
        *self.status.lock().unwrap() = strip_ansi(text);
    }
}

/// Splits `text` on newlines, converts each line's embedded ANSI codes
/// (this codebase only ever emits the small fixed set in `reporter.rs`,
/// never arbitrary terminal input) into a styled `Line`, and inserts the
/// whole block above the pinned viewport. A no-op on empty text so a
/// guard-only call (no actual line) doesn't push a blank row.
fn insert_text(terminal: &Mutex<InlineTerminal>, text: &str) {
    if text.is_empty() {
        return;
    }
    // `text` always ends with the newline the caller already appended
    // (mirrors `println!`/`eprintln!`) -- drop the trailing empty split.
    let mut lines: Vec<&str> = text.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    let rendered: Vec<Line> = lines.iter().map(|l| ansi_to_line(l)).collect();
    if rendered.is_empty() {
        return;
    }

    let mut term = terminal.lock().unwrap();
    let width = term.size().map(|s| s.width).unwrap_or(80).max(1);
    let paragraph = Paragraph::new(Text::from(rendered)).wrap(Wrap { trim: false });
    let height = paragraph.line_count(width) as u16;
    let _ = term.insert_before(height, |buf| paragraph.render(buf.area, buf));
}

/// Parses the handful of SGR codes `reporter.rs`'s `paint`/format helpers
/// ever emit (`RESET`/`DIM`/`BOLD`/`GREEN`/`RED`/`YELLOW`/`CYAN`) into a
/// styled `Line`, since a ratatui `Buffer` renders `Style`d spans, not raw
/// escape bytes. Unrecognized bytes (there shouldn't be any, given the
/// closed set of codes this codebase produces) pass through as plain text.
fn ansi_to_line(line: &str) -> Line<'static> {
    let mut spans = Vec::new();
    let mut style = Style::default();
    let mut rest = line;

    while let Some(esc_pos) = rest.find('\x1b') {
        if esc_pos > 0 {
            spans.push(Span::styled(rest[..esc_pos].to_string(), style));
        }
        rest = &rest[esc_pos..];
        let Some((code, after)) = split_sgr_code(rest) else {
            // Not a recognized SGR sequence -- emit the escape byte itself
            // as plain text rather than looping forever on it.
            spans.push(Span::styled(rest[..1].to_string(), style));
            rest = &rest[1..];
            continue;
        };
        style = apply_sgr(style, code);
        rest = after;
    }
    if !rest.is_empty() {
        spans.push(Span::styled(rest.to_string(), style));
    }
    Line::from(spans)
}

/// Recognizes exactly the escape sequences in `reporter.rs`'s color consts
/// (`\x1b[0m`, `\x1b[1m`, `\x1b[2m`, `\x1b[3Xm`) at the start of `s`,
/// returning the numeric code and the remainder past it.
fn split_sgr_code(s: &str) -> Option<(u8, &str)> {
    let body = s.strip_prefix("\x1b[")?;
    let end = body.find('m')?;
    let code: u8 = body[..end].parse().ok()?;
    Some((code, &body[end + 1..]))
}

fn apply_sgr(style: Style, code: u8) -> Style {
    match code {
        0 => Style::default(),
        1 => style.add_modifier(Modifier::BOLD),
        2 => style.add_modifier(Modifier::DIM),
        31 => style.fg(ratatui::style::Color::Red),
        32 => style.fg(ratatui::style::Color::Green),
        33 => style.fg(ratatui::style::Color::Yellow),
        36 => style.fg(ratatui::style::Color::Cyan),
        _ => style,
    }
}

/// Strips this codebase's SGR codes down to plain text -- used for the
/// status line, which the input box re-styles itself rather than rendering
/// pre-colored text.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(esc_pos) = rest.find('\x1b') {
        out.push_str(&rest[..esc_pos]);
        rest = &rest[esc_pos..];
        match split_sgr_code(rest) {
            Some((_, after)) => rest = after,
            None => {
                out.push_str(&rest[..1]);
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}
