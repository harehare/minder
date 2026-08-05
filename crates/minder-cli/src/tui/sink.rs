use std::io::{BufWriter, Stdout, Write};
use std::sync::{Arc, Mutex};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Widget, Wrap};

use super::input_box::InputMode;

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

pub(crate) type InlineTerminal = Terminal<CrosstermBackend<BufWriter<Stdout>>>;

/// The pinned box's latest known contents, cheap to clone/lock -- kept in
/// sync by `tui::run_tui_repl` on every keystroke/mode change so
/// `InlineViewportSink` can redraw the box on its own right after clearing
/// it (see `insert_text`), without needing the REPL loop's actual
/// `InputBoxState` (which it never has direct access to).
#[derive(Clone, Default)]
pub(crate) struct PinnedInputSnapshot {
    pub(crate) buffer: String,
    pub(crate) cursor: usize,
    pub(crate) mode: InputMode,
}

/// The three `Arc`s every pinned-box-aware function needs, bundled together
/// since none of them are ever useful without the other two -- threading
/// them as separate parameters was what tripped clippy's
/// `too_many_arguments` on `tui::run_turn_pinned`.
#[derive(Clone)]
pub(crate) struct PinnedHandles {
    pub(crate) terminal: Arc<Mutex<InlineTerminal>>,
    pub(crate) status: Arc<Mutex<String>>,
    /// See `PinnedInputSnapshot`.
    pub(crate) input: Arc<Mutex<PinnedInputSnapshot>>,
}

/// Pushes formatted lines permanently into the terminal's real scrollback
/// (above the pinned input box) via `Terminal::insert_before`, instead of
/// printing directly -- see `tui::run_tui_repl`. Status/spinner text is
/// captured into `status` instead, for the input box's own render pass to
/// pick up on its next redraw rather than clobbering a terminal row itself.
pub(crate) struct InlineViewportSink {
    handles: PinnedHandles,
    color: bool,
}

impl InlineViewportSink {
    pub(crate) fn new(handles: PinnedHandles, color: bool) -> Self {
        Self { handles, color }
    }
}

impl OutputSink for InlineViewportSink {
    fn print_stdout(&self, text: &str) {
        insert_text(
            &self.handles.terminal,
            &self.handles.input,
            &self.handles.status,
            self.color,
            text,
        );
    }

    fn print_stderr(&self, text: &str) {
        insert_text(
            &self.handles.terminal,
            &self.handles.input,
            &self.handles.status,
            self.color,
            text,
        );
    }

    fn redraw_status(&self, text: &str) {
        *self.handles.status.lock().unwrap() = strip_ansi(text);
    }
}

/// Splits `text` on newlines, converts each line's embedded ANSI codes
/// (this codebase only ever emits the small fixed set in `reporter.rs`,
/// never arbitrary terminal input) into a styled `Line`, and inserts the
/// whole block above the pinned viewport. A no-op on empty text so a
/// guard-only call (no actual line) doesn't push a blank row.
///
/// Without ratatui's `scrolling-regions` feature (not enabled here --
/// terminal-support is inconsistent enough across emulators/multiplexers
/// that the plain fallback is safer), `Terminal::insert_before` clears the
/// viewport it's inserting above and relies on the *next* `Terminal::draw`
/// to repaint it. Left alone, that next repaint was whatever the 90ms
/// ticker or the next keystroke happened to trigger in `tui::run_tui_repl`
/// -- during a burst of streamed tokens or fast tool output, each
/// `insert_before` clears the box again before that redraw lands, so the
/// pinned box (and the spinner/status line riding along with it) reads as
/// blank or stuck the whole time nothing is typed. Redrawing right here,
/// immediately after every insert, closes that window instead of waiting
/// on the next unrelated tick.
fn insert_text(
    terminal: &Mutex<InlineTerminal>,
    input: &Mutex<PinnedInputSnapshot>,
    status: &Mutex<String>,
    color: bool,
    text: &str,
) {
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

    let snapshot = input.lock().unwrap().clone();
    let status_text = status.lock().unwrap().clone();
    let _ = term.draw(|frame| {
        let area = frame.area();
        super::input_box::render_pinned(
            frame,
            area,
            &snapshot.buffer,
            snapshot.cursor,
            snapshot.mode,
            &status_text,
            color,
        );
        if let Some(input_row) = super::input_box::input_row(area) {
            frame.set_cursor_position((
                super::input_box::cursor_column_for(input_row, &snapshot.buffer, snapshot.cursor),
                input_row.y,
            ));
        }
    });
}

/// Parses embedded ANSI SGR sequences -- both `reporter.rs`'s own fixed
/// palette and `crate::markdown`'s syntect-highlighted code blocks (24-bit
/// and 256-color codes) -- into a styled `Line`, since a ratatui `Buffer`
/// renders `Style`d spans, not raw escape bytes. A malformed sequence with
/// no closing `m` (shouldn't happen given what this codebase emits) skips
/// just the escape byte rather than looping forever on it.
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

/// Recognizes any `\x1b[<params>m` SGR sequence at the start of `s`
/// (`params` is the raw `;`-separated body, not just `reporter.rs`'s closed
/// set of single-number codes) and returns it alongside the remainder past
/// it. Deliberately doesn't validate/parse `params` here -- `crate::markdown`
/// renders syntax-highlighted code blocks through `syntect`, which emits
/// multi-parameter 24-bit-color codes like `38;2;249;38;114` (see
/// `apply_sgr`); this used to reject those and fall through to the
/// "unrecognized" path, which only skips the escape *byte* and leaves the
/// rest of the sequence (`[38;2;249;38;114m`) to leak into the visible
/// output as literal text.
fn split_sgr_code(s: &str) -> Option<(&str, &str)> {
    let body = s.strip_prefix("\x1b[")?;
    let end = body.find('m')?;
    Some((&body[..end], &body[end + 1..]))
}

/// Applies one SGR sequence's `;`-separated parameters to `style`, covering
/// both `reporter.rs`'s own fixed palette and the 24-bit/256-color codes
/// `syntect` emits for highlighted code blocks (see `split_sgr_code`).
/// Parameters this doesn't recognize are no-ops, same as before.
fn apply_sgr(style: Style, params: &str) -> Style {
    use ratatui::style::Color;

    let mut parts = params.split(';').map(|p| p.parse::<u32>().unwrap_or(u32::MAX));
    match parts.next() {
        None | Some(0) => Style::default(),
        Some(1) => style.add_modifier(Modifier::BOLD),
        Some(2) => style.add_modifier(Modifier::DIM),
        Some(3) => style.add_modifier(Modifier::ITALIC),
        Some(4) => style.add_modifier(Modifier::UNDERLINED),
        Some(31) => style.fg(Color::Red),
        Some(32) => style.fg(Color::Green),
        Some(33) => style.fg(Color::Yellow),
        Some(34) => style.fg(Color::Blue),
        Some(35) => style.fg(Color::Magenta),
        Some(36) => style.fg(Color::Cyan),
        Some(39) => style.fg(Color::Reset),
        Some(38) => match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some(2), Some(r), Some(g), Some(b)) if [r, g, b].iter().all(|c| *c <= 255) => {
                style.fg(Color::Rgb(r as u8, g as u8, b as u8))
            }
            (Some(5), Some(n), _, _) if n <= 255 => style.fg(Color::Indexed(n as u8)),
            _ => style,
        },
        Some(48) => match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some(2), Some(r), Some(g), Some(b)) if [r, g, b].iter().all(|c| *c <= 255) => {
                style.bg(Color::Rgb(r as u8, g as u8, b as u8))
            }
            (Some(5), Some(n), _, _) if n <= 255 => style.bg(Color::Indexed(n as u8)),
            _ => style,
        },
        Some(49) => style.bg(Color::Reset),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_text(line: &Line<'static>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    /// Regression test for the bug this fix addresses: `crate::markdown`
    /// renders syntax-highlighted code blocks with syntect's 24-bit
    /// truecolor SGR codes (`\x1b[38;2;R;G;Bm`), which the old
    /// single-number `split_sgr_code` couldn't parse -- it fell through to
    /// the "unrecognized sequence" path, which only skipped the escape byte
    /// and left the rest of the code (`[38;2;249;38;114m`) as literal
    /// visible text, garbling both the content and the width `insert_text`
    /// wraps/scrolls by.
    #[test]
    fn truecolor_sgr_codes_do_not_leak_into_the_rendered_text() {
        let line = ansi_to_line("\x1b[38;2;249;38;114mfn\x1b[0m main() {}");
        assert_eq!(plain_text(&line), "fn main() {}");
    }

    #[test]
    fn truecolor_foreground_sets_an_rgb_style() {
        let line = ansi_to_line("\x1b[38;2;249;38;114mfn\x1b[0m");
        assert_eq!(line.spans[0].style.fg, Some(ratatui::style::Color::Rgb(249, 38, 114)));
    }

    #[test]
    fn indexed_256_color_codes_do_not_leak_either() {
        let line = ansi_to_line("\x1b[38;5;208morange\x1b[0m");
        assert_eq!(plain_text(&line), "orange");
        assert_eq!(line.spans[0].style.fg, Some(ratatui::style::Color::Indexed(208)));
    }

    #[test]
    fn background_truecolor_codes_are_applied_and_not_leaked() {
        let line = ansi_to_line("\x1b[48;2;1;2;3mtext\x1b[0m");
        assert_eq!(plain_text(&line), "text");
        assert_eq!(line.spans[0].style.bg, Some(ratatui::style::Color::Rgb(1, 2, 3)));
    }

    #[test]
    fn reporters_own_fixed_palette_still_works() {
        let line = ansi_to_line("\x1b[1m\x1b[36mheading\x1b[0m plain");
        assert_eq!(plain_text(&line), "heading plain");
        assert_eq!(line.spans[0].style.fg, Some(ratatui::style::Color::Cyan));
        assert!(line.spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn strip_ansi_removes_truecolor_codes_too() {
        assert_eq!(strip_ansi("\x1b[38;2;249;38;114mfn\x1b[0m"), "fn");
    }
}
