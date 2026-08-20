use std::io::{BufWriter, Stdout, Write};
use std::sync::{Arc, Mutex};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Wrap};

use super::input_box::InputMode;

/// Where a `TerminalReporter`'s formatted output actually goes -- shared
/// between the plain-print fallback REPL and the fullscreen ratatui one.
pub(crate) trait OutputSink: Send + Sync {
    /// Assistant text / conversational content.
    fn print_stdout(&self, text: &str);
    /// Tool trace / diagnostic content.
    fn print_stderr(&self, text: &str);
    /// Redraws the transient spinner/status line in place.
    fn redraw_status(&self, text: &str);
}

/// Raw `print!`/`eprint!` to the real terminal -- used by non-interactive
/// paths (`--output json`, `loop`, one-shot) and the REPL's non-TTY fallback.
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

pub(crate) type AppTerminal = Terminal<CrosstermBackend<BufWriter<Stdout>>>;

/// The pinned box's latest known contents, cheap to clone/lock -- lets
/// `FullscreenSink` redraw the box on its own without a live `InputBoxState`.
#[derive(Clone, Default)]
pub(crate) struct PinnedInputSnapshot {
    pub(crate) buffer: String,
    pub(crate) cursor: usize,
    pub(crate) mode: InputMode,
}

/// The full conversation transcript, redrawn as part of every frame --
/// replaces real terminal scrollback, which an alt-screen app doesn't have.
/// Deliberately unbounded: a chat session's text is small enough (a few MB
/// even for a long one) that cloning it into a `Paragraph` per redraw stays
/// cheap; revisit only if that proves false.
pub(crate) struct Transcript {
    lines: Vec<Line<'static>>,
    /// `true` (the default) auto-scrolls to the bottom as new content
    /// arrives. A manual scroll that leaves the bottom edge clears this;
    /// scrolling back down to the bottom edge re-arms it.
    follow: bool,
    /// Wrapped-line offset from the top; only meaningful while `!follow`.
    scroll: u16,
}

impl Default for Transcript {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            follow: true,
            scroll: 0,
        }
    }
}

impl Transcript {
    /// Appends newly printed lines (already parsed via `ansi_to_line`).
    pub(crate) fn push(&mut self, mut new_lines: Vec<Line<'static>>) {
        self.lines.append(&mut new_lines);
    }

    /// Scrolls by `delta` wrapped lines (negative = back through history);
    /// leaving/returning to the bottom edge toggles `follow`.
    pub(crate) fn scroll_by(&mut self, delta: i32, width: u16, visible_height: u16) {
        if width == 0 {
            return;
        }
        let total = self.wrapped_line_count(width);
        let max_offset = total.saturating_sub(visible_height);
        let current = if self.follow { max_offset } else { self.scroll };
        let next = (i32::from(current) + delta).clamp(0, i32::from(max_offset)) as u16;
        self.scroll = next;
        self.follow = next >= max_offset;
    }

    fn wrapped_line_count(&self, width: u16) -> u16 {
        if width == 0 || self.lines.is_empty() {
            return 0;
        }
        Paragraph::new(Text::from(self.lines.clone()))
            .wrap(Wrap { trim: false })
            .line_count(width) as u16
    }

    /// Renders into `area`, sticking to the bottom while `follow` is set
    /// and clamping a stale manual `scroll` (e.g. after a resize).
    pub(crate) fn render(&mut self, frame: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        let paragraph = Paragraph::new(Text::from(self.lines.clone())).wrap(Wrap { trim: false });
        let total = paragraph.line_count(area.width) as u16;
        let max_offset = total.saturating_sub(area.height);
        self.scroll = if self.follow {
            max_offset
        } else {
            self.scroll.min(max_offset)
        };
        frame.render_widget(paragraph.scroll((self.scroll, 0)), area);
    }
}

/// The `Arc`s every pinned-box-aware function needs, bundled to avoid clippy's `too_many_arguments`.
#[derive(Clone)]
pub(crate) struct PinnedHandles {
    pub(crate) terminal: Arc<Mutex<AppTerminal>>,
    pub(crate) status: Arc<Mutex<String>>,
    /// See `PinnedInputSnapshot`.
    pub(crate) input: Arc<Mutex<PinnedInputSnapshot>>,
    /// See `Transcript`.
    pub(crate) transcript: Arc<Mutex<Transcript>>,
}

/// Appends formatted lines to the in-memory `Transcript` and redraws the
/// whole frame, instead of printing directly -- see `tui::run_tui_repl`.
pub(crate) struct FullscreenSink {
    handles: PinnedHandles,
    color: bool,
}

impl FullscreenSink {
    pub(crate) fn new(handles: PinnedHandles, color: bool) -> Self {
        Self { handles, color }
    }
}

impl OutputSink for FullscreenSink {
    fn print_stdout(&self, text: &str) {
        append_text(
            &self.handles.terminal,
            &self.handles.transcript,
            &self.handles.input,
            &self.handles.status,
            self.color,
            text,
        );
    }

    fn print_stderr(&self, text: &str) {
        append_text(
            &self.handles.terminal,
            &self.handles.transcript,
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

/// Splits `text` into ANSI-styled `Line`s, appends them to the shared
/// transcript, and redraws the full frame. No-op on empty text.
///
/// Locks `terminal` before `transcript`, same order as `tui::redraw` -- keep
/// it consistent across both call sites or it deadlocks.
fn append_text(
    terminal: &Mutex<AppTerminal>,
    transcript: &Mutex<Transcript>,
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

    let snapshot = input.lock().unwrap().clone();
    let status_text = status.lock().unwrap().clone();

    let mut term = terminal.lock().unwrap();
    let mut t = transcript.lock().unwrap();
    t.push(rendered);
    let _ = term.draw(|frame| {
        let area = frame.area();
        let bottom = super::input_box::bottom_area(area, &snapshot.buffer);
        t.render(frame, super::input_box::transcript_area(area, &snapshot.buffer));
        super::input_box::render_pinned(
            frame,
            bottom,
            &snapshot.buffer,
            snapshot.cursor,
            snapshot.mode,
            &status_text,
            color,
        );
        if bottom.height > 0 {
            frame.set_cursor_position(super::input_box::cursor_screen_position_for(
                bottom,
                &snapshot.buffer,
                snapshot.cursor,
            ));
        }
    });
}

/// Parses embedded ANSI SGR sequences into a styled `Line` (ratatui renders
/// `Style`d spans, not raw escape bytes).
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

/// Recognizes any `\x1b[<params>m` SGR sequence and returns it alongside the
/// remainder. `params` is the raw `;`-separated body, not validated here --
/// `syntect`'s truecolor codes (`38;2;r;g;b`) need the full body, and used
/// to leak into visible text when this only matched single-number codes.
fn split_sgr_code(s: &str) -> Option<(&str, &str)> {
    let body = s.strip_prefix("\x1b[")?;
    let end = body.find('m')?;
    Some((&body[..end], &body[end + 1..]))
}

/// Applies one SGR sequence's params to `style`; unrecognized ones are no-ops.
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

/// Strips SGR codes to plain text -- the input box re-styles the status line itself.
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

    /// Regression: syntect's truecolor codes used to leak into visible text
    /// (old `split_sgr_code` only matched single-number SGR codes).
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

    fn line(text: &str) -> Line<'static> {
        Line::from(text.to_string())
    }

    #[test]
    fn transcript_follows_the_bottom_by_default_as_content_grows() {
        let mut t = Transcript::default();
        for i in 0..10 {
            t.push(vec![line(&format!("line {i}"))]);
        }
        // 10 lines, a 4-row viewport: following should sit at offset 6 (so
        // rows 6..10 -- the last 4 -- are visible).
        let total = t.wrapped_line_count(80);
        assert_eq!(total, 10);
        t.scroll_by(0, 80, 4); // no-op nudge just to exercise the follow path
        assert!(t.follow);
    }

    #[test]
    fn scrolling_up_disables_follow_and_scrolling_back_down_re_enables_it() {
        let mut t = Transcript::default();
        for i in 0..20 {
            t.push(vec![line(&format!("line {i}"))]);
        }
        t.scroll_by(-5, 80, 4);
        assert!(!t.follow);

        // Scroll far enough down to land back on the bottom edge.
        t.scroll_by(100, 80, 4);
        assert!(t.follow);
    }

    #[test]
    fn scroll_by_clamps_at_the_top() {
        let mut t = Transcript::default();
        for i in 0..5 {
            t.push(vec![line(&format!("line {i}"))]);
        }
        t.scroll_by(-1000, 80, 2);
        assert!(!t.follow);
        assert_eq!(t.scroll, 0);
    }
}
