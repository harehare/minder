use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};
use unicode_width::UnicodeWidthChar;

use super::sink::PinnedInputSnapshot;
use crate::mentions;

/// Window for a second idle Ctrl-C to quit -- see `InputOutcome::CtrlCHint`.
const CTRL_C_QUIT_WINDOW: Duration = Duration::from_secs(2);

/// Whether a turn is running -- changes what Enter/Esc do and how the box renders.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum InputMode {
    #[default]
    Idle,
    Running,
}

/// What happened to a single key event -- `run_tui_repl` acts on this.
pub(crate) enum InputOutcome {
    /// Consumed, nothing for the caller to do.
    Handled,
    /// Enter on a non-empty buffer -- a new turn when idle, steering text when running.
    Submit(String),
    /// Esc or Ctrl-C while a turn is running.
    CancelTurn,
    /// Ctrl-D on an empty buffer, or a second idle Ctrl-C within the window.
    Quit,
    /// First idle Ctrl-C on an empty buffer -- shows a "press again" hint.
    CtrlCHint,
}

/// The always-visible input box's state: buffer, cursor, and history --
/// unifies what used to be split between `rustyline` (idle) and
/// `input_watcher.rs` (mid-turn). See `tui::run_tui_repl`.
pub(crate) struct InputBoxState {
    buffer: String,
    cursor: usize,
    history: Vec<String>,
    /// `Some(i)` while browsing history via Up/Down.
    history_index: Option<usize>,
    /// Draft text saved before Up first moved into history.
    draft: String,
    /// When the last idle Ctrl-C landed -- see `CTRL_C_QUIT_WINDOW`.
    last_idle_ctrl_c: Option<Instant>,
}

impl InputBoxState {
    pub(crate) fn new(history: Vec<String>) -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            history,
            history_index: None,
            draft: String::new(),
            last_idle_ctrl_c: None,
        }
    }

    /// Records the buffer in history (skipping immediate repeats) and resets it.
    fn commit(&mut self) -> String {
        let line = std::mem::take(&mut self.buffer);
        self.cursor = 0;
        self.history_index = None;
        self.draft.clear();
        if self.history.last() != Some(&line) {
            self.history.push(line.clone());
        }
        line
    }

    pub(crate) fn handle_key(&mut self, key: KeyEvent, mode: InputMode, working_dir: &Path) -> InputOutcome {
        let is_ctrl_c = key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        let is_ctrl_d = key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL);
        // Only an idle Ctrl-C on an already-empty buffer is a quit candidate
        // -- one with text to clear, or Esc, or anything mid-turn, takes its
        // own branch below and never arms/consumes the quit window.
        let is_idle_empty_ctrl_c = is_ctrl_c && mode == InputMode::Idle && self.buffer.is_empty();
        let ctrl_c_armed =
            is_idle_empty_ctrl_c && self.last_idle_ctrl_c.is_some_and(|t| t.elapsed() < CTRL_C_QUIT_WINDOW);
        if !is_idle_empty_ctrl_c {
            self.last_idle_ctrl_c = None;
        }

        if is_ctrl_d {
            return if self.buffer.is_empty() {
                InputOutcome::Quit
            } else {
                InputOutcome::Handled
            };
        }
        // Alt+Enter inserts a newline instead of submitting (Shift+Enter too, where a terminal reports it).
        if key.code == KeyCode::Enter && key.modifiers.intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) {
            let idx = char_boundary(&self.buffer, self.cursor);
            self.buffer.insert(idx, '\n');
            self.cursor += 1;
            return InputOutcome::Handled;
        }
        if key.code == KeyCode::Esc || is_ctrl_c {
            if mode == InputMode::Running {
                return InputOutcome::CancelTurn;
            }
            if is_idle_empty_ctrl_c {
                if ctrl_c_armed {
                    return InputOutcome::Quit;
                }
                self.last_idle_ctrl_c = Some(Instant::now());
                return InputOutcome::CtrlCHint;
            }
            self.buffer.clear();
            self.cursor = 0;
            self.history_index = None;
            return InputOutcome::Handled;
        }

        match key.code {
            KeyCode::Enter => {
                if self.buffer.is_empty() {
                    InputOutcome::Handled
                } else {
                    InputOutcome::Submit(self.commit())
                }
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let idx = char_boundary(&self.buffer, self.cursor - 1);
                    self.buffer.remove(idx);
                    self.cursor -= 1;
                }
                InputOutcome::Handled
            }
            KeyCode::Delete => {
                if self.cursor < self.buffer.chars().count() {
                    let idx = char_boundary(&self.buffer, self.cursor);
                    self.buffer.remove(idx);
                }
                InputOutcome::Handled
            }
            KeyCode::Left => {
                self.cursor = self.cursor.saturating_sub(1);
                InputOutcome::Handled
            }
            KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.buffer.chars().count());
                InputOutcome::Handled
            }
            KeyCode::Home => {
                let (line, _) = line_col(&self.buffer, self.cursor);
                self.cursor = cursor_from_line_col(&self.buffer, line, 0);
                InputOutcome::Handled
            }
            KeyCode::End => {
                let (line, _) = line_col(&self.buffer, self.cursor);
                self.cursor = cursor_from_line_col(&self.buffer, line, usize::MAX);
                InputOutcome::Handled
            }
            KeyCode::Up => {
                if self.buffer.contains('\n') {
                    let (line, col) = line_col(&self.buffer, self.cursor);
                    if line > 0 {
                        self.cursor = cursor_from_line_col(&self.buffer, line - 1, col);
                    }
                } else {
                    self.navigate_history(-1);
                }
                InputOutcome::Handled
            }
            KeyCode::Down => {
                if self.buffer.contains('\n') {
                    let (line, col) = line_col(&self.buffer, self.cursor);
                    if line + 1 < total_lines(&self.buffer) {
                        self.cursor = cursor_from_line_col(&self.buffer, line + 1, col);
                    }
                } else {
                    self.navigate_history(1);
                }
                InputOutcome::Handled
            }
            KeyCode::Tab => {
                self.complete(working_dir);
                InputOutcome::Handled
            }
            KeyCode::Char(c) => {
                let idx = char_boundary(&self.buffer, self.cursor);
                self.buffer.insert(idx, c);
                self.cursor += 1;
                InputOutcome::Handled
            }
            _ => InputOutcome::Handled,
        }
    }

    fn navigate_history(&mut self, delta: i32) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.history_index {
            None if delta < 0 => {
                self.draft = self.buffer.clone();
                self.history.len() - 1
            }
            None => return,
            Some(i) if delta < 0 => {
                if i == 0 {
                    return;
                }
                i - 1
            }
            Some(i) => {
                if i + 1 >= self.history.len() {
                    self.buffer = std::mem::take(&mut self.draft);
                    self.cursor = self.buffer.chars().count();
                    self.history_index = None;
                    return;
                }
                i + 1
            }
        };
        self.buffer = self.history[next].clone();
        self.cursor = self.buffer.chars().count();
        self.history_index = Some(next);
    }

    /// Completes a `/`-command (single match) or `@mention` (first candidate) at the cursor.
    fn complete(&mut self, working_dir: &Path) {
        if let Some(matches) = crate::matching_slash_commands(&self.buffer, self.cursor)
            && let [only] = matches.as_slice()
        {
            self.buffer = format!("/{only} ");
            self.cursor = self.buffer.chars().count();
            return;
        }
        if let Some((start, prefix)) = mentions::at_mention_token(&self.buffer, self.cursor) {
            let candidates = mentions::complete_at_mention(prefix, working_dir);
            if let Some(first) = candidates.into_iter().next() {
                self.buffer.replace_range(start.., &first.replacement);
                self.cursor = self.buffer.chars().count();
            }
        }
    }

    /// Renders the status line and pending (unanswered mid-turn) questions
    /// above the bordered input panel into `area`.
    pub(crate) fn render(
        &self,
        frame: &mut ratatui::Frame,
        area: Rect,
        mode: InputMode,
        status: &str,
        pending: &[String],
        color: bool,
    ) {
        render_pinned(frame, area, &self.buffer, self.cursor, mode, status, pending, color);
    }

    /// Cursor's on-screen `(x, y)`, for `Frame::set_cursor_position`.
    pub(crate) fn cursor_screen_position(&self, area: Rect, pending: &[String]) -> (u16, u16) {
        cursor_screen_position_for(area, &self.buffer, self.cursor, pending)
    }

    /// The current buffer text -- used to size the pinned box before a live `render` call.
    pub(crate) fn buffer(&self) -> &str {
        &self.buffer
    }

    /// A cheap, `Send`-able copy for `sink::FullscreenSink` to redraw the box on its own.
    pub(crate) fn snapshot(&self, mode: InputMode) -> PinnedInputSnapshot {
        PinnedInputSnapshot {
            buffer: self.buffer.clone(),
            cursor: self.cursor,
            mode,
        }
    }
}

/// Panel background, a shade darker than the default terminal background so
/// the box reads as a distinct panel (Codex/Claude Code-style) instead of
/// just a bordered rule -- fixed rather than derived from the terminal's own
/// palette so it looks the same everywhere. Only used when `color` is on.
const PANEL_BG: Color = Color::Rgb(30, 32, 38);
/// Foreground for text drawn on `PANEL_BG`, explicit rather than the
/// terminal's own default -- a dark panel over a light-theme default
/// foreground would otherwise be unreadable.
const PANEL_FG: Color = Color::Rgb(220, 220, 225);

/// Most input lines shown at once before the box scrolls internally instead of growing further.
const MAX_INPUT_LINES: usize = 6;

/// Most pinned pending-question lines shown at once above the input box.
const MAX_PENDING_LINES: usize = 3;

fn pending_line_count(pending: &[String]) -> usize {
    pending.len().min(MAX_PENDING_LINES)
}

/// Number of `\n`-separated lines in `buffer` (always >= 1).
fn total_lines(buffer: &str) -> usize {
    buffer.matches('\n').count() + 1
}

/// How many of `buffer`'s lines are shown at once, capped at `MAX_INPUT_LINES`.
fn visible_line_count(buffer: &str) -> usize {
    total_lines(buffer).min(MAX_INPUT_LINES)
}

/// First visible line when showing `visible` lines of `buffer` -- sticks to the bottom by default, follows the cursor up.
fn scroll_offset(buffer: &str, cursor_line: usize, visible: usize) -> usize {
    let total = total_lines(buffer);
    let max_offset = total.saturating_sub(visible);
    cursor_line.min(max_offset)
}

/// `(line, col)` of `cursor` (both 0-based, in chars) within `buffer`'s `\n`-separated lines.
fn line_col(buffer: &str, cursor: usize) -> (usize, usize) {
    let mut line = 0usize;
    let mut col = 0usize;
    for (i, c) in buffer.chars().enumerate() {
        if i == cursor {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Inverse of `line_col`; clamps `target_col` to the line's length (`usize::MAX` means "end of line").
fn cursor_from_line_col(buffer: &str, target_line: usize, target_col: usize) -> usize {
    let mut idx = 0usize;
    let mut line = 0usize;
    let mut col = 0usize;
    for c in buffer.chars() {
        if line == target_line && col == target_col {
            return idx;
        }
        if c == '\n' {
            if line == target_line {
                return idx;
            }
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
        idx += 1;
    }
    idx
}

/// Free-function core of `InputBoxState::render`, taking buffer + cursor
/// instead of `&InputBoxState` -- lets `sink::append_text` redraw the box
/// from its own cheap snapshot without a live `InputBoxState`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_pinned(
    frame: &mut ratatui::Frame,
    area: Rect,
    buffer: &str,
    cursor: usize,
    mode: InputMode,
    status: &str,
    pending: &[String],
    color: bool,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let accent = if mode == InputMode::Running {
        Color::LightYellow
    } else {
        Color::Yellow
    };

    let (status_area, pending_area, box_area, interior) = split_bottom(area, pending);

    if let Some(status_area) = status_area {
        let style = if color {
            Style::default().fg(accent).add_modifier(Modifier::DIM)
        } else {
            Style::default()
        };
        frame.render_widget(Paragraph::new(Line::styled(status.to_string(), style)), status_area);
    }

    if let Some(pending_area) = pending_area {
        render_pending(frame, pending_area, pending, color);
    }

    let Some(box_area) = box_area else { return };

    let panel_style = if color {
        Style::default().bg(PANEL_BG).fg(PANEL_FG)
    } else {
        Style::default()
    };
    let border_style = if color { panel_style.fg(accent) } else { Style::default() };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .style(panel_style)
        .border_style(border_style);
    frame.render_widget(&block, box_area);

    let Some(interior) = interior else { return };

    let glyph = match mode {
        InputMode::Idle => "❯",
        InputMode::Running => "»",
    };
    let glyph_style = if color {
        panel_style.fg(accent).add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };

    let (cursor_line, _) = line_col(buffer, cursor);
    let visible = interior.height as usize;
    let offset = scroll_offset(buffer, cursor_line, visible);
    let raw_lines: Vec<&str> = buffer.split('\n').collect();

    let rendered: Vec<Line> = (0..visible)
        .map(|i| {
            let prefix = if i == 0 {
                Span::styled(format!("{glyph} "), glyph_style)
            } else {
                Span::styled("  ", panel_style)
            };
            let text = raw_lines.get(offset + i).copied().unwrap_or("");
            Line::from(vec![prefix, Span::styled(text.to_string(), panel_style)])
        })
        .collect();
    frame.render_widget(Paragraph::new(rendered).style(panel_style), interior);
}

/// Renders up to `MAX_PENDING_LINES` pending mid-turn questions (dim, prefixed
/// with an hourglass), collapsing any remainder into a "+N more" summary line.
fn render_pending(frame: &mut ratatui::Frame, area: Rect, pending: &[String], color: bool) {
    let style = if color {
        Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)
    } else {
        Style::default()
    };
    let visible = area.height as usize;
    if visible == 0 {
        return;
    }
    let overflow = pending.len() > visible;
    let shown = if overflow { visible.saturating_sub(1) } else { pending.len().min(visible) };
    let mut lines: Vec<Line> = pending
        .iter()
        .take(shown)
        .map(|q| Line::styled(format!("⏳ {}", first_line(q)), style))
        .collect();
    if overflow {
        lines.push(Line::styled(format!("… +{} more waiting", pending.len() - shown), style));
    }
    frame.render_widget(Paragraph::new(lines), area);
}

fn first_line(text: &str) -> &str {
    text.split('\n').next().unwrap_or("")
}

/// Cursor's on-screen `(x, y)` for `buffer`/`cursor` rendered into `area` by `render_pinned`.
pub(crate) fn cursor_screen_position_for(area: Rect, buffer: &str, cursor: usize, pending: &[String]) -> (u16, u16) {
    let Some(interior) = split_bottom(area, pending).3 else {
        return (area.x, area.y);
    };
    let (line, col) = line_col(buffer, cursor);
    let visible = interior.height as usize;
    let offset = scroll_offset(buffer, line, visible);
    let row = interior.y + (line - offset) as u16;

    let raw_lines: Vec<&str> = buffer.split('\n').collect();
    let line_text = raw_lines.get(line).copied().unwrap_or("");
    let col_width: u16 = line_text
        .chars()
        .take(col)
        .map(|c| c.width().unwrap_or(0) as u16)
        .sum();
    let prefix_width = 2u16; // glyph/indent + one space, both single-column
    (interior.x + prefix_width + col_width, row)
}

/// Rows reserved at the bottom of a full frame: status line + pending-question
/// lines (up to `MAX_PENDING_LINES`) + bordered input box (up to `MAX_INPUT_LINES` lines).
pub(crate) fn bottom_area(frame_area: Rect, buffer: &str, pending: &[String]) -> Rect {
    let content_lines = visible_line_count(buffer) as u16;
    let pending_lines = pending_line_count(pending) as u16;
    let height = (1 + pending_lines + 2 + content_lines).min(frame_area.height);
    Rect {
        y: frame_area.y + frame_area.height - height,
        height,
        ..frame_area
    }
}

/// Everything above `bottom_area(frame_area, buffer, pending)` -- the scrollable transcript pane.
pub(crate) fn transcript_area(frame_area: Rect, buffer: &str, pending: &[String]) -> Rect {
    Rect {
        height: frame_area.height - bottom_area(frame_area, buffer, pending).height,
        ..frame_area
    }
}

/// Splits a full bottom `area` into (status line, pending-question panel,
/// bordered box, box interior) -- each `None` once too short to fit.
#[allow(clippy::type_complexity)]
fn split_bottom(area: Rect, pending: &[String]) -> (Option<Rect>, Option<Rect>, Option<Rect>, Option<Rect>) {
    if area.height == 0 {
        return (None, None, None, None);
    }
    let status = Some(Rect { height: 1, ..area });
    if area.height == 1 {
        return (status, None, None, None);
    }
    let rest = Rect {
        y: area.y + 1,
        height: area.height - 1,
        ..area
    };

    let wanted_pending = pending_line_count(pending) as u16;
    let (pending_area, rest) = if wanted_pending == 0 {
        (None, rest)
    } else {
        let take = wanted_pending.min(rest.height.saturating_sub(1));
        if take == 0 {
            (None, rest)
        } else {
            (
                Some(Rect { height: take, ..rest }),
                Rect {
                    y: rest.y + take,
                    height: rest.height - take,
                    ..rest
                },
            )
        }
    };

    if rest.height == 0 {
        return (status, pending_area, None, None);
    }
    let box_area = rest;
    let interior = (box_area.height > 2).then(|| box_area.inner(Margin::new(1, 1)));
    (status, pending_area, Some(box_area), interior)
}

fn char_boundary(s: &str, char_idx: usize) -> usize {
    s.char_indices().nth(char_idx).map(|(i, _)| i).unwrap_or(s.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn dir() -> std::path::PathBuf {
        std::env::current_dir().unwrap()
    }

    #[test]
    fn typing_appends_to_the_buffer() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(key(KeyCode::Char('h')), InputMode::Idle, &dir());
        box_state.handle_key(key(KeyCode::Char('i')), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "hi");
        assert_eq!(box_state.cursor, 2);
    }

    #[test]
    fn enter_on_empty_buffer_is_a_noop() {
        let mut box_state = InputBoxState::new(Vec::new());
        assert!(matches!(
            box_state.handle_key(key(KeyCode::Enter), InputMode::Idle, &dir()),
            InputOutcome::Handled
        ));
    }

    #[test]
    fn enter_on_non_empty_buffer_submits_and_clears() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(key(KeyCode::Char('x')), InputMode::Idle, &dir());
        match box_state.handle_key(key(KeyCode::Enter), InputMode::Idle, &dir()) {
            InputOutcome::Submit(line) => assert_eq!(line, "x"),
            _ => panic!("expected Submit"),
        }
        assert!(box_state.buffer.is_empty());
    }

    #[test]
    fn esc_while_idle_clears_the_buffer_instead_of_cancelling() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(key(KeyCode::Char('x')), InputMode::Idle, &dir());
        assert!(matches!(
            box_state.handle_key(key(KeyCode::Esc), InputMode::Idle, &dir()),
            InputOutcome::Handled
        ));
        assert!(box_state.buffer.is_empty());
    }

    #[test]
    fn esc_while_running_cancels_the_turn() {
        let mut box_state = InputBoxState::new(Vec::new());
        assert!(matches!(
            box_state.handle_key(key(KeyCode::Esc), InputMode::Running, &dir()),
            InputOutcome::CancelTurn
        ));
    }

    fn ctrl_c() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    #[test]
    fn ctrl_c_on_a_non_empty_idle_buffer_clears_it_instead_of_arming_quit() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(key(KeyCode::Char('x')), InputMode::Idle, &dir());
        assert!(matches!(
            box_state.handle_key(ctrl_c(), InputMode::Idle, &dir()),
            InputOutcome::Handled
        ));
        assert!(box_state.buffer.is_empty());
        // Buffer's empty now, but this Ctrl-C was the "clear" one, not a
        // quit candidate -- a following Ctrl-C should only arm, not quit.
        assert!(matches!(
            box_state.handle_key(ctrl_c(), InputMode::Idle, &dir()),
            InputOutcome::CtrlCHint
        ));
    }

    #[test]
    fn first_idle_ctrl_c_on_an_empty_buffer_only_hints() {
        let mut box_state = InputBoxState::new(Vec::new());
        assert!(matches!(
            box_state.handle_key(ctrl_c(), InputMode::Idle, &dir()),
            InputOutcome::CtrlCHint
        ));
    }

    #[test]
    fn a_second_immediate_idle_ctrl_c_quits() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(ctrl_c(), InputMode::Idle, &dir());
        assert!(matches!(
            box_state.handle_key(ctrl_c(), InputMode::Idle, &dir()),
            InputOutcome::Quit
        ));
    }

    #[test]
    fn typing_between_two_idle_ctrl_cs_disarms_the_quit_window() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(ctrl_c(), InputMode::Idle, &dir());
        box_state.handle_key(key(KeyCode::Char('x')), InputMode::Idle, &dir());
        box_state.handle_key(key(KeyCode::Backspace), InputMode::Idle, &dir());
        // Buffer's empty again, but the intervening keystrokes should have
        // reset the armed state -- this Ctrl-C hints again rather than quitting.
        assert!(matches!(
            box_state.handle_key(ctrl_c(), InputMode::Idle, &dir()),
            InputOutcome::CtrlCHint
        ));
    }

    #[test]
    fn ctrl_d_on_empty_buffer_quits() {
        let mut box_state = InputBoxState::new(Vec::new());
        let ctrl_d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL);
        assert!(matches!(
            box_state.handle_key(ctrl_d, InputMode::Idle, &dir()),
            InputOutcome::Quit
        ));
    }

    #[test]
    fn ctrl_d_on_non_empty_buffer_is_a_noop() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(key(KeyCode::Char('x')), InputMode::Idle, &dir());
        let ctrl_d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL);
        assert!(matches!(
            box_state.handle_key(ctrl_d, InputMode::Idle, &dir()),
            InputOutcome::Handled
        ));
        assert_eq!(box_state.buffer, "x");
    }

    #[test]
    fn backspace_removes_the_char_before_the_cursor() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(key(KeyCode::Char('a')), InputMode::Idle, &dir());
        box_state.handle_key(key(KeyCode::Char('b')), InputMode::Idle, &dir());
        box_state.handle_key(key(KeyCode::Backspace), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "a");
        assert_eq!(box_state.cursor, 1);
    }

    #[test]
    fn up_recalls_the_most_recent_history_entry() {
        let mut box_state = InputBoxState::new(vec!["first".to_string(), "second".to_string()]);
        box_state.handle_key(key(KeyCode::Up), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "second");
    }

    #[test]
    fn down_past_the_newest_entry_restores_the_in_progress_draft() {
        let mut box_state = InputBoxState::new(vec!["first".to_string()]);
        box_state.handle_key(key(KeyCode::Char('d')), InputMode::Idle, &dir());
        box_state.handle_key(key(KeyCode::Up), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "first");
        box_state.handle_key(key(KeyCode::Down), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "d");
    }

    #[test]
    fn submitting_appends_to_history_but_skips_immediate_repeats() {
        let mut box_state = InputBoxState::new(Vec::new());
        for line in ["hi", "hi"] {
            for c in line.chars() {
                box_state.handle_key(key(KeyCode::Char(c)), InputMode::Idle, &dir());
            }
            box_state.handle_key(key(KeyCode::Enter), InputMode::Idle, &dir());
        }
        assert_eq!(box_state.history, ["hi"]);
    }

    #[test]
    fn cursor_screen_position_accounts_for_double_width_chars_before_the_cursor() {
        // Status row + top border + 1 input line + bottom border.
        let area = Rect::new(0, 0, 20, 4);
        // "あ" is one char but two display columns -- the column after it
        // should advance by 2, not 1.
        assert_eq!(cursor_screen_position_for(area, "あ", 1, &[]), (area.x + 1 + 2 + 2, area.y + 2));
        assert_eq!(cursor_screen_position_for(area, "あい", 2, &[]), (area.x + 1 + 2 + 4, area.y + 2));
    }

    #[test]
    fn bottom_area_is_the_last_four_rows_of_a_tall_frame_for_a_single_line_buffer() {
        let area = Rect::new(0, 0, 80, 24);
        assert_eq!(bottom_area(area, "", &[]), Rect::new(0, 20, 80, 4));
        assert_eq!(transcript_area(area, "", &[]), Rect::new(0, 0, 80, 20));
    }

    #[test]
    fn bottom_area_shrinks_instead_of_overflowing_a_short_frame() {
        let area = Rect::new(0, 0, 80, 2);
        assert_eq!(bottom_area(area, "", &[]), Rect::new(0, 0, 80, 2));
        assert_eq!(transcript_area(area, "", &[]), Rect::new(0, 0, 80, 0));
    }

    #[test]
    fn bottom_area_grows_with_extra_buffer_lines_up_to_the_cap() {
        let area = Rect::new(0, 0, 80, 24);
        assert_eq!(bottom_area(area, "one\ntwo", &[]).height, 5); // status + 2 borders + 2 lines
        // 10 lines is more than MAX_INPUT_LINES -- height caps instead of growing further.
        let many = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10";
        assert_eq!(bottom_area(area, many, &[]).height, 3 + MAX_INPUT_LINES as u16);
    }

    #[test]
    fn bottom_area_grows_for_pending_questions_and_caps_at_max_pending_lines() {
        let area = Rect::new(0, 0, 80, 24);
        let one = vec!["question one".to_string()];
        assert_eq!(bottom_area(area, "", &one).height, 5); // status + 1 pending + 2 borders + 1 input line
        let many: Vec<String> = (0..5).map(|i| format!("q{i}")).collect();
        assert_eq!(bottom_area(area, "", &many).height, (1 + MAX_PENDING_LINES + 2 + 1) as u16);
    }

    #[test]
    fn pending_questions_render_above_the_box_capped_with_an_overflow_summary() {
        let area = Rect::new(0, 0, 80, 24);
        let many: Vec<String> = (0..5).map(|i| format!("q{i}")).collect();
        let bottom = bottom_area(area, "", &many);
        let (_, pending_area, box_area, _) = split_bottom(bottom, &many);
        assert_eq!(pending_area.unwrap().height, MAX_PENDING_LINES as u16);
        assert!(box_area.is_some());
    }

    #[test]
    fn alt_enter_inserts_a_newline_instead_of_submitting() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(key(KeyCode::Char('a')), InputMode::Idle, &dir());
        let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
        assert!(matches!(
            box_state.handle_key(alt_enter, InputMode::Idle, &dir()),
            InputOutcome::Handled
        ));
        box_state.handle_key(key(KeyCode::Char('b')), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "a\nb");
    }

    #[test]
    fn plain_enter_still_submits_a_multi_line_buffer() {
        let mut box_state = InputBoxState::new(Vec::new());
        box_state.handle_key(key(KeyCode::Char('a')), InputMode::Idle, &dir());
        box_state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT), InputMode::Idle, &dir());
        box_state.handle_key(key(KeyCode::Char('b')), InputMode::Idle, &dir());
        match box_state.handle_key(key(KeyCode::Enter), InputMode::Idle, &dir()) {
            InputOutcome::Submit(line) => assert_eq!(line, "a\nb"),
            _ => panic!("expected Submit"),
        }
    }

    #[test]
    fn up_and_down_move_between_lines_instead_of_history_once_the_buffer_has_a_newline() {
        let mut box_state = InputBoxState::new(vec!["old history entry".to_string()]);
        box_state.handle_key(key(KeyCode::Char('a')), InputMode::Idle, &dir());
        box_state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT), InputMode::Idle, &dir());
        box_state.handle_key(key(KeyCode::Char('b')), InputMode::Idle, &dir());
        assert_eq!(box_state.cursor, 3); // after "a\nb"

        box_state.handle_key(key(KeyCode::Up), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "a\nb", "Up should move the cursor, not recall history");
        // Column is preserved where possible, clamped to line 0's length ("a" -> col 1).
        assert_eq!(box_state.cursor, 1);

        box_state.handle_key(key(KeyCode::Down), InputMode::Idle, &dir());
        assert_eq!(box_state.cursor, 3); // back to the end of "b"
    }

    #[test]
    fn home_and_end_operate_on_the_current_line_of_a_multi_line_buffer() {
        let mut box_state = InputBoxState::new(Vec::new());
        for c in "ab".chars() {
            box_state.handle_key(key(KeyCode::Char(c)), InputMode::Idle, &dir());
        }
        box_state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT), InputMode::Idle, &dir());
        for c in "cd".chars() {
            box_state.handle_key(key(KeyCode::Char(c)), InputMode::Idle, &dir());
        }
        // Cursor is after "cd" on the second line.
        box_state.handle_key(key(KeyCode::Home), InputMode::Idle, &dir());
        assert_eq!(box_state.cursor, 3); // start of "cd", right after "ab\n"
        box_state.handle_key(key(KeyCode::End), InputMode::Idle, &dir());
        assert_eq!(box_state.cursor, 5); // end of "cd"
    }

    #[test]
    fn tab_completes_an_unambiguous_slash_command() {
        let mut box_state = InputBoxState::new(Vec::new());
        for c in "/hel".chars() {
            box_state.handle_key(key(KeyCode::Char(c)), InputMode::Idle, &dir());
        }
        box_state.handle_key(key(KeyCode::Tab), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "/help ");
    }
}
