use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
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
                self.cursor = 0;
                InputOutcome::Handled
            }
            KeyCode::End => {
                self.cursor = self.buffer.chars().count();
                InputOutcome::Handled
            }
            KeyCode::Up => {
                self.navigate_history(-1);
                InputOutcome::Handled
            }
            KeyCode::Down => {
                self.navigate_history(1);
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

    /// Renders the box into `area` (the bottom rule/status/input rows): a
    /// dim rule, a status line, and the prompt + buffer pinned to the last row.
    pub(crate) fn render(&self, frame: &mut ratatui::Frame, area: Rect, mode: InputMode, status: &str, color: bool) {
        render_pinned(frame, area, &self.buffer, self.cursor, mode, status, color);
    }

    /// Cursor column within the input row, for `Frame::set_cursor_position`.
    pub(crate) fn cursor_column(&self, area: Rect) -> u16 {
        cursor_column_for(area, &self.buffer, self.cursor)
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

/// Free-function core of `InputBoxState::render`, taking buffer + cursor
/// instead of `&InputBoxState` -- lets `sink::append_text` redraw the box
/// from its own cheap snapshot without a live `InputBoxState`.
pub(crate) fn render_pinned(
    frame: &mut ratatui::Frame,
    area: Rect,
    buffer: &str,
    _cursor: usize,
    mode: InputMode,
    status: &str,
    color: bool,
) {
    if area.height == 0 {
        return;
    }
    let rows = split_rows(area);

    if let Some(rule_area) = rows.0 {
        let rule = "-".repeat(rule_area.width as usize);
        let style = if color {
            Style::default().add_modifier(Modifier::DIM)
        } else {
            Style::default()
        };
        frame.render_widget(Paragraph::new(Line::styled(rule, style)), rule_area);
    }

    if let Some(status_area) = rows.2 {
        let style = if color {
            Style::default().add_modifier(Modifier::DIM)
        } else {
            Style::default()
        };
        frame.render_widget(Paragraph::new(Line::styled(status.to_string(), style)), status_area);
    }

    if let Some(input_area) = rows.1 {
        let glyph = match mode {
            InputMode::Idle => "❯",
            InputMode::Running => "»",
        };
        let glyph_style = if !color {
            Style::default()
        } else if mode == InputMode::Running {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        };
        let line = Line::from(vec![
            Span::styled(format!("{glyph} "), glyph_style),
            Span::raw(buffer.to_string()),
        ]);
        frame.render_widget(Paragraph::new(line), input_area);
    }
}

/// Free-function core of `InputBoxState::cursor_column` -- see `render_pinned`.
/// Sums display width, not char count, so double-width (CJK/emoji) chars don't
/// throw off the on-screen cursor position.
pub(crate) fn cursor_column_for(area: Rect, buffer: &str, cursor: usize) -> u16 {
    let prefix_width = 2u16; // glyph + one space, both single-column
    let col: u16 = buffer[..char_boundary(buffer, cursor)]
        .chars()
        .map(|c| c.width().unwrap_or(0) as u16)
        .sum();
    area.x + prefix_width + col
}

/// The input row within `area` -- the bottom-most row, or the second row if
/// too short for a status line. Shared by `tui::redraw` and `sink::append_text`.
pub(crate) fn input_row(area: Rect) -> Option<Rect> {
    split_rows(area).1
}

/// Rows reserved at the bottom of a full frame for rule/status/input;
/// everything above is the scrollable transcript pane.
pub(crate) const BOTTOM_ROWS: u16 = 3;

/// The bottom `BOTTOM_ROWS` rows of a full frame (fewer if the terminal is
/// shorter than that) -- what `render_pinned`/`input_row` treat as `area`.
pub(crate) fn bottom_area(frame_area: Rect) -> Rect {
    let height = BOTTOM_ROWS.min(frame_area.height);
    Rect {
        y: frame_area.y + frame_area.height - height,
        height,
        ..frame_area
    }
}

/// Everything above `bottom_area(frame_area)` -- the scrollable transcript pane.
pub(crate) fn transcript_area(frame_area: Rect) -> Rect {
    Rect {
        height: frame_area.height - bottom_area(frame_area).height,
        ..frame_area
    }
}

/// Splits a 3-row area into (rule, input, status), input pinned to the
/// *last* row so it's never the one pushed offscreen. Degrades gracefully
/// on a shorter area (`None` for rows that don't fit; input wins over
/// status when there's only room for 2).
fn split_rows(area: Rect) -> (Option<Rect>, Option<Rect>, Option<Rect>) {
    let rule = (area.height >= 1).then_some(Rect { height: 1, ..area });
    let status = (area.height >= 3).then(|| Rect {
        y: area.y + 1,
        height: 1,
        ..area
    });
    let input_y = if area.height >= 3 { area.y + 2 } else { area.y + 1 };
    let input = (area.height >= 2).then_some(Rect {
        y: input_y,
        height: 1,
        ..area
    });
    (rule, input, status)
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
    fn cursor_column_accounts_for_double_width_chars_before_the_cursor() {
        let area = Rect::new(0, 0, 20, 1);
        // "あ" is one char but two display columns -- the column after it
        // should advance by 2, not 1.
        assert_eq!(cursor_column_for(area, "あ", 1), area.x + 2 + 2);
        assert_eq!(cursor_column_for(area, "あい", 2), area.x + 2 + 4);
    }

    #[test]
    fn bottom_area_is_the_last_three_rows_of_a_tall_frame() {
        let area = Rect::new(0, 0, 80, 24);
        assert_eq!(bottom_area(area), Rect::new(0, 21, 80, 3));
        assert_eq!(transcript_area(area), Rect::new(0, 0, 80, 21));
    }

    #[test]
    fn bottom_area_shrinks_instead_of_overflowing_a_short_frame() {
        let area = Rect::new(0, 0, 80, 2);
        assert_eq!(bottom_area(area), Rect::new(0, 0, 80, 2));
        assert_eq!(transcript_area(area), Rect::new(0, 0, 80, 0));
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
