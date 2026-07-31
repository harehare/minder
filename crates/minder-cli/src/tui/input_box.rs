use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::mentions;

/// Whether a turn is currently running -- changes both what Enter/Esc do
/// and how the box renders (see `InputBoxState::render`), so a user can
/// tell at a glance whether they're about to submit a new turn or steer a
/// running one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputMode {
    Idle,
    Running,
}

/// What happened to a single key event -- `run_tui_repl` acts on this
/// instead of the box managing its own control flow, so cancel/quit/submit
/// semantics stay in one place next to `AgentSession`'s turn-running calls.
pub(crate) enum InputOutcome {
    /// Consumed, nothing for the caller to do (typing, cursor movement, a
    /// completion, an idle empty-Enter, etc.).
    Handled,
    /// Enter on a non-empty buffer -- a new turn when idle, steering text
    /// when a turn is running.
    Submit(String),
    /// Esc or Ctrl-C while a turn is running.
    CancelTurn,
    /// Ctrl-D (or `exit`/`quit`, checked by the caller) on an empty buffer.
    Quit,
}

/// The always-visible input box's state: text buffer, cursor, and a simple
/// in-memory history -- replaces both `rustyline`'s line editing (used only
/// between turns before this) and `input_watcher.rs`'s hand-rolled raw-mode
/// echo (used only during a turn before this), unifying them into one
/// widget that's on screen the whole time. See `tui::run_tui_repl`.
pub(crate) struct InputBoxState {
    buffer: String,
    cursor: usize,
    history: Vec<String>,
    /// `Some(i)` while browsing history via Up/Down; `None` once back at a
    /// freshly typed (or not-yet-submitted) line.
    history_index: Option<usize>,
    /// What was being typed before Up first moved into history, restored
    /// on Down past the most recent entry.
    draft: String,
}

impl InputBoxState {
    pub(crate) fn new(history: Vec<String>) -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            history,
            history_index: None,
            draft: String::new(),
        }
    }

    /// Records `line` in history (skipping immediate repeats, same as
    /// `rustyline`'s default `HistoryDuplicates::IgnoreConsecutive`) and
    /// resets the buffer for the next line.
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

    /// Completes a `/`-command (single match only, mirroring
    /// `SlashCommandHelper`'s Tab behavior) or a `@mention` path (first
    /// candidate) at the cursor -- only ever active at end-of-line, same
    /// restriction `matching_slash_commands`/`at_mention_token` already
    /// enforce.
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

    /// Renders the box into `area` (expected to be the bottom 3 rows of the
    /// inline viewport: rule / input / status): a dim rule, the prompt glyph
    /// (different in `Running` mode so idle vs. mid-turn is visually
    /// distinct -- see the plan's open design questions) plus the buffer,
    /// and a status line built by the caller (spinner/provider while
    /// running, keyboard hints while idle).
    pub(crate) fn render(&self, frame: &mut ratatui::Frame, area: Rect, mode: InputMode, status: &str, color: bool) {
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
                Span::raw(self.buffer.clone()),
            ]);
            frame.render_widget(Paragraph::new(line), input_area);
        }

        if let Some(status_area) = rows.2 {
            let style = if color {
                Style::default().add_modifier(Modifier::DIM)
            } else {
                Style::default()
            };
            frame.render_widget(Paragraph::new(Line::styled(status.to_string(), style)), status_area);
        }
    }

    /// Cursor column within `area` (the input row specifically), for
    /// `Frame::set_cursor_position` -- accounts for the `"❯ "`/`"» "` glyph
    /// prefix both prompt styles share.
    pub(crate) fn cursor_column(&self, area: Rect) -> u16 {
        let prefix_width = 2u16; // glyph + one space, both single-column
        let col = self.buffer[..char_boundary(&self.buffer, self.cursor)].chars().count() as u16;
        area.x + prefix_width + col
    }
}

/// Splits a 3-row area into (rule, input, status); any row beyond the first
/// is `None` if `area` is shorter than expected (defensive -- a 1-row
/// terminal shouldn't panic, just draw what fits).
fn split_rows(area: Rect) -> (Option<Rect>, Option<Rect>, Option<Rect>) {
    let rule = (area.height >= 1).then(|| Rect { height: 1, ..area });
    let input = (area.height >= 2).then(|| Rect {
        y: area.y + 1,
        height: 1,
        ..area
    });
    let status = (area.height >= 3).then(|| Rect {
        y: area.y + 2,
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
    fn tab_completes_an_unambiguous_slash_command() {
        let mut box_state = InputBoxState::new(Vec::new());
        for c in "/hel".chars() {
            box_state.handle_key(key(KeyCode::Char(c)), InputMode::Idle, &dir());
        }
        box_state.handle_key(key(KeyCode::Tab), InputMode::Idle, &dir());
        assert_eq!(box_state.buffer, "/help ");
    }
}
