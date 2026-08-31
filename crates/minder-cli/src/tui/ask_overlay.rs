//! Modal state for `ask_user_question`, driven by `tui::run_turn_pinned`'s `ask_rx` branch.

use crossterm::event::{KeyCode, KeyEvent};
use minder_core::{AskAnswer, AskQuestion};
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::input_box::{PANEL_BG, PANEL_FG};

/// Placeholder recorded for a question the user backed out of with `Esc`.
const SKIPPED: &str = "(skipped)";

enum Mode {
    List,
    TextInput,
}

pub(crate) enum AskOverlayOutcome {
    Continue,
    Finished(Vec<AskAnswer>),
}

pub(crate) struct AskOverlayState {
    questions: Vec<AskQuestion>,
    current: usize,
    highlighted: usize,
    checked: Vec<bool>,
    mode: Mode,
    text_buffer: String,
    answers: Vec<AskAnswer>,
}

impl AskOverlayState {
    pub(crate) fn new(questions: Vec<AskQuestion>) -> Self {
        let checked = vec![false; questions[0].options.len()];
        Self {
            questions,
            current: 0,
            highlighted: 0,
            checked,
            mode: Mode::List,
            text_buffer: String::new(),
            answers: Vec::new(),
        }
    }

    fn current_question(&self) -> &AskQuestion {
        &self.questions[self.current]
    }

    fn row_count(&self) -> usize {
        self.current_question().options.len() + 1
    }

    pub(crate) fn handle_key(&mut self, key: KeyEvent) -> AskOverlayOutcome {
        match self.mode {
            Mode::TextInput => self.handle_text_input_key(key),
            Mode::List => self.handle_list_key(key),
        }
    }

    fn handle_list_key(&mut self, key: KeyEvent) -> AskOverlayOutcome {
        let options_len = self.current_question().options.len();
        let row_count = self.row_count();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.highlighted = (self.highlighted + row_count - 1) % row_count;
                AskOverlayOutcome::Continue
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.highlighted = (self.highlighted + 1) % row_count;
                AskOverlayOutcome::Continue
            }
            KeyCode::Char(' ') if self.current_question().multi_select && self.highlighted < options_len => {
                self.checked[self.highlighted] = !self.checked[self.highlighted];
                AskOverlayOutcome::Continue
            }
            KeyCode::Char(c) if c.is_ascii_digit() && c != '0' => {
                let idx = c.to_digit(10).unwrap() as usize - 1;
                if idx >= options_len {
                    return AskOverlayOutcome::Continue;
                }
                self.highlighted = idx;
                if self.current_question().multi_select {
                    self.checked[idx] = !self.checked[idx];
                    AskOverlayOutcome::Continue
                } else {
                    self.select_option(idx)
                }
            }
            KeyCode::Enter if self.highlighted == options_len => {
                self.mode = Mode::TextInput;
                self.text_buffer.clear();
                AskOverlayOutcome::Continue
            }
            KeyCode::Enter if self.current_question().multi_select => {
                if self.checked.iter().any(|&c| c) {
                    self.finish_multi_select()
                } else {
                    AskOverlayOutcome::Continue
                }
            }
            KeyCode::Enter => self.select_option(self.highlighted),
            KeyCode::Esc => AskOverlayOutcome::Finished(self.skip_remaining()),
            _ => AskOverlayOutcome::Continue,
        }
    }

    fn handle_text_input_key(&mut self, key: KeyEvent) -> AskOverlayOutcome {
        match key.code {
            KeyCode::Esc => {
                self.mode = Mode::List;
                self.text_buffer.clear();
                AskOverlayOutcome::Continue
            }
            KeyCode::Backspace => {
                self.text_buffer.pop();
                AskOverlayOutcome::Continue
            }
            KeyCode::Char(c) => {
                self.text_buffer.push(c);
                AskOverlayOutcome::Continue
            }
            KeyCode::Enter => {
                if self.text_buffer.trim().is_empty() {
                    return AskOverlayOutcome::Continue;
                }
                let text = std::mem::take(&mut self.text_buffer);
                let mut selected = self.checked_labels();
                selected.push(text);
                self.push_answer_and_advance(selected)
            }
            _ => AskOverlayOutcome::Continue,
        }
    }

    fn checked_labels(&self) -> Vec<String> {
        self.current_question()
            .options
            .iter()
            .zip(&self.checked)
            .filter(|(_, checked)| **checked)
            .map(|(o, _)| o.label.clone())
            .collect()
    }

    fn select_option(&mut self, idx: usize) -> AskOverlayOutcome {
        let label = self.current_question().options[idx].label.clone();
        self.push_answer_and_advance(vec![label])
    }

    fn finish_multi_select(&mut self) -> AskOverlayOutcome {
        let selected = self.checked_labels();
        self.push_answer_and_advance(selected)
    }

    fn push_answer_and_advance(&mut self, selected: Vec<String>) -> AskOverlayOutcome {
        let header = self.current_question().header.clone();
        self.answers.push(AskAnswer { header, selected });
        self.current += 1;
        if self.current == self.questions.len() {
            return AskOverlayOutcome::Finished(std::mem::take(&mut self.answers));
        }
        self.highlighted = 0;
        self.checked = vec![false; self.current_question().options.len()];
        self.mode = Mode::List;
        AskOverlayOutcome::Continue
    }

    fn skip_remaining(&mut self) -> Vec<AskAnswer> {
        while self.current < self.questions.len() {
            let header = self.questions[self.current].header.clone();
            self.answers.push(AskAnswer {
                header,
                selected: vec![SKIPPED.to_string()],
            });
            self.current += 1;
        }
        std::mem::take(&mut self.answers)
    }

    pub(crate) fn render(&self, frame: &mut ratatui::Frame, area: Rect, color: bool) {
        let modal = centered(area, 70, 60.min(area.height.saturating_sub(2)));
        frame.render_widget(ratatui::widgets::Clear, modal);

        let panel_style = if color {
            Style::default().bg(PANEL_BG).fg(PANEL_FG)
        } else {
            Style::default()
        };
        let border_style = if color {
            panel_style.fg(Color::Yellow)
        } else {
            Style::default()
        };
        let title = format!(" Question {} of {} ", self.current + 1, self.questions.len());
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title(title)
            .style(panel_style)
            .border_style(border_style);
        let inner = block.inner(modal);
        frame.render_widget(block, modal);

        let question = self.current_question();
        let mut lines = vec![
            Line::from(Span::styled(
                question.header.clone(),
                panel_style.add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(question.question.clone(), panel_style)),
            Line::from(""),
        ];

        match self.mode {
            Mode::List => {
                for (i, option) in question.options.iter().enumerate() {
                    lines.push(self.option_line(i, &option.label, &option.description, panel_style, color));
                }
                lines.push(self.option_line(
                    question.options.len(),
                    "Other",
                    "type your own answer",
                    panel_style,
                    color,
                ));
                lines.push(Line::from(""));
                let hint = if question.multi_select {
                    "↑↓ move · Space toggle · 1-9 jump · Enter confirm · Esc skip"
                } else {
                    "↑↓ move · 1-9 select · Enter select · Esc skip"
                };
                lines.push(Line::from(Span::styled(hint, panel_style.add_modifier(Modifier::DIM))));
            }
            Mode::TextInput => {
                lines.push(Line::from(Span::styled(
                    format!("> {}_", self.text_buffer),
                    panel_style,
                )));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "Enter confirm · Esc back",
                    panel_style.add_modifier(Modifier::DIM),
                )));
            }
        }

        frame.render_widget(
            Paragraph::new(lines).style(panel_style).alignment(Alignment::Left),
            inner,
        );
    }

    fn option_line(
        &self,
        idx: usize,
        label: &str,
        description: &str,
        panel_style: Style,
        color: bool,
    ) -> Line<'static> {
        let is_highlighted = idx == self.highlighted;
        let marker = if self.current_question().multi_select && idx < self.checked.len() {
            if self.checked[idx] { "[x] " } else { "[ ] " }
        } else if is_highlighted {
            "❯ "
        } else {
            "  "
        };
        let style = if color && is_highlighted {
            panel_style.add_modifier(Modifier::BOLD).fg(Color::Yellow)
        } else {
            panel_style
        };
        let text = if description.is_empty() {
            format!("{marker}{}. {label}", idx + 1)
        } else {
            format!("{marker}{}. {label} -- {description}", idx + 1)
        };
        Line::from(Span::styled(text, style))
    }
}

/// A `width`x`height` rect centered within `area`, clamped to fit.
fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    Rect::new(x, y, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question(header: &str, multi_select: bool) -> AskQuestion {
        AskQuestion {
            header: header.to_string(),
            question: format!("{header}?"),
            options: vec![
                minder_core::AskOption {
                    label: "A".to_string(),
                    description: String::new(),
                },
                minder_core::AskOption {
                    label: "B".to_string(),
                    description: String::new(),
                },
            ],
            multi_select,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, crossterm::event::KeyModifiers::NONE)
    }

    #[test]
    fn digit_key_selects_and_finishes_a_single_question_single_select() {
        let mut state = AskOverlayState::new(vec![question("Approach", false)]);
        match state.handle_key(key(KeyCode::Char('2'))) {
            AskOverlayOutcome::Finished(answers) => {
                assert_eq!(answers.len(), 1);
                assert_eq!(answers[0].selected, vec!["B".to_string()]);
            }
            AskOverlayOutcome::Continue => panic!("expected the only question to finish the batch"),
        }
    }

    #[test]
    fn multi_select_requires_at_least_one_checked_option_before_enter_confirms() {
        let mut state = AskOverlayState::new(vec![question("Features", true)]);
        assert!(matches!(
            state.handle_key(key(KeyCode::Enter)),
            AskOverlayOutcome::Continue
        ));
        assert!(matches!(
            state.handle_key(key(KeyCode::Char('1'))),
            AskOverlayOutcome::Continue
        ));
        match state.handle_key(key(KeyCode::Enter)) {
            AskOverlayOutcome::Finished(answers) => assert_eq!(answers[0].selected, vec!["A".to_string()]),
            AskOverlayOutcome::Continue => panic!("expected Enter with one checked option to finish"),
        }
    }

    #[test]
    fn other_switches_to_text_input_and_enter_confirms_the_typed_answer() {
        let mut state = AskOverlayState::new(vec![question("Approach", false)]);
        state.highlighted = 2; // the synthetic "Other" row
        assert!(matches!(
            state.handle_key(key(KeyCode::Enter)),
            AskOverlayOutcome::Continue
        ));
        for c in "custom".chars() {
            assert!(matches!(
                state.handle_key(key(KeyCode::Char(c))),
                AskOverlayOutcome::Continue
            ));
        }
        match state.handle_key(key(KeyCode::Enter)) {
            AskOverlayOutcome::Finished(answers) => assert_eq!(answers[0].selected, vec!["custom".to_string()]),
            AskOverlayOutcome::Continue => panic!("expected Enter on the free-text buffer to finish"),
        }
    }

    #[test]
    fn esc_at_list_level_skips_every_remaining_question() {
        let mut state = AskOverlayState::new(vec![question("First", false), question("Second", false)]);
        match state.handle_key(key(KeyCode::Esc)) {
            AskOverlayOutcome::Finished(answers) => {
                assert_eq!(answers.len(), 2);
                assert!(answers.iter().all(|a| a.selected == vec![SKIPPED.to_string()]));
            }
            AskOverlayOutcome::Continue => panic!("expected Esc to finish the whole batch"),
        }
    }

    #[test]
    fn answering_the_first_of_two_questions_advances_without_finishing() {
        let mut state = AskOverlayState::new(vec![question("First", false), question("Second", false)]);
        assert!(matches!(
            state.handle_key(key(KeyCode::Char('1'))),
            AskOverlayOutcome::Continue
        ));
        assert_eq!(state.current, 1);
        assert_eq!(state.highlighted, 0);
    }
}
