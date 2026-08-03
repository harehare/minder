//! Pinned-input-box REPL: a `ratatui` inline viewport that keeps a single
//! input box always visible at the bottom of the terminal, whether or not a
//! turn is currently running, so typing is never gated on first noticing
//! the turn finished -- replacing the old split between `rustyline`
//! (between turns) and `input_watcher::InputWatcher` (during a turn) with
//! one continuous mechanism. Falls back entirely to the old split when the
//! terminal can't do raw-mode key watching at all (see
//! `input_watcher::supports_key_watching`, checked by the caller before
//! any of this module is reached).

pub(crate) mod input_box;
pub(crate) mod sink;

use std::io::IsTerminal;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::event::{Event, EventStream, KeyEventKind};
use futures_util::StreamExt;
use minder_core::{AgentError, AgentSession, Message, Reporter};
use ratatui::layout::Rect;

use crate::reporter::{BOLD, CYAN, RESET, YELLOW};

use input_box::{InputBoxState, InputMode, InputOutcome};
use sink::PinnedInputSnapshot;
pub(crate) use sink::{DirectPrintSink, InlineViewportSink, OutputSink, PinnedHandles};

/// How often the pinned box redraws on its own while a turn is running (so
/// the spinner's elapsed-seconds counter advances even with no keystrokes)
/// -- matches `reporter::SPINNER_INTERVAL`.
const REDRAW_INTERVAL: Duration = Duration::from_millis(90);

/// Same grace period `run_turn_interruptible` uses -- see its doc comment
/// for why: time for a cancelled tool (e.g. `bash` killing its child) to
/// wind down on its own before a second interrupt force-aborts.
const INTERRUPT_GRACE_PERIOD: Duration = Duration::from_millis(1500);

/// Fixed rows reserved at the bottom of the terminal: a dim rule, the
/// prompt/input line, and a status line (spinner-or-hints). Not
/// dynamically resized for multi-line input yet -- see the plan's open
/// design questions.
const VIEWPORT_HEIGHT: u16 = 3;

/// Initializes the ratatui inline viewport and its shared status-line
/// handle. Returns `Err` (never panics) if raw mode / terminal init fails
/// even though `input_watcher::supports_key_watching()` said it should
/// work -- callers fall back to the plain REPL in that case.
pub(crate) fn init() -> std::io::Result<PinnedHandles> {
    let terminal = ratatui::try_init_with_options(ratatui::TerminalOptions {
        viewport: ratatui::Viewport::Inline(VIEWPORT_HEIGHT),
    })?;
    Ok(PinnedHandles {
        terminal: Arc::new(Mutex::new(terminal)),
        status: Arc::new(Mutex::new(String::new())),
        input: Arc::new(Mutex::new(PinnedInputSnapshot::default())),
    })
}

/// Drives one interactive session end-to-end -- same shape as
/// `run_repl_fallback`, but idle input and mid-turn steering both go
/// through the same `InputBoxState` instead of `rustyline` vs.
/// `InputWatcher`. Slash commands (including `/plan`'s own nested turn and
/// y/N confirmation) still run through the existing `handle_slash_command`
/// unchanged, briefly borrowing the terminal the old way -- `clear()`
/// after each one discards ratatui's diff cache so the next redraw doesn't
/// assume anything about what a nested `rustyline`/`InputWatcher` call left
/// on screen.
pub(crate) async fn run_tui_repl(
    built: &mut crate::BuiltSession,
    dir: &Path,
    record: &mut crate::session_store::SessionRecord,
    handles: PinnedHandles,
) {
    let color = crate::color_enabled(std::io::stdout().is_terminal());
    let history_path = crate::session_store::history_path(dir).ok();
    let initial_history = history_path.as_deref().map(load_history).unwrap_or_default();
    let mut box_state = InputBoxState::new(initial_history);
    let mut events = EventStream::new();

    loop {
        let idle_status = idle_status_text(&built.session, dir);
        let Some(line) = read_line(&handles, &mut box_state, &mut events, dir, &idle_status, color).await else {
            break;
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        echo_submitted(&built.reporter, InputMode::Idle, &line, color).await;
        if line == "exit" || line == "quit" {
            break;
        }
        if let Some(path) = &history_path {
            append_history(path, &line);
        }

        if let Some(command) = line.strip_prefix('/') {
            let mut editor: crate::ReplEditor = rustyline::Editor::new().expect("failed to initialize line editor");
            crate::handle_slash_command(command, built, dir, record, &mut editor).await;
            // A nested `/plan` turn (`run_turn_interruptible`) spawns its own
            // `InputWatcher`, which unconditionally disables raw mode on
            // exit -- put it back, then discard ratatui's stale diff/cursor
            // assumptions so the next redraw repaints cleanly instead of
            // corrupting whatever `rustyline`/`InputWatcher` left on screen.
            let _ = crossterm::terminal::enable_raw_mode();
            let _ = handles.terminal.lock().unwrap().clear();
            continue;
        }

        let expanded = crate::mentions::expand_mentions(&line, dir);
        built.checkpoint.start_turn();
        let result = run_turn_pinned(
            &mut built.session,
            &expanded,
            &handles,
            &mut box_state,
            &mut events,
            dir,
            color,
            &built.reporter,
        )
        .await;
        if let Err(e) = &result {
            print_turn_error_pinned(&built.reporter, e, &built.checkpoint).await;
        }
        crate::persist(dir, record, &built.session);
    }

    let _ = handles.terminal.lock().unwrap().clear();
    let _ = ratatui::try_restore();
}

/// Echoes what the user just submitted into the permanent scrollback.
/// `InputBoxState::commit` (see its doc comment) clears the pinned box's own
/// owned region on submit, unlike the old `rustyline`/`InputWatcher` split
/// this replaced -- there, the typed line stayed on screen as a side effect
/// of normal terminal echo. Without this, a submitted line (especially
/// mid-turn steering text) vanishes with no record it was ever sent.
/// `TerminalReporter::on_steering_message`'s doc comment already assumes
/// this echo happened upstream -- this is what makes that assumption true.
async fn echo_submitted(reporter: &Arc<dyn Reporter>, mode: InputMode, text: &str, color: bool) {
    let (glyph, glyph_color) = match mode {
        InputMode::Idle => ("❯", CYAN),
        InputMode::Running => ("»", YELLOW),
    };
    let line = if color {
        format!("{BOLD}{glyph_color}{glyph}{RESET} {text}")
    } else {
        format!("{glyph} {text}")
    };
    reporter.on_notice(&line).await;
}

fn idle_status_text(session: &AgentSession, dir: &Path) -> String {
    format!(
        "{} ({}) · {} · Ctrl-D/exit/Ctrl-C twice quits · Tab completes",
        session.provider_id(),
        session.model(),
        dir.display()
    )
}

/// Reads one line from the pinned box while idle (no turn running):
/// redraws on every keystroke, returns `None` on Ctrl-D/stream end (the
/// caller treats that like `rustyline`'s `ReadlineError::Eof`).
async fn read_line(
    handles: &PinnedHandles,
    box_state: &mut InputBoxState,
    events: &mut EventStream,
    dir: &Path,
    status_text: &str,
    color: bool,
) -> Option<String> {
    redraw(handles, box_state, InputMode::Idle, status_text, color);
    loop {
        let event = match events.next().await {
            None => return None,
            Some(Err(_)) => continue,
            Some(Ok(e)) => e,
        };
        let Event::Key(key) = event else { continue };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }
        // Shows a "press again to exit" hint for exactly the one redraw
        // right after a first idle Ctrl-C, in place of the usual idle
        // status text -- the actual quit window is timed by
        // `InputBoxState` itself (see `CTRL_C_QUIT_WINDOW`); this is purely
        // what the box displays in the meantime.
        let show_ctrl_c_hint = match box_state.handle_key(key, InputMode::Idle, dir) {
            InputOutcome::Submit(line) => return Some(line),
            InputOutcome::Quit => return None,
            InputOutcome::CtrlCHint => true,
            InputOutcome::Handled | InputOutcome::CancelTurn => false,
        };
        let text = if show_ctrl_c_hint {
            "Ctrl-C again to exit"
        } else {
            status_text
        };
        redraw(handles, box_state, InputMode::Idle, text, color);
    }
}

/// Runs one turn to completion while keeping the pinned box live: same
/// cancel/steering contract `run_turn_interruptible` gives the fallback
/// REPL (first Esc/Ctrl-C cancels and starts a grace period, a second one
/// or the grace period elapsing force-aborts), but driven by this module's
/// own `EventStream` poll instead of a per-turn `InputWatcher`, so the box
/// never disappears or hands off to a different input mechanism.
#[allow(clippy::too_many_arguments)]
async fn run_turn_pinned(
    session: &mut AgentSession,
    task: &str,
    handles: &PinnedHandles,
    box_state: &mut InputBoxState,
    events: &mut EventStream,
    dir: &Path,
    color: bool,
    reporter: &Arc<dyn Reporter>,
) -> Result<Message, AgentError> {
    let pre_turn_len = session.messages().len();
    let cancel = session.reset_cancel_token();
    let steering_tx = session.enable_steering();

    redraw(
        handles,
        box_state,
        InputMode::Running,
        &handles.status.lock().unwrap().clone(),
        color,
    );

    // Scoped so `turn` (and the mutable borrow of `session` it holds) ends
    // before `discard_interrupted_turn` below needs its own borrow -- same
    // shape `run_turn_interruptible` uses in `main.rs` and for the same
    // reason.
    let result = {
        let turn = session.run_turn(task);
        tokio::pin!(turn);

        let mut cancelled = false;
        let mut deadline: Option<tokio::time::Instant> = None;
        let mut ticker = tokio::time::interval(REDRAW_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            let wait_for_deadline = async {
                match deadline {
                    Some(d) => tokio::time::sleep_until(d).await,
                    None => std::future::pending::<()>().await,
                }
            };

            tokio::select! {
                biased;
                result = &mut turn => break result,
                () = wait_for_deadline => break Err(AgentError::Interrupted),
                _ = ticker.tick() => {
                    redraw(handles, box_state, InputMode::Running, &handles.status.lock().unwrap().clone(), color);
                }
                event = events.next() => {
                    let event = match event {
                        None => continue,
                        Some(Err(_)) => continue,
                        Some(Ok(e)) => e,
                    };
                    let Event::Key(key) = event else { continue };
                    if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                        continue;
                    }
                    match box_state.handle_key(key, InputMode::Running, dir) {
                        InputOutcome::CancelTurn => {
                            if cancelled {
                                break Err(AgentError::Interrupted);
                            }
                            cancelled = true;
                            cancel.cancel();
                            deadline = Some(tokio::time::Instant::now() + INTERRUPT_GRACE_PERIOD);
                        }
                        InputOutcome::Submit(text) => {
                            echo_submitted(reporter, InputMode::Running, &text, color).await;
                            let _ = steering_tx.send(text);
                        }
                        // `handle_key` never returns `Quit`/`CtrlCHint` while
                        // `mode == Running` -- those are idle-only outcomes.
                        InputOutcome::Quit | InputOutcome::CtrlCHint | InputOutcome::Handled => {}
                    }
                    redraw(handles, box_state, InputMode::Running, &handles.status.lock().unwrap().clone(), color);
                }
            }
        }
    };

    if matches!(result, Err(AgentError::Interrupted)) {
        session.discard_interrupted_turn(pre_turn_len);
    }
    result
}

/// Draws the pinned box and, first, publishes its current contents into
/// `handles.input` -- the shared snapshot `InlineViewportSink::insert_text`
/// reads from to redraw the box itself right after scrollback output clears
/// it (see `sink::PinnedInputSnapshot`). Updating it here, right before every
/// draw this module does directly, keeps it correct without a second place
/// that has to remember to touch it.
fn redraw(handles: &PinnedHandles, box_state: &InputBoxState, mode: InputMode, status_text: &str, color: bool) {
    *handles.input.lock().unwrap() = box_state.snapshot(mode);

    let mut term = handles.terminal.lock().unwrap();
    let _ = term.draw(|frame| {
        let area = frame.area();
        box_state.render(frame, area, mode, status_text, color);
        if area.height >= 2 {
            let input_row = Rect {
                y: area.y + 1,
                height: 1,
                ..area
            };
            frame.set_cursor_position((box_state.cursor_column(input_row), input_row.y));
        }
    });
}

/// Same wording as `print_turn_error` in `main.rs`, but routed through the
/// reporter (`on_notice`) instead of raw `println!`/`eprintln!`, so it
/// lands above the pinned box via `insert_before` instead of corrupting it.
async fn print_turn_error_pinned(
    reporter: &Arc<dyn Reporter>,
    err: &AgentError,
    checkpoint: &minder_tools::Checkpoint,
) {
    if matches!(err, AgentError::Interrupted) {
        reporter.on_notice("Interrupted.").await;
        if !checkpoint.is_empty() {
            reporter
                .on_notice("note: this turn already edited file(s) on disk before being interrupted -- run /undo to revert them.")
                .await;
        }
    } else {
        reporter.on_notice(&format!("error: {err}")).await;
    }
}

/// Rustyline's `FileHistory` v2 on-disk format: a `#V2` header line, then
/// one entry per line with `\`/`\n` escaped -- read here (best-effort, not
/// used to persist compaction/expiry rules `FileHistory` itself has) so
/// history carries over regardless of which REPL mode wrote it, and
/// written in the same shape so a later `rustyline`-backed session can
/// still read it.
const HISTORY_VERSION_HEADER: &str = "#V2";

fn load_history(path: &Path) -> Vec<String> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut lines = contents.lines();
    let mut entries = Vec::new();
    if let Some(first) = lines.next()
        && first != HISTORY_VERSION_HEADER
        && !first.is_empty()
    {
        entries.push(unescape_history_line(first));
    }
    for line in lines {
        if line.is_empty() {
            continue;
        }
        entries.push(unescape_history_line(line));
    }
    entries
}

fn unescape_history_line(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn append_history(path: &Path, line: &str) {
    use std::io::Write;
    let escaped = line.replace('\\', r"\\").replace('\n', r"\n");
    let is_new = !path.exists();
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        if is_new {
            let _ = writeln!(file, "{HISTORY_VERSION_HEADER}");
        }
        let _ = writeln!(file, "{escaped}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_entries_through_the_v2_escaped_format() {
        let dir = std::env::temp_dir().join(format!("minder-tui-history-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chat.history");

        append_history(&path, "plain line");
        append_history(&path, "line with\nan embedded newline");
        append_history(&path, r"line with a \ backslash");

        let loaded = load_history(&path);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(
            loaded,
            vec![
                "plain line".to_string(),
                "line with\nan embedded newline".to_string(),
                r"line with a \ backslash".to_string(),
            ]
        );
    }

    #[test]
    fn missing_history_file_loads_as_empty() {
        let path = std::env::temp_dir().join(format!("minder-tui-history-missing-{}", uuid::Uuid::new_v4()));
        assert!(load_history(&path).is_empty());
    }
}
