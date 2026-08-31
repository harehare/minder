//! Fullscreen REPL: a `ratatui` alternate-screen viewport with an
//! always-visible input box pinned to the bottom and the conversation kept
//! in an in-memory `sink::Transcript`, redrawn as a scrollable pane above it
//! every frame -- replaces an earlier design that pushed output into the
//! *real* terminal scrollback on a non-alt-screen viewport, which raced its
//! own re-repaint under bursty output. Falls back to the old split
//! (`rustyline` idle, `input_watcher::InputWatcher` mid-turn) when the
//! terminal can't do raw-mode key watching at all.

mod ask_overlay;
pub(crate) mod input_box;
pub(crate) mod sink;

use std::io::IsTerminal;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event, EventStream, KeyCode, KeyEventKind, MouseEventKind,
};
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use futures_util::StreamExt;
use minder_core::{
    AgentError, AgentSession, AskChannel, AskReceiver, HookPort, LlmProvider, Message, Reporter, Tool, ToolContext,
};

use crate::reporter::{BOLD, DIM, RED, RESET, YELLOW};

use ask_overlay::{AskOverlayOutcome, AskOverlayState};
use input_box::{InputBoxState, InputMode, InputOutcome};
use sink::PinnedInputSnapshot;
pub(crate) use sink::{DirectPrintSink, FullscreenSink, OutputSink, PinnedHandles};

/// How often the pinned box redraws on its own while running -- matches `reporter::SPINNER_INTERVAL`.
const REDRAW_INTERVAL: Duration = Duration::from_millis(90);

/// Grace period before a second interrupt force-aborts -- see `run_turn_interruptible`.
const INTERRUPT_GRACE_PERIOD: Duration = Duration::from_millis(1500);

/// Wrapped lines a mouse-wheel notch scrolls (`PageUp`/`PageDown` scroll a full pane instead).
const MOUSE_SCROLL_LINES: i32 = 3;

/// Initializes the fullscreen/alt-screen viewport and shared handles.
/// `Err` (never panics) on init failure -- callers fall back to the plain REPL.
///
/// Backend is `BufWriter`-wrapped rather than going through `ratatui::try_init`
/// (raw `stdout()`): `CrosstermBackend::draw` issues one `write()` per changed
/// cell, so unbuffered `Stdout` turns each into its own syscall.
pub(crate) fn init() -> std::io::Result<PinnedHandles> {
    set_panic_hook();
    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let backend = ratatui::backend::CrosstermBackend::new(std::io::BufWriter::new(std::io::stdout()));
    let terminal = ratatui::Terminal::new(backend)?;
    Ok(PinnedHandles {
        terminal: Arc::new(Mutex::new(terminal)),
        status: Arc::new(Mutex::new(String::new())),
        input: Arc::new(Mutex::new(PinnedInputSnapshot::default())),
        transcript: Arc::new(Mutex::new(sink::Transcript::default())),
        pending: Arc::new(Mutex::new(Vec::new())),
    })
}

/// Undoes `init`'s terminal setup -- shared by the panic hook and `run_tui_repl`'s own cleanup.
pub(crate) fn restore_terminal() {
    let _ = crossterm::execute!(std::io::stdout(), DisableMouseCapture);
    let _ = ratatui::try_restore();
}

/// Same panic hook `ratatui::try_init` installs internally, replicated since we use our own backend.
fn set_panic_hook() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        hook(info);
    }));
}

/// Drives one interactive session end-to-end. Slash commands still run
/// through `handle_slash_command` unchanged, printing straight to
/// stdout/rustyline outside ratatui's frame -- so the alternate screen is
/// left before the call and re-entered (plus `terminal.clear()`) after.
pub(crate) async fn run_tui_repl(
    built: &mut crate::BuiltSession,
    dir: &Path,
    record: &mut crate::session_store::SessionRecord,
    handles: PinnedHandles,
    mut ask_rx: AskReceiver,
) {
    let color = crate::color_enabled(std::io::stdout().is_terminal());
    let history_path = crate::session_store::history_path(dir).ok();
    let initial_history = history_path.as_deref().map(load_history).unwrap_or_default();
    let known_models = built.provider.list_models().await.unwrap_or_default();
    let mut box_state = InputBoxState::new(initial_history).with_known_models(known_models);
    // Discards keystrokes buffered before any event loop existed to read
    // them (e.g. during a first-run model pull) -- else they'd replay as a
    // surprise burst, possibly auto-submitting a stray Enter.
    while crossterm::event::poll(Duration::ZERO).unwrap_or(false) {
        if crossterm::event::read().is_err() {
            break;
        }
    }
    let mut events = EventStream::new();
    // What `spawn_side_question` needs to build its own ephemeral sessions
    // for text submitted while a turn is running -- same provider/tools/hooks
    // as the main session, minus `agent` itself (a side question shouldn't
    // spawn its own nested subagents). Captured once here rather than
    // re-cloned per keystroke.
    let side_provider = built.provider.clone();
    let side_tools: Vec<Arc<dyn Tool>> = built.tools.iter().filter(|t| t.name() != "agent").cloned().collect();
    let side_hooks = built.hooks.clone();

    loop {
        let idle_status = idle_status_text(&built.session, dir, &handles);
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
            let _ = crossterm::execute!(std::io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
            crate::handle_slash_command(command, built, dir, record).await;
            // Slash commands print/prompt straight to the real terminal outside
            // ratatui's frame -- put raw mode back, then discard ratatui's stale
            // diff/cursor state before the next redraw.
            let _ = crossterm::terminal::enable_raw_mode();
            let _ = crossterm::execute!(std::io::stdout(), EnterAlternateScreen, EnableMouseCapture);
            let _ = handles.terminal.lock().unwrap().clear();
            continue;
        }

        let expanded = crate::mentions::expand_mentions(&line, dir);
        built.checkpoint.start_turn();
        crate::mark_turn_started(dir, record);
        let result = run_turn_pinned(
            &mut built.session,
            &expanded,
            &handles,
            &mut box_state,
            &mut events,
            dir,
            color,
            &built.reporter,
            &side_provider,
            &side_tools,
            &side_hooks,
            &built.tool_ctx,
            &mut ask_rx,
        )
        .await;
        if let Err(e) = &result {
            print_turn_error_pinned(&built.reporter, e, &built.checkpoint).await;
        }
        crate::persist(dir, record, &built.session);
    }

    let _ = handles.terminal.lock().unwrap().clear();
    restore_terminal();
}

/// Echoes what the user just submitted into the transcript -- `commit`
/// clears the input box on submit, so without this the line would vanish
/// with no record it was sent.
async fn echo_submitted(reporter: &Arc<dyn Reporter>, mode: InputMode, text: &str, color: bool) {
    let (glyph, glyph_color) = match mode {
        InputMode::Idle => ("❯", YELLOW),
        InputMode::Running => ("»", YELLOW),
    };
    let line = if color {
        format!("{BOLD}{glyph_color}{glyph}{RESET} {text}")
    } else {
        format!("{glyph} {text}")
    };
    reporter.on_notice(&line).await;
}

fn idle_status_text(session: &AgentSession, dir: &Path, handles: &PinnedHandles) -> String {
    let base = format!(
        "{} ({}) · {} · Alt+Enter for newline · Ctrl-D/exit/Ctrl-C twice quits · Tab completes · PgUp/PgDn scrolls",
        session.provider_id(),
        session.model(),
        dir.display()
    );
    match handles.pending.lock().unwrap().len() {
        0 => base,
        n => format!("{base} · {n} answering"),
    }
}

/// `(width, visible_height)` of the transcript pane for the current terminal size.
fn transcript_metrics(handles: &PinnedHandles) -> (u16, u16) {
    let size = handles.terminal.lock().unwrap().size().unwrap_or_default();
    let buffer = handles.input.lock().unwrap().buffer.clone();
    let pending = handles.pending.lock().unwrap().clone();
    let frame_area = ratatui::layout::Rect::new(0, 0, size.width, size.height);
    let height = input_box::transcript_area(frame_area, &buffer, &pending).height;
    (size.width, height)
}

/// Scrolls on `PageUp`/`PageDown`/mouse-wheel; `false` for anything else, so
/// the caller falls through to normal handling. Bails before locking
/// anything for non-scroll events -- mouse capture reports every
/// motion/click, not just wheel notches.
fn handle_scroll_event(handles: &PinnedHandles, event: &Event) -> bool {
    enum Scroll {
        PageUp,
        PageDown,
        WheelUp,
        WheelDown,
    }

    let action = match event {
        Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => match key.code {
            KeyCode::PageUp => Scroll::PageUp,
            KeyCode::PageDown => Scroll::PageDown,
            _ => return false,
        },
        Event::Mouse(m) => match m.kind {
            MouseEventKind::ScrollUp => Scroll::WheelUp,
            MouseEventKind::ScrollDown => Scroll::WheelDown,
            _ => return false,
        },
        _ => return false,
    };

    let (width, height) = transcript_metrics(handles);
    let page = i32::from(height.max(1));
    let delta = match action {
        Scroll::PageUp => -page,
        Scroll::PageDown => page,
        Scroll::WheelUp => -MOUSE_SCROLL_LINES,
        Scroll::WheelDown => MOUSE_SCROLL_LINES,
    };
    handles.transcript.lock().unwrap().scroll_by(delta, width, height);
    true
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
        if matches!(event, Event::Resize(_, _)) {
            // Autoresize only runs inside `Terminal::draw`, so force one now.
            redraw(handles, box_state, InputMode::Idle, status_text, color);
            continue;
        }
        if handle_scroll_event(handles, &event) {
            redraw(handles, box_state, InputMode::Idle, status_text, color);
            continue;
        }
        let Event::Key(key) = event else { continue };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }
        // Shows a "press again to exit" hint for one redraw after an idle Ctrl-C
        // (the quit window itself is timed inside `InputBoxState`).
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

/// The running status line, with a suffix noting how many `spawn_side_question`
/// calls haven't answered yet -- so a submission that hasn't come back reads
/// as "still working on it" rather than "lost".
fn running_status(handles: &PinnedHandles) -> String {
    let base = handles.status.lock().unwrap().clone();
    match handles.pending.lock().unwrap().len() {
        0 => base,
        n => format!("{base} · {n} answering"),
    }
}

/// System prompt for the ephemeral session `spawn_side_question` builds --
/// deliberately not the main turn's own system prompt: this session exists
/// for exactly one reply and runs concurrently with whatever the main task
/// is doing in the same working directory, so it's told to act accordingly
/// rather than as the primary agent.
const SIDE_QUESTION_SYSTEM_PROMPT: &str = "You're answering a quick side question the user asked while \
    their main task keeps running concurrently in this same working directory. Use the available tools \
    if you need to, but keep any changes conservative to avoid conflicting with whatever the main task \
    might be doing at the same time. Reply concisely -- this is the only message the user will see from you.";

/// Spawns `question` as an independent `AgentSession` turn -- runs
/// concurrently with whatever the main turn is doing, touches none of its
/// state, and prints its answer into the transcript (via `reporter`) the
/// moment it's ready instead of waiting for the main turn to end. Used for
/// text typed and submitted while a turn is running -- splicing it into
/// that turn's own context instead used to derail the model into addressing
/// the interruption rather than finishing what it was already doing.
///
/// Deliberately built with no reporter of its own (defaults to a no-op)
/// rather than sharing the main session's: `TerminalReporter` tracks the
/// live spinner under one fixed key per session
/// (`reporter::TURN_LABEL_KEY`), so two sessions driving it at once -- this
/// one and the main turn -- would stomp each other's spinner state and
/// status line. Only this function's own final `on_notice` call touches the
/// real reporter.
#[allow(clippy::too_many_arguments)]
fn spawn_side_question(
    provider: Arc<dyn LlmProvider>,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
    question: String,
    tool_ctx: ToolContext,
    reporter: Arc<dyn Reporter>,
    pending: Arc<Mutex<Vec<String>>>,
    color: bool,
) {
    pending.lock().unwrap().push(question.clone());
    tokio::spawn(async move {
        let child_ctx = ToolContext {
            working_dir: tool_ctx.working_dir,
            session_id: format!("{}:aside", tool_ctx.session_id),
            // Independent of the main turn's own cancel token -- Esc-ing
            // out of the main task shouldn't take a side question down
            // with it, and vice versa.
            cancel: tokio_util::sync::CancellationToken::new(),
            mailbox: None,
            ask: AskChannel::unavailable(), // detached, nothing would poll a real receiver
        };
        let mut session = AgentSession::new(provider, tools, hooks, SIDE_QUESTION_SYSTEM_PROMPT, child_ctx);
        let result = session.run_turn(&question).await;
        {
            let mut pending = pending.lock().unwrap();
            if let Some(idx) = pending.iter().position(|q| q == &question) {
                pending.remove(idx);
            }
        }

        let (glyph, glyph_color, text) = match &result {
            Ok(message) => ("↩", DIM, message.text()),
            Err(e) => ("✗", RED, e.to_string()),
        };
        let line = if color {
            format!("{glyph_color}{glyph}{RESET} {text}")
        } else {
            format!("{glyph} {text}")
        };
        reporter.on_notice(&line).await;
    });
}

/// Runs one turn while keeping the pinned box live: same cancel contract as
/// `run_turn_interruptible` (first Esc/Ctrl-C starts a grace period, a
/// second or the deadline force-aborts), driven by this module's own
/// `EventStream` poll instead of a per-turn `InputWatcher`. Anything typed
/// and submitted while this runs is answered concurrently rather than
/// spliced into this turn -- see `spawn_side_question`.
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
    side_provider: &Arc<dyn LlmProvider>,
    side_tools: &[Arc<dyn Tool>],
    side_hooks: &Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
    tool_ctx: &ToolContext,
    ask_rx: &mut AskReceiver,
) -> Result<Message, AgentError> {
    let pre_turn_len = session.messages().len();
    let cancel = session.reset_cancel_token();

    redraw(handles, box_state, InputMode::Running, &running_status(handles), color);

    // Scoped so `turn`'s borrow of `session` ends before `discard_interrupted_turn` needs its own.
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
                    redraw(handles, box_state, InputMode::Running, &running_status(handles), color);
                }
                event = events.next() => {
                    let event = match event {
                        None => continue,
                        Some(Err(_)) => continue,
                        Some(Ok(e)) => e,
                    };
                    if handle_scroll_event(handles, &event) {
                        redraw(handles, box_state, InputMode::Running, &running_status(handles), color);
                        continue;
                    }
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
                            spawn_side_question(
                                side_provider.clone(),
                                side_tools.to_vec(),
                                side_hooks.clone(),
                                text,
                                tool_ctx.clone(),
                                reporter.clone(),
                                handles.pending.clone(),
                                color,
                            );
                        }
                        // `handle_key` never returns `Quit`/`CtrlCHint` while
                        // `mode == Running` -- those are idle-only outcomes.
                        InputOutcome::Quit | InputOutcome::CtrlCHint | InputOutcome::Handled => {}
                    }
                    redraw(handles, box_state, InputMode::Running, &running_status(handles), color);
                }
                Some(request) = ask_rx.recv() => {
                    // `turn` is parked inside `ctx.ask.ask().await`, so nothing else needs `events` meanwhile.
                    let answers = run_ask_overlay(handles, events, request.questions, color).await;
                    let _ = request.reply.send(answers);
                    redraw(handles, box_state, InputMode::Running, &running_status(handles), color);
                }
            }
        }
    };

    if matches!(result, Err(AgentError::Interrupted)) {
        session.discard_interrupted_turn(pre_turn_len);
    }
    result
}

async fn run_ask_overlay(
    handles: &PinnedHandles,
    events: &mut EventStream,
    questions: Vec<minder_core::AskQuestion>,
    color: bool,
) -> Vec<minder_core::AskAnswer> {
    let mut state = AskOverlayState::new(questions);
    redraw_ask_overlay(handles, &state, color);
    loop {
        let event = match events.next().await {
            None => return Vec::new(),
            Some(Err(_)) => continue,
            Some(Ok(e)) => e,
        };
        let Event::Key(key) = event else { continue };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }
        match state.handle_key(key) {
            AskOverlayOutcome::Continue => redraw_ask_overlay(handles, &state, color),
            AskOverlayOutcome::Finished(answers) => return answers,
        }
    }
}

fn redraw_ask_overlay(handles: &PinnedHandles, state: &AskOverlayState, color: bool) {
    let mut term = handles.terminal.lock().unwrap();
    let _ = term.draw(|frame| state.render(frame, frame.area(), color));
}

/// Publishes the box's contents into `handles.input` (so `sink::append_text`
/// can redraw it without a live `InputBoxState`), then draws both it and the
/// transcript pane.
///
/// Locks `terminal` before `transcript`, same order as `sink::append_text` -- keep it consistent or it deadlocks.
fn redraw(handles: &PinnedHandles, box_state: &InputBoxState, mode: InputMode, status_text: &str, color: bool) {
    *handles.input.lock().unwrap() = box_state.snapshot(mode);
    let pending = handles.pending.lock().unwrap().clone();

    let mut term = handles.terminal.lock().unwrap();
    let mut transcript = handles.transcript.lock().unwrap();
    let _ = term.draw(|frame| {
        let area = frame.area();
        let bottom = input_box::bottom_area(area, box_state.buffer(), &pending);
        transcript.render(frame, input_box::transcript_area(area, box_state.buffer(), &pending));
        box_state.render(frame, bottom, mode, status_text, &pending, color);
        if bottom.height > 0 {
            frame.set_cursor_position(box_state.cursor_screen_position(bottom, &pending));
        }
    });
}

/// Same wording as `print_turn_error` in `main.rs`, but routed through the
/// reporter (`on_notice`) instead of raw `println!`/`eprintln!`, so it
/// lands above the pinned box via the transcript instead of corrupting it.
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

/// Rustyline's `FileHistory` v2 format: a `#V2` header, then one `\`/`\n`-escaped entry per line.
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
