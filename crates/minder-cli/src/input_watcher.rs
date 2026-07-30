use std::io::{IsTerminal, Write};

use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Printed at the start of a line once the user starts typing over a
/// running turn, so their in-progress input reads as a distinct aside from
/// whatever the agent is streaming above it.
const STEERING_PROMPT: &str = "» ";

/// Watches stdin for the duration of one turn so it can be interrupted or
/// steered without waiting for it to finish -- see `run_turn_interruptible`.
///
/// On a real terminal, puts stdin into raw mode and treats Esc or Ctrl-C as
/// a cancel request (both fire `cancel` and a `next_cancel` event), while
/// any other typed line ending in Enter is forwarded to a `steering_tx` for
/// `AgentSession` to splice into the transcript (see
/// `AgentSession::enable_steering`). Raw mode disables the terminal's own
/// line editing and local echo, so keystrokes are echoed back here by hand.
///
/// Off a real TTY (piped input, `--output json`, `loop` mode) there's
/// nothing to type into, so this just falls back to listening for a real
/// SIGINT, same as before this existed.
pub struct InputWatcher {
    handle: tokio::task::JoinHandle<()>,
    cancel_rx: mpsc::UnboundedReceiver<()>,
    raw_mode_enabled: bool,
    shutdown: CancellationToken,
}

/// Checked once at REPL startup so it can warn instead of silently falling
/// back to signal-only Ctrl-C every turn. Round-trips raw mode rather than
/// just checking `is_terminal()` -- a real TTY can still fail to enable it.
pub fn supports_key_watching() -> bool {
    std::io::stdin().is_terminal() && {
        let enabled = crossterm::terminal::enable_raw_mode().is_ok();
        if enabled {
            let _ = crossterm::terminal::disable_raw_mode();
        }
        enabled
    }
}

impl InputWatcher {
    pub fn spawn(cancel: CancellationToken, steering_tx: mpsc::UnboundedSender<String>) -> Self {
        let raw_mode_enabled = std::io::stdin().is_terminal() && crossterm::terminal::enable_raw_mode().is_ok();
        let (cancel_tx, cancel_rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();

        let handle = tokio::spawn({
            let shutdown = shutdown.clone();
            async move {
                if raw_mode_enabled {
                    watch_keys(cancel, steering_tx, cancel_tx, shutdown).await;
                } else {
                    watch_sigint_only(cancel, cancel_tx, shutdown).await;
                }
            }
        });

        Self {
            handle,
            cancel_rx,
            raw_mode_enabled,
            shutdown,
        }
    }

    /// Resolves once per Esc/Ctrl-C (key or real SIGINT) -- `cancel` is
    /// already reflected by the time this returns. `None` once the watcher
    /// itself has stopped.
    pub async fn next_cancel(&mut self) -> Option<()> {
        self.cancel_rx.recv().await
    }

    /// Stops watching and restores the terminal (a no-op off a real TTY).
    /// Signals the task to return on its own next `select!` turn rather than
    /// `abort()`-ing it -- this runs at the end of *every* turn, including
    /// the vast majority that were never interrupted, while the task is
    /// almost always parked inside `EventStream::next()`. Force-aborting a
    /// task mid-poll there left crossterm's raw-fd reader in a state that
    /// occasionally broke the *next* turn's key watching or left the
    /// terminal in raw mode -- exactly the "Esc stopped canceling" / "the
    /// prompt didn't show" reports this was written to fix. Awaiting the
    /// task after a cooperative exit guarantees raw mode is back off before
    /// this returns -- the REPL's next `readline()` call needs a normal,
    /// cooked terminal.
    pub async fn stop(self) {
        self.shutdown.cancel();
        let _ = self.handle.await;
        if self.raw_mode_enabled {
            let _ = crossterm::terminal::disable_raw_mode();
        }
    }
}

async fn watch_keys(
    cancel: CancellationToken,
    steering_tx: mpsc::UnboundedSender<String>,
    cancel_tx: mpsc::UnboundedSender<()>,
    shutdown: CancellationToken,
) {
    let mut events = EventStream::new();
    let mut buffer = String::new();

    loop {
        let event = tokio::select! {
            biased;
            _ = shutdown.cancelled() => return,
            event = events.next() => event,
        };
        // `None` means the stream itself ended (e.g. stdin closed) -- stop
        // rather than spin. A transient `Err` from a single read is not the
        // same thing and used to fall through this same arm, silently
        // ending the whole watcher (and Esc/steering with it) for the rest
        // of the turn the first time crossterm hiccuped -- keep polling
        // instead.
        let Some(event) = event else { return };
        let Ok(event) = event else { continue };
        let Event::Key(key) = event else { continue };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue; // ignore key-release events crossterm reports on some platforms
        }

        let is_ctrl_c = key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        if key.code == KeyCode::Esc || is_ctrl_c {
            clear_echoed_buffer(&mut buffer);
            cancel.cancel();
            if cancel_tx.send(()).is_err() {
                return; // the REPL stopped listening -- nothing left to watch for
            }
            continue;
        }

        match key.code {
            KeyCode::Enter if !buffer.is_empty() => {
                echo("\r\n");
                if steering_tx.send(std::mem::take(&mut buffer)).is_err() {
                    return;
                }
            }
            KeyCode::Backspace => {
                if buffer.pop().is_some() {
                    echo("\u{8} \u{8}");
                }
            }
            KeyCode::Char(c) => {
                if buffer.is_empty() {
                    echo("\r\n");
                    echo(STEERING_PROMPT);
                }
                buffer.push(c);
                let mut tmp = [0u8; 4];
                echo(c.encode_utf8(&mut tmp));
            }
            _ => {}
        }
    }
}

async fn watch_sigint_only(
    cancel: CancellationToken,
    cancel_tx: mpsc::UnboundedSender<()>,
    shutdown: CancellationToken,
) {
    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => return,
            result = tokio::signal::ctrl_c() => {
                if result.is_err() {
                    return;
                }
            }
        }
        cancel.cancel();
        if cancel_tx.send(()).is_err() {
            return;
        }
    }
}

/// Erases whatever's been typed so far (not yet submitted) so a cancel
/// doesn't leave a half-finished line behind, then moves to a fresh line.
fn clear_echoed_buffer(buffer: &mut String) {
    if buffer.is_empty() {
        return;
    }
    for _ in buffer.chars() {
        echo("\u{8} \u{8}");
    }
    echo("\r\n");
    buffer.clear();
}

fn echo(s: &str) {
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(s.as_bytes());
    let _ = stdout.flush();
}
