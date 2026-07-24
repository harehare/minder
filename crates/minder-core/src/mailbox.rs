use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// A single message sent through a `Mailbox`.
#[derive(Debug, Clone)]
pub struct MailboxMessage {
    pub from: String,
    pub to: String,
    pub content: String,
}

/// Best-effort inbox shared by subagents running concurrently in one `agent`
/// batch (see `AgentSession::run_turn`'s concurrent-tool-call block), so
/// siblings can coordinate via `send_message`/`check_messages` tool calls.
/// Targeted delivery only, no broadcast -- a message nobody ever checks for
/// is simply gone once the batch ends.
#[derive(Clone, Default)]
pub struct Mailbox(Arc<Mutex<VecDeque<MailboxMessage>>>);

impl Mailbox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn send(&self, msg: MailboxMessage) {
        self.0.lock().unwrap().push_back(msg);
    }

    /// Drains and returns every pending message addressed to `name`, in send order.
    pub fn take_for(&self, name: &str) -> Vec<MailboxMessage> {
        let mut inner = self.0.lock().unwrap();
        let mut mine = Vec::new();
        let mut rest = VecDeque::new();
        for msg in inner.drain(..) {
            if msg.to == name {
                mine.push(msg);
            } else {
                rest.push_back(msg);
            }
        }
        *inner = rest;
        mine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(from: &str, to: &str, content: &str) -> MailboxMessage {
        MailboxMessage {
            from: from.to_string(),
            to: to.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn take_for_returns_only_messages_addressed_to_that_name_in_send_order() {
        let mailbox = Mailbox::new();
        mailbox.send(msg("a", "b", "first"));
        mailbox.send(msg("a", "c", "not for b"));
        mailbox.send(msg("a", "b", "second"));

        let mine = mailbox.take_for("b");
        assert_eq!(mine.len(), 2);
        assert_eq!(mine[0].content, "first");
        assert_eq!(mine[1].content, "second");
    }

    #[test]
    fn take_for_drains_so_a_message_is_delivered_at_most_once() {
        let mailbox = Mailbox::new();
        mailbox.send(msg("a", "b", "only once"));

        assert_eq!(mailbox.take_for("b").len(), 1);
        assert_eq!(mailbox.take_for("b").len(), 0);
    }

    #[test]
    fn messages_for_other_recipients_are_left_untouched() {
        let mailbox = Mailbox::new();
        mailbox.send(msg("a", "c", "for c"));

        assert_eq!(mailbox.take_for("b").len(), 0);
        assert_eq!(mailbox.take_for("c").len(), 1);
    }
}
