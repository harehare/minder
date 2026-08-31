use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub struct AskOption {
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct AskQuestion {
    pub header: String,
    pub question: String,
    pub options: Vec<AskOption>,
    pub multi_select: bool,
}

/// The selected option label(s), or a single free-text entry if the user picked "Other".
#[derive(Debug, Clone)]
pub struct AskAnswer {
    pub header: String,
    pub selected: Vec<String>,
}

pub struct AskRequest {
    pub questions: Vec<AskQuestion>,
    pub reply: oneshot::Sender<Vec<AskAnswer>>,
}

pub type AskReceiver = mpsc::UnboundedReceiver<AskRequest>;

/// Modeled on `Mailbox`'s thin `Clone` wrapper around a channel. Cloned into
/// every `ToolContext` (including subagents'), so there's exactly one true
/// owner of the receiver (the TUI REPL) and every other context just holds a
/// sender nobody's listening on.
#[derive(Clone)]
pub struct AskChannel(mpsc::UnboundedSender<AskRequest>);

impl AskChannel {
    pub fn channel() -> (Self, AskReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self(tx), rx)
    }

    /// The matching receiver is dropped immediately, so `ask` fails fast
    /// instead of hanging forever waiting for an answer nobody can give.
    pub fn unavailable() -> Self {
        let (tx, _rx) = mpsc::unbounded_channel();
        Self(tx)
    }

    /// `None` if nobody's listening, or the receiver dropped before
    /// replying -- callers should degrade gracefully, not treat it as fatal.
    pub async fn ask(&self, questions: Vec<AskQuestion>) -> Option<Vec<AskAnswer>> {
        let (reply, reply_rx) = oneshot::channel();
        self.0.send(AskRequest { questions, reply }).ok()?;
        reply_rx.await.ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question() -> AskQuestion {
        AskQuestion {
            header: "Approach".to_string(),
            question: "Which approach?".to_string(),
            options: vec![
                AskOption {
                    label: "A".to_string(),
                    description: String::new(),
                },
                AskOption {
                    label: "B".to_string(),
                    description: String::new(),
                },
            ],
            multi_select: false,
        }
    }

    #[tokio::test]
    async fn unavailable_channel_returns_none_immediately_instead_of_hanging() {
        let channel = AskChannel::unavailable();
        assert!(channel.ask(vec![question()]).await.is_none());
    }

    #[tokio::test]
    async fn connected_channel_round_trips_a_reply() {
        let (channel, mut rx) = AskChannel::channel();
        let listener = tokio::spawn(async move {
            let request = rx.recv().await.expect("a request was sent");
            assert_eq!(request.questions.len(), 1);
            request
                .reply
                .send(vec![AskAnswer {
                    header: "Approach".to_string(),
                    selected: vec!["A".to_string()],
                }])
                .unwrap();
        });

        let answers = channel.ask(vec![question()]).await.expect("a reply came back");
        assert_eq!(answers[0].selected, vec!["A".to_string()]);
        listener.await.unwrap();
    }

    #[tokio::test]
    async fn dropped_receiver_makes_a_pending_ask_resolve_to_none() {
        let (channel, rx) = AskChannel::channel();
        drop(rx);
        assert!(channel.ask(vec![question()]).await.is_none());
    }
}
