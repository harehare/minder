use async_trait::async_trait;
use minder_core::{Mailbox, MailboxMessage, Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;

/// Lets a subagent send a message to a named sibling running concurrently in
/// the same `agent` batch. Only offered when `ToolContext::mailbox` is set --
/// see `AgentTool::execute`.
pub struct SendMessageTool {
    pub mailbox: Mailbox,
    pub from: String,
}

#[derive(Deserialize)]
struct SendArgs {
    to: String,
    content: String,
}

#[async_trait]
impl Tool for SendMessageTool {
    fn name(&self) -> &str {
        "send_message"
    }

    fn description(&self) -> &str {
        "Sends a short message to another subagent running concurrently in this same batch. \
         Delivery is best-effort: if the recipient never calls `check_messages`, the message is \
         simply never read."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Name of the sibling subagent to message"},
                "content": {"type": "string"}
            },
            "required": ["to", "content"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let args: SendArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => {
                return ToolExecOutcome {
                    content: format!("invalid arguments: {e}"),
                    is_error: true,
                    metadata: serde_json::Value::Null,
                };
            }
        };
        self.mailbox.send(MailboxMessage {
            from: self.from.clone(),
            to: args.to.clone(),
            content: args.content,
        });
        ToolExecOutcome {
            content: format!("message sent to '{}'", args.to),
            is_error: false,
            metadata: serde_json::Value::Null,
        }
    }
}

/// Lets a subagent read messages sent to it by siblings via `SendMessageTool`.
pub struct CheckMessagesTool {
    pub mailbox: Mailbox,
    pub name: String,
}

#[async_trait]
impl Tool for CheckMessagesTool {
    fn name(&self) -> &str {
        "check_messages"
    }

    fn description(&self) -> &str {
        "Returns any messages sibling subagents in this batch have sent you since you last checked."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn execute(&self, _arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let messages = self.mailbox.take_for(&self.name);
        let content = if messages.is_empty() {
            "no messages".to_string()
        } else {
            messages
                .iter()
                .map(|m| format!("{}: {}", m.from, m.content))
                .collect::<Vec<_>>()
                .join("\n")
        };
        ToolExecOutcome {
            content,
            is_error: false,
            metadata: serde_json::Value::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::AskChannel;

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
            mailbox: None,
            ask: AskChannel::unavailable(),
        }
    }

    #[tokio::test]
    async fn a_sent_message_is_delivered_to_the_named_recipient() {
        let mailbox = Mailbox::new();
        let sender = SendMessageTool {
            mailbox: mailbox.clone(),
            from: "alice".to_string(),
        };
        let receiver = CheckMessagesTool {
            mailbox: mailbox.clone(),
            name: "bob".to_string(),
        };

        let outcome = sender
            .execute(serde_json::json!({"to": "bob", "content": "hi bob"}), &ctx())
            .await;
        assert!(!outcome.is_error);

        let received = receiver.execute(serde_json::json!({}), &ctx()).await;
        assert_eq!(received.content, "alice: hi bob");
    }

    #[tokio::test]
    async fn checking_with_no_pending_messages_says_so() {
        let receiver = CheckMessagesTool {
            mailbox: Mailbox::new(),
            name: "bob".to_string(),
        };
        let outcome = receiver.execute(serde_json::json!({}), &ctx()).await;
        assert_eq!(outcome.content, "no messages");
    }
}
