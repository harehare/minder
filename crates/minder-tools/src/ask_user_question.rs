use async_trait::async_trait;
use minder_core::{AskAnswer, AskOption, AskQuestion, Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;

#[derive(Deserialize)]
struct ArgOption {
    label: String,
    #[serde(default)]
    description: String,
}

#[derive(Deserialize)]
struct ArgQuestion {
    header: String,
    question: String,
    options: Vec<ArgOption>,
    #[serde(default)]
    #[serde(rename = "multiSelect")]
    multi_select: bool,
}

#[derive(Deserialize)]
struct Args {
    questions: Vec<ArgQuestion>,
}

/// Only wired to a live UI in the fullscreen TUI REPL (see `ToolContext::ask`);
/// everywhere else it degrades to listing the choices as text.
pub struct AskUserQuestionTool;

#[async_trait]
impl Tool for AskUserQuestionTool {
    fn name(&self) -> &str {
        "ask_user_question"
    }

    fn description(&self) -> &str {
        "Ask the user one or more multiple-choice questions and get back their picks -- use this \
         for genuine decision points (which of these approaches, which file, yes/no) instead of \
         asking in a plain reply and waiting for free-text. Not available in every context (e.g. \
         non-interactive runs); if it isn't, fall back to asking in your reply text."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "header": { "type": "string", "description": "Short label (a few words), e.g. \"Auth method\"" },
                            "question": { "type": "string" },
                            "options": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": { "type": "string" },
                                        "description": { "type": "string" }
                                    },
                                    "required": ["label"]
                                }
                            },
                            "multiSelect": { "type": "boolean", "description": "Allow picking more than one option" }
                        },
                        "required": ["header", "question", "options"]
                    }
                }
            },
            "required": ["questions"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => {
                return ToolExecOutcome {
                    content: format!("invalid arguments: {e}"),
                    is_error: true,
                    metadata: serde_json::Value::Null,
                };
            }
        };
        if let Err(e) = validate(&args.questions) {
            return ToolExecOutcome {
                content: e,
                is_error: true,
                metadata: serde_json::Value::Null,
            };
        }

        let questions: Vec<AskQuestion> = args.questions.into_iter().map(Into::into).collect();
        match ctx.ask.ask(questions.clone()).await {
            Some(answers) => ToolExecOutcome {
                content: format_answers(&answers),
                is_error: false,
                metadata: serde_json::Value::Null,
            },
            None => ToolExecOutcome {
                content: format_unavailable(&questions),
                is_error: true,
                metadata: serde_json::Value::Null,
            },
        }
    }
}

fn validate(questions: &[ArgQuestion]) -> Result<(), String> {
    if questions.is_empty() {
        return Err("questions must not be empty".to_string());
    }
    for q in questions {
        if !(2..=4).contains(&q.options.len()) {
            return Err(format!(
                "question \"{}\" must have 2-4 options, got {}",
                q.header,
                q.options.len()
            ));
        }
    }
    Ok(())
}

impl From<ArgOption> for AskOption {
    fn from(o: ArgOption) -> Self {
        AskOption {
            label: o.label,
            description: o.description,
        }
    }
}

impl From<ArgQuestion> for AskQuestion {
    fn from(q: ArgQuestion) -> Self {
        AskQuestion {
            header: q.header,
            question: q.question,
            options: q.options.into_iter().map(Into::into).collect(),
            multi_select: q.multi_select,
        }
    }
}

fn format_answers(answers: &[AskAnswer]) -> String {
    answers
        .iter()
        .map(|a| format!("{}: {}", a.header, a.selected.join(", ")))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_unavailable(questions: &[AskQuestion]) -> String {
    let mut out = String::from(
        "Interactive selection isn't available in this context. Ask the user directly in your \
         reply text instead and wait for their next message. The questions were:\n",
    );
    for q in questions {
        out.push_str(&format!("- {} ({})\n", q.question, q.header));
        for o in &q.options {
            if o.description.is_empty() {
                out.push_str(&format!("  * {}\n", o.label));
            } else {
                out.push_str(&format!("  * {} -- {}\n", o.label, o.description));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::AskChannel;

    fn ctx_with(ask: AskChannel) -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
            mailbox: None,
            ask,
        }
    }

    fn args_json() -> serde_json::Value {
        serde_json::json!({
            "questions": [{
                "header": "Approach",
                "question": "Which approach?",
                "options": [
                    {"label": "A", "description": "first"},
                    {"label": "B", "description": "second"}
                ]
            }]
        })
    }

    #[tokio::test]
    async fn answers_come_back_as_readable_text_when_a_ui_is_wired_up() {
        let (ask, mut rx) = AskChannel::channel();
        tokio::spawn(async move {
            let request = rx.recv().await.unwrap();
            request
                .reply
                .send(vec![AskAnswer {
                    header: "Approach".to_string(),
                    selected: vec!["A".to_string()],
                }])
                .unwrap();
        });

        let outcome = AskUserQuestionTool.execute(args_json(), &ctx_with(ask)).await;
        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "Approach: A");
    }

    #[tokio::test]
    async fn falls_back_to_listing_the_choices_as_text_with_no_ui() {
        let outcome = AskUserQuestionTool
            .execute(args_json(), &ctx_with(AskChannel::unavailable()))
            .await;
        assert!(outcome.is_error);
        assert!(outcome.content.contains("Which approach?"));
        assert!(outcome.content.contains("A -- first"));
    }

    #[tokio::test]
    async fn rejects_a_question_with_fewer_than_two_options() {
        let args = serde_json::json!({
            "questions": [{
                "header": "Approach",
                "question": "Which approach?",
                "options": [{"label": "A"}]
            }]
        });
        let outcome = AskUserQuestionTool
            .execute(args, &ctx_with(AskChannel::unavailable()))
            .await;
        assert!(outcome.is_error);
        assert!(outcome.content.contains("2-4 options"));
    }
}
