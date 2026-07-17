use async_trait::async_trait;
use minder_core::{AgentSession, HookPort, LlmProvider, Reporter, Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A named, isolated `AgentSession` the main loop can delegate a task to via
/// `AgentTool`. Defined like a `Skill`: a directory with a frontmatter file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subagent {
    pub name: String,
    pub description: String,
    /// Allow-list of tool names, by name. `None` means every parent tool
    /// except `agent` itself (see `AgentTool::new`).
    pub tools: Option<Vec<String>>,
    pub system_prompt: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SubagentLoadError {
    #[error("failed to read agents directory {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("agent file {0} is missing '---' delimited frontmatter")]
    MissingFrontmatter(PathBuf),
    #[error("agent file {0} frontmatter is missing required field '{1}'")]
    MissingField(PathBuf, &'static str),
    #[error("duplicate agent name '{name}' in {first} and {second} -- agent names must be unique")]
    DuplicateName {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },
}

/// Subagents available with zero project config, so `agent` always works.
/// A project can override any of these via `.agent/agents/<name>/AGENT.md`
/// with a matching name.
pub fn builtin_subagents() -> Vec<Subagent> {
    vec![Subagent {
        name: "general-purpose".to_string(),
        description: "General-purpose agent for open-ended research, multi-step tasks, or any \
                       self-contained piece of work you'd rather hand off than do inline. Has \
                       access to every tool the parent has."
            .to_string(),
        tools: None,
        system_prompt: "You are a focused subagent completing a single delegated task. Use the \
                         available tools to accomplish it directly, then reply with a concise, \
                         complete answer -- your caller only ever sees this final reply, none of \
                         your intermediate tool calls."
            .to_string(),
    }]
}

/// Discovers subagents from `agent_dir/agents/*/AGENT.md`, one directory per
/// subagent (mirrors `discover_skills`). Returns an empty vec if the agents
/// directory doesn't exist -- subagents are fully optional, like skills.
pub fn discover_subagents(agent_dir: &Path) -> Result<Vec<Subagent>, SubagentLoadError> {
    let agents_dir = agent_dir.join("agents");
    if !agents_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&agents_dir)
        .map_err(|e| SubagentLoadError::Io(agents_dir.clone(), e))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir())
        .collect();
    entries.sort();

    let mut subagents: Vec<Subagent> = Vec::new();
    let mut sources: Vec<PathBuf> = Vec::new(); // parallel to `subagents`, for error messages
    for dir in entries {
        let agent_md = dir.join("AGENT.md");
        if !agent_md.is_file() {
            continue;
        }
        let raw = std::fs::read_to_string(&agent_md).map_err(|e| SubagentLoadError::Io(agent_md.clone(), e))?;
        let subagent = parse_subagent(&agent_md, &raw)?;

        if let Some(idx) = subagents.iter().position(|s| s.name == subagent.name) {
            return Err(SubagentLoadError::DuplicateName {
                name: subagent.name,
                first: sources[idx].clone(),
                second: agent_md,
            });
        }
        sources.push(agent_md);
        subagents.push(subagent);
    }

    Ok(subagents)
}

fn parse_subagent(path: &Path, raw: &str) -> Result<Subagent, SubagentLoadError> {
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw); // tolerate a UTF-8 BOM
    let rest = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))
        .ok_or_else(|| SubagentLoadError::MissingFrontmatter(path.to_path_buf()))?;
    let end = rest
        .find("\n---")
        .ok_or_else(|| SubagentLoadError::MissingFrontmatter(path.to_path_buf()))?;
    let frontmatter = &rest[..end];
    let body = rest[end..]
        .trim_start_matches("\n---")
        .trim_start_matches("\r\n---")
        .trim_start_matches(['\r', '\n']);

    let mut name = None;
    let mut description = None;
    let mut tools = None;
    for line in frontmatter.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        match key.trim() {
            "name" => name = Some(value.to_string()),
            "description" => description = Some(value.to_string()),
            "tools" if !value.is_empty() => {
                tools = Some(value.split(',').map(|t| t.trim().to_string()).collect::<Vec<_>>())
            }
            _ => {}
        }
    }

    let name = name.ok_or_else(|| SubagentLoadError::MissingField(path.to_path_buf(), "name"))?;
    let description = description.ok_or_else(|| SubagentLoadError::MissingField(path.to_path_buf(), "description"))?;

    Ok(Subagent {
        name,
        description,
        tools,
        system_prompt: body.trim().to_string(),
    })
}

/// Exposes discovered subagents as a single `agent` tool, mirroring
/// `SkillTool`: calling it with `{name, task}` runs that subagent's own
/// `AgentSession` to completion in-process and returns its final answer.
/// Provider and base tools are shared (`Arc`) with the parent rather than
/// rebuilt per call.
pub struct AgentTool {
    subagents: Vec<Subagent>,
    provider: Arc<dyn LlmProvider>,
    /// The parent's tools minus `agent` itself, so subagents can't recurse.
    base_tools: Vec<Arc<dyn Tool>>,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
    reporter: Arc<dyn Reporter>,
    description: String,
}

impl AgentTool {
    pub fn new(
        subagents: Vec<Subagent>,
        provider: Arc<dyn LlmProvider>,
        base_tools: Vec<Arc<dyn Tool>>,
        hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
        reporter: Arc<dyn Reporter>,
    ) -> Self {
        let list = subagents
            .iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");
        let description = format!(
            "Delegates a task to a named subagent, running it to completion in an isolated \
             session and returning its final answer. Use this to hand off a well-scoped piece \
             of work (e.g. a focused review or search) instead of doing it inline, especially \
             when it would otherwise clutter this conversation with intermediate tool calls.\n\n\
             Available subagents:\n{list}"
        );
        Self {
            subagents,
            provider,
            base_tools: base_tools.into_iter().filter(|t| t.name() != "agent").collect(),
            hooks,
            reporter,
            description,
        }
    }
}

#[derive(Deserialize)]
struct Args {
    name: String,
    task: String,
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let names: Vec<&str> = self.subagents.iter().map(|s| s.name.as_str()).collect();
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the subagent to delegate to",
                    "enum": names
                },
                "task": {
                    "type": "string",
                    "description": "The task to hand off, in enough detail for the subagent to act without further clarification (it starts with no conversation history)"
                }
            },
            "required": ["name", "task"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: &ToolContext) -> ToolExecOutcome {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return error(format!("invalid arguments: {e}")),
        };

        let Some(subagent) = self.subagents.iter().find(|s| s.name == args.name) else {
            return error(format!(
                "unknown subagent '{}' -- available: {}",
                args.name,
                self.subagents
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        };

        let tools: Vec<Arc<dyn Tool>> = match &subagent.tools {
            Some(allowed) => self
                .base_tools
                .iter()
                .filter(|t| allowed.iter().any(|name| name == t.name()))
                .cloned()
                .collect(),
            None => self.base_tools.clone(),
        };

        let child_ctx = ToolContext {
            working_dir: ctx.working_dir.clone(),
            session_id: format!("{}:agent:{}", ctx.session_id, subagent.name),
            cancel: ctx.cancel.clone(),
        };

        let mut session = AgentSession::new(
            self.provider.clone(),
            tools,
            self.hooks.clone(),
            subagent.system_prompt.clone(),
            child_ctx,
        )
        .with_reporter(self.reporter.clone());

        match session.run_turn(&args.task).await {
            Ok(message) => ToolExecOutcome {
                content: message.text(),
                is_error: false,
                metadata: serde_json::Value::Null,
            },
            Err(e) => error(format!("subagent '{}' failed: {e}", subagent.name)),
        }
    }
}

fn error(message: String) -> ToolExecOutcome {
    ToolExecOutcome {
        content: message,
        is_error: true,
        metadata: serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minder_core::{
        ContentBlock, Message, ProviderError, ProviderResponse, Role, StopReason, ToolCall, ToolSpec, Usage,
    };
    use std::sync::Mutex as StdMutex;

    fn scratch_dir() -> PathBuf {
        std::env::temp_dir().join(format!("minder-subagent-test-{}", uuid::Uuid::new_v4()))
    }

    fn write_agent(agent_dir: &Path, dir_name: &str, contents: &str) {
        let dir = agent_dir.join("agents").join(dir_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("AGENT.md"), contents).unwrap();
    }

    #[test]
    fn discovers_no_subagents_when_agents_dir_is_absent() {
        let agent_dir = scratch_dir();
        let subagents = discover_subagents(&agent_dir).unwrap();
        assert!(subagents.is_empty());
    }

    #[test]
    fn builtin_subagents_includes_general_purpose_with_no_tool_restriction() {
        let builtins = builtin_subagents();
        let general_purpose = builtins.iter().find(|s| s.name == "general-purpose");
        assert!(general_purpose.is_some());
        assert_eq!(general_purpose.unwrap().tools, None);
    }

    #[test]
    fn discovers_and_parses_a_subagent() {
        let agent_dir = scratch_dir();
        write_agent(
            &agent_dir,
            "reviewer",
            "---\nname: reviewer\ndescription: Reviews a diff for bugs\ntools: read_file, grep\n---\n# Reviewer\n\nLook for bugs.\n",
        );

        let subagents = discover_subagents(&agent_dir).unwrap();
        assert_eq!(subagents.len(), 1);
        assert_eq!(subagents[0].name, "reviewer");
        assert_eq!(subagents[0].description, "Reviews a diff for bugs");
        assert_eq!(
            subagents[0].tools,
            Some(vec!["read_file".to_string(), "grep".to_string()])
        );
        assert_eq!(subagents[0].system_prompt, "# Reviewer\n\nLook for bugs.");
    }

    #[test]
    fn tools_field_is_optional() {
        let agent_dir = scratch_dir();
        write_agent(
            &agent_dir,
            "generalist",
            "---\nname: generalist\ndescription: Does anything\n---\nbody\n",
        );
        let subagents = discover_subagents(&agent_dir).unwrap();
        assert_eq!(subagents[0].tools, None);
    }

    #[test]
    fn duplicate_agent_names_are_an_error() {
        let agent_dir = scratch_dir();
        write_agent(&agent_dir, "a", "---\nname: dup\ndescription: first\n---\nbody\n");
        write_agent(&agent_dir, "b", "---\nname: dup\ndescription: second\n---\nbody\n");
        let err = discover_subagents(&agent_dir).unwrap_err();
        assert!(matches!(err, SubagentLoadError::DuplicateName { name, .. } if name == "dup"));
    }

    struct ScriptedProvider(StdMutex<std::collections::VecDeque<ProviderResponse>>);

    #[async_trait]
    impl LlmProvider for ScriptedProvider {
        fn id(&self) -> &'static str {
            "scripted"
        }
        fn model(&self) -> &str {
            "scripted-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(self.0.lock().unwrap().pop_front().expect("script exhausted"))
        }
    }

    fn text_response(text: &str) -> ProviderResponse {
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(text.to_string())],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    fn tool_use_response(call_id: &str, tool: &str) -> ProviderResponse {
        ProviderResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse(ToolCall {
                    id: call_id.to_string(),
                    name: tool.to_string(),
                    arguments: serde_json::json!({}),
                })],
                metadata: serde_json::Value::Null,
            },
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        }
    }

    struct RecursionProbeTool(Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait]
    impl Tool for RecursionProbeTool {
        fn name(&self) -> &str {
            "agent"
        }
        fn description(&self) -> &str {
            "should never be reachable from inside a subagent"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ToolExecOutcome {
                content: "should not run".to_string(),
                is_error: false,
                metadata: serde_json::Value::Null,
            }
        }
    }

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: std::env::temp_dir(),
            session_id: "test".to_string(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn delegates_to_a_subagent_and_returns_its_final_text() {
        let provider: Arc<dyn LlmProvider> = Arc::new(ScriptedProvider(StdMutex::new(
            vec![text_response("done: reviewed and found nothing")].into(),
        )));
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let base_tools: Vec<Arc<dyn Tool>> = vec![Arc::new(RecursionProbeTool(call_count.clone()))];

        let tool = AgentTool::new(
            vec![Subagent {
                name: "reviewer".to_string(),
                description: "Reviews code".to_string(),
                tools: None,
                system_prompt: "You review code.".to_string(),
            }],
            provider,
            base_tools,
            None,
            Arc::new(minder_core::NoopReporter),
        );

        let outcome = tool
            .execute(serde_json::json!({"name": "reviewer", "task": "review this"}), &ctx())
            .await;

        assert!(!outcome.is_error);
        assert_eq!(outcome.content, "done: reviewed and found nothing");
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the `agent` tool must never be exposed to a subagent's own session"
        );
    }

    #[tokio::test]
    async fn subagent_cannot_call_the_agent_tool_itself() {
        // Child session's provider tries calling "agent"; the unknown-tool
        // result now comes back to the model as a normal (error) tool result
        // instead of aborting the turn, so it gets a chance to recover.
        let provider: Arc<dyn LlmProvider> = Arc::new(ScriptedProvider(StdMutex::new(
            vec![
                tool_use_response("call_1", "agent"),
                text_response("gave up on 'agent'"),
            ]
            .into(),
        )));
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let base_tools: Vec<Arc<dyn Tool>> = vec![Arc::new(RecursionProbeTool(call_count.clone()))];

        let tool = AgentTool::new(
            vec![Subagent {
                name: "reviewer".to_string(),
                description: "Reviews code".to_string(),
                tools: None,
                system_prompt: "You review code.".to_string(),
            }],
            provider,
            base_tools,
            None,
            Arc::new(minder_core::NoopReporter),
        );

        let outcome = tool
            .execute(serde_json::json!({"name": "reviewer", "task": "review this"}), &ctx())
            .await;

        assert!(!outcome.is_error, "expected the subagent to recover, got: {outcome:?}");
        assert_eq!(outcome.content, "gave up on 'agent'");
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the `agent` tool must never be reachable from inside a subagent"
        );
    }

    #[tokio::test]
    async fn unknown_subagent_name_is_an_error() {
        let tool = AgentTool::new(
            vec![],
            Arc::new(ScriptedProvider(StdMutex::new(vec![].into()))),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
        );
        let outcome = tool
            .execute(serde_json::json!({"name": "nope", "task": "x"}), &ctx())
            .await;
        assert!(outcome.is_error);
    }
}
