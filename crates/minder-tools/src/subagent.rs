use async_trait::async_trait;
use minder_core::{AgentError, AgentSession, HookPort, LlmProvider, Reporter, Tool, ToolContext, ToolExecOutcome};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::agent_registry::AgentRegistry;
use crate::mailbox_tools::{CheckMessagesTool, SendMessageTool};

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
    /// Static per-subagent model/provider override, resolved through
    /// `AgentTool`'s `ProviderFactory` -- `None` means use the parent's
    /// default provider. A per-call `model` tool argument wins over this.
    pub model: Option<String>,
    pub provider: Option<String>,
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
        model: None,
        provider: None,
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
    let mut model = None;
    let mut provider = None;
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
            "model" if !value.is_empty() => model = Some(value.to_string()),
            "provider" if !value.is_empty() => provider = Some(value.to_string()),
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
        model,
        provider,
    })
}

/// Builds a provider for a given (provider name, model) pair, e.g. wrapping
/// `provider_select::build_provider` -- lets `AgentTool` resolve a model
/// override without depending on `minder-providers` itself.
pub type ProviderFactory = dyn Fn(&str, &str) -> Result<Arc<dyn LlmProvider>, String> + Send + Sync;

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
    /// Resolves a model override (per-call `model` argument, or a
    /// `Subagent`'s static `model`/`provider`) into a real provider. `None`
    /// means overrides aren't supported in this context -- the `model`
    /// argument is left out of the schema entirely (see `parameters_schema`).
    provider_factory: Option<Arc<ProviderFactory>>,
    /// Tracks runs started with `background: true` so `list_agents`/
    /// `agent_output`/`agent_stop` (registered alongside this tool, see
    /// `main.rs::build_session`) can inspect or cancel them later.
    registry: Arc<AgentRegistry>,
}

impl AgentTool {
    pub fn new(
        subagents: Vec<Subagent>,
        provider: Arc<dyn LlmProvider>,
        base_tools: Vec<Arc<dyn Tool>>,
        hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
        reporter: Arc<dyn Reporter>,
        provider_factory: Option<Arc<ProviderFactory>>,
        registry: Arc<AgentRegistry>,
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
             when it would otherwise clutter this conversation with intermediate tool calls. \
             Call it more than once in the same turn to run subagents concurrently -- they can \
             then coordinate with each other via send_message/check_messages. Pass \
             `background: true` to start it and return immediately instead of waiting -- check \
             on it later with `list_agents`/`agent_output`, or cancel it with `agent_stop`.\n\n\
             Available subagents:\n{list}"
        );
        Self {
            subagents,
            provider,
            base_tools: base_tools.into_iter().filter(|t| t.name() != "agent").collect(),
            hooks,
            reporter,
            description,
            provider_factory,
            registry,
        }
    }
}

#[derive(Deserialize)]
struct Args {
    name: String,
    task: String,
    model: Option<String>,
    #[serde(default)]
    background: bool,
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
        let mut properties = serde_json::json!({
            "name": {
                "type": "string",
                "description": "Name of the subagent to delegate to",
                "enum": names
            },
            "task": {
                "type": "string",
                "description": "The task to hand off, in enough detail for the subagent to act without further clarification (it starts with no conversation history)"
            }
        });
        if self.provider_factory.is_some() {
            properties["model"] = serde_json::json!({
                "type": "string",
                "description": "Override the model used for this call, same provider as the \
                                 default -- e.g. a smaller/faster model for a simple task, a \
                                 stronger one for a complex one. Omit to use the default."
            });
        }
        properties["background"] = serde_json::json!({
            "type": "boolean",
            "description": "Run this subagent in the background instead of waiting for its \
                             final answer -- returns an id immediately. Check on it with \
                             `list_agents`/`agent_output`, or cancel it with `agent_stop`. \
                             Default false (wait for the result)."
        });
        serde_json::json!({
            "type": "object",
            "properties": properties,
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

        // Precedence: per-call `model` argument > the subagent's own static
        // `model`/`provider` default > the parent's provider (no override,
        // no factory call at all).
        let provider = match (&args.model, &subagent.model, &subagent.provider) {
            (None, None, None) => self.provider.clone(),
            (model_arg, subagent_model, subagent_provider) => {
                let Some(factory) = &self.provider_factory else {
                    return error(format!(
                        "subagent '{}' requested a model override but none is configured in this context",
                        subagent.name
                    ));
                };
                let provider_name = subagent_provider.as_deref().unwrap_or(self.provider.id());
                let model_name = model_arg
                    .as_deref()
                    .or(subagent_model.as_deref())
                    .unwrap_or(self.provider.model());
                match factory(provider_name, model_name) {
                    Ok(p) => p,
                    Err(e) => {
                        return error(format!(
                            "subagent '{}' model override '{model_name}' failed: {e}",
                            subagent.name
                        ));
                    }
                }
            }
        };

        let mut tools: Vec<Arc<dyn Tool>> = match &subagent.tools {
            Some(allowed) => self
                .base_tools
                .iter()
                .filter(|t| allowed.iter().any(|name| name == t.name()))
                .cloned()
                .collect(),
            None => self.base_tools.clone(),
        };

        // Only present when this call is running alongside siblings in one
        // concurrent `agent` batch (see `AgentSession::run_turn`) -- gives
        // the subagent a way to coordinate with them.
        let mut system_prompt = subagent.system_prompt.clone();
        if let Some(mailbox) = &ctx.mailbox {
            tools.push(Arc::new(SendMessageTool {
                mailbox: mailbox.clone(),
                from: subagent.name.clone(),
            }));
            tools.push(Arc::new(CheckMessagesTool {
                mailbox: mailbox.clone(),
                name: subagent.name.clone(),
            }));
            system_prompt.push_str(
                "\n\nYou're running alongside other subagents in this batch. Use `send_message`/\
                 `check_messages` to coordinate if useful -- delivery is best-effort.",
            );
        }

        if !args.background {
            let child_ctx = ToolContext {
                working_dir: ctx.working_dir.clone(),
                session_id: format!("{}:agent:{}", ctx.session_id, subagent.name),
                cancel: ctx.cancel.clone(),
                mailbox: None,
            };
            return run_subagent_to_completion(
                subagent.name.clone(),
                args.task.clone(),
                provider,
                tools,
                self.hooks.clone(),
                system_prompt,
                self.reporter.clone(),
                child_ctx,
            )
            .await;
        }

        // Child of the caller's own token: an interrupt of the whole turn
        // takes this background run down with it, but `agent_stop` cancelling
        // just this run doesn't touch its siblings (see `CancellationToken`'s
        // parent/child semantics).
        let cancel = ctx.cancel.child_token();
        let id = self.registry.start(&subagent.name, &args.task, cancel.clone());
        let child_ctx = ToolContext {
            working_dir: ctx.working_dir.clone(),
            session_id: format!("{}:agent:{}", ctx.session_id, subagent.name),
            cancel,
            mailbox: None,
        };

        let registry = self.registry.clone();
        let reporter = self.reporter.clone();
        let hooks = self.hooks.clone();
        let subagent_name = subagent.name.clone();
        let task = args.task.clone();
        let run_id = id.clone();
        tokio::spawn(async move {
            let outcome = run_subagent_to_completion(
                subagent_name,
                task,
                provider,
                tools,
                hooks,
                system_prompt,
                reporter,
                child_ctx,
            )
            .await;
            registry.finish(&run_id, outcome.content, outcome.is_error);
        });

        ToolExecOutcome {
            content: format!(
                "Started subagent '{}' in the background as {id}. Use `list_agents` to check \
                 its status, `agent_output` to fetch its result, or `agent_stop` to cancel it.",
                subagent.name
            ),
            is_error: false,
            metadata: serde_json::json!({ "id": id, "background": true }),
        }
    }
}

/// Runs one subagent turn to completion, retrying a transient provider error
/// up to `MAX_SUBAGENT_RETRIES` times with a fresh session each attempt (not
/// a resume of the failed one). Shared by `AgentTool::execute`'s foreground
/// and background (`tokio::spawn`-ed) paths, so both get identical retry
/// behavior.
#[allow(clippy::too_many_arguments)]
async fn run_subagent_to_completion(
    subagent_name: String,
    task: String,
    provider: Arc<dyn LlmProvider>,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Option<Arc<tokio::sync::Mutex<Box<dyn HookPort>>>>,
    system_prompt: String,
    reporter: Arc<dyn Reporter>,
    child_ctx: ToolContext,
) -> ToolExecOutcome {
    let mut attempt = 0u32;
    loop {
        let mut session = AgentSession::new(
            provider.clone(),
            tools.clone(),
            hooks.clone(),
            system_prompt.clone(),
            child_ctx.clone(),
        )
        .with_reporter(reporter.clone());

        match session.run_turn(&task).await {
            Ok(message) => {
                return ToolExecOutcome {
                    content: message.text(),
                    is_error: false,
                    metadata: serde_json::Value::Null,
                };
            }
            // Only provider errors are retried -- a hook block or interrupt is never one to retry.
            Err(e @ AgentError::Provider(_)) if attempt < MAX_SUBAGENT_RETRIES => {
                attempt += 1;
                reporter
                    .on_retry(
                        attempt as usize,
                        MAX_SUBAGENT_RETRIES as usize,
                        SUBAGENT_RETRY_DELAY,
                        &format!("subagent '{subagent_name}': {e}"),
                    )
                    .await;
                tokio::time::sleep(SUBAGENT_RETRY_DELAY).await;
            }
            Err(e) => {
                return error(format!(
                    "subagent '{subagent_name}' failed after {} attempt(s): {e}",
                    attempt + 1
                ));
            }
        }
    }
}

/// A failed subagent gets this many retries (fresh session each time, not a
/// resume of the failed one) before giving up -- see `AgentTool::execute`.
const MAX_SUBAGENT_RETRIES: u32 = 2;
const SUBAGENT_RETRY_DELAY: Duration = Duration::from_secs(2);

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
            mailbox: None,
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
            vec![reviewer_subagent()],
            provider,
            base_tools,
            None,
            Arc::new(minder_core::NoopReporter),
            None,
            Arc::new(AgentRegistry::new()),
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
            vec![reviewer_subagent()],
            provider,
            base_tools,
            None,
            Arc::new(minder_core::NoopReporter),
            None,
            Arc::new(AgentRegistry::new()),
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
            None,
            Arc::new(AgentRegistry::new()),
        );
        let outcome = tool
            .execute(serde_json::json!({"name": "nope", "task": "x"}), &ctx())
            .await;
        assert!(outcome.is_error);
    }

    struct FlakyProvider {
        calls: StdMutex<usize>,
        fail_times: usize,
    }

    #[async_trait]
    impl LlmProvider for FlakyProvider {
        fn id(&self) -> &'static str {
            "flaky"
        }
        fn model(&self) -> &str {
            "flaky-model"
        }
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
            _system_prompt: Option<&str>,
        ) -> Result<ProviderResponse, ProviderError> {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            if *calls <= self.fail_times {
                // status 400 is non-transient, so session.rs won't retry it internally --
                // each call here is exactly one subagent-level attempt.
                Err(ProviderError::Api {
                    status: 400,
                    body: "flaky".to_string(),
                })
            } else {
                Ok(text_response("recovered"))
            }
        }
    }

    fn reviewer_subagent() -> Subagent {
        Subagent {
            name: "reviewer".to_string(),
            description: "Reviews code".to_string(),
            tools: None,
            system_prompt: "You review code.".to_string(),
            model: None,
            provider: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn subagent_recovers_after_failures_below_the_retry_cap() {
        let provider = Arc::new(FlakyProvider {
            calls: StdMutex::new(0),
            fail_times: MAX_SUBAGENT_RETRIES as usize,
        });
        let tool = AgentTool::new(
            vec![reviewer_subagent()],
            provider,
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            None,
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(serde_json::json!({"name": "reviewer", "task": "review this"}), &ctx())
            .await;

        assert!(!outcome.is_error, "expected recovery, got: {outcome:?}");
        assert_eq!(outcome.content, "recovered");
    }

    #[tokio::test(start_paused = true)]
    async fn subagent_gives_up_after_exhausting_retries_and_reports_attempt_count() {
        let provider = Arc::new(FlakyProvider {
            calls: StdMutex::new(0),
            fail_times: MAX_SUBAGENT_RETRIES as usize + 1,
        });
        let tool = AgentTool::new(
            vec![reviewer_subagent()],
            provider,
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            None,
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(serde_json::json!({"name": "reviewer", "task": "review this"}), &ctx())
            .await;

        assert!(outcome.is_error);
        assert!(
            outcome
                .content
                .contains(&format!("after {} attempt(s)", MAX_SUBAGENT_RETRIES + 1)),
            "expected attempt count in: {}",
            outcome.content
        );
    }

    struct BlockingHooks;

    #[async_trait]
    impl HookPort for BlockingHooks {
        async fn before_agent_start(&mut self, _system_prompt: &str) -> minder_core::HookDecision<String> {
            minder_core::HookDecision::Block("policy says no".to_string())
        }
        async fn on_context(&mut self, messages: &[Message]) -> minder_core::HookDecision<Vec<Message>> {
            minder_core::HookDecision::Allow(messages.to_vec())
        }
        async fn on_tool_call(&mut self, call: &ToolCall) -> minder_core::ToolCallDecision {
            minder_core::ToolCallDecision::Allow(call.clone())
        }
        async fn on_tool_result(&mut self, result: &minder_core::ToolResultInfo) -> minder_core::HookDecision<String> {
            minder_core::HookDecision::Allow(result.content.clone())
        }
        async fn before_compact(&mut self, _messages: &[Message]) -> minder_core::HookDecision<()> {
            minder_core::HookDecision::Allow(())
        }
    }

    #[tokio::test]
    async fn a_hook_block_is_never_retried() {
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        struct CountingProvider(Arc<std::sync::atomic::AtomicUsize>);
        #[async_trait]
        impl LlmProvider for CountingProvider {
            fn id(&self) -> &'static str {
                "counting"
            }
            fn model(&self) -> &str {
                "counting-model"
            }
            async fn complete(
                &self,
                _messages: &[Message],
                _tools: &[ToolSpec],
                _system_prompt: Option<&str>,
            ) -> Result<ProviderResponse, ProviderError> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(text_response("should not be reached"))
            }
        }
        let hooks: Box<dyn HookPort> = Box::new(BlockingHooks);
        let tool = AgentTool::new(
            vec![reviewer_subagent()],
            Arc::new(CountingProvider(call_count.clone())),
            vec![],
            Some(Arc::new(tokio::sync::Mutex::new(hooks))),
            Arc::new(minder_core::NoopReporter),
            None,
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(serde_json::json!({"name": "reviewer", "task": "review this"}), &ctx())
            .await;

        assert!(outcome.is_error);
        assert!(
            outcome.content.contains("after 1 attempt(s)"),
            "got: {}",
            outcome.content
        );
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a hook block is a deterministic policy decision -- retrying it is pointless"
        );
    }

    fn text_response_provider(text: &'static str) -> Arc<dyn LlmProvider> {
        Arc::new(ScriptedProvider(StdMutex::new(vec![text_response(text)].into())))
    }

    #[tokio::test]
    async fn a_per_call_model_argument_routes_to_the_override_provider() {
        let override_provider = text_response_provider("used override");
        let factory_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = factory_calls.clone();
        let factory: Arc<ProviderFactory> = Arc::new(move |_provider_name, model| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            (model == "cheap-model")
                .then(|| override_provider.clone())
                .ok_or_else(|| format!("unexpected model '{model}'"))
        });

        let tool = AgentTool::new(
            vec![reviewer_subagent()],
            text_response_provider("used default"),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            Some(factory),
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(
                serde_json::json!({"name": "reviewer", "task": "review this", "model": "cheap-model"}),
                &ctx(),
            )
            .await;

        assert!(!outcome.is_error, "got: {}", outcome.content);
        assert_eq!(outcome.content, "used override");
        assert_eq!(factory_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_subagent_static_model_default_is_used_with_no_per_call_override() {
        let override_provider = text_response_provider("used pinned default");
        let factory: Arc<ProviderFactory> = Arc::new(move |_name, model| {
            (model == "pinned-model")
                .then(|| override_provider.clone())
                .ok_or_else(|| format!("unexpected model '{model}'"))
        });

        let mut subagent = reviewer_subagent();
        subagent.model = Some("pinned-model".to_string());

        let tool = AgentTool::new(
            vec![subagent],
            text_response_provider("used default"),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            Some(factory),
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(serde_json::json!({"name": "reviewer", "task": "review this"}), &ctx())
            .await;

        assert!(!outcome.is_error, "got: {}", outcome.content);
        assert_eq!(outcome.content, "used pinned default");
    }

    #[tokio::test]
    async fn a_per_call_model_argument_wins_over_the_subagents_static_default() {
        let pinned_provider = text_response_provider("used pinned");
        let call_provider = text_response_provider("used call override");
        let factory: Arc<ProviderFactory> = Arc::new(move |_name, model| match model {
            "pinned-model" => Ok(pinned_provider.clone()),
            "cheap-model" => Ok(call_provider.clone()),
            other => Err(format!("unexpected model '{other}'")),
        });

        let mut subagent = reviewer_subagent();
        subagent.model = Some("pinned-model".to_string());

        let tool = AgentTool::new(
            vec![subagent],
            text_response_provider("used default"),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            Some(factory),
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(
                serde_json::json!({"name": "reviewer", "task": "review this", "model": "cheap-model"}),
                &ctx(),
            )
            .await;

        assert_eq!(outcome.content, "used call override");
    }

    #[tokio::test]
    async fn no_override_means_the_factory_is_never_invoked() {
        let factory_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = factory_calls.clone();
        let factory: Arc<ProviderFactory> = Arc::new(move |_name, _model| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err("should never be called".to_string())
        });

        let tool = AgentTool::new(
            vec![reviewer_subagent()],
            text_response_provider("used default"),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            Some(factory),
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(serde_json::json!({"name": "reviewer", "task": "review this"}), &ctx())
            .await;

        assert!(!outcome.is_error, "got: {}", outcome.content);
        assert_eq!(outcome.content, "used default");
        assert_eq!(
            factory_calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the default (no-override) path must never call the factory"
        );
    }

    #[tokio::test]
    async fn a_failing_factory_is_a_tool_error_not_a_panic() {
        let factory: Arc<ProviderFactory> = Arc::new(|_name, _model| Err("missing API key".to_string()));

        let tool = AgentTool::new(
            vec![reviewer_subagent()],
            text_response_provider("used default"),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            Some(factory),
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(
                serde_json::json!({"name": "reviewer", "task": "review this", "model": "unknown-model"}),
                &ctx(),
            )
            .await;

        assert!(outcome.is_error);
        assert!(outcome.content.contains("missing API key"), "got: {}", outcome.content);
    }

    #[tokio::test]
    async fn requesting_an_override_with_no_factory_configured_is_a_tool_error() {
        let tool = AgentTool::new(
            vec![reviewer_subagent()],
            text_response_provider("used default"),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            None,
            Arc::new(AgentRegistry::new()),
        );

        let outcome = tool
            .execute(
                serde_json::json!({"name": "reviewer", "task": "review this", "model": "cheap-model"}),
                &ctx(),
            )
            .await;

        assert!(outcome.is_error);
    }

    #[test]
    fn model_argument_is_only_advertised_when_a_provider_factory_is_configured() {
        let without_factory = AgentTool::new(
            vec![reviewer_subagent()],
            text_response_provider("used default"),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            None,
            Arc::new(AgentRegistry::new()),
        );
        assert!(without_factory.parameters_schema()["properties"].get("model").is_none());

        let factory: Arc<ProviderFactory> = Arc::new(|_name, _model| Err("unused".to_string()));
        let with_factory = AgentTool::new(
            vec![reviewer_subagent()],
            text_response_provider("used default"),
            vec![],
            None,
            Arc::new(minder_core::NoopReporter),
            Some(factory),
            Arc::new(AgentRegistry::new()),
        );
        assert!(with_factory.parameters_schema()["properties"].get("model").is_some());
    }

    #[test]
    fn frontmatter_model_and_provider_overrides_are_parsed() {
        let agent_dir = scratch_dir();
        write_agent(
            &agent_dir,
            "quick",
            "---\nname: quick\ndescription: Fast searches\nmodel: llama3.2\nprovider: ollama\n---\nbody\n",
        );
        let subagents = discover_subagents(&agent_dir).unwrap();
        assert_eq!(subagents[0].model, Some("llama3.2".to_string()));
        assert_eq!(subagents[0].provider, Some("ollama".to_string()));
    }
}
