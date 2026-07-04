use agent_core::{
    HookDecision, HookPort, Message, RenderDecision, ToolCall, ToolCallDecision, ToolExecOutcome,
    ToolResultInfo,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailMode {
    Open,
    Closed,
}

#[derive(Debug, thiserror::Error)]
pub enum HookLoadError {
    #[error("failed to read hooks directory {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("failed to load hook module '{0}': {1}")]
    Mq(String, Box<mq_lang::Error>),
}

pub struct HookEngine {
    engine: mq_lang::DefaultEngine,
    compiled: HashMap<String, mq_lang::CompiledProgram>,
}

/// Convenience accessors over the `messages`/`call`/`result` shapes hook
/// scripts already receive (`agent_tool_calls`, `agent_error_count`, ...) --
/// see `agent.mq` for the full list. Always evaluated into the engine before
/// any user hook file, so these are bare-callable from any hook without an
/// explicit `import`; a user hook file redefining one of these names simply
/// shadows it (loaded after, in the same flat namespace).
const AGENT_MODULE_SOURCE: &str = include_str!("agent.mq");

/// Contract for hooks that transform a value: `{"action": "allow", "value": T}`
/// or `{"action": "block", "reason": "..."}`.
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum RawDecision<T> {
    Allow { value: T },
    Block { reason: String },
}

/// Contract for hooks that only gate (no payload to transform), e.g.
/// `before_compact`: `{"action": "allow"}` or `{"action": "block", "reason": "..."}`.
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum RawGate {
    Allow,
    Block { reason: String },
}

/// Contract for `on_tool_call`: like `RawDecision<ToolCall>` but with a third
/// option, `{"action": "override", "value": {"content": ..., "is_error": ...,
/// "metadata": ...}}`, to supply the tool's result directly instead of
/// allowing/blocking the call -- real middleware, not just an observer.
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum RawToolCallDecision {
    Allow { value: ToolCall },
    Block { reason: String },
    Override { value: ToolExecOutcome },
}

/// Contract for `render_tool_call`/`render_tool_result`: `{"action": "default"}`,
/// `{"action": "text", "value": "...", "style": "green"}` (`style` optional),
/// or `{"action": "hide"}`.
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum RawRenderDecision {
    Default,
    Text {
        value: String,
        #[serde(default)]
        style: Option<String>,
    },
    Hide,
}

impl From<RawRenderDecision> for RenderDecision {
    fn from(raw: RawRenderDecision) -> Self {
        match raw {
            RawRenderDecision::Default => RenderDecision::Default,
            RawRenderDecision::Text { value, style } => RenderDecision::Text { value, style },
            RawRenderDecision::Hide => RenderDecision::Hide,
        }
    }
}

/// Bundles a tool call with its outcome into the single JSON value
/// `render_tool_result` receives (the `__hook_arg`-single-value convention
/// every hook argument follows).
#[derive(Serialize)]
struct RenderResultArg<'a> {
    call: &'a ToolCall,
    outcome: &'a ToolExecOutcome,
}

impl HookEngine {
    /// Loads hook scripts from `agent_dir` (typically `.agent`): either
    /// `agent_dir/hooks/*.mq` (one shared namespace across files) or the
    /// single-file fallback `agent_dir/hooks.mq`. Returns `Ok(None)` if
    /// neither exists -- hooks are fully optional.
    pub fn load(agent_dir: &Path) -> Result<Option<Self>, HookLoadError> {
        let hooks_dir = agent_dir.join("hooks");
        let single_file = agent_dir.join("hooks.mq");

        let (search_dir, mut module_names) = if hooks_dir.is_dir() {
            let mut names = Vec::new();
            for entry in std::fs::read_dir(&hooks_dir)
                .map_err(|e| HookLoadError::Io(hooks_dir.clone(), e))?
            {
                let entry = entry.map_err(|e| HookLoadError::Io(hooks_dir.clone(), e))?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("mq")
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                {
                    names.push(stem.to_string());
                }
            }
            (hooks_dir, names)
        } else if single_file.is_file() {
            (agent_dir.to_path_buf(), vec!["hooks".to_string()])
        } else {
            return Ok(None);
        };

        if module_names.is_empty() {
            return Ok(None);
        }
        module_names.sort(); // deterministic load order

        let mut engine = mq_lang::DefaultEngine::default();
        engine.load_builtin_module();
        engine.set_search_paths(vec![search_dir]);

        // Plain `eval` of a source string full of top-level `def`s persists
        // those defs into the engine's shared env the same way `load_module`
        // does (verified against mq-lang directly) -- no temp file needed to
        // make our own bundled module available the same way a user's would
        // be via `load_module`.
        engine
            .eval(AGENT_MODULE_SOURCE, mq_lang::null_input().into_iter())
            .map_err(|e| HookLoadError::Mq("agent".to_string(), e))?;

        for name in &module_names {
            // unqualified load: every file's top-level defs land in one
            // shared namespace, so hook functions are called bare (e.g.
            // `on_tool_call(...)`), not `hook::on_tool_call(...)`.
            engine
                .load_module(name)
                .map_err(|e| HookLoadError::Mq(name.clone(), e))?;
        }

        Ok(Some(Self {
            engine,
            compiled: HashMap::new(),
        }))
    }

    fn compile_call(&mut self, fn_name: &str) -> Result<(), String> {
        if self.compiled.contains_key(fn_name) {
            return Ok(());
        }
        match self.engine.compile(&format!("{fn_name}(__hook_arg)")) {
            Ok(program) => {
                self.compiled.insert(fn_name.to_string(), program);
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// mq-lang has no public API to detect "this function was never defined"
    /// from outside the crate (the relevant error variant isn't exported),
    /// so this matches the exact thiserror-generated message for
    /// `RuntimeError::NotDefined` ("\"{name}\" is not defined"). A hook that
    /// was simply never written is always a no-op, regardless of fail_mode.
    fn is_not_defined(err: &mq_lang::Error, fn_name: &str) -> bool {
        err.to_string() == format!("\"{fn_name}\" is not defined")
    }

    async fn invoke<In: Serialize, Out: DeserializeOwned>(
        &mut self,
        fn_name: &str,
        arg: &In,
        default: HookDecision<Out>,
        fail_mode: FailMode,
    ) -> HookDecision<Out> {
        let json = match serde_json::to_value(arg) {
            Ok(v) => v,
            Err(e) => {
                return self.on_error(
                    fn_name,
                    format!("failed to serialize hook argument: {e}"),
                    default,
                    fail_mode,
                );
            }
        };
        self.engine
            .define_value("__hook_arg", mq_lang::RuntimeValue::from(json));

        if let Err(msg) = self.compile_call(fn_name) {
            return self.on_error(fn_name, msg, default, fail_mode);
        }
        let compiled = self.compiled.get(fn_name).expect("just compiled");

        match self
            .engine
            .eval_compiled(compiled, mq_lang::null_input().into_iter())
        {
            Ok(values) => {
                let result_json = values
                    .values()
                    .first()
                    .cloned()
                    .unwrap_or(mq_lang::RuntimeValue::NONE)
                    .to_json_value();
                match serde_json::from_value::<RawDecision<Out>>(result_json) {
                    Ok(RawDecision::Allow { value }) => HookDecision::Allow(value),
                    Ok(RawDecision::Block { reason }) => HookDecision::Block(reason),
                    Err(e) => self.on_error(
                        fn_name,
                        format!("malformed hook return value: {e}"),
                        default,
                        fail_mode,
                    ),
                }
            }
            Err(err) if Self::is_not_defined(&err, fn_name) => default,
            Err(err) => self.on_error(fn_name, err.to_string(), default, fail_mode),
        }
    }

    async fn invoke_gate<In: Serialize>(
        &mut self,
        fn_name: &str,
        arg: &In,
        default: HookDecision<()>,
        fail_mode: FailMode,
    ) -> HookDecision<()> {
        let json = match serde_json::to_value(arg) {
            Ok(v) => v,
            Err(e) => {
                return self.on_error(
                    fn_name,
                    format!("failed to serialize hook argument: {e}"),
                    default,
                    fail_mode,
                );
            }
        };
        self.engine
            .define_value("__hook_arg", mq_lang::RuntimeValue::from(json));

        if let Err(msg) = self.compile_call(fn_name) {
            return self.on_error(fn_name, msg, default, fail_mode);
        }
        let compiled = self.compiled.get(fn_name).expect("just compiled");

        match self
            .engine
            .eval_compiled(compiled, mq_lang::null_input().into_iter())
        {
            Ok(values) => {
                let result_json = values
                    .values()
                    .first()
                    .cloned()
                    .unwrap_or(mq_lang::RuntimeValue::NONE)
                    .to_json_value();
                match serde_json::from_value::<RawGate>(result_json) {
                    Ok(RawGate::Allow) => HookDecision::Allow(()),
                    Ok(RawGate::Block { reason }) => HookDecision::Block(reason),
                    Err(e) => self.on_error(
                        fn_name,
                        format!("malformed hook return value: {e}"),
                        default,
                        fail_mode,
                    ),
                }
            }
            Err(err) if Self::is_not_defined(&err, fn_name) => default,
            Err(err) => self.on_error(fn_name, err.to_string(), default, fail_mode),
        }
    }

    fn on_error<T>(
        &self,
        fn_name: &str,
        message: String,
        default: HookDecision<T>,
        fail_mode: FailMode,
    ) -> HookDecision<T> {
        tracing::error!(hook = fn_name, error = %message, "hook script error");
        match fail_mode {
            FailMode::Open => default,
            FailMode::Closed => HookDecision::Block(message),
        }
    }

    /// `on_tool_call` gets its own invoke path (rather than reusing the
    /// generic `invoke<In, Out>`) because its decision type has a third,
    /// non-`HookDecision`-shaped option (`Override`) -- and it always fails
    /// closed, same as before this was split out.
    async fn invoke_tool_call(&mut self, call: &ToolCall) -> ToolCallDecision {
        let json = match serde_json::to_value(call) {
            Ok(v) => v,
            Err(e) => {
                return self.tool_call_error(format!("failed to serialize hook argument: {e}"));
            }
        };
        self.engine
            .define_value("__hook_arg", mq_lang::RuntimeValue::from(json));

        if let Err(msg) = self.compile_call("on_tool_call") {
            return self.tool_call_error(msg);
        }
        let compiled = self.compiled.get("on_tool_call").expect("just compiled");

        match self
            .engine
            .eval_compiled(compiled, mq_lang::null_input().into_iter())
        {
            Ok(values) => {
                let result_json = values
                    .values()
                    .first()
                    .cloned()
                    .unwrap_or(mq_lang::RuntimeValue::NONE)
                    .to_json_value();
                match serde_json::from_value::<RawToolCallDecision>(result_json) {
                    Ok(RawToolCallDecision::Allow { value }) => ToolCallDecision::Allow(value),
                    Ok(RawToolCallDecision::Block { reason }) => ToolCallDecision::Block(reason),
                    Ok(RawToolCallDecision::Override { value }) => {
                        ToolCallDecision::Override(value)
                    }
                    Err(e) => self.tool_call_error(format!("malformed hook return value: {e}")),
                }
            }
            Err(err) if Self::is_not_defined(&err, "on_tool_call") => {
                ToolCallDecision::Allow(call.clone())
            }
            Err(err) => self.tool_call_error(err.to_string()),
        }
    }

    fn tool_call_error(&self, message: String) -> ToolCallDecision {
        tracing::error!(hook = "on_tool_call", error = %message, "hook script error");
        ToolCallDecision::Block(message)
    }

    /// Shared by `render_tool_call`/`render_tool_result`: unlike every other
    /// hook point, a display hook always fails open to `RenderDecision::Default`
    /// -- a broken or undefined render script can only ever fall back to the
    /// harness's built-in formatting, never affect execution.
    async fn invoke_render<In: Serialize>(&mut self, fn_name: &str, arg: &In) -> RenderDecision {
        let json = match serde_json::to_value(arg) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(hook = fn_name, error = %e, "hook script error");
                return RenderDecision::Default;
            }
        };
        self.engine
            .define_value("__hook_arg", mq_lang::RuntimeValue::from(json));

        if let Err(msg) = self.compile_call(fn_name) {
            tracing::error!(hook = fn_name, error = %msg, "hook script error");
            return RenderDecision::Default;
        }
        let compiled = self.compiled.get(fn_name).expect("just compiled");

        match self
            .engine
            .eval_compiled(compiled, mq_lang::null_input().into_iter())
        {
            Ok(values) => {
                let result_json = values
                    .values()
                    .first()
                    .cloned()
                    .unwrap_or(mq_lang::RuntimeValue::NONE)
                    .to_json_value();
                match serde_json::from_value::<RawRenderDecision>(result_json) {
                    Ok(raw) => raw.into(),
                    Err(e) => {
                        tracing::error!(hook = fn_name, error = %e, "malformed hook return value");
                        RenderDecision::Default
                    }
                }
            }
            Err(err) if Self::is_not_defined(&err, fn_name) => RenderDecision::Default,
            Err(err) => {
                tracing::error!(hook = fn_name, error = %err, "hook script error");
                RenderDecision::Default
            }
        }
    }
}

#[async_trait]
impl HookPort for HookEngine {
    async fn before_agent_start(&mut self, system_prompt: &str) -> HookDecision<String> {
        self.invoke(
            "before_agent_start",
            &system_prompt,
            HookDecision::Allow(system_prompt.to_string()),
            FailMode::Open,
        )
        .await
    }

    async fn on_context(&mut self, messages: &[Message]) -> HookDecision<Vec<Message>> {
        self.invoke(
            "on_context",
            &messages,
            HookDecision::Allow(messages.to_vec()),
            FailMode::Open,
        )
        .await
    }

    async fn on_tool_call(&mut self, call: &ToolCall) -> ToolCallDecision {
        self.invoke_tool_call(call).await
    }

    async fn on_tool_result(&mut self, result: &ToolResultInfo) -> HookDecision<String> {
        self.invoke(
            "on_tool_result",
            result,
            HookDecision::Allow(result.content.clone()),
            FailMode::Open,
        )
        .await
    }

    async fn before_compact(&mut self, messages: &[Message]) -> HookDecision<()> {
        self.invoke_gate(
            "before_compact",
            &messages,
            HookDecision::Allow(()),
            FailMode::Open,
        )
        .await
    }

    async fn render_tool_call(&mut self, call: &ToolCall) -> RenderDecision {
        self.invoke_render("render_tool_call", call).await
    }

    async fn render_tool_result(
        &mut self,
        call: &ToolCall,
        outcome: &ToolExecOutcome,
    ) -> RenderDecision {
        self.invoke_render("render_tool_result", &RenderResultArg { call, outcome })
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_agent_dir() -> PathBuf {
        std::env::temp_dir().join(format!("minder-hooks-test-{}", uuid::Uuid::new_v4()))
    }

    fn write_hook(agent_dir: &Path, filename: &str, content: &str) {
        let hooks_dir = agent_dir.join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(hooks_dir.join(filename), content).unwrap();
    }

    fn bash_call(id: &str, command: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({ "command": command }),
        }
    }

    const SECURITY_HOOK: &str = r#"
def on_tool_call(call):
  if (call["name"] == "bash" && contains(call["arguments"]["command"], "rm -rf")):
    {"action": "block", "reason": "destructive bash command blocked by policy"}
  else:
    {"action": "allow", "value": call};
"#;

    #[test]
    fn load_returns_none_when_no_hooks_present() {
        let agent_dir = temp_agent_dir();
        std::fs::create_dir_all(&agent_dir).unwrap();
        assert!(HookEngine::load(&agent_dir).unwrap().is_none());
    }

    #[test]
    fn load_finds_single_file_convention() {
        let agent_dir = temp_agent_dir();
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(agent_dir.join("hooks.mq"), SECURITY_HOOK).unwrap();
        assert!(HookEngine::load(&agent_dir).unwrap().is_some());
    }

    #[tokio::test]
    async fn on_tool_call_blocks_destructive_command() {
        let agent_dir = temp_agent_dir();
        write_hook(&agent_dir, "security.mq", SECURITY_HOOK);
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let decision = engine
            .on_tool_call(&bash_call("1", "rm -rf /tmp/foo"))
            .await;
        match decision {
            ToolCallDecision::Block(reason) => assert!(reason.contains("destructive")),
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_tool_call_allows_safe_command() {
        let agent_dir = temp_agent_dir();
        write_hook(&agent_dir, "security.mq", SECURITY_HOOK);
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let call = bash_call("2", "ls -la");
        let decision = engine.on_tool_call(&call).await;
        match decision {
            ToolCallDecision::Allow(c) => assert_eq!(c.arguments["command"], "ls -la"),
            other => panic!("expected Allow, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_tool_call_can_override_the_result_without_running_the_real_tool() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "mock.mq",
            r#"
def on_tool_call(call):
  if (call["name"] == "web_fetch"):
    {"action": "override", "value": {"content": "mocked response", "is_error": false, "metadata": None}}
  else:
    {"action": "allow", "value": call};
"#,
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let call = ToolCall {
            id: "1".to_string(),
            name: "web_fetch".to_string(),
            arguments: serde_json::json!({"url": "https://example.com"}),
        };
        match engine.on_tool_call(&call).await {
            ToolCallDecision::Override(outcome) => {
                assert_eq!(outcome.content, "mocked response");
                assert!(!outcome.is_error);
            }
            other => panic!("expected Override, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn undefined_hook_is_a_no_op() {
        let agent_dir = temp_agent_dir();
        write_hook(&agent_dir, "security.mq", SECURITY_HOOK); // only defines on_tool_call
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let result = agent_core::ToolResultInfo {
            tool_name: "bash".to_string(),
            content: "some output".to_string(),
            is_error: false,
        };
        let decision = engine.on_tool_result(&result).await;
        match decision {
            HookDecision::Allow(content) => assert_eq!(content, "some output"),
            other => panic!("expected Allow(default), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn buggy_tool_call_hook_fails_closed() {
        let agent_dir = temp_agent_dir();
        // references an undefined variable -- a genuine script bug, not "undefined hook"
        write_hook(
            &agent_dir,
            "buggy.mq",
            "def on_tool_call(call): does_not_exist(call);",
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let decision = engine.on_tool_call(&bash_call("3", "ls")).await;
        assert!(
            matches!(decision, ToolCallDecision::Block(_)),
            "tool_call must fail closed on a script bug"
        );
    }

    #[tokio::test]
    async fn buggy_transform_hook_fails_open() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "buggy.mq",
            "def on_tool_result(result): does_not_exist(result);",
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let result = agent_core::ToolResultInfo {
            tool_name: "bash".to_string(),
            content: "original output".to_string(),
            is_error: false,
        };
        let decision = engine.on_tool_result(&result).await;
        match decision {
            HookDecision::Allow(content) => assert_eq!(content, "original output"),
            other => panic!("on_tool_result must fail open on a script bug, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn before_compact_gate_allows_by_default() {
        let agent_dir = temp_agent_dir();
        write_hook(&agent_dir, "security.mq", SECURITY_HOOK); // doesn't define before_compact
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let messages = vec![Message::user_text("hi")];
        assert!(matches!(
            engine.before_compact(&messages).await,
            HookDecision::Allow(())
        ));
    }

    #[tokio::test]
    async fn before_compact_can_block() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "compact.mq",
            r#"def before_compact(messages): {"action": "block", "reason": "not now"};"#,
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let messages = vec![Message::user_text("hi")];
        match engine.before_compact(&messages).await {
            HookDecision::Block(reason) => assert_eq!(reason, "not now"),
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_context_can_filter_messages() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "context.mq",
            r#"def on_context(messages): {"action": "allow", "value": []};"#,
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let messages = vec![Message::user_text("secret stuff")];
        match engine.on_context(&messages).await {
            HookDecision::Allow(filtered) => assert!(filtered.is_empty()),
            other => panic!("expected Allow([]), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn before_agent_start_can_modify_system_prompt() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "prompt.mq",
            r#"def before_agent_start(prompt): {"action": "allow", "value": prompt + " Be extra careful."};"#,
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        match engine.before_agent_start("You are an agent.").await {
            HookDecision::Allow(prompt) => {
                assert_eq!(prompt, "You are an agent. Be extra careful.")
            }
            other => panic!("expected Allow, got {other:?}"),
        }
    }

    // -- embedded `agent` module (agent.mq) --

    use agent_core::{ContentBlock, Role, ToolResult, ToolResultContent};

    fn tool_result_message(tool_call_id: &str, content: &str, is_error: bool) -> Message {
        Message::tool_results(vec![ToolResult {
            tool_call_id: tool_call_id.to_string(),
            content: ToolResultContent::Text(content.to_string()),
            is_error,
        }])
    }

    #[tokio::test]
    async fn agent_module_consecutive_errors_gates_compaction() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "compact.mq",
            r#"
def before_compact(messages):
  if (agent_consecutive_errors(agent_tool_results(messages)) >= 2):
    {"action": "block", "reason": "too many consecutive errors"}
  else:
    {"action": "allow"};
"#,
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let messages = vec![
            tool_result_message("1", "ok", false),
            tool_result_message("2", "fail", true),
            tool_result_message("3", "fail again", true),
        ];

        match engine.before_compact(&messages).await {
            HookDecision::Block(reason) => assert!(reason.contains("consecutive")),
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn agent_module_functions_compose_inside_a_hook() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "trim.mq",
            r#"
def on_context(messages):
  if (agent_error_count(messages) > 0):
    {"action": "allow", "value": agent_last_n(messages, 1)}
  else:
    {"action": "allow", "value": messages};
"#,
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let messages = vec![
            Message::user_text("do something"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse(bash_call("1", "false"))],
                metadata: serde_json::Value::Null,
            },
            tool_result_message("1", "command failed", true),
        ];

        match engine.on_context(&messages).await {
            HookDecision::Allow(filtered) => assert_eq!(filtered.len(), 1),
            other => panic!("expected Allow, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn user_hook_file_can_shadow_an_agent_module_function() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "shadow.mq",
            r#"
def agent_tool_names(messages): "shadowed";
def before_agent_start(prompt): {"action": "allow", "value": prompt + " " + agent_tool_names([])};
"#,
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        match engine.before_agent_start("base").await {
            HookDecision::Allow(prompt) => assert_eq!(prompt, "base shadowed"),
            other => panic!("expected Allow, got {other:?}"),
        }
    }

    // -- render_tool_call / render_tool_result --

    const RENDER_HOOK: &str = r#"
def render_tool_call(call):
  if (call["name"] == "bash"):
    {"action": "text", "value": "$ " + call["arguments"]["command"], "style": "cyan"}
  elif (call["name"] == "git_status"):
    {"action": "hide"}
  else:
    {"action": "default"};

def render_tool_result(arg):
  if (arg["call"]["name"] == "bash" && arg["outcome"]["is_error"]):
    {"action": "text", "value": "command failed", "style": "red"}
  else:
    {"action": "default"};
"#;

    #[tokio::test]
    async fn render_tool_call_undefined_falls_back_to_default() {
        let agent_dir = temp_agent_dir();
        write_hook(&agent_dir, "security.mq", SECURITY_HOOK); // doesn't define render_tool_call
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let decision = engine.render_tool_call(&bash_call("1", "ls")).await;
        assert!(matches!(decision, RenderDecision::Default));
    }

    #[tokio::test]
    async fn render_tool_call_can_customize_text_and_style() {
        let agent_dir = temp_agent_dir();
        write_hook(&agent_dir, "render.mq", RENDER_HOOK);
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        match engine.render_tool_call(&bash_call("1", "ls -la")).await {
            RenderDecision::Text { value, style } => {
                assert_eq!(value, "$ ls -la");
                assert_eq!(style.as_deref(), Some("cyan"));
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn render_tool_call_can_hide() {
        let agent_dir = temp_agent_dir();
        write_hook(&agent_dir, "render.mq", RENDER_HOOK);
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let call = ToolCall {
            id: "2".to_string(),
            name: "git_status".to_string(),
            arguments: serde_json::json!({}),
        };
        assert!(matches!(
            engine.render_tool_call(&call).await,
            RenderDecision::Hide
        ));
    }

    #[tokio::test]
    async fn render_tool_result_receives_both_call_and_outcome() {
        let agent_dir = temp_agent_dir();
        write_hook(&agent_dir, "render.mq", RENDER_HOOK);
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let call = bash_call("3", "false");
        let outcome = ToolExecOutcome {
            content: "exit 1".to_string(),
            is_error: true,
            metadata: serde_json::Value::Null,
        };
        match engine.render_tool_result(&call, &outcome).await {
            RenderDecision::Text { value, style } => {
                assert_eq!(value, "command failed");
                assert_eq!(style.as_deref(), Some("red"));
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_buggy_render_hook_falls_back_to_default_not_an_error() {
        let agent_dir = temp_agent_dir();
        write_hook(
            &agent_dir,
            "buggy.mq",
            "def render_tool_call(call): does_not_exist(call);",
        );
        let mut engine = HookEngine::load(&agent_dir).unwrap().unwrap();

        let decision = engine.render_tool_call(&bash_call("4", "ls")).await;
        assert!(
            matches!(decision, RenderDecision::Default),
            "a broken render hook must fall back to Default, not break/block"
        );
    }
}
