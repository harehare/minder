use agent_core::{HookDecision, HookPort, Message, ToolCall, ToolResultInfo};
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

    async fn on_tool_call(&mut self, call: &ToolCall) -> HookDecision<ToolCall> {
        self.invoke(
            "on_tool_call",
            call,
            HookDecision::Allow(call.clone()),
            FailMode::Closed,
        )
        .await
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
            HookDecision::Block(reason) => assert!(reason.contains("destructive")),
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
            HookDecision::Allow(c) => assert_eq!(c.arguments["command"], "ls -la"),
            other => panic!("expected Allow, got {other:?}"),
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
            matches!(decision, HookDecision::Block(_)),
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
}
