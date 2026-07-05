mod manifest;

use agent_core::{Tool, ToolContext, ToolExecOutcome};
use async_trait::async_trait;
pub use manifest::{Manifest, ManifestError, ServerConfig};
use rmcp::model::CallToolRequestParams;
use rmcp::service::{ClientInitializeError, RunningService};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum McpToolError {
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error("mcp server \"{name}\" failed to start `{command}`: {source}")]
    Spawn {
        name: String,
        command: String,
        #[source]
        source: std::io::Error,
    },
    #[error("mcp server \"{name}\" failed to initialize: {source}")]
    Initialize {
        name: String,
        #[source]
        source: Box<ClientInitializeError>,
    },
    #[error("mcp server \"{name}\" failed to list tools: {source}")]
    ListTools {
        name: String,
        #[source]
        source: rmcp::ServiceError,
    },
}

type McpClient = RunningService<RoleClient, ()>;

/// A tool backed by a remote MCP server tool, reached over the server's
/// stdio JSON-RPC session. Implements the same `Tool` trait as built-ins and
/// WASM plugins, so it slots into the existing `Vec<Box<dyn Tool>>` with no
/// changes to `AgentSession`.
struct McpTool {
    name: String,
    description: String,
    parameters_schema: serde_json::Value,
    remote_name: String,
    client: Arc<McpClient>,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.parameters_schema.clone()
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: &ToolContext) -> ToolExecOutcome {
        let mut params = CallToolRequestParams::new(self.remote_name.clone());
        if let Some(object) = arguments.as_object() {
            params = params.with_arguments(object.clone());
        }

        match self.client.call_tool(params).await {
            Ok(result) => {
                let content = result
                    .content
                    .iter()
                    .filter_map(|block| block.as_text())
                    .map(|text| text.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolExecOutcome {
                    content,
                    is_error: result.is_error.unwrap_or(false),
                    metadata: serde_json::Value::Null,
                }
            }
            Err(e) => ToolExecOutcome {
                content: format!("mcp call to \"{}\" failed: {e}", self.remote_name),
                is_error: true,
                metadata: serde_json::Value::Null,
            },
        }
    }
}

/// Discovers `.agent/mcp.toml`, spawns each configured server as a child
/// process over the stdio transport, and exposes every tool it advertises as
/// an `agent_core::Tool` named `mcp__<server>__<tool>` (mirroring the
/// `mcp__server__tool` naming other agent harnesses use, so remote tools
/// stay unambiguous across servers). Mirrors `agent_tools_wasm::load_plugins`'
/// optionality: a missing manifest is not an error, `Ok(vec![])`. A server
/// that fails to start, initialize, or list its tools is a hard error --
/// same as a wasm plugin load failure.
pub async fn load_mcp_tools(agent_dir: &Path) -> Result<Vec<Box<dyn Tool>>, McpToolError> {
    let manifest_path = agent_dir.join("mcp.toml");
    if !manifest_path.exists() {
        return Ok(Vec::new());
    }

    let manifest = Manifest::load(&manifest_path).await?;
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();
    for server in manifest.servers {
        tools.extend(connect_server(server).await?);
    }
    Ok(tools)
}

async fn connect_server(server: ServerConfig) -> Result<Vec<Box<dyn Tool>>, McpToolError> {
    let ServerConfig {
        name,
        command,
        args,
        env,
    } = server;

    let mut cmd = tokio::process::Command::new(&command);
    cmd.envs(&env);
    let transport = TokioChildProcess::new(cmd.configure(|c| {
        c.args(&args);
    }))
    .map_err(|e| McpToolError::Spawn {
        name: name.clone(),
        command: command.clone(),
        source: e,
    })?;

    let client: McpClient = ()
        .serve(transport)
        .await
        .map_err(|e| McpToolError::Initialize {
            name: name.clone(),
            source: Box::new(e),
        })?;
    let client = Arc::new(client);

    let remote_tools =
        client
            .list_all_tools()
            .await
            .map_err(|e| McpToolError::ListTools {
                name: name.clone(),
                source: e,
            })?;

    Ok(remote_tools
        .into_iter()
        .map(|remote_tool| -> Box<dyn Tool> {
            Box::new(McpTool {
                name: format!("mcp__{name}__{}", remote_tool.name),
                description: remote_tool
                    .description
                    .map(|d| d.to_string())
                    .unwrap_or_default(),
                parameters_schema: serde_json::Value::Object((*remote_tool.input_schema).clone()),
                remote_name: remote_tool.name.to_string(),
                client: client.clone(),
            })
        })
        .collect())
}
