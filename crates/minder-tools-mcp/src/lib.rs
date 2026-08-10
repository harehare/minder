mod manifest;
mod plugin_manifest;

use async_trait::async_trait;
pub use manifest::{Manifest, ManifestError, ServerConfig};
use minder_core::{Tool, ToolContext, ToolExecOutcome};
pub use plugin_manifest::{PluginMcpManifest, PluginMcpManifestError, PluginServerConfig};
use rmcp::model::CallToolRequestParams;
use rmcp::service::{ClientInitializeError, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum McpToolError {
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(transparent)]
    PluginManifest(#[from] PluginMcpManifestError),
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
    #[error("mcp server \"{name}\" has an invalid header {header:?}")]
    InvalidHeader { name: String, header: String },
    // rmcp has no client-side transport for the legacy HTTP+SSE protocol
    // (only stdio and streamable-http) -- see connect_sse's doc comment.
    #[error(
        "mcp server \"{name}\" uses the \"sse\" transport, which minder doesn't support yet -- use \"streamable-http\" instead"
    )]
    UnsupportedTransport { name: String },
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
/// an `minder_core::Tool` named `mcp__<server>__<tool>` (mirroring the
/// `mcp__server__tool` naming other agent harnesses use, so remote tools
/// stay unambiguous across servers). Mirrors `minder_tools_wasm::load_plugins`'
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
    connect_stdio(name, command, args, env, None).await
}

async fn connect_stdio(
    name: String,
    command: String,
    args: Vec<String>,
    env: std::collections::HashMap<String, String>,
    cwd: Option<String>,
) -> Result<Vec<Box<dyn Tool>>, McpToolError> {
    let mut cmd = tokio::process::Command::new(&command);
    cmd.envs(&env);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let transport = TokioChildProcess::new(cmd.configure(|c| {
        c.args(&args);
    }))
    .map_err(|e| McpToolError::Spawn {
        name: name.clone(),
        command: command.clone(),
        source: e,
    })?;

    let client: McpClient = ().serve(transport).await.map_err(|e| McpToolError::Initialize {
        name: name.clone(),
        source: Box::new(e),
    })?;
    wrap_client_tools(name, client).await
}

/// Connects to a `streamable-http` MCP server over HTTPS/HTTP via `reqwest`,
/// the successor to the legacy HTTP+SSE transport in the MCP spec.
async fn connect_streamable_http(
    name: String,
    url: String,
    headers: std::collections::HashMap<String, String>,
) -> Result<Vec<Box<dyn Tool>>, McpToolError> {
    let mut custom_headers = std::collections::HashMap::new();
    for (key, value) in headers {
        let header_name = http::HeaderName::from_bytes(key.as_bytes()).map_err(|_| McpToolError::InvalidHeader {
            name: name.clone(),
            header: key.clone(),
        })?;
        let header_value = http::HeaderValue::from_str(&value).map_err(|_| McpToolError::InvalidHeader {
            name: name.clone(),
            header: key,
        })?;
        custom_headers.insert(header_name, header_value);
    }

    let config = StreamableHttpClientTransportConfig::with_uri(url).custom_headers(custom_headers);
    let transport = StreamableHttpClientTransport::with_client(reqwest::Client::default(), config);

    let client: McpClient = ().serve(transport).await.map_err(|e| McpToolError::Initialize {
        name: name.clone(),
        source: Box::new(e),
    })?;
    wrap_client_tools(name, client).await
}

async fn wrap_client_tools(name: String, client: McpClient) -> Result<Vec<Box<dyn Tool>>, McpToolError> {
    let client = Arc::new(client);
    let remote_tools = client.list_all_tools().await.map_err(|e| McpToolError::ListTools {
        name: name.clone(),
        source: e,
    })?;

    Ok(remote_tools
        .into_iter()
        .map(|remote_tool| -> Box<dyn Tool> {
            Box::new(McpTool {
                name: format!("mcp__{name}__{}", remote_tool.name),
                description: remote_tool.description.map(|d| d.to_string()).unwrap_or_default(),
                parameters_schema: serde_json::Value::Object((*remote_tool.input_schema).clone()),
                remote_name: remote_tool.name.to_string(),
                client: client.clone(),
            })
        })
        .collect())
}

/// Discovers a plugin's `mcp.json` (Agent Plugins spec) at
/// `plugin_root/mcp.json` and connects every configured server, the same
/// way [`load_mcp_tools`] does for the project's `.agent/mcp.toml`. Missing
/// file -> `Ok(vec![])`; `sse`-transport servers are a hard error since rmcp
/// has no client-side legacy-SSE transport (only stdio and streamable-http).
pub async fn load_plugin_mcp_tools(plugin_root: &Path) -> Result<Vec<Box<dyn Tool>>, McpToolError> {
    let manifest_path = plugin_root.join("mcp.json");
    if !manifest_path.exists() {
        return Ok(Vec::new());
    }

    let manifest = PluginMcpManifest::load(&manifest_path).await?;
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();
    for (name, server) in manifest.servers {
        let server_tools = match server {
            PluginServerConfig::Stdio {
                command,
                args,
                env,
                cwd,
            } => connect_stdio(name, command, args, env, cwd).await?,
            PluginServerConfig::StreamableHttp { url, headers } => connect_streamable_http(name, url, headers).await?,
            PluginServerConfig::Sse { .. } => return Err(McpToolError::UnsupportedTransport { name }),
        };
        tools.extend(server_tools);
    }
    Ok(tools)
}
