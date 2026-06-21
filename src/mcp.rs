use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rmcp::model::{CallToolRequestParams, CallToolResult, Content, RawContent, Tool};
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::process::Command;
use tracing::{debug, info, instrument, warn};

const MCP_CONFIG_PATH: &str = ".ferrix/mcp.json";
const DEFAULT_SEARCH_LIMIT: usize = 10;
const MAX_SEARCH_LIMIT: usize = 50;

#[derive(Debug)]
pub struct McpRegistry {
    servers: Vec<McpServer>,
    tools: Vec<McpToolEntry>,
}

impl McpRegistry {
    pub fn empty() -> Self {
        Self {
            servers: Vec::new(),
            tools: Vec::new(),
        }
    }

    pub async fn from_workspace(workspace_root: &Path) -> Result<Self> {
        let config_path = workspace_root.join(MCP_CONFIG_PATH);
        let Some(config) = McpConfig::load(&config_path)? else {
            debug!(path = %config_path.display(), "mcp config not found");
            return Ok(Self::empty());
        };

        Self::connect(workspace_root, config).await
    }

    async fn connect(workspace_root: &Path, config: McpConfig) -> Result<Self> {
        config.validate()?;

        let mut registry = Self::empty();
        for server in config.servers.into_iter().filter(|server| !server.disabled) {
            match connect_server(workspace_root, server).await {
                Ok((server, mut tools)) => {
                    registry.tools.append(&mut tools);
                    registry.servers.push(server);
                }
                Err(error) => {
                    warn!(error = %error, "failed to connect MCP server");
                }
            }
        }

        info!(
            server_count = registry.servers.len(),
            tool_count = registry.tools.len(),
            "mcp registry initialized"
        );
        Ok(registry)
    }

    pub fn search(&self, query: &str, limit: Option<usize>) -> McpSearchOutput {
        let limit = limit.unwrap_or(DEFAULT_SEARCH_LIMIT).min(MAX_SEARCH_LIMIT);
        let query_terms = search_terms(query);
        let mut matches = self
            .tools
            .iter()
            .filter_map(|tool| tool.search_match(&query_terms))
            .collect::<Vec<_>>();

        matches.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.server.cmp(&right.server))
                .then_with(|| left.tool.cmp(&right.tool))
        });
        matches.truncate(limit);

        McpSearchOutput {
            query: query.to_string(),
            matches,
            total_tools: self.tools.len(),
        }
    }

    #[instrument(skip_all, fields(server = %server_name, tool = %tool_name))]
    pub async fn call(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Value,
    ) -> Result<McpCallOutput> {
        let arguments = match arguments {
            Value::Object(arguments) => arguments,
            _ => bail!("MCP tool arguments must be a JSON object"),
        };

        if !self
            .tools
            .iter()
            .any(|tool| tool.server == server_name && tool.tool.name.as_ref() == tool_name)
        {
            bail!("unknown MCP tool `{server_name}/{tool_name}`");
        }

        let server = self
            .servers
            .iter()
            .find(|server| server.name == server_name)
            .with_context(|| format!("unknown MCP server `{server_name}`"))?;
        let result = server
            .service
            .peer()
            .call_tool(CallToolRequestParams::new(tool_name.to_string()).with_arguments(arguments))
            .await
            .with_context(|| format!("MCP tool `{server_name}/{tool_name}` failed"))?;

        Ok(normalize_call_result(server_name, tool_name, result))
    }

    #[cfg(test)]
    fn from_tools(tools: impl IntoIterator<Item = McpToolEntry>) -> Self {
        Self {
            servers: Vec::new(),
            tools: tools.into_iter().collect(),
        }
    }
}

#[derive(Debug)]
struct McpServer {
    name: String,
    service: RunningService<RoleClient, ()>,
}

#[derive(Debug, Clone)]
pub struct McpToolEntry {
    server: String,
    tool: Tool,
}

impl McpToolEntry {
    fn search_match(&self, query_terms: &[String]) -> Option<McpToolSearchResult> {
        let haystack = self.haystack();
        if !query_terms.iter().all(|term| haystack.contains(term)) {
            return None;
        }

        Some(McpToolSearchResult {
            server: self.server.clone(),
            tool: self.tool.name.to_string(),
            qualified_name: format!("{}/{}", self.server, self.tool.name),
            title: self.tool.title.clone(),
            description: self
                .tool
                .description
                .as_ref()
                .map(std::string::ToString::to_string),
            input_schema: self.tool.schema_as_json_value(),
            score: self.score(query_terms),
        })
    }

    fn haystack(&self) -> String {
        [
            self.server.as_str(),
            self.tool.name.as_ref(),
            self.tool.title.as_deref().unwrap_or_default(),
            self.tool
                .description
                .as_ref()
                .map(|description| description.as_ref())
                .unwrap_or_default(),
        ]
        .join(" ")
        .to_lowercase()
    }

    fn score(&self, query_terms: &[String]) -> u64 {
        if query_terms.is_empty() {
            return 1;
        }

        query_terms
            .iter()
            .map(|term| {
                let name = self.tool.name.to_lowercase();
                let title = self.tool.title.clone().unwrap_or_default().to_lowercase();
                let description = self
                    .tool
                    .description
                    .as_ref()
                    .map(|description| description.to_lowercase())
                    .unwrap_or_default();

                if name == *term {
                    100
                } else if name.contains(term) {
                    60
                } else if title.contains(term) {
                    30
                } else if description.contains(term) {
                    15
                } else {
                    1
                }
            })
            .sum()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct McpSearchOutput {
    pub query: String,
    pub matches: Vec<McpToolSearchResult>,
    pub total_tools: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct McpToolSearchResult {
    pub server: String,
    pub tool: String,
    pub qualified_name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub input_schema: Value,
    pub score: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct McpCallOutput {
    pub server: String,
    pub tool: String,
    pub ok: bool,
    pub content: String,
    pub data: Value,
}

#[derive(Debug, Deserialize)]
struct McpConfig {
    #[serde(default)]
    servers: Vec<McpServerConfig>,
}

impl McpConfig {
    fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }

        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read MCP config `{}`", path.display()))?;
        let config = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse MCP config `{}`", path.display()))?;
        Ok(Some(config))
    }

    fn validate(&self) -> Result<()> {
        let mut names = BTreeSet::new();
        for server in &self.servers {
            if server.name.trim().is_empty() {
                bail!("MCP server name must not be empty");
            }
            if server.command.trim().is_empty() {
                bail!("MCP server `{}` command must not be empty", server.name);
            }
            if !names.insert(server.name.as_str()) {
                bail!("duplicate MCP server `{}`", server.name);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct McpServerConfig {
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    disabled: bool,
}

async fn connect_server(
    workspace_root: &Path,
    config: McpServerConfig,
) -> Result<(McpServer, Vec<McpToolEntry>)> {
    let server_name = config.name.clone();
    let transport = TokioChildProcess::new(Command::new(&config.command).configure(|command| {
        command.args(&config.args);
        command.envs(&config.env);
        if let Some(cwd) = &config.cwd {
            let cwd = if cwd.is_absolute() {
                cwd.clone()
            } else {
                workspace_root.join(cwd)
            };
            command.current_dir(cwd);
        }
    }))
    .with_context(|| format!("failed to spawn MCP server `{server_name}`"))?;

    let service = ()
        .serve(transport)
        .await
        .with_context(|| format!("failed to initialize MCP server `{server_name}`"))?;
    let tools = service
        .peer()
        .list_all_tools()
        .await
        .with_context(|| format!("failed to list MCP tools for `{server_name}`"))?
        .into_iter()
        .map(|tool| McpToolEntry {
            server: server_name.clone(),
            tool,
        })
        .collect::<Vec<_>>();

    info!(
        server = %server_name,
        tool_count = tools.len(),
        "connected MCP server"
    );

    Ok((
        McpServer {
            name: server_name,
            service,
        },
        tools,
    ))
}

fn normalize_call_result(server: &str, tool: &str, result: CallToolResult) -> McpCallOutput {
    let ok = !result.is_error.unwrap_or(false);
    let text_content = text_content(&result.content);
    let content = result
        .structured_content
        .as_ref()
        .map(Value::to_string)
        .or_else(|| (!text_content.is_empty()).then(|| text_content.join("\n")))
        .unwrap_or_else(|| serde_json::to_string(&result.content).unwrap_or_default());
    let data = json!({
        "server": server,
        "tool": tool,
        "ok": ok,
        "content": result.content,
        "structured_content": result.structured_content,
        "meta": result.meta
    });

    McpCallOutput {
        server: server.to_string(),
        tool: tool.to_string(),
        ok,
        content,
        data,
    }
}

fn text_content(contents: &[Content]) -> Vec<String> {
    contents
        .iter()
        .filter_map(|content| match &content.raw {
            RawContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

fn search_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use rmcp::model::{Annotated, RawContent, Tool};
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_mcp_config_defaults() {
        let config: McpConfig = serde_json::from_value(json!({
            "servers": [{
                "name": "git",
                "command": "uvx"
            }]
        }))
        .expect("parse config");

        assert_eq!(config.servers[0].name, "git");
        assert_eq!(config.servers[0].args, Vec::<String>::new());
        assert_eq!(config.servers[0].env, BTreeMap::new());
        assert_eq!(config.servers[0].cwd, None);
        assert!(!config.servers[0].disabled);
        config.validate().expect("valid config");
    }

    #[test]
    fn rejects_duplicate_server_names() {
        let config: McpConfig = serde_json::from_value(json!({
            "servers": [
                { "name": "git", "command": "uvx" },
                { "name": "git", "command": "node" }
            ]
        }))
        .expect("parse config");

        let error = config.validate().expect_err("duplicate rejected");

        assert!(error.to_string().contains("duplicate MCP server"));
    }

    #[test]
    fn searches_tools_by_name_title_and_description() {
        let registry = McpRegistry::from_tools([
            tool_entry(
                "git",
                Tool::new(
                    "git_status",
                    "Show repository status",
                    Arc::new(serde_json::Map::new()),
                )
                .with_title("Git Status"),
            ),
            tool_entry(
                "docs",
                Tool::new(
                    "search_docs",
                    "Search project documentation",
                    Arc::new(serde_json::Map::new()),
                ),
            ),
        ]);

        let output = registry.search("git status", Some(5));

        assert_eq!(output.matches.len(), 1);
        assert_eq!(output.matches[0].qualified_name, "git/git_status");
    }

    #[test]
    fn normalizes_structured_call_results() {
        let result = CallToolResult::structured(json!({
            "status": "clean"
        }));

        let output = normalize_call_result("git", "git_status", result);

        assert!(output.ok);
        assert_eq!(output.content, "{\"status\":\"clean\"}");
        assert_eq!(output.data["server"], "git");
        assert_eq!(output.data["tool"], "git_status");
    }

    #[test]
    fn normalizes_text_call_results() {
        let result = CallToolResult::success(vec![Annotated::new(RawContent::text("hello"), None)]);

        let output = normalize_call_result("server", "echo", result);

        assert!(output.ok);
        assert_eq!(output.content, "hello");
    }

    fn tool_entry(server: &str, mut tool: Tool) -> McpToolEntry {
        tool.description = tool
            .description
            .map(|description| Cow::Owned(description.into_owned()));
        McpToolEntry {
            server: server.to_string(),
            tool,
        }
    }
}
