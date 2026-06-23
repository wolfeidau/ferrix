pub mod bash;
pub mod fs;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{info, instrument};

use crate::mcp::McpRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub name: String,
    pub ok: bool,
    pub content: String,
    pub data: Value,
}

#[derive(Debug)]
pub struct ToolRegistry {
    workspace_root: PathBuf,
    mcp: McpRegistry,
}

impl ToolRegistry {
    #[cfg(test)]
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            mcp: McpRegistry::empty(),
        }
    }

    pub fn with_mcp(workspace_root: PathBuf, mcp: McpRegistry) -> Self {
        Self {
            workspace_root,
            mcp,
        }
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "read".to_string(),
                description: "Read a UTF-8 text file from disk.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute file path to read."
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "write".to_string(),
                description: "Write full UTF-8 contents to a file.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute file path to write."
                        },
                        "contents": {
                            "type": "string",
                            "description": "Complete file contents."
                        },
                        "create_dirs": {
                            "type": "boolean",
                            "description": "Create missing parent directories before writing. Use false when parent directories must already exist."
                        }
                    },
                    "required": ["path", "contents", "create_dirs"],
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "edit".to_string(),
                description: "Replace one exact string in a UTF-8 text file.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute file path to edit."
                        },
                        "old": {
                            "type": "string",
                            "description": "Exact text to replace. Must occur once."
                        },
                        "new": {
                            "type": "string",
                            "description": "Replacement text."
                        }
                    },
                    "required": ["path", "old", "new"],
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "bash".to_string(),
                description: "Run a shell command and stream output to the terminal.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to run through the shell."
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "tool_search".to_string(),
                description: "Search tools exposed by configured MCP servers.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words to match against MCP server names, tool names, titles, and descriptions. Use an empty string to list the first tools."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of matches to return. Use 10 for the default."
                        }
                    },
                    "required": ["query", "limit"],
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "mcp_call".to_string(),
                description: "Call a tool exposed by a configured MCP server after finding it with tool_search.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "server": {
                            "type": "string",
                            "description": "MCP server name returned by tool_search."
                        },
                        "tool": {
                            "type": "string",
                            "description": "MCP tool name returned by tool_search."
                        },
                        "arguments": {
                            "type": "string",
                            "description": "JSON object string containing arguments for the MCP tool, matching the input_schema returned by tool_search."
                        }
                    },
                    "required": ["server", "tool", "arguments"],
                    "additionalProperties": false
                }),
            },
        ]
    }

    #[instrument(skip_all, fields(tool = %call.name, call_id = %call.call_id))]
    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        let result = match call.name.as_str() {
            "read" => fs::read_file(&self.workspace_root, call),
            "write" => fs::write_file(&self.workspace_root, call),
            "edit" => fs::edit_file(&self.workspace_root, call),
            "bash" => bash::run_command(&self.workspace_root, call),
            "tool_search" => self.search_mcp_tools(call),
            "mcp_call" => self.call_mcp_tool(call).await,
            _ => Err(anyhow::anyhow!("unknown tool `{}`", call.name)),
        };

        match result {
            Ok(result) => {
                info!(ok = result.ok, "tool completed");
                result
            }
            Err(error) => {
                tracing::warn!(error = %error, "tool failed");
                ToolResult {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    ok: false,
                    content: error.to_string(),
                    data: json!({ "error": error.to_string() }),
                }
            }
        }
    }

    fn search_mcp_tools(&self, call: &ToolCall) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct Args {
            query: String,
            limit: Option<usize>,
        }

        let args = parse_args::<Args>(call)?;
        let output = self.mcp.search(&args.query, args.limit);
        let data = serde_json::to_value(&output)?;

        Ok(ToolResult {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            ok: true,
            content: data.to_string(),
            data,
        })
    }

    async fn call_mcp_tool(&self, call: &ToolCall) -> Result<ToolResult> {
        #[derive(Deserialize)]
        struct Args {
            server: String,
            tool: String,
            arguments: Value,
        }

        let args = parse_args::<Args>(call)?;
        let arguments = parse_mcp_arguments(args.arguments)?;
        let output = self.mcp.call(&args.server, &args.tool, arguments).await?;

        Ok(ToolResult {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            ok: output.ok,
            content: output.content,
            data: output.data,
        })
    }
}

fn parse_mcp_arguments(arguments: Value) -> Result<Value> {
    let arguments = match arguments {
        Value::String(arguments) => serde_json::from_str(&arguments)
            .context("mcp_call arguments must be a JSON object string")?,
        arguments => arguments,
    };

    if !arguments.is_object() {
        bail!("mcp_call arguments must decode to a JSON object");
    }

    Ok(arguments)
}

pub fn resolve_path(workspace_root: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        workspace_root.join(path)
    }
}

pub(crate) fn parse_args<T>(call: &ToolCall) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(call.arguments.clone())
        .map_err(|error| anyhow::anyhow!("invalid arguments for `{}`: {error}", call.name))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn tool_definitions_are_strict_schema_compatible() {
        let tools = ToolRegistry::new(std::env::temp_dir()).definitions();

        for tool in tools {
            assert_strict_objects(&tool.name, &tool.parameters);

            let properties = tool.parameters["properties"]
                .as_object()
                .expect("tool properties");
            let required = tool.parameters["required"]
                .as_array()
                .expect("required tools");
            let required = required
                .iter()
                .map(|value| value.as_str().expect("required string"))
                .collect::<BTreeSet<_>>();
            let property_names = properties
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();

            assert_no_union_types(&tool.name, &tool.parameters);
            assert_eq!(
                required, property_names,
                "{} must require every declared property for strict mode",
                tool.name
            );
        }
    }

    #[test]
    fn parses_mcp_call_arguments_from_json_string() {
        let arguments =
            parse_mcp_arguments(json!("{\"pipeline\":\"ferrix\"}")).expect("parse arguments");

        assert_eq!(arguments["pipeline"], "ferrix");
    }

    #[test]
    fn rejects_non_object_mcp_call_arguments() {
        let error = parse_mcp_arguments(json!("[]")).expect_err("array rejected");

        assert!(error.to_string().contains("must decode to a JSON object"));
    }

    fn assert_strict_objects(tool_name: &str, schema: &Value) {
        let Some(object) = schema.as_object() else {
            return;
        };

        let schema_type = object.get("type");
        let is_object_schema = schema_type == Some(&json!("object"));
        if is_object_schema {
            assert_eq!(
                object.get("additionalProperties"),
                Some(&json!(false)),
                "{tool_name} object schemas must reject extra arguments"
            );
        }

        for value in object.values() {
            match value {
                Value::Array(values) => {
                    for value in values {
                        assert_strict_objects(tool_name, value);
                    }
                }
                Value::Object(_) => assert_strict_objects(tool_name, value),
                _ => {}
            }
        }
    }

    fn assert_no_union_types(tool_name: &str, schema: &Value) {
        match schema {
            Value::Object(object) => {
                assert!(
                    !matches!(object.get("type"), Some(Value::Array(_))),
                    "{tool_name} schemas should avoid union `type` arrays for provider compatibility"
                );

                for value in object.values() {
                    assert_no_union_types(tool_name, value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    assert_no_union_types(tool_name, value);
                }
            }
            _ => {}
        }
    }
}
