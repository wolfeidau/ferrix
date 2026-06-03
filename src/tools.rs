pub mod bash;
pub mod fs;

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{info, instrument};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
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

#[derive(Debug, Clone)]
pub struct ToolRegistry {
    workspace_root: PathBuf,
}

impl ToolRegistry {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
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
                            "description": "Create missing parent directories before writing.",
                            "default": false
                        }
                    },
                    "required": ["path", "contents"],
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
        ]
    }

    #[instrument(skip_all, fields(tool = %call.name, call_id = %call.id))]
    pub fn execute(&self, call: &ToolCall) -> ToolResult {
        let result = match call.name.as_str() {
            "read" => fs::read_file(&self.workspace_root, call),
            "write" => fs::write_file(&self.workspace_root, call),
            "edit" => fs::edit_file(&self.workspace_root, call),
            "bash" => bash::run_command(&self.workspace_root, call),
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
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    ok: false,
                    content: error.to_string(),
                    data: json!({ "error": error.to_string() }),
                }
            }
        }
    }
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
