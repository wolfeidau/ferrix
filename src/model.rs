use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tracing::{debug, instrument};
use uuid::Uuid;

use crate::tools::{ToolCall, ToolDefinition};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ConversationMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelMetadata {
    pub provider: String,
    pub model: String,
    pub endpoint: String,
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub execution_plan: Option<Value>,
}

pub trait Model {
    fn metadata(&self) -> ModelMetadata;
    fn complete(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
    ) -> Result<ModelResponse>;
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleModel {
    client: Client,
    provider: String,
    model: String,
    endpoint: String,
    api_key: Option<String>,
}

impl OpenAiCompatibleModel {
    pub fn from_env() -> Result<Self> {
        let provider = std::env::var("FERRIX_MODEL_PROVIDER")
            .unwrap_or_else(|_| "openai-compatible".to_string());
        let model = std::env::var("FERRIX_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_string());
        let base_url = std::env::var("FERRIX_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let endpoint = chat_completions_endpoint(&base_url);
        let api_key = std::env::var("FERRIX_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .ok();

        Ok(Self {
            client: Client::builder()
                .build()
                .context("failed to build HTTP client")?,
            provider,
            model,
            endpoint,
            api_key,
        })
    }
}

impl Model for OpenAiCompatibleModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            provider: self.provider.clone(),
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
        }
    }

    #[instrument(skip_all, fields(model = %self.model, endpoint = %self.endpoint))]
    fn complete(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
    ) -> Result<ModelResponse> {
        let api_key = self
            .api_key
            .as_deref()
            .context("set FERRIX_API_KEY or OPENAI_API_KEY to call the model")?;

        let request = json!({
            "model": self.model,
            "messages": messages.iter().map(openai_message).collect::<Vec<_>>(),
            "tools": tools.iter().map(openai_tool).collect::<Vec<_>>(),
            "tool_choice": "auto"
        });

        debug!(
            message_count = messages.len(),
            tool_count = tools.len(),
            "sending model request"
        );

        let response: Value = self
            .client
            .post(&self.endpoint)
            .bearer_auth(api_key)
            .json(&request)
            .send()
            .context("model request failed")?
            .error_for_status()
            .context("model returned an error status")?
            .json()
            .context("failed to parse model response")?;

        parse_openai_response(response)
    }
}

fn chat_completions_endpoint(base_url: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    if base_url.ends_with("/chat/completions") {
        base_url.to_string()
    } else {
        format!("{base_url}/chat/completions")
    }
}

fn openai_message(message: &ConversationMessage) -> Value {
    let mut object = Map::new();
    object.insert("role".to_string(), json!(message.role));

    if let Some(content) = &message.content {
        object.insert("content".to_string(), json!(content));
    } else {
        object.insert("content".to_string(), Value::Null);
    }

    if !message.tool_calls.is_empty() {
        object.insert(
            "tool_calls".to_string(),
            Value::Array(message.tool_calls.iter().map(openai_tool_call).collect()),
        );
    }

    if let Some(tool_call_id) = &message.tool_call_id {
        object.insert("tool_call_id".to_string(), json!(tool_call_id));
    }

    Value::Object(object)
}

fn openai_tool_call(call: &ToolCall) -> Value {
    json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments.to_string()
        }
    })
}

fn openai_tool(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
    })
}

fn parse_openai_response(response: Value) -> Result<ModelResponse> {
    let execution_plan = extract_execution_plan(&response);
    let message = response
        .pointer("/choices/0/message")
        .context("model response did not include choices[0].message")?;

    let content = message.get("content").and_then(content_to_string);
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(parse_tool_call)
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();

    if content.is_none() && tool_calls.is_empty() {
        bail!("model response contained neither content nor tool calls");
    }

    Ok(ModelResponse {
        content,
        tool_calls,
        execution_plan,
    })
}

fn content_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(content) => Some(content.clone()),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .or_else(|| part.get("content").and_then(Value::as_str))
                })
                .collect::<Vec<_>>()
                .join("");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn parse_tool_call(value: &Value) -> Result<ToolCall> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let function = value
        .get("function")
        .context("tool call missing function payload")?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .context("tool call missing function name")?
        .to_string();
    let raw_arguments = function
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let arguments = serde_json::from_str(raw_arguments)
        .unwrap_or_else(|_| json!({ "raw_arguments": raw_arguments }));

    Ok(ToolCall {
        id,
        name,
        arguments,
    })
}

fn extract_execution_plan(value: &Value) -> Option<Value> {
    let mut plans = Vec::new();
    collect_execution_plans(value, &mut plans);

    match plans.len() {
        0 => None,
        1 => plans.pop(),
        _ => Some(Value::Array(plans)),
    }
}

fn collect_execution_plans(value: &Value, plans: &mut Vec<Value>) {
    match value {
        Value::Object(object) => {
            if let Some(plan) = object.get("execution_plan") {
                plans.push(plan.clone());
            }

            if object
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind == "execution_plan")
            {
                plans.push(Value::Object(object.clone()));
            }

            for value in object.values() {
                collect_execution_plans(value, plans);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_execution_plans(value, plans);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_tool_call_response() {
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": "{\"path\":\"src/main.rs\"}"
                        }
                    }]
                }
            }]
        });

        let parsed = parse_openai_response(response).expect("parse response");

        assert_eq!(parsed.tool_calls[0].name, "read");
        assert_eq!(parsed.tool_calls[0].arguments["path"], "src/main.rs");
    }

    #[test]
    fn extracts_execution_plan_payloads() {
        let response = json!({
            "execution_plan": { "steps": ["inspect", "edit"] },
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "done"
                }
            }]
        });

        let parsed = parse_openai_response(response).expect("parse response");

        assert_eq!(parsed.execution_plan.unwrap()["steps"][0], "inspect");
    }
}
