use std::path::Path;

use anyhow::{Context, Result, bail};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::responses::{
        CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput,
        FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, InputItem, InputParam, Item,
        OutputItem, OutputStatus, PromptCacheRetention, Reasoning, ReasoningEffort,
        ResponseStreamEvent, Role, Tool, ToolChoiceOptions, ToolChoiceParam,
    },
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{debug, instrument};
use uuid::Uuid;

use crate::config::ModelConfig;
use crate::tools::{ToolCall, ToolDefinition};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub response_items: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ConversationMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            response_items: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            response_items: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            response_items: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant_tool_calls(
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
        response_items: Vec<Value>,
    ) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            tool_calls,
            response_items,
            tool_call_id: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            response_items: Vec::new(),
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

#[derive(Debug, Clone, Serialize)]
pub struct PromptCacheSettings {
    pub retention: Option<String>,
    pub key: Option<String>,
    pub store_responses: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ModelTurnContext {
    pub previous_response_id: Option<String>,
    pub incremental: bool,
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub response_items: Vec<Value>,
    pub usage: Option<Value>,
    pub execution_plan: Option<Value>,
    pub response_id: Option<String>,
}

pub trait Model {
    fn metadata(&self) -> ModelMetadata;

    fn prompt_cache_settings(&self) -> PromptCacheSettings;

    fn store_responses(&self) -> bool;

    async fn complete_streaming(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        context: &ModelTurnContext,
        on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<ModelResponse>;
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleModel {
    client: Client<OpenAIConfig>,
    provider: String,
    model: String,
    api_base: String,
    endpoint: String,
    api_key: Option<String>,
    api_key_env_var: &'static str,
    reasoning_effort: Option<ReasoningEffort>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    prompt_cache_retention: Option<PromptCacheRetention>,
    prompt_cache_key: Option<String>,
    store_responses: bool,
}

impl OpenAiCompatibleModel {
    #[allow(dead_code)]
    pub fn from_workspace(workspace_root: &Path) -> Result<Self> {
        Self::from_workspace_with_cache_key(workspace_root, None)
    }

    pub fn from_workspace_with_cache_key(
        workspace_root: &Path,
        session_cache_key: Option<String>,
    ) -> Result<Self> {
        let model_config =
            ModelConfig::from_workspace(workspace_root)?.with_prompt_cache_key(session_cache_key);
        Self::from_config(model_config)
    }

    fn from_config(model_config: ModelConfig) -> Result<Self> {
        let config = openai_config(&model_config)?;
        let prompt_cache = model_config.prompt_cache;

        Ok(Self {
            client: Client::with_config(config),
            provider: model_config.provider,
            model: model_config.model,
            api_base: model_config.api_base,
            endpoint: model_config.endpoint,
            api_key: model_config.api_key,
            api_key_env_var: model_config.api_key_env_var,
            reasoning_effort: model_config.reasoning_effort,
            max_output_tokens: model_config.max_output_tokens,
            temperature: model_config.temperature,
            top_p: model_config.top_p,
            prompt_cache_retention: prompt_cache.retention,
            prompt_cache_key: prompt_cache.key,
            store_responses: prompt_cache.store_responses,
        })
    }

    fn response_request(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        context: &ModelTurnContext,
    ) -> Result<CreateResponse> {
        Ok(CreateResponse {
            model: Some(self.model.clone()),
            input: InputParam::Items(response_input_items(messages)?),
            tools: (!tools.is_empty()).then(|| tools.iter().map(response_tool).collect()),
            tool_choice: (!tools.is_empty())
                .then_some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)),
            stream: Some(true),
            store: Some(self.store_responses),
            previous_response_id: context.previous_response_id.clone(),
            prompt_cache_key: self.prompt_cache_key.clone(),
            prompt_cache_retention: self.prompt_cache_retention,
            max_output_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            reasoning: self.reasoning_effort.clone().map(|effort| Reasoning {
                effort: Some(effort),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    fn prompt_cache_settings(&self) -> PromptCacheSettings {
        PromptCacheSettings {
            retention: self
                .prompt_cache_retention
                .map(prompt_cache_retention_label),
            key: self.prompt_cache_key.clone(),
            store_responses: self.store_responses,
        }
    }
}

fn prompt_cache_retention_label(retention: PromptCacheRetention) -> String {
    match retention {
        PromptCacheRetention::InMemory => "in_memory".to_string(),
        PromptCacheRetention::Hours24 => "24h".to_string(),
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

    fn prompt_cache_settings(&self) -> PromptCacheSettings {
        self.prompt_cache_settings()
    }

    fn store_responses(&self) -> bool {
        self.store_responses
    }

    #[instrument(skip_all, fields(model = %self.model, api_base = %self.api_base))]
    async fn complete_streaming(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        context: &ModelTurnContext,
        on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<ModelResponse> {
        self.api_key
            .as_deref()
            .with_context(|| format!("set {} to call the model", self.api_key_env_var))?;

        let request = self.response_request(messages, tools, context)?;

        debug!(
            message_count = messages.len(),
            tool_count = tools.len(),
            incremental = context.incremental,
            previous_response_id = ?context.previous_response_id,
            "sending streaming responses request"
        );

        let mut stream = self
            .client
            .responses()
            .create_stream(request)
            .await
            .context("model request failed")?;
        let mut accumulator = StreamAccumulator::default();

        while let Some(event) = stream.next().await {
            let event = event.context("failed to read model response stream")?;
            if let Ok(value) = serde_json::to_value(&event) {
                collect_execution_plans(&value, &mut accumulator.execution_plans);
            }
            accumulator.apply_event(event, on_text_delta)?;
        }

        accumulator.finish()
    }
}

fn openai_config(config: &ModelConfig) -> Result<OpenAIConfig> {
    let mut openai_config = OpenAIConfig::new()
        .with_api_base(config.api_base.clone())
        .with_api_key(config.api_key.clone().unwrap_or_default());

    if let Some(referer) = &config.openrouter.referer {
        openai_config = openai_config
            .with_header("HTTP-Referer", referer)
            .context("invalid OpenRouter referer header")?;
    }
    if let Some(title) = &config.openrouter.title {
        openai_config = openai_config
            .with_header("X-OpenRouter-Title", title)
            .context("invalid OpenRouter title header")?;
    }
    if let Some(categories) = &config.openrouter.categories {
        openai_config = openai_config
            .with_header("X-OpenRouter-Categories", categories)
            .context("invalid OpenRouter categories header")?;
    }

    Ok(openai_config)
}

fn response_input_items(messages: &[ConversationMessage]) -> Result<Vec<InputItem>> {
    let mut items = Vec::new();
    for message in messages {
        items.extend(response_input_items_for_message(message)?);
    }
    Ok(items)
}

fn response_input_items_for_message(message: &ConversationMessage) -> Result<Vec<InputItem>> {
    match message.role.as_str() {
        "system" => Ok(vec![easy_message(
            Role::System,
            message.content.clone().unwrap_or_default(),
        )]),
        "user" => Ok(vec![easy_message(
            Role::User,
            message.content.clone().unwrap_or_default(),
        )]),
        "assistant" => {
            let mut items = Vec::new();
            if let Some(content) = &message.content
                && !content.is_empty()
            {
                items.push(easy_message(Role::Assistant, content.clone()));
            }

            if !message.tool_calls.is_empty() {
                items.extend(
                    message
                        .tool_calls
                        .iter()
                        .map(response_function_call_item)
                        .map(InputItem::Item),
                );
            } else if !message.response_items.is_empty() {
                items.extend(
                    message
                        .response_items
                        .iter()
                        .cloned()
                        .map(serde_json::from_value::<Item>)
                        .collect::<Result<Vec<_>, _>>()
                        .context("failed to parse stored response item")?
                        .into_iter()
                        .map(InputItem::Item),
                );
            }

            Ok(items)
        }
        "tool" => Ok(vec![InputItem::Item(Item::FunctionCallOutput(
            FunctionCallOutputItemParam {
                call_id: message
                    .tool_call_id
                    .clone()
                    .context("tool message missing tool_call_id")?,
                output: FunctionCallOutput::Text(message.content.clone().unwrap_or_default()),
                id: None,
                status: Some(OutputStatus::Completed),
            },
        ))]),
        role => bail!("unsupported conversation role `{role}`"),
    }
}

fn easy_message(role: Role, content: String) -> InputItem {
    InputItem::EasyMessage(EasyInputMessage {
        role,
        content: EasyInputContent::Text(content),
        phase: None,
        ..Default::default()
    })
}

fn response_function_call_item(call: &ToolCall) -> Item {
    Item::FunctionCall(FunctionToolCall {
        call_id: call.call_id.clone(),
        id: call.item_id.clone(),
        name: call.name.clone(),
        namespace: None,
        arguments: call.arguments.to_string(),
        status: Some(OutputStatus::Completed),
    })
}

fn response_tool(tool: &ToolDefinition) -> Tool {
    Tool::Function(FunctionTool {
        name: tool.name.clone(),
        description: Some(tool.description.clone()),
        parameters: Some(tool.parameters.clone()),
        strict: Some(true),
        defer_loading: None,
    })
}

#[derive(Default)]
struct StreamAccumulator {
    content: String,
    tool_calls: Vec<PartialToolCall>,
    response_items: Vec<Value>,
    usage: Option<Value>,
    execution_plans: Vec<Value>,
    response_id: Option<String>,
}

impl StreamAccumulator {
    fn apply_event(
        &mut self,
        event: ResponseStreamEvent,
        on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<()> {
        match event {
            ResponseStreamEvent::ResponseOutputTextDelta(event) => {
                on_text_delta(&event.delta)?;
                self.content.push_str(&event.delta);
            }
            ResponseStreamEvent::ResponseOutputTextDone(event) => {
                self.content = event.text;
            }
            ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(event) => {
                self.partial_tool_call(event.output_index)
                    .arguments
                    .push_str(&event.delta);
            }
            ResponseStreamEvent::ResponseFunctionCallArgumentsDone(event) => {
                let tool_call = self.partial_tool_call(event.output_index);
                if let Some(name) = event.name {
                    tool_call.name = name;
                }
                tool_call.item_id.get_or_insert(event.item_id);
                tool_call.arguments = event.arguments;
            }
            ResponseStreamEvent::ResponseOutputItemAdded(event) => {
                self.apply_output_item(event.output_index, event.item);
            }
            ResponseStreamEvent::ResponseOutputItemDone(event) => {
                self.apply_output_item(event.output_index, event.item);
            }
            ResponseStreamEvent::ResponseCompleted(event) => {
                self.response_id = Some(event.response.id.clone());
                if let Some(usage) = event.response.usage {
                    self.set_usage(usage);
                }
            }
            ResponseStreamEvent::ResponseFailed(event) => {
                bail!("model response failed: {:?}", event.response.error);
            }
            ResponseStreamEvent::ResponseIncomplete(event) => {
                bail!(
                    "model response incomplete: {:?}",
                    event.response.incomplete_details
                );
            }
            ResponseStreamEvent::ResponseError(event) => {
                bail!("model stream error: {}", event.message);
            }
            _ => {}
        }

        Ok(())
    }

    fn apply_output_item(&mut self, output_index: u32, item: OutputItem) {
        if matches!(item, OutputItem::FunctionCall(_) | OutputItem::Reasoning(_))
            && let Ok(value) = serde_json::to_value(&item)
        {
            self.store_response_item(output_index, response_item_for_input(value));
        }

        if let OutputItem::FunctionCall(call) = item {
            let tool_call = self.partial_tool_call(output_index);
            tool_call.call_id.get_or_insert(call.call_id);
            tool_call.item_id = call.id;
            tool_call.name = call.name;
            tool_call.arguments = call.arguments;
        }
    }

    fn store_response_item(&mut self, output_index: u32, item: Value) {
        let index = output_index as usize;
        if self.response_items.len() <= index {
            self.response_items.resize(index + 1, Value::Null);
        }
        self.response_items[index] = item;
    }

    fn set_usage(&mut self, usage: impl Serialize) {
        self.usage = serde_json::to_value(usage).ok();
    }

    fn partial_tool_call(&mut self, output_index: u32) -> &mut PartialToolCall {
        let index = output_index as usize;
        if self.tool_calls.len() <= index {
            self.tool_calls
                .resize_with(index + 1, PartialToolCall::default);
        }
        &mut self.tool_calls[index]
    }

    fn finish(self) -> Result<ModelResponse> {
        let content = (!self.content.is_empty()).then_some(self.content);
        let tool_calls = self
            .tool_calls
            .into_iter()
            .filter(|call| !call.name.is_empty())
            .map(PartialToolCall::finish)
            .collect::<Result<Vec<_>>>()?;

        if content.is_none() && tool_calls.is_empty() {
            bail!("model response contained neither content nor tool calls");
        }

        Ok(ModelResponse {
            content,
            tool_calls,
            response_items: self
                .response_items
                .into_iter()
                .filter(|item| !item.is_null())
                .collect(),
            usage: self.usage,
            execution_plan: execution_plan_from_many(self.execution_plans),
            response_id: self.response_id,
        })
    }
}

#[derive(Default)]
struct PartialToolCall {
    call_id: Option<String>,
    item_id: Option<String>,
    name: String,
    arguments: String,
}

impl PartialToolCall {
    fn finish(self) -> Result<ToolCall> {
        let arguments = if self.arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&self.arguments)
                .unwrap_or_else(|_| json!({ "raw_arguments": self.arguments }))
        };

        Ok(ToolCall {
            call_id: self.call_id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            item_id: self.item_id,
            name: self.name,
            arguments,
        })
    }
}

fn execution_plan_from_many(mut plans: Vec<Value>) -> Option<Value> {
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

fn response_item_for_input(mut item: Value) -> Value {
    if let Value::Object(object) = &mut item {
        object.remove("id");
    }
    item
}

#[cfg(test)]
mod tests {
    use async_openai::config::Config as _;
    use async_openai::types::responses::{
        FunctionToolCall, ResponseFunctionCallArgumentsDeltaEvent,
        ResponseFunctionCallArgumentsDoneEvent, ResponseTextDeltaEvent, ResponseTextDoneEvent,
    };
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use crate::config::{OpenRouterConfig, PromptCacheConfig};

    use super::*;

    #[test]
    fn response_request_sets_configured_request_options() {
        let model = test_model();
        let model = OpenAiCompatibleModel {
            max_output_tokens: Some(4096),
            temperature: Some(0.2),
            top_p: Some(0.9),
            ..model
        };

        let request = model
            .response_request(
                &[ConversationMessage::user("hello")],
                &[],
                &ModelTurnContext::default(),
            )
            .expect("build request");

        assert_eq!(request.max_output_tokens, Some(4096));
        assert_eq!(request.temperature, Some(0.2));
        assert_eq!(request.top_p, Some(0.9));
    }

    #[tokio::test]
    async fn missing_provider_api_key_reports_expected_env_var() {
        let model = OpenAiCompatibleModel {
            api_key: None,
            api_key_env_var: "OPENROUTER_API_KEY",
            ..test_model()
        };

        let error = model
            .complete_streaming(
                &[ConversationMessage::user("hello")],
                &[],
                &ModelTurnContext::default(),
                &mut |_| Ok(()),
            )
            .await
            .expect_err("missing key should fail");

        assert!(error.to_string().contains("OPENROUTER_API_KEY"));
    }

    #[test]
    fn openrouter_headers_are_added_when_configured() {
        let config = ModelConfig {
            provider: "openrouter".to_string(),
            model: "openai/gpt-5.2".to_string(),
            api_base: "https://openrouter.ai/api/v1".to_string(),
            endpoint: "https://openrouter.ai/api/v1/responses".to_string(),
            api_key: Some("sk-or".to_string()),
            api_key_env_var: "OPENROUTER_API_KEY",
            reasoning_effort: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            openrouter: OpenRouterConfig {
                referer: Some("https://example.com".to_string()),
                title: Some("Ferrix".to_string()),
                categories: Some("cli-agent".to_string()),
            },
            prompt_cache: PromptCacheConfig::default(),
        };

        let headers = openai_config(&config).expect("build config").headers();

        assert_eq!(
            headers
                .get("HTTP-Referer")
                .expect("referer header")
                .to_str()
                .unwrap(),
            "https://example.com"
        );
        assert_eq!(
            headers
                .get("X-OpenRouter-Title")
                .expect("title header")
                .to_str()
                .unwrap(),
            "Ferrix"
        );
        assert_eq!(
            headers
                .get("X-OpenRouter-Categories")
                .expect("categories header")
                .to_str()
                .unwrap(),
            "cli-agent"
        );
    }

    #[test]
    fn response_request_sets_prompt_cache_options() {
        let model = OpenAiCompatibleModel {
            prompt_cache_retention: Some(PromptCacheRetention::Hours24),
            prompt_cache_key: Some("session-key".to_string()),
            store_responses: true,
            ..test_model()
        };

        let request = model
            .response_request(
                &[ConversationMessage::user("hello")],
                &[],
                &ModelTurnContext {
                    previous_response_id: Some("resp_prev".to_string()),
                    incremental: true,
                },
            )
            .expect("build request");

        assert_eq!(
            request.prompt_cache_retention,
            Some(PromptCacheRetention::Hours24)
        );
        assert_eq!(request.prompt_cache_key.as_deref(), Some("session-key"));
        assert_eq!(request.store, Some(true));
        assert_eq!(request.previous_response_id.as_deref(), Some("resp_prev"));
    }

    #[test]
    fn response_request_disables_remote_storage_by_default() {
        let model = test_model();

        let request = model
            .response_request(
                &[ConversationMessage::user("hello")],
                &[],
                &ModelTurnContext::default(),
            )
            .expect("build request");

        assert_eq!(request.stream, Some(true));
        assert_eq!(request.store, Some(false));
    }

    #[test]
    fn response_request_sets_configured_reasoning_effort() {
        let model = OpenAiCompatibleModel {
            reasoning_effort: Some(ReasoningEffort::Low),
            ..test_model()
        };

        let request = model
            .response_request(
                &[ConversationMessage::user("hello")],
                &[],
                &ModelTurnContext::default(),
            )
            .expect("build request");

        assert_eq!(
            request.reasoning.and_then(|reasoning| reasoning.effort),
            Some(ReasoningEffort::Low)
        );
    }

    #[test]
    fn response_request_enables_strict_function_tools() {
        let model = test_model();
        let tools = vec![ToolDefinition {
            name: "read".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }];

        let request = model
            .response_request(
                &[ConversationMessage::user("hello")],
                &tools,
                &ModelTurnContext::default(),
            )
            .expect("build request");
        let tools = request.tools.expect("tools");

        assert!(matches!(
            &tools[0],
            Tool::Function(tool) if tool.strict == Some(true)
        ));
    }

    #[test]
    fn strips_response_item_ids_before_replay() {
        let item = response_item_for_input(json!({
            "type": "reasoning",
            "id": "rs_123",
            "summary": []
        }));

        assert_eq!(item["type"], "reasoning");
        assert!(item.get("id").is_none());
    }

    #[test]
    fn includes_usage_from_completed_stream() {
        let mut accumulator = StreamAccumulator::default();
        accumulator
            .apply_event(text_delta_event("done"), &mut |_| Ok(()))
            .expect("text delta");
        accumulator.set_usage(json!({
            "input_tokens": 12,
            "input_tokens_details": { "cached_tokens": 3 },
            "output_tokens": 7,
            "output_tokens_details": { "reasoning_tokens": 4 },
            "total_tokens": 19
        }));

        let response = accumulator.finish().expect("finish response");

        assert_eq!(response.usage.unwrap()["total_tokens"], 19);
    }

    #[test]
    fn builds_response_input_items_for_tool_round() {
        let messages = vec![
            ConversationMessage::assistant_tool_calls(
                None,
                vec![ToolCall {
                    call_id: "call_1".to_string(),
                    item_id: Some("fc_1".to_string()),
                    name: "read".to_string(),
                    arguments: json!({ "path": "src/main.rs" }),
                }],
                Vec::new(),
            ),
            ConversationMessage::tool_result("call_1", "{\"ok\":true}"),
        ];

        let items = response_input_items(&messages).expect("convert messages");

        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], InputItem::Item(Item::FunctionCall(_))));
        assert!(matches!(
            items[1],
            InputItem::Item(Item::FunctionCallOutput(_))
        ));
    }

    #[test]
    fn replays_tool_calls_with_item_ids_before_tool_results() {
        let messages = vec![
            ConversationMessage::assistant_tool_calls(
                None,
                vec![ToolCall {
                    call_id: "call_1".to_string(),
                    item_id: Some("fc_1".to_string()),
                    name: "tool_search".to_string(),
                    arguments: json!({ "query": "buildkite", "limit": 10 }),
                }],
                vec![json!({
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "tool_search",
                    "arguments": "{\"query\":\"buildkite\",\"limit\":10}"
                })],
            ),
            ConversationMessage::tool_result("call_1", "{\"ok\":true}"),
        ];

        let items = response_input_items(&messages).expect("convert messages");

        assert_eq!(items.len(), 2);
        let InputItem::Item(Item::FunctionCall(call)) = &items[0] else {
            panic!("first item should be a function call");
        };
        assert_eq!(call.id.as_deref(), Some("fc_1"));
        assert_eq!(call.call_id, "call_1");
        assert!(matches!(
            items[1],
            InputItem::Item(Item::FunctionCallOutput(_))
        ));
    }

    #[test]
    fn assembles_streamed_text_response() {
        let mut streamed = String::new();
        let mut accumulator = StreamAccumulator::default();

        accumulator
            .apply_event(text_delta_event("hel"), &mut |delta| {
                streamed.push_str(delta);
                Ok(())
            })
            .expect("first chunk");
        accumulator
            .apply_event(text_delta_event("lo"), &mut |delta| {
                streamed.push_str(delta);
                Ok(())
            })
            .expect("second chunk");
        accumulator
            .apply_event(text_done_event("hello"), &mut |_| Ok(()))
            .expect("done chunk");

        let response = accumulator.finish().expect("finish response");

        assert_eq!(streamed, "hello");
        assert_eq!(response.content.as_deref(), Some("hello"));
        assert!(response.tool_calls.is_empty());
    }

    #[test]
    fn assembles_streamed_tool_call_response() {
        let mut accumulator = StreamAccumulator::default();
        accumulator
            .apply_event(
                ResponseStreamEvent::ResponseOutputItemAdded(
                    async_openai::types::responses::ResponseOutputItemAddedEvent {
                        sequence_number: 1,
                        output_index: 0,
                        item: OutputItem::FunctionCall(FunctionToolCall {
                            arguments: String::new(),
                            call_id: "call_1".to_string(),
                            namespace: None,
                            name: "read".to_string(),
                            id: Some("item_1".to_string()),
                            status: Some(OutputStatus::InProgress),
                        }),
                    },
                ),
                &mut |_| Ok(()),
            )
            .expect("item added");
        accumulator
            .apply_event(function_args_delta("{\"path\""), &mut |_| Ok(()))
            .expect("argument delta");
        accumulator
            .apply_event(
                function_args_done("{\"path\":\"src/main.rs\"}"),
                &mut |_| Ok(()),
            )
            .expect("arguments done");

        let response = accumulator.finish().expect("finish response");

        assert_eq!(response.tool_calls[0].call_id, "call_1");
        assert_eq!(response.tool_calls[0].item_id.as_deref(), Some("item_1"));
        assert_eq!(response.tool_calls[0].name, "read");
        assert_eq!(response.tool_calls[0].arguments["path"], "src/main.rs");
        assert_eq!(response.response_items.len(), 1);
        assert_eq!(response.response_items[0]["type"], "function_call");
        assert_eq!(response.response_items[0]["call_id"], "call_1");
        assert!(response.response_items[0].get("id").is_none());
    }

    #[test]
    fn extracts_execution_plan_payloads() {
        let response = json!({
            "execution_plan": { "steps": ["inspect", "edit"] },
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "done"
                }]
            }]
        });

        let mut plans = Vec::new();
        collect_execution_plans(&response, &mut plans);
        let plan = execution_plan_from_many(plans).expect("extract plan");

        assert_eq!(plan["steps"][0], "inspect");
    }

    fn text_delta_event(delta: &str) -> ResponseStreamEvent {
        ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
            sequence_number: 1,
            item_id: "msg_1".to_string(),
            output_index: 0,
            content_index: 0,
            delta: delta.to_string(),
            logprobs: None,
        })
    }

    fn text_done_event(text: &str) -> ResponseStreamEvent {
        ResponseStreamEvent::ResponseOutputTextDone(ResponseTextDoneEvent {
            sequence_number: 1,
            item_id: "msg_1".to_string(),
            output_index: 0,
            content_index: 0,
            text: text.to_string(),
            logprobs: None,
        })
    }

    fn function_args_delta(delta: &str) -> ResponseStreamEvent {
        ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(
            ResponseFunctionCallArgumentsDeltaEvent {
                sequence_number: 1,
                item_id: "item_1".to_string(),
                output_index: 0,
                delta: delta.to_string(),
            },
        )
    }

    fn function_args_done(arguments: &str) -> ResponseStreamEvent {
        ResponseStreamEvent::ResponseFunctionCallArgumentsDone(
            ResponseFunctionCallArgumentsDoneEvent {
                name: Some("read".to_string()),
                sequence_number: 2,
                item_id: "item_1".to_string(),
                output_index: 0,
                arguments: arguments.to_string(),
            },
        )
    }

    #[test]
    fn captures_response_id_from_completed_stream() {
        let mut accumulator = StreamAccumulator::default();
        accumulator
            .apply_event(text_delta_event("hello"), &mut |_| Ok(()))
            .expect("text delta");
        let event: ResponseStreamEvent = serde_json::from_value(json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_123",
                "created_at": 0,
                "model": "gpt-test",
                "object": "response",
                "output": [],
                "status": "completed"
            }
        }))
        .expect("completed event");
        accumulator
            .apply_event(event, &mut |_| Ok(()))
            .expect("completed event");

        let response = accumulator.finish().expect("finish response");

        assert_eq!(response.response_id.as_deref(), Some("resp_123"));
    }

    fn test_model() -> OpenAiCompatibleModel {
        OpenAiCompatibleModel {
            client: Client::with_config(OpenAIConfig::new().with_api_key("test")),
            provider: "test".to_string(),
            model: "gpt-test".to_string(),
            api_base: "https://api.openai.com/v1".to_string(),
            endpoint: "https://api.openai.com/v1/responses".to_string(),
            api_key: Some("test".to_string()),
            api_key_env_var: "OPENAI_API_KEY",
            reasoning_effort: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            prompt_cache_retention: None,
            prompt_cache_key: None,
            store_responses: false,
        }
    }
}
