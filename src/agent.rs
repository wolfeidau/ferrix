use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::json;
use tracing::{debug, info, instrument, warn};

use crate::model::{ConversationMessage, Model, ModelTurnContext};
use crate::runs::RunRecorder;
use crate::tools::{ToolCall, ToolDefinition, ToolRegistry};

const MAX_AGENT_ITERATIONS: usize = 16;

#[derive(Debug, Default, serde::Serialize)]
struct ModelUsageTotals {
    model_calls: usize,
    duration_ms: u128,
    input_tokens: u64,
    output_tokens: u64,
    reasoning_tokens: u64,
    cached_input_tokens: u64,
    total_tokens: u64,
    missing_usage_iterations: Vec<usize>,
}

impl ModelUsageTotals {
    fn add_response(
        &mut self,
        iteration: usize,
        duration_ms: u128,
        usage: Option<&serde_json::Value>,
    ) {
        self.model_calls += 1;
        self.duration_ms += duration_ms;

        let Some(usage) = usage else {
            self.missing_usage_iterations.push(iteration);
            return;
        };

        self.input_tokens += usage_u64(usage, &["input_tokens"]);
        self.output_tokens += usage_u64(usage, &["output_tokens"]);
        self.total_tokens += usage_u64(usage, &["total_tokens"]);
        self.cached_input_tokens += usage_u64(usage, &["input_tokens_details", "cached_tokens"]);
        self.reasoning_tokens += usage_u64(usage, &["output_tokens_details", "reasoning_tokens"]);
    }
}

fn usage_u64(usage: &serde_json::Value, path: &[&str]) -> u64 {
    path.iter()
        .try_fold(usage, |value, key| value.get(*key))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default()
}

pub fn initial_history() -> Vec<ConversationMessage> {
    vec![ConversationMessage::system(
        "You are Ferrix, a small coding agent. \
Use tools when you need to inspect, modify, or run the project. \
Prefer exact, minimal edits. The available tools are read, write, edit, bash, tool_search, and mcp_call. \
Use tool_search before mcp_call when you need to discover tools exposed by configured MCP servers. \
When you are done, respond with a concise final answer.",
    )]
}

#[derive(Debug)]
pub struct Agent<M> {
    model: M,
    tools: ToolRegistry,
    workspace_root: PathBuf,
}

impl<M> Agent<M>
where
    M: Model,
{
    pub fn new(model: M, tools: ToolRegistry, workspace_root: PathBuf) -> Self {
        Self {
            model,
            tools,
            workspace_root,
        }
    }

    #[instrument(skip_all, fields(prompt_len = user_input.len()))]
    pub async fn run_turn(
        &self,
        user_input: &str,
        history: &mut Vec<ConversationMessage>,
        session_last_response_id: &mut Option<String>,
        mut on_text_delta: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        let recorder = RunRecorder::new(&self.workspace_root)?;
        let metadata = self.model.metadata();
        let prompt_cache = self.model.prompt_cache_settings();
        let store_responses = self.model.store_responses();

        recorder.record(
            "run_started",
            json!({
                "prompt": user_input,
                "model": metadata,
                "prompt_cache": prompt_cache
            }),
        )?;

        info!(run_id = %recorder.run_id(), "agent run started");
        history.push(ConversationMessage::user(user_input));

        let tool_definitions = self.tools.definitions();
        let mut model_usage = ModelUsageTotals::default();
        let mut previous_response_id = store_responses
            .then(|| session_last_response_id.clone())
            .flatten();
        let mut incremental_messages: Option<Vec<ConversationMessage>> = None;

        for iteration in 1..=MAX_AGENT_ITERATIONS {
            let incremental = incremental_messages.take();
            let messages = incremental.as_deref().unwrap_or(history.as_slice());
            let tools: &[ToolDefinition] = &tool_definitions;
            let context = ModelTurnContext {
                previous_response_id: previous_response_id.clone(),
                incremental: incremental.is_some(),
            };

            debug!(iteration, "agent iteration started");
            recorder.record(
                "model_request",
                json!({
                    "iteration": iteration,
                    "message_count": messages.len(),
                    "tool_count": tools.len(),
                    "incremental": context.incremental,
                    "previous_response_id": context.previous_response_id
                }),
            )?;

            let mut streamed_text = String::new();
            let model_started = Instant::now();
            let response = match self
                .model
                .complete_streaming(messages, tools, &context, &mut |delta| {
                    streamed_text.push_str(delta);
                    Ok(())
                })
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    recorder.record(
                        "run_completed",
                        json!({
                            "status": "failed",
                            "reason": "model completion failed",
                            "model_usage": model_usage
                        }),
                    )?;
                    return Err(error).context("model completion failed");
                }
            };
            let model_duration_ms = model_started.elapsed().as_millis();
            model_usage.add_response(iteration, model_duration_ms, response.usage.as_ref());

            if let Some(usage) = &response.usage {
                let cached_tokens = usage_u64(usage, &["input_tokens_details", "cached_tokens"]);
                if cached_tokens > 0 {
                    debug!(
                        iteration,
                        cached_tokens, "prompt cache hit reported by model"
                    );
                }
            }

            let response_id = response.response_id.clone();
            previous_response_id = store_responses.then(|| response_id.clone()).flatten();

            if let Some(plan) = &response.execution_plan {
                recorder.record(
                    "execution_plan",
                    json!({
                        "iteration": iteration,
                        "extra": plan
                    }),
                )?;
            }

            recorder.record(
                "model_response",
                json!({
                    "iteration": iteration,
                    "content": response.content.clone(),
                    "tool_call_count": response.tool_calls.len(),
                    "tool_calls": response.tool_calls.clone(),
                    "response_item_count": response.response_items.len(),
                    "response_items": response.response_items.clone(),
                    "duration_ms": model_duration_ms,
                    "usage": response.usage.clone(),
                    "response_id": response.response_id,
                    "has_execution_plan": response.execution_plan.is_some()
                }),
            )?;

            if !response.tool_calls.is_empty() {
                history.push(ConversationMessage::assistant_tool_calls(
                    response.content.clone(),
                    response.tool_calls.clone(),
                    response.response_items.clone(),
                ));
                let tool_results_start = history.len();
                self.execute_tool_calls(&recorder, &response.tool_calls, history)
                    .await?;
                if store_responses && response_id.is_some() {
                    incremental_messages = Some(history[tool_results_start..].to_vec());
                }
                continue;
            }

            let answer = response.content.unwrap_or_default();
            if !streamed_text.is_empty() {
                on_text_delta(&streamed_text)?;
            }
            history.push(ConversationMessage::assistant(answer.clone()));
            *session_last_response_id = store_responses.then_some(response.response_id).flatten();
            recorder.record(
                "run_completed",
                json!({
                    "status": "completed",
                    "answer": answer,
                    "model_usage": model_usage
                }),
            )?;
            info!(run_id = %recorder.run_id(), "agent run completed");
            return Ok(answer);
        }

        recorder.record(
            "run_completed",
            json!({
                "status": "failed",
                "reason": "maximum agent iterations reached",
                "model_usage": model_usage
            }),
        )?;
        bail!("agent reached the maximum of {MAX_AGENT_ITERATIONS} iterations");
    }

    async fn execute_tool_calls(
        &self,
        recorder: &RunRecorder,
        tool_calls: &[ToolCall],
        history: &mut Vec<ConversationMessage>,
    ) -> Result<()> {
        for call in tool_calls {
            recorder.record("tool_call", call)?;
            let result = self.tools.execute(call).await;

            if !result.ok {
                warn!(tool = %result.name, call_id = %result.call_id, "tool returned an error");
            }

            recorder.record("tool_result", &result)?;
            history.push(ConversationMessage::tool_result(
                result.call_id,
                serde_json::to_string(&result.data).unwrap_or(result.content),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::fs;
    use std::sync::Mutex;

    use pretty_assertions::assert_eq;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;
    use crate::model::{ModelMetadata, ModelResponse, ModelTurnContext, PromptCacheSettings};
    use crate::tools::ToolDefinition;

    use std::sync::Arc;

    #[derive(Debug, Clone)]
    struct MockModel {
        inner: Arc<MockModelInner>,
    }

    #[derive(Debug)]
    struct MockModelInner {
        responses: Mutex<VecDeque<ModelResponse>>,
        store_responses: bool,
        captured_requests: Mutex<Vec<CapturedRequest>>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CapturedRequest {
        message_count: usize,
        tool_count: usize,
        incremental: bool,
        previous_response_id: Option<String>,
    }

    impl MockModel {
        fn new(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
            Self {
                inner: Arc::new(MockModelInner {
                    responses: Mutex::new(responses.into_iter().collect()),
                    store_responses: false,
                    captured_requests: Mutex::new(Vec::new()),
                }),
            }
        }

        fn with_store_responses(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
            Self {
                inner: Arc::new(MockModelInner {
                    responses: Mutex::new(responses.into_iter().collect()),
                    store_responses: true,
                    captured_requests: Mutex::new(Vec::new()),
                }),
            }
        }

        fn captured_requests(&self) -> Vec<CapturedRequest> {
            self.inner
                .captured_requests
                .lock()
                .expect("lock captured requests")
                .clone()
        }
    }

    impl Model for MockModel {
        fn metadata(&self) -> ModelMetadata {
            ModelMetadata {
                provider: "test".to_string(),
                model: "test".to_string(),
                endpoint: "test".to_string(),
            }
        }

        fn prompt_cache_settings(&self) -> PromptCacheSettings {
            PromptCacheSettings {
                retention: None,
                key: None,
                store_responses: self.inner.store_responses,
            }
        }

        fn store_responses(&self) -> bool {
            self.inner.store_responses
        }

        async fn complete_streaming(
            &self,
            messages: &[ConversationMessage],
            tools: &[ToolDefinition],
            context: &ModelTurnContext,
            on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<ModelResponse> {
            self.inner
                .captured_requests
                .lock()
                .expect("lock captured requests")
                .push(CapturedRequest {
                    message_count: messages.len(),
                    tool_count: tools.len(),
                    incremental: context.incremental,
                    previous_response_id: context.previous_response_id.clone(),
                });

            let response = self
                .inner
                .responses
                .lock()
                .expect("lock responses")
                .pop_front()
                .expect("mock response");

            if let Some(content) = &response.content {
                on_text_delta(content)?;
            }

            Ok(response)
        }
    }

    #[tokio::test]
    async fn suppresses_streamed_text_from_tool_call_iterations() {
        let workspace = std::env::temp_dir().join(format!("ferrix-agent-{}", Uuid::new_v4()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let model = MockModel::new([
            ModelResponse {
                content: Some("tool preface".to_string()),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".to_string(),
                    item_id: Some("fc_1".to_string()),
                    name: "unknown".to_string(),
                    arguments: json!({}),
                }],
                response_items: vec![json!({
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "unknown",
                    "arguments": "{}"
                })],
                usage: Some(json!({
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": { "cached_tokens": 3 },
                    "output_tokens_details": { "reasoning_tokens": 4 }
                })),
                execution_plan: None,
                response_id: Some("resp_tool".to_string()),
            },
            ModelResponse {
                content: Some("final answer".to_string()),
                tool_calls: Vec::new(),
                response_items: Vec::new(),
                usage: None,
                execution_plan: None,
                response_id: Some("resp_final".to_string()),
            },
        ]);
        let tools = ToolRegistry::new(workspace.clone());
        let agent = Agent::new(model, tools, workspace.clone());
        let mut history = initial_history();
        let mut session_last_response_id = None;
        let mut streamed = String::new();

        let answer = agent
            .run_turn(
                "hello",
                &mut history,
                &mut session_last_response_id,
                |delta| {
                    streamed.push_str(delta);
                    Ok(())
                },
            )
            .await
            .expect("run turn");

        assert_eq!(answer, "final answer");
        assert_eq!(streamed, "final answer");
        assert_eq!(session_last_response_id, None);

        let runs_dir = workspace.join(".ferrix").join("runs");
        let run_file = fs::read_dir(&runs_dir)
            .expect("read runs dir")
            .next()
            .expect("run file")
            .expect("run file entry")
            .path();
        let run_events = fs::read_to_string(run_file).expect("read run events");
        let events = run_events
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("parse event"))
            .collect::<Vec<_>>();
        let started_event = events
            .iter()
            .find(|event| event["kind"] == "run_started")
            .expect("run started event");
        assert_eq!(
            started_event["payload"]["prompt_cache"]["store_responses"],
            false
        );

        let tool_response_event = events
            .iter()
            .find(|event| {
                event["kind"] == "model_response"
                    && event["payload"]["tool_call_count"].as_u64() == Some(1)
            })
            .expect("tool response event");

        assert_eq!(
            tool_response_event["payload"]["tool_calls"][0]["call_id"],
            "call_1"
        );
        assert_eq!(
            tool_response_event["payload"]["response_items"][0]["type"],
            "function_call"
        );
        assert!(
            tool_response_event["payload"]["duration_ms"]
                .as_u64()
                .is_some()
        );
        assert_eq!(tool_response_event["payload"]["usage"]["total_tokens"], 15);

        let completed_event = events
            .iter()
            .find(|event| event["kind"] == "run_completed")
            .expect("run completed event");
        assert_eq!(completed_event["payload"]["status"], "completed");
        assert_eq!(completed_event["payload"]["model_usage"]["model_calls"], 2);
        assert!(
            completed_event["payload"]["model_usage"]["duration_ms"]
                .as_u64()
                .is_some()
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["input_tokens"],
            10
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["output_tokens"],
            5
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["total_tokens"],
            15
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["cached_input_tokens"],
            3
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["reasoning_tokens"],
            4
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["missing_usage_iterations"],
            json!([2])
        );

        fs::remove_dir_all(workspace).expect("remove workspace");
    }

    #[tokio::test]
    async fn records_model_usage_totals_on_max_iteration_failure() {
        let workspace = std::env::temp_dir().join(format!("ferrix-agent-{}", Uuid::new_v4()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let responses = (1..=MAX_AGENT_ITERATIONS).map(|iteration| ModelResponse {
            content: Some(format!("tool preface {iteration}")),
            tool_calls: vec![ToolCall {
                call_id: format!("call_{iteration}"),
                item_id: Some(format!("fc_{iteration}")),
                name: "unknown".to_string(),
                arguments: json!({}),
            }],
            response_items: vec![json!({
                "type": "function_call",
                "id": format!("fc_{iteration}"),
                "call_id": format!("call_{iteration}"),
                "name": "unknown",
                "arguments": "{}"
            })],
            usage: Some(json!({
                "input_tokens": iteration,
                "output_tokens": 1,
                "total_tokens": iteration + 1
            })),
            execution_plan: None,
            response_id: None,
        });
        let model = MockModel::new(responses);
        let tools = ToolRegistry::new(workspace.clone());
        let agent = Agent::new(model, tools, workspace.clone());
        let mut history = initial_history();
        let mut session_last_response_id = None;

        let error = agent
            .run_turn(
                "hello",
                &mut history,
                &mut session_last_response_id,
                |_delta| Ok(()),
            )
            .await
            .expect_err("max iteration error");

        assert_eq!(
            error.to_string(),
            format!("agent reached the maximum of {MAX_AGENT_ITERATIONS} iterations")
        );

        let runs_dir = workspace.join(".ferrix").join("runs");
        let run_file = fs::read_dir(&runs_dir)
            .expect("read runs dir")
            .next()
            .expect("run file")
            .expect("run file entry")
            .path();
        let run_events = fs::read_to_string(run_file).expect("read run events");
        let completed_event = run_events
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("parse event"))
            .find(|event| event["kind"] == "run_completed")
            .expect("run completed event");

        assert_eq!(completed_event["payload"]["status"], "failed");
        assert_eq!(
            completed_event["payload"]["reason"],
            "maximum agent iterations reached"
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["model_calls"],
            MAX_AGENT_ITERATIONS
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["input_tokens"],
            136
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["output_tokens"],
            MAX_AGENT_ITERATIONS
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["total_tokens"],
            152
        );
        assert_eq!(
            completed_event["payload"]["model_usage"]["missing_usage_iterations"],
            json!([])
        );

        fs::remove_dir_all(workspace).expect("remove workspace");
    }

    #[tokio::test]
    async fn uses_incremental_messages_when_store_responses_enabled() {
        let workspace = std::env::temp_dir().join(format!("ferrix-agent-{}", Uuid::new_v4()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let model = MockModel::with_store_responses([
            ModelResponse {
                content: Some("tool preface".to_string()),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".to_string(),
                    item_id: Some("fc_1".to_string()),
                    name: "unknown".to_string(),
                    arguments: json!({}),
                }],
                response_items: Vec::new(),
                usage: None,
                execution_plan: None,
                response_id: Some("resp_tool".to_string()),
            },
            ModelResponse {
                content: Some("final answer".to_string()),
                tool_calls: Vec::new(),
                response_items: Vec::new(),
                usage: None,
                execution_plan: None,
                response_id: Some("resp_final".to_string()),
            },
        ]);
        let tools = ToolRegistry::new(workspace.clone());
        let agent = Agent::new(model.clone(), tools, workspace.clone());
        let mut history = initial_history();
        let mut session_last_response_id = None;

        let answer = agent
            .run_turn(
                "hello",
                &mut history,
                &mut session_last_response_id,
                |_delta| Ok(()),
            )
            .await
            .expect("run turn");

        assert_eq!(answer, "final answer");
        let captured = model.captured_requests();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].message_count, 2);
        assert!(captured[0].tool_count > 0);
        assert!(!captured[0].incremental);
        assert_eq!(captured[0].previous_response_id, None);
        assert_eq!(captured[1].message_count, 1);
        assert_eq!(captured[1].tool_count, captured[0].tool_count);
        assert!(captured[1].incremental);
        assert_eq!(
            captured[1].previous_response_id.as_deref(),
            Some("resp_tool")
        );

        fs::remove_dir_all(workspace).expect("remove workspace");
    }

    #[tokio::test]
    async fn does_not_chain_response_ids_when_store_responses_disabled() {
        let workspace = std::env::temp_dir().join(format!("ferrix-agent-{}", Uuid::new_v4()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let model = MockModel::new([
            ModelResponse {
                content: Some("first".to_string()),
                tool_calls: Vec::new(),
                response_items: Vec::new(),
                usage: None,
                execution_plan: None,
                response_id: Some("resp_first".to_string()),
            },
            ModelResponse {
                content: Some("second".to_string()),
                tool_calls: Vec::new(),
                response_items: Vec::new(),
                usage: None,
                execution_plan: None,
                response_id: Some("resp_second".to_string()),
            },
        ]);
        let tools = ToolRegistry::new(workspace.clone());
        let agent = Agent::new(model.clone(), tools, workspace.clone());
        let mut history = initial_history();
        let mut session_last_response_id = None;

        agent
            .run_turn(
                "one",
                &mut history,
                &mut session_last_response_id,
                |_delta| Ok(()),
            )
            .await
            .expect("first turn");
        agent
            .run_turn(
                "two",
                &mut history,
                &mut session_last_response_id,
                |_delta| Ok(()),
            )
            .await
            .expect("second turn");

        assert_eq!(session_last_response_id, None);
        assert_eq!(
            model
                .captured_requests()
                .into_iter()
                .map(|request| request.previous_response_id)
                .collect::<Vec<_>>(),
            vec![None, None]
        );

        fs::remove_dir_all(workspace).expect("remove workspace");
    }

    #[tokio::test]
    async fn falls_back_to_full_history_when_store_response_id_is_missing() {
        let workspace = std::env::temp_dir().join(format!("ferrix-agent-{}", Uuid::new_v4()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let model = MockModel::with_store_responses([
            ModelResponse {
                content: Some("tool preface".to_string()),
                tool_calls: vec![ToolCall {
                    call_id: "call_1".to_string(),
                    item_id: Some("fc_1".to_string()),
                    name: "unknown".to_string(),
                    arguments: json!({}),
                }],
                response_items: Vec::new(),
                usage: None,
                execution_plan: None,
                response_id: None,
            },
            ModelResponse {
                content: Some("final answer".to_string()),
                tool_calls: Vec::new(),
                response_items: Vec::new(),
                usage: None,
                execution_plan: None,
                response_id: Some("resp_final".to_string()),
            },
        ]);
        let tools = ToolRegistry::new(workspace.clone());
        let agent = Agent::new(model.clone(), tools, workspace.clone());
        let mut history = initial_history();
        let mut session_last_response_id = None;

        agent
            .run_turn(
                "hello",
                &mut history,
                &mut session_last_response_id,
                |_delta| Ok(()),
            )
            .await
            .expect("run turn");

        let captured = model.captured_requests();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].message_count, 2);
        assert!(captured[0].tool_count > 0);
        assert!(!captured[0].incremental);
        assert_eq!(captured[0].previous_response_id, None);
        assert_eq!(captured[1].message_count, 4);
        assert_eq!(captured[1].tool_count, captured[0].tool_count);
        assert!(!captured[1].incremental);
        assert_eq!(captured[1].previous_response_id, None);

        fs::remove_dir_all(workspace).expect("remove workspace");
    }
}
