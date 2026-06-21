use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::json;
use tracing::{debug, info, instrument, warn};

use crate::model::{ConversationMessage, Model};
use crate::runs::RunRecorder;
use crate::tools::{ToolCall, ToolRegistry};

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
Prefer exact, minimal edits. The available tools are read, write, edit, and bash. \
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
        mut on_text_delta: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        let recorder = RunRecorder::new(&self.workspace_root)?;
        let metadata = self.model.metadata();

        recorder.record(
            "run_started",
            json!({
                "prompt": user_input,
                "model": metadata
            }),
        )?;

        info!(run_id = %recorder.run_id(), "agent run started");
        history.push(ConversationMessage::user(user_input));

        let tool_definitions = self.tools.definitions();
        let mut model_usage = ModelUsageTotals::default();

        for iteration in 1..=MAX_AGENT_ITERATIONS {
            debug!(iteration, "agent iteration started");
            recorder.record(
                "model_request",
                json!({
                    "iteration": iteration,
                    "message_count": history.len(),
                    "tool_count": tool_definitions.len()
                }),
            )?;

            let mut streamed_text = String::new();
            let model_started = Instant::now();
            let response = match self
                .model
                .complete_streaming(history, &tool_definitions, &mut |delta| {
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
                    "has_execution_plan": response.execution_plan.is_some()
                }),
            )?;

            if !response.tool_calls.is_empty() {
                history.push(ConversationMessage::assistant_tool_calls(
                    response.content.clone(),
                    response.tool_calls.clone(),
                    response.response_items.clone(),
                ));
                self.execute_tool_calls(&recorder, &response.tool_calls, history)?;
                continue;
            }

            let answer = response.content.unwrap_or_default();
            if !streamed_text.is_empty() {
                on_text_delta(&streamed_text)?;
            }
            history.push(ConversationMessage::assistant(answer.clone()));
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

    fn execute_tool_calls(
        &self,
        recorder: &RunRecorder,
        tool_calls: &[ToolCall],
        history: &mut Vec<ConversationMessage>,
    ) -> Result<()> {
        for call in tool_calls {
            recorder.record("tool_call", call)?;
            let result = self.tools.execute(call);

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
    use crate::model::{ModelMetadata, ModelResponse};
    use crate::tools::ToolDefinition;

    #[derive(Debug)]
    struct MockModel {
        responses: Mutex<VecDeque<ModelResponse>>,
    }

    impl MockModel {
        fn new(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().collect()),
            }
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

        async fn complete_streaming(
            &self,
            _messages: &[ConversationMessage],
            _tools: &[ToolDefinition],
            on_text_delta: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<ModelResponse> {
            let response = self
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
            },
            ModelResponse {
                content: Some("final answer".to_string()),
                tool_calls: Vec::new(),
                response_items: Vec::new(),
                usage: None,
                execution_plan: None,
            },
        ]);
        let tools = ToolRegistry::new(workspace.clone());
        let agent = Agent::new(model, tools, workspace.clone());
        let mut history = initial_history();
        let mut streamed = String::new();

        let answer = agent
            .run_turn("hello", &mut history, |delta| {
                streamed.push_str(delta);
                Ok(())
            })
            .await
            .expect("run turn");

        assert_eq!(answer, "final answer");
        assert_eq!(streamed, "final answer");

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
        });
        let model = MockModel::new(responses);
        let tools = ToolRegistry::new(workspace.clone());
        let agent = Agent::new(model, tools, workspace.clone());
        let mut history = initial_history();

        let error = agent
            .run_turn("hello", &mut history, |_delta| Ok(()))
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
}
