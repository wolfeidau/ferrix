use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde_json::json;
use tracing::{debug, info, instrument, warn};

use crate::model::{ConversationMessage, Model};
use crate::runs::RunRecorder;
use crate::tools::{ToolCall, ToolRegistry};

const MAX_AGENT_ITERATIONS: usize = 16;

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

            let response = self
                .model
                .complete_streaming(history, &tool_definitions, &mut on_text_delta)
                .await
                .context("model completion failed")?;

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
                    "has_execution_plan": response.execution_plan.is_some()
                }),
            )?;

            if !response.tool_calls.is_empty() {
                history.push(ConversationMessage::assistant_tool_calls(
                    response.content.clone(),
                    response.tool_calls.clone(),
                ));
                self.execute_tool_calls(&recorder, &response.tool_calls, history)?;
                continue;
            }

            let answer = response.content.unwrap_or_default();
            history.push(ConversationMessage::assistant(answer.clone()));
            recorder.record(
                "run_completed",
                json!({
                    "status": "completed",
                    "answer": answer
                }),
            )?;
            info!(run_id = %recorder.run_id(), "agent run completed");
            return Ok(answer);
        }

        recorder.record(
            "run_completed",
            json!({
                "status": "failed",
                "reason": "maximum agent iterations reached"
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
