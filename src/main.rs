mod agent;
mod config;
mod logging;
mod mcp;
mod model;
mod runs;
mod terminal;
mod tools;

use std::cell::RefCell;
use std::io;

use agent::{Agent, RunTurnError, SessionStats};
use anyhow::Context;
use config::UiConfig;
use mcp::McpRegistry;
use model::OpenAiCompatibleModel;
use terminal::TerminalUi;
use tools::ToolRegistry;
use uuid::Uuid;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    logging::init();

    let workspace_root = std::env::current_dir().context("failed to determine workspace root")?;
    let ui_config = UiConfig::from_workspace(&workspace_root)?;
    let session_cache_key = Some(format!("ferrix-{}", Uuid::new_v4()));
    let model =
        OpenAiCompatibleModel::from_workspace_with_cache_key(&workspace_root, session_cache_key)?;
    let mcp = McpRegistry::from_workspace(&workspace_root).await?;
    let tools = ToolRegistry::with_mcp(workspace_root.clone(), mcp);
    let agent = Agent::new(model, tools, workspace_root);
    let mut history = agent::initial_history();
    let mut session_last_response_id = None;
    let terminal = RefCell::new(TerminalUi::stdout(ui_config));
    let mut session_stats = SessionStats::default();

    let stdin = io::stdin();

    loop {
        terminal.borrow_mut().write_prompt()?;

        let mut input = String::new();
        let bytes_read = stdin
            .read_line(&mut input)
            .context("failed to read user input")?;

        if bytes_read == 0 {
            terminal.borrow_mut().finish_answer()?;
            break;
        }

        let input = input.trim();
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }

        if input.is_empty() {
            continue;
        }

        let mut streamed_answer = false;
        match agent
            .run_turn(
                input,
                &mut history,
                &mut session_last_response_id,
                |delta| {
                    streamed_answer = true;
                    terminal.borrow_mut().write_answer_delta(delta)
                },
                |event| terminal.borrow_mut().handle_agent_event(&event),
            )
            .await
        {
            Ok(result) if streamed_answer => {
                session_stats.add_turn(&result.stats);
                terminal.borrow_mut().finish_answer()?;
            }
            Ok(result) if !result.answer.trim().is_empty() => {
                session_stats.add_turn(&result.stats);
                terminal.borrow_mut().write_answer_delta(&result.answer)?;
                terminal.borrow_mut().finish_answer()?;
            }
            Ok(result) => {
                session_stats.add_turn(&result.stats);
            }
            Err(error) => {
                if let Some(agent_error) = error.downcast_ref::<RunTurnError>() {
                    session_stats.add_failed_turn(&agent_error.stats);
                }
                terminal.borrow_mut().write_error(&error)?;
            }
        }
    }

    terminal
        .borrow_mut()
        .write_session_summary(&session_stats)?;

    Ok(())
}
