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
use anyhow::{Context, anyhow, bail};
use config::UiConfig;
use mcp::McpRegistry;
use model::{ConversationMessage, OpenAiCompatibleModel};
use terminal::TerminalUi;
use tools::ToolRegistry;
use uuid::Uuid;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    logging::init();

    let startup_mode = parse_startup_mode(std::env::args().skip(1))?;
    if startup_mode == StartupMode::Version {
        println!("ferrix {}", app_version());
        return Ok(());
    }

    let workspace_root = std::env::current_dir().context("failed to determine workspace root")?;
    let ui_config = UiConfig::from_workspace(&workspace_root)?;
    let session_cache_key = Some(format!("ferrix-{}", Uuid::new_v4()));
    let model =
        OpenAiCompatibleModel::from_workspace_with_cache_key(&workspace_root, session_cache_key)?;
    let mcp = McpRegistry::from_workspace(&workspace_root).await?;
    let mcp_instructions = mcp.server_instructions().to_vec();
    let tools = ToolRegistry::with_mcp(workspace_root.clone(), mcp);
    let agent = Agent::new(model, tools, workspace_root);
    let mut history = agent::initial_history_with_mcp(&mcp_instructions);
    let mut session_last_response_id = None;
    let terminal = RefCell::new(TerminalUi::stdout(ui_config));
    let mut session_stats = SessionStats::default();

    match startup_mode {
        StartupMode::Repl => {
            run_repl(
                &agent,
                &mut history,
                &mut session_last_response_id,
                &terminal,
                &mut session_stats,
            )
            .await?;
        }
        StartupMode::Prompt(input) => {
            run_prompt(
                &agent,
                &input,
                &mut history,
                &mut session_last_response_id,
                &terminal,
                &mut session_stats,
            )
            .await?;
        }
        StartupMode::Version => unreachable!("version mode exits before agent startup"),
    }

    terminal
        .borrow_mut()
        .write_session_summary(&session_stats)?;

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StartupMode {
    Repl,
    Prompt(String),
    Version,
}

fn parse_startup_mode(args: impl IntoIterator<Item = String>) -> anyhow::Result<StartupMode> {
    let mut args = args.into_iter();

    let Some(first) = args.next() else {
        return Ok(StartupMode::Repl);
    };

    match first.as_str() {
        "--version" => {
            if let Some(extra) = args.next() {
                bail!("--version cannot be combined with {extra}");
            }
            Ok(StartupMode::Version)
        }
        "--prompt" | "-p" => {
            let prompt = args
                .next()
                .ok_or_else(|| anyhow!("{first} requires a prompt"))?;
            if let Some(extra) = args.next() {
                bail!("unexpected argument after prompt: {extra}");
            }
            Ok(StartupMode::Prompt(prompt))
        }
        _ => bail!("unknown argument: {first}"),
    }
}

fn app_version() -> &'static str {
    option_env!("APP_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))
}

async fn run_repl(
    agent: &Agent<OpenAiCompatibleModel>,
    history: &mut Vec<ConversationMessage>,
    session_last_response_id: &mut Option<String>,
    terminal: &RefCell<TerminalUi<io::Stdout>>,
    session_stats: &mut SessionStats,
) -> anyhow::Result<()> {
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

        run_prompt(
            agent,
            input,
            history,
            session_last_response_id,
            terminal,
            session_stats,
        )
        .await?;
    }

    Ok(())
}

async fn run_prompt(
    agent: &Agent<OpenAiCompatibleModel>,
    input: &str,
    history: &mut Vec<ConversationMessage>,
    session_last_response_id: &mut Option<String>,
    terminal: &RefCell<TerminalUi<io::Stdout>>,
    session_stats: &mut SessionStats,
) -> anyhow::Result<()> {
    let mut streamed_answer = false;
    match agent
        .run_turn(
            input,
            history,
            session_last_response_id,
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn parses_repl_mode_without_args() {
        assert_eq!(
            parse_startup_mode([]).expect("parse mode"),
            StartupMode::Repl
        );
    }

    #[test]
    fn parses_version_mode() {
        assert_eq!(
            parse_startup_mode(["--version".to_string()]).expect("parse mode"),
            StartupMode::Version
        );
    }

    #[test]
    fn parses_prompt_mode() {
        assert_eq!(
            parse_startup_mode(["--prompt".to_string(), "hello".to_string()]).expect("parse mode"),
            StartupMode::Prompt("hello".to_string())
        );
        assert_eq!(
            parse_startup_mode(["-p".to_string(), "hello".to_string()]).expect("parse mode"),
            StartupMode::Prompt("hello".to_string())
        );
    }

    #[test]
    fn rejects_invalid_startup_args() {
        assert!(parse_startup_mode(["--version".to_string(), "-p".to_string()]).is_err());
        assert!(parse_startup_mode(["--prompt".to_string()]).is_err());
        assert!(
            parse_startup_mode([
                "--prompt".to_string(),
                "hello".to_string(),
                "again".to_string()
            ])
            .is_err()
        );
        assert!(parse_startup_mode(["--unknown".to_string()]).is_err());
    }
}
