mod agent;
mod logging;
mod model;
mod runs;
mod tools;

use std::io::{self, Write};

use agent::Agent;
use anyhow::Context;
use model::OpenAiCompatibleModel;
use tools::ToolRegistry;

fn main() -> anyhow::Result<()> {
    logging::init();

    let workspace_root = std::env::current_dir().context("failed to determine workspace root")?;
    let model = OpenAiCompatibleModel::from_env()?;
    let tools = ToolRegistry::new(workspace_root.clone());
    let agent = Agent::new(model, tools, workspace_root);
    let mut history = agent::initial_history();

    let stdin = io::stdin();

    loop {
        print!("ferrix> ");
        io::stdout().flush().context("failed to flush prompt")?;

        let mut input = String::new();
        let bytes_read = stdin
            .read_line(&mut input)
            .context("failed to read user input")?;

        if bytes_read == 0 {
            println!();
            break;
        }

        let input = input.trim();
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }

        if input.is_empty() {
            continue;
        }

        match agent.run_turn(input, &mut history) {
            Ok(answer) if !answer.trim().is_empty() => println!("{answer}"),
            Ok(_) => {}
            Err(error) => eprintln!("error: {error:#}"),
        }
    }

    Ok(())
}
