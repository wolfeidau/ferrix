use std::io::{self, IsTerminal, Write};

use anyhow::{Context, Result};

use crate::agent::{AgentEvent, SessionStats};
use crate::config::UiConfig;

const CLEAR_LINE: &str = "\r\x1b[2K";
const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const DIM: &str = "\x1b[2m";

#[derive(Debug)]
pub struct TerminalUi<W> {
    writer: W,
    status_enabled: bool,
    color_enabled: bool,
    status_visible: bool,
}

impl TerminalUi<io::Stdout> {
    pub fn stdout(config: UiConfig) -> Self {
        let stdout = io::stdout();
        let interactive = stdout.is_terminal();
        Self::new(
            stdout,
            config.status_line && interactive,
            config.color && interactive,
        )
    }
}

impl<W> TerminalUi<W>
where
    W: Write,
{
    pub fn new(writer: W, status_enabled: bool, color_enabled: bool) -> Self {
        Self {
            writer,
            status_enabled,
            color_enabled,
            status_visible: false,
        }
    }

    pub fn write_prompt(&mut self) -> Result<()> {
        self.clear_status()?;
        write!(self.writer, "ferrix> ").context("failed to write prompt")?;
        self.writer.flush().context("failed to flush prompt")
    }

    pub fn set_status(&mut self, status: &str) -> Result<()> {
        if !self.status_enabled {
            return Ok(());
        }

        write!(self.writer, "{CLEAR_LINE}{status}").context("failed to write status line")?;
        self.writer.flush().context("failed to flush status line")?;
        self.status_visible = true;
        Ok(())
    }

    pub fn clear_status(&mut self) -> Result<()> {
        if self.status_visible {
            write!(self.writer, "{CLEAR_LINE}").context("failed to clear status line")?;
            self.writer
                .flush()
                .context("failed to flush status clear")?;
            self.status_visible = false;
        }
        Ok(())
    }

    pub fn log_event(&mut self, level: LogLevel, tag: &str, message: &str) -> Result<()> {
        self.clear_status()?;
        writeln!(self.writer, "{} {tag} {message}", self.level_prefix(level))
            .context("failed to write terminal log event")?;
        self.writer.flush().context("failed to flush terminal log")
    }

    pub fn write_answer_delta(&mut self, delta: &str) -> Result<()> {
        self.clear_status()?;
        write!(self.writer, "{delta}").context("failed to write streamed response")?;
        self.writer
            .flush()
            .context("failed to flush streamed response")
    }

    pub fn finish_answer(&mut self) -> Result<()> {
        self.clear_status()?;
        writeln!(self.writer).context("failed to finish response line")
    }

    pub fn write_error(&mut self, error: &anyhow::Error) -> Result<()> {
        self.log_event(LogLevel::Err, "[RUN]", &format!("error: {error:#}"))
    }

    pub fn handle_agent_event(&mut self, event: &AgentEvent) -> Result<()> {
        match event {
            AgentEvent::ModelStarted { iteration, model } => {
                self.set_status(&format!("thinking | iter {iteration} | model {model}"))
            }
            AgentEvent::ModelCompleted {
                iteration,
                model,
                duration_ms,
                usage_totals,
                tool_call_count,
                ..
            } => self.log_event(
                LogLevel::Ok,
                "[MODEL]",
                &format!(
                    "{model} iter={iteration} completed in {} tokens={} cache={} tool_calls={tool_call_count}",
                    format_duration(*duration_ms),
                    format_tokens(usage_totals.total_tokens),
                    format_cache_percent(
                        usage_totals.cached_input_tokens,
                        usage_totals.input_tokens
                    )
                ),
            ),
            AgentEvent::ToolStarted { tag, name, call_id } => {
                self.set_status(&format!("tool {name} running | {tag} call={call_id}"))
            }
            AgentEvent::ToolCompleted {
                tag,
                name,
                call_id,
                ok,
                duration_ms,
                exit_code,
                stdout_bytes,
                stderr_bytes,
                truncated,
            } => {
                let mut message = format!(
                    "{name} call={call_id} completed in {}",
                    format_duration(*duration_ms)
                );
                if let Some(exit_code) = exit_code {
                    message.push_str(&format!(" exit={exit_code}"));
                }
                if let Some(stdout_bytes) = stdout_bytes {
                    message.push_str(&format!(" stdout={}", format_bytes(*stdout_bytes)));
                }
                if let Some(stderr_bytes) = stderr_bytes {
                    message.push_str(&format!(" stderr={}", format_bytes(*stderr_bytes)));
                }
                if *truncated {
                    message.push_str(" truncated=true");
                }
                self.log_event(
                    if *ok { LogLevel::Ok } else { LogLevel::Err },
                    tag,
                    &message,
                )
            }
        }
    }

    pub fn write_session_summary(&mut self, stats: &SessionStats) -> Result<()> {
        if stats.turns == 0 {
            return Ok(());
        }

        self.log_event(
            LogLevel::Ok,
            "[SESSION]",
            &format!(
                "complete turns={} failed_turns={} model_calls={} tokens={} cache={} reasoning={} tools={} failed_tools={} tool_time={}",
                stats.turns,
                stats.failed_turns,
                stats.model_usage.model_calls,
                format_tokens(stats.model_usage.total_tokens),
                format_cache_percent(
                    stats.model_usage.cached_input_tokens,
                    stats.model_usage.input_tokens
                ),
                format_tokens(stats.model_usage.reasoning_tokens),
                stats.tool_usage.tool_calls,
                stats.tool_usage.failed_tool_calls,
                format_duration(stats.tool_usage.duration_ms)
            ),
        )
    }

    fn level_prefix(&self, level: LogLevel) -> String {
        let label = match level {
            LogLevel::Ok => "[OK]",
            LogLevel::Err => "[ERR]",
        };

        if !self.color_enabled {
            return label.to_string();
        }

        let color = match level {
            LogLevel::Ok => GREEN,
            LogLevel::Err => RED,
        };
        format!("{color}{label}{RESET}")
    }

    #[cfg(test)]
    pub fn into_inner(self) -> W {
        self.writer
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Ok,
    Err,
}

pub fn format_cache_percent(cached_tokens: u64, input_tokens: u64) -> String {
    if input_tokens == 0 {
        return "n/a".to_string();
    }

    format!("{}%", cached_tokens.saturating_mul(100) / input_tokens)
}

pub fn format_tokens(tokens: u64) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 1_000_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        format!("{:.1}m", tokens as f64 / 1_000_000.0)
    }
}

pub fn format_duration(duration_ms: u128) -> String {
    if duration_ms < 1_000 {
        format!("{duration_ms}ms")
    } else {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    }
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes < 1_024 {
        format!("{bytes}B")
    } else if bytes < 1_024 * 1_024 {
        format!("{:.1}KB", bytes as f64 / 1_024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1_024.0 * 1_024.0))
    }
}

#[allow(dead_code)]
fn dim(value: &str) -> String {
    format!("{DIM}{value}{RESET}")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn formats_stats_for_log_lines() {
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(12_345), "12.3k");
        assert_eq!(format_cache_percent(300, 1_000), "30%");
        assert_eq!(format_cache_percent(0, 0), "n/a");
        assert_eq!(format_duration(842), "842ms");
        assert_eq!(format_duration(1_850), "1.9s");
        assert_eq!(format_bytes(16_384), "16.0KB");
    }

    #[test]
    fn clears_status_before_log_event() {
        let mut terminal = TerminalUi::new(Vec::new(), true, false);

        terminal.set_status("thinking").expect("status");
        terminal
            .log_event(LogLevel::Ok, "[MODEL]", "done")
            .expect("log event");

        let output = String::from_utf8(terminal.into_inner()).expect("utf8 output");
        assert_eq!(output, "\r\x1b[2Kthinking\r\x1b[2K[OK] [MODEL] done\n");
    }

    #[test]
    fn colors_status_prefix_when_enabled() {
        let mut terminal = TerminalUi::new(Vec::new(), false, true);

        terminal
            .log_event(LogLevel::Err, "[TOOL:bash]", "failed")
            .expect("log event");

        let output = String::from_utf8(terminal.into_inner()).expect("utf8 output");
        assert_eq!(output, "\x1b[31m[ERR]\x1b[0m [TOOL:bash] failed\n");
    }

    #[test]
    fn session_summary_includes_failed_turns() {
        let mut terminal = TerminalUi::new(Vec::new(), false, false);
        let stats = SessionStats {
            turns: 2,
            failed_turns: 1,
            model_usage: crate::agent::ModelUsageTotals {
                model_calls: 3,
                ..Default::default()
            },
            ..Default::default()
        };

        terminal
            .write_session_summary(&stats)
            .expect("session summary");

        let output = String::from_utf8(terminal.into_inner()).expect("utf8 output");
        assert!(output.contains("turns=2 failed_turns=1 model_calls=3"));
    }
}
