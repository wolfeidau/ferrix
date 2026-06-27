use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;
use tracing::info;

use crate::tools::{ToolCall, ToolResult, parse_args};

const MAX_CAPTURE_BYTES: usize = 16 * 1024;

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
}

#[derive(Debug, Default)]
struct Capture {
    bytes: Vec<u8>,
    total_bytes: u64,
    truncated: bool,
}

pub fn run_command(workspace_root: &Path, call: &ToolCall) -> Result<ToolResult> {
    let args: BashArgs = parse_args(call)?;
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());

    info!(command = %args.command, "starting bash command");

    let mut child = Command::new(shell)
        .arg("-lc")
        .arg(&args.command)
        .current_dir(workspace_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn command `{}`", args.command))?;

    let stdout = child.stdout.take().context("failed to capture stdout")?;
    let stderr = child.stderr.take().context("failed to capture stderr")?;
    let stdout_capture = Arc::new(Mutex::new(Capture::default()));
    let stderr_capture = Arc::new(Mutex::new(Capture::default()));

    let stdout_thread = capture_pipe(stdout, stdout_capture.clone());
    let stderr_thread = capture_pipe(stderr, stderr_capture.clone());

    let status = child
        .wait()
        .with_context(|| format!("failed to wait for command `{}`", args.command))?;

    stdout_thread
        .join()
        .map_err(|_| anyhow::anyhow!("stdout capture thread panicked"))??;
    stderr_thread
        .join()
        .map_err(|_| anyhow::anyhow!("stderr capture thread panicked"))??;

    let stdout = finish_capture(&stdout_capture);
    let stderr = finish_capture(&stderr_capture);
    let exit_code = status.code();
    let ok = status.success();

    info!(?exit_code, ok, "bash command completed");

    Ok(ToolResult {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        ok,
        content: match exit_code {
            Some(code) => format!("command exited with status {code}"),
            None => "command terminated by signal".to_string(),
        },
        data: json!({
            "command": args.command,
            "exit_code": exit_code,
            "success": ok,
            "stdout": stdout.text,
            "stderr": stderr.text,
            "stdout_bytes": stdout.total_bytes,
            "stderr_bytes": stderr.total_bytes,
            "stdout_truncated": stdout.truncated,
            "stderr_truncated": stderr.truncated
        }),
    })
}

fn capture_pipe<R>(mut reader: R, capture: Arc<Mutex<Capture>>) -> thread::JoinHandle<Result<()>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut buffer = [0_u8; 8192];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }

            let mut capture = capture
                .lock()
                .map_err(|_| anyhow::anyhow!("capture lock was poisoned"))?;
            capture.total_bytes += read as u64;
            let remaining = MAX_CAPTURE_BYTES.saturating_sub(capture.bytes.len());
            if remaining > 0 {
                let to_take = remaining.min(read);
                capture.bytes.extend_from_slice(&buffer[..to_take]);
            }
            if read > remaining {
                capture.truncated = true;
            }
        }

        Ok(())
    })
}

struct CapturedText {
    text: String,
    total_bytes: u64,
    truncated: bool,
}

fn finish_capture(capture: &Arc<Mutex<Capture>>) -> CapturedText {
    let capture = capture.lock().expect("capture lock");
    CapturedText {
        text: String::from_utf8_lossy(&capture.bytes).into_owned(),
        total_bytes: capture.total_bytes,
        truncated: capture.truncated,
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn bash_captures_output_without_streaming() {
        let call = ToolCall {
            call_id: "call_1".to_string(),
            item_id: None,
            name: "bash".to_string(),
            arguments: json!({ "command": "printf ferrix" }),
        };

        let result = run_command(&std::env::temp_dir(), &call).expect("run command");

        assert!(result.ok);
        assert_eq!(result.data["exit_code"], 0);
        assert_eq!(result.data["stdout"], "ferrix");
        assert_eq!(result.data["stdout_bytes"], 6);
    }
}
