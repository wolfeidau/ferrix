use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

#[derive(Debug)]
pub struct RunRecorder {
    run_id: String,
    turn_id: String,
    path: PathBuf,
}

impl RunRecorder {
    pub fn new(workspace_root: &Path) -> Result<Self> {
        let run_id = Uuid::new_v4().to_string();
        let turn_id = Uuid::new_v4().to_string();
        let runs_dir = workspace_root.join(".ferrix").join("runs");
        fs::create_dir_all(&runs_dir)
            .with_context(|| format!("failed to create `{}`", runs_dir.display()))?;

        Ok(Self {
            path: runs_dir.join(format!("{run_id}.jsonl")),
            run_id,
            turn_id,
        })
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn record<T>(&self, kind: &str, payload: T) -> Result<()>
    where
        T: Serialize,
    {
        let event = json!({
            "timestamp_ms": timestamp_ms(),
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "kind": kind,
            "payload": payload
        });

        self.record_value(event)
    }

    fn record_value(&self, event: Value) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("failed to open `{}`", self.path.display()))?;

        serde_json::to_writer(&mut file, &event).context("failed to serialize run event")?;
        file.write_all(b"\n").context("failed to write run event")?;
        Ok(())
    }
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pretty_assertions::assert_eq;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;

    #[test]
    fn writes_jsonl_run_events() {
        let workspace = std::env::temp_dir().join(format!("ferrix-runs-{}", Uuid::new_v4()));
        fs::create_dir_all(&workspace).expect("create workspace");
        let recorder = RunRecorder::new(&workspace).expect("create recorder");

        recorder
            .record("execution_plan", json!({ "steps": ["read", "edit"] }))
            .expect("record event");

        let run_file = workspace
            .join(".ferrix")
            .join("runs")
            .join(format!("{}.jsonl", recorder.run_id()));
        let contents = fs::read_to_string(run_file).expect("read run file");
        let event: Value = serde_json::from_str(contents.trim()).expect("parse event");

        assert_eq!(event["kind"], "execution_plan");
        assert_eq!(event["payload"]["steps"][0], "read");
    }
}
