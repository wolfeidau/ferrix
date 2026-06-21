use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;

use crate::tools::{ToolCall, ToolResult, parse_args, resolve_path};

#[derive(Debug, Deserialize)]
struct ReadArgs {
    path: String,
}

#[derive(Debug, Deserialize)]
struct WriteArgs {
    path: String,
    contents: String,
    create_dirs: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct EditArgs {
    path: String,
    old: String,
    new: String,
}

pub fn read_file(workspace_root: &Path, call: &ToolCall) -> Result<ToolResult> {
    let args: ReadArgs = parse_args(call)?;
    let path = resolve_path(workspace_root, &args.path);
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read UTF-8 file `{}`", path.display()))?;

    Ok(ToolResult {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        ok: true,
        content: contents.clone(),
        data: json!({
            "path": path,
            "bytes": contents.len()
        }),
    })
}

pub fn write_file(workspace_root: &Path, call: &ToolCall) -> Result<ToolResult> {
    let args: WriteArgs = parse_args(call)?;
    let path = resolve_path(workspace_root, &args.path);

    if args.create_dirs.unwrap_or(false)
        && let Some(parent) = path.parent()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create `{}`", parent.display()))?;
    }

    fs::write(&path, &args.contents)
        .with_context(|| format!("failed to write `{}`", path.display()))?;

    Ok(ToolResult {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        ok: true,
        content: format!(
            "wrote {} bytes to `{}`",
            args.contents.len(),
            path.display()
        ),
        data: json!({
            "path": path,
            "bytes": args.contents.len()
        }),
    })
}

pub fn edit_file(workspace_root: &Path, call: &ToolCall) -> Result<ToolResult> {
    let args: EditArgs = parse_args(call)?;
    if args.old.is_empty() {
        bail!("edit `old` text must not be empty");
    }

    let path = resolve_path(workspace_root, &args.path);
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read UTF-8 file `{}`", path.display()))?;
    let match_count = contents.matches(&args.old).count();

    match match_count {
        0 => bail!("edit text was not found in `{}`", path.display()),
        1 => {}
        count => bail!("edit text matched {count} times in `{}`", path.display()),
    }

    let updated = contents.replace(&args.old, &args.new);
    fs::write(&path, updated).with_context(|| format!("failed to write `{}`", path.display()))?;

    Ok(ToolResult {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        ok: true,
        content: format!("edited `{}`", path.display()),
        data: json!({
            "path": path,
            "replacements": 1
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use pretty_assertions::assert_eq;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;

    fn temp_workspace() -> PathBuf {
        let path = std::env::temp_dir().join(format!("ferrix-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&path).expect("create temp workspace");
        path
    }

    fn edit_call(path: &str, old: &str, new: &str) -> ToolCall {
        ToolCall {
            call_id: "call_1".to_string(),
            item_id: None,
            name: "edit".to_string(),
            arguments: json!({
                "path": path,
                "old": old,
                "new": new
            }),
        }
    }

    #[test]
    fn edit_replaces_one_exact_match() {
        let workspace = temp_workspace();
        let file = workspace.join("main.rs");
        fs::write(&file, "fn main() {\n    todo!();\n}\n").expect("write file");

        let result = edit_file(
            &workspace,
            &edit_call("main.rs", "todo!();", "println!(\"hi\");"),
        )
        .expect("edit succeeds");

        assert!(result.ok);
        assert_eq!(
            fs::read_to_string(file).expect("read file"),
            "fn main() {\n    println!(\"hi\");\n}\n"
        );
    }

    #[test]
    fn edit_fails_when_match_is_missing() {
        let workspace = temp_workspace();
        fs::write(workspace.join("main.rs"), "fn main() {}\n").expect("write file");

        let error = edit_file(&workspace, &edit_call("main.rs", "missing", "new"))
            .expect_err("edit should fail");

        assert!(error.to_string().contains("not found"));
    }

    #[test]
    fn edit_fails_when_match_is_ambiguous() {
        let workspace = temp_workspace();
        fs::write(workspace.join("main.rs"), "same\nsame\n").expect("write file");

        let error = edit_file(&workspace, &edit_call("main.rs", "same", "new"))
            .expect_err("edit should fail");

        assert!(error.to_string().contains("matched 2 times"));
    }
}
