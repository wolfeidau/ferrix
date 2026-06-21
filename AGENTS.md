# Ferrix Agent Guide

Ferrix is a small Rust coding-agent CLI. Keep changes focused on the current shape of the app: a simple REPL, an OpenAI-compatible model client, local tools, `tracing` diagnostics, and JSONL run artifacts.

## Core Behavior

- The CLI prompt is exactly `ferrix> _`; preserve that string unless the user explicitly asks to change it.
- The user exits with `exit`, `quit`, or EOF.
- The agent loop sends conversation history and tool schemas to the model, executes requested tools locally, appends tool results, and repeats until a final answer is returned.
- Keep the model layer provider-neutral through the `Model` trait. Provider-specific details belong in the concrete model client or raw `extra`/JSON fields.

## Output

- Keep terminal output plain and predictable: prompt, streamed bash output, final assistant text, and clear errors.
- Do not add rich terminal layout, alternate screens, progress spinners, or interactive UI state unless the user explicitly asks for it.
- Bash stdout/stderr should pass through to the user's shell as it is produced; diagnostics belong in `tracing` and durable run artifacts.

## Local Tools

- `read` reads UTF-8 files and should report missing or non-text files as tool errors.
- `write` writes complete file contents. Do not silently invent partial-write semantics.
- `edit` replaces one exact `old` string with `new`; it must fail when the match is missing or ambiguous.
- `bash` runs through the shell from the workspace root, streams stdout/stderr to the terminal, and returns exit status plus bounded captured output.

## Run Artifacts And Logging

- Preserve durable run analysis data under `.ferrix/runs/` as JSONL events.
- Record model metadata, execution-plan payloads when present, model responses, tool calls, tool results, and final run status.
- Keep provider-specific execution-plan data as raw JSON so the core agent loop does not become tied to one API shape.
- Use `tracing` for internal diagnostics. Respect `FERRIX_LOG`/`RUST_LOG`, and keep logs separate from user-visible streamed bash output where practical.

## Model Configuration

- The default backend is the OpenAI-compatible HTTP Responses API.
- Configuration comes from `FERRIX_MODEL_PROVIDER`, `FERRIX_MODEL`, `FERRIX_BASE_URL`, and `FERRIX_API_KEY`; `OPENAI_API_KEY` is accepted as a fallback.
- Do not hard-code secrets, model keys, or user-specific endpoints.

## Development Standards

- Prefer small, direct Rust modules over broad abstractions until real duplication appears.
- Use `anyhow::Context` on fallible IO, HTTP, and serialization paths.
- Add tests around behavior that can damage files or affect run history, especially filesystem tools and artifact serialization.
- Run `cargo fmt` and `cargo test` after substantive Rust changes.
- Do not edit `.ferrix/` artifacts into source control; that directory is local runtime state.
