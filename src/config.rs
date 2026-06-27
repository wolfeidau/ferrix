use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use async_openai::types::responses::{PromptCacheRetention, ReasoningEffort};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::Value;

const FERRIX_CONFIG_PATH: &str = ".ferrix/config.json";
const DEFAULT_PROVIDER: &str = "openai-compatible";
const DEFAULT_MODEL: &str = "gpt-5.5";
const OPENAI_API_BASE: &str = "https://api.openai.com/v1";
const OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub provider: String,
    pub model: String,
    pub api_base: String,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub api_key_env_var: &'static str,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub openrouter: OpenRouterConfig,
    pub prompt_cache: PromptCacheConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiConfig {
    pub status_line: bool,
    pub color: bool,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PromptCacheConfig {
    pub retention: Option<PromptCacheRetention>,
    pub key: Option<String>,
    pub store_responses: bool,
}

impl ModelConfig {
    pub fn from_workspace(workspace_root: &Path) -> Result<Self> {
        Self::from_workspace_with_env(workspace_root, env_optional)
    }

    fn from_workspace_with_env(
        workspace_root: &Path,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let workspace_config = FerrixConfig::load(workspace_root)?;
        Self::from_parts(workspace_config.model, env)
    }

    fn from_parts(
        workspace_model: Option<WorkspaceModelConfig>,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let workspace_model = workspace_model.unwrap_or_default();
        let mut openrouter = workspace_model.openrouter.unwrap_or_default();

        let provider = env("FERRIX_MODEL_PROVIDER")
            .or(workspace_model.provider)
            .unwrap_or_else(|| DEFAULT_PROVIDER.to_string());
        let provider_kind = provider_kind(&provider);
        let model = env("FERRIX_MODEL")
            .or(workspace_model.name)
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        let default_base_url = match provider_kind {
            ProviderKind::OpenRouter => OPENROUTER_API_BASE,
            ProviderKind::OpenAiCompatible => OPENAI_API_BASE,
        };
        let api_base = normalize_api_base(
            &env("FERRIX_BASE_URL")
                .or(workspace_model.base_url)
                .unwrap_or_else(|| default_base_url.to_string()),
        )?;
        let endpoint = responses_endpoint(&api_base);

        let api_key_env_var = match provider_kind {
            ProviderKind::OpenRouter => "OPENROUTER_API_KEY",
            ProviderKind::OpenAiCompatible => "OPENAI_API_KEY",
        };
        let api_key = env(api_key_env_var);

        if let Some(value) = env("FERRIX_OPENROUTER_REFERER") {
            openrouter.referer = Some(value);
        }
        if let Some(value) = env("FERRIX_OPENROUTER_TITLE") {
            openrouter.title = Some(value);
        }
        if let Some(value) = env("FERRIX_OPENROUTER_CATEGORIES") {
            openrouter.categories = Some(value);
        }

        let reasoning_effort = match env("FERRIX_REASONING_EFFORT") {
            Some(effort) => Some(parse_reasoning_effort(&effort)?),
            None => workspace_model.reasoning_effort.map(Into::into),
        };

        let prompt_cache = workspace_model.prompt_cache.unwrap_or_default();
        let prompt_cache_retention = match env("FERRIX_PROMPT_CACHE_RETENTION") {
            Some(value) => Some(parse_prompt_cache_retention(&value)?),
            None => prompt_cache.retention.map(Into::into),
        };
        let prompt_cache_key = env("FERRIX_PROMPT_CACHE_KEY").or(prompt_cache.key);
        let store_responses = parse_bool_option(
            "FERRIX_STORE_RESPONSES",
            env("FERRIX_STORE_RESPONSES"),
            prompt_cache.store_responses,
        )?;

        Ok(Self {
            provider,
            model,
            api_base,
            endpoint,
            api_key,
            api_key_env_var,
            reasoning_effort,
            max_output_tokens: parse_u32_option(
                "FERRIX_MAX_OUTPUT_TOKENS",
                env("FERRIX_MAX_OUTPUT_TOKENS"),
                workspace_model.max_output_tokens,
            )?,
            temperature: parse_f32_range_option(
                "FERRIX_TEMPERATURE",
                env("FERRIX_TEMPERATURE"),
                workspace_model.temperature,
                0.0,
                2.0,
            )?,
            top_p: parse_f32_range_option(
                "FERRIX_TOP_P",
                env("FERRIX_TOP_P"),
                workspace_model.top_p,
                0.0,
                1.0,
            )?,
            openrouter,
            prompt_cache: PromptCacheConfig {
                retention: prompt_cache_retention,
                key: prompt_cache_key,
                store_responses,
            },
        })
    }

    pub fn with_prompt_cache_key(mut self, key: Option<String>) -> Self {
        if self.prompt_cache.key.is_none() {
            self.prompt_cache.key = key;
        }
        self
    }
}

impl UiConfig {
    pub fn from_workspace(workspace_root: &Path) -> Result<Self> {
        Self::from_workspace_with_env(workspace_root, env_optional)
    }

    fn from_workspace_with_env(
        workspace_root: &Path,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let workspace_config = FerrixConfig::load(workspace_root)?;
        Self::from_parts(workspace_config.ui, env)
    }

    fn from_parts(
        workspace_ui: Option<WorkspaceUiConfig>,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let workspace_ui = workspace_ui.unwrap_or_default();
        let status_line = parse_bool_option(
            "FERRIX_STATUS_LINE",
            env("FERRIX_STATUS_LINE"),
            workspace_ui.status_line.or(Some(true)),
        )?;
        let color = if env("NO_COLOR").is_some() {
            false
        } else {
            parse_bool_option(
                "FERRIX_COLOR",
                env("FERRIX_COLOR"),
                workspace_ui.color.or(Some(true)),
            )?
        };

        Ok(Self { status_line, color })
    }
}

#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterConfig {
    #[serde(default)]
    #[schemars(length(min = 1))]
    pub referer: Option<String>,
    #[serde(default)]
    #[schemars(length(min = 1))]
    pub title: Option<String>,
    #[serde(default)]
    #[schemars(length(min = 1))]
    pub categories: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct FerrixConfig {
    #[serde(default)]
    model: Option<WorkspaceModelConfig>,
    #[serde(default)]
    ui: Option<WorkspaceUiConfig>,
}

impl FerrixConfig {
    fn load(workspace_root: &Path) -> Result<Self> {
        let config_path = workspace_root.join(FERRIX_CONFIG_PATH);
        if !config_path.exists() {
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read `{}`", config_path.display()))?;

        let value: Value = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse `{}`", config_path.display()))?;
        validate_ferrix_config(&value)
            .with_context(|| format!("failed to validate `{}`", config_path.display()))?;
        serde_json::from_value(value)
            .with_context(|| format!("failed to parse `{}`", config_path.display()))
    }
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WorkspaceModelConfig {
    #[serde(default)]
    #[schemars(length(min = 1))]
    provider: Option<String>,
    #[serde(default)]
    #[schemars(length(min = 1))]
    name: Option<String>,
    #[serde(default)]
    #[schemars(length(min = 1))]
    base_url: Option<String>,
    #[serde(default)]
    reasoning_effort: Option<WorkspaceReasoningEffort>,
    #[serde(default)]
    #[schemars(range(min = 1))]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    #[schemars(range(min = 0.0, max = 2.0))]
    temperature: Option<f32>,
    #[serde(default)]
    #[schemars(range(min = 0.0, max = 1.0))]
    top_p: Option<f32>,
    #[serde(default)]
    openrouter: Option<OpenRouterConfig>,
    #[serde(default)]
    prompt_cache: Option<WorkspacePromptCacheConfig>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WorkspacePromptCacheConfig {
    #[serde(default)]
    retention: Option<WorkspacePromptCacheRetention>,
    #[serde(default)]
    #[schemars(length(min = 1))]
    key: Option<String>,
    #[serde(default)]
    store_responses: Option<bool>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WorkspaceUiConfig {
    #[serde(default)]
    status_line: Option<bool>,
    #[serde(default)]
    color: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(rename_all = "lowercase")]
enum WorkspacePromptCacheRetention {
    #[serde(rename = "in_memory")]
    InMemory,
    #[serde(rename = "24h")]
    Hours24,
}

impl From<WorkspacePromptCacheRetention> for PromptCacheRetention {
    fn from(retention: WorkspacePromptCacheRetention) -> Self {
        match retention {
            WorkspacePromptCacheRetention::InMemory => Self::InMemory,
            WorkspacePromptCacheRetention::Hours24 => Self::Hours24,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(rename_all = "lowercase")]
enum WorkspaceReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl From<WorkspaceReasoningEffort> for ReasoningEffort {
    fn from(effort: WorkspaceReasoningEffort) -> Self {
        match effort {
            WorkspaceReasoningEffort::None => Self::None,
            WorkspaceReasoningEffort::Minimal => Self::Minimal,
            WorkspaceReasoningEffort::Low => Self::Low,
            WorkspaceReasoningEffort::Medium => Self::Medium,
            WorkspaceReasoningEffort::High => Self::High,
            WorkspaceReasoningEffort::Xhigh => Self::Xhigh,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderKind {
    OpenRouter,
    OpenAiCompatible,
}

fn env_optional(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn validate_ferrix_config(config: &Value) -> Result<()> {
    let schema = ferrix_config_schema();
    let validator = jsonschema::validator_for(&schema).context("failed to build config schema")?;
    let errors = validator
        .iter_errors(config)
        .map(|error| format!("{}: {}", error.instance_path(), error))
        .collect::<Vec<_>>();

    if !errors.is_empty() {
        bail!("invalid Ferrix config:\n{}", errors.join("\n"));
    }

    Ok(())
}

fn ferrix_config_schema() -> Value {
    serde_json::to_value(schema_for!(FerrixConfig)).expect("Ferrix config schema serializes")
}

fn provider_kind(provider: &str) -> ProviderKind {
    match provider.trim().to_ascii_lowercase().as_str() {
        "openrouter" => ProviderKind::OpenRouter,
        _ => ProviderKind::OpenAiCompatible,
    }
}

fn parse_u32_option(
    name: &str,
    env_value: Option<String>,
    config_value: Option<u32>,
) -> Result<Option<u32>> {
    let value = match env_value {
        Some(value) => value
            .parse::<u32>()
            .map(Some)
            .with_context(|| format!("invalid {name} `{value}`; expected a positive integer")),
        None => Ok(config_value),
    }?;

    if let Some(value) = value
        && value == 0
    {
        bail!("invalid {name} `0`; expected a positive integer");
    }

    Ok(value)
}

fn parse_f32_range_option(
    name: &str,
    env_value: Option<String>,
    config_value: Option<f32>,
    min: f32,
    max: f32,
) -> Result<Option<f32>> {
    let value = match env_value {
        Some(value) => Some(value.parse::<f32>().with_context(|| {
            format!("invalid {name} `{value}`; expected a number between {min} and {max}")
        })?),
        None => config_value,
    };

    if let Some(value) = value
        && !(min..=max).contains(&value)
    {
        bail!("invalid {name} `{value}`; expected a number between {min} and {max}");
    }

    Ok(value)
}

fn normalize_api_base(base_url: &str) -> Result<String> {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.is_empty() {
        bail!("FERRIX_BASE_URL must not be empty");
    }
    if trimmed.ends_with("/chat/completions") {
        bail!(
            "FERRIX_BASE_URL must point to a Responses API base URL, not a chat completions endpoint"
        );
    }

    Ok(trimmed
        .strip_suffix("/responses")
        .unwrap_or(trimmed)
        .to_string())
}

fn responses_endpoint(base_url: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    if base_url.ends_with("/responses") {
        base_url.to_string()
    } else {
        format!("{base_url}/responses")
    }
}

fn parse_bool_option(
    name: &str,
    env_value: Option<String>,
    config_value: Option<bool>,
) -> Result<bool> {
    match env_value {
        Some(value) => parse_bool(&value).with_context(|| format!("invalid {name} `{value}`")),
        None => Ok(config_value.unwrap_or(false)),
    }
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => bail!("invalid boolean `{other}`; expected true or false"),
    }
}

fn parse_prompt_cache_retention(value: &str) -> Result<PromptCacheRetention> {
    match value.trim().to_ascii_lowercase().as_str() {
        "in_memory" => Ok(PromptCacheRetention::InMemory),
        "24h" => Ok(PromptCacheRetention::Hours24),
        other => {
            bail!("invalid FERRIX_PROMPT_CACHE_RETENTION `{other}`; expected in_memory or 24h")
        }
    }
}

fn parse_reasoning_effort(value: &str) -> Result<ReasoningEffort> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(ReasoningEffort::None),
        "minimal" => Ok(ReasoningEffort::Minimal),
        "low" => Ok(ReasoningEffort::Low),
        "medium" => Ok(ReasoningEffort::Medium),
        "high" => Ok(ReasoningEffort::High),
        "xhigh" => Ok(ReasoningEffort::Xhigh),
        other => bail!(
            "invalid FERRIX_REASONING_EFFORT `{other}`; expected one of none, minimal, low, medium, high, xhigh"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;
    use uuid::Uuid;

    use super::*;

    #[test]
    fn normalizes_responses_api_base_url() {
        let base = normalize_api_base("https://api.openai.com/v1").expect("normalize base");

        assert_eq!(base, "https://api.openai.com/v1");
        assert_eq!(
            responses_endpoint(&base),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn normalizes_responses_endpoint_url() {
        let base =
            normalize_api_base("https://api.openai.com/v1/responses").expect("normalize endpoint");

        assert_eq!(base, "https://api.openai.com/v1");
        assert_eq!(
            responses_endpoint(&base),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn rejects_chat_completions_endpoint_url() {
        let error = normalize_api_base("https://api.openai.com/v1/chat/completions")
            .expect_err("chat completions endpoint should be rejected");

        assert!(
            error
                .to_string()
                .contains("Responses API base URL, not a chat completions endpoint")
        );
    }

    #[test]
    fn rejects_empty_responses_api_base_url() {
        let error = normalize_api_base("").expect_err("empty endpoint should be rejected");

        assert!(error.to_string().contains("must not be empty"));
    }

    #[test]
    fn loads_workspace_model_config_from_ferrix_config_json() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "model": {
                    "provider": "openrouter",
                    "name": "openai/gpt-5.2",
                    "reasoning_effort": "low",
                    "max_output_tokens": 9000,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "openrouter": {
                        "referer": "https://example.com",
                        "title": "Ferrix",
                        "categories": "cli-agent"
                    }
                }
            }"#,
        );

        let config = ModelConfig::from_workspace_with_env(
            &workspace,
            env_from(&[("OPENROUTER_API_KEY", "sk-or")]),
        )
        .expect("load config");

        assert_eq!(config.provider, "openrouter");
        assert_eq!(config.model, "openai/gpt-5.2");
        assert_eq!(config.api_base, OPENROUTER_API_BASE);
        assert_eq!(config.endpoint, "https://openrouter.ai/api/v1/responses");
        assert_eq!(config.api_key.as_deref(), Some("sk-or"));
        assert_eq!(config.api_key_env_var, "OPENROUTER_API_KEY");
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::Low));
        assert_eq!(config.max_output_tokens, Some(9000));
        assert_eq!(config.temperature, Some(0.2));
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(
            config.openrouter.referer.as_deref(),
            Some("https://example.com")
        );
        assert_eq!(config.openrouter.title.as_deref(), Some("Ferrix"));
        assert_eq!(config.openrouter.categories.as_deref(), Some("cli-agent"));
    }

    #[test]
    fn env_overrides_workspace_model_config() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "model": {
                    "provider": "openai-compatible",
                    "name": "gpt-config",
                    "base_url": "https://api.openai.com/v1",
                    "reasoning_effort": "low",
                    "max_output_tokens": 1000,
                    "temperature": 0.3,
                    "top_p": 0.8
                }
            }"#,
        );

        let config = ModelConfig::from_workspace_with_env(
            &workspace,
            env_from(&[
                ("FERRIX_MODEL_PROVIDER", "openrouter"),
                ("FERRIX_MODEL", "openai/gpt-5.2"),
                ("FERRIX_BASE_URL", "https://openrouter.ai/api/v1/responses"),
                ("FERRIX_REASONING_EFFORT", "medium"),
                ("FERRIX_MAX_OUTPUT_TOKENS", "2000"),
                ("FERRIX_TEMPERATURE", "0.5"),
                ("FERRIX_TOP_P", "0.7"),
                ("OPENROUTER_API_KEY", "sk-or"),
            ]),
        )
        .expect("load config");

        assert_eq!(config.provider, "openrouter");
        assert_eq!(config.model, "openai/gpt-5.2");
        assert_eq!(config.api_base, OPENROUTER_API_BASE);
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::Medium));
        assert_eq!(config.max_output_tokens, Some(2000));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.top_p, Some(0.7));
        assert_eq!(config.api_key.as_deref(), Some("sk-or"));
    }

    #[test]
    fn rejects_unknown_config_fields() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "model": {
                    "provider": "openrouter",
                    "nam": "openai/gpt-5.2"
                }
            }"#,
        );

        let error = ModelConfig::from_workspace_with_env(&workspace, env_from(&[]))
            .expect_err("unknown config field should fail schema validation");
        let error = format!("{error:#}");

        assert!(error.contains("failed to validate"));
        assert!(error.contains("nam"));
    }

    #[test]
    fn rejects_config_values_outside_schema_ranges() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "model": {
                    "temperature": 3,
                    "top_p": 2,
                    "max_output_tokens": 0
                }
            }"#,
        );

        let error = ModelConfig::from_workspace_with_env(&workspace, env_from(&[]))
            .expect_err("out-of-range config values should fail schema validation");
        let error = format!("{error:#}");

        assert!(error.contains("temperature"));
        assert!(error.contains("top_p"));
        assert!(error.contains("max_output_tokens"));
    }

    #[test]
    fn rejects_invalid_config_types() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "model": {
                    "provider": 42,
                    "openrouter": {
                        "title": false
                    }
                }
            }"#,
        );

        let error = ModelConfig::from_workspace_with_env(&workspace, env_from(&[]))
            .expect_err("invalid config types should fail schema validation");
        let error = format!("{error:#}");

        assert!(error.contains("provider"));
        assert!(error.contains("title"));
    }

    #[test]
    fn openrouter_provider_uses_openrouter_base_url() {
        let config = ModelConfig::from_parts(
            Some(WorkspaceModelConfig {
                provider: Some("openrouter".to_string()),
                ..Default::default()
            }),
            env_from(&[]),
        )
        .expect("resolve config");

        assert_eq!(config.api_base, OPENROUTER_API_BASE);
        assert_eq!(config.endpoint, "https://openrouter.ai/api/v1/responses");
    }

    #[test]
    fn openrouter_provider_uses_openrouter_api_key() {
        let config = ModelConfig::from_parts(
            Some(WorkspaceModelConfig {
                provider: Some("openrouter".to_string()),
                ..Default::default()
            }),
            env_from(&[
                ("OPENROUTER_API_KEY", "sk-or"),
                ("OPENAI_API_KEY", "sk-openai"),
            ]),
        )
        .expect("resolve config");

        assert_eq!(config.api_key.as_deref(), Some("sk-or"));
        assert_eq!(config.api_key_env_var, "OPENROUTER_API_KEY");
    }

    #[test]
    fn openai_provider_uses_openai_api_key() {
        let config = ModelConfig::from_parts(
            Some(WorkspaceModelConfig {
                provider: Some("openai".to_string()),
                ..Default::default()
            }),
            env_from(&[
                ("OPENROUTER_API_KEY", "sk-or"),
                ("OPENAI_API_KEY", "sk-openai"),
            ]),
        )
        .expect("resolve config");

        assert_eq!(config.api_key.as_deref(), Some("sk-openai"));
        assert_eq!(config.api_key_env_var, "OPENAI_API_KEY");
    }

    #[test]
    fn generic_provider_keeps_existing_openai_defaults() {
        let config = ModelConfig::from_parts(None, env_from(&[])).expect("resolve config");

        assert_eq!(config.provider, DEFAULT_PROVIDER);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.api_base, OPENAI_API_BASE);
        assert_eq!(config.endpoint, "https://api.openai.com/v1/responses");
        assert_eq!(config.api_key_env_var, "OPENAI_API_KEY");
    }

    #[test]
    fn parses_reasoning_effort_values() {
        assert_eq!(
            parse_reasoning_effort("none").unwrap(),
            ReasoningEffort::None
        );
        assert_eq!(
            parse_reasoning_effort("minimal").unwrap(),
            ReasoningEffort::Minimal
        );
        assert_eq!(parse_reasoning_effort("low").unwrap(), ReasoningEffort::Low);
        assert_eq!(
            parse_reasoning_effort("medium").unwrap(),
            ReasoningEffort::Medium
        );
        assert_eq!(
            parse_reasoning_effort("HIGH").unwrap(),
            ReasoningEffort::High
        );
        assert_eq!(
            parse_reasoning_effort("xhigh").unwrap(),
            ReasoningEffort::Xhigh
        );
    }

    #[test]
    fn rejects_invalid_reasoning_effort_value() {
        let error =
            parse_reasoning_effort("maximum").expect_err("invalid reasoning effort should fail");

        assert!(
            error
                .to_string()
                .contains("invalid FERRIX_REASONING_EFFORT")
        );
    }

    #[test]
    fn rejects_invalid_request_options() {
        let error = ModelConfig::from_parts(
            None,
            env_from(&[
                ("FERRIX_MAX_OUTPUT_TOKENS", "many"),
                ("FERRIX_TEMPERATURE", "0.2"),
                ("FERRIX_TOP_P", "0.9"),
            ]),
        )
        .expect_err("invalid max output tokens should fail");
        assert!(error.to_string().contains("FERRIX_MAX_OUTPUT_TOKENS"));

        let error = ModelConfig::from_parts(
            Some(WorkspaceModelConfig {
                max_output_tokens: Some(0),
                ..Default::default()
            }),
            env_from(&[]),
        )
        .expect_err("zero max output tokens should fail");
        assert!(error.to_string().contains("FERRIX_MAX_OUTPUT_TOKENS"));

        let error = ModelConfig::from_parts(
            None,
            env_from(&[("FERRIX_TEMPERATURE", "3"), ("FERRIX_TOP_P", "0.9")]),
        )
        .expect_err("invalid temperature should fail");
        assert!(error.to_string().contains("FERRIX_TEMPERATURE"));

        let error = ModelConfig::from_parts(
            None,
            env_from(&[("FERRIX_TEMPERATURE", "0.2"), ("FERRIX_TOP_P", "2")]),
        )
        .expect_err("invalid top_p should fail");
        assert!(error.to_string().contains("FERRIX_TOP_P"));
    }

    #[test]
    fn loads_prompt_cache_config_from_ferrix_config_json() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "model": {
                    "prompt_cache": {
                        "retention": "24h",
                        "key": "my-session",
                        "store_responses": true
                    }
                }
            }"#,
        );

        let config =
            ModelConfig::from_workspace_with_env(&workspace, env_from(&[])).expect("load config");

        assert_eq!(
            config.prompt_cache.retention,
            Some(PromptCacheRetention::Hours24)
        );
        assert_eq!(config.prompt_cache.key.as_deref(), Some("my-session"));
        assert!(config.prompt_cache.store_responses);
    }

    #[test]
    fn env_overrides_prompt_cache_config() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "model": {
                    "prompt_cache": {
                        "retention": "24h",
                        "key": "config-key",
                        "store_responses": false
                    }
                }
            }"#,
        );

        let config = ModelConfig::from_workspace_with_env(
            &workspace,
            env_from(&[
                ("FERRIX_PROMPT_CACHE_RETENTION", "in_memory"),
                ("FERRIX_PROMPT_CACHE_KEY", "env-key"),
                ("FERRIX_STORE_RESPONSES", "true"),
            ]),
        )
        .expect("load config");

        assert_eq!(
            config.prompt_cache.retention,
            Some(PromptCacheRetention::InMemory)
        );
        assert_eq!(config.prompt_cache.key.as_deref(), Some("env-key"));
        assert!(config.prompt_cache.store_responses);
    }

    #[test]
    fn with_prompt_cache_key_fills_missing_key_only() {
        let config = ModelConfig::from_parts(None, env_from(&[]))
            .expect("resolve config")
            .with_prompt_cache_key(Some("session-key".to_string()));

        assert_eq!(config.prompt_cache.key.as_deref(), Some("session-key"));

        let config = ModelConfig::from_parts(
            Some(WorkspaceModelConfig {
                prompt_cache: Some(WorkspacePromptCacheConfig {
                    key: Some("configured".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            env_from(&[]),
        )
        .expect("resolve config")
        .with_prompt_cache_key(Some("session-key".to_string()));

        assert_eq!(config.prompt_cache.key.as_deref(), Some("configured"));
    }

    #[test]
    fn loads_workspace_ui_config_from_ferrix_config_json() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "ui": {
                    "status_line": false,
                    "color": false
                }
            }"#,
        );

        let config =
            UiConfig::from_workspace_with_env(&workspace, env_from(&[])).expect("load ui config");

        assert_eq!(
            config,
            UiConfig {
                status_line: false,
                color: false
            }
        );
    }

    #[test]
    fn env_overrides_workspace_ui_config() {
        let workspace = temp_workspace();
        write_config(
            &workspace,
            r#"{
                "ui": {
                    "status_line": false,
                    "color": false
                }
            }"#,
        );

        let config = UiConfig::from_workspace_with_env(
            &workspace,
            env_from(&[("FERRIX_STATUS_LINE", "true"), ("FERRIX_COLOR", "true")]),
        )
        .expect("load ui config");

        assert!(config.status_line);
        assert!(config.color);
    }

    #[test]
    fn no_color_disables_ui_color() {
        let config =
            UiConfig::from_parts(None, env_from(&[("NO_COLOR", "1")])).expect("resolve ui config");

        assert!(!config.color);
    }

    #[test]
    fn rejects_invalid_prompt_cache_retention() {
        let error = parse_prompt_cache_retention("forever").expect_err("invalid retention");

        assert!(
            error
                .to_string()
                .contains("invalid FERRIX_PROMPT_CACHE_RETENTION")
        );
    }

    fn env_from(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + 'static {
        let vars = vars
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect::<BTreeMap<_, _>>();
        move |name| vars.get(name).cloned()
    }

    fn temp_workspace() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ferrix-model-{}", Uuid::new_v4()))
    }

    fn write_config(workspace: &Path, contents: &str) {
        let ferrix_dir = workspace.join(".ferrix");
        fs::create_dir_all(&ferrix_dir).expect("create .ferrix");
        fs::write(ferrix_dir.join("config.json"), contents).expect("write config");
    }
}
