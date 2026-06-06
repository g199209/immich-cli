use crate::config::LlmConfig;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Minimal chat-completion backend the `ask` command talks to. Trait so
/// tests can swap in a fake without touching HTTP.
pub trait ChatBackend {
    /// Send the messages, force JSON output, return the assistant message
    /// content (still a string — caller parses the JSON).
    fn chat_json(&self, messages: &[Message]) -> Result<String>;
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: &'static str,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system",
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user",
            content: content.into(),
        }
    }
}

/// OpenAI-compatible client. Works with OneAPI, LiteLLM, OpenAI itself,
/// Ollama's /v1 shim, etc. The gateway decides which model `model` maps to.
pub struct OpenAiClient {
    base_url: String,
    api_key: String,
    model: String,
    http: reqwest::blocking::Client,
}

impl OpenAiClient {
    pub fn new(cfg: &LlmConfig) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .context("failed to build LLM HTTP client")?;
        Ok(Self {
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            http,
        })
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    temperature: f32,
    response_format: ResponseFormat,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: String,
}

impl ChatBackend for OpenAiClient {
    fn chat_json(&self, messages: &[Message]) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = ChatRequest {
            model: &self.model,
            messages,
            temperature: 0.0,
            response_format: ResponseFormat {
                kind: "json_object",
            },
        };
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .with_context(|| format!("HTTP POST {url} failed"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            bail!("LLM returned {status}: {body}");
        }
        let parsed: ChatResponse = resp
            .json()
            .with_context(|| format!("failed to decode LLM response from {url}"))?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("LLM returned no choices"))?;
        if content.trim().is_empty() {
            bail!("LLM returned empty content");
        }
        Ok(content)
    }
}
