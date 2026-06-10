use crate::config::LlmConfig;
use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Minimal chat-completion backend the `ask` command talks to. Trait so
/// tests can swap in a fake without touching HTTP.
pub trait ChatBackend {
    /// Send the messages, force JSON output, return the assistant message
    /// content (still a string — caller parses the JSON).
    fn chat_json(&self, messages: &[Message]) -> Result<String>;
}

/// Vision captioning backend used by `update-descriptions`. Single-shot:
/// system + user text + one inline image, returns free-form text.
pub trait CaptionLlm {
    /// `image_bytes` is the raw JPEG (or PNG/WebP) the model sees.
    /// `mime` is something like `"image/jpeg"`. Caller bounds the size.
    /// `system_prompt` and `user_prompt` are sent as text; `max_tokens`
    /// is the upper bound on completion (large for reasoning models).
    fn caption(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        image_bytes: &[u8],
        mime: &str,
        max_tokens: u32,
    ) -> Result<String>;
}

/// Vision backend used by `dedup` to pick the best photo from a group of
/// near-duplicates. Single shot: system + user text + N inline images,
/// returns the assistant's raw message content (caller parses as JSON
/// since `response_format` is forced to `json_object`).
pub trait MultiImageVisionLlm {
    /// `images` is a list of `(bytes, mime)` pairs; the prompt should
    /// refer to them by 0-based index in the order given.
    fn pick_best(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        images: &[(Vec<u8>, &str)],
        max_tokens: u32,
    ) -> Result<String>;
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
    /// Text-only model. Used by [`ChatBackend::chat_json`].
    model: String,
    /// Optional vision model. Used by [`CaptionLlm::caption`]. Calling
    /// `caption` without configuring this errors at runtime.
    vision_model: Option<String>,
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
            vision_model: cfg.vision_model.clone(),
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

// Multimodal request: messages are a mix of text and image blocks. The
// model field is the vision model. `response_format` is optional so the
// same struct backs both the free-form caption call and the JSON-only
// dedup pick.
#[derive(Serialize)]
struct VisionRequest<'a> {
    model: &'a str,
    messages: Vec<VisionMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct VisionMessage<'a> {
    role: &'static str,
    /// Either a plain string (for system) or an array of typed parts (for
    /// user). serde_json::Value covers both shapes.
    content: serde_json::Value,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<&'a ()>,
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
    /// May be null for reasoning models that ran out of tokens before
    /// producing visible output (mimo-v2.5, o1-style). We surface that
    /// as a clear error rather than letting the deserializer fail.
    #[serde(default)]
    content: Option<String>,
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
        unpack_chat_content(resp, &url)
    }
}

impl CaptionLlm for OpenAiClient {
    fn caption(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        image_bytes: &[u8],
        mime: &str,
        max_tokens: u32,
    ) -> Result<String> {
        let model = self
            .vision_model
            .as_deref()
            .ok_or_else(|| anyhow!("config.llm.vision_model is not set"))?;
        let url = format!("{}/v1/chat/completions", self.base_url);
        let data_url = format!("data:{mime};base64,{}", B64.encode(image_bytes));

        let body = VisionRequest {
            model,
            temperature: 0.0,
            max_tokens,
            response_format: None,
            messages: vec![
                VisionMessage {
                    role: "system",
                    content: serde_json::Value::String(system_prompt.to_string()),
                    _phantom: std::marker::PhantomData,
                },
                VisionMessage {
                    role: "user",
                    content: serde_json::json!([
                        { "type": "text", "text": user_prompt },
                        { "type": "image_url", "image_url": { "url": data_url } },
                    ]),
                    _phantom: std::marker::PhantomData,
                },
            ],
        };
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .with_context(|| format!("HTTP POST {url} (vision) failed"))?;
        unpack_chat_content(resp, &url)
    }
}

impl MultiImageVisionLlm for OpenAiClient {
    fn pick_best(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        images: &[(Vec<u8>, &str)],
        max_tokens: u32,
    ) -> Result<String> {
        let model = self
            .vision_model
            .as_deref()
            .ok_or_else(|| anyhow!("config.llm.vision_model is not set"))?;
        if images.is_empty() {
            bail!("pick_best called with zero images");
        }
        let url = format!("{}/v1/chat/completions", self.base_url);

        // Build the multipart user content: a text block, then one
        // labelled text+image pair per candidate so the model sees the
        // 0-based index it should reference in its JSON answer.
        let mut content_parts = Vec::with_capacity(1 + images.len() * 2);
        content_parts.push(serde_json::json!({ "type": "text", "text": user_prompt }));
        for (i, (bytes, mime)) in images.iter().enumerate() {
            let data_url = format!("data:{mime};base64,{}", B64.encode(bytes));
            content_parts.push(serde_json::json!({
                "type": "text",
                "text": format!("候选 {i}："),
            }));
            content_parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": { "url": data_url },
            }));
        }

        let body = VisionRequest {
            model,
            temperature: 0.0,
            max_tokens,
            response_format: Some(ResponseFormat {
                kind: "json_object",
            }),
            messages: vec![
                VisionMessage {
                    role: "system",
                    content: serde_json::Value::String(system_prompt.to_string()),
                    _phantom: std::marker::PhantomData,
                },
                VisionMessage {
                    role: "user",
                    content: serde_json::Value::Array(content_parts),
                    _phantom: std::marker::PhantomData,
                },
            ],
        };
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .with_context(|| format!("HTTP POST {url} (vision pick) failed"))?;
        unpack_chat_content(resp, &url)
    }
}

fn unpack_chat_content(resp: reqwest::blocking::Response, url: &str) -> Result<String> {
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
        .ok_or_else(|| anyhow!("LLM returned no choices"))?
        .message
        .content
        .ok_or_else(|| {
            anyhow!(
                "LLM returned null content — reasoning model likely \
                 hit max_tokens before emitting output; try raising it"
            )
        })?;
    if content.trim().is_empty() {
        bail!("LLM returned empty content");
    }
    Ok(content)
}
