use log::debug;
use serde_derive::Deserialize;
use serde_derive::Serialize;

use crate::inference::{
    errors::{InferenceError, Result},
    model::AutoRegressiveModel,
    tokenization::Tokenizer,
};

const DEFAULT_MAX_TOKENS: i32 = 100;
const DEFAULT_MIN_P: f32 = 0.5;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_TOP_K: i64 = 5;
const DEFAULT_TOP_P: f32 = 1.0;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct TextGenerationParameters {
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i64>,
    pub top_p: Option<f32>,
}

impl Default for TextGenerationParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGenerationParameters {
    pub fn new() -> Self {
        Self {
            // max_tokens: 1024,
            // temperature: 0.5,
            // top_k: 5,
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: Some(1.0),
        }
    }
}

pub struct LLMService {
    model: AutoRegressiveModel,
    tokenizer: Tokenizer,
}

impl LLMService {
    pub fn new(model: AutoRegressiveModel, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    // Returns the name of the underlying model.
    ///
    /// # Returns
    ///
    /// A cloned string containing the model's name
    pub fn model_name(&self) -> String {
        self.model.name.clone()
    }

    pub fn chat(&mut self, chat: ChatRequest, params: TextGenerationParameters) -> Result<String> {
        // TODO: we need to get the stuff going for the tokenizer to handle BOS,EOS,UNK tokens
        let input = self.tokenizer.chat_template.format_request(
            &chat,
            false,
            self.model
                .generate_cfg
                .bos_token_id
                .as_ref()
                .map(|v| v.to_string()),
            self.model
                .generate_cfg
                .eos_token_id
                .as_ref()
                .map(|v| v.to_string()),
            None,
        )?;
        self.generate(input, params)
    }

    pub fn generate(&mut self, input: String, params: TextGenerationParameters) -> Result<String> {
        debug!("Generating from: {:?}", params);
        debug!("Input: {:?}", input);
        let max_tokens = params.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let min_p = self.model.generate_cfg.min_p.unwrap_or(DEFAULT_MIN_P);
        let temperature = params.temperature.unwrap_or(
            self.model
                .generate_cfg
                .temperature
                .unwrap_or(DEFAULT_TEMPERATURE),
        );
        let top_k = params
            .top_k
            .unwrap_or(self.model.generate_cfg.top_k.unwrap_or(DEFAULT_TOP_K));
        let top_p = params
            .top_p
            .unwrap_or(self.model.generate_cfg.top_p.unwrap_or(DEFAULT_TOP_P));

        let input_tokens = self.tokenize_input(&input)?;
        let generated_tokens =
            self.model
                .generate(input_tokens, max_tokens, min_p, temperature, top_k, top_p)?;
        self.decode_output(&generated_tokens)
    }

    fn decode_output(&self, generated_tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(generated_tokens, true) {
            Ok(output) => Ok(output.trim_start().to_string()),
            Err(e) => Err(InferenceError::TextGenerationError(format!(
                "Failed to decode output: {}",
                e
            ))),
        }
    }

    fn tokenize_input(&self, text: &str) -> Result<Vec<i64>> {
        let tokens = self.tokenizer.encode(text, false)?;
        Ok(tokens.get_ids().iter().map(|i| *i as i64).collect())
    }
}
