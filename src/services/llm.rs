use log::debug;
use log::trace;
use rand::Rng;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use serde_json::Value;

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
        let input = self
            .tokenizer
            .chat_template
            .format_request(&chat, false, None, None, None)?;
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

        self.generate_inner(input, max_tokens, min_p, temperature, top_k, top_p)
    }

    fn generate_inner(
        &mut self,
        input: String,
        max_tokens: i32,
        min_p: f32,
        temperature: f32,
        top_k: i64,
        top_p: f32,
    ) -> Result<String> {
        let mut input_tokens = self.tokenize_input(&input)?;
        let mut generated_tokens = Vec::new();
        let mut rng = rand::rng();
        for _ in 0..max_tokens {
            let output = self.model.run(&input_tokens)?; // TODO: this probaby will not work
            let logits = output.logits()?;
            let tokens = self.process_logits(logits, temperature, top_p)?;
            if tokens.is_empty() {
                break;
            }
            let top_k_size = top_k.min(tokens.len() as i64);
            let selected = tokens[rng.random_range(0..top_k_size as usize)];
            let token = selected.0 as i64;
            let probability = selected.1;
            if probability < min_p {
                debug!(
                    "Improbable token generated (token={}, prob={}, count={})",
                    token,
                    probability,
                    generated_tokens.len()
                );
                // continue;
            }

            if self.end_of_sequence(token) {
                debug!("EOS found on token {}", generated_tokens.len());
                break;
            }
            input_tokens.push(token);
            generated_tokens.push(token as u32);
        }
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

    fn end_of_sequence(&self, token: i64) -> bool {
        if let Some(eos_token) = &self.model.generate_cfg.eos_token_id {
            match eos_token {
                Value::Array(data) => data.iter().any(|v| v.as_i64() == Some(token)),
                Value::Number(val) => val.as_i64() == Some(token),
                _ => false,
            }
        } else {
            false
        }
    }

    fn process_logits(
        &self,
        logits: Vec<(usize, f32)>,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<(usize, f32)>> {
        // Temperature scaling:
        // Divide logits by temperature before softmax.
        // T < 1.0 → sharper distribution (more deterministic).
        // T > 1.0 → flatter distribution (more random).
        // Ref: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.temperature
        let inv_temp = 1.0 / temperature;

        // Numerical stability trick:
        // Subtract the maximum logit to avoid overflow in exp().
        // Ref: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        let max_logit = logits
            .iter()
            .map(|(_, logit)| *logit * inv_temp)
            .fold(f32::NEG_INFINITY, f32::max);

        // Apply exp((logit / T) - max_logit) for each token.
        // This transforms logits into positive unnormalized values.
        // Ref: https://en.wikipedia.org/wiki/Softmax_function
        let exp_vals: Vec<(usize, f32)> = logits
            .iter()
            .map(|(id, logit)| {
                let scaled = (*logit * inv_temp) - max_logit;
                (*id, scaled.exp())
            })
            .collect();

        // Normalize to probabilities: p_i = exp_i / Σ exp_j
        // Ensures probabilities sum to 1.
        let sum: f32 = exp_vals.iter().map(|(_, val)| *val).sum();
        let mut probs: Vec<(usize, f32)> = exp_vals
            .into_iter()
            .map(|(id, val)| (id, val / sum))
            .collect();

        // Sort tokens by probability (descending) so top-k or top-p
        // filtering can be applied easily downstream.
        // Ref: https://huggingface.co/blog/how-to-generate
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        // Apply top-p (nucleus) filtering if requested
        // Value of 1.0 means we are not filtering.
        if top_p == 1.0 {
            return Ok(probs);
        }

        if 0.0 < top_p && top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut cutoff = 0;
            for &(_, p) in &probs {
                // Always return >=1 elements
                if cumulative + p > top_p && cutoff > 0 {
                    break;
                }
                cumulative += p;
                cutoff += 1;
            }
            probs.truncate(cutoff);

            // Optional: renormalize after filtering so probs sum to 1.0
            let sum_p: f32 = probs.iter().map(|&(_, p)| p).sum();
            if sum_p > 0.0 {
                for (_, p) in probs.iter_mut() {
                    *p /= sum_p;
                }
            }
            trace!("{:?}", probs);
        } else {
            return Err(InferenceError::TextGenerationError(format!(
                "Invalid top-p parameter '{}'",
                top_p
            )));
        }

        Ok(probs)
    }
}
