use log::debug;
use rand::Rng;
use serde_derive::Deserialize;
use serde_derive::Serialize;

use crate::inference::{
    errors::{InferenceError, Result},
    model::AutoRegressiveModel,
    tokenization::Tokenizer,
};

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
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: usize,
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
            max_tokens: 1024,
            temperature: 0.5,
            top_k: 5,
            top_p: Some(1.0),
        }
    }
}

pub struct LLM {
    model: AutoRegressiveModel,
    tokenizer: Tokenizer,
}

impl LLM {
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
        let input = self.tokenizer.chat_template
            .format_request(&chat, true,None,None, None)?;
        self.generate(input, params)
    }

    pub fn generate(&mut self, input: String, params: TextGenerationParameters) -> Result<String> {
        debug!("Generating from: {:?}", params);
        debug!("Input: {:?}", input);
        let mut input_tokens = self.tokenize_input(&input)?;
        let mut generated_tokens = Vec::new();
        let mut rng = rand::rng();

        for _ in 0..params.max_tokens {
            let output = self.model.run(&input_tokens)?; // TODO: this probaby will not work
            let logits = output.logits()?;
            let tokens = self.process_logits(logits, &params)?;
            if tokens.is_empty() {
                break;
            }
            let top_k_size = params.top_k.min(tokens.len());
            let token = tokens[rng.random_range(0..top_k_size)].0 as i64;
            input_tokens.push(token);
            generated_tokens.push(token as u32);
        }

        self.tokenizer.decode(&generated_tokens, true)
    }

    fn tokenize_input(&self, text: &str) -> Result<Vec<i64>> {
        let tokens = self.tokenizer.encode(text, false)?;
        Ok(tokens.get_ids().iter().map(|i| *i as i64).collect())
    }

    fn process_logits(
        &self,
        logits: Vec<(usize, f32)>,
        params: &TextGenerationParameters,
    ) -> Result<Vec<(usize, f32)>> {
        // Temperature scaling:
        // Divide logits by temperature before softmax.
        // T < 1.0 → sharper distribution (more deterministic).
        // T > 1.0 → flatter distribution (more random).
        // Ref: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.temperature
        let inv_temp = 1.0 / params.temperature;

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
        if let Some(top_p) = params.top_p
            && top_p != 1.0
        {
            if top_p < 1.0 {
                let mut cumulative = 0.0;
                probs.retain(|&(_, p)| {
                    cumulative += p;
                    cumulative <= top_p
                });

                // Optional: renormalize after filtering so probs sum to 1.0
                let sum_p: f32 = probs.iter().map(|&(_, p)| p).sum();
                if sum_p > 0.0 {
                    for (_, p) in probs.iter_mut() {
                        *p /= sum_p;
                    }
                }
            } else {
                return Err(InferenceError::TextGenerationError(format!(
                    "Invalid top-p parameter '{}'",
                    top_p
                )));
            }
        }
        Ok(probs)
    }
}
