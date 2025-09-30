use log::{debug, info};
use ort::session::builder::GraphOptimizationLevel;
use rand::Rng;
use tokenizers::Tokenizer;

use crate::inference::{
    errors::{InferenceError, Result},
    model::{AutoRegressiveModel, AutoRegressiveModelConfig},
    prompting::{NO_OP, PromptTemplate},
    tokenization::{TextGenerationTokenizerConfig, new_tokenizer},
};

const DEFAULT_MODEL_NAME_VAR: &str = "INFERENCE_DEFAULT_MODEL_NAME";
const DEFAULT_MODEL_PATH_VAR: &str = "INFERENCE_DEFAULT_MODEL_PATH";
const DEFAULT_TOKENIZER_VAR: &str = "INFERENCE_DEFAULT_TOKENIZER";

// https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
pub fn load_default_model() -> Result<AutoRegressiveModel> {
    let cfg = AutoRegressiveModelConfig {
        model_name: get_env(DEFAULT_MODEL_NAME_VAR, "gpt2"),
        model_path: get_env(
            DEFAULT_MODEL_PATH_VAR,
            "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/gpt2.onnx",
        ),
        intra_threads: 4,
        optimization_level: GraphOptimizationLevel::Level3,
    };
    info!("Loading model: {:?}", cfg);
    AutoRegressiveModel::new(cfg)
}

pub fn load_default_tokenizer() -> Result<Tokenizer> {
    let cfg = TextGenerationTokenizerConfig {
        pretrained_identifier: Some(get_env(DEFAULT_TOKENIZER_VAR, "openai-community/gpt2")),
        filepath: None,
    };
    info!("Loading tokenizer: {:?}", cfg);
    new_tokenizer(cfg)
}

#[derive(Debug)]
pub struct TextGenerationConfig {
    pub text: String,
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: Option<f32>,
    pub template: Option<PromptTemplate>,
}

impl TextGenerationConfig {
    pub fn new(text: String) -> Self {
        Self {
            text,
            max_tokens: 1024,
            temperature: 0.5,
            top_k: 5,
            top_p: Some(1.0),
            template: None,
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

    fn process_logits(
        &self,
        logits: Vec<(usize, f32)>,
        params: &TextGenerationConfig,
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

    pub fn generate(&mut self, req: TextGenerationConfig) -> Result<String> {
        debug!("Generating from: {:?}", req);
        let template = req.template.as_ref().unwrap_or(&NO_OP);
        let input = template.apply(&req.text, "");
        let mut input_tokens = self.tokenize_input(&input)?;
        let mut generated_tokens = Vec::new();
        let mut rng = rand::rng();

        for _ in 0..req.max_tokens {
            let output = self.model.run(&input_tokens)?; // TODO: this probaby will not work
            let logits = output.logits()?;
            let tokens = self.process_logits(logits, &req)?;
            if tokens.is_empty() {
                break;
            }
            let top_k_size = req.top_k.min(tokens.len());
            let token = tokens[rng.random_range(0..top_k_size)].0 as i64;
            input_tokens.push(token);
            generated_tokens.push(token as u32);
        }

        self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| InferenceError::Tokenization(e.to_string()))
    }

    fn tokenize_input(&self, text: &str) -> Result<Vec<i64>> {
        let tokens = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| InferenceError::Tokenization(e.to_string()))?;
        Ok(tokens.get_ids().iter().map(|i| *i as i64).collect())
    }
}

fn get_env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
