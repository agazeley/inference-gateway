use log::{debug, info, trace};
use ort::{inputs, session::builder::GraphOptimizationLevel, value::TensorRef};
use rand::Rng;
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::inference::{
    errors::{InferenceError, Result},
    model::{TextGenerationModel, TextGenerationModelConfig},
    tokenization::{TextGenerationTokenizerConfig, new_tokenizer},
};

// TODO: Need a way to configure the defaults dynamically
// https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
pub fn load_default_model() -> Result<TextGenerationModel> {
    info!("Loading model...");
    let model_cfg = TextGenerationModelConfig {
        model_name: "gpt2".to_string(),
        model_path: "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/gpt2.onnx".to_string(),
        intra_threads: 4,
        optimization_level: GraphOptimizationLevel::Level3,
    };

    match TextGenerationModel::new(model_cfg) {
        Ok(m) => {
            debug!("Model loaded");
            Ok(m)
        }
        Err(e) => Err(e),
    }
}

pub fn load_default_tokenizer() -> Result<Tokenizer> {
    info!("Loading tokenizer...");
    let cfg = TextGenerationTokenizerConfig {
        pretrained_identifier: Some("openai-community/gpt2".to_string()),
        filepath: None,
    };
    match new_tokenizer(cfg) {
        Ok(t) => {
            debug!("Tokenizer loaded");
            Ok(t)
        }
        Err(e) => Err(e),
    }
}

#[derive(Serialize)]
pub struct TextGenerationConfig {
    pub text: String,
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: Option<f32>,
}

impl TextGenerationConfig {
    pub fn new(text: String) -> Self {
        Self {
            text,
            max_tokens: 1024,
            temperature: 0.5,
            top_k: 5,
            top_p: Some(1.0),
        }
    }
}

pub struct LLM {
    model: TextGenerationModel,
    tokenizer: Tokenizer,
}

impl LLM {
    pub fn new(model: TextGenerationModel, tokenizer: Tokenizer) -> Self {
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
        trace!("Processing {} logits", logits.len());
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

        trace!("Processed down to {} logits", probs.len());
        Ok(probs)
    }

    pub fn generate(&mut self, req: TextGenerationConfig) -> Result<String> {
        debug!(
            "Generating from: {:?}",
            serde_json::to_string(&req).unwrap()
        );
        let mut input_tokens = self.tokenize_input(&req.text)?;
        let mut generated_tokens = Vec::new();
        let mut rng = rand::rng();

        for _ in 0..req.max_tokens {
            let logits = self.run_inference(&input_tokens)?;
            let tokens = self.process_logits(logits, &req)?;
            if tokens.is_empty() {
                break;
            }
            let token = self.sample_token(&tokens, req.top_k, &mut rng);
            input_tokens.push(token);
            generated_tokens.push(token as u32);
        }

        self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| InferenceError::Tokenization(e.to_string()))
    }

    fn run_inference(&mut self, tokens: &[i64]) -> Result<Vec<(usize, f32)>> {
        // Raw tensor construction takes a tuple of (shape, data).
        // The model expects our input to have shape [B, _, S]
        //
        //    Shape [B, _, S] here:
        //    B = 1 (batch size is 1 request)
        //    _ = 1 (sometimes used as number of heads/layers, depends on model’s input signature)
        //    S = tokens.len() (sequence length = how many tokens you’ve generated so far, including prompt).
        //

        // Our implementation:
        let input = TensorRef::from_array_view((vec![1, 1, tokens.len() as i64], tokens))?;
        let outputs = self.model.run(inputs![input])?;
        let (dim, mut probabilities) = match outputs["output1"].try_extract_tensor() {
            Ok((dim, probabilities)) => (dim, probabilities),
            Err(e) => {
                return Err(InferenceError::TextGenerationError(format!(
                    "error extracting tensor: {}",
                    e
                )));
            }
        };

        // The output tensor will have shape [B, _, S, V]
        // We want only the probabilities for the last token in this sequence, which will be the next most likely token
        // according to the model
        // Output shape = [B, _, S, V]:
        //    B = 1 batch
        //    _ = 1 dummy dimension
        //    S = sequence length
        //    V = vocab size
        // That means at each time step, you get a vector of length V representing the logits/probabilities of every possible next token.
        // You don’t care about the probabilities for earlier positions (you already generated those).
        //
        // So you slice into the last V chunk:
        // The output tensor will have shape [B, _, S, V]
        // We want only the probabilities for the last token in this sequence, which will be the next most likely token
        // according to the model
        let (seq_len, vocab_size) = (dim[2] as usize, dim[3] as usize);
        probabilities = &probabilities[(seq_len - 1) * vocab_size..];
        Ok(probabilities.iter().copied().enumerate().collect())
    }

    fn tokenize_input(&self, text: &str) -> Result<Vec<i64>> {
        let tokens = self.tokenizer.encode(text, false).unwrap();
        Ok(tokens.get_ids().iter().map(|i| *i as i64).collect())
    }

    fn sample_token(
        &self,
        probabilities: &[(usize, f32)],
        top_k: usize,
        rng: &mut impl Rng,
    ) -> i64 {
        let top_k_size = top_k.min(probabilities.len());
        probabilities[rng.random_range(0..top_k_size)].0 as i64
    }
}
