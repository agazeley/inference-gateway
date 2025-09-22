use log::{debug, info};
use ort::{inputs, session::builder::GraphOptimizationLevel, value::TensorRef};
use rand::Rng;
use tokenizers::Tokenizer;

use crate::inference::{
    errors::Result,
    model::{TextGenerationModel, TextGenerationModelConfig},
    tokenization::{TextGenerationTokenizerConfig, new_tokenizer},
};

// TODO: Need a way to configure the defaults dynamically
// https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
pub fn load_default_model() -> Result<TextGenerationModel> {
    info!("Loading model...");
    let model_cfg = TextGenerationModelConfig {
        model_path: "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/gpt2.onnx".to_string(),
        intra_threads: 4,
        optimization_level: GraphOptimizationLevel::Level3,
    };

    let m = TextGenerationModel::new(model_cfg);
    debug!("Model loaded");
    m
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

pub struct TextGenerationConfig {
    text: String,
    max_tokens: Option<i32>,
    top_k: Option<usize>,
}

impl TextGenerationConfig {
    pub fn new(text: String) -> Self {
        Self {
            text,
            max_tokens: Some(128),
            top_k: Some(5),
        }
    }
}

pub struct LLM {
    model: TextGenerationModel,
    tokenizer: Tokenizer,
    max_tokens: i32,
    top_k: usize,
}

impl LLM {
    pub fn new(
        model: TextGenerationModel,
        tokenizer: Tokenizer,
        max_tokens: i32,
        top_k: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            max_tokens,
            top_k,
        }
    }

    pub fn generate(&mut self, req: TextGenerationConfig) -> Result<String> {
        debug!("Generating from: {}", req.text);

        let max_tokens = match req.max_tokens {
            Some(val) => val,
            None => self.max_tokens,
        };
        let top_k = match req.top_k {
            Some(val) => val,
            None => self.top_k,
        };

        let tokens = self.tokenizer.encode(req.text, false).unwrap();
        let mut tokens = tokens
            .get_ids()
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>();
        let mut rng = rand::rng();
        let mut generated = String::new();
        for _ in 0..max_tokens {
            // Raw tensor construction takes a tuple of (shape, data).
            // The model expects our input to have shape [B, _, S]
            let input =
                TensorRef::from_array_view((vec![1, 1, tokens.len() as i64], tokens.as_slice()))?;
            let outputs = self.model.run(inputs![input])?;
            let (dim, mut probabilities) = outputs["output1"].try_extract_tensor()?;

            // The output tensor will have shape [B, _, S, V]
            // We want only the probabilities for the last token in this sequence, which will be the next most likely token
            // according to the model
            let (seq_len, vocab_size) = (dim[2] as usize, dim[3] as usize);
            probabilities = &probabilities[(seq_len - 1) * vocab_size..];

            // Sort each token by probability
            let mut probabilities: Vec<(usize, f32)> =
                probabilities.iter().copied().enumerate().collect();
            probabilities
                .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

            // Sample using top-k sampling
            let token = probabilities[rng.random_range(0..=top_k)].0 as i64;

            // Add our generated token to the input sequence
            tokens.push(token);

            let token_str = self.tokenizer.decode(&[token as u32], true).unwrap();
            generated.push_str(&token_str);
        }
        Ok(generated)
    }
}
