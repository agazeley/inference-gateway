use crate::inference::errors::{InferenceError, Result};

use tokenizers::Tokenizer;

const DEFAULT_TOKENIZER_PATH: &str = "data/tokenizer.json";

#[derive(Debug)]
pub struct TextGenerationTokenizerConfig {
    pub filepath: Option<String>,
    pub pretrained_identifier: Option<String>,
}

impl Default for TextGenerationTokenizerConfig {
    fn default() -> Self {
        Self {
            filepath: Some(DEFAULT_TOKENIZER_PATH.to_string()),
            pretrained_identifier: None,
        }
    }
}

pub fn new_tokenizer(cfg: TextGenerationTokenizerConfig) -> Result<Tokenizer> {
    if cfg.pretrained_identifier.is_none() && cfg.filepath.is_none() {
        return Err(InferenceError::Configuration(
            "missing tokenizer file or identifier".to_string(),
        ));
    }

    if cfg.filepath.is_some() {
        let path = cfg.filepath.unwrap();
        let tokenizer = Tokenizer::from_file(path);
        match tokenizer {
            Ok(t) => return Ok(t),
            Err(e) => return Err(InferenceError::Tokenization(e.to_string())),
        }
    }

    let identifier = cfg.pretrained_identifier.unwrap();
    match Tokenizer::from_pretrained(identifier, None) {
        Ok(t) => Ok(t),
        Err(e) => Err(InferenceError::Tokenization(e.to_string())),
    }
}
