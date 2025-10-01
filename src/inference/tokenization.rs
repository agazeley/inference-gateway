use crate::inference::{
    errors::{InferenceError, Result},
    prompting::{CHAT, ChatTemplate},
};
use tokenizers::{Encoding, Tokenizer as TokenizerBase};

const DEFAULT_TOKENIZER_PATH: &str = "data/tokenizer.json";

#[derive(Debug)]
pub struct TokenizerConfig {
    pub filepath: Option<String>,
    pub pretrained_identifier: Option<String>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            filepath: Some(DEFAULT_TOKENIZER_PATH.to_string()),
            pretrained_identifier: None,
        }
    }
}

pub struct Tokenizer {
    inner: TokenizerBase,
    pub chat_template: Box<dyn ChatTemplate>,
}

// TODO: unique error messages for exposed methods?
impl Tokenizer {
    pub fn new(cfg: TokenizerConfig, chat_template: Option<Box<dyn ChatTemplate>>) -> Result<Tokenizer> {
        if cfg.pretrained_identifier.is_none() && cfg.filepath.is_none() {
            return Err(InferenceError::Configuration(
                "missing tokenizer file or identifier".to_string(),
            ));
        }
        let chat_template = chat_template.unwrap_or(Box::new(CHAT));

        if cfg.filepath.is_some() {
            let path = cfg.filepath.unwrap();
            let tokenizer = TokenizerBase::from_file(path);
            match tokenizer {
                Ok(inner) => {
                    return Ok(Self {
                        inner,
                        chat_template,
                    });
                }
                Err(e) => return Err(InferenceError::Tokenization(e.to_string())),
            }
        }

        let identifier = cfg.pretrained_identifier.unwrap();
        let inner = TokenizerBase::from_pretrained(identifier, None)
            .map_err(|e| InferenceError::Tokenization(e.to_string()))?;

        Ok(Self {
            inner,
            chat_template,
        })
    }

    // Re-expose selected methods
    
    pub fn tokenize_to_ids(&self, text: &str) -> Result<Vec<u32>> {
        let enc: Encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| InferenceError::Tokenization(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        self.inner
            .encode(text, add_special_tokens)
            .map_err(|e| InferenceError::Tokenization(e.to_string()))
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| InferenceError::Tokenization(e.to_string()))
    }
}

// Optional: forward method calls via Deref
impl std::ops::Deref for Tokenizer {
    type Target = TokenizerBase;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
