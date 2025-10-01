use log::{info, error};
use ort::session::builder::GraphOptimizationLevel;

use crate::inference::errors::Result;
use crate::inference::model::{AutoRegressiveModel, AutoRegressiveModelConfig};
use crate::inference::prompting::{ChatTemplate, JinjaChatTemplate};
use crate::inference::tokenization::{Tokenizer, TokenizerConfig};

pub mod errors;
pub mod llm;
pub mod prompting;

mod inputs;
mod model;
mod tensors;
mod tokenization;

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
    let model_id = get_env(DEFAULT_TOKENIZER_VAR, "openai-community/gpt2");
    let cfg = TokenizerConfig {
        pretrained_identifier: Some(model_id.clone()),
        filepath: None,
    };

    let chat_template:Option<Box<dyn ChatTemplate>> = match JinjaChatTemplate::from_hf(model_id){
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            error!("{}", e);
            None
        },
    };
    info!("Loading tokenizer: {:?}", cfg);
    Tokenizer::new(cfg, chat_template)
}

fn get_env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
