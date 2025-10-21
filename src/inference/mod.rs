use log::{error, info};
use ort::session::builder::GraphOptimizationLevel;

use crate::inference::errors::Result;
use crate::inference::models::{
    config::{HfModelLoadConfig, ModelConfig, ModelLoadConfig},
    text_generation::TextGenerationModel,
};
use crate::inference::prompting::{ChatTemplate, JinjaChatTemplate};
use crate::inference::tokenization::{Tokenizer, TokenizerConfig};
use crate::utils::get_env;

pub mod errors;
pub mod models;
pub mod tokenization;

mod inputs;
mod prompting;
mod tensors;

const DEFAULT_MODEL_NAME_VAR: &str = "INFERENCE_DEFAULT_MODEL_NAME";
const DEFAULT_HF_MODEL_ID: &str = "INFERENCE_DEFAULT_HF_MODEL_ID";
const DEFAULT_HF_MODEL_FILENAME: &str = "INFERENCE_DEFAULT_HF_MODEL_FILENAME";

// https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
pub fn load_default_model() -> Result<TextGenerationModel> {
    let model_id = get_env(DEFAULT_HF_MODEL_ID, "openai-community/gpt2");
    let model_filename = match get_env(DEFAULT_HF_MODEL_FILENAME, "onnx/decoder_model.onnx") {
        s if s.is_empty() => None,
        s => Some(s),
    };
    let model_cfg = ModelLoadConfig::Hf(HfModelLoadConfig {
        id: model_id,
        filename: model_filename,
    });

    let cfg = ModelConfig {
        model_name: get_env(DEFAULT_MODEL_NAME_VAR, "gpt2"),
        model_cfg,
        intra_threads: 4,
        optimization_level: GraphOptimizationLevel::Level3,
    };
    info!("Loading model: {:?}", cfg);
    TextGenerationModel::new(cfg)
}

pub fn load_default_tokenizer() -> Result<Tokenizer> {
    let model_id = get_env(DEFAULT_HF_MODEL_ID, "openai-community/gpt2");
    let cfg = TokenizerConfig {
        pretrained_identifier: Some(model_id.clone()),
        filepath: None,
    };

    let chat_template: Option<Box<dyn ChatTemplate>> = match JinjaChatTemplate::from_hf(model_id) {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            error!("{}", e);
            None
        }
    };
    info!("Loading tokenizer: {:?}", cfg);
    Tokenizer::new(cfg, chat_template)
}
