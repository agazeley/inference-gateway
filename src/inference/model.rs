use std::path::PathBuf;

use crate::inference::{
    errors::{InferenceError, Result},
    inputs::{InputBuilder, ModelOutput, OutputBuilder},
};
use hf_hub::api::sync::Api;
use log::debug;
use ort::{
    execution_providers::{CUDAExecutionProvider, ExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
};
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL_NAME: &str = "unknown";
const DEFAULT_MODEL_PATH: &str = "data/model.onnx";

// Subset of supported HF transforms Generation Config
// https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GenerationConfig {
    pub do_sample: Option<bool>,
    pub bos_token_id: Option<serde_json::Value>,
    pub eos_token_id: Option<serde_json::Value>, // i64 || vec<i64>
    pub pad_token_id: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub top_k: Option<i64>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub transformers_version: Option<String>,
}

impl GenerationConfig {
    pub fn from_string(data: String) -> Result<Self> {
        let c: Self = serde_json::from_str(&data).map_err(|e| {
            debug!("{}", data);
            InferenceError::ModelLoading(format!(
                "Failed to serialize generation config data: {}",
                e
            ))
        })?;
        Ok(c)
    }

    pub fn from_file(path: PathBuf) -> Result<Self> {
        let data = std::fs::read_to_string(&path).map_err(|e| {
            InferenceError::ModelLoading(format!("Failed to read generation config file: {}", e))
        })?;
        Self::from_string(data)
    }

    pub fn from_url(_url: String) -> Result<Self> {
        Ok(Self::default()) // TODO: impl
    }

    pub fn from_hf(model_id: String) -> Result<Self> {
        let api = Api::new().unwrap();
        let repo = api.model(model_id);
        let path = repo.get("generation_config.json").map_err(|e| {
            InferenceError::ModelLoading(format!("unable to fetch model generation config: {}", e))
        })?;
        Self::from_file(path)
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct RemoteModelLoadConfig {
    pub url: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct HfModelLoadConfig {
    pub id: String,
    pub filename: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct LocalModelLoadConfig {
    pub path: PathBuf,
}

#[derive(Debug, Serialize, Clone)]
pub enum ModelLoadConfig {
    Remote(RemoteModelLoadConfig),
    Hf(HfModelLoadConfig),
    Local(LocalModelLoadConfig),
}

impl ModelLoadConfig {
    fn get_generation_config(self) -> Result<GenerationConfig> {
        match self {
            ModelLoadConfig::Remote(cfg) => GenerationConfig::from_hf(cfg.url),
            ModelLoadConfig::Hf(cfg) => GenerationConfig::from_hf(cfg.id),
            ModelLoadConfig::Local(cfg) => GenerationConfig::from_file(cfg.path),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AutoRegressiveModelConfig {
    pub model_name: String,
    pub model_cfg: ModelLoadConfig,
    pub intra_threads: usize,
    #[serde(skip_serializing)]
    pub optimization_level: GraphOptimizationLevel,
}

impl Clone for AutoRegressiveModelConfig {
    fn clone(&self) -> Self {
        Self {
            model_name: self.model_name.clone(),
            model_cfg: self.model_cfg.clone(),
            intra_threads: self.intra_threads,
            optimization_level: match self.optimization_level {
                GraphOptimizationLevel::Disable => GraphOptimizationLevel::Disable,
                GraphOptimizationLevel::Level1 => GraphOptimizationLevel::Level1,
                GraphOptimizationLevel::Level2 => GraphOptimizationLevel::Level2,
                GraphOptimizationLevel::Level3 => GraphOptimizationLevel::Level3,
            },
        }
    }
}

impl AutoRegressiveModelConfig {
    /// Builds a new inference session based on the current `AutoRegressiveModelConfig`.
    ///
    /// This method configures the session builder with the specified optimization level and intra-thread count.
    /// If a CUDA execution provider is available, it attempts to register it for GPU acceleration, logging any errors encountered.
    /// The model is then loaded either from a URL or a local file, depending on the `model_path`.
    ///
    /// # Errors
    ///
    /// Returns an `InferenceError::ModelLoading` if any step in the session building process fails,
    /// including setting optimization level, intra-thread count, registering CUDA provider, or loading the model.
    ///
    /// # Returns
    ///
    /// On success, returns a constructed `Session` ready for inference.
    pub fn build_session(self) -> Result<Session> {
        let mut builder = Session::builder()?
            .with_optimization_level(self.optimization_level)
            .map_err(|e| {
                InferenceError::ModelLoading(format!("Failed to set optimization level: {}", e))
            })?
            .with_intra_threads(self.intra_threads)
            .map_err(|e| {
                InferenceError::ModelLoading(format!("Failed to set intra threads: {}", e))
            })?;

        // Register CUDA provider if available
        let cuda = CUDAExecutionProvider::default();
        if cuda.is_available().unwrap_or(false) {
            if let Err(e) = cuda.register(&mut builder) {
                log::error!("Failed to register CUDA provider: {}", e);
            } else {
                log::debug!("CUDA provider registered");
            }
        }

        // Load model from URL or local file
        let session = match self.model_cfg {
            ModelLoadConfig::Remote(cfg) => builder.commit_from_url(&cfg.url).map_err(|e| {
                InferenceError::ModelLoading(format!("Failed to commit from remote URL: {}", e))
            })?,
            ModelLoadConfig::Hf(cfg) => {
                let api = Api::new().unwrap();
                let repo = api.model(cfg.id);
                let filename = cfg.filename.unwrap_or("model.onnx".to_string());
                let path = repo.get(filename.as_str()).map_err(|e| {
                    InferenceError::ModelLoading(format!(
                        "Failed to fetch model from HuggingFace: {}",
                        e
                    ))
                })?;
                builder.commit_from_file(path).map_err(|e| {
                    InferenceError::ModelLoading(format!(
                        "Failed to commit from HuggingFace file: {}",
                        e
                    ))
                })?
            }
            ModelLoadConfig::Local(cfg) => builder.commit_from_file(cfg.path).map_err(|e| {
                InferenceError::ModelLoading(format!("Failed to commit from local file: {}", e))
            })?,
        };
        Ok(session)
    }
}

impl Default for AutoRegressiveModelConfig {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_MODEL_NAME.to_string(),
            model_cfg: ModelLoadConfig::Local(LocalModelLoadConfig {
                path: PathBuf::from(DEFAULT_MODEL_PATH),
            }),
            intra_threads: 4,
            optimization_level: GraphOptimizationLevel::Level3,
        }
    }
}

pub struct AutoRegressiveModel {
    pub name: String,
    pub generate_cfg: GenerationConfig,
    session: Session,
    input_builder: InputBuilder,
    output_builder: OutputBuilder,
}

/// Implementation of the `AutoRegressiveModel` struct, providing methods for model initialization and inference.
///
/// # Methods
///
/// - `new(cfg: AutoRegressiveModelConfig) -> Result<Self>`
///   Constructs a new `AutoRegressiveModel` instance using the provided configuration.
///   Initializes the model session, retrieves metadata (name, description, version), and sets up input/output builders.
///   Returns an error if session creation or metadata retrieval fails.
///
/// - `run<T: PrimitiveTensorElementType + Copy>(&mut self, tokens: &[i64]) -> Result<ModelOutput<T>>`
///   Runs inference on the model using the provided token sequence.
///   Builds input values, executes the session, and extracts the first output.
///   Returns the model output or an error if inference fails.
impl AutoRegressiveModel {
    pub fn new(cfg: AutoRegressiveModelConfig) -> Result<Self> {
        let name = cfg.model_name.clone();
        let session = cfg.clone().build_session()?;

        // Access metadata and signatures BEFORE creating the model to avoid borrowing issues
        let metadata = session.metadata().map_err(|e| {
            InferenceError::ModelMetadataError(format!("Failed to get metadata: {}", e))
        })?;
        let metadata_name = metadata.name().map_err(|e| {
            InferenceError::ModelMetadataError(format!("Failed to get model name: {}", e))
        })?;
        let description = metadata.description().map_err(|e| {
            InferenceError::ModelMetadataError(format!("Failed to get model description: {}", e))
        })?;
        let version = metadata.version().map_err(|e| {
            InferenceError::ModelMetadataError(format!("Failed to get model version: {}", e))
        })?;

        // Debug output
        let banner = "*".repeat(41);
        debug!("{}{}", banner, banner);
        debug!("Name: {}", metadata_name);
        debug!("Description: {}", description);
        debug!("Version: {}", version);
        let input_builder = InputBuilder::from_session_inputs(&session.inputs);
        let output_builder = OutputBuilder::from_session_outputs(&session.outputs);
        debug!("{}{}", banner, banner);

        let model_cfg = cfg.model_cfg.clone();
        let session = cfg.build_session()?;
        let generate_cfg = model_cfg.get_generation_config()?;

        Ok(Self {
            session,
            generate_cfg,
            input_builder,
            output_builder,
            name,
        })
    }

    pub fn run(&mut self, tokens: &[i64]) -> Result<ModelOutput<f32>> {
        let input_values = self.input_builder.build(tokens)?;
        match self.session.run(input_values) {
            Ok(r) => self.output_builder.get_first(r),
            Err(e) => Err(InferenceError::OnnxRuntime(e)),
        }
    }
}
