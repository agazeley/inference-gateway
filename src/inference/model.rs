use crate::inference::{
    errors::{InferenceError, Result},
    inputs::{InputBuilder, ModelOutput, OutputBuilder},
};
use log::debug;
use ort::{
    execution_providers::{CUDAExecutionProvider, ExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
};
use serde::Serialize;

const DEFAULT_MODEL_NAME: &str = "unknown";
const DEFAULT_MODEL_PATH: &str = "data/model.onnx";

#[derive(Debug, Serialize)]
pub struct AutoRegressiveModelConfig {
    pub model_name: String,
    pub model_path: String,
    pub intra_threads: usize,
    #[serde(skip_serializing)]
    pub optimization_level: GraphOptimizationLevel,
}

impl Clone for AutoRegressiveModelConfig {
    fn clone(&self) -> Self {
        Self {
            model_name: self.model_name.clone(),
            model_path: self.model_path.clone(),
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
        let session = if self.model_path.starts_with("http") {
            builder.commit_from_url(self.model_path).map_err(|e| {
                InferenceError::ModelLoading(format!("Failed to commit from URL: {}", e))
            })?
        } else {
            builder.commit_from_file(self.model_path).map_err(|e| {
                InferenceError::ModelLoading(format!("Failed to commit from file: {}", e))
            })?
        };
        Ok(session)
    }
}

impl Default for AutoRegressiveModelConfig {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_MODEL_NAME.to_string(),
            model_path: DEFAULT_MODEL_PATH.to_string(),
            intra_threads: 4,
            optimization_level: GraphOptimizationLevel::Level3,
        }
    }
}

pub struct AutoRegressiveModel {
    pub name: String,
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

        Ok(Self {
            session: cfg.build_session()?,
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
