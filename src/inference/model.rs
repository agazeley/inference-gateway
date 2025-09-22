use crate::inference::errors::{InferenceError, Result};
use ort::{
    execution_providers::{CUDAExecutionProvider, ExecutionProvider},
    session::{Session, SessionInputs, SessionOutputs, builder::GraphOptimizationLevel},
};

const DEFAULT_MODEL_PATH: &str = "data/model.onnx";

#[derive(Debug)]
pub struct TextGenerationModelConfig {
    pub model_path: String,
    pub intra_threads: usize,
    pub optimization_level: GraphOptimizationLevel,
}

impl Default for TextGenerationModelConfig {
    fn default() -> Self {
        Self {
            model_path: DEFAULT_MODEL_PATH.to_string(),
            intra_threads: 4,
            optimization_level: GraphOptimizationLevel::Level3,
        }
    }
}

pub struct TextGenerationModel {
    session: Session,
}

impl TextGenerationModel {
    pub fn new(cfg: TextGenerationModelConfig) -> Result<Self> {
        let mut builder = Session::builder()?;

        // Register with the cuda provider if it exists
        let cuda = CUDAExecutionProvider::default();
        if cuda.is_available().unwrap_or(false) {
            if let Err(e) = cuda.register(&mut builder) {
                log::error!("Failed to register CUDA provider: {}", e);
            } else {
                log::debug!("CUDA provider registered");
            }
        }

        // Apply configuration options
        let builder = builder
            .with_optimization_level(cfg.optimization_level)?
            .with_intra_threads(cfg.intra_threads)?;

        // In dev mode we often use http models (they are cached)
        // Check for http path URL and load from there - otherwise local file system
        if cfg.model_path.starts_with("http") {
            return Ok(Self {
                session: builder.commit_from_url(cfg.model_path)?,
            });
        }

        Ok(Self {
            session: builder.commit_from_file(cfg.model_path)?,
        })
    }

    pub fn run<'s, 'i, 'v: 'i, const N: usize>(
        &'s mut self,
        input_values: impl Into<SessionInputs<'i, 'v, N>>,
    ) -> Result<SessionOutputs<'s>> {
        match self.session.run(input_values) {
            Ok(r) => Ok(r),
            Err(e) => Err(InferenceError::OnnxRuntime(e)),
        }
    }
}
