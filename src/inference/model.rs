use crate::inference::errors::{InferenceError, Result};
use crate::inference::inputs::DynamicInputBuilder;
use ort::{
    execution_providers::{CUDAExecutionProvider, ExecutionProvider},
    session::{Session, SessionInputs, SessionOutputs, builder::GraphOptimizationLevel},
    value::Value,
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

impl AutoRegressiveModelConfig {
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
}

impl AutoRegressiveModel {
    pub fn new(cfg: AutoRegressiveModelConfig) -> Result<Self> {
        let name = cfg.model_name.clone();
        let session = cfg.build_session()?;

        Ok(Self {
            session,
            name,
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
