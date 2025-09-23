use thiserror::Error as ThisError;

/// Result type alias
pub type Result<T> = std::result::Result<T, InferenceError>;

/// Main error type for the inference lib.
#[derive(ThisError, Debug)]
pub enum InferenceError {
    /// ONNX Runtime errors
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(#[from] ort::Error),

    // Generation errors
    #[error("Text generation error: {0}")]
    TextGenerationError(String),

    /// Tokenization errors
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Model loading errors
    #[error("Model loading error: {0}")]
    ModelLoading(String),

    /// Lock acquisition errors
    #[error("Lock acquisition error: {0}")]
    LockAcquisition(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),
}
