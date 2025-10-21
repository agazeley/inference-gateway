use thiserror::Error as ThisError;

// Result type alias
pub type Result<T> = std::result::Result<T, InferenceError>;

// Main error type for the inference lib.
#[derive(ThisError, Debug)]
pub enum InferenceError {
    // ONNX Runtime errors
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(#[from] ort::Error),

    // Error while generating dynamic inputs
    #[error("Input generation error: {0}")]
    InputGenerationError(String),

    // Error while generating dynamic outputs
    #[error("Output generation error: {0}")]
    OutputGenerationError(String),

    // Error working with model metadata
    #[error("Model metadata error: {0}")]
    ModelMetadataError(String),

    // Error while generating text
    #[error("Text generation error: {0}")]
    TextGenerationError(String),

    // Tokenization errors
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Chat template error: {0}")]
    ChatTemplateError(String),

    // Model loading errors
    #[error("Model loading error: {0}")]
    ModelLoading(String),

    // Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    // Unsupported data type errors
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),

    #[error("Cache error: {0}")]
    CacheError(String),
}
