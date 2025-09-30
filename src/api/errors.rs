use thiserror::Error as ThisError;

use crate::inference::errors::InferenceError;

// Result type alias
pub type Result<T> = std::result::Result<T, ApiError>;

// Main error type for the inference lib.
#[derive(ThisError, Debug)]
pub enum ApiError {
    // Inference errors
    #[error("Inference initialization error: {0}")]
    InferenceInit(InferenceError),
}
