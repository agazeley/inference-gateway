use thiserror::Error as ThisError;

use crate::inference::errors::InferenceError;
use crate::repository::errors::RepositoryError;

// Result type alias
pub type Result<T> = std::result::Result<T, ApiError>;

// Simple API error type
#[derive(ThisError, Debug)]
pub enum ApiError {
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    #[error("Repository error: {0}")]
    Repository(#[from] RepositoryError),
}
