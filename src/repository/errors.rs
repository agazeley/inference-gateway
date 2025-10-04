use thiserror::Error as ThisError;

// Result type alias
pub type Result<T> = std::result::Result<T, RepositoryError>;

// Main error type for the inference lib.
#[derive(ThisError, Debug)]
pub enum RepositoryError {
    #[error("Initialization error: {0}")]
    InitializationError(String),

    #[error("SQL execution error: {0}")]
    SqlExecutionError(#[from] rusqlite::Error),

    #[error("Connection locked error: {0}")]
    ConnectionLockedError(String),
}
