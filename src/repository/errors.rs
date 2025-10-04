use thiserror::Error;

#[derive(Error, Debug)]
pub enum RepositoryError {
    #[error("Initialization error: {0}")]
    InitializationError(String),

    #[error("Connection locked error: {0}")]
    ConnectionLockedError(String),

    #[error("SQL execution error: {0}")]
    SqlExecutionError(#[from] sqlx::Error),

    #[error("Transaction creation error: {0}")]
    TransactionCreationError(String),

    #[error("Transaction retrieval error: {0}")]
    TransactionRetrievalError(String),

    #[error("Migration error: {0}")]
    MigrationError(String),
}

pub type Result<T> = std::result::Result<T, RepositoryError>;
