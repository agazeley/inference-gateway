use log::info;

use crate::repository::TransactionRepository;
use crate::repository::errors::RepositoryError;
use crate::repository::models::Transaction;

pub struct TransactionService<T: TransactionRepository> {
    repo: T,
}

impl<T: TransactionRepository> TransactionService<T> {
    pub async fn new(mut repo: T) -> Result<Self, RepositoryError> {
        repo.initialize().await?;
        info!("Transaction service initialized");
        Ok(Self { repo })
    }

    pub async fn get_transactions(&self) -> Result<Vec<Transaction>, RepositoryError> {
        self.repo.get_transactions().await
    }

    pub async fn create_transaction(&self, t: Transaction) -> Result<Transaction, RepositoryError> {
        self.repo.create_transaction(t).await
    }
}
