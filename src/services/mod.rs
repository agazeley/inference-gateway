use log::info;

use crate::repository::TransactionRepository;
use crate::repository::errors::RepositoryError;
use crate::repository::models::Transaction; // Add this import if Transaction is defined in models.rs

pub struct TransactionService {
    repo: Box<dyn TransactionRepository>,
}

impl TransactionService {
    pub fn new(mut repo: Box<dyn TransactionRepository>) -> Result<Self, RepositoryError> {
        repo.initialize()?;
        info!("Transaction service initialized");
        Ok(Self { repo })
    }
    pub fn get_transactions(&self) -> Result<Vec<Transaction>, RepositoryError> {
        self.repo.get_transactions()
    }

    pub fn create_transaction(&self, t: Transaction) -> Result<Transaction, RepositoryError>{
        self.repo.create_transaction(t)
    }
}
