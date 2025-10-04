pub mod errors;
pub mod models;
pub mod sqlite;

use crate::repository::{errors::Result, models::Transaction};

use std::future::Future;

#[derive(Debug, Clone, Copy)]
pub enum MigrationMode {
    /// Create tables if they don't exist (safe, preserves data)
    CreateIfNotExists,
    /// Drop and recreate all tables (destructive, loses data)
    DropAndRecreate,
}

pub trait TransactionRepository: Send + Sync {
    fn initialize(&mut self) -> impl Future<Output = Result<()>> + Send;
    fn create_transaction(
        &self,
        t: Transaction,
    ) -> impl Future<Output = Result<Transaction>> + Send;
    fn get_transactions(&self) -> impl Future<Output = Result<Vec<Transaction>>> + Send;
}
