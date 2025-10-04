pub mod errors;
pub mod models;
pub mod sqlite;

use crate::repository::{errors::Result, models::Transaction};

#[derive(Debug, Clone, Copy)]
pub enum MigrationMode {
    /// Create tables if they don't exist (safe, preserves data)
    CreateIfNotExists,
    /// Drop and recreate all tables (destructive, loses data)
    DropAndRecreate,
}

pub trait TransactionRepository: Send + Sync {
    fn initialize(&mut self) -> Result<()>;
    fn create_transaction(&self, t: Transaction) -> Result<Transaction>;
    fn get_transactions(&self) -> Result<Vec<Transaction>>;
}
