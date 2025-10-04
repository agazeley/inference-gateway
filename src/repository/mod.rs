use std::sync::Mutex;

use crate::repository::{
    errors::{RepositoryError, Result},
    models::Transaction,
};
use log::info;
use rusqlite::{Connection, ToSql};

pub mod errors;
pub mod models;

pub trait TransactionRepository: Send + Sync {
    fn initialize(&mut self) -> Result<()>; // create database tables if do not exist; probably remove this later?

    fn create_transaction(&self, t: Transaction) -> Result<Transaction>; 

    fn get_transactions(&self) -> Result<Vec<Transaction>>;
}

pub struct SqlliteTransactionRepository {
    conn: Mutex<Connection>,
    initialized: bool,
}

impl SqlliteTransactionRepository {
    pub fn new() -> Result<Self> {
        let path = "./database.db";
        let conn = Connection::open(path)
            .map_err(|e| RepositoryError::InitializationError(e.to_string()))?;
        info!("Sqllite database {} connection created", path);
        Ok(Self {
            conn: Mutex::new(conn),
            initialized: false
        })
    }

    fn execute(&self, sql: &str, params: &[&dyn ToSql]) -> Result<usize> {
        let conn_guard = self.conn.lock().map_err(|_| {
            RepositoryError::ConnectionLockedError("failed to acquire lock".to_string())
        })?;
        let mut stmt = conn_guard
            .prepare(sql)
            .map_err(RepositoryError::SqlExecutionError)?;
        let rows = stmt
            .execute(params)
            .map_err(RepositoryError::SqlExecutionError)?;
        Ok(rows)
    }
}

impl TransactionRepository for SqlliteTransactionRepository {
    fn initialize(&mut self) -> Result<()> {
        if self.initialized{
            return Ok(())
        }
        let _ = self.execute(
            "CREATE TABLE IF NOT EXISTS transactions (
                id BIGINT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL
            );",
            &[], // empty list of parameters.
        )?;
        self.initialized = true;
        info!("Transaction table created");
        Ok(())
    }

    fn create_transaction(&self, t: Transaction) -> Result<Transaction> {
        let conn_guard = self.conn.lock().map_err(|_| {
            RepositoryError::ConnectionLockedError("failed to acquire lock".to_string())
        })?;
        let sql = "INSERT INTO transactions (prompt, response) VALUES (?1, ?2)";
        conn_guard
            .prepare(sql)
            .map_err(RepositoryError::SqlExecutionError)?
            .execute(&[&t.prompt, &t.response])
            .map_err(RepositoryError::SqlExecutionError)?;

        let id = conn_guard.last_insert_rowid();
        Ok(Transaction {
            id,
            prompt: t.prompt,
            response: t.response,
        })
    }

    fn get_transactions(&self) -> Result<Vec<Transaction>> {
        let conn_guard = self.conn.lock().map_err(|_| {
            RepositoryError::ConnectionLockedError("failed to acquire lock".to_string())
        })?;
        let mut stmt = conn_guard
            .prepare("SELECT * FROM transactions")
            .map_err(RepositoryError::SqlExecutionError)?;

        let transactions: Vec<Transaction> = stmt
            .query_map([], |row| {
                Ok(Transaction {
                    id: row.get(0)?,
                    prompt: row.get(1)?,
                    response: row.get(2)?,
                })
            })
            .map_err(RepositoryError::SqlExecutionError)?
            .filter_map(|result| result.ok())
            .collect();
        Ok(transactions)
    }
}
