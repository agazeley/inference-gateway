use std::sync::Mutex;

use crate::{
    repository::{
        MigrationMode, TransactionRepository,
        errors::{RepositoryError, Result},
        models::Transaction,
    },
    utils::get_env,
};
use log::{info, warn};
use rusqlite::{Connection, ToSql};

const DATABASE_MIGRATION_MODE_VAR: &str = "DATABASE_MIGRATION_MODE";
const DATABASE_PATH_VAR: &str = "DATABASE_PATH";

pub struct SQLiteTransactionRepository {
    conn: Mutex<Connection>,
    initialized: bool,
}

impl SQLiteTransactionRepository {
    pub fn new() -> Result<Self> {
        let path = get_env(DATABASE_PATH_VAR, "/runtime/database.db");
        let conn = Connection::open(&path)
            .map_err(|e| RepositoryError::InitializationError(e.to_string()))?;
        info!("SQLite database {} connection created", path);
        Ok(Self {
            conn: Mutex::new(conn),
            initialized: false,
        })
    }

    fn initialize_with_mode(&mut self, mode: MigrationMode) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        match mode {
            MigrationMode::CreateIfNotExists => {
                info!("Initializing database with CREATE IF NOT EXISTS mode");
                self.create_tables_if_not_exists()?;
            }
            MigrationMode::DropAndRecreate => {
                warn!(
                    "Initializing database with DROP AND RECREATE mode - THIS WILL DELETE ALL DATA"
                );
                self.drop_and_recreate_tables()?;
            }
        }

        self.initialized = true;
        info!("Database initialization complete");
        Ok(())
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

    fn create_tables_if_not_exists(&self) -> Result<()> {
        let sql = r#"
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        "#;

        self.execute(sql, &[])?;
        info!("Transaction table created (if not exists)");
        Ok(())
    }

    fn drop_and_recreate_tables(&self) -> Result<()> {
        // Drop existing tables
        let drop_sql = "DROP TABLE IF EXISTS transactions;";
        self.execute(drop_sql, &[])?;
        warn!("Dropped existing transaction table");

        // Recreate with updated schema
        let create_sql = r#"
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        "#;

        self.execute(create_sql, &[])?;
        info!("Transaction table recreated");
        Ok(())
    }
}

impl TransactionRepository for SQLiteTransactionRepository {
    fn initialize(&mut self) -> Result<()> {
        // Choose migration mode based on environment variable
        let mode = match get_env(DATABASE_MIGRATION_MODE_VAR, "").as_str() {
            "drop_recreate" => {
                log::warn!("Using DROP_RECREATE migration mode - THIS WILL DELETE ALL DATA");
                MigrationMode::DropAndRecreate
            }
            _ => {
                log::info!("Using CREATE_IF_NOT_EXISTS migration mode (safe)");
                MigrationMode::CreateIfNotExists
            }
        };
        self.initialize_with_mode(mode)
    }

    fn create_transaction(&self, t: Transaction) -> Result<Transaction> {
        let conn_guard = self.conn.lock().map_err(|_| {
            RepositoryError::ConnectionLockedError("failed to acquire lock".to_string())
        })?;
        let sql = "INSERT INTO transactions (prompt, response, model_name) VALUES (?1, ?2, ?3)";
        conn_guard
            .prepare(sql)
            .map_err(RepositoryError::SqlExecutionError)?
            .execute([&t.prompt, &t.response, &t.model_name])
            .map_err(RepositoryError::SqlExecutionError)?;

        let id = conn_guard.last_insert_rowid();
        Ok(Transaction {
            id: Some(id),
            prompt: t.prompt,
            response: t.response,
            model_name: t.model_name,
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
                    model_name: row.get(3)?,
                })
            })
            .map_err(RepositoryError::SqlExecutionError)?
            .filter_map(|result| result.ok())
            .collect();
        Ok(transactions)
    }
}
