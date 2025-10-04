use std::{fs, path::Path};

use crate::{
    repository::{
        MigrationMode, TransactionRepository,
        errors::{RepositoryError, Result},
        models::Transaction,
    },
    utils::get_env,
};
use log::{info, warn};
use sqlx::{AnyPool, Row};
use std::env;

const DATABASE_MIGRATION_MODE_VAR: &str = "DATABASE_MIGRATION_MODE";
const DATABASE_CONNECTION_STRING_VAR: &str = "DATABASE_CONNECTION_STRING";
const SQLITE_FILE_PREFIX: &str = "sqlite:file:";

#[derive(Clone)]
pub struct SQLTransactionRepository {
    pool: AnyPool,
}

impl SQLTransactionRepository {
    pub async fn new() -> Result<Self> {
        let default_path = env::current_dir()
            .expect("Failed to get current directory")
            .join("data/database.db")
            .to_str()
            .expect("Failed to convert path to string")
            .to_string();
        let connection_string: String = get_env(
            DATABASE_CONNECTION_STRING_VAR,
            &format!("sqlite:file:{}?mode=rwc", default_path),
        );

        if connection_string.contains(SQLITE_FILE_PREFIX) {
            // Ensure the directory for the DB exists
            if let Some(path) = connection_string
                .strip_prefix(SQLITE_FILE_PREFIX)
                .and_then(|s| s.split('?').next())
                && let Some(dir) = Path::new(path).parent()
            {
                fs::create_dir_all(dir).ok();
                info!("Ensured directory exists: {:?}", dir);
            }
        }

        let pool = AnyPool::connect(&connection_string).await.map_err(|e| {
            RepositoryError::InitializationError(format!(
                "unable to open with connection string {}: {}",
                connection_string, e
            ))
        })?;

        info!("Database {} connection pool created", connection_string);

        let mut repo = Self { pool };

        // Auto-initialize with default mode
        repo.initialize().await?;

        Ok(repo)
    }

    async fn initialize_with_mode(&mut self, mode: MigrationMode) -> Result<()> {
        match mode {
            MigrationMode::CreateIfNotExists => {
                info!("Initializing database with CREATE IF NOT EXISTS mode");
                self.create_tables_if_not_exists().await?;
            }
            MigrationMode::DropAndRecreate => {
                warn!(
                    "Initializing database with DROP AND RECREATE mode - THIS WILL DELETE ALL DATA"
                );
                self.drop_and_recreate_tables().await?;
            }
        }
        info!("Database initialization complete");
        Ok(())
    }

    async fn create_tables_if_not_exists(&self) -> Result<()> {
        let sql = r#"
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        "#;

        sqlx::query(sql)
            .execute(&self.pool)
            .await
            .map_err(RepositoryError::SqlExecutionError)?;

        info!("Transaction table created (if not exists)");
        Ok(())
    }

    async fn drop_and_recreate_tables(&self) -> Result<()> {
        // Drop existing tables
        let drop_sql = "DROP TABLE IF EXISTS transactions";
        sqlx::query(drop_sql)
            .execute(&self.pool)
            .await
            .map_err(RepositoryError::SqlExecutionError)?;
        warn!("Dropped existing transaction table");

        // Recreate with updated schema
        let create_sql = r#"
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        "#;

        sqlx::query(create_sql)
            .execute(&self.pool)
            .await
            .map_err(RepositoryError::SqlExecutionError)?;
        info!("Transaction table recreated");
        Ok(())
    }
}

impl TransactionRepository for SQLTransactionRepository {
    async fn initialize(&mut self) -> Result<()> {
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
        self.initialize_with_mode(mode).await
    }

    async fn create_transaction(&self, t: Transaction) -> Result<Transaction> {
        let sql = "INSERT INTO transactions (prompt, response, model_name) VALUES (?, ?, ?)";
        let result = sqlx::query(sql)
            .bind(&t.prompt)
            .bind(&t.response)
            .bind(&t.model_name)
            .execute(&self.pool)
            .await
            .map_err(RepositoryError::SqlExecutionError)?;
        let id = result.last_insert_id();
        Ok(Transaction {
            id,
            prompt: t.prompt,
            response: t.response,
            model_name: t.model_name,
        })
    }

    async fn get_transactions(&self) -> Result<Vec<Transaction>> {
        let rows = sqlx::query("SELECT id, prompt, response, model_name FROM transactions")
            .fetch_all(&self.pool)
            .await
            .map_err(RepositoryError::SqlExecutionError)?;

        let transactions = rows
            .into_iter()
            .map(|row| Transaction {
                id: row.get("id"),
                prompt: row.get("prompt"),
                response: row.get("response"),
                model_name: row.get("model_name"),
            })
            .collect();
        Ok(transactions)
    }
}
