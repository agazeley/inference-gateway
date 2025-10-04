use crate::{
    repository::{
        MigrationMode, TransactionRepository,
        errors::{RepositoryError, Result},
        models::Transaction,
    },
    utils::get_env,
};
use log::{info, warn};
use sqlx::{Row, SqlitePool};

const DATABASE_MIGRATION_MODE_VAR: &str = "DATABASE_MIGRATION_MODE";
const DATABASE_PATH_VAR: &str = "DATABASE_PATH";

pub struct SQLiteTransactionRepository {
    pool: SqlitePool,
    initialized: bool,
}

impl SQLiteTransactionRepository {
    pub async fn new() -> Result<Self> {
        let path = get_env(DATABASE_PATH_VAR, "./data/database.db");

        // Format the connection string properly for SQLite
        let connection_string = format!("sqlite:{}?mode=rwc", path);

        let pool = SqlitePool::connect(&connection_string)
            .await
            .map_err(|e| RepositoryError::InitializationError(e.to_string()))?;

        info!("SQLite database {} connection pool created", path);

        let mut repo = Self {
            pool,
            initialized: false,
        };

        // Auto-initialize with default mode
        repo.initialize().await?;

        Ok(repo)
    }

    async fn initialize_with_mode(&mut self, mode: MigrationMode) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

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

        self.initialized = true;
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

impl TransactionRepository for SQLiteTransactionRepository {
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

        let id = result.last_insert_rowid();
        Ok(Transaction {
            id: Some(id),
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
