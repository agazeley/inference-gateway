use std::sync::Arc;

use log::{error, info, warn};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::task::JoinHandle;

use crate::repository::TransactionRepository;
use crate::repository::errors::RepositoryError;
use crate::repository::models::Transaction;

pub struct TransactionService<T: TransactionRepository + Send + Sync + 'static> {
    repo: Arc<T>,
    tx: UnboundedSender<Transaction>,
    handle: Option<JoinHandle<()>>,
}

impl<T: TransactionRepository + Send + Sync + 'static> TransactionService<T> {
    pub fn start(repo: T) -> Result<Self, RepositoryError> {
        let repo = Arc::new(repo);
        let (tx, mut rx) = mpsc::unbounded_channel::<Transaction>();
        let repo_bg = Arc::clone(&repo);
        let handle = tokio::spawn(async move {
            info!("TransactionService background task started");
            while let Some(transaction) = rx.recv().await {
                match repo_bg.create_transaction(transaction.clone()).await {
                    Ok(_) => {
                        // Optionally log successful writes at debug level
                        log::debug!("Transaction written to database: {:?}", transaction.id);
                    }
                    Err(e) => {
                        // TODO: implement production features
                        // 1. Retry with exponential backoff
                        // 2. Write to a dead letter queue
                        // 3. Send to a monitoring system
                        error!(
                            "Failed to write transaction to database: {} - Transaction: {:?}",
                            e, transaction
                        );
                    }
                }
            }

            rx.close();
            warn!("TransactionService background task stopped");
        });
        info!("Transaction service initialized");
        Ok(Self {
            repo,
            tx,
            handle: Some(handle),
        })
    }

    pub async fn get_transactions(&self) -> Result<Vec<Transaction>, RepositoryError> {
        self.repo.get_transactions().await
    }

    pub async fn create_transaction(&self, t: Transaction) -> Result<Transaction, RepositoryError> {
        self.repo.create_transaction(t).await
    }

    /// Asynchronously queues a transaction for writing to the database.
    /// This method is non-blocking and will not fail if the channel is full.
    pub fn write_transaction(&self, transaction: Transaction) -> Result<(), RepositoryError> {
        self.tx.send(transaction).map_err(|_| {
            RepositoryError::InitializationError(
                "Failed to queue transaction for writing".to_string(),
            )
        })
    }

    /// Gracefully shutdown the service, waiting for background tasks to complete
    pub async fn shutdown(mut self) -> Result<(), RepositoryError> {
        // Drop the sender to close the channel and signal background task to finish
        drop(std::mem::replace(&mut self.tx, {
            // Create a dummy sender that's immediately dropped
            let (dummy_tx, _) = tokio::sync::mpsc::unbounded_channel();
            dummy_tx
        }));

        // Wait for background task to complete processing
        if let Some(handle) = self.handle.take() {
            match handle.await {
                Ok(_) => {
                    info!("TransactionService shutdown gracefully");
                    Ok(())
                }
                Err(e) => {
                    error!("Error during TransactionService shutdown: {}", e);
                    Err(RepositoryError::InitializationError(format!(
                        "Shutdown error: {}",
                        e
                    )))
                }
            }
        } else {
            warn!("TransactionService background task already taken or shutdown");
            Ok(())
        }
    }
}

impl<T: TransactionRepository + Send + Sync + 'static> Drop for TransactionService<T> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
            warn!("TransactionService dropped without graceful shutdown");
        }
    }
}
#[cfg(test)]
#[allow(clippy::manual_async_fn)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone)]
    struct MockTransactionRepository {
        transactions: Arc<Mutex<Vec<Transaction>>>,
        fail_on_create: bool,
    }

    #[async_trait]
    impl TransactionRepository for MockTransactionRepository {
        fn get_transactions(
            &self,
        ) -> impl Future<Output = Result<Vec<Transaction>, RepositoryError>> + Send {
            let transactions = self.transactions.lock().unwrap().clone();
            async move { Ok(transactions) }
        }

        fn create_transaction(
            &self,
            t: Transaction,
        ) -> impl Future<Output = Result<Transaction, RepositoryError>> + Send {
            let fail_on_create = self.fail_on_create;
            let transactions = Arc::clone(&self.transactions);
            async move {
                if fail_on_create {
                    Err(RepositoryError::InitializationError("fail".into()))
                } else {
                    transactions.lock().unwrap().push(t.clone());
                    Ok(t)
                }
            }
        }

        fn initialize(&mut self) -> impl Future<Output = Result<(), RepositoryError>> + Send {
            async { Ok(()) }
        }
    }

    fn sample_transaction(id: i32) -> Transaction {
        Transaction {
            id: Some(id as i64),
            prompt: Some("hello world".to_string()),
            response: None,
            model_name: Some("LLM-1".to_string()),
        }
    }

    #[tokio::test]
    async fn test_get_transactions_returns_all() {
        let repo = MockTransactionRepository {
            transactions: Arc::new(Mutex::new(vec![
                sample_transaction(1),
                sample_transaction(2),
            ])),
            fail_on_create: false,
        };
        let service = TransactionService::start(repo).unwrap();
        let txs = service.get_transactions().await.unwrap();
        assert_eq!(txs.len(), 2);
        assert_eq!(txs[0].id, Some(1));
        assert_eq!(txs[1].id, Some(2));
    }

    #[tokio::test]
    async fn test_create_transaction_success() {
        let repo = MockTransactionRepository {
            transactions: Arc::new(Mutex::new(vec![])),
            fail_on_create: false,
        };
        let service = TransactionService::start(repo).unwrap();
        let t = sample_transaction(42);
        let result = service.create_transaction(t.clone()).await.unwrap();
        assert_eq!(result.id, Some(42));
        let txs = service.get_transactions().await.unwrap();
        assert_eq!(txs.len(), 1);
        assert_eq!(txs[0].id, Some(42));
    }

    #[tokio::test]
    async fn test_create_transaction_failure() {
        let repo = MockTransactionRepository {
            transactions: Arc::new(Mutex::new(vec![])),
            fail_on_create: true,
        };
        let service = TransactionService::start(repo).unwrap();
        let t = sample_transaction(99);
        let result = service.create_transaction(t).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_transaction_queues_and_persists() {
        let repo = MockTransactionRepository {
            transactions: Arc::new(Mutex::new(vec![])),
            fail_on_create: false,
        };
        let service = TransactionService::start(repo.clone()).unwrap();
        let t = sample_transaction(123);
        service.write_transaction(t.clone()).unwrap();

        // Wait for background task to process
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let txs = repo.transactions.lock().unwrap();
        assert!(txs.iter().any(|x| x.id == Some(123)));
    }

    #[tokio::test]
    async fn test_write_transaction_channel_closed_returns_error() {
        let repo = MockTransactionRepository {
            transactions: Arc::new(Mutex::new(vec![])),
            fail_on_create: false,
        };
        let mut service = TransactionService::start(repo).unwrap();
        // Drop sender to close channel
        drop(std::mem::replace(&mut service.tx, {
            let (dummy_tx, _) = tokio::sync::mpsc::unbounded_channel();
            dummy_tx
        }));
        let t = sample_transaction(1);
        let result = service.write_transaction(t);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_shutdown_waits_for_background_task() {
        let repo = MockTransactionRepository {
            transactions: Arc::new(Mutex::new(vec![])),
            fail_on_create: false,
        };
        let service = TransactionService::start(repo.clone()).unwrap();
        let t = sample_transaction(555);
        service.write_transaction(t.clone()).unwrap();
        let _ = service.shutdown().await;
        let txs = repo.transactions.lock().unwrap();
        assert!(txs.iter().any(|x| x.id == Some(555)));
    }

    #[tokio::test]
    async fn test_drop_aborts_background_task() {
        let repo = MockTransactionRepository {
            transactions: Arc::new(Mutex::new(vec![])),
            fail_on_create: false,
        };
        let service = TransactionService::start(repo).unwrap();
        drop(service); // Should abort background task without panic
    }
}
