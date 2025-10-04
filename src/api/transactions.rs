use std::sync::Arc;

use axum::{Extension, Json, http::StatusCode};
use log::error;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{repository::models::Transaction, services::TransactionService};

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ListTransactionResponse {
    transactions: Vec<Transaction>,
    count: usize,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PostTransactionsRequest {
    transaction: Transaction
}

pub async fn post_transactions(
    Extension(svc): Extension<Arc<TransactionService>>,
    Json(req): Json<PostTransactionsRequest>,
) -> (StatusCode, Json<Value>){
    let t = match svc.create_transaction(req.transaction) {
        Ok(t) => t,
        Err(e) => {
            error!("Transaction creation error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "transaction creation error",
                })),
            );
        }
    };
    // I do not like this but the analyzer dislikes it more.
    (StatusCode::ACCEPTED, Json(json!(PostTransactionsRequest{transaction: t})))
}

pub async fn get_transactions(
    Extension(svc): Extension<Arc<TransactionService>>,
) -> (StatusCode, Json<Value>) {
    let transactions = match svc.get_transactions() {
        Ok(t) => t,
        Err(e) => {
            error!("Transaction list error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": "transaction list error",
                })),
            );
        }
    };
    let count = transactions.len();
    (
        StatusCode::OK,
        Json(json!(ListTransactionResponse {
            transactions,
            count
        })),
    )
}
