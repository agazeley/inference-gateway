mod errors;
mod inference;
mod transactions;

use crate::inference::llm::LLM;
use crate::inference::{load_default_model, load_default_tokenizer};
use crate::repository::sqlite::SQLiteTransactionRepository;
use crate::router::errors::ApiError;
use crate::router::inference::post_inference;
use crate::router::transactions::{get_transactions, post_transactions};
use crate::services::TransactionService;

use axum::routing::get;
use axum::{Extension, Router, routing::post};
use std::sync::{Arc, Mutex};

/// Creates and returns an Axum `Router` configured with the necessary routes and layers.
///
/// This function initializes the default language model and tokenizer, wraps them in a
/// thread-safe `Arc<Mutex<LLM>>`, and sets up the `/inference` endpoint. The shared model
/// is made accessible to the endpoint via the `Extension` layer.
///
/// Returns axum Router or an Error.
pub async fn get_router() -> errors::Result<Router> {
    let model = load_default_model().map_err(ApiError::Inference)?;
    let tokenizer = load_default_tokenizer().map_err(ApiError::Inference)?;
    let language_model = LLM::new(model, tokenizer);
    let shared_model = Arc::new(Mutex::new(language_model));

    let repository = SQLiteTransactionRepository::new()
        .await
        .map_err(ApiError::Repository)?;
    let svc = TransactionService::new(repository)
        .await
        .map_err(ApiError::Repository)?;
    let shared_svc = Arc::new(svc);

    // IMPORTANT: Add routes first, then apply layers so the layers wrap them.
    // Previously the Extension layer was added before the route, so the route
    // did not have access to the shared model, producing a missing Extension error.
    Ok(Router::new()
        .route("/inference", post(post_inference))
        .route("/transactions", get(get_transactions))
        .route("/transactions", post(post_transactions))
        .layer(Extension(shared_model))
        .layer(Extension(shared_svc)))
}
