pub mod handlers;

use crate::api::handlers::post_inference;
use crate::inference::llm::{LLM, load_default_model, load_default_tokenizer};
use axum::{Extension, Router, routing::post};
use std::{
    fmt::Error,
    sync::{Arc, Mutex},
};

/// Creates and returns an Axum `Router` configured with the necessary routes and layers.
///
/// This function initializes the default language model and tokenizer, wraps them in a
/// thread-safe `Arc<Mutex<LLM>>`, and sets up the `/inference` endpoint. The shared model
/// is made accessible to the endpoint via the `Extension` layer.
///
/// Returns axum Router or an Error.
pub fn get_router() -> Result<Router, Error> {
    let model = load_default_model().unwrap();
    let tokenizer = load_default_tokenizer().unwrap();
    let language_model = LLM::new(model, tokenizer);
    let shared_model = Arc::new(Mutex::new(language_model));

    // IMPORTANT: Add routes first, then apply layers so the layers wrap them.
    // Previously the Extension layer was added before the route, so the route
    // did not have access to the shared model, producing a missing Extension error.
    Ok(Router::new()
        .route("/inference", post(post_inference))
        .layer(Extension(shared_model)))
}
