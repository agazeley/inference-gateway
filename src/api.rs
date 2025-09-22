use std::sync::{Arc, Mutex};

use axum::{Extension, Json, Router, http::StatusCode, routing::post};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::inference::errors::Result;
use crate::inference::llm::{
    LLM, TextGenerationConfig, load_default_model, load_default_tokenizer,
};

/// Creates and returns an Axum `Router` configured with the necessary routes and layers.
///
/// This function initializes the default language model and tokenizer, wraps them in a
/// thread-safe `Arc<Mutex<LLM>>`, and sets up the `/inference` endpoint. The shared model
/// is made accessible to the endpoint via the `Extension` layer.
///
/// Returns axum Router or an Error.
pub fn get_router() -> Result<Router> {
    let model = load_default_model()?;
    let tokenizer = load_default_tokenizer()?;
    let language_model = LLM::new(model, tokenizer, 128, 5);
    let shared_model = Arc::new(Mutex::new(language_model));

    // IMPORTANT: Add routes first, then apply layers so the layers wrap them.
    // Previously the Extension layer was added before the route, so the route
    // did not have access to the shared model, producing a missing Extension error.
    Ok(Router::new()
        .route("/inference", post(post_inference))
        .layer(Extension(shared_model)))
}

#[derive(Deserialize, Serialize, Debug)]
struct PostInferenceRequest {
    text: String,
}

#[derive(Serialize, Debug)]
struct PostInferenceResponse {
    text: String,
}

/// Handles POST requests to the `/inference` endpoint.
///
/// This asynchronous function takes a JSON payload containing a `text` field,
/// processes it using a shared language model, and returns a generated response
/// as JSON. The function uses a mutex to ensure thread-safe access to the shared
/// language model.
///
/// # Arguments
///
/// * `Extension(llm)` - An `Extension` layer providing access to the shared
///   `Arc<Mutex<LLM>>` instance, which represents the language model.
/// * `Json(req)` - A `Json`-deserialized `PostInferenceRequest` containing the
///   input text for inference.
///
/// # Returns
///
/// A tuple containing:
/// * `StatusCode` - The HTTP status code indicating the result of the operation.
/// * `Json<Value>` - A JSON response containing either the generated text or
///   an error message.
///
/// # Errors
///
/// Returns a `500 Internal Server Error` status code if the language model
/// fails to generate a response.
///
/// # Example
///
/// ```json
/// // Request
/// {
///   "text": "Hello, world!"
/// }
///
/// // Response
/// {
///   "text": "Hello, world! How can I assist you today?"
/// }
/// ```
async fn post_inference(
    Extension(llm): Extension<Arc<Mutex<LLM>>>,
    Json(req): Json<PostInferenceRequest>,
) -> (StatusCode, Json<Value>) {
    debug!("Inference text: {}", req.text);
    // Limit the scope of the lock so we don't hold it while serializing / responding
    let generated = {
        let mut llm_guard = llm.lock().unwrap();
        match llm_guard.generate(TextGenerationConfig::new(req.text)) {
            Ok(t) => t,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": format!("generation failed: {e}"),
                    })),
                );
            }
        }
    };
    debug!("Inference result: {}", generated);
    (
        StatusCode::OK,
        Json(json!(PostInferenceResponse { text: generated })),
    )
}
