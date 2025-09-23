use std::sync::{Arc, Mutex};

use axum::{Extension, Json, http::StatusCode};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::inference::llm::{LLM, TextGenerationConfig};

#[derive(Deserialize, Serialize, Debug)]
pub struct PostInferenceRequest {
    text: String,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

// Implement conversion from API type to internal config
impl From<PostInferenceRequest> for TextGenerationConfig {
    fn from(req: PostInferenceRequest) -> Self {
        let mut cfg = Self::new(req.text);
        if req.max_tokens.is_some() {
            cfg.max_tokens = req.max_tokens.unwrap()
        }
        if req.temperature.is_some() {
            cfg.temperature = req.temperature.unwrap()
        }
        if req.top_p.is_some() {
            cfg.top_p = req.top_p
        }

        cfg
    }
}

#[derive(Serialize, Debug)]
pub struct PostInferenceResponse {
    text: String,
    model: String,
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
///   "text": "Hello, world! How can I assist you today?",
///   "model": "gpt-3.5-turbo"
/// }
/// ```
pub async fn post_inference(
    Extension(llm): Extension<Arc<Mutex<LLM>>>,
    Json(req): Json<PostInferenceRequest>,
) -> (StatusCode, Json<Value>) {
    // Limit the scope of the lock so we don't hold it while serializing / responding
    let (generated, model_name) = {
        let mut llm_guard = llm.lock().unwrap();
        let model_name = llm_guard.model_name();
        match llm_guard.generate(TextGenerationConfig::from(req)) {
            Ok(t) => (t, model_name),
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
        Json(json!(PostInferenceResponse {
            text: generated,
            model: model_name
        })),
    )
}
