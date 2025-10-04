use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use axum::{Extension, Json, http::StatusCode};
use log::{debug, error};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    inference::llm::{ChatRequest, LLM, Message, TextGenerationParameters},
    repository::models::Transaction,
    services::TransactionService,
};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PostInferenceRequest {
    text: String,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

// Implement conversion from API type to internal config
impl From<PostInferenceRequest> for ChatRequest {
    fn from(req: PostInferenceRequest) -> Self {
        Self {
            messages: vec![Message {
                role: "user".to_string(),
                content: req.text,
            }],
        }
    }
}

// Implement conversion from API type to internal config
impl From<PostInferenceRequest> for TextGenerationParameters {
    fn from(req: PostInferenceRequest) -> Self {
        let mut params = Self::new();
        if req.max_tokens.is_some() {
            params.max_tokens = req.max_tokens
        }
        if req.temperature.is_some() {
            params.temperature = req.temperature
        }
        if req.top_p.is_some() {
            params.top_p = req.top_p
        }

        params
    }
}

#[derive(Serialize, Debug)]
pub struct PostInferenceResponse {
    text: String,
    model: String,
    metadata: HashMap<String, String>,
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
    Extension(svc): Extension<Arc<TransactionService>>,
    Json(req): Json<PostInferenceRequest>,
) -> (StatusCode, Json<Value>) {
    // Limit the scope of the lock so we don't hold it while serializing / responding
    let mut t = Transaction::new(req.text.clone());
    let (generated, model_name) = {
        let mut llm_guard = llm.lock().unwrap();
        let model_name = llm_guard.model_name();
        let params = TextGenerationParameters::from(req.clone());
        let chat_request = ChatRequest::from(req);
        match llm_guard.chat(chat_request, params) {
            Ok(t) => (t, model_name),
            Err(e) => {
                error!("Generate error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": "generate error",
                    })),
                );
            }
        }
    };
    debug!("Inference result: {}", generated);
    t.set_response(generated.clone());
    t.set_model(model_name.clone());
    if let Err(e) = svc.create_transaction(t) {
        error!("Failed to create transaction: {}", e);
    }

    (
        StatusCode::OK,
        Json(json!(PostInferenceResponse {
            text: generated,
            model: model_name,
            metadata: HashMap::new()
        })),
    )
}
