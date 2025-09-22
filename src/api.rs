use std::sync::{Arc, Mutex};

use axum::{Extension, Json, Router, http::StatusCode, routing::post};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::inference::errors::Result;
use crate::inference::llm::{
    LLM, TextGenerationConfig, load_default_model, load_default_tokenizer,
};

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
pub struct PostInferenceRequest {
    text: String,
}

#[derive(Serialize, Debug)]
pub struct PostInferenceResponse {
    text: String,
}

pub async fn post_inference(
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
