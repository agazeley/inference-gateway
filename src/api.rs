use axum::{http::StatusCode, routing::post, Json, Router};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};



pub fn get_router() -> Router{
    Router::new().route("/inference", post(post_inference))
}



#[derive(Deserialize, Serialize, Debug)]
pub struct PostInferenceRequest {
    text: String,
}

#[derive(Serialize, Debug)]
pub struct PostInferenceResponse {}

pub async fn post_inference(Json(req): Json<PostInferenceRequest>) -> (StatusCode, Json<Value>) {
    info!("Inference text: {}", req.text);
    (StatusCode::OK, Json(json!(PostInferenceResponse {})))
}
