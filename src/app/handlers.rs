use std::sync::{Arc, MutexGuard};

use axum::{Json, extract::State, http::StatusCode};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Mutex;

use crate::app::{App, AppStatus};

#[derive(Serialize, Debug)]
struct AppStatusResponse {
    name: String,
    status: AppStatus,
}

impl AppStatusResponse {
    fn from_state(state: MutexGuard<'_, App>) -> Self {
        Self {
            name: state.name.clone(),
            status: state.status,
        }
    }
}

// handlers
pub async fn healthz(State(state): State<Arc<Mutex<App>>>) -> Json<Value> {
    let state_guard = state.lock().unwrap();
    Json(json!(AppStatusResponse::from_state(state_guard)))
}

pub async fn readyz(State(state): State<Arc<Mutex<App>>>) -> (StatusCode, Json<Value>) {
    let state_guard = state.lock().unwrap();
    if state_guard.status != AppStatus::Running {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!(AppStatusResponse::from_state(state_guard))),
        );
    }
    (
        StatusCode::OK,
        Json(json!(AppStatusResponse::from_state(state_guard))),
    )
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
