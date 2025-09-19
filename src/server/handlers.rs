use std::sync::{Arc, MutexGuard};

use axum::{Extension, Json, http::StatusCode};
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::Mutex;

use crate::server::{Server, ServerStatus};

#[derive(Serialize, Debug)]
struct AppStatusResponse {
    name: String,
    status: ServerStatus,
}

impl AppStatusResponse {
    fn from_state(state: MutexGuard<'_, Server>) -> Self {
        Self {
            name: state.name.clone(),
            status: state.status,
        }
    }
}

// handlers
pub async fn healthz(Extension(state): Extension<Arc<Mutex<Server>>>) -> Json<Value> {
    let state_guard = state.lock().unwrap();
    Json(json!(AppStatusResponse::from_state(state_guard)))
}

pub async fn readyz(Extension(state): Extension<Arc<Mutex<Server>>>) -> (StatusCode, Json<Value>) {
    let state_guard = state.lock().unwrap();
    if state_guard.status != ServerStatus::Running {
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
