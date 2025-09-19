use std::sync::{Arc, MutexGuard};

use axum::{http::StatusCode, Extension, Json};
use serde::{ Serialize};
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
pub async fn healthz(Extension(state): Extension<Arc<Mutex<App>>>) -> Json<Value> {
    let state_guard = state.lock().unwrap();
    Json(json!(AppStatusResponse::from_state(state_guard)))
}

pub async fn readyz(Extension(state): Extension<Arc<Mutex<App>>>) -> (StatusCode, Json<Value>) {
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