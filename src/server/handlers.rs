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

/// Health check handler that returns the current status of the application.
///
/// This endpoint is typically used to verify that the application is running
/// and able to respond to requests. It returns a JSON response containing
/// the application's name and status.
///
/// # Arguments
///
/// * `Extension(state)` - Shared application state wrapped in an `Arc<Mutex<Server>>`.
///
/// # Returns
///
/// A JSON response with the application's status.
pub async fn healthz(Extension(state): Extension<Arc<Mutex<Server>>>) -> Json<Value> {
    let state_guard = state.lock().unwrap();
    Json(json!(AppStatusResponse::from_state(state_guard)))
}

/// Readiness check handler that verifies if the application is ready to handle requests.
///
/// This endpoint is used to determine if the application is in a "ready" state
/// to process incoming requests. If the server status is not `Running`, it
/// returns a 500 Internal Server Error status code along with the application's
/// current status. Otherwise, it returns a 200 OK status code with the status.
///
/// # Arguments
///
/// * `Extension(state)` - Shared application state wrapped in an `Arc<Mutex<Server>>`.
///
/// # Returns
///
/// A tuple containing:
/// - HTTP status code (`StatusCode`)
/// - A JSON response with the application's status.
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
