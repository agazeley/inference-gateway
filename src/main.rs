use axum::{
    Json, Router,
    extract::State,
    http::{HeaderName, StatusCode},
    routing::get,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tower::ServiceBuilder;
use tower_http::{propagate_header::PropagateHeaderLayer, trace::TraceLayer};

#[derive(Serialize, Copy, Clone, PartialEq)]
enum AppStatus {
    None,
    Starting,
    Stopped,
    Running,
    // Unhealthy, // TODO: find out what unhealthy state is and make a way to set it
}

// State
struct AppState {
    name: String,
    status: AppStatus,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            name: String::from("Inference gateway!"),
            status: AppStatus::None,
        }
    }
}

// handlers
async fn healthz(State(state): State<Arc<Mutex<AppState>>>) -> Json<Value> {
    let state_guard = state.lock().unwrap();
    Json(json!({ "status": state_guard.status, "name": state_guard.name }))
}

async fn readyz(State(state): State<Arc<Mutex<AppState>>>) -> (StatusCode, Json<Value>) {
    let state_guard = state.lock().unwrap();
    let status = json!({
        "status": state_guard.status,
        "name": state_guard.name,
    });
    if state_guard.status != AppStatus::Running {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(status));
    }
    (StatusCode::OK, Json(status))
}

#[tokio::main]
async fn main() {
    env_logger::Builder::new()
        .parse_env(env_logger::Env::new().filter_or("", "info"))
        .init();

    let state = AppState::default();
    let shared_state = Arc::new(Mutex::new(state));

    // Set initial status to Starting
    {
        let mut state_guard = shared_state.lock().unwrap();
        state_guard.status = AppStatus::Starting;
    }

    // Create our service
    let service = ServiceBuilder::new()
        // High level logging of requests and responses
        .layer(TraceLayer::new_for_http())
        // Propagate `X-Request-Id`s from requests to responses
        .layer(PropagateHeaderLayer::new(HeaderName::from_static(
            "x-request-id",
        )));

    // build our application with a single route
    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .layer(service)
        .with_state(shared_state.clone());

    // Set status to Running before starting the server
    {
        let mut state_guard = shared_state.lock().unwrap();
        state_guard.status = AppStatus::Running;
    }

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

    // Set status to Stopped once we have stopped the server
    {
        let mut state_guard = shared_state.lock().unwrap();
        state_guard.status = AppStatus::Stopped
    }
}
