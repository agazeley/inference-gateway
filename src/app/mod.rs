pub mod handlers;

use axum::{
    Router,
    http::HeaderName,
    routing::{get, post},
};
use log::{error, info};
use serde::Serialize;
use std::sync::{Arc, Mutex};

use handlers::{healthz, readyz};
use tower::ServiceBuilder;
use tower_http::{propagate_header::PropagateHeaderLayer, trace::TraceLayer};

use crate::app::handlers::post_inference;

#[derive(Serialize, Copy, Clone, PartialEq, Debug)]
pub enum AppStatus {
    None,
    Starting,
    Stopped,
    Running,
    // Unhealthy, // TODO: find out what unhealthy state is and make a way to set it
}

pub struct App {
    pub name: String,
    pub status: AppStatus,
}

impl Default for App {
    fn default() -> Self {
        Self {
            name: String::from("inference-gateway"),
            status: AppStatus::None,
        }
    }
}

impl App {
    pub async fn serve(self, addr: String) {
        let shared_state = Arc::new(Mutex::new(self));

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

        let api_router = Router::new().route("/inference", post(post_inference));

        // build our application with a single route
        let app = Router::new()
            .nest("/api/v1", api_router)
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
        let listener = match tokio::net::TcpListener::bind(addr.clone()).await {
            Ok(listener) => listener,
            Err(e) => {
                error!("Failed to bind to {}: {}", addr, e);
                return;
            }
        };

        info!("Server starting on {}", addr);

        match axum::serve(listener, app).await {
            Ok(_) => info!("Server shutdown gracefully"),
            Err(e) => error!("Server error: {}", e),
        }

        // Set status to Stopped once we have stopped the server
        {
            let mut state_guard = shared_state.lock().unwrap();
            state_guard.status = AppStatus::Stopped;
        }
    }
}
