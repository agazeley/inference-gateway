pub mod handlers;

use axum::{Extension, Router, routing::get};
use log::{debug, error, info};
use serde::Serialize;
use std::sync::{Arc, Mutex};

use handlers::{healthz, readyz};
use tower::ServiceBuilder;

use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::{
    DefaultMakeSpan, DefaultOnFailure, DefaultOnRequest, DefaultOnResponse, TraceLayer,
};
use tracing::Level;

#[derive(Serialize, Copy, Clone, PartialEq, Debug)]
pub enum ServerStatus {
    None,
    Starting,
    Stopped,
    Running,
    // Unhealthy, // TODO: find out what unhealthy state is and make a way to set it
}

pub struct Server {
    pub name: String,
    pub status: ServerStatus,
    router: Router,
}

impl Default for Server {
    fn default() -> Self {
        // Create our service
        let service = ServiceBuilder::new()
            // High level logging of requests and responses
            .layer(TraceLayer::new_for_http())
            // Propagate `X-Request-Id`s from requests to responses
            ;
        let router = Router::new()
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .layer(service);
        Self {
            name: "inference-gateway".to_string(),
            status: ServerStatus::None,
            router,
        }
    }
}

impl Server {
    pub fn add_router(&mut self, path: &str, router: Router) {
        self.router = self.router.clone().nest(path, router);
    }

    pub async fn serve(mut self, addr: String) {
        // Set initial status to Starting
        self.status = ServerStatus::Starting;
        let router = self.router.clone();
        let shared_state = Arc::new(Mutex::new(self));

        // Create router with shared state
        let router = router
            .clone()
            .layer(Extension(shared_state.clone()))
            // Add request-id layers (optional but useful)
            .layer(PropagateRequestIdLayer::x_request_id())
            .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
            // Add request/response logging
            .layer(
                TraceLayer::new_for_http()
                    // log when the request is received
                    .on_request(DefaultOnRequest::new().level(Level::INFO))
                    // include method, uri, version, and request-id in the span
                    .make_span_with(
                        DefaultMakeSpan::new()
                            .level(Level::INFO)
                            .include_headers(false), // set true if you want headers in the span
                    )
                    // log when the response is sent (includes latency)
                    .on_response(
                        DefaultOnResponse::new()
                            .level(Level::INFO)
                            .include_headers(false),
                    )
                    // log failures (timeouts, errors)
                    .on_failure(DefaultOnFailure::new().level(Level::ERROR)),
            );

        debug!("Initialized state");

        // Set status to Running before starting the server
        {
            let mut state_guard = shared_state.lock().unwrap();
            state_guard.status = ServerStatus::Running;
        }

        // run our app with hyper
        let listener = match tokio::net::TcpListener::bind(addr.clone()).await {
            Ok(listener) => listener,
            Err(e) => {
                error!("Failed to bind to {}: {}", addr, e);
                return;
            }
        };

        info!("Server starting on {}", addr);

        match axum::serve(listener, router).await {
            Ok(_) => info!("Server shutdown gracefully"),
            Err(e) => error!("Server error: {}", e),
        }

        // Set status to Stopped once we have stopped the server
        {
            let mut state_guard = shared_state.lock().unwrap();
            state_guard.status = ServerStatus::Stopped;
        }
    }
}
