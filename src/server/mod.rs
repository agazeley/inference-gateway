pub mod handlers;

use axum::{Extension, Router, routing::get};
use axum_prometheus::{Handle, MakeDefaultHandle, PrometheusMetricLayerBuilder};
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
enum ServerStatus {
    None,
    Starting,
    Stopped,
    Running,
    // Unhealthy, // TODO: find out what unhealthy state is and make a way to set it
}

pub struct Server {
    name: String,
    status: ServerStatus,
    router: Router,
}

impl Default for Server {
    fn default() -> Self {
        let name = "inference-gateway".to_string();
        // create metric handler and add prom layer to router
        let prometheus_layer = PrometheusMetricLayerBuilder::new()
            .with_prefix(name.clone().replace("-", "_"))
            .build();
        let metric_handle = Handle::make_default_handle(Handle::default());

        // Create our service
        let service = ServiceBuilder::new()
            // High level logging of requests and responses
            .layer(TraceLayer::new_for_http());

        let router = Router::new()
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .route("/metrics", get(|| async move { metric_handle.render() }))
            .layer(service)
            .layer(prometheus_layer);
        Self {
            name,
            status: ServerStatus::None,
            router,
        }
    }
}

impl Server {
    /// Adds a new router to the server at the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - A string slice that specifies the path where the router will be nested.
    /// * `router` - The `Router` instance to be added to the server.
    ///
    pub fn add_router(&mut self, path: &str, router: Router) {
        self.router = self.router.clone().nest(path, router);
    }

    /// Starts the server and begins serving requests on the specified address.
    ///
    /// # Arguments
    ///
    /// * `addr` - A `String` specifying the address (e.g., "127.0.0.1:8080") where the server will listen for incoming requests.
    ///
    /// # Behavior
    ///
    /// - Sets the server status to `Starting` before initializing.
    /// - Configures middleware layers for request/response logging, request ID propagation, and error handling.
    /// - Sets the server status to `Running` once the server is ready to accept requests.
    /// - Updates the server status to `Stopped` after the server shuts down.
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

        // Run our app with hyper
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
