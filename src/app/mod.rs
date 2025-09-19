pub mod handlers;

use axum::{
    http::HeaderName, routing::get, Extension, Router
};
use log::{debug, error, info};
use serde::Serialize;
use std::sync::{Arc, Mutex};

use handlers::{healthz, readyz};
use tower::ServiceBuilder;
use tower_http::{propagate_header::PropagateHeaderLayer, trace::TraceLayer};

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
    router: Router,
}

impl Default for App {
    fn default() -> Self {
        // Create our service
        let service = ServiceBuilder::new()
            // High level logging of requests and responses
            .layer(TraceLayer::new_for_http())
            // Propagate `X-Request-Id`s from requests to responses
            .layer(PropagateHeaderLayer::new(HeaderName::from_static(
                "x-request-id",
            )));
        let router = Router::new()
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .layer(service);
        Self {
            name: String::from("inference-gateway"),
            status: AppStatus::None,
            router,
        }
    }
}

impl App {

    pub fn add_router(&mut self, path: &str, router: Router){
        self.router = self.router.clone().nest(path, router);
    }

    pub async fn serve(mut self, addr: String) {
        // Set initial status to Starting
        self.status = AppStatus::Starting;
        let router = self.router.clone();
        let shared_state = Arc::new(Mutex::new(self));

        // Create router with shared state
        let router = router.clone().layer(Extension(shared_state.clone()));
        
        debug!("Initialized state");

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

        match axum::serve(listener, router).await {
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
