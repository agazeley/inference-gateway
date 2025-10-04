use inference_gateway::{router, server::Server};
use sqlx::any;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() {
    // Initialize tracing with env filter, e.g. RUST_LOG=info,tower_http=info
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn,tower_http=info".into()),
        )
        .with(fmt::layer().compact())
        .init();

    any::install_default_drivers();

    let addr = "0.0.0.0:3000";
    let mut srv = Server::default();

    let api_router = match router::get_router().await {
        Ok(r) => r,
        Err(e) => {
            panic!("error creating router: {:?}", e.to_string());
        }
    };
    srv.add_router("/api/v1", api_router);
    srv.serve(addr.to_string()).await;
}
