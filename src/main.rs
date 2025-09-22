use inference_gateway::{api, server::Server};
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

    let addr = "0.0.0.0:3000";
    let mut srv = Server::default();

    let api_router = api::get_router().expect("Failed to create API router");
    srv.add_router("/api/v1", api_router);
    srv.serve(addr.to_string()).await;
}
