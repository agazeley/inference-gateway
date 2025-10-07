use inference_gateway::{router, server::Server};
use sqlx::any;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

use clap::{Parser, ValueEnum};

use std::fmt::{Display, Formatter, Result};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(default_value = "0.0.0.0")]
    addr: String,
    #[arg(default_value_t = 3000)]
    port: i32,
    #[arg(default_value = "info")]
    log_level: LogLevel,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Error,
}

impl Display for LogLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let level = match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Error => "error",
        };
        write!(f, "{}", level)
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize tracing with env filter, e.g. RUST_LOG=info,tower_http=info
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("{},ort=warn,tower_http=info", args.log_level).into()),
        )
        .with(fmt::layer().compact())
        .init();

    any::install_default_drivers();

    let addr = format!("{}:{}", args.addr, args.port);
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
