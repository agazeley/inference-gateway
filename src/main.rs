use inference_gateway::{api, app::App};

#[tokio::main]
async fn main() {
    env_logger::Builder::new()
        .parse_env(env_logger::Env::new().filter_or("", "info"))
        .init();

    let addr = "0.0.0.0:3000";
    let mut app = App::default();
    
    let api_router = api::get_router();
    app.add_router("/api/v1", api_router);
    app.serve(addr.to_string()).await;
}
