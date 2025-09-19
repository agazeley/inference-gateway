use inference_gateway::app::App;

#[tokio::main]
async fn main() {
    env_logger::Builder::new()
        .parse_env(env_logger::Env::new().filter_or("", "info"))
        .init();

    let addr = "0.0.0.0:3000";
    let app = App::default();
    app.serve(addr.to_string()).await;
}
