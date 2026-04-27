//! src/bin/server.rs
use burn_recsys::server::{run, Settings};
use burn_recsys::telemetry::init_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize telemetry
    let log_format = std::env::var("LOG_FORMAT").unwrap_or_else(|_| "compact".to_string());
    init_subscriber("burn_recsys=info,info".into(), log_format);

    // Load configuration
    let builder = config::Config::builder()
        .add_source(config::File::with_name("config/default.toml"))
        .add_source(config::Environment::with_prefix("APP").separator("__"));
    
    let settings: Settings = builder.build()?.try_deserialize()?;

    // Run the server
    run(settings).await
}
