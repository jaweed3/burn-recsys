//! tests/server.rs
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn_recsys::{
    models::ncf::NeuMFConfig,
    server::{run, RecommendRequest, RecommendResponse, Settings},
    telemetry::{init_metrics, init_subscriber},
};
use once_cell::sync::Lazy;
use portpicker::pick_unused_port;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

// Ensure that the `tracing` stack is only initialised once using `once_cell`
static TRACING: Lazy<()> = Lazy::new(|| {
    let default_filter = "info".to_string();
    let format = "compact".to_string();
    init_subscriber(default_filter, format);
});

struct TestApp {
    pub addr: SocketAddr,
    // Keep files in scope to prevent deletion
    _model_file: NamedTempFile,
    _data_file: NamedTempFile,
}

/// Creates a dummy model and saves it to a temporary file.
fn create_dummy_model() -> (NamedTempFile, NamedTempFile, Settings) {
    let model_file = NamedTempFile::new().expect("Failed to create temp file");
    
    // Create a dummy data file for retrieval tests
    let data_file = NamedTempFile::new().expect("Failed to create temp data file");
    std::fs::write(data_file.path(), "user_id,app_name,timestamp\n1,app1,123456\n").unwrap();

    let num_users = 100;
    let num_items = 100;
    let mlp_layers = vec![16, 8];
    let mlp_embed_dim = 8;
    let max_candidates = 200;
    
    let settings = Settings {
        model: model_file.path().to_str().unwrap().to_string(),
        model_type: "neumf".to_string(),
        port: pick_unused_port().expect("No ports free"),
        num_users,
        num_items,
        gmf_dim: 8,
        mlp_embed_dim,
        mlp_layers: mlp_layers.clone(),
        valid_api_keys: "admin_bismillah".to_string(),
        retrieval_limit: 10,
        data_path: data_file.path().to_str().unwrap().to_string(),
        max_candidates: max_candidates.clone(),
    };

    let model_config = NeuMFConfig {
        num_users: settings.num_users,
        num_items: settings.num_items,
        gmf_dim: settings.gmf_dim,
        mlp_layers,
        mlp_embed_dim: settings.mlp_embed_dim,
    };

    let model = model_config.init::<Autodiff<NdArray>>(&Default::default());
    model
        .save_file(model_file.path(), &CompactRecorder::new())
        .expect("Failed to save dummy model");

    (model_file, data_file, settings)
}

// Helper to spawn the app in the background
async fn spawn_app() -> TestApp {
    Lazy::force(&TRACING);

    let (model_file, data_file, settings) = create_dummy_model();
    let addr = SocketAddr::from(([127, 0, 0, 1], settings.port));

    let metrics = init_metrics();
    tokio::spawn(run(settings, metrics));

    let client = reqwest::Client::new(); 
    let deadline = Instant::now() + Duration::from_secs(5);

    loop {
        if Instant::now() > deadline {
            panic!("server failed to start within timeout!");
        }

        match client
            .get(format!("http://{}/health", addr))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => break,
            _ => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }

    TestApp { 
        addr, 
        _model_file: model_file,
        _data_file: data_file,
    }
}

#[tokio::test]
async fn health_check_works() {
    let app = spawn_app().await;
    let client = reqwest::Client::new();

    let response = client
        .get(&format!("http://{}/health", app.addr))
        .send()
        .await
        .expect("Failed to execute request.");

    assert!(response.status().is_success());
    let health = response.json::<serde_json::Value>().await.unwrap();
    assert_eq!(health["status"], "ok");
}

#[tokio::test]
async fn recommend_returns_200_for_valid_data() {
    let app = spawn_app().await;
    let client = reqwest::Client::new();
    let body = RecommendRequest {
        user_id: 1,
        candidates: Some(vec![1, 2, 3, 4, 5]),
    };

    let response = client
        .post(&format!("http://{}/recommend", app.addr))
        .header("x-api-key", "admin_bismillah")
        .json(&body)
        .send()
        .await
        .expect("Failed to execute request.");

    assert_eq!(response.status().as_u16(), 200);
}

#[tokio::test]
async fn recommend_works_without_candidates() {
    let app = spawn_app().await;
    let client = reqwest::Client::new();
    let body = RecommendRequest {
        user_id: 1,
        candidates: None,
    };

    let response = client
        .post(&format!("http://{}/recommend", app.addr))
        .header("x-api-key", "admin_bismillah")
        .json(&body)
        .send()
        .await
        .expect("Failed to execute request.");

    assert_eq!(response.status().as_u16(), 200);
    let res = response.json::<RecommendResponse>().await.unwrap();
    assert!(!res.ranked.is_empty());
}

#[tokio::test]
async fn recommend_returns_422_for_invalid_data() {
    let app = spawn_app().await;
    let client = reqwest::Client::new();
    let body = RecommendRequest {
        user_id: 999, // Out of bounds
        candidates: Some(vec![1, 2, 3]),
    };

    let response = client
        .post(&format!("http://{}/recommend", app.addr))
        .header("x-api-key", "admin_bismillah")
        .json(&body)
        .send()
        .await
        .expect("Failed to execute request.");

    assert_eq!(response.status().as_u16(), 422);
}

#[tokio::test]
async fn recommend_fails_id_candidated_too_long() {
    let app = spawn_app().await;
    let client = reqwest::Client::new();

    let body = RecommendRequest {
        user_id: 1,
        candidates: Some((0..201).collect()),
    };

    let response = client
        .post(&format!("http://{}/recommend", app.addr))
        .header("x-api-key", "admin_bismillah")
        .json(&body)
        .send()
        .await
        .expect("failed to execute response.");

    assert_eq!(response.status().as_u16(), 413);
}
