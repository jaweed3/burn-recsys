//! src/server.rs
use crate::models::ncf::{NeuMF, NeuMFConfig};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use burn::{
    backend::NdArray,
    module::Module,
    record::CompactRecorder,
    tensor::{activation::sigmoid, backend::Backend, Int, Tensor},
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;
use tracing::info;

type B = NdArray<f32>;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct Settings {
    pub model: String,
    pub port: u16,
    pub num_users: usize,
    pub num_items: usize,
    pub gmf_dim: usize,
    pub mlp_embed_dim: usize,
    pub mlp_layers: Vec<usize>,
}

// ── Shared state ──────────────────────────────────────────────────────────────

pub struct AppState {
    /// Tokio Mutex: NeuMF's Param<Tensor> is not Sync.
    model: Mutex<NeuMF<B>>,
    device: <B as Backend>::Device,
    num_users: usize,
    num_items: usize,
}

use std::time::Instant;
use tracing::warn;

// ... other code ...

// ── DTOs & Handlers ───────────────────────────────────────────────────────────

#[derive(Deserialize, Serialize)]
pub struct RecommendRequest {
    pub user_id: u32,
    pub candidates: Vec<u32>,
}

#[derive(Serialize)]
pub struct RecommendResponse {
    pub user_id: u32,
    pub ranked: Vec<u32>,
    pub latency_ms: f64,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub num_users: usize,
    pub num_items: usize,
}

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok",
            num_users: state.num_users,
            num_items: state.num_items,
        }),
    )
}

pub async fn recommend(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RecommendRequest>,
) -> impl IntoResponse {
    if payload.user_id as usize >= state.num_users {
        warn!(user_id = payload.user_id, num_users = state.num_users, "user_id out of range");
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": format!(
                    "user_id {} is out of range (num_users={})",
                    payload.user_id, state.num_users
                )
            })),
        ).into_response();
    }

    let candidates = if payload.candidates.is_empty() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": format!(
                    "array user_id {} is empty!! cannot processing entity.",
                    payload.user_id
                )
            })),
        ).into_response();
    } else if payload.candidates.len() > 200 {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": format!(
                    "user_id {} array is too long.",
                    payload.user_id
                )
            }))
        ).into_response();
    } else {
        payload.candidates
    };

    if let Some(&bad) = candidates.iter().find(|&&c| c as usize >= state.num_items) {
        warn!(item_id = bad, num_items = state.num_items, "candidate item_id out of range");
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": format!(
                    "candidate item_id {} is out of range (num_items={})",
                    bad, state.num_items
                )
            })),
        ).into_response();
    }

    let t0 = Instant::now();
    let user_id = payload.user_id;
    let n = candidates.len();

    let users: Vec<i32> = vec![user_id as i32; n];
    let items: Vec<i32> = candidates.iter().map(|&x| x as i32).collect();

    // Lock model → run inference → release
    let ranked = {
        let model = state.model.lock().await;
        let user_t = Tensor::<B, 1, Int>::from_ints(users.as_slice(), &state.device);
        let item_t = Tensor::<B, 1, Int>::from_ints(items.as_slice(), &state.device);

        let scores = sigmoid(model.forward(user_t, item_t));
        let scores_vec: Vec<f32> = scores.into_data().to_vec::<f32>().unwrap_or_default();

        let mut idx_scores: Vec<(usize, f32)> = scores_vec.into_iter().enumerate().collect();
        idx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        idx_scores.into_iter().map(|(i, _)| candidates[i]).collect::<Vec<u32>>()
    };

    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
    info!(n = ranked.len(), latency_ms, "recommend ok");

    Json(RecommendResponse { user_id, ranked, latency_ms }).into_response()
}


pub async fn run(settings: Settings) -> anyhow::Result<()> {
    info!("Configuration: {:?}", settings);
    info!("Loading model from {}...", settings.model);
    let device: <B as Backend>::Device = Default::default();

    let model_config = NeuMFConfig {
        num_users: settings.num_users,
        num_items: settings.num_items,
        gmf_dim: settings.gmf_dim,
        mlp_layers: settings.mlp_layers.clone(),
        mlp_embed_dim: settings.mlp_embed_dim,
    };

    let model = model_config
        .init::<B>(&device)
        .load_file(&settings.model, &CompactRecorder::new(), &device)
        .map_err(|e| anyhow::anyhow!("Failed to load {}: {e}", settings.model))?;

    info!("Model loaded ({} users, {} items)", settings.num_users, settings.num_items);

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        device,
        num_users: settings.num_users,
        num_items: settings.num_items,
    });

    let app = Router::new()
        .route("/recommend", post(recommend))
        .route("/health", get(health))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], settings.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Listening on {}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}
