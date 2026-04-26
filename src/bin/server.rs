/// Recommendation serving API (Axum + Shuttle).
///
/// Endpoints:
///   POST /recommend   body: {"user_id": 0, "candidates": [1, 2, 3, ...]}
///   GET  /health
///
///   curl -s http://localhost:8000/health
///   curl -s -X POST http://localhost:8000/recommend \
///       -H 'Content-Type: application/json' \
///       -d '{"user_id": 0, "candidates": [1,2,3,4,5]}'
use shuttle_axum::axum::{
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
use burn_recsys::models::ncf::{NeuMF, NeuMFConfig};
use serde::{Deserialize, Serialize};
use shuttle_runtime::{CustomError, SecretStore};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::Mutex;
use tracing::{info, instrument, warn};

type B = NdArray<f32>;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Settings {
    // Note: The `model` path is now handled by `StaticFolder`
    // and injected into the main service function.
    num_users: usize,
    num_items: usize,
    gmf_dim: usize,
    mlp_embed_dim: usize,
}

// ── Shared state ──────────────────────────────────────────────────────────────

struct AppState {
    /// Tokio Mutex: NeuMF's Param<Tensor> is not Sync.
    model: Mutex<NeuMF<B>>,
    device: <B as Backend>::Device,
    num_users: usize,
    num_items: usize,
}

// ── DTOs ──────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RecommendRequest {
    user_id: u32,
    /// Candidate item IDs to score and rank.
    /// If empty, all items are ranked (expensive for large catalogs).
    candidates: Vec<u32>,
}

#[derive(Serialize)]
struct RecommendResponse {
    user_id: u32,
    ranked: Vec<u32>,
    latency_ms: f64,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    num_users: usize,
    num_items: usize,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

#[instrument(skip(state, payload), fields(user_id = payload.user_id))]
async fn recommend(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RecommendRequest>,
) -> impl IntoResponse {
    // ── Input validation ─────────────────────────────────────────────────────
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
        )
            .into_response();
    }

    let candidates = if payload.candidates.is_empty() {
        (0..state.num_items as u32).collect::<Vec<_>>()
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
        )
            .into_response();
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

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (StatusCode::OK, Json(HealthResponse {
        status: "ok",
        num_users: state.num_users,
        num_items: state.num_items,
    }))
}

// ── Main: Shuttle Entrypoint ──────────────────────────────────────────────────

#[shuttle_runtime::main]
async fn axum(
    #[shuttle_runtime::Secrets] secrets: SecretStore,
) -> shuttle_axum::ShuttleAxum {
    let model_folder = PathBuf::from("checkpoints");
    // Manually parse settings from the secrets map.
    let settings = Settings {
        num_users: secrets
            .get("num_users")
            .ok_or_else(|| CustomError::msg("num user not found"))?
            .parse()
            .map_err(|_| CustomError::msg("num user invalid"))?,
        num_items: secrets
            .get("num_items")
            .ok_or_else(|| CustomError::msg("num item not found"))?
            .parse()
            .map_err(|_| CustomError::msg("num item invalid"))?,
        gmf_dim: secrets
            .get("gmf_dim")
            .ok_or_else(|| CustomError::msg("gmf dim not found"))?
            .parse()
            .map_err(|_| CustomError::msg("gmf dim invalid"))?,
        mlp_embed_dim: secrets
            .get("mlp_embed_dim")
            .ok_or_else(|| CustomError::msg("mlp embed dim not found"))?
            .parse()
            .map_err(|_| CustomError::msg("mlp embed dim invalid"))?,
    };

    // The model file is expected to be 'best.mpk' inside the static folder.
    let model_path = model_folder.join("best.mpk");
    info!("Loading model from {:?}...", &model_path);
    let device: <B as Backend>::Device = Default::default();

    let model_config = NeuMFConfig {
        num_users: settings.num_users,
        num_items: settings.num_items,
        gmf_dim: settings.gmf_dim,
        mlp_layers: vec![128, 64, 32, 16],
        mlp_embed_dim: settings.mlp_embed_dim,
    };

    let model = model_config
        .init::<B>(&device)
        .load_file(&model_path, &CompactRecorder::new(), &device)
        .map_err(|e| CustomError::msg(format!("model not found! failed to load model {}", e)))?;

    info!("Model loaded ({} users, {} items)", settings.num_users, settings.num_items);

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        device,
        num_users: settings.num_users,
        num_items: settings.num_items,
    });

    let router = Router::new()
        .route("/recommend", post(recommend))
        .route("/health", get(health))
        .with_state(state);

    Ok(router.into())
}
