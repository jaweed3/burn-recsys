//! src/server.rs
use axum::{
    middleware,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use crate::middleware::layer::api_key_middleware;
use crate::models::{
    Scorable,
    NeuMF,
    DeepFM,
    GMF,
    ncf::NeuMFConfig,
    deepfm::DeepFMConfig,
    gmf::GMFConfig
};
use burn::{
    backend::NdArray,
    module::Module,
    record::CompactRecorder,
    tensor::{Int, Tensor, activation::sigmoid, backend::Backend},
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::{Arc, atomic::{AtomicBool, Ordering}}};
use tokio::sync::{mpsc, oneshot};
use tracing::info;

type B = NdArray<f32>;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct Settings {
    pub model: String,
    pub model_type: String,
    pub port: u16,
    pub num_users: usize,
    pub num_items: usize,
    pub gmf_dim: usize,
    pub mlp_embed_dim: usize,
    pub mlp_layers: Vec<usize>,
    pub valid_api_keys: String,
}

// ── Shared state ──────────────────────────────────────────────────────────────

pub struct InferenceJob {
    pub user_id: u32,
    pub candidates: Vec<u32>,
    pub resp: oneshot::Sender<Vec<u32>>,
}

pub struct AppState {
    pub tx: mpsc::Sender<InferenceJob>,
    pub num_users: usize,
    pub num_items: usize,
    pub ready: Arc<AtomicBool>,
    pub device: <B as Backend>::Device,
}

use std::time::Instant;
use tracing::warn;


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

fn run_inference (
    model: &NeuMF<B>,
    device: &<B as Backend>::Device,
    user_id: u32,
    candidates: &[u32]
    ) -> Vec<u32> {
    let n = candidates.len();

    let users = vec![user_id as i32; n];
    let items: Vec<i32> = candidates.iter().map(|&x| x as i32).collect();

    let user_t = Tensor::<B, 1, Int>::from_ints(users.as_slice(), device);
    let item_t = Tensor::<B, 1, Int>::from_ints(items.as_slice(), device);

    let scores = sigmoid(model.forward(user_t, item_t));
    let scores_vec: Vec<f32> =
        scores.into_data().to_vec::<f32>().unwrap_or_default();

    let mut idx_scores: Vec<(usize, f32)> = 
        scores_vec.into_iter().enumerate().collect();

    idx_scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap()
    });

    idx_scores
        .into_iter()
        .map(|(i, _)| candidates[i] )
        .collect()
}

pub async fn get_ready (
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // if the model loaded, turn ready into true, response 200
    // else return 503
    if state.ready.load(Ordering::Acquire) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
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
    let (resp_tx, resp_rx) = oneshot::channel();

    state.tx.send(InferenceJob {
        user_id,
        candidates: candidates.clone(),
        resp: resp_tx,
    }).await.unwrap();

    let ranked = resp_rx.await.unwrap();

    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
    info!(n = ranked.len(), latency_ms, "recommend ok");

    Json(RecommendResponse { user_id, ranked, latency_ms }).into_response()
}


pub async fn run(settings: Settings) -> anyhow::Result<()> {
    info!("Configuration: {:?}", settings);
    info!("Loading model from {}...", settings.model);
    let device: <B as Backend>::Device = Default::default();
    let ready = Arc::new(AtomicBool::new(false));

    info!("Model loaded ({} users, {} items)", settings.num_users, settings.num_items);

    let (tx, rx) = mpsc::channel::<InferenceJob>(1024);
    let rx = Arc::new(tokio::sync::Mutex::new(rx));
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    for _ in 0..workers {
        let model = load_model(&settings, &device)?;
        let rx = rx.clone();
        let device = device.clone();

        tokio::spawn(async move {
            loop {
                let job = {
                    let mut guard = rx.lock().await;
                    guard.recv().await
                };

                let Some(job) = job else { break };

                let ranked = run_inference(
                    &model,
                    &device,
                    job.user_id,
                    &job.candidates
                );

                let _ = job.resp.send(ranked);
            }
        });
    }

    let state = Arc::new(AppState { 
        tx, 
        num_users: settings.num_users, 
        num_items: settings.num_items,
        ready: ready.clone(),
    });

    let protected = Router::new()
        .route("/recommend", post(recommend))
        .layer(middleware::from_fn(api_key_middleware));
    
    let app = Router::new()
        .route("/ready", get(get_ready))
        .route("/health", get(health))
        .merge(protected)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], settings.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Listening on {}", addr);
    
    ready.store(true, Ordering::Release);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

fn load_model(
    settings: &Settings,
    device: &<B as Backend>::Device,
) -> anyhow::Result<Arc<dyn Scorable<B> + Send + Sync>>{
    match settings.model_type.as_str() {
        "neumf" => {
            let config = NeuMFConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                gmf_dim: settings.gmf_dim,
                mlp_layers: settings.mlp_layers,
                mlp_embed_dim: settings.mlp_embed_dim,
            };

            let model = config
                .init::<B>(device)
                .load_file(&settings.model, &CompactRecorder::new(), device)?;

            Ok(Arc::new(model))
        }

        "deepfm" => {
            let config = DeepFMConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                embedding_dim: settings.gmf_dim,
                mlp_layers: settings.mlp_layers,
            };

            let model = config
                .init::<B>(device)
                .load_file(&settings.model, CompactRecorder, device)>;

            Ok(Arc::new(model))
        }

        "gmf" => {
            let config = GMFConfig {
                num_users: settings.num_users,
                num_items: settings.num_items,
                embedding_dim: settings.gmf_dim,
            };
        }
    }
}
