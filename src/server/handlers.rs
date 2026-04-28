use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use crate::models::Scorable;
use burn::{
    backend::NdArray,
    tensor::{Int, Tensor, activation::sigmoid, backend::Backend},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, atomic::Ordering};
use std::collections::HashSet;
use tokio::sync::oneshot;
use tracing::{info, warn};
use std::time::Instant;

use super::state::{AppState, InferenceJob};

type B = NdArray<f32>;

// ── DTOs ──────────────────────────────────────────────────────────────────────

#[derive(Deserialize, Serialize)]
pub struct RecommendRequest {
    pub user_id: u32,
    pub candidates: Option<Vec<u32>>,
}

#[derive(Deserialize, Serialize)]
pub struct RecommendResponse {
    pub user_id: u32,
    pub ranked: Vec<u32>,
    pub latency_ms: f64,
}

#[derive(Deserialize, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub num_users: usize,
    pub num_items: usize,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

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

pub async fn get_ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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

    let user_id = payload.user_id;

    // --- Retrieval Stage ---
    let candidates = match payload.candidates {
        Some(c) => {
            if c.is_empty() {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(serde_json::json!({
                        "error": "candidates array provided but empty"
                    })),
                ).into_response();
            }
            else if c.len() > state.max_candidates {
                return (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    Json(serde_json::json!({
                        "error": "candidates array is too long (max 200)"
                    }))
                ).into_response();
            }
            c
        }
        None => {
            let empty_set = HashSet::new();
            let exclude = state.user_positives.get(&user_id).unwrap_or(&empty_set);
            state.retriever.generate(user_id, state.retrieval_limit, exclude)
        }


    };

    if candidates.is_empty() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "failed to generate or find candidates"
            })),
        ).into_response();
    }

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
    let (resp_tx, resp_rx) = oneshot::channel();

    if state.tx.send(InferenceJob {
        user_id,
        candidates: candidates.clone(),
        resp: resp_tx,
    }).await.is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "inference queue unavailable"
            })),
        ).into_response();
    }

    let ranked = match resp_rx.await {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "worker failed to return inference result."
                })),
            ).into_response();
        }
    };

    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
    info!(n = ranked.len(), latency_ms, "recommend ok");

    Json(RecommendResponse { user_id, ranked, latency_ms }).into_response()
}

pub fn run_inference(
    model: &(dyn Scorable<B> + Send),
    device: &<B as Backend>::Device,
    user_id: u32,
    candidates: &[u32]
) -> Vec<u32> {
    let n = candidates.len();

    let users = vec![user_id as i32; n];
    let items: Vec<i32> = candidates.iter().map(|&x| x as i32).collect();

    let user_t = Tensor::<B, 1, Int>::from_ints(users.as_slice(), device);
    let item_t = Tensor::<B, 1, Int>::from_ints(items.as_slice(), device);

    let scores = sigmoid(model.score(user_t, item_t));
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
