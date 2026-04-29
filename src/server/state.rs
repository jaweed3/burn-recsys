use serde::Deserialize;
use std::sync::{Arc, atomic::AtomicBool};
use std::collections::{HashMap, HashSet};
use tokio::sync::{mpsc, oneshot};
use crate::telemetry::Metrics;
use super::retrieval::CandidateGenerator;

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
    pub retrieval_limit: usize,
    pub data_path: String,
    pub max_candidates: usize,
}

// ── Shared state ──────────────────────────────────────────────────────────────

pub struct InferenceJob {
    pub user_id: u32,
    pub candidates: Option<Vec<u32>>,
    pub resp: oneshot::Sender<Vec<u32>>,
}

pub struct AppState {
    pub tx: mpsc::Sender<InferenceJob>,
    pub num_users: usize,
    pub num_items: usize,
    pub ready: Arc<AtomicBool>,
    pub valid_api_keys: String,
    pub user_positives: HashMap<u32, HashSet<u32>>,
    pub retriever: Arc<dyn CandidateGenerator>,
    pub retrieval_limit: usize,
    pub max_candidates: usize,
    pub metrics: Metrics,
    pub model_type: String,
}
