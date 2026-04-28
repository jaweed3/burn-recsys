use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use crate::middleware::layer::api_key_middleware;
use super::state::AppState;
use super::handlers::{health, get_ready, recommend};

pub fn create_router(state: Arc<AppState>) -> Router {
    let protected = Router::new()
        .route("/recommend", post(recommend))
        .layer(middleware::from_fn_with_state(state.clone(), api_key_middleware));
    
    Router::new()
        .route("/ready", get(get_ready))
        .route("/health", get(health))
        .merge(protected)
        .with_state(state)
}
