//! src/middleware/layer.rs
use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use crate::server::AppState;
use std::sync::Arc;


pub async fn api_key_middleware (
    State(state): State<Arc<AppState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode>{
    let auth_header = req.headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok());

    let expected = &state.valid_api_keys;
    
    match auth_header {
        Some(key) if key == expected => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
