//! src/middleware/layer.rs
use axum::{
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};


pub async fn api_key_middleware (
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode>{
    let auth_header = req.headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok());

    let expected = match  std::env::var("API_KEY") {
        Ok(v) => v,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };
    
    match auth_header {
        Some(key) if key == expected => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
