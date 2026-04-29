use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use std::sync::Arc;
use crate::middleware::layer::api_key_middleware;
use super::state::AppState;
use super::handlers::{self, health, get_ready, recommend};

#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::health,
        handlers::get_ready,
        handlers::recommend,
    ),
    components(
        schemas(handlers::HealthResponse, handlers::ReadyResponse, handlers::RecommendRequest, handlers::RecommendResponse)
    ),
    tags(
        (name = "Recommendation API", description = "endpoints for serving ML models")
    )
)]
struct ApiDoc;

pub fn create_router(state: Arc<AppState>) -> Router {
    let protected = Router::new()
        .route("/recommend", post(recommend))
        .layer(middleware::from_fn_with_state(state.clone(), api_key_middleware));
    
    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/ready", get(get_ready))
        .route("/health", get(health))
        .merge(protected)
        .with_state(state)
}
