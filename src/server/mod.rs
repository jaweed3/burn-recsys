pub mod handlers;
pub mod model;
pub mod router;
pub mod state;
pub mod retrieval;

use std::net::SocketAddr;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::collections::{HashMap, HashSet};
use tokio::sync::mpsc;
use tracing::info;
use burn::backend::NdArray;
use burn::tensor::backend::Backend;

pub use state::{Settings, AppState, InferenceJob};
pub use handlers::{RecommendRequest, RecommendResponse, HealthResponse};
use model::load_model;
use handlers::run_inference;
use router::create_router;
use retrieval::SimpleRetriever;
use crate::data::{PolarsDataset, RecsysDataset};

type B = NdArray<f32>;

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Shutdown signal received. Starting cleaning...");
}

pub async fn run(settings: Settings) -> anyhow::Result<()> {
    info!("Configuration: {:?}", settings);
    info!("Loading dataset interactions from {} for retrieval...", settings.data_path);
    
    // Load interactions to know what to exclude during retrieval
    let dataset = PolarsDataset::myket(&settings.data_path)?;
    let mut user_positives: HashMap<u32, HashSet<u32>> = HashMap::new();
    for &(u, i) in dataset.interactions() {
        user_positives.entry(u).or_default().insert(i);
    }
    info!("Loaded interactions for {} users", user_positives.len());

    info!("Loading model from {}...", settings.model);
    let device: <B as Backend>::Device = Default::default();
    let ready = Arc::new(AtomicBool::new(false));
    
    // Test if base model loads correctly
    let _base_model = load_model(&settings, &device)?;
    info!("Model loaded ({} users, {} items)", settings.num_users, settings.num_items);

    let (tx, rx) = mpsc::channel::<InferenceJob>(1024);
    let rx = Arc::new(tokio::sync::Mutex::new(rx));
    
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    for i in 0..workers {
        let model = load_model(&settings, &device)?;
        let rx = rx.clone();
        let device = device.clone();

        tokio::spawn(async move {
            info!("Inference worker {} started", i);
            loop {
                let job = {
                    let mut guard = rx.lock().await;
                    guard.recv().await
                };

                let Some(job) = job else { break };

                let ranked = run_inference(
                    model.as_ref(),
                    &device,
                    job.user_id,
                    &job.candidates
                );

                let _ = job.resp.send(ranked);
            }
        });
    }

    let retriever = Arc::new(SimpleRetriever::new(settings.num_items));

    let state = Arc::new(AppState {
        tx,
        num_users: settings.num_users, 
        num_items: settings.num_items,
        ready: ready.clone(),
        valid_api_keys: settings.valid_api_keys.clone(),
        user_positives,
        retriever,
        retrieval_limit: settings.retrieval_limit,
    });

    let app = create_router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], settings.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Listening on {}", addr);
    
    ready.store(true, Ordering::Release);
    
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;
    
    Ok(())
}
