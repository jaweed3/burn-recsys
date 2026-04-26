/// Train NeuMF on the MovieLens 1M dataset (temporal leave-one-out eval).
///
/// Download data first:
///   uv run python scripts/download_movielens.py
///
/// Usage:
///   cargo run --release --example movielens_ncf
///   cargo run --release --example movielens_ncf -- --data data/movielens.csv --epochs 10
use burn::backend::{Autodiff, NdArray};
use burn_recsys::{
    data::{PolarsDataset, RecsysDataset},
    models::ncf::NeuMFConfig,
    trainer::{TrainConfig, Trainer},
};
use clap::Parser;

type B = Autodiff<NdArray<f32>>;

#[derive(Parser, Debug)]
#[command(about = "Train NeuMF on MovieLens dataset")]
struct Args {
    /// Path to the MovieLens CSV file (columns: user_id, app_id)
    #[arg(long, default_value = "data/movielens.csv")]
    data: String,

    /// Number of training epochs
    #[arg(long, default_value_t = 20)]
    epochs: usize,

    /// Embedding dimension
    #[arg(long, default_value_t = 64)]
    embed_dim: usize,

    /// Checkpoint directory (set to empty string to disable)
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Loading MovieLens dataset from: {}", args.data);
    let dataset = PolarsDataset::movielens(&args.data)?;
    println!(
        "Loaded: {} users, {} items, {} interactions",
        dataset.num_users(), dataset.num_items(), dataset.interactions().len()
    );

    let (train_ds, val_interactions) = dataset.leave_one_out();
    println!(
        "Split: {} train | {} val (leave-one-out)",
        train_ds.interactions().len(), val_interactions.len()
    );

    let checkpoint_dir = if args.checkpoint_dir.is_empty() {
        None
    } else {
        Some(std::path::PathBuf::from(&args.checkpoint_dir))
    };

    let config = TrainConfig {
        num_epochs: args.epochs,
        embedding_dim: args.embed_dim,
        checkpoint_dir,
        ..Default::default()
    };

    let model_config = NeuMFConfig {
        num_users: dataset.num_users(),
        num_items: dataset.num_items(),
        gmf_dim: config.embedding_dim,
        mlp_layers: config.mlp_layers.clone(),
        mlp_embed_dim: config.embedding_dim,
    };

    let device = Default::default();
    let model = model_config.init::<B>(&device);
    println!("NeuMF params: {}", model.num_params());

    let trainer = Trainer::<B>::new(config, device);
    let _trained = trainer.train(model, &train_ds, &val_interactions);

    println!("\nDone. Best checkpoint saved to {}/best", args.checkpoint_dir);
    Ok(())
}
