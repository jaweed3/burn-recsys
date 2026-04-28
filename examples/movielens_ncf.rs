/// Train NeuMF on the MovieLens 1M dataset (temporal leave-one-out eval).
///
/// Download data first:
///   uv run python scripts/download_movielens.py
///
/// Usage:
///   cargo run --release --example movielens_ncf
///   APP_epochs=10 cargo run --release --example movielens_ncf
use burn::backend::{Autodiff, NdArray};
use burn_recsys::{
    data::{PolarsDataset, RecsysDataset},
    models::ncf::NeuMFConfig,
    trainer::{TrainConfig, Trainer, TrainerSettings},
};

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Load configuration
    let builder = config::Config::builder()
        .add_source(config::File::with_name("config/train_movielens.toml"))
        .add_source(config::Environment::with_prefix("APP")
            .try_parsing(true)
            .separator("__"));
    
    let config_built = builder.build()?;
    let settings: TrainerSettings = config_built.try_deserialize()?;

    println!("Configuration: {:?}", settings);
    println!("Loading MovieLens dataset from: {}", settings.data_path);
    
    let dataset = PolarsDataset::movielens(&settings.data_path)?;
    println!(
        "Loaded: {} users, {} items, {} interactions",
        dataset.num_users(), dataset.num_items(), dataset.interactions().len()
    );

    let (train_ds, val_interactions) = dataset.leave_one_out();
    println!(
        "Split: {} train | {} val (leave-one-out)",
        train_ds.interactions().len(), val_interactions.len()
    );

    let checkpoint_dir = settings.checkpoint_dir.as_ref().map(std::path::PathBuf::from);

    let config = TrainConfig {
        num_epochs: settings.epochs,
        batch_size: settings.batch_size,
        learning_rate: settings.learning_rate,
        embedding_dim: settings.embedding_dim,
        neg_ratio: settings.neg_ratio,
        checkpoint_dir,
        ..Default::default()
    };

    let model_config = NeuMFConfig {
        num_users: dataset.num_users(),
        num_items: dataset.num_items(),
        gmf_dim: config.embedding_dim,
        mlp_layers: settings.mlp_layers(),
        mlp_embed_dim: config.embedding_dim,
    };

    let device = Default::default();
    let model = model_config.init::<B>(&device);
    println!("NeuMF params: {}", model.num_params());

    let trainer = Trainer::<B>::new(config, device);
    let _trained = trainer.train(model, &train_ds, &val_interactions);

    println!("\nDone. Training session completed.");
    Ok(())
}
