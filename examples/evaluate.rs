/// Evaluate GMF vs NeuMF with leave-one-out HR@10 and NDCG@10.
///
/// Usage:
///   cargo run --release --example evaluate
///   cargo run --release --example evaluate -- --users 500 --epochs 10
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::tensor::activation::sigmoid;
use burn_recsys::{
    data::{PolarsDataset, RecsysDataset},
    metrics::evaluate,
    models::{gmf::GMFConfig, ncf::NeuMFConfig, Scorable},
    trainer::{TrainConfig, Trainer},
};
use clap::Parser;

type BInner = NdArray<f32>;
type B = Autodiff<BInner>;

#[derive(Parser, Debug)]
#[command(about = "Compare GMF vs NeuMF with leave-one-out HR@10 and NDCG@10")]
struct Args {
    /// Path to the Myket CSV file
    #[arg(long, default_value = "data/myket.csv")]
    data: String,

    /// Max users to evaluate (for speed)
    #[arg(long, default_value_t = 1000)]
    users: usize,

    /// Number of training epochs
    #[arg(long, default_value_t = 10)]
    epochs: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let dataset = PolarsDataset::myket(&args.data)?;
    let (train_ds, val_interactions) = dataset.leave_one_out();
    println!(
        "Dataset: {} users | {} items | train={} | val={}",
        dataset.num_users(), dataset.num_items(),
        train_ds.interactions().len(), val_interactions.len()
    );

    let config = TrainConfig {
        num_epochs: args.epochs,
        checkpoint_dir: None, // skip saving during eval comparison
        ..Default::default()
    };
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let eval_device: <BInner as burn::tensor::backend::Backend>::Device = Default::default();
    let k = 10;
    let max_users = args.users;

    // ── GMF ──────────────────────────────────────────────────────────────────
    println!("\n[GMF] Training {} epochs...", args.epochs);
    let gmf_cfg = GMFConfig {
        num_users: dataset.num_users(),
        num_items: dataset.num_items(),
        embedding_dim: config.embedding_dim,
    };
    let gmf_model = gmf_cfg.init::<B>(&device);
    let gmf_trainer = Trainer::<B>::new(config.clone(), device.clone());
    let gmf_trained = gmf_trainer.train(gmf_model, &train_ds, &val_interactions);

    println!("\n[GMF] Evaluating (k={k}, max_users={max_users})...");
    let gmf_inner = gmf_trained.valid();
    let gmf_result = evaluate::<BInner, _, _>(
        |users, items| sigmoid(gmf_inner.score(users, items)),
        &train_ds,
        &val_interactions,
        &eval_device,
        k,
        max_users,
    );

    // ── NeuMF ─────────────────────────────────────────────────────────────────
    println!("\n[NeuMF] Training {} epochs...", args.epochs);
    let ncf_cfg = NeuMFConfig {
        num_users: dataset.num_users(),
        num_items: dataset.num_items(),
        gmf_dim: config.embedding_dim,
        mlp_layers: config.mlp_layers.clone(),
        mlp_embed_dim: config.embedding_dim,
    };
    let ncf_model = ncf_cfg.init::<B>(&device);
    let ncf_trainer = Trainer::<B>::new(config, device);
    let ncf_trained = ncf_trainer.train(ncf_model, &train_ds, &val_interactions);

    println!("\n[NeuMF] Evaluating (k={k}, max_users={max_users})...");
    let ncf_inner = ncf_trained.valid();
    let ncf_result = evaluate::<BInner, _, _>(
        |users, items| sigmoid(ncf_inner.score(users, items)),
        &train_ds,
        &val_interactions,
        &eval_device,
        k,
        max_users,
    );

    // ── Results table ─────────────────────────────────────────────────────────
    println!("\n╔══════════╦══════════╦══════════╗");
    println!("║  Model   ║  HR@{k:<3}  ║ NDCG@{k:<3} ║");
    println!("╠══════════╬══════════╬══════════╣");
    println!("║ GMF      ║ {:<8.4} ║ {:<8.4} ║", gmf_result.hr_at_k, gmf_result.ndcg_at_k);
    println!("║ NeuMF    ║ {:<8.4} ║ {:<8.4} ║", ncf_result.hr_at_k, ncf_result.ndcg_at_k);
    println!("╚══════════╩══════════╩══════════╝");

    Ok(())
}
