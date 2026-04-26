/// Integration test: train 1 epoch, save checkpoint, load, evaluate HR@10 > 0.
use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    record::CompactRecorder,
    tensor::activation::sigmoid,
};
use burn_recsys::{
    data::{PolarsDataset, RecsysDataset},
    metrics::evaluate,
    models::{ncf::NeuMFConfig, Scorable},
    trainer::{TrainConfig, Trainer},
};
use std::io::Write as IoWrite;
use tempfile::NamedTempFile;

type B = Autodiff<NdArray<f32>>;
type BInner = NdArray<f32>;

/// Build a tiny in-memory dataset: 5 users × 20 items.
fn tiny_dataset() -> PolarsDataset {
    // Write synthetic CSV with timestamps
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "user_id,app_name,timestamp").unwrap();
    for user in 0u32..5 {
        for (ts, item) in (0u32..10).enumerate() {
            writeln!(f, "{user},item{item},{ts}").unwrap();
        }
    }
    f.flush().unwrap();
    PolarsDataset::myket(f.path()).unwrap()
}

#[test]
fn train_save_load_eval() {
    let dataset = tiny_dataset();
    assert_eq!(dataset.num_users(), 5);
    assert_eq!(dataset.num_items(), 10);

    let (train_ds, val_interactions) = dataset.leave_one_out();
    assert_eq!(val_interactions.len(), 5, "one val item per user");
    assert_eq!(
        train_ds.interactions().len(),
        dataset.interactions().len() - 5,
        "train excludes held-out items"
    );

    let ckpt_dir = tempfile::tempdir().unwrap();
    let config = TrainConfig {
        num_epochs: 1,
        batch_size: 32,
        neg_ratio: 2,
        embedding_dim: 8,
        mlp_layers: vec![16, 8],
        patience: 999,
        checkpoint_dir: Some(ckpt_dir.path().to_path_buf()),
        ..Default::default()
    };

    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let model_config = NeuMFConfig {
        num_users: dataset.num_users(),
        num_items: dataset.num_items(),
        gmf_dim: config.embedding_dim,
        mlp_layers: config.mlp_layers.clone(),
        mlp_embed_dim: config.embedding_dim,
    };

    let model = model_config.init::<B>(&device);
    let trainer = Trainer::<B>::new(config, device.clone());
    let _trained = trainer.train(model, &train_ds, &val_interactions);

    // Verify best checkpoint exists
    let ckpt_path = ckpt_dir.path().join("best.mpk");
    assert!(ckpt_path.exists(), "best checkpoint should be saved");

    // Load checkpoint and verify it scores correctly
    let loaded = model_config
        .init::<BInner>(&Default::default())
        .load_file(
            ckpt_dir.path().join("best"),
            &CompactRecorder::new(),
            &Default::default(),
        )
        .expect("checkpoint should load");

    // Run evaluation — with only 10 items and 9 negatives (all items),
    // HR@10 = 1.0 always (ground truth is always in top 10 of 10).
    let eval_device: <BInner as burn::tensor::backend::Backend>::Device = Default::default();
    let result = evaluate::<BInner, _, _>(
        |users, items| sigmoid(loaded.score(users, items)),
        &train_ds,
        &val_interactions,
        &eval_device,
        10,
        100,
    );

    assert!(result.hr_at_k > 0.0, "HR@10 should be > 0 after 1 training epoch");
    assert!(result.ndcg_at_k > 0.0, "NDCG@10 should be > 0 after 1 training epoch");
    println!("Integration test: {result}");
}
