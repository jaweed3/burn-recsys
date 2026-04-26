use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io::Write as IoWrite,
    path::PathBuf,
    time::Instant,
};

use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{
        activation::sigmoid,
        backend::AutodiffBackend,
        ElementConversion, Int, Tensor,
    },
};
use rand::{seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::data::{NegativeSampler, RecsysDataset};
use crate::metrics::evaluate;
use crate::models::Scorable;

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub embedding_dim: usize,
    pub mlp_layers: Vec<usize>,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub neg_ratio: usize,
    /// Stop if validation HR@k doesn't improve for this many epochs.
    pub patience: usize,
    /// k for the validation metric used in early stopping.
    pub eval_k: usize,
    /// How many users to sample for per-epoch validation (fast approximation).
    pub val_samples: usize,
    /// Directory to save the single best checkpoint. None = no checkpointing.
    pub checkpoint_dir: Option<PathBuf>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            mlp_layers: vec![128, 64, 32, 16],
            learning_rate: 1e-3,
            batch_size: 256,
            num_epochs: 20,
            neg_ratio: 4,
            patience: 3,
            eval_k: 10,
            val_samples: 300,
            checkpoint_dir: Some(PathBuf::from("checkpoints")),
        }
    }
}

// ── Trainer ──────────────────────────────────────────────────────────────────

pub struct Trainer<B: AutodiffBackend> {
    pub config: TrainConfig,
    pub device: B::Device,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(config: TrainConfig, device: B::Device) -> Self {
        Self { config, device }
    }

    /// Train any model that implements `Scorable<B>`.
    ///
    /// - `train_ds`        — training interactions.
    /// - `val_interactions` — leave-one-out test pairs (one per user). Used for
    ///                        per-epoch HR@k tracking and early stopping.
    ///                        Pass an empty slice to disable validation.
    ///
    /// Returns the model state at the best validation HR@k seen during training.
    pub fn train<D, M>(&self, model: M, train_ds: &D, val_interactions: &[(u32, u32)]) -> M
    where
        D: RecsysDataset,
        M: Scorable<B> + AutodiffModule<B> + Clone,
        M::InnerModule: Scorable<B::InnerBackend>,
    {
        let use_validation = !val_interactions.is_empty();

        // Write config.toml to checkpoint dir before training starts
        if let Some(dir) = &self.config.checkpoint_dir {
            std::fs::create_dir_all(dir).ok();
            let config_path = dir.join("config.toml");
            if let Ok(s) = toml::to_string_pretty(&self.config) {
                std::fs::write(&config_path, s).ok();
            }
        }

        let interactions = train_ds.interactions();
        let sampler = NegativeSampler::new(interactions, train_ds.num_items(), self.config.neg_ratio);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let mut optimizer = AdamConfig::new().init::<B, M>();
        let mut model = model;

        // Best state tracking
        let mut best_hr: f32 = -1.0;
        let mut best_ndcg: f32 = 0.0;
        let mut best_model = model.clone();
        let mut best_epoch = 0usize;
        let mut no_improve = 0usize;

        // Pre-allocate the pairs buffer — refill each epoch, never reallocate
        let capacity = interactions.len() * (1 + self.config.neg_ratio);
        let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(capacity);

        let inner_device = self.device.clone();

        for epoch in 1..=self.config.num_epochs {
            let t0 = Instant::now();

            // ── Build training pairs ─────────────────────────────────────────
            pairs.clear();
            for &(uid, iid) in interactions {
                pairs.push((uid, iid, 1.0));
                for neg in sampler.sample(uid, &mut rng) {
                    pairs.push((uid, neg, 0.0));
                }
            }
            pairs.shuffle(&mut rng);

            // ── Gradient steps ───────────────────────────────────────────────
            let mut epoch_loss = 0.0f32;
            let mut num_batches = 0usize;

            for chunk in pairs.chunks(self.config.batch_size) {
                let users:  Vec<i32> = chunk.iter().map(|&(u,_,_)| u as i32).collect();
                let items:  Vec<i32> = chunk.iter().map(|&(_,i,_)| i as i32).collect();
                let labels: Vec<f32> = chunk.iter().map(|&(_,_,l)| l).collect();
                let n = users.len();

                let user_t  = Tensor::<B, 1, Int>::from_ints(users.as_slice(),  &self.device);
                let item_t  = Tensor::<B, 1, Int>::from_ints(items.as_slice(),  &self.device);
                let label_t = Tensor::<B, 1>::from_floats(labels.as_slice(), &self.device);

                let logits = model.score(user_t, item_t);
                let loss   = bce_loss(logits, label_t, n);

                epoch_loss += loss.clone().into_scalar().elem::<f32>();
                num_batches += 1;

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(self.config.learning_rate, model, grads);
            }

            let avg_loss = epoch_loss / num_batches.max(1) as f32;
            let train_secs = t0.elapsed().as_secs_f32();

            // ── Validation ──────────────────────────────────────────────────
            let (val_hr, val_ndcg) = if use_validation {
                let valid_model = model.valid();
                let result = evaluate::<B::InnerBackend, _, _>(
                    |u, i| sigmoid(valid_model.score(u, i)),
                    train_ds,
                    val_interactions,
                    &inner_device,
                    self.config.eval_k,
                    self.config.val_samples,
                );
                (result.hr_at_k, result.ndcg_at_k)
            } else {
                (-1.0, -1.0)
            };

            let total_secs = t0.elapsed().as_secs_f32();

            if use_validation {
                println!(
                    "Epoch {epoch:>3}/{} | loss={avg_loss:.6} | \
                     HR@{}={val_hr:.4} NDCG@{}={val_ndcg:.4} | {total_secs:.1}s",
                    self.config.num_epochs, self.config.eval_k, self.config.eval_k
                );
            } else {
                println!(
                    "Epoch {epoch:>3}/{} | loss={avg_loss:.6} | {train_secs:.1}s",
                    self.config.num_epochs
                );
            }

            // ── Best checkpoint ──────────────────────────────────────────────
            let improved = if use_validation {
                val_hr > best_hr + 1e-5
            } else {
                avg_loss < -best_hr  // reuse sentinel; improve = loss goes down
            };

            if improved || epoch == 1 {
                best_hr    = val_hr;
                best_ndcg  = val_ndcg;
                best_epoch = epoch;
                best_model = model.clone();
                no_improve = 0;

                // Save best checkpoint (overwrites previous best)
                if let Some(dir) = &self.config.checkpoint_dir {
                    let path = dir.join("best");
                    if let Err(e) = model.clone().save_file(path, &CompactRecorder::new()) {
                        log::warn!("Best checkpoint save failed: {e}");
                    }
                }
            } else {
                no_improve += 1;
                if no_improve >= self.config.patience {
                    println!(
                        "Early stopping at epoch {epoch} \
                         (best epoch={best_epoch}, HR@{}={best_hr:.4})",
                        self.config.eval_k
                    );
                    break;
                }
            }
        }

        // ── Experiment log ───────────────────────────────────────────────────
        log_experiment(&self.config, best_epoch, best_hr, best_ndcg);

        println!(
            "\nBest epoch: {best_epoch} | HR@{}={best_hr:.4} | NDCG@{}={best_ndcg:.4}",
            self.config.eval_k, self.config.eval_k
        );

        best_model
    }
}

// ── Loss ──────────────────────────────────────────────────────────────────────

/// Binary cross-entropy loss on logits.
fn bce_loss<B: AutodiffBackend>(
    logits: Tensor<B, 1>,
    labels: Tensor<B, 1>,
    n: usize,
) -> Tensor<B, 1> {
    let eps = 1e-7_f32;
    let probs = sigmoid(logits).clamp(eps, 1.0 - eps);
    let ones = Tensor::<B, 1>::ones([n], &probs.device());
    ((labels.clone() * probs.clone().log())
        + ((ones.clone() - labels) * (ones - probs).log()))
        .mean()
        .neg()
}

// ── Experiment log ────────────────────────────────────────────────────────────

fn log_experiment(config: &TrainConfig, best_epoch: usize, best_hr: f32, best_ndcg: f32) {
    let path = PathBuf::from("experiments.csv");
    let write_header = !path.exists();

    let config_hash = {
        let mut h = DefaultHasher::new();
        format!("{:?}", config).hash(&mut h);
        h.finish()
    };

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|_| return std::fs::File::create(&path).unwrap());

    if write_header {
        writeln!(
            file,
            "timestamp,config_hash,embedding_dim,mlp_layers,lr,\
             batch_size,epochs,neg_ratio,best_epoch,best_hr10,best_ndcg10"
        ).ok();
    }

    writeln!(
        file,
        "{timestamp},{config_hash:#x},{},{:?},{},{},{},{},{best_epoch},{best_hr:.4},{best_ndcg:.4}",
        config.embedding_dim,
        config.mlp_layers,
        config.learning_rate,
        config.batch_size,
        config.num_epochs,
        config.neg_ratio,
    ).ok();
}
