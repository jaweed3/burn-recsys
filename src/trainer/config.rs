use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct TrainerSettings {
    pub data_path: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub embedding_dim: usize,
    pub neg_ratio: usize,
    pub checkpoint_dir: Option<String>,
}

impl Default for TrainerSettings {
    fn default() -> Self {
        Self {
            data_path: "data/myket.csv".to_string(),
            epochs: 20,
            batch_size: 128,
            learning_rate: 1e-3,
            embedding_dim: 64,
            neg_ratio: 4,
            checkpoint_dir: Some("checkpoints".to_string()),
        }
    }
}

impl TrainerSettings {
    pub fn mlp_layers(&self) -> Vec<usize> {
        // Standard NCF tower
        vec![
            self.embedding_dim * 2,
            self.embedding_dim,
            self.embedding_dim / 2,
            self.embedding_dim / 4,
        ]
    }
}
