/// DeepFM — Guo et al. 2017 (https://arxiv.org/abs/1703.04247)
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Int, Tensor},
};
use crate::models::Retrievable;

#[derive(Module, Debug)]
pub struct DeepFM<B: Backend> {
    // First-order bias terms (one per field)
    user_bias: Embedding<B>,
    item_bias: Embedding<B>,

    // Second-order embedding vectors (shared with Deep path)
    user_emb: Embedding<B>,
    item_emb: Embedding<B>,

    // Deep MLP layers
    deep_layers: Vec<Linear<B>>,
    relu: Relu,

    // Final projection for Deep path
    deep_out: Linear<B>,
}

#[derive(Debug, Clone)]
pub struct DeepFMConfig {
    pub num_users: usize,
    pub num_items: usize,
    /// Embedding dim k — shared for FM second-order vectors and Deep input.
    pub embedding_dim: usize,
    /// MLP hidden layer sizes. Input dim = 2 * embedding_dim.
    pub mlp_layers: Vec<usize>,
}

impl DeepFMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeepFM<B> {
        assert!(!self.mlp_layers.is_empty(), "mlp_layers must not be empty");

        let deep_linears: Vec<Linear<B>> = {
            let mut sizes = vec![self.embedding_dim * 2];
            sizes.extend_from_slice(&self.mlp_layers);
            sizes.windows(2)
                .map(|w| LinearConfig::new(w[0], w[1]).init(device))
                .collect()
        };

        let last_hidden = *self.mlp_layers.last().unwrap();

        DeepFM {
            user_bias: EmbeddingConfig::new(self.num_users, 1).init(device),
            item_bias: EmbeddingConfig::new(self.num_items, 1).init(device),
            user_emb:  EmbeddingConfig::new(self.num_users, self.embedding_dim).init(device),
            item_emb:  EmbeddingConfig::new(self.num_items, self.embedding_dim).init(device),
            deep_layers: deep_linears,
            relu: Relu::new(),
            deep_out: LinearConfig::new(last_hidden, 1).with_bias(true).init(device),
        }
    }
}

impl<B: Backend> DeepFM<B> {
    /// Returns logits (pre-sigmoid), shape [batch].
    pub fn forward(
        &self,
        user_ids: Tensor<B, 1, Int>,
        item_ids: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let n = user_ids.dims()[0];

        // ---- embeddings ----
        let embed = |emb: &Embedding<B>, ids: Tensor<B, 1, Int>, dim: usize| {
            emb.forward(ids.reshape([n, 1])).reshape([n, dim])
        };
        let k = self.user_emb.weight.dims()[1];
        let vu = embed(&self.user_emb, user_ids.clone(), k); // [n, k]
        let vi = embed(&self.item_emb, item_ids.clone(), k); // [n, k]

        // ---- FM first-order ----
        let bu = self.user_bias.forward(user_ids.clone().reshape([n, 1]))
            .reshape([n]);                                   // [n]
        let bi = self.item_bias.forward(item_ids.clone().reshape([n, 1]))
            .reshape([n]);                                   // [n]
        let fm_first = bu + bi;                              // [n]

        // ---- FM second-order (two-field case: only user × item cross-term) ----
        // For two fields: 0.5 * [(v_u + v_i)² - v_u² - v_i²] = v_u ⊙ v_i, summed over k
        let fm_second = (vu.clone() * vi.clone()).sum_dim(1).squeeze(1); // [n]

        let fm_out = fm_first + fm_second;                   // [n]

        // ---- Deep path ----
        let mut h = Tensor::cat(vec![vu, vi], 1);            // [n, 2k]
        for layer in &self.deep_layers {
            h = self.relu.forward(layer.forward(h));
        }
        let deep_out = self.deep_out.forward(h).squeeze(1);  // [n]

        // ---- Combine ----
        fm_out + deep_out                                    // [n] logits
    }

    pub fn num_params(&self) -> usize {
        let emb = self.user_bias.weight.dims()[0]            // user bias table
            + self.item_bias.weight.dims()[0]                // item bias table
            + self.user_emb.weight.dims()[0] * self.user_emb.weight.dims()[1]
            + self.item_emb.weight.dims()[0] * self.item_emb.weight.dims()[1];

        let mlp: usize = self.deep_layers.iter().map(|l| {
            let w = l.weight.dims();
            let b = l.bias.as_ref().map(|b| b.dims()[0]).unwrap_or(0);
            w[0] * w[1] + b
        }).sum();

        let out_w = self.deep_out.weight.dims();
        let out_b = self.deep_out.bias.as_ref().map(|b| b.dims()[0]).unwrap_or(0);

        emb + mlp + out_w[0] * out_w[1] + out_b
    }
}

impl<B: Backend> Retrievable<B> for DeepFM<B> {
    fn item_embeddings(&self) -> Vec<Vec<f32>> {
        let tensor = self.item_emb.weight.val();
        let shape = tensor.shape();
        let dim = shape.dims[1];
        let flat_data: Vec<f32> = tensor.into_data().to_vec().unwrap();
        flat_data
            .chunks_exact(dim)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    fn user_embedding(&self, user_id: u32) -> Vec<f32> {
        let weights = self.user_emb.weight.val();
        let start = user_id as usize;
        let user_tensor = weights.slice([start..start + 1]);
        user_tensor
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to export user embedding")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::NdArray, tensor::activation::sigmoid};

    type B = NdArray<f32>;

    #[test]
    fn deepfm_output_shape_and_range() {
        let device = Default::default();
        let model = DeepFMConfig {
            num_users: 50,
            num_items: 100,
            embedding_dim: 16,
            mlp_layers: vec![64, 32],
        }.init::<B>(&device);

        let users = Tensor::<B, 1, Int>::from_ints([0, 1, 2], &device);
        let items = Tensor::<B, 1, Int>::from_ints([5, 10, 15], &device);
        let logits = model.forward(users, items);

        assert_eq!(logits.dims(), [3]);

        let probs = sigmoid(logits);
        let vals = probs.into_data().to_vec::<f32>().unwrap();
        for v in vals {
            assert!(v > 0.0 && v < 1.0, "sigmoid output {v} out of (0,1)");
        }
    }

    #[test]
    fn deepfm_param_count_reasonable() {
        let device = Default::default();
        let model = DeepFMConfig {
            num_users: 10_000,
            num_items: 7_988,
            embedding_dim: 64,
            mlp_layers: vec![128, 64],
        }.init::<B>(&device);
        let p = model.num_params();
        assert!(p > 100_000, "too few params: {p}");
        assert!(p < 5_000_000, "too many params: {p}");
        println!("DeepFM params: {p}");
    }
}
