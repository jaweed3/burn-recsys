use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Int, Tensor},
};

/// Neural Matrix Factorization (NeuMF) — combines GMF path and MLP path.
#[derive(Module, Debug)]
pub struct NeuMF<B: Backend> {
    // GMF path
    gmf_user_emb: Embedding<B>,
    gmf_item_emb: Embedding<B>,

    // MLP path
    mlp_user_emb: Embedding<B>,
    mlp_item_emb: Embedding<B>,
    mlp_layers: Vec<Linear<B>>,
    relu: Relu,

    // Output
    output: Linear<B>,
}

#[derive(Debug, Clone)]
pub struct NeuMFConfig {
    pub num_users: usize,
    pub num_items: usize,
    /// Embedding dim for the GMF path.
    pub gmf_dim: usize,
    /// MLP layer sizes. First element must equal 2 * mlp_embed_dim.
    /// e.g. [128, 64, 32, 16] with mlp_embed_dim=64.
    pub mlp_layers: Vec<usize>,
    /// Embedding dim for MLP path (concat gives mlp_layers[0]).
    pub mlp_embed_dim: usize,
}

impl NeuMFConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NeuMF<B> {
        assert_eq!(
            self.mlp_layers[0],
            self.mlp_embed_dim * 2,
            "mlp_layers[0] must equal 2 * mlp_embed_dim"
        );

        let mlp_linears: Vec<Linear<B>> = self.mlp_layers
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();

        let output_dim = self.gmf_dim + self.mlp_layers.last().copied().unwrap_or(16);

        NeuMF {
            gmf_user_emb: EmbeddingConfig::new(self.num_users, self.gmf_dim).init(device),
            gmf_item_emb: EmbeddingConfig::new(self.num_items, self.gmf_dim).init(device),
            mlp_user_emb: EmbeddingConfig::new(self.num_users, self.mlp_embed_dim).init(device),
            mlp_item_emb: EmbeddingConfig::new(self.num_items, self.mlp_embed_dim).init(device),
            mlp_layers: mlp_linears,
            relu: Relu::new(),
            output: LinearConfig::new(output_dim, 1).with_bias(true).init(device),
        }
    }
}

impl<B: Backend> NeuMF<B> {
    /// Returns logits (pre-sigmoid), shape [batch].
    pub fn forward(
        &self,
        user_ids: Tensor<B, 1, Int>,
        item_ids: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let n = user_ids.dims()[0];

        // Helper: embedding lookup [n] → [n, dim]
        let embed = |emb: &Embedding<B>, ids: Tensor<B, 1, Int>| {
            let dim = emb.weight.dims()[1];
            emb.forward(ids.reshape([n, 1])).reshape([n, dim])
        };

        // --- GMF path ---
        let gu = embed(&self.gmf_user_emb, user_ids.clone()); // [n, gmf_dim]
        let gi = embed(&self.gmf_item_emb, item_ids.clone()); // [n, gmf_dim]
        let gmf_out = gu * gi;                                // element-wise

        // --- MLP path ---
        let mu = embed(&self.mlp_user_emb, user_ids);         // [n, mlp_embed]
        let mi = embed(&self.mlp_item_emb, item_ids);         // [n, mlp_embed]
        let mut mlp_out = Tensor::cat(vec![mu, mi], 1);       // [n, 2*mlp_embed]
        for layer in &self.mlp_layers {
            mlp_out = self.relu.forward(layer.forward(mlp_out));
        }

        // --- Concat & output ---
        let combined = Tensor::cat(vec![gmf_out, mlp_out], 1); // [n, gmf_dim + mlp_last]
        self.output.forward(combined).squeeze(1)                // [n]
    }

    /// Count total trainable parameters.
    pub fn num_params(&self) -> usize {
        let d = |dims: [usize; 2]| dims[0] * dims[1];

        let emb = d(self.gmf_user_emb.weight.dims())
            + d(self.gmf_item_emb.weight.dims())
            + d(self.mlp_user_emb.weight.dims())
            + d(self.mlp_item_emb.weight.dims());

        let mlp: usize = self.mlp_layers.iter().map(|l| {
            let w = l.weight.dims();
            let bias = l.bias.as_ref().map(|b| b.dims()[0]).unwrap_or(0);
            w[0] * w[1] + bias
        }).sum();

        let out_w = self.output.weight.dims();
        let out_b = self.output.bias.as_ref().map(|b| b.dims()[0]).unwrap_or(0);
        let out = out_w[0] * out_w[1] + out_b;

        emb + mlp + out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::NdArray, tensor::activation::sigmoid};

    type B = NdArray<f32>;

    #[test]
    fn neumf_output_shape_and_range() {
        let device = Default::default();
        let config = NeuMFConfig {
            num_users: 100,
            num_items: 200,
            gmf_dim: 64,
            mlp_layers: vec![128, 64, 32, 16],
            mlp_embed_dim: 64,
        };
        let model = config.init::<B>(&device);

        let users = Tensor::<B, 1, Int>::from_ints([0, 1, 2], &device);
        let items = Tensor::<B, 1, Int>::from_ints([5, 10, 15], &device);
        let logits = model.forward(users, items);

        assert_eq!(logits.dims(), [3], "output shape must be [batch]");

        let probs = sigmoid(logits);
        let data = probs.to_data();
        for &v in data.as_slice::<f32>().unwrap() {
            assert!(v > 0.0 && v < 1.0, "sigmoid output {v} out of (0,1)");
        }
    }
}
