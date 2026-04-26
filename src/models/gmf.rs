use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor},
};

#[derive(Module, Debug)]
pub struct GMF<B: Backend> {
    user_embedding: Embedding<B>,
    item_embedding: Embedding<B>,
    output: Linear<B>,
}

#[derive(Debug, Clone)]
pub struct GMFConfig {
    pub num_users: usize,
    pub num_items: usize,
    pub embedding_dim: usize,
}

impl GMFConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GMF<B> {
        GMF {
            user_embedding: EmbeddingConfig::new(self.num_users, self.embedding_dim)
                .init(device),
            item_embedding: EmbeddingConfig::new(self.num_items, self.embedding_dim)
                .init(device),
            output: LinearConfig::new(self.embedding_dim, 1).with_bias(false).init(device),
        }
    }
}

impl<B: Backend> GMF<B> {
    /// Returns logits (pre-sigmoid), shape [batch].
    pub fn forward(
        &self,
        user_ids: Tensor<B, 1, Int>,
        item_ids: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let n = user_ids.dims()[0];
        // Embedding requires [batch, seq_len] input → reshape [n] to [n, 1]
        let u = self.user_embedding.forward(user_ids.reshape([n, 1]))  // [n, 1, dim]
            .reshape([n, self.user_embedding.weight.dims()[1]]);        // [n, dim]
        let i = self.item_embedding.forward(item_ids.reshape([n, 1]))
            .reshape([n, self.item_embedding.weight.dims()[1]]);
        let gmf = u * i;                                  // element-wise product
        self.output.forward(gmf).squeeze(1)               // [batch]
    }

    pub fn num_params(&self) -> usize {
        let d = |dims: [usize; 2]| dims[0] * dims[1];
        let bias = self.output.bias.as_ref().map(|b| b.dims()[0]).unwrap_or(0);
        d(self.user_embedding.weight.dims())
            + d(self.item_embedding.weight.dims())
            + d(self.output.weight.dims())
            + bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type MyBackend = NdArray<f32>;

    #[test]
    fn gmf_output_shape() {
        let device = Default::default();
        let model = GMFConfig { num_users: 100, num_items: 200, embedding_dim: 64 }
            .init::<MyBackend>(&device);

        let users = Tensor::<MyBackend, 1, Int>::from_ints([0, 1, 2], &device);
        let items = Tensor::<MyBackend, 1, Int>::from_ints([5, 10, 15], &device);
        let out = model.forward(users, items);

        assert_eq!(out.dims(), [3]);
    }
}
