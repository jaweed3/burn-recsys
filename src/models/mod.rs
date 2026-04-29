pub mod deepfm;
pub mod gmf;
pub mod ncf;

pub use deepfm::DeepFM;
pub use gmf::GMF;
pub use ncf::NeuMF;

use burn::tensor::{backend::Backend, Int, Tensor};

/// Implemented by every model in this crate.
///
/// Returns **logits** (pre-sigmoid) of shape `[batch]`.
/// Sigmoid is applied externally — by the loss function during training
/// and by the serving layer during inference.
pub trait Scorable<B: Backend> {
    fn score(
        &self, 
        users: Tensor<B, 1, Int>, 
        items: Tensor<B, 1, Int>
) -> Tensor<B, 1>;
}

pub trait Retrievable<B: Backend> {
    // taking embedding matrix for all items
    fn item_embeddings(&self) -> Vec<Vec<f32>>;
    fn user_embedding(&self, user_id: u32) -> Vec<f32>;
}

impl<B: Backend> Scorable<B> for GMF<B> {
    fn score(
        &self,
        users: Tensor<B, 1, Int>, 
        items: Tensor<B, 1, Int>
) -> Tensor<B, 1> {
        self.forward(users, items)
    }
}

impl<B: Backend> Scorable<B> for NeuMF<B> {
    fn score(
        &self,
        users: Tensor<B, 1, Int>, 
        items: Tensor<B, 1, Int>
) -> Tensor<B, 1> {
        self.forward(users, items)
    }
}

impl<B: Backend> Scorable<B> for DeepFM<B> {
    fn score(
        &self, 
        users: Tensor<B, 1, Int>, 
        items: Tensor<B, 1, Int>
) -> Tensor<B, 1> {
        self.forward(users, items)
    }
}

pub trait RecsysModel<B: Backend>: Scorable<B> + Retrievable<B> + Send {}

impl<B: Backend, T> RecsysModel<B> for T where T: Scorable<B> + Retrievable<B> + Send {}
