/// Core trait every dataset must implement.
/// Keeps the model and training loop dataset-agnostic.
pub trait RecsysDataset {
    fn num_users(&self) -> usize;
    fn num_items(&self) -> usize;

    /// All (user_id, item_id) positive interactions, 0-based contiguous indices.
    fn interactions(&self) -> &[(u32, u32)];

    /// Split into (train, test) by `ratio` fraction going to train.
    fn train_test_split(&self, ratio: f32) -> (Self, Self)
    where
        Self: Sized;
}
