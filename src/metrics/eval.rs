use burn::tensor::{backend::Backend, Int, Tensor};
use rand::SeedableRng;

use crate::data::{NegativeSampler, RecsysDataset};
use crate::metrics::{hit_rate_at_k, ndcg_at_k};

/// Leave-one-out evaluation result.
pub struct EvalResult {
    pub hr_at_k: f32,
    pub ndcg_at_k: f32,
    pub k: usize,
    pub num_users: usize,
}

impl std::fmt::Display for EvalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HR@{k}={hr:.4}  NDCG@{k}={ndcg:.4}  (n={n} users)",
            k = self.k,
            hr = self.hr_at_k,
            ndcg = self.ndcg_at_k,
            n = self.num_users,
        )
    }
}

/// Run leave-one-out evaluation for any model that exposes a `forward` compatible closure.
///
/// Protocol:
/// - For each user in the test set, take their last interaction as ground truth.
/// - Sample 99 random negatives (not in the user's known positives).
/// - Score all 100 candidates, rank descending, check if ground truth is in top-k.
pub fn evaluate<B, D, F>(
    score_fn: F,
    train_ds: &D,
    test_interactions: &[(u32, u32)],
    device: &B::Device,
    k: usize,
    max_users: usize,
) -> EvalResult
where
    B: Backend,
    D: RecsysDataset,
    F: Fn(Tensor<B, 1, Int>, Tensor<B, 1, Int>) -> Tensor<B, 1>,
{
    let sampler = NegativeSampler::new(train_ds.interactions(), train_ds.num_items(), 4);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(2025);

    let mut hr_sum = 0.0f32;
    let mut ndcg_sum = 0.0f32;
    let mut count = 0usize;

    for &(user_id, gt_item) in test_interactions.iter().take(max_users) {
        // 99 negatives + 1 ground truth = 100 candidates
        let mut negatives = sampler.sample_eval_negatives(user_id, &mut rng);
        negatives.push(gt_item);
        let n = negatives.len(); // 100

        let users: Vec<i32> = vec![user_id as i32; n];
        let items: Vec<i32> = negatives.iter().map(|&x| x as i32).collect();

        let user_t = Tensor::<B, 1, Int>::from_ints(users.as_slice(), device);
        let item_t = Tensor::<B, 1, Int>::from_ints(items.as_slice(), device);

        let scores = score_fn(user_t, item_t);
        let scores_vec: Vec<f32> = scores
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();

        // Rank candidates by score descending
        let mut idx_scores: Vec<(usize, f32)> = scores_vec.into_iter().enumerate().collect();
        idx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let ranked: Vec<u32> = idx_scores.iter().map(|&(i, _)| negatives[i]).collect();

        hr_sum   += hit_rate_at_k(&ranked, gt_item, k);
        ndcg_sum += ndcg_at_k(&ranked, gt_item, k);
        count    += 1;
    }

    EvalResult {
        hr_at_k:   hr_sum   / count as f32,
        ndcg_at_k: ndcg_sum / count as f32,
        k,
        num_users: count,
    }
}
