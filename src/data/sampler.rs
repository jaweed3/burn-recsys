use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Samples negative items for each user that are not in their observed positives.
pub struct NegativeSampler {
    num_items: usize,
    user_positives: HashMap<u32, HashSet<u32>>,
    /// How many negatives to draw per positive interaction.
    pub ratio: usize,
}

impl NegativeSampler {
    pub fn new(interactions: &[(u32, u32)], num_items: usize, ratio: usize) -> Self {
        let mut user_positives: HashMap<u32, HashSet<u32>> = HashMap::new();
        for &(user_id, item_id) in interactions {
            user_positives.entry(user_id).or_default().insert(item_id);
        }
        Self { num_items, user_positives, ratio }
    }

    /// Returns `ratio` negative item IDs for `user_id`, guaranteed not in their positive set.
    pub fn sample(&self, user_id: u32, rng: &mut impl Rng) -> Vec<u32> {
        let positives = self.user_positives.get(&user_id);
        let mut negatives = Vec::with_capacity(self.ratio);
        while negatives.len() < self.ratio {
            let candidate = rng.gen_range(0..self.num_items) as u32;
            let is_positive = positives.map_or(false, |s| s.contains(&candidate));
            if !is_positive {
                negatives.push(candidate);
            }
        }
        negatives
    }

    /// Returns 99 random negatives for leave-one-out evaluation (no overlap with positives).
    pub fn sample_eval_negatives(&self, user_id: u32, rng: &mut impl Rng) -> Vec<u32> {
        let positives = self.user_positives.get(&user_id);
        let mut negatives = Vec::with_capacity(99);
        while negatives.len() < 99 {
            let candidate = rng.gen_range(0..self.num_items) as u32;
            let is_positive = positives.map_or(false, |s| s.contains(&candidate));
            if !is_positive {
                negatives.push(candidate);
            }
        }
        negatives
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn negatives_dont_overlap_positives() {
        let interactions = vec![(0u32, 0u32), (0, 1), (0, 2), (1, 5)];
        let sampler = NegativeSampler::new(&interactions, 100, 4);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let negs = sampler.sample(0, &mut rng);
        assert_eq!(negs.len(), 4);
        for n in &negs {
            assert!(*n != 0 && *n != 1 && *n != 2, "negative {n} is a known positive");
        }
    }
}
