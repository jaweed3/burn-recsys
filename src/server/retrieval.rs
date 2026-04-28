use rand::prelude::*;
use std::collections::HashSet;

pub trait CandidateGenerator: Send + Sync {
    fn generate(&self, user_id: u32, limit: usize, exclude: &HashSet<u32>) -> Vec<u32>;
}

pub struct SimpleRetriever {
    num_items: usize,
}

impl SimpleRetriever {
    pub fn new(num_items: usize) -> Self {
        Self { num_items }
    }
}

impl CandidateGenerator for SimpleRetriever {
    fn generate(&self, _user_id: u32, limit: usize, exclude: &HashSet<u32>) -> Vec<u32> {
        let mut rng = thread_rng();
        let mut candidates = HashSet::with_capacity(limit);
        
        // Simple random retrieval, avoiding already interacted items
        // In a real system, this would use an ANN index (like Faiss/Hnsw)
        let mut attempts = 0;
        while candidates.len() < limit && attempts < limit * 2 {
            let item_id = rng.gen_range(0..self.num_items) as u32;
            if !exclude.contains(&item_id) {
                candidates.insert(item_id);
            }
            attempts += 1;
        }

        candidates.into_iter().collect()
    }
}
