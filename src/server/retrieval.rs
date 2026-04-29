use instant_distance::{Builder, Hnsw, MapItem};
use serde::{Deserialize, Serialize};
use rand::{prelude::*, seq::index};
use std::collections::HashSet;

#[derive(Clone, Serialize, Deserialize)]
struct Point {
    vector: Vec<f32>,
    item_id: u32,
}

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        let dist: f32 = self.vector.iter()
            .zip(&other.vector)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        dist.sqrt()
    }
}

pub struct VectorRetriever {
    index: Hnsw<Point>,
}

impl VectorRetriever {
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        let points: Vec<Point> = embeddings.into_iter().enumerate()
            .map(|(i, v)| Point {vector: v, item_id: i as u32})
            .collect();

        let index = Builder::default().build(points, vec![(); points.len()]);

        Self { index }
    }
}

impl CandidateGenerator for VectorRetriever {
    fn generate(&self, user_id: u32, limit: usize, exclude: &HashSet<u32>) -> Vec<u32> {
        // TODO: take user embedding from model to search nearest relevant item.
        vec![]
    }
}

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
