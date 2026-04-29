use instant_distance::{Builder, HnswMap, Search};
use rand::prelude::*;
use std::collections::HashSet;

pub trait CandidateGenerator: Send + Sync {
    fn generate(&self, user_id: u32, user_vector: Option<Vec<f32>>, limit: usize, exclude: &HashSet<u32>) -> Vec<u32>;
}

// Point in the ANN space
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Point {
    pub vector: Vec<f32>,
    pub item_id: u32,
}

// Tell instant-distance how to compute distance between Points
impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.vector.iter()
            .zip(&other.vector)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

pub struct VectorRetriever {
    index: HnswMap<Point, ()>,
    num_items: usize,
}

impl VectorRetriever {
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        let num_items = embeddings.len();
        let points: Vec<Point> = embeddings.into_iter().enumerate()
            .map(|(i, v)| Point { vector: v, item_id: i as u32 })
            .collect();

        let values = vec![(); points.len()];
        let index = Builder::default().build(points, values);
        
        Self { index, num_items }
    }
}

impl CandidateGenerator for VectorRetriever {
    fn generate(&self, _user_id: u32, user_vector: Option<Vec<f32>>, limit: usize, exclude: &HashSet<u32>) -> Vec<u32> {
        if let Some(vec) = user_vector {
            // Approximate nearest neighbor search
            let query = Point { vector: vec, item_id: 0 };
            let mut search = Search::default();

            self.index.search(&query, &mut search)
                .map(|item| item.point.item_id)
                .filter(|id| !exclude.contains(id))
                .take(limit)
                .collect()
        } else {
            // Fallback to random if no vector available
            let mut rng = thread_rng();
            let mut candidates = HashSet::with_capacity(limit);
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
}
