/// NDCG@k: normalized discounted cumulative gain assuming binary relevance.
/// Since there's exactly one relevant item, ideal DCG = 1.0, so NDCG = DCG.
pub fn ndcg_at_k(ranked: &[u32], ground_truth: u32, k: usize) -> f32 {
    for (rank, &id) in ranked.iter().take(k).enumerate() {
        if id == ground_truth {
            // rank is 0-indexed, so denominator is log2(rank + 2)
            return 1.0 / (rank as f32 + 2.0).log2();
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_top_position() {
        let ranked = vec![5, 3, 1, 9];
        // hit at rank 0 → 1/log2(2) = 1.0
        let v = ndcg_at_k(&ranked, 5, 10);
        assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
    }

    #[test]
    fn ndcg_second_position() {
        let ranked = vec![3, 5, 1, 9];
        // hit at rank 1 → 1/log2(3) ≈ 0.631
        let v = ndcg_at_k(&ranked, 5, 10);
        let expected = 1.0_f32 / 3.0_f32.log2();
        assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
    }

    #[test]
    fn ndcg_not_in_top_k() {
        let ranked = vec![3, 5, 1, 9, 2];
        assert_eq!(ndcg_at_k(&ranked, 5, 1), 0.0);
    }
}
