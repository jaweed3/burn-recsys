/// HR@k: 1.0 if `ground_truth` appears in the top-k of `ranked`, else 0.0.
pub fn hit_rate_at_k(ranked: &[u32], ground_truth: u32, k: usize) -> f32 {
    ranked.iter().take(k).any(|&id| id == ground_truth) as u8 as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_at_k_found() {
        let ranked = vec![3, 7, 1, 5, 9];
        assert_eq!(hit_rate_at_k(&ranked, 1, 3), 1.0);
    }

    #[test]
    fn hit_at_k_not_found() {
        let ranked = vec![3, 7, 1, 5, 9];
        assert_eq!(hit_rate_at_k(&ranked, 99, 3), 0.0);
    }

    #[test]
    fn hit_at_k_boundary() {
        let ranked = vec![3, 7, 1, 5, 9];
        // item 1 is at position 2 (0-indexed), so within top-3
        assert_eq!(hit_rate_at_k(&ranked, 1, 3), 1.0);
        // but not within top-2
        assert_eq!(hit_rate_at_k(&ranked, 1, 2), 0.0);
    }
}
