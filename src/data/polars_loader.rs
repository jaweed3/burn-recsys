use std::collections::{HashMap, HashSet};
use std::path::Path;
use anyhow::{Context, Result};
use polars::prelude::*;

use super::dataset::RecsysDataset;

/// The single dataset loader for this project.
///
/// Backed by Polars lazy evaluation: fast, deduplicating, and optionally
/// timestamp-sorted for correct temporal train/test splits.
pub struct PolarsDataset {
    /// All interactions as (user_idx, item_idx), in loading order.
    /// If ts_col was provided at construction, interactions are sorted
    /// by (user, timestamp) — enabling leave_one_out() to work correctly.
    pub interactions: Vec<(u32, u32)>,
    pub num_users: usize,
    pub num_items: usize,
    /// Map from original user string → contiguous u32 index.
    pub user_index: HashMap<String, u32>,
    /// Map from original item string → contiguous u32 index.
    pub item_index: HashMap<String, u32>,
    /// Last item index per user (by timestamp if available, else by row order).
    /// Used by leave_one_out().
    user_last_item: HashMap<u32, u32>,
}

impl PolarsDataset {
    /// Load from CSV.
    ///
    /// - `user_col`: column name for user IDs (any type — cast to string internally).
    /// - `item_col`: column name for item IDs.
    /// - `ts_col`:   optional timestamp column. If Some, interactions are sorted
    ///               per user by timestamp ascending, making leave_one_out() temporally correct.
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        user_col: &str,
        item_col: &str,
        ts_col: Option<&str>,
    ) -> Result<Self> {
        let path = path.as_ref();

        // Build column selection
        let mut select_exprs = vec![
            col(user_col).cast(DataType::String).alias("_user"),
            col(item_col).cast(DataType::String).alias("_item"),
        ];
        if let Some(ts) = ts_col {
            select_exprs.push(col(ts).cast(DataType::Float64).alias("_ts"));
        }

        let mut lf = LazyCsvReader::new(path)
            .with_has_header(true)
            .finish()
            .with_context(|| format!("Cannot scan {}", path.display()))?
            .select(select_exprs)
            .drop_nulls(None)
            .unique(
                Some(vec!["_user".into(), "_item".into()]),
                UniqueKeepStrategy::First,
            );

        // Sort by user then timestamp so that, within each user, interactions
        // are in chronological order. The last interaction per user is their most
        // recent one — what leave_one_out() holds out as ground truth.
        if ts_col.is_some() {
            lf = lf.sort_by_exprs(
                [col("_user"), col("_ts")],
                SortMultipleOptions::default().with_order_descending_multi([false, false]),
            );
        }

        let df = lf.collect().context("Polars collect failed")?;

        let user_ca = df.column("_user")?.str().context("_user not string")?;
        let item_ca = df.column("_item")?.str().context("_item not string")?;

        let mut user_index: HashMap<String, u32> = HashMap::new();
        let mut item_index: HashMap<String, u32> = HashMap::new();
        let mut interactions: Vec<(u32, u32)> = Vec::with_capacity(df.height());

        for (u_opt, i_opt) in user_ca.into_iter().zip(item_ca.into_iter()) {
            let u_str = u_opt.context("null user")?;
            let i_str = i_opt.context("null item")?;

            let next_u = user_index.len() as u32;
            let uid = *user_index.entry(u_str.to_string()).or_insert(next_u);
            let next_i = item_index.len() as u32;
            let iid = *item_index.entry(i_str.to_string()).or_insert(next_i);

            interactions.push((uid, iid));
        }

        let num_users = user_index.len();
        let num_items = item_index.len();

        // Track last item per user. Because interactions are sorted by (user, ts),
        // a simple linear scan picks up the last occurrence per user.
        let mut user_last_item: HashMap<u32, u32> = HashMap::with_capacity(num_users);
        for &(uid, iid) in &interactions {
            user_last_item.insert(uid, iid); // overwrites — last write wins
        }

        log::info!(
            "PolarsDataset: {} users, {} items, {} interactions (ts_sorted={})",
            num_users, num_items, interactions.len(), ts_col.is_some()
        );

        Ok(Self { interactions, num_users, num_items, user_index, item_index, user_last_item })
    }

    /// Convenience: load Myket CSV with temporal sort.
    /// Columns: user_id, app_name, timestamp.
    pub fn myket<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_csv(path, "user_id", "app_name", Some("timestamp"))
    }

    /// Convenience: load MovieLens 1M CSV.
    /// Columns: user_id, app_id. No timestamp in the exported CSV,
    /// so row order is used (original MovieLens data is implicitly sorted).
    pub fn movielens<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_csv(path, "user_id", "app_id", None)
    }

    /// Temporal leave-one-out split.
    ///
    /// Returns `(train_dataset, val_interactions)` where:
    /// - `val_interactions` contains exactly one `(user_id, item_id)` per user
    ///   — their **last** interaction by timestamp (or row order if no timestamp).
    /// - `train_dataset` is a PolarsDataset containing all other interactions,
    ///   with the same vocabulary (num_users, num_items, indexes) as `self`.
    ///
    /// Users with only one interaction appear in val only; train has no interactions
    /// for them (their embedding will remain untrained / random).
    pub fn leave_one_out(&self) -> (PolarsDataset, Vec<(u32, u32)>) {
        // val: last item per user
        let val_set: HashSet<(u32, u32)> = self.user_last_item
            .iter()
            .map(|(&uid, &iid)| (uid, iid))
            .collect();

        let mut val: Vec<(u32, u32)> = val_set.iter().copied().collect();
        val.sort_unstable(); // deterministic order

        // train: everything except the val interactions.
        // Iterate backwards to remove only the LAST occurrence per user
        // (handles duplicate interactions correctly).
        let mut removed: HashSet<u32> = HashSet::new();
        let mut rev_train: Vec<(u32, u32)> = Vec::with_capacity(self.interactions.len());
        for &(uid, iid) in self.interactions.iter().rev() {
            let gt = self.user_last_item[&uid];
            if iid == gt && !removed.contains(&uid) {
                removed.insert(uid); // skip — this is the held-out item
            } else {
                rev_train.push((uid, iid));
            }
        }
        let train_interactions: Vec<(u32, u32)> = rev_train.into_iter().rev().collect();

        // Rebuild user_last_item for the train slice.
        let mut train_last: HashMap<u32, u32> = HashMap::with_capacity(self.num_users);
        for &(uid, iid) in &train_interactions {
            train_last.insert(uid, iid);
        }

        let train_ds = PolarsDataset::from_parts(
            train_interactions,
            self.num_users,
            self.num_items,
            self.user_index.clone(),
            self.item_index.clone(),
            train_last,
        );

        (train_ds, val)
    }

    fn from_parts(
        interactions: Vec<(u32, u32)>,
        num_users: usize,
        num_items: usize,
        user_index: HashMap<String, u32>,
        item_index: HashMap<String, u32>,
        user_last_item: HashMap<u32, u32>,
    ) -> Self {
        Self { interactions, num_users, num_items, user_index, item_index, user_last_item }
    }
}

impl RecsysDataset for PolarsDataset {
    fn num_users(&self) -> usize { self.num_users }
    fn num_items(&self) -> usize { self.num_items }
    fn interactions(&self) -> &[(u32, u32)] { &self.interactions }

    fn train_test_split(&self, ratio: f32) -> (Self, Self) {
        // Kept for compatibility. Prefer leave_one_out() for eval correctness.
        let split = (self.interactions.len() as f32 * ratio) as usize;
        let train = self.interactions[..split].to_vec();
        let test  = self.interactions[split..].to_vec();

        let mut last_train: HashMap<u32, u32> = HashMap::new();
        for &(u, i) in &train { last_train.insert(u, i); }
        let mut last_test: HashMap<u32, u32> = HashMap::new();
        for &(u, i) in &test { last_test.insert(u, i); }

        (
            Self::from_parts(train, self.num_users, self.num_items,
                             self.user_index.clone(), self.item_index.clone(), last_train),
            Self::from_parts(test, self.num_users, self.num_items,
                             self.user_index.clone(), self.item_index.clone(), last_test),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{}", content).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn dedup_and_count() {
        let f = write_csv(
            "user_id,app_name,timestamp\n\
             1,com.foo,1.0\n\
             2,com.bar,2.0\n\
             1,com.baz,3.0\n\
             1,com.foo,4.0\n"  // duplicate of row 1 → deduped
        );
        let ds = PolarsDataset::myket(f.path()).unwrap();
        assert_eq!(ds.num_users(), 2);
        assert_eq!(ds.num_items(), 3);
        assert_eq!(ds.interactions().len(), 3);
    }

    #[test]
    fn leave_one_out_is_temporal() {
        // user 0: interacted with items 10, 20, 30 at ts 1, 2, 3
        // user 1: interacted with items 40, 50 at ts 1, 2
        // Expected test: user0→item30, user1→item50
        let f = write_csv(
            "user_id,app_name,timestamp\n\
             0,item10,1.0\n\
             0,item20,2.0\n\
             0,item30,3.0\n\
             1,item40,1.0\n\
             1,item50,2.0\n"
        );
        let ds = PolarsDataset::from_csv(f.path(), "user_id", "app_name", Some("timestamp")).unwrap();
        let (train, val) = ds.leave_one_out();

        // val has exactly 2 entries (one per user)
        assert_eq!(val.len(), 2);
        // train has 3 entries (all minus last per user)
        assert_eq!(train.interactions().len(), 3);

        // the val items must be the "last" by timestamp
        let u0_idx = ds.user_index["0"];
        let u1_idx = ds.user_index["1"];
        let item30_idx = ds.item_index["item30"];
        let item50_idx = ds.item_index["item50"];

        assert!(val.contains(&(u0_idx, item30_idx)), "user0 val item should be item30");
        assert!(val.contains(&(u1_idx, item50_idx)), "user1 val item should be item50");

        // val items must NOT appear in train
        for &(u, i) in &val {
            assert!(!train.interactions().contains(&(u, i)), "val item ({u},{i}) found in train");
        }
    }

    #[test]
    fn single_interaction_user_in_both() {
        let f = write_csv(
            "user_id,app_name,timestamp\n\
             1,com.foo,1.0\n\
             2,com.bar,1.0\n\
             2,com.baz,2.0\n"
        );
        let ds = PolarsDataset::myket(f.path()).unwrap();
        let (train, val) = ds.leave_one_out();
        // user1 has 1 interaction → in val, not in train
        // user2 has 2 → com.baz in val, com.bar in train
        assert_eq!(val.len(), 2);
        assert_eq!(train.interactions().len(), 1); // only user2's com.bar
    }
}
