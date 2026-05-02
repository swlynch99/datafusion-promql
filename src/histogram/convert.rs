//! Conversions between sparse `(indices, counts)` rows and the
//! `iopsystems/histogram` crate's dense [`histogram::Histogram`].
//!
//! `HistogramArray` rows are already sparse and zero-stripped. The
//! `histogram` crate carries the analytics implementation we want to call
//! into (quantiles, sums, etc.), so the engine pivots between the two:
//! load a row into a dense `Histogram` to run a UDF; pivot a dense
//! `Histogram` back into a sparse row to write a result column.
//!
//! Counts here are per-bucket — not a running cumulative — matching
//! [`histogram::SparseHistogram`] / [`histogram::Histogram`] semantics.
//! Cumulative-vs-delta logic lives elsewhere (see `plans/histograms.md`,
//! Task 1.4).

use histogram::{Config, Histogram};

use super::HistogramConfig;

impl From<HistogramConfig> for Config {
    fn from(c: HistogramConfig) -> Self {
        // The HistogramConfig newtype is constructed from validated metadata
        // upstream (parquet ingest or Field::metadata round-trip), but we
        // still surface the upstream error rather than panic to keep the
        // boundary honest if a future caller hand-builds one.
        Config::new(c.grouping_power, c.max_value_power)
            .expect("HistogramConfig encodes a valid histogram::Config")
    }
}

impl From<Config> for HistogramConfig {
    fn from(c: Config) -> Self {
        Self::new(c.grouping_power(), c.max_value_power())
    }
}

/// Build a dense [`histogram::Histogram`] from a sparse row.
///
/// `indices` are bucket positions into the dense buckets array; `counts[i]`
/// is the per-bucket count at `indices[i]`. The two slices must have the
/// same length and every index must be `< config.total_buckets()`. Indices
/// are not required to be sorted or unique — duplicate indices have their
/// counts summed (wrapping on overflow, matching [`Histogram::add`]).
///
/// # Panics
///
/// Panics if `indices.len() != counts.len()`, if any index does not fit in
/// `usize`, or if any index is out of range for `config`.
pub fn row_to_histogram(indices: &[u64], counts: &[u64], config: &HistogramConfig) -> Histogram {
    assert_eq!(
        indices.len(),
        counts.len(),
        "row_to_histogram: indices.len ({}) != counts.len ({})",
        indices.len(),
        counts.len(),
    );

    let cfg: Config = (*config).into();
    let mut h = Histogram::with_config(&cfg);
    let total = cfg.total_buckets();
    let buckets = h.as_mut_slice();

    for (&idx, &cnt) in indices.iter().zip(counts.iter()) {
        let i = usize::try_from(idx).expect("bucket index fits in usize");
        assert!(
            i < total,
            "row_to_histogram: index {i} out of range for config with {total} buckets",
        );
        buckets[i] = buckets[i].wrapping_add(cnt);
    }

    h
}

/// Extract a sparse `(indices, counts)` row from a dense
/// [`histogram::Histogram`], skipping zero-count buckets.
///
/// The output is sorted ascending by bucket index — a property of iterating
/// the dense slice — and `(indices, counts)` round-trips through
/// [`row_to_histogram`] at the same `Config`.
pub fn histogram_to_row(h: &Histogram) -> (Vec<u64>, Vec<u64>) {
    let buckets = h.as_slice();
    let nonzero = buckets.iter().filter(|&&c| c != 0).count();

    let mut indices = Vec::with_capacity(nonzero);
    let mut counts = Vec::with_capacity(nonzero);
    for (i, &c) in buckets.iter().enumerate() {
        if c == 0 {
            continue;
        }
        indices.push(i as u64);
        counts.push(c);
    }
    (indices, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic LCG so tests don't pull in a `rand` dev-dep.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            // Numerical Recipes constants — fine for test fixtures, not
            // for anything with security or distribution requirements.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }

        fn gen_range(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    fn cfg() -> HistogramConfig {
        // grouping_power=4, max_value_power=16 ⇒ 192 buckets — small enough
        // for the all-buckets-nonzero case to stay cheap.
        HistogramConfig::new(4, 16)
    }

    #[test]
    fn config_round_trip_through_histogram_crate() {
        let ours = HistogramConfig::new(7, 32);
        let theirs: Config = ours.into();
        assert_eq!(theirs.grouping_power(), 7);
        assert_eq!(theirs.max_value_power(), 32);
        assert_eq!(HistogramConfig::from(theirs), ours);
    }

    #[test]
    fn empty_row_yields_empty_histogram() {
        let config = cfg();
        let h = row_to_histogram(&[], &[], &config);
        assert_eq!(h.config(), Config::from(config));
        assert!(h.as_slice().iter().all(|&c| c == 0));

        let (idx, cnt) = histogram_to_row(&h);
        assert!(idx.is_empty());
        assert!(cnt.is_empty());
    }

    #[test]
    fn single_nonzero_bucket_round_trips() {
        let config = cfg();
        let indices = vec![17u64];
        let counts = vec![42u64];

        let h = row_to_histogram(&indices, &counts, &config);
        assert_eq!(h.as_slice()[17], 42);
        // Only that one bucket should be set.
        assert_eq!(h.as_slice().iter().filter(|&&c| c != 0).count(), 1);

        let (out_idx, out_cnt) = histogram_to_row(&h);
        assert_eq!(out_idx, indices);
        assert_eq!(out_cnt, counts);
    }

    #[test]
    fn dense_all_buckets_nonzero_round_trips() {
        let config = cfg();
        let total = Config::from(config).total_buckets();

        let indices: Vec<u64> = (0..total as u64).collect();
        let counts: Vec<u64> = (0..total as u64).map(|i| i + 1).collect();

        let h = row_to_histogram(&indices, &counts, &config);
        // Every bucket holds exactly its (1-based) count.
        for (i, &c) in h.as_slice().iter().enumerate() {
            assert_eq!(c, (i + 1) as u64);
        }

        let (out_idx, out_cnt) = histogram_to_row(&h);
        assert_eq!(out_idx, indices);
        assert_eq!(out_cnt, counts);
    }

    #[test]
    fn random_sparse_rows_round_trip() {
        let config = cfg();
        let total = Config::from(config).total_buckets();

        let mut rng = Lcg::new(0xDEADBEEFCAFEBABE);
        for _ in 0..32 {
            let n = 1 + rng.gen_range(total / 4);
            // Generate sorted unique indices so the input itself matches
            // what `histogram_to_row` produces.
            let mut chosen: Vec<u64> = Vec::with_capacity(n);
            let mut seen = vec![false; total];
            while chosen.len() < n {
                let i = rng.gen_range(total);
                if !seen[i] {
                    seen[i] = true;
                    chosen.push(i as u64);
                }
            }
            chosen.sort_unstable();

            let counts: Vec<u64> = (0..n).map(|_| 1 + rng.next_u64() % 1_000_000).collect();

            let h = row_to_histogram(&chosen, &counts, &config);
            let (out_idx, out_cnt) = histogram_to_row(&h);
            assert_eq!(out_idx, chosen);
            assert_eq!(out_cnt, counts);
        }
    }

    #[test]
    fn duplicate_indices_are_summed() {
        let config = cfg();
        let h = row_to_histogram(&[3, 3, 5], &[10, 7, 1], &config);
        assert_eq!(h.as_slice()[3], 17);
        assert_eq!(h.as_slice()[5], 1);

        let (idx, cnt) = histogram_to_row(&h);
        assert_eq!(idx, vec![3, 5]);
        assert_eq!(cnt, vec![17, 1]);
    }

    #[test]
    fn histogram_to_row_skips_zero_buckets() {
        let cfg_native: Config = cfg().into();
        let mut h = Histogram::with_config(&cfg_native);
        h.as_mut_slice()[0] = 0;
        h.as_mut_slice()[1] = 5;
        h.as_mut_slice()[2] = 0;
        h.as_mut_slice()[3] = 9;
        h.as_mut_slice()[4] = 0;

        let (idx, cnt) = histogram_to_row(&h);
        assert_eq!(idx, vec![1, 3]);
        assert_eq!(cnt, vec![5, 9]);
    }

    #[test]
    #[should_panic(expected = "indices.len")]
    fn mismatched_lengths_panic() {
        let _ = row_to_histogram(&[1, 2], &[3], &cfg());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_index_panics() {
        let config = cfg();
        let total = Config::from(config).total_buckets() as u64;
        let _ = row_to_histogram(&[total], &[1], &config);
    }
}
