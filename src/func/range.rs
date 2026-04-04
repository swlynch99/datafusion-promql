use std::fmt;

/// Range vector functions that operate on a window of samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RangeFunction {
    /// Per-second rate of increase (counter metric).
    Rate,
    /// Instant rate using only the last two samples (counter metric).
    Irate,
    /// Total increase over the range window (counter metric).
    Increase,
    /// Difference between last and first sample (gauge metric).
    Delta,
    /// Difference between the last two samples (gauge equivalent of irate).
    Idelta,
}

impl fmt::Display for RangeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rate => write!(f, "rate"),
            Self::Irate => write!(f, "irate"),
            Self::Increase => write!(f, "increase"),
            Self::Delta => write!(f, "delta"),
            Self::Idelta => write!(f, "idelta"),
        }
    }
}

/// Look up a range function by name.
pub(crate) fn lookup_range_function(name: &str) -> Option<RangeFunction> {
    match name {
        "rate" => Some(RangeFunction::Rate),
        "irate" => Some(RangeFunction::Irate),
        "increase" => Some(RangeFunction::Increase),
        "delta" => Some(RangeFunction::Delta),
        "idelta" => Some(RangeFunction::Idelta),
        _ => None,
    }
}

impl RangeFunction {
    /// Evaluate the range function over a window of `(timestamp_ns, value)` samples.
    ///
    /// Samples must be sorted by timestamp. Returns `None` if there are
    /// insufficient samples to compute a result.
    pub fn evaluate(&self, samples: &[(i64, f64)]) -> Option<f64> {
        if samples.len() < 2 {
            return None;
        }

        match self {
            Self::Rate => {
                let (first_ts, _) = samples[0];
                let (last_ts, _) = samples[samples.len() - 1];
                let dt_secs = (last_ts - first_ts) as f64 / 1_000_000_000.0;
                if dt_secs == 0.0 {
                    return None;
                }
                let increase = counter_increase(samples);
                Some(increase / dt_secs)
            }
            Self::Irate => {
                let n = samples.len();
                let (prev_ts, prev_val) = samples[n - 2];
                let (last_ts, last_val) = samples[n - 1];
                let dt_secs = (last_ts - prev_ts) as f64 / 1_000_000_000.0;
                if dt_secs == 0.0 {
                    return None;
                }
                // Handle counter reset: if value decreased, assume reset and
                // use last_val as the increase.
                let increase = if last_val < prev_val {
                    last_val
                } else {
                    last_val - prev_val
                };
                Some(increase / dt_secs)
            }
            Self::Increase => Some(counter_increase(samples)),
            Self::Delta => {
                let (_, first_val) = samples[0];
                let (_, last_val) = samples[samples.len() - 1];
                Some(last_val - first_val)
            }
            Self::Idelta => {
                let n = samples.len();
                let (_, prev_val) = samples[n - 2];
                let (_, last_val) = samples[n - 1];
                Some(last_val - prev_val)
            }
        }
    }
}

/// Compute the total counter increase across the samples, handling resets.
///
/// A counter reset is detected when a value is less than the preceding value.
/// In that case, the new value is added as-is (assuming it reset from 0).
fn counter_increase(samples: &[(i64, f64)]) -> f64 {
    let mut total = 0.0;
    for i in 1..samples.len() {
        let delta = samples[i].1 - samples[i - 1].1;
        if delta >= 0.0 {
            total += delta;
        } else {
            // Counter reset: add the new value (increase from 0).
            total += samples[i].1;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_basic() {
        // 10 per second increase over 5 seconds
        let samples = vec![
            (0, 0.0),
            (1_000_000_000, 10.0),
            (2_000_000_000, 20.0),
            (3_000_000_000, 30.0),
            (4_000_000_000, 40.0),
            (5_000_000_000, 50.0),
        ];
        let result = RangeFunction::Rate.evaluate(&samples).unwrap();
        assert!(
            (result - 10.0).abs() < f64::EPSILON,
            "expected 10.0, got {result}"
        );
    }

    #[test]
    fn test_rate_with_counter_reset() {
        // Counter goes 0, 10, 20, 5 (reset), 15
        // Increases: 10, 10, 5(reset), 10 = 35 total over 4 seconds
        let samples = vec![
            (0, 0.0),
            (1_000_000_000, 10.0),
            (2_000_000_000, 20.0),
            (3_000_000_000, 5.0), // reset
            (4_000_000_000, 15.0),
        ];
        let result = RangeFunction::Rate.evaluate(&samples).unwrap();
        assert!(
            (result - 8.75).abs() < f64::EPSILON,
            "expected 8.75, got {result}"
        );
    }

    #[test]
    fn test_rate_insufficient_samples() {
        let samples = vec![(1_000_000_000, 10.0)];
        assert!(RangeFunction::Rate.evaluate(&samples).is_none());
    }

    #[test]
    fn test_rate_zero_duration() {
        let samples = vec![(1_000_000_000, 10.0), (1_000_000_000, 20.0)];
        assert!(RangeFunction::Rate.evaluate(&samples).is_none());
    }

    #[test]
    fn test_irate_basic() {
        // Only uses last two samples: (4s, 40) and (5s, 50) -> 10/1 = 10
        let samples = vec![
            (0, 0.0),
            (1_000_000_000, 10.0),
            (2_000_000_000, 20.0),
            (3_000_000_000, 30.0),
            (4_000_000_000, 40.0),
            (5_000_000_000, 50.0),
        ];
        let result = RangeFunction::Irate.evaluate(&samples).unwrap();
        assert!(
            (result - 10.0).abs() < f64::EPSILON,
            "expected 10.0, got {result}"
        );
    }

    #[test]
    fn test_irate_with_counter_reset() {
        // Last two: (2s, 20) and (3s, 5) -> reset, increase = 5, dt = 1s
        let samples = vec![
            (0, 0.0),
            (1_000_000_000, 10.0),
            (2_000_000_000, 20.0),
            (3_000_000_000, 5.0),
        ];
        let result = RangeFunction::Irate.evaluate(&samples).unwrap();
        assert!(
            (result - 5.0).abs() < f64::EPSILON,
            "expected 5.0, got {result}"
        );
    }

    #[test]
    fn test_increase_basic() {
        let samples = vec![
            (0, 100.0),
            (1_000_000_000, 110.0),
            (2_000_000_000, 120.0),
            (3_000_000_000, 130.0),
        ];
        let result = RangeFunction::Increase.evaluate(&samples).unwrap();
        assert!(
            (result - 30.0).abs() < f64::EPSILON,
            "expected 30.0, got {result}"
        );
    }

    #[test]
    fn test_increase_with_counter_reset() {
        // 0 -> 10 -> 20 -> 5(reset) -> 15: increases = 10 + 10 + 5 + 10 = 35
        let samples = vec![
            (0, 0.0),
            (1_000_000_000, 10.0),
            (2_000_000_000, 20.0),
            (3_000_000_000, 5.0),
            (4_000_000_000, 15.0),
        ];
        let result = RangeFunction::Increase.evaluate(&samples).unwrap();
        assert!(
            (result - 35.0).abs() < f64::EPSILON,
            "expected 35.0, got {result}"
        );
    }

    #[test]
    fn test_delta_basic() {
        let samples = vec![
            (0, 10.0),
            (1_000_000_000, 15.0),
            (2_000_000_000, 12.0),
            (3_000_000_000, 18.0),
        ];
        let result = RangeFunction::Delta.evaluate(&samples).unwrap();
        assert!(
            (result - 8.0).abs() < f64::EPSILON,
            "expected 8.0, got {result}"
        );
    }

    #[test]
    fn test_delta_negative() {
        let samples = vec![(0, 20.0), (1_000_000_000, 15.0), (2_000_000_000, 10.0)];
        let result = RangeFunction::Delta.evaluate(&samples).unwrap();
        assert!(
            (result - (-10.0)).abs() < f64::EPSILON,
            "expected -10.0, got {result}"
        );
    }

    #[test]
    fn test_delta_insufficient_samples() {
        let samples = vec![(1_000_000_000, 10.0)];
        assert!(RangeFunction::Delta.evaluate(&samples).is_none());
    }

    #[test]
    fn test_idelta_basic() {
        // idelta uses only the last two samples: (3s, 18) - (2s, 12) = 6.0
        let samples = vec![
            (0, 10.0),
            (1_000_000_000, 15.0),
            (2_000_000_000, 12.0),
            (3_000_000_000, 18.0),
        ];
        let result = RangeFunction::Idelta.evaluate(&samples).unwrap();
        assert!(
            (result - 6.0).abs() < f64::EPSILON,
            "expected 6.0, got {result}"
        );
    }

    #[test]
    fn test_idelta_negative() {
        // Last two: (2s, 10) - (1s, 15) = -5.0
        let samples = vec![(0, 20.0), (1_000_000_000, 15.0), (2_000_000_000, 10.0)];
        let result = RangeFunction::Idelta.evaluate(&samples).unwrap();
        assert!(
            (result - (-5.0)).abs() < f64::EPSILON,
            "expected -5.0, got {result}"
        );
    }

    #[test]
    fn test_idelta_insufficient_samples() {
        let samples = vec![(1_000_000_000, 10.0)];
        assert!(RangeFunction::Idelta.evaluate(&samples).is_none());
    }
}
