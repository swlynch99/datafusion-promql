use std::fmt;

/// Range vector functions that operate on a window of samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RangeFunction {
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
    /// Average value of all samples in the range.
    AvgOverTime,
    /// Per-second derivative using simple linear regression (gauge metric).
    Deriv,
    /// Predict value t seconds in the future using simple linear regression.
    PredictLinear,
    /// Count of all samples in the range.
    CountOverTime,
    /// Sum of all sample values in the range.
    SumOverTime,
    /// Minimum sample value in the range.
    MinOverTime,
    /// Maximum sample value in the range.
    MaxOverTime,
    /// Population standard deviation of values in the range.
    StddevOverTime,
    /// Population variance of values in the range.
    StdvarOverTime,
    /// Most recent sample value in the range.
    LastOverTime,
    /// Returns 1 if the range vector has any samples.
    PresentOverTime,
    /// φ-quantile of values in the range. Requires a scalar argument `φ`.
    QuantileOverTime,
}

impl fmt::Display for RangeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rate => write!(f, "rate"),
            Self::Irate => write!(f, "irate"),
            Self::Increase => write!(f, "increase"),
            Self::Delta => write!(f, "delta"),
            Self::Idelta => write!(f, "idelta"),
            Self::AvgOverTime => write!(f, "avg_over_time"),
            Self::Deriv => write!(f, "deriv"),
            Self::PredictLinear => write!(f, "predict_linear"),
            Self::CountOverTime => write!(f, "count_over_time"),
            Self::SumOverTime => write!(f, "sum_over_time"),
            Self::MinOverTime => write!(f, "min_over_time"),
            Self::MaxOverTime => write!(f, "max_over_time"),
            Self::StddevOverTime => write!(f, "stddev_over_time"),
            Self::StdvarOverTime => write!(f, "stdvar_over_time"),
            Self::LastOverTime => write!(f, "last_over_time"),
            Self::PresentOverTime => write!(f, "present_over_time"),
            Self::QuantileOverTime => write!(f, "quantile_over_time"),
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
        "avg_over_time" => Some(RangeFunction::AvgOverTime),
        "deriv" => Some(RangeFunction::Deriv),
        "predict_linear" => Some(RangeFunction::PredictLinear),
        "count_over_time" => Some(RangeFunction::CountOverTime),
        "sum_over_time" => Some(RangeFunction::SumOverTime),
        "min_over_time" => Some(RangeFunction::MinOverTime),
        "max_over_time" => Some(RangeFunction::MaxOverTime),
        "stddev_over_time" => Some(RangeFunction::StddevOverTime),
        "stdvar_over_time" => Some(RangeFunction::StdvarOverTime),
        "last_over_time" => Some(RangeFunction::LastOverTime),
        "present_over_time" => Some(RangeFunction::PresentOverTime),
        "quantile_over_time" => Some(RangeFunction::QuantileOverTime),
        _ => None,
    }
}

impl RangeFunction {
    /// Position of the scalar argument in the PromQL call, if any.
    ///
    /// Returns `None` for range functions that take no scalar argument.
    /// Otherwise returns the zero-based argument index where the scalar
    /// appears in the PromQL source (the range vector takes the remaining
    /// slot).
    pub(crate) fn scalar_arg_position(&self) -> Option<usize> {
        match self {
            // `predict_linear(v range-vector, t scalar)`
            Self::PredictLinear => Some(1),
            // `quantile_over_time(φ scalar, v range-vector)`
            Self::QuantileOverTime => Some(0),
            _ => None,
        }
    }

    /// Evaluate the range function over a window of `(timestamp_ns, value)` samples.
    ///
    /// Samples must be sorted by timestamp. Returns `None` if there are
    /// insufficient samples to compute a result.
    ///
    /// `eval_ts_ns` is the evaluation timestamp in nanoseconds (used by
    /// `deriv` and `predict_linear`). `scalar_arg` is the extra scalar
    /// argument for functions like `predict_linear` and `quantile_over_time`.
    pub fn evaluate(
        &self,
        samples: &[(u64, f64)],
        eval_ts_ns: u64,
        scalar_arg: Option<f64>,
    ) -> Option<f64> {
        if samples.is_empty() {
            return None;
        }

        // Functions that only need 1 sample.
        match self {
            Self::AvgOverTime => {
                let sum: f64 = samples.iter().map(|(_, v)| v).sum();
                return Some(sum / samples.len() as f64);
            }
            Self::CountOverTime => {
                return Some(samples.len() as f64);
            }
            Self::SumOverTime => {
                let sum: f64 = samples.iter().map(|(_, v)| v).sum();
                return Some(sum);
            }
            Self::MinOverTime => {
                let min = samples
                    .iter()
                    .map(|(_, v)| *v)
                    .fold(f64::INFINITY, f64::min);
                return Some(min);
            }
            Self::MaxOverTime => {
                let max = samples
                    .iter()
                    .map(|(_, v)| *v)
                    .fold(f64::NEG_INFINITY, f64::max);
                return Some(max);
            }
            Self::StddevOverTime => {
                return Some(population_variance(samples).sqrt());
            }
            Self::StdvarOverTime => {
                return Some(population_variance(samples));
            }
            Self::LastOverTime => {
                // Samples are sorted by timestamp.
                let (_, last_val) = samples[samples.len() - 1];
                return Some(last_val);
            }
            Self::PresentOverTime => {
                return Some(1.0);
            }
            Self::QuantileOverTime => {
                let phi = scalar_arg.expect("quantile_over_time requires a scalar argument");
                return Some(quantile(samples, phi));
            }
            _ => {}
        }

        // All remaining functions need at least 2 samples.
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
            Self::AvgOverTime
            | Self::CountOverTime
            | Self::SumOverTime
            | Self::MinOverTime
            | Self::MaxOverTime
            | Self::StddevOverTime
            | Self::StdvarOverTime
            | Self::LastOverTime
            | Self::PresentOverTime
            | Self::QuantileOverTime => unreachable!(),
            Self::Deriv => {
                let (_intercept, slope) = linear_regression(samples, eval_ts_ns);
                Some(slope)
            }
            Self::PredictLinear => {
                let t_seconds = scalar_arg.expect("predict_linear requires a scalar argument");
                let (intercept, slope) = linear_regression(samples, eval_ts_ns);
                Some(slope * t_seconds + intercept)
            }
        }
    }
}

/// Population variance of the sample values, computed with Welford's
/// algorithm for numerical stability. Returns `0.0` for a single sample.
fn population_variance(samples: &[(u64, f64)]) -> f64 {
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut count = 0_u64;
    for &(_, v) in samples {
        count += 1;
        let delta = v - mean;
        mean += delta / count as f64;
        let delta2 = v - mean;
        m2 += delta * delta2;
    }
    m2 / count as f64
}

/// φ-quantile of the sample values, matching Prometheus semantics:
///
/// - φ < 0 returns `-Inf`
/// - φ > 1 returns `+Inf`
/// - φ = NaN returns NaN
/// - Otherwise performs linear interpolation between the two nearest
///   ranks on the sorted values.
fn quantile(samples: &[(u64, f64)], phi: f64) -> f64 {
    if phi.is_nan() {
        return f64::NAN;
    }
    if phi < 0.0 {
        return f64::NEG_INFINITY;
    }
    if phi > 1.0 {
        return f64::INFINITY;
    }

    let mut values: Vec<f64> = samples.iter().map(|(_, v)| *v).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = values.len();
    if n == 1 {
        return values[0];
    }

    let rank = phi * (n - 1) as f64;
    let lower_idx = rank.floor() as usize;
    let upper_idx = (lower_idx + 1).min(n - 1);
    let weight = rank - lower_idx as f64;
    values[lower_idx] * (1.0 - weight) + values[upper_idx] * weight
}

/// Simple linear regression over `(timestamp_ns, value)` samples.
///
/// Computes intercept and slope where x-values are seconds relative to
/// `intercept_time_ns`. This matches the Prometheus implementation:
/// intercept is the predicted value at `intercept_time_ns`, and slope is
/// the per-second rate of change.
///
/// Returns `(intercept, slope)`.
fn linear_regression(samples: &[(u64, f64)], intercept_time_ns: u64) -> (f64, f64) {
    let n = samples.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;

    for &(ts, val) in samples {
        let x = (ts as f64 - intercept_time_ns as f64) / 1_000_000_000.0;
        sum_x += x;
        sum_y += val;
        sum_xy += x * val;
        sum_x2 += x * x;
    }

    let cov_xy = sum_xy - sum_x * sum_y / n;
    let var_x = sum_x2 - sum_x * sum_x / n;

    let slope = cov_xy / var_x;
    let intercept = sum_y / n - slope * sum_x / n;

    (intercept, slope)
}

/// Compute the total counter increase across the samples, handling resets.
///
/// A counter reset is detected when a value is less than the preceding value.
/// In that case, the new value is added as-is (assuming it reset from 0).
fn counter_increase(samples: &[(u64, f64)]) -> f64 {
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
        let result = RangeFunction::Rate.evaluate(&samples, 0, None).unwrap();
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
        let result = RangeFunction::Rate.evaluate(&samples, 0, None).unwrap();
        assert!(
            (result - 8.75).abs() < f64::EPSILON,
            "expected 8.75, got {result}"
        );
    }

    #[test]
    fn test_rate_insufficient_samples() {
        let samples = vec![(1_000_000_000, 10.0)];
        assert!(RangeFunction::Rate.evaluate(&samples, 0, None).is_none());
    }

    #[test]
    fn test_rate_zero_duration() {
        let samples = vec![(1_000_000_000, 10.0), (1_000_000_000, 20.0)];
        assert!(RangeFunction::Rate.evaluate(&samples, 0, None).is_none());
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
        let result = RangeFunction::Irate.evaluate(&samples, 0, None).unwrap();
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
        let result = RangeFunction::Irate.evaluate(&samples, 0, None).unwrap();
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
        let result = RangeFunction::Increase.evaluate(&samples, 0, None).unwrap();
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
        let result = RangeFunction::Increase.evaluate(&samples, 0, None).unwrap();
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
        let result = RangeFunction::Delta.evaluate(&samples, 0, None).unwrap();
        assert!(
            (result - 8.0).abs() < f64::EPSILON,
            "expected 8.0, got {result}"
        );
    }

    #[test]
    fn test_delta_negative() {
        let samples = vec![(0, 20.0), (1_000_000_000, 15.0), (2_000_000_000, 10.0)];
        let result = RangeFunction::Delta.evaluate(&samples, 0, None).unwrap();
        assert!(
            (result - (-10.0)).abs() < f64::EPSILON,
            "expected -10.0, got {result}"
        );
    }

    #[test]
    fn test_delta_insufficient_samples() {
        let samples = vec![(1_000_000_000, 10.0)];
        assert!(RangeFunction::Delta.evaluate(&samples, 0, None).is_none());
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
        let result = RangeFunction::Idelta.evaluate(&samples, 0, None).unwrap();
        assert!(
            (result - 6.0).abs() < f64::EPSILON,
            "expected 6.0, got {result}"
        );
    }

    #[test]
    fn test_idelta_negative() {
        // Last two: (2s, 10) - (1s, 15) = -5.0
        let samples = vec![(0, 20.0), (1_000_000_000, 15.0), (2_000_000_000, 10.0)];
        let result = RangeFunction::Idelta.evaluate(&samples, 0, None).unwrap();
        assert!(
            (result - (-5.0)).abs() < f64::EPSILON,
            "expected -5.0, got {result}"
        );
    }

    #[test]
    fn test_idelta_insufficient_samples() {
        let samples = vec![(1_000_000_000, 10.0)];
        assert!(RangeFunction::Idelta.evaluate(&samples, 0, None).is_none());
    }

    #[test]
    fn test_avg_over_time_basic() {
        let samples = vec![
            (0, 10.0),
            (1_000_000_000, 20.0),
            (2_000_000_000, 30.0),
            (3_000_000_000, 40.0),
        ];
        let result = RangeFunction::AvgOverTime
            .evaluate(&samples, 0, None)
            .unwrap();
        assert!(
            (result - 25.0).abs() < f64::EPSILON,
            "expected 25.0, got {result}"
        );
    }

    #[test]
    fn test_avg_over_time_single_sample() {
        let samples = vec![(1_000_000_000, 42.0)];
        let result = RangeFunction::AvgOverTime
            .evaluate(&samples, 0, None)
            .unwrap();
        assert!(
            (result - 42.0).abs() < f64::EPSILON,
            "expected 42.0, got {result}"
        );
    }

    #[test]
    fn test_avg_over_time_empty() {
        let samples: Vec<(u64, f64)> = vec![];
        assert!(
            RangeFunction::AvgOverTime
                .evaluate(&samples, 0, None)
                .is_none()
        );
    }

    #[test]
    fn test_deriv_constant_slope() {
        // Linear increase of 10 per second over 4 seconds.
        let samples = vec![
            (0, 0.0),
            (1_000_000_000, 10.0),
            (2_000_000_000, 20.0),
            (3_000_000_000, 30.0),
            (4_000_000_000, 40.0),
        ];
        // Eval at t=4s
        let result = RangeFunction::Deriv
            .evaluate(&samples, 4_000_000_000, None)
            .unwrap();
        assert!((result - 10.0).abs() < 1e-9, "expected 10.0, got {result}");
    }

    #[test]
    fn test_deriv_insufficient_samples() {
        let samples = vec![(1_000_000_000, 10.0)];
        assert!(
            RangeFunction::Deriv
                .evaluate(&samples, 1_000_000_000, None)
                .is_none()
        );
    }

    #[test]
    fn test_predict_linear_basic() {
        // Linear increase of 10 per second. Predict 10 seconds into the future.
        // At eval time (4s), value is 40. After 10 more seconds, should be 140.
        let samples = vec![
            (0, 0.0),
            (1_000_000_000, 10.0),
            (2_000_000_000, 20.0),
            (3_000_000_000, 30.0),
            (4_000_000_000, 40.0),
        ];
        let result = RangeFunction::PredictLinear
            .evaluate(&samples, 4_000_000_000, Some(10.0))
            .unwrap();
        // intercept at eval_ts (4s) = 40, slope = 10/s, predict at +10s = 40 + 100 = 140
        assert!(
            (result - 140.0).abs() < 1e-9,
            "expected 140.0, got {result}"
        );
    }

    #[test]
    fn test_predict_linear_negative_slope() {
        // Decreasing by 5 per second.
        let samples = vec![(0, 100.0), (1_000_000_000, 95.0), (2_000_000_000, 90.0)];
        // Predict 20 seconds from eval time (2s).
        // intercept at 2s = 90, slope = -5/s, predict = 90 + (-5)*20 = -10
        let result = RangeFunction::PredictLinear
            .evaluate(&samples, 2_000_000_000, Some(20.0))
            .unwrap();
        assert!(
            (result - (-10.0)).abs() < 1e-9,
            "expected -10.0, got {result}"
        );
    }

    #[test]
    fn test_predict_linear_insufficient_samples() {
        let samples = vec![(1_000_000_000, 10.0)];
        assert!(
            RangeFunction::PredictLinear
                .evaluate(&samples, 1_000_000_000, Some(10.0))
                .is_none()
        );
    }

    fn over_time_samples() -> Vec<(u64, f64)> {
        vec![
            (0, 10.0),
            (1_000_000_000, 20.0),
            (2_000_000_000, 30.0),
            (3_000_000_000, 40.0),
        ]
    }

    #[test]
    fn test_sum_over_time_basic() {
        let result = RangeFunction::SumOverTime
            .evaluate(&over_time_samples(), 0, None)
            .unwrap();
        assert!(
            (result - 100.0).abs() < f64::EPSILON,
            "expected 100.0, got {result}"
        );
    }

    #[test]
    fn test_sum_over_time_single_sample() {
        let samples = vec![(1_000_000_000, 42.0)];
        let result = RangeFunction::SumOverTime
            .evaluate(&samples, 0, None)
            .unwrap();
        assert!(
            (result - 42.0).abs() < f64::EPSILON,
            "expected 42.0, got {result}"
        );
    }

    #[test]
    fn test_min_over_time_basic() {
        let samples = vec![
            (0, 20.0),
            (1_000_000_000, 5.0),
            (2_000_000_000, 15.0),
            (3_000_000_000, 10.0),
        ];
        let result = RangeFunction::MinOverTime
            .evaluate(&samples, 0, None)
            .unwrap();
        assert!(
            (result - 5.0).abs() < f64::EPSILON,
            "expected 5.0, got {result}"
        );
    }

    #[test]
    fn test_max_over_time_basic() {
        let samples = vec![
            (0, 20.0),
            (1_000_000_000, 5.0),
            (2_000_000_000, 35.0),
            (3_000_000_000, 10.0),
        ];
        let result = RangeFunction::MaxOverTime
            .evaluate(&samples, 0, None)
            .unwrap();
        assert!(
            (result - 35.0).abs() < f64::EPSILON,
            "expected 35.0, got {result}"
        );
    }

    #[test]
    fn test_stdvar_over_time_basic() {
        // Values: 10, 20, 30, 40. Mean = 25. Variance = ((15^2)+(5^2)+(5^2)+(15^2))/4 = 500/4 = 125
        let result = RangeFunction::StdvarOverTime
            .evaluate(&over_time_samples(), 0, None)
            .unwrap();
        assert!(
            (result - 125.0).abs() < 1e-9,
            "expected 125.0, got {result}"
        );
    }

    #[test]
    fn test_stddev_over_time_basic() {
        // sqrt(125) ≈ 11.180339887498949
        let result = RangeFunction::StddevOverTime
            .evaluate(&over_time_samples(), 0, None)
            .unwrap();
        assert!(
            (result - 125.0_f64.sqrt()).abs() < 1e-9,
            "expected sqrt(125), got {result}"
        );
    }

    #[test]
    fn test_stddev_over_time_single_sample() {
        let samples = vec![(1_000_000_000, 42.0)];
        let result = RangeFunction::StddevOverTime
            .evaluate(&samples, 0, None)
            .unwrap();
        assert!(result.abs() < f64::EPSILON, "expected 0.0, got {result}");
    }

    #[test]
    fn test_last_over_time_basic() {
        // Samples are assumed sorted by timestamp.
        let result = RangeFunction::LastOverTime
            .evaluate(&over_time_samples(), 0, None)
            .unwrap();
        assert!(
            (result - 40.0).abs() < f64::EPSILON,
            "expected 40.0, got {result}"
        );
    }

    #[test]
    fn test_last_over_time_single_sample() {
        let samples = vec![(1_000_000_000, 42.0)];
        let result = RangeFunction::LastOverTime
            .evaluate(&samples, 0, None)
            .unwrap();
        assert!(
            (result - 42.0).abs() < f64::EPSILON,
            "expected 42.0, got {result}"
        );
    }

    #[test]
    fn test_present_over_time_returns_one() {
        let result = RangeFunction::PresentOverTime
            .evaluate(&over_time_samples(), 0, None)
            .unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_present_over_time_empty() {
        let samples: Vec<(u64, f64)> = vec![];
        assert!(
            RangeFunction::PresentOverTime
                .evaluate(&samples, 0, None)
                .is_none()
        );
    }

    #[test]
    fn test_quantile_over_time_median() {
        // Sorted values: 10, 20, 30, 40. φ=0.5 → interpolate between index 1
        // and 2: 0.5 * (20 + 30) = 25.
        let result = RangeFunction::QuantileOverTime
            .evaluate(&over_time_samples(), 0, Some(0.5))
            .unwrap();
        assert!((result - 25.0).abs() < 1e-9, "expected 25.0, got {result}");
    }

    #[test]
    fn test_quantile_over_time_endpoints() {
        let min = RangeFunction::QuantileOverTime
            .evaluate(&over_time_samples(), 0, Some(0.0))
            .unwrap();
        assert!((min - 10.0).abs() < f64::EPSILON);

        let max = RangeFunction::QuantileOverTime
            .evaluate(&over_time_samples(), 0, Some(1.0))
            .unwrap();
        assert!((max - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quantile_over_time_out_of_range() {
        let neg = RangeFunction::QuantileOverTime
            .evaluate(&over_time_samples(), 0, Some(-0.1))
            .unwrap();
        assert!(neg.is_infinite() && neg < 0.0);

        let pos = RangeFunction::QuantileOverTime
            .evaluate(&over_time_samples(), 0, Some(1.1))
            .unwrap();
        assert!(pos.is_infinite() && pos > 0.0);
    }

    #[test]
    fn test_quantile_over_time_single_sample() {
        let samples = vec![(1_000_000_000, 42.0)];
        let result = RangeFunction::QuantileOverTime
            .evaluate(&samples, 0, Some(0.75))
            .unwrap();
        assert!((result - 42.0).abs() < f64::EPSILON);
    }
}
