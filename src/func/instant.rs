use std::fmt;

/// Instant vector functions that transform each sample value pointwise.
#[derive(Debug, Clone, Copy)]
pub(crate) enum InstantFunction {
    /// Round each value to the nearest multiple of `to_nearest`.
    Round { to_nearest: f64 },
}

impl InstantFunction {
    /// Apply the function to a single sample value.
    pub fn evaluate(&self, value: f64) -> f64 {
        match self {
            Self::Round { to_nearest } => promql_round(value, *to_nearest),
        }
    }
}

impl fmt::Display for InstantFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Round { to_nearest } => write!(f, "round(to_nearest={to_nearest})"),
        }
    }
}

/// Look up an instant function by name, given any extra scalar arguments beyond the
/// first (vector) argument.
///
/// Returns `None` if the name is not a known instant function.
pub(crate) fn lookup_instant_function(name: &str, extra_args: &[f64]) -> Option<InstantFunction> {
    match name {
        "round" => {
            let to_nearest = extra_args.first().copied().unwrap_or(1.0);
            Some(InstantFunction::Round { to_nearest })
        }
        _ => None,
    }
}

/// PromQL `round(v, to_nearest)` semantics.
///
/// Rounds `value` to the nearest multiple of `to_nearest`.
/// Uses the algorithm: `floor(value / to_nearest + 0.5) * to_nearest`.
/// If `to_nearest` is 0, the value is returned unchanged.
fn promql_round(value: f64, to_nearest: f64) -> f64 {
    if to_nearest == 0.0 {
        return value;
    }
    (value / to_nearest + 0.5).floor() * to_nearest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_default_to_nearest_one() {
        let f = InstantFunction::Round { to_nearest: 1.0 };
        assert!((f.evaluate(2.3) - 2.0).abs() < f64::EPSILON, "2.3 -> 2");
        assert!((f.evaluate(2.5) - 3.0).abs() < f64::EPSILON, "2.5 -> 3");
        assert!((f.evaluate(2.7) - 3.0).abs() < f64::EPSILON, "2.7 -> 3");
        assert!((f.evaluate(-1.5) - (-1.0)).abs() < f64::EPSILON, "-1.5 -> -1");
        assert!((f.evaluate(0.0) - 0.0).abs() < f64::EPSILON, "0 -> 0");
    }

    #[test]
    fn test_round_to_nearest_point_one() {
        let f = InstantFunction::Round { to_nearest: 0.1 };
        // 3.14159 -> 3.1 (rounds to nearest 0.1)
        let result = f.evaluate(3.14159);
        assert!((result - 3.1).abs() < 1e-10, "expected 3.1, got {result}");
    }

    #[test]
    fn test_round_to_nearest_five() {
        let f = InstantFunction::Round { to_nearest: 5.0 };
        assert!((f.evaluate(12.0) - 10.0).abs() < f64::EPSILON, "12 -> 10");
        assert!((f.evaluate(13.0) - 15.0).abs() < f64::EPSILON, "13 -> 15");
        assert!((f.evaluate(17.5) - 20.0).abs() < f64::EPSILON, "17.5 -> 20");
    }

    #[test]
    fn test_round_to_nearest_zero() {
        // When to_nearest is 0, value passes through unchanged.
        let f = InstantFunction::Round { to_nearest: 0.0 };
        assert!((f.evaluate(3.7) - 3.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_round_negative_values() {
        let f = InstantFunction::Round { to_nearest: 1.0 };
        // floor(-2.3 + 0.5) = floor(-1.8) = -2
        assert!((f.evaluate(-2.3) - (-2.0)).abs() < f64::EPSILON, "-2.3 -> -2");
        // floor(-2.7 + 0.5) = floor(-2.2) = -3
        assert!((f.evaluate(-2.7) - (-3.0)).abs() < f64::EPSILON, "-2.7 -> -3");
    }

    #[test]
    fn test_lookup_instant_function_round_no_args() {
        let f = lookup_instant_function("round", &[]).unwrap();
        match f {
            InstantFunction::Round { to_nearest } => {
                assert!((to_nearest - 1.0).abs() < f64::EPSILON)
            }
        }
    }

    #[test]
    fn test_lookup_instant_function_round_with_arg() {
        let f = lookup_instant_function("round", &[0.5]).unwrap();
        match f {
            InstantFunction::Round { to_nearest } => {
                assert!((to_nearest - 0.5).abs() < f64::EPSILON)
            }
        }
    }

    #[test]
    fn test_lookup_instant_function_unknown() {
        assert!(lookup_instant_function("unknown_func", &[]).is_none());
    }
}
