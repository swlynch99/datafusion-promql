use std::fmt;

/// Instant vector functions that apply an element-wise transformation to sample values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum InstantFunction {
    /// Base-2 logarithm of each sample value.
    Log2,
}

impl fmt::Display for InstantFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Log2 => write!(f, "log2"),
        }
    }
}

/// Look up an instant function by name.
pub(crate) fn lookup_instant_function(name: &str) -> Option<InstantFunction> {
    match name {
        "log2" => Some(InstantFunction::Log2),
        _ => None,
    }
}

impl InstantFunction {
    /// Apply the function to a single sample value.
    pub fn evaluate(&self, value: f64) -> f64 {
        match self {
            Self::Log2 => value.log2(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_power_of_two() {
        assert!((InstantFunction::Log2.evaluate(1.0) - 0.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Log2.evaluate(2.0) - 1.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Log2.evaluate(4.0) - 2.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Log2.evaluate(8.0) - 3.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Log2.evaluate(1024.0) - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_log2_zero() {
        // log2(0) = -inf in IEEE 754
        let result = InstantFunction::Log2.evaluate(0.0);
        assert!(
            result.is_infinite() && result.is_sign_negative(),
            "log2(0) should be -inf, got {result}"
        );
    }

    #[test]
    fn test_log2_negative() {
        // log2 of a negative number is NaN
        let result = InstantFunction::Log2.evaluate(-1.0);
        assert!(result.is_nan(), "log2(-1) should be NaN, got {result}");
    }

    #[test]
    fn test_lookup_log2() {
        assert_eq!(lookup_instant_function("log2"), Some(InstantFunction::Log2));
    }

    #[test]
    fn test_lookup_unknown() {
        assert_eq!(lookup_instant_function("log10"), None);
        assert_eq!(lookup_instant_function("unknown"), None);
        assert_eq!(lookup_instant_function(""), None);
    }
}
