use std::fmt;
use std::hash::{Hash, Hasher};

/// Instant vector functions that transform each sample value pointwise.
#[derive(Debug, Clone, Copy)]
pub(crate) enum InstantFunction {
    /// Absolute value of each sample value.
    Abs,
    /// Arccosine of each sample value (radians).
    Acos,
    /// Inverse hyperbolic cosine of each sample value.
    Acosh,
    /// Arcsine of each sample value (radians).
    Asin,
    /// Inverse hyperbolic sine of each sample value.
    Asinh,
    /// Arctangent of each sample value (radians).
    Atan,
    /// Inverse hyperbolic tangent of each sample value.
    Atanh,
    /// Round each sample value up to the nearest integer.
    Ceil,
    /// Cosine of each sample value (radians).
    Cos,
    /// Hyperbolic cosine of each sample value.
    Cosh,
    /// Convert each sample value from radians to degrees.
    Deg,
    /// Exponential function: e raised to the power of the sample value.
    Exp,
    /// Round each sample value down to the nearest integer.
    Floor,
    Ln,
    /// Base-2 logarithm of each sample value.
    Log2,
    /// Base-10 logarithm of each sample value.
    Log10,
    /// Convert each sample value from degrees to radians.
    Rad,
    /// Round each value to the nearest multiple of `to_nearest`.
    Round {
        to_nearest: f64,
    },
    /// Returns the sign of each sample: -1 if negative, 0 if zero, 1 if positive.
    Sgn,
    /// Sine of each sample value (radians).
    Sin,
    /// Hyperbolic sine of each sample value.
    Sinh,
    /// Square root of each sample value. Returns NaN for negative inputs.
    Sqrt,
    /// Tangent of each sample value (radians).
    Tan,
    /// Hyperbolic tangent of each sample value.
    Tanh,
}

impl fmt::Display for InstantFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Abs => write!(f, "abs"),
            Self::Acos => write!(f, "acos"),
            Self::Acosh => write!(f, "acosh"),
            Self::Asin => write!(f, "asin"),
            Self::Asinh => write!(f, "asinh"),
            Self::Atan => write!(f, "atan"),
            Self::Atanh => write!(f, "atanh"),
            Self::Ceil => write!(f, "ceil"),
            Self::Cos => write!(f, "cos"),
            Self::Cosh => write!(f, "cosh"),
            Self::Deg => write!(f, "deg"),
            Self::Exp => write!(f, "exp"),
            Self::Floor => write!(f, "floor"),
            Self::Ln => write!(f, "ln"),
            Self::Log2 => write!(f, "log2"),
            Self::Log10 => write!(f, "log10"),
            Self::Rad => write!(f, "rad"),
            Self::Round { to_nearest } => write!(f, "round(to_nearest={to_nearest})"),
            Self::Sgn => write!(f, "sgn"),
            Self::Sin => write!(f, "sin"),
            Self::Sinh => write!(f, "sinh"),
            Self::Sqrt => write!(f, "sqrt"),
            Self::Tan => write!(f, "tan"),
            Self::Tanh => write!(f, "tanh"),
        }
    }
}

impl InstantFunction {
    /// Apply the function to a single sample value.
    #[cfg(test)]
    pub fn evaluate(&self, value: f64) -> f64 {
        match self {
            Self::Abs => value.abs(),
            Self::Acos => value.acos(),
            Self::Acosh => value.acosh(),
            Self::Asin => value.asin(),
            Self::Asinh => value.asinh(),
            Self::Atan => value.atan(),
            Self::Atanh => value.atanh(),
            Self::Ceil => value.ceil(),
            Self::Cos => value.cos(),
            Self::Cosh => value.cosh(),
            Self::Deg => value.to_degrees(),
            Self::Exp => value.exp(),
            Self::Floor => value.floor(),
            Self::Ln => value.ln(),
            Self::Log2 => value.log2(),
            Self::Log10 => value.log10(),
            Self::Rad => value.to_radians(),
            Self::Round { to_nearest } => super::udf::promql_round(value, *to_nearest),
            Self::Sgn => {
                if value.is_nan() {
                    f64::NAN
                } else if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Sin => value.sin(),
            Self::Sinh => value.sinh(),
            Self::Sqrt => value.sqrt(),
            Self::Tan => value.tan(),
            Self::Tanh => value.tanh(),
        }
    }

    /// Whether this function drops the `__name__` label from its output.
    ///
    /// All math instant functions drop the metric name because the result is
    /// a different quantity from the input.
    #[cfg(test)]
    pub fn drops_metric_name(&self) -> bool {
        true
    }
}

// f64 doesn't implement Eq/Hash, so we implement them manually using bit representation.
impl PartialEq for InstantFunction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Abs, Self::Abs)
            | (Self::Acos, Self::Acos)
            | (Self::Acosh, Self::Acosh)
            | (Self::Asin, Self::Asin)
            | (Self::Asinh, Self::Asinh)
            | (Self::Atan, Self::Atan)
            | (Self::Atanh, Self::Atanh)
            | (Self::Ceil, Self::Ceil)
            | (Self::Cos, Self::Cos)
            | (Self::Cosh, Self::Cosh)
            | (Self::Deg, Self::Deg)
            | (Self::Exp, Self::Exp)
            | (Self::Floor, Self::Floor)
            | (Self::Ln, Self::Ln)
            | (Self::Log2, Self::Log2)
            | (Self::Log10, Self::Log10)
            | (Self::Rad, Self::Rad)
            | (Self::Sgn, Self::Sgn)
            | (Self::Sin, Self::Sin)
            | (Self::Sinh, Self::Sinh)
            | (Self::Sqrt, Self::Sqrt)
            | (Self::Tan, Self::Tan)
            | (Self::Tanh, Self::Tanh) => true,
            (Self::Round { to_nearest: a }, Self::Round { to_nearest: b }) => {
                a.to_bits() == b.to_bits()
            }
            _ => false,
        }
    }
}
impl Eq for InstantFunction {}

impl Hash for InstantFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let Self::Round { to_nearest } = self {
            to_nearest.to_bits().hash(state);
        }
    }
}

/// Look up an instant function by name, given any extra scalar arguments beyond the
/// first (vector) argument.
///
/// Returns `None` if the name is not a known instant function.
pub(crate) fn lookup_instant_function(name: &str, extra_args: &[f64]) -> Option<InstantFunction> {
    match name {
        "abs" => Some(InstantFunction::Abs),
        "acos" => Some(InstantFunction::Acos),
        "acosh" => Some(InstantFunction::Acosh),
        "asin" => Some(InstantFunction::Asin),
        "asinh" => Some(InstantFunction::Asinh),
        "atan" => Some(InstantFunction::Atan),
        "atanh" => Some(InstantFunction::Atanh),
        "ceil" => Some(InstantFunction::Ceil),
        "cos" => Some(InstantFunction::Cos),
        "cosh" => Some(InstantFunction::Cosh),
        "deg" => Some(InstantFunction::Deg),
        "exp" => Some(InstantFunction::Exp),
        "floor" => Some(InstantFunction::Floor),
        "ln" => Some(InstantFunction::Ln),
        "log2" => Some(InstantFunction::Log2),
        "log10" => Some(InstantFunction::Log10),
        "rad" => Some(InstantFunction::Rad),
        "round" => {
            let to_nearest = extra_args.first().copied().unwrap_or(1.0);
            Some(InstantFunction::Round { to_nearest })
        }
        "sgn" => Some(InstantFunction::Sgn),
        "sin" => Some(InstantFunction::Sin),
        "sinh" => Some(InstantFunction::Sinh),
        "sqrt" => Some(InstantFunction::Sqrt),
        "tan" => Some(InstantFunction::Tan),
        "tanh" => Some(InstantFunction::Tanh),
        _ => None,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // --- abs tests ---

    #[test]
    fn test_abs_positive() {
        assert_eq!(InstantFunction::Abs.evaluate(3.5), 3.5);
    }

    #[test]
    fn test_abs_negative() {
        assert_eq!(InstantFunction::Abs.evaluate(-3.5), 3.5);
    }

    #[test]
    fn test_abs_zero() {
        assert_eq!(InstantFunction::Abs.evaluate(0.0), 0.0);
    }

    #[test]
    fn test_abs_large() {
        assert_eq!(InstantFunction::Abs.evaluate(-1e300), 1e300);
    }

    #[test]
    fn test_abs_nan_stays_nan() {
        assert!(InstantFunction::Abs.evaluate(f64::NAN).is_nan());
    }

    #[test]
    fn test_lookup_abs() {
        assert!(matches!(
            lookup_instant_function("abs", &[]),
            Some(InstantFunction::Abs)
        ));
    }

    // --- ceil tests ---

    #[test]
    fn test_ceil_positive_fractional() {
        assert_eq!(InstantFunction::Ceil.evaluate(1.2), 2.0);
        assert_eq!(InstantFunction::Ceil.evaluate(0.1), 1.0);
        assert_eq!(InstantFunction::Ceil.evaluate(99.999), 100.0);
    }

    #[test]
    fn test_ceil_exact_integer() {
        assert_eq!(InstantFunction::Ceil.evaluate(1.0), 1.0);
        assert_eq!(InstantFunction::Ceil.evaluate(0.0), 0.0);
        assert_eq!(InstantFunction::Ceil.evaluate(42.0), 42.0);
    }

    #[test]
    fn test_ceil_negative_fractional() {
        assert_eq!(InstantFunction::Ceil.evaluate(-1.2), -1.0);
        assert_eq!(InstantFunction::Ceil.evaluate(-0.1), 0.0);
    }

    #[test]
    fn test_ceil_negative_exact_integer() {
        assert_eq!(InstantFunction::Ceil.evaluate(-1.0), -1.0);
        assert_eq!(InstantFunction::Ceil.evaluate(-42.0), -42.0);
    }

    #[test]
    fn test_ceil_nan() {
        assert!(InstantFunction::Ceil.evaluate(f64::NAN).is_nan());
    }

    #[test]
    fn test_ceil_infinity() {
        assert_eq!(InstantFunction::Ceil.evaluate(f64::INFINITY), f64::INFINITY);
        assert_eq!(
            InstantFunction::Ceil.evaluate(f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_lookup_ceil() {
        assert!(matches!(
            lookup_instant_function("ceil", &[]),
            Some(InstantFunction::Ceil)
        ));
    }

    // --- exp tests ---

    #[test]
    fn test_exp_zero() {
        // e^0 = 1
        let result = InstantFunction::Exp.evaluate(0.0);
        assert!(
            (result - 1.0).abs() < f64::EPSILON,
            "expected 1.0, got {result}"
        );
    }

    #[test]
    fn test_exp_one() {
        // e^1 = e
        let result = InstantFunction::Exp.evaluate(1.0);
        assert!(
            (result - std::f64::consts::E).abs() < 1e-10,
            "expected e, got {result}"
        );
    }

    #[test]
    fn test_exp_negative() {
        // e^-1 = 1/e
        let result = InstantFunction::Exp.evaluate(-1.0);
        let expected = 1.0 / std::f64::consts::E;
        assert!(
            (result - expected).abs() < 1e-10,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_exp_large_positive() {
        // e^10
        let result = InstantFunction::Exp.evaluate(10.0);
        let expected = 10.0_f64.exp();
        assert!(
            (result - expected).abs() < 1e-6,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_exp_large_negative() {
        // e^-100 should be very close to 0
        let result = InstantFunction::Exp.evaluate(-100.0);
        assert!(
            result >= 0.0 && result < 1e-40,
            "expected near 0, got {result}"
        );
    }

    #[test]
    fn test_lookup_exp() {
        assert!(matches!(
            lookup_instant_function("exp", &[]),
            Some(InstantFunction::Exp)
        ));
    }

    // --- floor tests ---

    #[test]
    fn test_floor_positive() {
        assert_eq!(InstantFunction::Floor.evaluate(3.7), 3.0);
    }

    #[test]
    fn test_floor_negative() {
        assert_eq!(InstantFunction::Floor.evaluate(-3.2), -4.0);
    }

    #[test]
    fn test_floor_exact_integer() {
        assert_eq!(InstantFunction::Floor.evaluate(5.0), 5.0);
    }

    #[test]
    fn test_floor_zero() {
        assert_eq!(InstantFunction::Floor.evaluate(0.0), 0.0);
    }

    #[test]
    fn test_floor_nan() {
        assert!(InstantFunction::Floor.evaluate(f64::NAN).is_nan());
    }

    #[test]
    fn test_floor_infinity() {
        assert_eq!(
            InstantFunction::Floor.evaluate(f64::INFINITY),
            f64::INFINITY
        );
    }

    #[test]
    fn test_lookup_floor() {
        assert_eq!(
            lookup_instant_function("floor", &[]),
            Some(InstantFunction::Floor)
        );
    }

    // --- ln tests ---

    #[test]
    fn test_ln_positive() {
        let result = InstantFunction::Ln.evaluate(1.0);
        assert!(
            result.abs() < f64::EPSILON,
            "ln(1) should be 0, got {result}"
        );
    }

    #[test]
    fn test_ln_e() {
        let result = InstantFunction::Ln.evaluate(std::f64::consts::E);
        assert!(
            (result - 1.0).abs() < f64::EPSILON,
            "ln(e) should be 1, got {result}"
        );
    }

    #[test]
    fn test_ln_zero() {
        let result = InstantFunction::Ln.evaluate(0.0);
        assert!(
            result.is_infinite() && result.is_sign_negative(),
            "ln(0) should be -Inf, got {result}"
        );
    }

    #[test]
    fn test_ln_negative() {
        let result = InstantFunction::Ln.evaluate(-1.0);
        assert!(result.is_nan(), "ln(-1) should be NaN, got {result}");
    }

    #[test]
    fn test_ln_infinity() {
        let result = InstantFunction::Ln.evaluate(f64::INFINITY);
        assert!(
            result.is_infinite() && result.is_sign_positive(),
            "ln(+Inf) should be +Inf, got {result}"
        );
    }

    #[test]
    fn test_ln_lookup() {
        assert!(matches!(
            lookup_instant_function("ln", &[]),
            Some(InstantFunction::Ln)
        ));
    }

    #[test]
    fn test_ln_drops_metric_name() {
        assert!(InstantFunction::Ln.drops_metric_name());
    }

    #[test]
    fn test_ln_display() {
        assert_eq!(InstantFunction::Ln.to_string(), "ln");
    }

    // --- log2 tests ---

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
        let result = InstantFunction::Log2.evaluate(0.0);
        assert!(
            result.is_infinite() && result.is_sign_negative(),
            "log2(0) should be -inf, got {result}"
        );
    }

    #[test]
    fn test_log2_negative() {
        let result = InstantFunction::Log2.evaluate(-1.0);
        assert!(result.is_nan(), "log2(-1) should be NaN, got {result}");
    }

    #[test]
    fn test_lookup_log2() {
        assert!(matches!(
            lookup_instant_function("log2", &[]),
            Some(InstantFunction::Log2)
        ));
    }

    // --- round tests ---

    #[test]
    fn test_log10_basic() {
        assert!((InstantFunction::Log10.evaluate(100.0) - 2.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Log10.evaluate(1000.0) - 3.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Log10.evaluate(1.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_log10_fractional() {
        // log10(0.1) = -1
        assert!((InstantFunction::Log10.evaluate(0.1) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_log10_zero() {
        // log10(0) = -inf per IEEE 754
        assert!(InstantFunction::Log10.evaluate(0.0).is_infinite());
        assert!(InstantFunction::Log10.evaluate(0.0) < 0.0);
    }

    #[test]
    fn test_log10_negative() {
        // log10 of a negative number is NaN
        assert!(InstantFunction::Log10.evaluate(-1.0).is_nan());
    }

    #[test]
    fn test_lookup_log10() {
        assert!(matches!(
            lookup_instant_function("log10", &[]),
            Some(InstantFunction::Log10)
        ));
    }

    #[test]
    fn test_round_default_to_nearest_one() {
        let f = InstantFunction::Round { to_nearest: 1.0 };
        assert!((f.evaluate(2.3) - 2.0).abs() < f64::EPSILON, "2.3 -> 2");
        assert!((f.evaluate(2.5) - 3.0).abs() < f64::EPSILON, "2.5 -> 3");
        assert!((f.evaluate(2.7) - 3.0).abs() < f64::EPSILON, "2.7 -> 3");
        assert!(
            (f.evaluate(-1.5) - (-1.0)).abs() < f64::EPSILON,
            "-1.5 -> -1"
        );
        assert!((f.evaluate(0.0) - 0.0).abs() < f64::EPSILON, "0 -> 0");
    }

    #[test]
    fn test_round_to_nearest_point_one() {
        let f = InstantFunction::Round { to_nearest: 0.1 };
        let result = f.evaluate(1.24);
        assert!((result - 1.2).abs() < 1e-10, "expected 1.2, got {result}");
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
        let f = InstantFunction::Round { to_nearest: 0.0 };
        assert!((f.evaluate(3.7) - 3.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_round_negative_values() {
        let f = InstantFunction::Round { to_nearest: 1.0 };
        assert!(
            (f.evaluate(-2.3) - (-2.0)).abs() < f64::EPSILON,
            "-2.3 -> -2"
        );
        assert!(
            (f.evaluate(-2.7) - (-3.0)).abs() < f64::EPSILON,
            "-2.7 -> -3"
        );
    }

    // --- lookup tests ---

    #[test]
    fn test_lookup_instant_function_round_no_args() {
        let f = lookup_instant_function("round", &[]).unwrap();
        match f {
            InstantFunction::Round { to_nearest } => {
                assert!((to_nearest - 1.0).abs() < f64::EPSILON)
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn test_lookup_instant_function_round_with_arg() {
        let f = lookup_instant_function("round", &[0.5]).unwrap();
        match f {
            InstantFunction::Round { to_nearest } => {
                assert!((to_nearest - 0.5).abs() < f64::EPSILON)
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn test_lookup_instant_function_unknown() {
        assert!(lookup_instant_function("unknown_func", &[]).is_none());
    }

    #[test]
    fn test_sqrt_positive() {
        assert!((InstantFunction::Sqrt.evaluate(4.0) - 2.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Sqrt.evaluate(9.0) - 3.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Sqrt.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((InstantFunction::Sqrt.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sqrt_negative_is_nan() {
        assert!(InstantFunction::Sqrt.evaluate(-1.0).is_nan());
        assert!(InstantFunction::Sqrt.evaluate(-100.0).is_nan());
    }

    #[test]
    fn test_sqrt_fractional() {
        let result = InstantFunction::Sqrt.evaluate(2.0);
        assert!((result - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_lookup_sqrt() {
        assert!(matches!(
            lookup_instant_function("sqrt", &[]),
            Some(InstantFunction::Sqrt)
        ));
    }
}
