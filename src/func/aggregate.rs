use std::fmt;

use promql_parser::parser::token::TokenType;

use crate::error::{PromqlError, Result};

/// Aggregation functions that operate on a group of sample values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AggregateFunction {
    Sum,
    Avg,
    Count,
    Min,
    Max,
}

impl fmt::Display for AggregateFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Avg => write!(f, "avg"),
            Self::Count => write!(f, "count"),
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
        }
    }
}

/// Look up an aggregate function by its token type.
pub(crate) fn lookup_aggregate_function(op: TokenType) -> Result<AggregateFunction> {
    // TokenType displays its name, use that for matching since the
    // underlying token IDs are generated and not directly accessible as constants
    // from outside the crate in a stable way.
    match op.to_string().as_str() {
        "sum" => Ok(AggregateFunction::Sum),
        "avg" => Ok(AggregateFunction::Avg),
        "count" => Ok(AggregateFunction::Count),
        "min" => Ok(AggregateFunction::Min),
        "max" => Ok(AggregateFunction::Max),
        other => Err(PromqlError::NotImplemented(format!(
            "aggregation operator not yet supported: {other}"
        ))),
    }
}

impl AggregateFunction {
    /// Evaluate the aggregation function over a slice of values.
    ///
    /// The slice must not be empty.
    pub fn evaluate(&self, values: &[f64]) -> f64 {
        match self {
            Self::Sum => values.iter().sum(),
            Self::Avg => {
                let sum: f64 = values.iter().sum();
                sum / values.len() as f64
            }
            Self::Count => values.len() as f64,
            Self::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
            Self::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        assert!((AggregateFunction::Sum.evaluate(&[1.0, 2.0, 3.0]) - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg() {
        assert!((AggregateFunction::Avg.evaluate(&[2.0, 4.0, 6.0]) - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_count() {
        assert!(
            (AggregateFunction::Count.evaluate(&[1.0, 2.0, 3.0]) - 3.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_min() {
        assert!((AggregateFunction::Min.evaluate(&[3.0, 1.0, 2.0]) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max() {
        assert!((AggregateFunction::Max.evaluate(&[3.0, 1.0, 2.0]) - 3.0).abs() < f64::EPSILON);
    }
}
