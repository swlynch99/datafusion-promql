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
    Stddev,
    Stdvar,
    Group,
    TopK,
    BottomK,
    Quantile,
    CountValues,
    LimitK,
    LimitRatio,
}

impl fmt::Display for AggregateFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Avg => write!(f, "avg"),
            Self::Count => write!(f, "count"),
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
            Self::Stddev => write!(f, "stddev"),
            Self::Stdvar => write!(f, "stdvar"),
            Self::Group => write!(f, "group"),
            Self::TopK => write!(f, "topk"),
            Self::BottomK => write!(f, "bottomk"),
            Self::Quantile => write!(f, "quantile"),
            Self::CountValues => write!(f, "count_values"),
            Self::LimitK => write!(f, "limitk"),
            Self::LimitRatio => write!(f, "limit_ratio"),
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
        "stddev" => Ok(AggregateFunction::Stddev),
        "stdvar" => Ok(AggregateFunction::Stdvar),
        "group" => Ok(AggregateFunction::Group),
        "topk" => Ok(AggregateFunction::TopK),
        "bottomk" => Ok(AggregateFunction::BottomK),
        "quantile" => Ok(AggregateFunction::Quantile),
        "count_values" => Ok(AggregateFunction::CountValues),
        "limitk" => Ok(AggregateFunction::LimitK),
        "limit_ratio" => Ok(AggregateFunction::LimitRatio),
        other => Err(PromqlError::NotImplemented(format!(
            "aggregation operator not yet supported: {other}"
        ))),
    }
}
