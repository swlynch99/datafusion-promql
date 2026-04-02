use std::collections::HashSet;
use std::sync::Arc;

use datafusion::logical_expr::{Extension, LogicalPlan};
use promql_parser::parser::{self, Expr, LabelModifier};

use crate::datasource::MetricSource;
use crate::error::{PromqlError, Result};
use crate::func::{lookup_aggregate_function, lookup_range_function};
use crate::node::{
    AggregateEval, BinaryEval, InstantVectorEval, MatchCardinality, RangeVectorEval,
    ScalarBinaryEval, VectorMatching, convert_binary_op,
};
use crate::types::{DEFAULT_LOOKBACK_MS, TimeRange};

use super::selector::plan_vector_selector;

/// Parameters controlling how evaluation timestamps are generated.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EvalParams {
    /// For instant queries: the single evaluation timestamp (ms).
    /// `None` for range queries (timestamps generated from start/end/step).
    pub eval_ts_ms: Option<i64>,
    pub start_ms: i64,
    pub end_ms: i64,
    pub step_ms: i64,
}

/// Extract label column names from a schema (everything except timestamp/value).
fn label_columns_from_schema(schema: &datafusion::common::DFSchemaRef) -> Vec<String> {
    schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .filter(|n| n != "timestamp" && n != "value")
        .collect()
}

/// Translate a promql-parser AST `Expr` into a DataFusion `LogicalPlan`.
pub(crate) async fn plan_expr(
    expr: &Expr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    match expr {
        Expr::VectorSelector(vs) => {
            let (child_plan, label_columns) =
                plan_vector_selector(vs, source, time_range, 0).await?;

            let node = if let Some(ts) = params.eval_ts_ms {
                InstantVectorEval::instant(child_plan, ts, DEFAULT_LOOKBACK_MS, label_columns)
            } else {
                InstantVectorEval::range(
                    child_plan,
                    params.start_ms,
                    params.end_ms,
                    params.step_ms,
                    DEFAULT_LOOKBACK_MS,
                    label_columns,
                )
            };

            Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            }))
        }

        Expr::Call(call) => plan_call(call, source, time_range, params).await,

        Expr::MatrixSelector(_) => Err(PromqlError::Plan(
            "bare matrix selector is not allowed as a top-level expression; \
             use it inside a range function like rate()"
                .into(),
        )),

        Expr::NumberLiteral(_) | Expr::StringLiteral(_) => Err(PromqlError::NotImplemented(
            "scalar/string literals as top-level query not yet implemented".into(),
        )),

        Expr::Paren(paren) => Box::pin(plan_expr(&paren.expr, source, time_range, params)).await,

        Expr::Aggregate(agg) => plan_aggregate(agg, source, time_range, params).await,

        Expr::Binary(bin) => plan_binary(bin, source, time_range, params).await,

        Expr::Unary(unary) => plan_unary(unary, source, time_range, params).await,

        _ => Err(PromqlError::NotImplemented(format!(
            "expression type not yet supported: {expr:?}"
        ))),
    }
}

/// Plan a function call expression.
async fn plan_call(
    call: &parser::Call,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    let func_name = call.func.name;

    // Check if this is a range vector function.
    if let Some(range_func) = lookup_range_function(func_name) {
        // Range functions expect exactly one argument: a MatrixSelector.
        if call.args.args.len() != 1 {
            return Err(PromqlError::Plan(format!(
                "{func_name}() requires exactly 1 argument, got {}",
                call.args.args.len()
            )));
        }

        let arg = &call.args.args[0];
        let matrix = match arg.as_ref() {
            Expr::MatrixSelector(ms) => ms,
            _ => {
                return Err(PromqlError::Plan(format!(
                    "{func_name}() requires a range vector (matrix selector) argument"
                )));
            }
        };

        let range_ms = matrix.range.as_millis() as i64;

        // Plan the inner vector selector with extra range expansion.
        let (child_plan, label_columns) =
            plan_vector_selector(&matrix.vs, source, time_range, range_ms).await?;

        // Wrap in RangeVectorEval node.
        let node = if let Some(ts) = params.eval_ts_ms {
            RangeVectorEval::instant(child_plan, ts, range_ms, range_func, label_columns)
        } else {
            RangeVectorEval::range(
                child_plan,
                params.start_ms,
                params.end_ms,
                params.step_ms,
                range_ms,
                range_func,
                label_columns,
            )
        };

        Ok(LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        }))
    } else {
        Err(PromqlError::NotImplemented(format!(
            "function not yet supported: {func_name}"
        )))
    }
}

/// Plan an aggregation expression (sum, avg, count, min, max).
async fn plan_aggregate(
    agg: &parser::AggregateExpr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    let func = lookup_aggregate_function(agg.op)?;

    // Recursively plan the inner expression.
    let child_plan = Box::pin(plan_expr(&agg.expr, source, time_range, params)).await?;

    // Determine grouping labels from the modifier and child schema.
    let child_label_cols = label_columns_from_schema(child_plan.schema());
    let grouping_labels = compute_grouping_labels(&agg.modifier, &child_label_cols);

    let node = AggregateEval::new(child_plan, func, grouping_labels)?;

    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    }))
}

/// Compute grouping labels from a LabelModifier and the available child label columns.
fn compute_grouping_labels(
    modifier: &Option<LabelModifier>,
    child_label_cols: &[String],
) -> Vec<String> {
    match modifier {
        Some(LabelModifier::Include(labels)) => {
            // by(...): keep only specified labels that exist in the child schema.
            // Exclude __name__ from grouping by default.
            labels
                .labels
                .iter()
                .filter(|l| child_label_cols.contains(l) && l.as_str() != "__name__")
                .cloned()
                .collect()
        }
        Some(LabelModifier::Exclude(labels)) => {
            // without(...): keep all child labels except specified ones and __name__.
            let exclude: HashSet<&str> = labels.labels.iter().map(|s| s.as_str()).collect();
            child_label_cols
                .iter()
                .filter(|l| !exclude.contains(l.as_str()) && l.as_str() != "__name__")
                .cloned()
                .collect()
        }
        None => {
            // No modifier: aggregate all into one group (no grouping labels).
            vec![]
        }
    }
}

/// Plan a binary expression.
async fn plan_binary(
    bin: &parser::BinaryExpr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    let op = convert_binary_op(bin.op)?;

    let lhs_is_scalar = matches!(bin.lhs.as_ref(), Expr::NumberLiteral(_));
    let rhs_is_scalar = matches!(bin.rhs.as_ref(), Expr::NumberLiteral(_));

    let return_bool = bin
        .modifier
        .as_ref()
        .map(|m| m.return_bool)
        .unwrap_or(false);

    match (lhs_is_scalar, rhs_is_scalar) {
        (true, true) => {
            // scalar op scalar: not yet implemented
            Err(PromqlError::NotImplemented(
                "scalar op scalar not yet supported".into(),
            ))
        }
        (true, false) => {
            // scalar op vector
            let scalar_val = extract_scalar(&bin.lhs)?;
            let rhs_plan = Box::pin(plan_expr(&bin.rhs, source, time_range, params)).await?;
            let node =
                ScalarBinaryEval::new(rhs_plan, scalar_val, op, true, return_bool)?;
            Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            }))
        }
        (false, true) => {
            // vector op scalar
            let scalar_val = extract_scalar(&bin.rhs)?;
            let lhs_plan = Box::pin(plan_expr(&bin.lhs, source, time_range, params)).await?;
            let node =
                ScalarBinaryEval::new(lhs_plan, scalar_val, op, false, return_bool)?;
            Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            }))
        }
        (false, false) => {
            // vector op vector
            let lhs_plan = Box::pin(plan_expr(&bin.lhs, source, time_range, params)).await?;
            let rhs_plan = Box::pin(plan_expr(&bin.rhs, source, time_range, params)).await?;

            let matching = extract_vector_matching(bin)?;
            let node = BinaryEval::new(lhs_plan, rhs_plan, op, return_bool, matching)?;

            Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            }))
        }
    }
}

/// Extract the scalar value from a NumberLiteral expression.
fn extract_scalar(expr: &Expr) -> Result<f64> {
    match expr {
        Expr::NumberLiteral(lit) => Ok(lit.val),
        _ => Err(PromqlError::Plan(
            "expected a number literal for scalar operand".into(),
        )),
    }
}

/// Extract VectorMatching from a BinaryExpr's modifier.
fn extract_vector_matching(bin: &parser::BinaryExpr) -> Result<VectorMatching> {
    let modifier = match &bin.modifier {
        Some(m) => m,
        None => return Ok(VectorMatching::default_matching()),
    };

    let card = match &modifier.card {
        parser::VectorMatchCardinality::OneToOne => MatchCardinality::OneToOne,
        parser::VectorMatchCardinality::ManyToOne(labels) => {
            MatchCardinality::ManyToOne(labels.labels.clone())
        }
        parser::VectorMatchCardinality::OneToMany(labels) => {
            MatchCardinality::OneToMany(labels.labels.clone())
        }
        parser::VectorMatchCardinality::ManyToMany => MatchCardinality::OneToOne,
    };

    let (on_labels, ignoring_labels) = match &modifier.matching {
        Some(LabelModifier::Include(labels)) => (Some(labels.labels.clone()), None),
        Some(LabelModifier::Exclude(labels)) => (None, Some(labels.labels.clone())),
        None => (None, None),
    };

    Ok(VectorMatching {
        card,
        on_labels,
        ignoring_labels,
    })
}

/// Plan a unary expression (negation).
async fn plan_unary(
    unary: &parser::UnaryExpr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    // Unary negation: multiply by -1
    let child_plan = Box::pin(plan_expr(&unary.expr, source, time_range, params)).await?;

    use crate::node::BinaryOp;
    let node = ScalarBinaryEval::new(child_plan, -1.0, BinaryOp::Mul, true, false)?;
    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    }))
}
