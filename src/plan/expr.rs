use std::sync::Arc;

use datafusion::logical_expr::{Extension, LogicalPlan};
use promql_parser::parser::Expr;

use crate::datasource::MetricSource;
use crate::error::{PromqlError, Result};
use crate::func::lookup_range_function;
use crate::node::{InstantVectorEval, RangeVectorEval};
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

        _ => Err(PromqlError::NotImplemented(format!(
            "expression type not yet supported: {expr:?}"
        ))),
    }
}

/// Plan a function call expression.
async fn plan_call(
    call: &promql_parser::parser::Call,
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
