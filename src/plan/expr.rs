use datafusion::logical_expr::{Extension, LogicalPlan};
use promql_parser::parser::Expr;

use crate::datasource::MetricSource;
use crate::error::{PromqlError, Result};
use crate::node::InstantVectorEval;
use crate::types::{TimeRange, DEFAULT_LOOKBACK_MS};

use super::selector::plan_vector_selector;

/// Translate a promql-parser AST `Expr` into a DataFusion `LogicalPlan`.
///
/// For Phase 1, only `VectorSelector` and literals are supported.
pub(crate) async fn plan_expr(
    expr: &Expr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    eval_ts_ms: Option<i64>,
) -> Result<LogicalPlan> {
    match expr {
        Expr::VectorSelector(vs) => {
            let (child_plan, label_columns) =
                plan_vector_selector(vs, source, time_range).await?;

            // Wrap in InstantVectorEval for step alignment.
            let node = if let Some(ts) = eval_ts_ms {
                InstantVectorEval::instant(child_plan, ts, DEFAULT_LOOKBACK_MS, label_columns)
            } else {
                return Err(PromqlError::NotImplemented(
                    "range query step evaluation not yet implemented".into(),
                ));
            };

            Ok(LogicalPlan::Extension(Extension {
                node: std::sync::Arc::new(node),
            }))
        }

        Expr::NumberLiteral(_) | Expr::StringLiteral(_) => Err(PromqlError::NotImplemented(
            "scalar/string literals as top-level query not yet implemented".into(),
        )),

        Expr::Paren(paren) => {
            Box::pin(plan_expr(&paren.expr, source, time_range, eval_ts_ms)).await
        }

        _ => Err(PromqlError::NotImplemented(format!(
            "expression type not yet supported: {expr:?}"
        ))),
    }
}
