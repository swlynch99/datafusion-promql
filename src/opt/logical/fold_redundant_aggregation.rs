use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

/// Optimizer rule that folds redundant nested aggregations for idempotent
/// functions.
///
/// When an `Aggregate` node wraps another `Aggregate` node with the same
/// grouping columns and the same idempotent aggregation function (sum, min, or
/// max), the outer aggregate is redundant and can be removed.
///
/// For example, `sum(sum(x)) GROUP BY t` → `sum(x) GROUP BY t`.
#[derive(Debug)]
pub struct FoldRedundantAggregation;

/// Set of aggregation functions that are idempotent when nested:
/// applying them twice yields the same result as applying once.
const IDEMPOTENT_FUNCTIONS: &[&str] = &["sum", "min", "max"];

impl OptimizerRule for FoldRedundantAggregation {
    fn name(&self) -> &str {
        "fold_redundant_aggregation"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Aggregate(ref outer) = plan else {
            return Ok(Transformed::no(plan));
        };

        let LogicalPlan::Aggregate(ref inner) = *outer.input else {
            return Ok(Transformed::no(plan));
        };

        // Both must have the same grouping columns.
        if outer.group_expr != inner.group_expr {
            return Ok(Transformed::no(plan));
        }

        // Both must have exactly one aggregation expression.
        if outer.aggr_expr.len() != 1 || inner.aggr_expr.len() != 1 {
            return Ok(Transformed::no(plan));
        }

        let Some(outer_name) = agg_func_name(&outer.aggr_expr[0]) else {
            return Ok(Transformed::no(plan));
        };
        let Some(inner_name) = agg_func_name(&inner.aggr_expr[0]) else {
            return Ok(Transformed::no(plan));
        };

        // Functions must match and be in the idempotent set.
        if outer_name != inner_name || !IDEMPOTENT_FUNCTIONS.contains(&outer_name) {
            return Ok(Transformed::no(plan));
        }

        // Fold: replace the outer+inner pair with just the inner aggregate.
        let LogicalPlan::Aggregate(outer) = plan else {
            unreachable!();
        };
        Ok(Transformed::yes(std::sync::Arc::unwrap_or_clone(
            outer.input,
        )))
    }
}

/// Extract the aggregation function name from an expression, looking through
/// an `Alias` wrapper if present.
fn agg_func_name(expr: &Expr) -> Option<&str> {
    let inner = match expr {
        Expr::Alias(a) => a.expr.as_ref(),
        other => other,
    };
    match inner {
        Expr::AggregateFunction(af) => Some(af.func.name()),
        _ => None,
    }
}
