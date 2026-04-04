use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, cast};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::node::DateTimeFunctionNode;

/// Optimizer rule that rewrites `DateTimeFunctionNode` extension nodes into
/// standard DataFusion `Projection` plans with the stored UDF expression
/// applied to produce the `value` column from the `timestamp` column.
#[derive(Debug)]
pub struct DateTimeFuncToProjection;

impl OptimizerRule for DateTimeFuncToProjection {
    fn name(&self) -> &str {
        "datetime_func_to_projection"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(ext) = &plan else {
            return Ok(Transformed::no(plan));
        };

        let Some(eval) = ext.node.as_any().downcast_ref::<DateTimeFunctionNode>() else {
            return Ok(Transformed::no(plan));
        };

        let child = eval.input.clone();
        let child_schema = child.schema();

        // Build projection expressions matching the output schema.
        let mut exprs: Vec<Expr> = Vec::new();
        for field in eval.output_schema.fields() {
            let name = field.name();
            let (qualifier, child_field) =
                child_schema.qualified_field_with_name(None, name.as_str())?;
            let col_expr = Expr::Column(datafusion::common::Column::from((qualifier, child_field)));

            if name == "value" {
                // Use the stored function expression, replacing the generic
                // col("timestamp") reference with the properly qualified column.
                let (ts_qualifier, ts_field) =
                    child_schema.qualified_field_with_name(None, "timestamp")?;
                let ts_col =
                    Expr::Column(datafusion::common::Column::from((ts_qualifier, ts_field)));
                let func_expr = replace_timestamp_col(&eval.func_expr, &ts_col);
                let casted = cast(func_expr, field.data_type().clone());
                exprs.push(casted.alias("value"));
            } else {
                // Pass through, aliasing to strip qualifier.
                exprs.push(col_expr.alias(name.as_str()));
            }
        }

        let new_plan = LogicalPlanBuilder::from(child).project(exprs)?.build()?;

        Ok(Transformed::yes(new_plan))
    }
}

/// Replace any `col("timestamp")` references in `expr` with the given
/// `replacement` expression. This handles the case where the stored
/// func_expr uses an unqualified `col("timestamp")` but the child schema
/// may require a qualified column reference.
fn replace_timestamp_col(expr: &Expr, replacement: &Expr) -> Expr {
    expr.clone()
        .transform(|e| {
            if let Expr::Column(ref c) = e
                && c.name == "timestamp"
                && c.relation.is_none()
            {
                return Ok(datafusion::common::tree_node::Transformed::yes(
                    replacement.clone(),
                ));
            }
            Ok(datafusion::common::tree_node::Transformed::no(e))
        })
        .expect("transform should not fail")
        .data
}
