use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, cast};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::node::InstantFunction;

/// Optimizer rule that rewrites `InstantFunction` extension nodes into
/// standard DataFusion `Projection` plans with the stored UDF expression
/// applied to the `value` column.
#[derive(Debug)]
pub struct InstantFuncToProjection;

impl OptimizerRule for InstantFuncToProjection {
    fn name(&self) -> &str {
        "instant_func_to_projection"
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

        let Some(eval) = ext.node.as_any().downcast_ref::<InstantFunction>() else {
            return Ok(Transformed::no(plan));
        };

        let child = eval.input.clone();
        let child_schema = child.schema();

        // Build projection expressions matching the output schema.
        // We alias every column to strip table qualifiers and ensure the
        // output schema matches the original InstantFunction output exactly.
        let mut exprs: Vec<Expr> = Vec::new();
        for field in eval.output_schema.fields() {
            let name = field.name();
            let (qualifier, child_field) =
                child_schema.qualified_field_with_name(None, name.as_str())?;
            let col_expr = Expr::Column(datafusion::common::Column::from((qualifier, child_field)));

            if name == "value" {
                // Use the stored function expression, replacing the generic
                // col("value") reference with the properly qualified column.
                let func_expr = replace_value_col(&eval.func_expr, &col_expr);
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

/// Replace any `col("value")` references in `expr` with the given
/// `replacement` expression. This handles the case where the stored
/// func_expr uses an unqualified `col("value")` but the child schema
/// may require a qualified column reference.
fn replace_value_col(expr: &Expr, replacement: &Expr) -> Expr {
    expr.clone()
        .transform(|e| {
            if let Expr::Column(ref c) = e
                && c.name == "value"
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
