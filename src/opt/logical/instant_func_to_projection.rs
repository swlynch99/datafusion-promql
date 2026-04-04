use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, cast};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::func::instant_func_to_expr;
use crate::node::InstantFuncEval;

/// Optimizer rule that rewrites `InstantFuncEval` extension nodes into
/// standard DataFusion `Projection` plans with UDF expressions on the
/// `value` column.
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

        let Some(eval) = ext.node.as_any().downcast_ref::<InstantFuncEval>() else {
            return Ok(Transformed::no(plan));
        };

        let child = eval.input.clone();
        let func = &eval.func;
        let child_schema = child.schema();

        // Build projection expressions matching the output schema.
        // We alias every column to strip table qualifiers and ensure the
        // output schema matches the original InstantFuncEval output exactly.
        let mut exprs: Vec<Expr> = Vec::new();
        for field in eval.output_schema.fields() {
            let name = field.name();
            let (qualifier, child_field) =
                child_schema.qualified_field_with_name(None, name.as_str())?;
            let col_expr = Expr::Column(datafusion::common::Column::from((qualifier, child_field)));

            if name == "value" {
                // Apply the instant function to the value column.
                // Cast to ensure non-nullable output matches the original schema.
                let func_expr = instant_func_to_expr(func, col_expr);
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
