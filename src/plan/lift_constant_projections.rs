use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::Column;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Projection, Union};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::lit;

/// Optimizer rule that lifts constant literal projections shared by all
/// branches of a `Union` into a single projection above the union.
///
/// For example, given:
///
/// ```text
/// Union
///   Projection [ts, value, 'cpu' AS __name__, 'host1' AS host]
///     Scan ...
///   Projection [ts, value, 'cpu' AS __name__, 'host2' AS host]
///     Scan ...
/// ```
///
/// If `__name__` has the same literal (`'cpu'`) in every branch, it is lifted:
///
/// ```text
/// Projection [ts, value, 'cpu' AS __name__, host]
///   Union
///     Projection [ts, value, 'host1' AS host]
///       Scan ...
///     Projection [ts, value, 'host2' AS host]
///       Scan ...
/// ```
///
/// This avoids materializing redundant constant columns in every branch of
/// the union, which is beneficial for the wide-to-long normalization that
/// produces a UNION ALL with many branches sharing the same metric name.
#[derive(Debug)]
pub struct LiftConstantProjections;

impl OptimizerRule for LiftConstantProjections {
    fn name(&self) -> &str {
        "lift_constant_projections"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Union(ref union) = plan else {
            return Ok(Transformed::no(plan));
        };

        if union.inputs.len() < 2 {
            return Ok(Transformed::no(plan));
        }

        // All branches must be projections.
        let projections: Vec<&Projection> = union
            .inputs
            .iter()
            .filter_map(|input| match input.as_ref() {
                LogicalPlan::Projection(p) => Some(p),
                _ => None,
            })
            .collect();

        if projections.len() != union.inputs.len() {
            return Ok(Transformed::no(plan));
        }

        let ncols = projections[0].expr.len();
        if projections.iter().any(|p| p.expr.len() != ncols) {
            return Ok(Transformed::no(plan));
        }

        // For each column position, check if ALL branches project the same
        // literal value. We store Some((scalar_value, alias_name)) for
        // constant columns, None otherwise.
        let mut constant_cols: Vec<Option<(datafusion::common::ScalarValue, String)>> =
            Vec::with_capacity(ncols);

        for col_idx in 0..ncols {
            constant_cols.push(find_shared_constant(&projections, col_idx));
        }

        // Nothing to lift if there are no shared constants.
        if constant_cols.iter().all(|c| c.is_none()) {
            return Ok(Transformed::no(plan));
        }

        // Need at least one non-constant column to form a valid inner union.
        let non_constant_count = constant_cols.iter().filter(|c| c.is_none()).count();
        if non_constant_count == 0 {
            return Ok(Transformed::no(plan));
        }

        // Build new inner projections without the constant columns.
        let LogicalPlan::Union(union) = plan else {
            unreachable!();
        };

        let mut new_inputs: Vec<Arc<LogicalPlan>> = Vec::with_capacity(union.inputs.len());
        for input in union.inputs {
            let proj = match Arc::try_unwrap(input) {
                Ok(LogicalPlan::Projection(p)) => p,
                Ok(_) => unreachable!(),
                Err(arc) => match arc.as_ref() {
                    LogicalPlan::Projection(p) => p.clone(),
                    _ => unreachable!(),
                },
            };

            let new_exprs: Vec<Expr> = proj
                .expr
                .into_iter()
                .enumerate()
                .filter(|(i, _)| constant_cols[*i].is_none())
                .map(|(_, e)| e)
                .collect();

            let new_plan = LogicalPlanBuilder::from(Arc::unwrap_or_clone(proj.input))
                .project(new_exprs)?
                .build()?;
            new_inputs.push(Arc::new(new_plan));
        }

        let new_union = LogicalPlan::Union(Union::try_new_with_loose_types(new_inputs)?);

        // Build outer projection that reassembles the original column order:
        // constant columns become literals, others reference the union output.
        let union_schema = new_union.schema().clone();
        let mut outer_exprs: Vec<Expr> = Vec::with_capacity(ncols);
        let mut union_col_idx = 0;

        for col_idx in 0..ncols {
            if let Some((ref value, ref name)) = constant_cols[col_idx] {
                outer_exprs.push(lit(value.clone()).alias(name.as_str()));
            } else {
                let (qualifier, field) = union_schema.qualified_field(union_col_idx);
                let col_expr =
                    Expr::Column(Column::from((qualifier, field.as_ref()))).alias(field.name());
                outer_exprs.push(col_expr);
                union_col_idx += 1;
            }
        }

        let result = LogicalPlanBuilder::from(new_union)
            .project(outer_exprs)?
            .build()?;

        Ok(Transformed::yes(result))
    }
}

/// Check if all branches share the same literal value at `col_idx`.
///
/// Returns `Some((scalar, alias_name))` if every branch has
/// `lit(scalar).alias(name)` (or a bare `Literal`) with the same scalar value.
/// Returns `None` if the expressions differ or are not literals.
fn find_shared_constant(
    projections: &[&Projection],
    col_idx: usize,
) -> Option<(datafusion::common::ScalarValue, String)> {
    let (first_value, first_name) = extract_literal(&projections[0].expr[col_idx])?;

    for proj in &projections[1..] {
        let (value, _) = extract_literal(&proj.expr[col_idx])?;
        if value != first_value {
            return None;
        }
    }

    Some((first_value, first_name))
}

/// Extract a scalar literal from an expression, returning `(value, column_name)`.
///
/// Handles both `Expr::Alias(Alias { expr: Literal(..), name, .. })` and bare
/// `Expr::Literal(..)`.
fn extract_literal(expr: &Expr) -> Option<(datafusion::common::ScalarValue, String)> {
    match expr {
        Expr::Alias(alias) => match alias.expr.as_ref() {
            Expr::Literal(value, _metadata) => Some((value.clone(), alias.name.clone())),
            _ => None,
        },
        Expr::Literal(value, _metadata) => Some((value.clone(), value.to_string())),
        _ => None,
    }
}
