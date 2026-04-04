use std::sync::Arc;

use datafusion::common::Column;
use datafusion::common::tree_node::TreeNode;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Projection, Union};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::lit;

/// Optimizer rule that lifts shared constant literal projections out of unions
/// and past sort nodes.
///
/// This rule handles two local patterns. DataFusion's bottom-up application
/// composes them to handle deeper trees automatically.
///
/// **Pattern 1 – Union with projection branches:**
///
/// ```text
/// Union                          Projection [constants + col refs]
///   Projection [cols + consts]     Union
///   Projection [cols + consts]       Projection [cols only]
///                            →       Projection [cols only]
/// ```
///
/// **Pattern 2 – Sort over a projection:**
///
/// ```text
/// Sort [key]                     Projection [constants + col refs]
///   Projection [cols + consts]     Sort [key]
///                            →       Projection [cols only]
/// ```
///
/// Together, a tree like `Sort -> Union -> [Sort -> Proj, Sort -> Proj]` is
/// simplified in multiple bottom-up passes without any special "wrapper"
/// logic.
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
        match &plan {
            LogicalPlan::Union(_) => rewrite_union(plan),
            LogicalPlan::Sort(_) => rewrite_sort(plan),
            _ => Ok(Transformed::no(plan)),
        }
    }
}

use datafusion::common::tree_node::Transformed;

/// Pattern 1: Lift shared constants out of a Union whose branches are all
/// Projections.
fn rewrite_union(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Union(ref union) = plan else {
        unreachable!();
    };

    if union.inputs.len() < 2 {
        return Ok(Transformed::no(plan));
    }

    // All branches must be projections with the same column count.
    let mut projections: Vec<&Projection> = Vec::with_capacity(union.inputs.len());
    for input in &union.inputs {
        let LogicalPlan::Projection(p) = input.as_ref() else {
            return Ok(Transformed::no(plan));
        };
        projections.push(p);
    }

    let ncols = projections[0].expr.len();
    if projections.iter().any(|p| p.expr.len() != ncols) {
        return Ok(Transformed::no(plan));
    }

    // Find which columns are the same constant literal across all branches.
    let constant_cols: Vec<Option<(datafusion::common::ScalarValue, String)>> = (0..ncols)
        .map(|i| find_shared_constant(&projections, i))
        .collect();

    if constant_cols.iter().all(|c| c.is_none()) {
        return Ok(Transformed::no(plan));
    }

    // Need at least one non-constant column to form a valid inner union.
    let non_constant_count = constant_cols.iter().filter(|c| c.is_none()).count();
    if non_constant_count == 0 {
        return Ok(Transformed::no(plan));
    }

    // Rebuild inner branches with constants stripped.
    let LogicalPlan::Union(union) = plan else {
        unreachable!();
    };

    let mut new_inputs: Vec<Arc<LogicalPlan>> = Vec::with_capacity(union.inputs.len());
    for input in union.inputs {
        let LogicalPlan::Projection(proj) = Arc::unwrap_or_clone(input) else {
            unreachable!();
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

    let inner_plan = LogicalPlan::Union(Union::try_new_with_loose_types(new_inputs)?);
    let result = build_outer_projection(inner_plan, &constant_cols)?;
    Ok(Transformed::yes(result))
}

/// Pattern 2: Lift constants from a Projection through a Sort that doesn't
/// reference them.
fn rewrite_sort(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Sort(ref sort) = plan else {
        unreachable!();
    };

    let LogicalPlan::Projection(ref proj) = *sort.input else {
        return Ok(Transformed::no(plan));
    };

    // Identify constant literals in the projection.
    let constant_cols: Vec<Option<(datafusion::common::ScalarValue, String)>> =
        proj.expr.iter().map(extract_literal).collect();

    if constant_cols.iter().all(|c| c.is_none()) {
        return Ok(Transformed::no(plan));
    }

    // Need at least one non-constant column.
    let non_constant_count = constant_cols.iter().filter(|c| c.is_none()).count();
    if non_constant_count == 0 {
        return Ok(Transformed::no(plan));
    }

    // The sort must not reference any constant columns.
    let constant_names: Vec<&str> = constant_cols
        .iter()
        .filter_map(|c| c.as_ref().map(|(_, name)| name.as_str()))
        .collect();

    for sort_expr in &sort.expr {
        if expr_references_columns(&sort_expr.expr, &constant_names) {
            return Ok(Transformed::no(plan));
        }
    }

    // Rebuild: Proj(constants + refs) -> Sort -> Proj(non-constants)
    let LogicalPlan::Sort(sort) = plan else {
        unreachable!();
    };
    let LogicalPlan::Projection(proj) = Arc::unwrap_or_clone(sort.input) else {
        unreachable!();
    };

    let new_exprs: Vec<Expr> = proj
        .expr
        .into_iter()
        .enumerate()
        .filter(|(i, _)| constant_cols[*i].is_none())
        .map(|(_, e)| e)
        .collect();

    let new_proj = LogicalPlanBuilder::from(Arc::unwrap_or_clone(proj.input))
        .project(new_exprs)?
        .build()?;

    let new_sort = LogicalPlan::Sort(datafusion::logical_expr::Sort {
        expr: sort.expr,
        input: Arc::new(new_proj),
        fetch: sort.fetch,
    });

    let result = build_outer_projection(new_sort, &constant_cols)?;
    Ok(Transformed::yes(result))
}

/// Build an outer projection that reassembles the original column order:
/// constant columns become literals, others reference the inner output.
fn build_outer_projection(
    inner_plan: LogicalPlan,
    constant_cols: &[Option<(datafusion::common::ScalarValue, String)>],
) -> Result<LogicalPlan> {
    let ncols = constant_cols.len();
    let inner_schema = inner_plan.schema().clone();
    let mut outer_exprs: Vec<Expr> = Vec::with_capacity(ncols);
    let mut inner_col_idx = 0;

    for constant_col in constant_cols {
        if let Some((value, name)) = constant_col {
            outer_exprs.push(lit(value.clone()).alias(name.as_str()));
        } else {
            let (qualifier, field) = inner_schema.qualified_field(inner_col_idx);
            let col_expr =
                Expr::Column(Column::from((qualifier, field.as_ref()))).alias(field.name());
            outer_exprs.push(col_expr);
            inner_col_idx += 1;
        }
    }

    LogicalPlanBuilder::from(inner_plan)
        .project(outer_exprs)?
        .build()
}

/// Check if an expression references any of the given column names.
fn expr_references_columns(expr: &Expr, column_names: &[&str]) -> bool {
    let mut references = false;
    expr.apply(|e| {
        if let Expr::Column(col) = e
            && column_names.contains(&col.name.as_str())
        {
            references = true;
            return Ok(datafusion::common::tree_node::TreeNodeRecursion::Stop);
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .expect("infallible closure should not fail");
    references
}

/// Check if all branches share the same literal value at `col_idx`.
fn find_shared_constant(
    projections: &[&Projection],
    col_idx: usize,
) -> Option<(datafusion::common::ScalarValue, String)> {
    let first_value = extract_literal_value(&projections[0].expr[col_idx])?;

    for proj in &projections[1..] {
        let value = extract_literal_value(&proj.expr[col_idx])?;
        if value != first_value {
            return None;
        }
    }

    // Only clone once we know all branches match.
    extract_literal(&projections[0].expr[col_idx])
}

/// Borrow the scalar value from a literal expression without cloning.
fn extract_literal_value(expr: &Expr) -> Option<&datafusion::common::ScalarValue> {
    match expr {
        Expr::Alias(alias) => match alias.expr.as_ref() {
            Expr::Literal(value, _) => Some(value),
            _ => None,
        },
        Expr::Literal(value, _) => Some(value),
        _ => None,
    }
}

/// Extract a scalar literal from an expression, returning `(value, column_name)`.
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
