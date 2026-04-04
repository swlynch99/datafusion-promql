use std::sync::Arc;

use datafusion::common::Column;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Projection, Sort, Union};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::lit;

/// Optimizer rule that lifts constant literal projections shared by all
/// branches of a `Union` into a single projection above the union.
///
/// The rule can lift through intermediate wrapper nodes (such as `Sort`) that
/// sit above the `Union`, inside each union branch (between the union and the
/// projection), or both. Each wrapper is only lifted past if it does not
/// reference any of the constant columns.
///
/// For example, given:
///
/// ```text
/// Sort [value ASC]
///   Union
///     Sort [ts ASC]
///       Projection [ts, value, 'cpu' AS __name__, 'host1' AS host]
///         Scan ...
///     Sort [ts ASC]
///       Projection [ts, value, 'cpu' AS __name__, 'host2' AS host]
///         Scan ...
/// ```
///
/// The shared constant `__name__` is lifted above the outer sort:
///
/// ```text
/// Projection [ts, value, 'cpu' AS __name__, host]
///   Sort [value ASC]
///     Union
///       Sort [ts ASC]
///         Projection [ts, value, 'host1' AS host]
///           Scan ...
///       Sort [ts ASC]
///         Projection [ts, value, 'host2' AS host]
///           Scan ...
/// ```
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
        // Walk down through eligible wrapper nodes to find a Union.
        let mut outer_wrappers: Vec<WrapperNode> = Vec::new();
        let mut current = &plan;

        loop {
            match current {
                LogicalPlan::Union(_) => break,
                _ => {
                    let Some(wrapper) = WrapperNode::try_from_plan(current) else {
                        return Ok(Transformed::no(plan));
                    };
                    outer_wrappers.push(wrapper);
                    current = current.inputs()[0];
                }
            }
        }

        // `current` points to the Union. Extract it from the original plan.
        let union_plan = extract_inner(plan, outer_wrappers.len());

        let LogicalPlan::Union(ref union) = union_plan else {
            unreachable!();
        };

        if union.inputs.len() < 2 {
            return Ok(Transformed::no(union_plan));
        }

        let ncols;
        let mut constant_cols: Vec<Option<(datafusion::common::ScalarValue, String)>> = Vec::new();

        // Walk each branch through wrappers to find the projection.
        // Collect (branch_wrappers, wrapper_depth) for each branch.
        // We also collect references to projections for constant analysis.
        let mut branch_wrappers_list: Vec<Vec<WrapperNode>> =
            Vec::with_capacity(union.inputs.len());
        let mut branch_depths: Vec<usize> = Vec::with_capacity(union.inputs.len());

        {
            // Scope the borrow of union for projection analysis.
            let mut projections: Vec<&Projection> = Vec::with_capacity(union.inputs.len());

            for input in &union.inputs {
                let mut branch_wrappers = Vec::new();
                let mut node = input.as_ref();

                loop {
                    match node {
                        LogicalPlan::Projection(p) => {
                            projections.push(p);
                            break;
                        }
                        _ => {
                            let Some(wrapper) = WrapperNode::try_from_plan(node) else {
                                return Ok(Transformed::no(union_plan));
                            };
                            branch_wrappers.push(wrapper);
                            node = node.inputs()[0];
                        }
                    }
                }

                branch_depths.push(branch_wrappers.len());
                branch_wrappers_list.push(branch_wrappers);
            }

            let ncols_first = projections[0].expr.len();
            if projections.iter().any(|p| p.expr.len() != ncols_first) {
                return Ok(Transformed::no(union_plan));
            }
            ncols = ncols_first;

            // For each column position, check if ALL branches project the same
            // literal value.
            for col_idx in 0..ncols {
                constant_cols.push(find_shared_constant(&projections, col_idx));
            }
        }

        // Nothing to lift if there are no shared constants.
        if constant_cols.iter().all(|c| c.is_none()) {
            return Ok(Transformed::no(union_plan));
        }

        // Need at least one non-constant column to form a valid inner union.
        let non_constant_count = constant_cols.iter().filter(|c| c.is_none()).count();
        if non_constant_count == 0 {
            return Ok(Transformed::no(union_plan));
        }

        // Check that no outer wrapper references the constant columns.
        let constant_names: Vec<&str> = constant_cols
            .iter()
            .filter_map(|c| c.as_ref().map(|(_, name)| name.as_str()))
            .collect();

        for wrapper in &outer_wrappers {
            if wrapper.references_columns(&constant_names) {
                return Ok(Transformed::no(union_plan));
            }
        }

        // Check that no branch wrapper references the constant columns.
        for branch_wrappers in &branch_wrappers_list {
            for wrapper in branch_wrappers {
                if wrapper.references_columns(&constant_names) {
                    return Ok(Transformed::no(union_plan));
                }
            }
        }

        // Build new inner branches: walk each branch to extract projection,
        // strip constants, and reconstruct the wrapper chain.
        let LogicalPlan::Union(union) = union_plan else {
            unreachable!();
        };

        let mut new_inputs: Vec<Arc<LogicalPlan>> = Vec::with_capacity(union.inputs.len());
        for (branch_idx, input) in union.inputs.into_iter().enumerate() {
            let branch_wrappers = &branch_wrappers_list[branch_idx];
            let depth = branch_depths[branch_idx];

            // Extract the projection from within the branch wrapper chain.
            let inner = extract_inner(Arc::unwrap_or_clone(input), depth);
            let LogicalPlan::Projection(proj) = inner else {
                unreachable!();
            };

            let new_exprs: Vec<Expr> = proj
                .expr
                .into_iter()
                .enumerate()
                .filter(|(i, _)| constant_cols[*i].is_none())
                .map(|(_, e)| e)
                .collect();

            let mut new_plan = LogicalPlanBuilder::from(Arc::unwrap_or_clone(proj.input))
                .project(new_exprs)?
                .build()?;

            // Reconstruct the branch wrapper chain (innermost first).
            for wrapper in branch_wrappers.iter().rev() {
                new_plan = wrapper.rebuild_ref(new_plan);
            }

            new_inputs.push(Arc::new(new_plan));
        }

        let mut inner_plan = LogicalPlan::Union(Union::try_new_with_loose_types(new_inputs)?);

        // Reconstruct the outer wrapper chain on top of the new union.
        for wrapper in outer_wrappers.into_iter().rev() {
            inner_plan = wrapper.rebuild(inner_plan);
        }

        // Build outer projection that reassembles the original column order:
        // constant columns become literals, others reference the inner output.
        let inner_schema = inner_plan.schema().clone();
        let mut outer_exprs: Vec<Expr> = Vec::with_capacity(ncols);
        let mut inner_col_idx = 0;

        for col_idx in 0..ncols {
            if let Some((ref value, ref name)) = constant_cols[col_idx] {
                outer_exprs.push(lit(value.clone()).alias(name.as_str()));
            } else {
                let (qualifier, field) = inner_schema.qualified_field(inner_col_idx);
                let col_expr =
                    Expr::Column(Column::from((qualifier, field.as_ref()))).alias(field.name());
                outer_exprs.push(col_expr);
                inner_col_idx += 1;
            }
        }

        let result = LogicalPlanBuilder::from(inner_plan)
            .project(outer_exprs)?
            .build()?;

        Ok(Transformed::yes(result))
    }
}

/// Represents an intermediate node that the projection can be lifted past.
/// Each variant stores the node's own data (without the input subtree).
///
/// To support lifting past a new node type, add a variant here and update
/// [`WrapperNode::try_from_plan`], [`WrapperNode::references_columns`],
/// [`WrapperNode::rebuild`], [`WrapperNode::rebuild_ref`], and
/// [`extract_inner`].
enum WrapperNode {
    Sort {
        expr: Vec<datafusion::logical_expr::SortExpr>,
        fetch: Option<usize>,
    },
}

impl WrapperNode {
    /// Try to interpret a plan node as a liftable wrapper. Returns `None` if
    /// the node type is not liftable.
    fn try_from_plan(plan: &LogicalPlan) -> Option<Self> {
        match plan {
            LogicalPlan::Sort(sort) => Some(WrapperNode::Sort {
                expr: sort.expr.clone(),
                fetch: sort.fetch,
            }),
            _ => None,
        }
    }

    /// Check if this wrapper node references any of the given column names.
    fn references_columns(&self, column_names: &[&str]) -> bool {
        match self {
            WrapperNode::Sort { expr, .. } => expr
                .iter()
                .any(|se| expr_references_columns(&se.expr, column_names)),
        }
    }

    /// Rebuild this wrapper node with a new input plan, consuming self.
    fn rebuild(self, input: LogicalPlan) -> LogicalPlan {
        match self {
            WrapperNode::Sort { expr, fetch } => LogicalPlan::Sort(Sort {
                expr,
                input: Arc::new(input),
                fetch,
            }),
        }
    }

    /// Rebuild this wrapper node with a new input plan, borrowing self.
    fn rebuild_ref(&self, input: LogicalPlan) -> LogicalPlan {
        match self {
            WrapperNode::Sort { expr, fetch } => LogicalPlan::Sort(Sort {
                expr: expr.clone(),
                input: Arc::new(input),
                fetch: *fetch,
            }),
        }
    }
}

/// Extract the innermost plan by peeling off `depth` layers of wrapper nodes.
fn extract_inner(plan: LogicalPlan, depth: usize) -> LogicalPlan {
    if depth == 0 {
        return plan;
    }

    match plan {
        LogicalPlan::Sort(sort) => extract_inner(Arc::unwrap_or_clone(sort.input), depth - 1),
        _ => unreachable!("extract_inner called with mismatched depth"),
    }
}

/// Check if an expression references any of the given column names.
fn expr_references_columns(expr: &Expr, column_names: &[&str]) -> bool {
    match expr {
        Expr::Column(col) => column_names.contains(&col.name.as_str()),
        _ => {
            let mut references = false;
            expr.apply(|e| {
                if let Expr::Column(col) = e {
                    if column_names.contains(&col.name.as_str()) {
                        references = true;
                        return Ok(datafusion::common::tree_node::TreeNodeRecursion::Stop);
                    }
                }
                Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
            })
            .ok();
            references
        }
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
