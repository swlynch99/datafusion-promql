use std::collections::HashSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::node::WideUnpack;

/// Optimizer rule that prunes unused output columns from a [`WideUnpack`]
/// node based on what its immediate parent `Projection` references.
///
/// DataFusion's built-in `OptimizeProjections` rule cannot rewrite the
/// schema of a [`UserDefinedLogicalNode`]. When the parent `Aggregate`
/// (or any other consumer) only needs a subset of `WideUnpack`'s output,
/// `OptimizeProjections` inserts a column-pruning `Projection` above the
/// unpack:
///
/// ```text
/// Aggregate(group=[timestamp, id], sum(value))
///   Projection: [timestamp, value, id]
///     WideUnpack(label_keys=[id, op])      // emits __name__, op too
/// ```
///
/// That projection is cheap to execute, but it sits between the aggregate
/// and the `WideUnpack`, blocking the much more impactful
/// [`PushSumThroughWideUnpack`] rewrite (which only matches
/// `Aggregate → WideUnpack` directly).
///
/// This rule fixes that by pushing the pruning into the `WideUnpack` itself.
/// Given `Projection → WideUnpack`, it inspects which of the unpack's output
/// columns the projection references and rebuilds the unpack with a smaller
/// `label_keys` and/or `include_name=false`. After this rewrite the
/// projection becomes an identity (same fields, same order), so
/// [`RemoveNoopProjections`] removes it on the next pass and the
/// `Aggregate → WideUnpack` shape is restored.
///
/// `timestamp` and `value` are always kept — every consumer needs them, and
/// `value` is the whole point of the unpack.
///
/// [`PushSumThroughWideUnpack`]: super::PushSumThroughWideUnpack
/// [`RemoveNoopProjections`]: super::RemoveNoopProjections
#[derive(Debug)]
pub struct PruneWideUnpackColumns;

impl OptimizerRule for PruneWideUnpackColumns {
    fn name(&self) -> &str {
        "prune_wide_unpack_columns"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Match: Projection whose input is Extension(WideUnpack).
        let LogicalPlan::Projection(ref proj) = plan else {
            return Ok(Transformed::no(plan));
        };

        let LogicalPlan::Extension(ref ext) = *proj.input else {
            return Ok(Transformed::no(plan));
        };

        let Some(unpack) = ext.node.as_any().downcast_ref::<WideUnpack>() else {
            return Ok(Transformed::no(plan));
        };

        // Collect the set of unqualified column names referenced by the
        // projection's expressions. Anything in the unpack output that isn't
        // in this set is unused and can be pruned.
        let mut referenced: HashSet<String> = HashSet::new();
        for expr in &proj.expr {
            expr.apply(|e| {
                if let Expr::Column(c) = e {
                    referenced.insert(c.name.clone());
                }
                Ok(TreeNodeRecursion::Continue)
            })?;
        }

        let new_include_name = unpack.include_name && referenced.contains("__name__");
        let new_label_keys: Vec<String> = unpack
            .label_keys
            .iter()
            .filter(|k| referenced.contains(k.as_str()))
            .cloned()
            .collect();

        // Already minimal — nothing to do.
        if new_include_name == unpack.include_name
            && new_label_keys.len() == unpack.label_keys.len()
        {
            return Ok(Transformed::no(plan));
        }

        let new_unpack = WideUnpack::new_with_options(
            unpack.input.clone(),
            unpack.columns.clone(),
            Arc::new(new_label_keys),
            new_include_name,
        )
        .map_err(|e| datafusion::error::DataFusionError::Plan(e.to_string()))?;

        let new_unpack_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(new_unpack),
        });

        // Rebuild the projection over the narrowed unpack. The schema below
        // shrank, so we can no longer reuse the cached projection schema.
        let new_plan = LogicalPlanBuilder::from(new_unpack_plan)
            .project(proj.expr.clone())?
            .build()?;

        Ok(Transformed::yes(new_plan))
    }
}
