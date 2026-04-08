use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Extension, LogicalPlan, Projection, Union};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::node::StreamingRangeFunctionEval;

/// Optimizer rule that pushes a `StreamingRangeFunctionEval` node down through
/// a `Union` when each branch of the union produces a disjoint set of series.
///
/// ```text
/// StreamingRangeFunctionEval(func, range, …)    Union
///   Union                                    →     StreamingRangeFunctionEval(…)
///     Branch1                                        Branch1
///     Branch2                                      StreamingRangeFunctionEval(…)
///                                                    Branch2
/// ```
///
/// Disjointness is detected by checking that all union branches are
/// `Projection` nodes whose label columns include at least one constant
/// literal that is unique per branch. This is the common case produced by
/// the wide-to-long normalization in `normalize.rs`.
///
/// The transformation is valid because `StreamingRangeFunctionEval` groups by
/// `label_columns` and applies a sliding-window range function per series.
/// If branches contain non-overlapping series, evaluating each branch
/// independently yields the same result as evaluating the combined union.
#[derive(Debug)]
pub struct PushStreamingRangeFuncEvalThroughUnion;

impl OptimizerRule for PushStreamingRangeFuncEvalThroughUnion {
    fn name(&self) -> &str {
        "push_streaming_range_func_eval_through_union"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Match: Extension(StreamingRangeFunctionEval) whose input is a Union.
        let LogicalPlan::Extension(ref ext) = plan else {
            return Ok(Transformed::no(plan));
        };

        let Some(eval) = ext
            .node
            .as_any()
            .downcast_ref::<StreamingRangeFunctionEval>()
        else {
            return Ok(Transformed::no(plan));
        };

        let LogicalPlan::Union(ref union) = eval.input else {
            return Ok(Transformed::no(plan));
        };

        if union.inputs.len() < 2 {
            return Ok(Transformed::no(plan));
        }

        // Check that branches have disjoint label sets.
        if !branches_have_disjoint_labels(&union.inputs, &eval.label_columns) {
            return Ok(Transformed::no(plan));
        }

        // Destructure to take ownership.
        let LogicalPlan::Extension(ext) = plan else {
            unreachable!();
        };
        let eval = ext
            .node
            .as_any()
            .downcast_ref::<StreamingRangeFunctionEval>()
            .unwrap()
            .clone();
        let LogicalPlan::Union(union) = eval.input else {
            unreachable!();
        };

        // Wrap each branch in its own StreamingRangeFunctionEval.
        let new_inputs: Vec<Arc<LogicalPlan>> = union
            .inputs
            .into_iter()
            .map(|branch| {
                let new_eval = StreamingRangeFunctionEval::new(
                    Arc::unwrap_or_clone(branch),
                    eval.func,
                    eval.scalar_arg,
                    eval.range_ns,
                    eval.eval_ts_ns,
                    eval.start_ns,
                    eval.end_ns,
                    eval.step_ns,
                    eval.offset_ns,
                    eval.label_columns.clone(),
                    eval.at_timestamp_ns,
                )
                .expect("schema computation should not fail for valid input");
                Arc::new(LogicalPlan::Extension(Extension {
                    node: Arc::new(new_eval),
                }))
            })
            .collect();

        let new_union = LogicalPlan::Union(Union::try_new_with_loose_types(new_inputs)?);
        Ok(Transformed::yes(new_union))
    }
}

/// Check whether all union branches produce disjoint series based on the given
/// label columns.
///
/// Returns `true` if every branch is a `Projection` that assigns constant
/// literal values to at least one label column, and the tuple of those constant
/// values is unique across all branches.
fn branches_have_disjoint_labels(inputs: &[Arc<LogicalPlan>], label_columns: &[String]) -> bool {
    if label_columns.is_empty() {
        return false;
    }

    // Collect the constant label fingerprint for each branch.
    let mut fingerprints: Vec<Vec<(usize, String)>> = Vec::with_capacity(inputs.len());

    for input in inputs {
        let LogicalPlan::Projection(ref proj) = **input else {
            return false;
        };

        let fp = extract_constant_label_fingerprint(proj, label_columns);
        if fp.is_empty() {
            // No constant label columns in this branch — cannot prove disjointness.
            return false;
        }
        fingerprints.push(fp);
    }

    // All fingerprints must be unique.
    let mut seen = std::collections::HashSet::new();
    for fp in &fingerprints {
        if !seen.insert(fp.clone()) {
            return false;
        }
    }

    true
}

/// For a `Projection`, extract the constant literal values for columns that
/// appear in `label_columns`. Returns a sorted vec of `(label_index, value)`
/// pairs for the constant label columns found.
fn extract_constant_label_fingerprint(
    proj: &Projection,
    label_columns: &[String],
) -> Vec<(usize, String)> {
    use datafusion::logical_expr::Expr;

    let mut result = Vec::new();

    for expr in &proj.expr {
        let (name, value) = match expr {
            Expr::Alias(alias) => match alias.expr.as_ref() {
                Expr::Literal(scalar, _) => (alias.name.as_str(), scalar.to_string()),
                _ => continue,
            },
            _ => continue,
        };

        if let Some(idx) = label_columns.iter().position(|lc| lc == name) {
            result.push((idx, value));
        }
    }

    result.sort();
    result
}
