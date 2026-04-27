use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::node::{StreamingRangeFunctionEval, WideStreamingRangeFunctionEval, WideUnpack};

/// Optimizer rule that pushes a [`StreamingRangeFunctionEval`] down through a
/// [`WideUnpack`] node, converting it into a [`WideStreamingRangeFunctionEval`]
/// operating on the wide-format input.
///
/// ```text
/// StreamingRangeFunctionEval(func, range, …)       WideUnpack(columns, labels)
///   WideUnpack(columns, labels)                 →    WideStreamingRangeFunctionEval(func, range, …)
///     WideInput                                         WideInput
/// ```
///
/// This is valid because:
///
///   * Each value column of the wide input corresponds to exactly one series
///     in the long-format output of `WideUnpack` (the series' `__name__` and
///     labels are the constants recorded in the column's [`WideColumnMeta`]).
///   * [`StreamingRangeFunctionEval`] groups by `label_columns` and applies a
///     per-series sliding-window range function. Applying the same function
///     independently to each column of the wide input yields identical
///     per-series results.
///   * The wide-format input is sorted by `timestamp ASC`, which is what
///     `WideStreamingRangeFunctionEval` requires.
///
/// The transformation has two main benefits:
///
///   * The range function runs on N (typically ~900) slim value columns rather
///     than on N × #timestamps long-format rows materialised by `WideUnpack`.
///   * After the range function, the number of rows drops from #samples to
///     #eval-timestamps, so `WideUnpack` then expands a much smaller table.
#[derive(Debug)]
pub struct PushStreamingRangeFuncEvalThroughWideUnpack;

impl OptimizerRule for PushStreamingRangeFuncEvalThroughWideUnpack {
    fn name(&self) -> &str {
        "push_streaming_range_func_eval_through_wide_unpack"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Match: Extension(StreamingRangeFunctionEval) whose input is
        //        Extension(WideUnpack).
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

        let LogicalPlan::Extension(ref inner_ext) = eval.input else {
            return Ok(Transformed::no(plan));
        };

        let Some(unpack) = inner_ext.node.as_any().downcast_ref::<WideUnpack>() else {
            return Ok(Transformed::no(plan));
        };

        // Only apply the push-down when the streaming evaluator's grouping
        // matches the per-column series identity produced by `WideUnpack`.
        // Each column of `WideUnpack` has a constant `__name__` and one
        // constant value for every label in `label_keys`, so the set of
        // label columns the streaming evaluator groups by must be exactly
        // `{"__name__"} ∪ label_keys`.
        if !label_columns_match_unpack(&eval.label_columns, &unpack.label_keys) {
            return Ok(Transformed::no(plan));
        }

        // Destructure to take ownership of the inner nodes without cloning
        // their sub-plans.
        let LogicalPlan::Extension(ext) = plan else {
            unreachable!();
        };
        let eval = ext
            .node
            .as_any()
            .downcast_ref::<StreamingRangeFunctionEval>()
            .unwrap()
            .clone();
        let LogicalPlan::Extension(inner_ext) = eval.input else {
            unreachable!();
        };
        let unpack = inner_ext
            .node
            .as_any()
            .downcast_ref::<WideUnpack>()
            .unwrap()
            .clone();

        // Collect the list of value column names from the unpack metadata.
        // These are the column names in the wide-format input plan that the
        // new `WideStreamingRangeFunctionEval` should process.
        let value_columns: Vec<String> =
            unpack.columns.iter().map(|m| m.col_name.clone()).collect();

        // Build the new plan:  WideUnpack( WideStreamingRangeFunctionEval( <wide input> ) )
        let wide_eval = WideStreamingRangeFunctionEval::new(
            unpack.input.clone(),
            eval.func,
            eval.scalar_arg,
            eval.range_ns,
            eval.eval_ts_ns,
            eval.start_ns,
            eval.end_ns,
            eval.step_ns,
            eval.offset_ns,
            Arc::new(value_columns),
            eval.at_timestamp_ns,
        )
        .map_err(|e| datafusion::error::DataFusionError::Plan(e.to_string()))?;

        let wide_eval_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(wide_eval),
        });

        let new_unpack = WideUnpack::new(wide_eval_plan, unpack.columns, unpack.label_keys)
            .map_err(|e| datafusion::error::DataFusionError::Plan(e.to_string()))?;

        let new_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(new_unpack),
        });

        Ok(Transformed::yes(new_plan))
    }
}

/// Check that `label_columns` equals `{"__name__"} ∪ label_keys` as an
/// unordered set. When this holds, each wide column maps to a unique series
/// in the streaming evaluator's grouping.
fn label_columns_match_unpack(label_columns: &[String], label_keys: &[String]) -> bool {
    if label_columns.len() != label_keys.len() + 1 {
        return false;
    }

    let mut saw_name = false;
    for lc in label_columns {
        if lc == "__name__" {
            saw_name = true;
        } else if !label_keys.iter().any(|k| k == lc) {
            return false;
        }
    }
    saw_name
}
