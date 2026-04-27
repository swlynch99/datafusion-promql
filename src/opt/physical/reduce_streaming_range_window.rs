use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;

use crate::exec::{StreamingRangeFuncExec, WideStreamingRangeFuncExec};

/// Physical optimizer rule that reduces the per-series sliding-window sample
/// cap of streaming range execs whose function does not need the full window.
///
/// Range functions vary in how many trailing samples they actually consult:
///
/// - `irate` / `idelta` only ever look at the last two samples;
/// - `last_over_time` / `present_over_time` only need the most recent sample;
/// - everything else (`rate`, `increase`, `*_over_time`, `deriv`,
///   `predict_linear`, `quantile_over_time`, …) requires the full window.
///
/// For the bounded-need cases this rule sets the exec's
/// `max_window_samples` cap so that older samples within the time window can
/// be dropped from the per-series deque, lowering peak memory without
/// changing the result. The time window (`range_ns`) itself is unchanged —
/// it still controls when results are produced and which samples count as
/// "in range" for the purposes of cap-eligibility.
///
/// For the wide-format variant (`WideStreamingRangeFuncExec`) all columns
/// share a single deque cap, so the cap is the maximum of every column's
/// sample need. Mixing in any column that needs the full window disables
/// the cap altogether.
#[derive(Debug, Default)]
pub struct ReduceStreamingRangeWindow;

impl ReduceStreamingRangeWindow {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for ReduceStreamingRangeWindow {
    fn name(&self) -> &str {
        "reduce_streaming_range_window"
    }

    fn schema_check(&self) -> bool {
        // The cap is purely a runtime memory bound; it does not change the
        // schema or the result of the exec.
        true
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            if let Some(exec) = node.as_any().downcast_ref::<StreamingRangeFuncExec>() {
                let cap = exec.func().max_samples_needed();
                if cap.is_some() && cap != exec.max_window_samples() {
                    let children = exec.children().into_iter().cloned().collect::<Vec<_>>();
                    let rebuilt = Arc::new(exec.clone_with_max_window_samples(children, cap));
                    return Ok(Transformed::yes(rebuilt as Arc<dyn ExecutionPlan>));
                }
            } else if let Some(exec) = node.as_any().downcast_ref::<WideStreamingRangeFuncExec>() {
                let cap = max_cap_for_columns(exec.funcs());
                if cap.is_some() && cap != exec.max_window_samples() {
                    let children = exec.children().into_iter().cloned().collect::<Vec<_>>();
                    let rebuilt = Arc::new(exec.clone_with_max_window_samples(children, cap));
                    return Ok(Transformed::yes(rebuilt as Arc<dyn ExecutionPlan>));
                }
            }
            Ok(Transformed::no(node))
        })
        .map(|t| t.data)
    }
}

/// Compute the shared cap for a set of wide-format column functions.
///
/// Returns the maximum of every column's [`RangeFunction::max_samples_needed`].
/// If any column needs the full window (returns `None`), the deque must
/// remain uncapped.
fn max_cap_for_columns(funcs: &[crate::node::ColumnRangeFunc]) -> Option<usize> {
    if funcs.is_empty() {
        return None;
    }
    let mut cap = 0usize;
    for cf in funcs {
        match cf.func.max_samples_needed() {
            Some(n) => cap = cap.max(n),
            None => return None,
        }
    }
    Some(cap)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::config::ConfigOptions;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::physical_plan::empty::EmptyExec;

    use super::ReduceStreamingRangeWindow;
    use crate::exec::{StreamingRangeFuncExec, WideStreamingRangeFuncExec};
    use crate::func::RangeFunction;
    use crate::node::ColumnRangeFunc;

    fn long_input_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("value", DataType::Float64, true),
            Field::new("instance", DataType::Utf8, true),
        ]))
    }

    fn wide_input_schema(n_cols: usize) -> Arc<Schema> {
        let mut fields = vec![Field::new("timestamp", DataType::UInt64, false)];
        for i in 0..n_cols {
            fields.push(Field::new(format!("col_{i}"), DataType::Float64, true));
        }
        Arc::new(Schema::new(fields))
    }

    fn make_long_exec(func: RangeFunction) -> Arc<dyn ExecutionPlan> {
        let child = Arc::new(EmptyExec::new(long_input_schema()));
        Arc::new(StreamingRangeFuncExec::new(
            child,
            func,
            None,
            60_000_000_000,
            vec![1_000_000_000],
            0,
            None,
            vec!["instance".to_string()],
        ))
    }

    fn make_wide_exec(funcs: Vec<RangeFunction>) -> Arc<dyn ExecutionPlan> {
        let value_columns: Vec<String> = (0..funcs.len()).map(|i| format!("col_{i}")).collect();
        let child = Arc::new(EmptyExec::new(wide_input_schema(funcs.len())));
        let column_funcs: Vec<ColumnRangeFunc> = funcs
            .into_iter()
            .map(|f| ColumnRangeFunc::new(f, None))
            .collect();
        Arc::new(WideStreamingRangeFuncExec::new(
            child,
            column_funcs,
            60_000_000_000,
            vec![1_000_000_000],
            0,
            None,
            value_columns,
        ))
    }

    fn run_rule(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let rule = ReduceStreamingRangeWindow::new();
        let cfg = ConfigOptions::new();
        rule.optimize(plan, &cfg).unwrap()
    }

    #[test]
    fn caps_irate_to_two() {
        let plan = make_long_exec(RangeFunction::Irate);
        let optimized = run_rule(plan);
        let exec = optimized
            .as_any()
            .downcast_ref::<StreamingRangeFuncExec>()
            .expect("rule must preserve node type");
        assert_eq!(exec.max_window_samples(), Some(2));
    }

    #[test]
    fn caps_idelta_to_two() {
        let plan = make_long_exec(RangeFunction::Idelta);
        let optimized = run_rule(plan);
        let exec = optimized
            .as_any()
            .downcast_ref::<StreamingRangeFuncExec>()
            .unwrap();
        assert_eq!(exec.max_window_samples(), Some(2));
    }

    #[test]
    fn caps_last_over_time_to_one() {
        let plan = make_long_exec(RangeFunction::LastOverTime);
        let optimized = run_rule(plan);
        let exec = optimized
            .as_any()
            .downcast_ref::<StreamingRangeFuncExec>()
            .unwrap();
        assert_eq!(exec.max_window_samples(), Some(1));
    }

    #[test]
    fn caps_present_over_time_to_one() {
        let plan = make_long_exec(RangeFunction::PresentOverTime);
        let optimized = run_rule(plan);
        let exec = optimized
            .as_any()
            .downcast_ref::<StreamingRangeFuncExec>()
            .unwrap();
        assert_eq!(exec.max_window_samples(), Some(1));
    }

    #[test]
    fn leaves_full_window_functions_uncapped() {
        for func in [
            RangeFunction::Rate,
            RangeFunction::Increase,
            RangeFunction::Delta,
            RangeFunction::AvgOverTime,
            RangeFunction::SumOverTime,
            RangeFunction::Deriv,
        ] {
            let plan = make_long_exec(func);
            let optimized = run_rule(plan);
            let exec = optimized
                .as_any()
                .downcast_ref::<StreamingRangeFuncExec>()
                .unwrap();
            assert_eq!(
                exec.max_window_samples(),
                None,
                "unexpected cap for {func:?}"
            );
        }
    }

    #[test]
    fn caps_wide_uniform_irate() {
        let plan = make_wide_exec(vec![RangeFunction::Irate, RangeFunction::Irate]);
        let optimized = run_rule(plan);
        let exec = optimized
            .as_any()
            .downcast_ref::<WideStreamingRangeFuncExec>()
            .unwrap();
        assert_eq!(exec.max_window_samples(), Some(2));
    }

    #[test]
    fn caps_wide_mixed_bounded_to_max() {
        // last_over_time needs 1, irate needs 2 → take the max so both
        // functions still see enough samples to compute.
        let plan = make_wide_exec(vec![RangeFunction::LastOverTime, RangeFunction::Irate]);
        let optimized = run_rule(plan);
        let exec = optimized
            .as_any()
            .downcast_ref::<WideStreamingRangeFuncExec>()
            .unwrap();
        assert_eq!(exec.max_window_samples(), Some(2));
    }

    #[test]
    fn leaves_wide_uncapped_when_any_column_needs_full_window() {
        // Mixing irate with rate disables the cap because rate needs every
        // sample in the time window to detect counter resets.
        let plan = make_wide_exec(vec![RangeFunction::Irate, RangeFunction::Rate]);
        let optimized = run_rule(plan);
        let exec = optimized
            .as_any()
            .downcast_ref::<WideStreamingRangeFuncExec>()
            .unwrap();
        assert_eq!(exec.max_window_samples(), None);
    }

    #[test]
    fn idempotent_when_already_capped() {
        // Running the rule on a plan whose cap is already set should not
        // produce a new node (and must keep the same cap).
        let child = Arc::new(EmptyExec::new(long_input_schema()));
        let exec = StreamingRangeFuncExec::new(
            child,
            RangeFunction::Irate,
            None,
            60_000_000_000,
            vec![1_000_000_000],
            0,
            None,
            vec!["instance".to_string()],
        )
        .with_max_window_samples(Some(2));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(exec);
        let optimized = run_rule(plan);
        let after = optimized
            .as_any()
            .downcast_ref::<StreamingRangeFuncExec>()
            .unwrap();
        assert_eq!(after.max_window_samples(), Some(2));
    }
}
