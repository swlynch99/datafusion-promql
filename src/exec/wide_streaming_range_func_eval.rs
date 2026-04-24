use std::any::Any;
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, Float64Builder, UInt64Builder};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexRequirement, OrderingRequirements, Partitioning, PhysicalSortExpr,
    PhysicalSortRequirement,
};
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::func::RangeFunction;

/// Physical plan node that applies a range function to every value column of
/// a wide-format input using a single-pass sliding-window algorithm.
///
/// Input schema: `(timestamp: UInt64, col_0: Float64, …, col_N: Float64)`
/// sorted by `timestamp ASC`.
///
/// Output schema: same column names as the input (`timestamp` plus the value
/// columns), but with one row per evaluation timestamp. Each value column
/// contains the range function's result for that series at the corresponding
/// eval timestamp, or null if the function returns `None` (e.g. too few
/// samples in the window for that series).
///
/// Rows in which every value column is null are skipped to avoid emitting
/// work for the downstream `WideUnpackExec` that would just be filtered out.
#[derive(Debug)]
pub(crate) struct WideStreamingRangeFuncExec {
    child: Arc<dyn ExecutionPlan>,
    func: RangeFunction,
    scalar_arg: Option<f64>,
    range_ns: u64,
    eval_timestamps: Vec<u64>,
    offset_ns: i64,
    at_timestamp_ns: Option<u64>,
    value_columns: Vec<String>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

fn compute_output_schema(value_columns: &[String]) -> SchemaRef {
    let mut fields = vec![Field::new("timestamp", DataType::UInt64, false)];
    for col in value_columns {
        fields.push(Field::new(col.as_str(), DataType::Float64, true));
    }
    Arc::new(Schema::new(fields))
}

impl WideStreamingRangeFuncExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        func: RangeFunction,
        scalar_arg: Option<f64>,
        range_ns: u64,
        eval_timestamps: Vec<u64>,
        offset_ns: i64,
        at_timestamp_ns: Option<u64>,
        value_columns: Vec<String>,
    ) -> Self {
        let output_schema = compute_output_schema(&value_columns);

        let asc_nulls_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let ordering = match Column::new_with_schema("timestamp", output_schema.as_ref()) {
            Ok(ts_col) => vec![PhysicalSortExpr::new(Arc::new(ts_col), asc_nulls_last)],
            Err(_) => Vec::new(),
        };
        let eq_properties = if ordering.is_empty() {
            EquivalenceProperties::new(Arc::clone(&output_schema))
        } else {
            EquivalenceProperties::new_with_orderings(Arc::clone(&output_schema), [ordering])
        };

        let properties = Arc::new(PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            child,
            func,
            scalar_arg,
            range_ns,
            eval_timestamps,
            offset_ns,
            at_timestamp_ns,
            value_columns,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for WideStreamingRangeFuncExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WideStreamingRangeFuncExec: func={}, range={}ns, columns={}",
            self.func,
            self.range_ns,
            self.value_columns.len()
        )
    }
}

impl ExecutionPlan for WideStreamingRangeFuncExec {
    fn name(&self) -> &str {
        "WideStreamingRangeFuncExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        // Require input sorted by timestamp.
        let child_schema = self.child.schema();
        let asc_nulls_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let reqs = match Column::new_with_schema("timestamp", child_schema.as_ref()) {
            Ok(ts_col) => vec![PhysicalSortRequirement::new(
                Arc::new(ts_col),
                Some(asc_nulls_last),
            )],
            Err(_) => return vec![None],
        };
        let lex = LexRequirement::new(reqs).map(OrderingRequirements::new);
        vec![lex]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            self.func,
            self.scalar_arg,
            self.range_ns,
            self.eval_timestamps.clone(),
            self.offset_ns,
            self.at_timestamp_ns,
            self.value_columns.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, Arc::clone(&context))?;
        let output_schema = Arc::clone(&self.output_schema);
        let func = self.func;
        let scalar_arg = self.scalar_arg;
        let range_ns = self.range_ns;
        let eval_timestamps = self.eval_timestamps.clone();
        let offset_ns = self.offset_ns;
        let at_timestamp_ns = self.at_timestamp_ns;
        let value_columns = self.value_columns.clone();
        let schema_for_stream = Arc::clone(&output_schema);
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            let mut batches = Vec::new();
            let mut stream = child_stream;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }

            let _timer = baseline_metrics.elapsed_compute().timer();

            let batch = compute_wide_streaming_windows(
                batches,
                &eval_timestamps,
                range_ns,
                offset_ns,
                at_timestamp_ns,
                func,
                scalar_arg,
                &value_columns,
                &output_schema,
            )?;
            Ok(batch.record_output(&baseline_metrics))
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema_for_stream,
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

#[inline]
fn effective_ts(eval_t: u64, at_timestamp_ns: Option<u64>, offset_ns: i64) -> u64 {
    let lookup = at_timestamp_ns.unwrap_or(eval_t);
    (lookup as i64 - offset_ns) as u64
}

/// Apply a range function to a sliding-window deque. Mirrors the hot-path
/// optimization in `streaming_range_func_eval.rs` for `irate`/`idelta`.
fn apply_func(
    func: RangeFunction,
    window: &VecDeque<(u64, f64)>,
    eval_t: u64,
    scalar_arg: Option<f64>,
) -> Option<f64> {
    let n = window.len();
    if n == 0 {
        return None;
    }

    match func {
        RangeFunction::Irate => {
            if n < 2 {
                return None;
            }
            let (prev_ts, prev_val) = window[n - 2];
            let (last_ts, last_val) = window[n - 1];
            let dt_secs = (last_ts - prev_ts) as f64 / 1_000_000_000.0;
            if dt_secs == 0.0 {
                return None;
            }
            let increase = if last_val < prev_val {
                last_val
            } else {
                last_val - prev_val
            };
            Some(increase / dt_secs)
        }
        RangeFunction::Idelta => {
            if n < 2 {
                return None;
            }
            let (_, prev_val) = window[n - 2];
            let (_, last_val) = window[n - 1];
            Some(last_val - prev_val)
        }
        _ => {
            let (s1, s2) = window.as_slices();
            if s2.is_empty() {
                func.evaluate(s1, eval_t, scalar_arg)
            } else {
                let samples: Vec<(u64, f64)> = s1.iter().chain(s2.iter()).copied().collect();
                func.evaluate(&samples, eval_t, scalar_arg)
            }
        }
    }
}

/// Core single-pass sliding-window computation over wide-format input.
///
/// Input rows are sorted by `timestamp ASC`. For each eval timestamp we
/// maintain one [`VecDeque`] per value column. Invariants mirror the
/// long-format version in `streaming_range_func_eval.rs`: a sample at
/// timestamp `ts` is pushed to its column's deque only *after* flushing
/// every eval timestamp whose effective end is strictly less than `ts`.
#[allow(clippy::too_many_arguments)]
fn compute_wide_streaming_windows(
    batches: Vec<RecordBatch>,
    eval_timestamps: &[u64],
    range_ns: u64,
    offset_ns: i64,
    at_timestamp_ns: Option<u64>,
    func: RangeFunction,
    scalar_arg: Option<f64>,
    value_columns: &[String],
    output_schema: &SchemaRef,
) -> Result<RecordBatch> {
    let n_cols = value_columns.len();

    let mut out_ts = UInt64Builder::new();
    let mut out_vals: Vec<Float64Builder> = value_columns
        .iter()
        .map(|_| Float64Builder::new())
        .collect();

    let mut windows: Vec<VecDeque<(u64, f64)>> = (0..n_cols).map(|_| VecDeque::new()).collect();
    let mut eval_idx: usize = 0;

    let flush_one_eval = |eval_t: u64,
                          windows: &mut [VecDeque<(u64, f64)>],
                          out_ts: &mut UInt64Builder,
                          out_vals: &mut [Float64Builder]| {
        let eff_ts = effective_ts(eval_t, at_timestamp_ns, offset_ns);
        let window_start = eff_ts.saturating_sub(range_ns);

        // Evict stale samples per column and compute results.
        let mut results: Vec<Option<f64>> = Vec::with_capacity(n_cols);
        let mut any_some = false;
        for w in windows.iter_mut() {
            while w.front().map(|(t, _)| *t < window_start).unwrap_or(false) {
                w.pop_front();
            }
            let r = apply_func(func, w, eval_t, scalar_arg);
            if r.is_some() {
                any_some = true;
            }
            results.push(r);
        }

        if !any_some {
            return;
        }

        out_ts.append_value(eval_t);
        for (c, r) in results.into_iter().enumerate() {
            match r {
                Some(v) => out_vals[c].append_value(v),
                None => out_vals[c].append_null(),
            }
        }
    };

    for batch in &batches {
        let ts_arr = batch
            .column_by_name("timestamp")
            .expect("missing timestamp column")
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .expect("timestamp must be UInt64");

        let val_arrs: Vec<&arrow::array::Float64Array> = value_columns
            .iter()
            .map(|name| {
                batch
                    .column_by_name(name.as_str())
                    .unwrap_or_else(|| panic!("missing value column: {name}"))
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap_or_else(|| panic!("value column {name} must be Float64"))
            })
            .collect();

        for row in 0..batch.num_rows() {
            let ts = ts_arr.value(row);

            // Flush any eval windows whose effective end is strictly before
            // the current sample's timestamp. After this, the current sample
            // can be safely added to each column's deque.
            while eval_idx < eval_timestamps.len() {
                let eval_t = eval_timestamps[eval_idx];
                let eff_ts = effective_ts(eval_t, at_timestamp_ns, offset_ns);
                if eff_ts >= ts {
                    break;
                }
                flush_one_eval(eval_t, &mut windows, &mut out_ts, &mut out_vals);
                eval_idx += 1;
            }

            // Push the sample into each column's deque (skipping nulls).
            for (c, arr) in val_arrs.iter().enumerate() {
                if arr.is_null(row) {
                    continue;
                }
                windows[c].push_back((ts, arr.value(row)));
            }
        }
    }

    // Flush remaining eval windows after all input has been consumed.
    while eval_idx < eval_timestamps.len() {
        let eval_t = eval_timestamps[eval_idx];
        flush_one_eval(eval_t, &mut windows, &mut out_ts, &mut out_vals);
        eval_idx += 1;
    }

    // Build output RecordBatch matching `output_schema` (timestamp then each
    // value column in the declared order).
    let mut arrays: Vec<arrow::array::ArrayRef> = Vec::with_capacity(1 + n_cols);
    arrays.push(Arc::new(out_ts.finish()));
    for builder in out_vals.iter_mut() {
        arrays.push(Arc::new(builder.finish()));
    }
    Ok(RecordBatch::try_new(Arc::clone(output_schema), arrays)?)
}
