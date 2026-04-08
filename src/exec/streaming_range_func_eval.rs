use std::any::Any;
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, OrderingRequirements, Partitioning};
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use super::label_timestamp_ordering;
use crate::func::RangeFunction;

/// Physical plan node that applies a range function to raw sorted samples
/// using a single-pass sliding-window algorithm.
///
/// Unlike the two-node `RangeVectorExec → RangeFunctionExec` pipeline this
/// node does **not** materialise intermediate `List<UInt64>` / `List<Float64>`
/// window arrays.  Instead it processes input rows in `(label_cols, timestamp)`
/// order, maintaining a per-series [`VecDeque`] as a sliding window and
/// emitting one output row per `(eval_timestamp, series)` pair as soon as the
/// window for that eval timestamp is complete.
///
/// For `irate` and `idelta` the window deque is accessed by index rather than
/// converted to a `Vec`, so no heap allocation occurs in the hot path.
#[derive(Debug)]
pub(crate) struct StreamingRangeFuncExec {
    child: Arc<dyn ExecutionPlan>,
    func: RangeFunction,
    scalar_arg: Option<f64>,
    range_ns: u64,
    eval_timestamps: Vec<u64>,
    offset_ns: i64,
    at_timestamp_ns: Option<u64>,
    label_columns: Vec<String>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

fn compute_output_schema(child_schema: &SchemaRef, label_columns: &[String]) -> SchemaRef {
    let mut fields = vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, true),
    ];
    for label in label_columns {
        let nullable = child_schema
            .field_with_name(label)
            .map(|f| f.is_nullable())
            .unwrap_or(true);
        fields.push(Field::new(label, DataType::Utf8, nullable));
    }
    Arc::new(Schema::new(fields))
}

impl StreamingRangeFuncExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        func: RangeFunction,
        scalar_arg: Option<f64>,
        range_ns: u64,
        eval_timestamps: Vec<u64>,
        offset_ns: i64,
        at_timestamp_ns: Option<u64>,
        label_columns: Vec<String>,
    ) -> Self {
        let output_schema = compute_output_schema(&child.schema(), &label_columns);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
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
            label_columns,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for StreamingRangeFuncExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StreamingRangeFuncExec: func={}, range={}ns",
            self.func, self.range_ns
        )
    }
}

impl ExecutionPlan for StreamingRangeFuncExec {
    fn name(&self) -> &str {
        "StreamingRangeFuncExec"
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
        vec![label_timestamp_ordering(
            &self.label_columns,
            &self.child.schema(),
        )]
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
            self.label_columns.clone(),
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
        let label_columns = self.label_columns.clone();
        let schema_for_stream = Arc::clone(&output_schema);

        // Collect all input first, then process synchronously in sorted order.
        // The processing is a single pass that maintains a per-series sliding
        // window deque, emitting one output row per (eval_timestamp, series)
        // as each window is finalised.  All output rows are accumulated into
        // a single RecordBatch following the same pattern used by the other
        // exec nodes in this crate.
        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            let mut batches = Vec::new();
            let mut stream = child_stream;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }

            compute_streaming_windows(
                batches,
                &eval_timestamps,
                range_ns,
                offset_ns,
                at_timestamp_ns,
                func,
                scalar_arg,
                &label_columns,
                &output_schema,
            )
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema_for_stream,
            stream,
        )))
    }
}

/// Compute the "effective" window-end timestamp for `eval_t`, applying the
/// `@` modifier and offset.
#[inline]
fn effective_ts(eval_t: u64, at_timestamp_ns: Option<u64>, offset_ns: i64) -> u64 {
    let lookup = at_timestamp_ns.unwrap_or(eval_t);
    (lookup as i64 - offset_ns) as u64
}

/// Apply the range function to the current window deque without converting it
/// to a `Vec` for `irate` and `idelta` (the primary targets of this node).
/// For all other functions a temporary `Vec` is allocated.
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
        // Hot path: only the last two samples are needed — access by index
        // to avoid a Vec allocation.
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
        // General path: build a contiguous slice and delegate to RangeFunction.
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

/// Emit one output row by appending to the builders.
fn emit_row(
    eval_t: u64,
    value: f64,
    series_key: &[String],
    out_ts: &mut UInt64Builder,
    out_val: &mut Float64Builder,
    out_labels: &mut [StringBuilder],
) {
    out_ts.append_value(eval_t);
    out_val.append_value(value);
    for (i, label_val) in series_key.iter().enumerate() {
        out_labels[i].append_value(label_val);
    }
}

/// Finalise the Arrow builders into a single output [`RecordBatch`].
fn finish_builders(
    output_schema: &SchemaRef,
    out_ts: &mut UInt64Builder,
    out_val: &mut Float64Builder,
    out_labels: &mut [StringBuilder],
    label_columns: &[String],
) -> Result<RecordBatch> {
    let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();
    for field in output_schema.fields() {
        let name = field.name().as_str();
        if name == "timestamp" {
            columns.push(Arc::new(out_ts.finish()));
        } else if name == "value" {
            columns.push(Arc::new(out_val.finish()));
        } else if let Some(idx) = label_columns.iter().position(|n| n == name) {
            columns.push(Arc::new(out_labels[idx].finish()));
        }
    }
    Ok(RecordBatch::try_new(Arc::clone(output_schema), columns)?)
}

/// Flush all eval windows in `eval_timestamps[eval_idx..]` for `series_key`
/// given the current `window` contents.
///
/// This is called both when a series boundary is detected and at the end of
/// the input stream to finalise the last (or only) series.
#[allow(clippy::too_many_arguments)]
fn flush_remaining_eval_windows(
    series_key: &[String],
    window: &mut VecDeque<(u64, f64)>,
    eval_idx: &mut usize,
    eval_timestamps: &[u64],
    range_ns: u64,
    offset_ns: i64,
    at_timestamp_ns: Option<u64>,
    func: RangeFunction,
    scalar_arg: Option<f64>,
    out_ts: &mut UInt64Builder,
    out_val: &mut Float64Builder,
    out_labels: &mut [StringBuilder],
) {
    while *eval_idx < eval_timestamps.len() {
        let eval_t = eval_timestamps[*eval_idx];
        let eff_ts = effective_ts(eval_t, at_timestamp_ns, offset_ns);
        let window_start = eff_ts.saturating_sub(range_ns);

        // Evict samples that are now before the window start.
        while window
            .front()
            .map(|(t, _)| *t < window_start)
            .unwrap_or(false)
        {
            window.pop_front();
        }

        if let Some(value) = apply_func(func, window, eval_t, scalar_arg) {
            emit_row(eval_t, value, series_key, out_ts, out_val, out_labels);
        }

        *eval_idx += 1;
    }
}

/// Core single-pass sliding-window computation.
///
/// Invariant (proved by construction): when flushing eval timestamp `t` with
/// effective window end `eff_t`, every sample currently in `window` satisfies
/// `ts <= eff_t`.  This holds because:
///
/// 1. A sample `ts` is added to the deque only *after* flushing all eval
///    timestamps whose `eff_ts < ts`.
/// 2. Therefore all unflushed eval timestamps satisfy `eff_ts >= ts`.
/// 3. As a result, when we later flush those eval timestamps, every deque
///    entry (added with `ts' <= ts`) satisfies `ts' <= ts <= eff_ts`.
///
/// This means no upper-bound binary search is needed; after evicting the
/// front (samples older than `window_start`) the entire deque is the window.
#[allow(clippy::too_many_arguments)]
fn compute_streaming_windows(
    batches: Vec<RecordBatch>,
    eval_timestamps: &[u64],
    range_ns: u64,
    offset_ns: i64,
    at_timestamp_ns: Option<u64>,
    func: RangeFunction,
    scalar_arg: Option<f64>,
    label_columns: &[String],
    output_schema: &SchemaRef,
) -> Result<RecordBatch> {
    let mut out_ts = UInt64Builder::new();
    let mut out_val = Float64Builder::new();
    let mut out_labels: Vec<StringBuilder> =
        label_columns.iter().map(|_| StringBuilder::new()).collect();

    // Per-series sliding-window state.
    let mut current_series: Option<Vec<String>> = None;
    let mut window: VecDeque<(u64, f64)> = VecDeque::new();
    let mut eval_idx: usize = 0;

    for batch in &batches {
        let ts_arr = batch
            .column_by_name("timestamp")
            .expect("missing timestamp column")
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .expect("timestamp must be UInt64");

        let val_arr = batch
            .column_by_name("value")
            .expect("missing value column")
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("value must be Float64");

        let label_arrs: Vec<&arrow::array::StringArray> = label_columns
            .iter()
            .map(|name| {
                batch
                    .column_by_name(name)
                    .unwrap_or_else(|| panic!("missing label column: {name}"))
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .unwrap_or_else(|| panic!("label column {name} must be Utf8"))
            })
            .collect();

        for row in 0..batch.num_rows() {
            let ts = ts_arr.value(row);
            let val = val_arr.value(row);
            let series_key: Vec<String> = label_arrs
                .iter()
                .map(|arr| arr.value(row).to_string())
                .collect();

            // Detect series boundary.
            if current_series.as_deref() != Some(series_key.as_slice()) {
                // Flush all remaining eval windows for the outgoing series.
                if let Some(ref old_key) = current_series {
                    let old_key = old_key.clone();
                    flush_remaining_eval_windows(
                        &old_key,
                        &mut window,
                        &mut eval_idx,
                        eval_timestamps,
                        range_ns,
                        offset_ns,
                        at_timestamp_ns,
                        func,
                        scalar_arg,
                        &mut out_ts,
                        &mut out_val,
                        &mut out_labels,
                    );
                }

                current_series = Some(series_key.clone());
                window.clear();
                eval_idx = 0;
            }

            // Flush eval windows whose effective window end is before `ts`.
            // Once we see a sample at `ts`, no further samples can fall into
            // windows with `eff_ts < ts` for this series.
            while eval_idx < eval_timestamps.len() {
                let eval_t = eval_timestamps[eval_idx];
                let eff_ts = effective_ts(eval_t, at_timestamp_ns, offset_ns);
                if eff_ts >= ts {
                    break;
                }
                let window_start = eff_ts.saturating_sub(range_ns);

                // Evict samples older than the window start.
                while window
                    .front()
                    .map(|(t, _)| *t < window_start)
                    .unwrap_or(false)
                {
                    window.pop_front();
                }

                if let Some(value) = apply_func(func, &window, eval_t, scalar_arg) {
                    emit_row(
                        eval_t,
                        value,
                        &series_key,
                        &mut out_ts,
                        &mut out_val,
                        &mut out_labels,
                    );
                }

                eval_idx += 1;
            }

            // Add the current sample to the sliding window.
            window.push_back((ts, val));
        }
    }

    // Flush remaining eval windows for the last series.
    if let Some(ref last_key) = current_series {
        let last_key = last_key.clone();
        flush_remaining_eval_windows(
            &last_key,
            &mut window,
            &mut eval_idx,
            eval_timestamps,
            range_ns,
            offset_ns,
            at_timestamp_ns,
            func,
            scalar_arg,
            &mut out_ts,
            &mut out_val,
            &mut out_labels,
        );
    }

    finish_builders(
        output_schema,
        &mut out_ts,
        &mut out_val,
        &mut out_labels,
        label_columns,
    )
}
