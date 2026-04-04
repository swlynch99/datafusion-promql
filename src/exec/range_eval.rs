use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int64Builder, ListBuilder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

/// Physical plan node that groups samples by metric series and collects
/// them into per-window arrays at each evaluation timestamp.
///
/// For each step timestamp `t` and each series, this node collects all samples
/// in `[t - range_ns, t]` and outputs:
/// - `timestamp: Int64` — the evaluation timestamp
/// - `timestamps: List<Int64>` — sample timestamps within the window
/// - `values: List<Float64>` — sample values within the window
/// - label columns (Utf8)
#[derive(Debug)]
pub(crate) struct RangeVectorExec {
    child: Arc<dyn ExecutionPlan>,
    range_ns: i64,
    eval_ts_ns: Option<i64>,
    start_ns: i64,
    end_ns: i64,
    step_ns: i64,
    label_columns: Vec<String>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

/// Build the output Arrow schema for the windowing node.
fn compute_output_schema(label_columns: &[String]) -> SchemaRef {
    let mut fields = vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new(
            "timestamps",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ),
        Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
    ];
    for label in label_columns {
        fields.push(Field::new(label, DataType::Utf8, true));
    }
    Arc::new(Schema::new(fields))
}

impl RangeVectorExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        range_ns: i64,
        eval_ts_ns: Option<i64>,
        start_ns: i64,
        end_ns: i64,
        step_ns: i64,
        label_columns: Vec<String>,
    ) -> Self {
        let output_schema = compute_output_schema(&label_columns);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            child,
            range_ns,
            eval_ts_ns,
            start_ns,
            end_ns,
            step_ns,
            label_columns,
            output_schema,
            properties,
        }
    }

    fn eval_timestamps(&self) -> Vec<i64> {
        if let Some(ts) = self.eval_ts_ns {
            return vec![ts];
        }
        let mut timestamps = Vec::new();
        let mut t = self.start_ns;
        while t <= self.end_ns {
            timestamps.push(t);
            t += self.step_ns;
        }
        timestamps
    }
}

impl DisplayAs for RangeVectorExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RangeVectorExec: range={}ns", self.range_ns)
    }
}

impl ExecutionPlan for RangeVectorExec {
    fn name(&self) -> &str {
        "RangeVectorExec"
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

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            self.range_ns,
            self.eval_ts_ns,
            self.start_ns,
            self.end_ns,
            self.step_ns,
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
        let eval_timestamps = self.eval_timestamps();
        let range_ns = self.range_ns;
        let label_columns = self.label_columns.clone();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            // Collect all batches from the child stream.
            let mut batches = Vec::new();
            let mut stream = child_stream;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }

            // Build a map: series_key -> Vec<(timestamp, value)>
            let mut series_map: HashMap<Vec<String>, Vec<(i64, f64)>> = HashMap::new();

            for batch in &batches {
                let ts_col = batch
                    .column_by_name("timestamp")
                    .expect("missing timestamp column");
                let val_col = batch.column_by_name("value").expect("missing value column");

                let ts_arr = ts_col
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .expect("timestamp must be Int64");
                let val_arr = val_col
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .expect("value must be Float64");

                let label_arrays: Vec<&arrow::array::StringArray> = label_columns
                    .iter()
                    .map(|name| {
                        let col = batch
                            .column_by_name(name)
                            .unwrap_or_else(|| panic!("missing label column: {name}"));
                        col.as_any()
                            .downcast_ref::<arrow::array::StringArray>()
                            .unwrap_or_else(|| panic!("label column {name} must be Utf8"))
                    })
                    .collect();

                for row in 0..batch.num_rows() {
                    let key: Vec<String> = label_arrays
                        .iter()
                        .map(|arr| arr.value(row).to_string())
                        .collect();
                    let ts = ts_arr.value(row);
                    let val = val_arr.value(row);
                    series_map.entry(key).or_default().push((ts, val));
                }
            }

            // Sort each series by timestamp.
            for samples in series_map.values_mut() {
                samples.sort_by_key(|(ts, _)| *ts);
            }

            // Build output arrays: for each eval timestamp and each series,
            // collect samples in [t - range_ns, t] into list arrays.
            let mut out_ts = Int64Builder::new();
            let mut out_timestamps = ListBuilder::new(Int64Builder::new());
            let mut out_values = ListBuilder::new(Float64Builder::new());
            let mut out_labels: Vec<StringBuilder> =
                label_columns.iter().map(|_| StringBuilder::new()).collect();

            for &eval_ts in &eval_timestamps {
                let window_start = eval_ts - range_ns;
                for (key, samples) in &series_map {
                    // Find samples within [window_start, eval_ts].
                    let start_idx = samples.partition_point(|(ts, _)| *ts < window_start);
                    let end_idx = samples.partition_point(|(ts, _)| *ts <= eval_ts);

                    let window = &samples[start_idx..end_idx];
                    if window.is_empty() {
                        continue;
                    }

                    // Emit the evaluation timestamp.
                    out_ts.append_value(eval_ts);

                    // Emit window sample timestamps as a list.
                    for &(ts, _) in window {
                        out_timestamps.values().append_value(ts);
                    }
                    out_timestamps.append(true);

                    // Emit window sample values as a list.
                    for &(_, val) in window {
                        out_values.values().append_value(val);
                    }
                    out_values.append(true);

                    // Emit label values.
                    for (i, label_val) in key.iter().enumerate() {
                        out_labels[i].append_value(label_val);
                    }
                }
            }

            // Build output columns in schema order.
            let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();
            for field in output_schema.fields() {
                let name = field.name().as_str();
                if name == "timestamp" {
                    columns.push(Arc::new(out_ts.finish()));
                } else if name == "timestamps" {
                    columns.push(Arc::new(out_timestamps.finish()));
                } else if name == "values" {
                    columns.push(Arc::new(out_values.finish()));
                } else if let Some(idx) = label_columns.iter().position(|n| n == name) {
                    columns.push(Arc::new(out_labels[idx].finish()));
                }
            }

            let batch = RecordBatch::try_new(output_schema, columns)?;
            Ok(batch)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}
