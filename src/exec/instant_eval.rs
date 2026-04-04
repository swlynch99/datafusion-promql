use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int64Builder, StringBuilder};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

/// Physical plan node that aligns raw samples to evaluation timestamps
/// using the lookback window.
///
/// For each series (grouped by label columns), picks the most recent sample
/// within `[eval_ts - lookback, eval_ts]`.
#[derive(Debug)]
pub(crate) struct InstantVectorExec {
    child: Arc<dyn ExecutionPlan>,
    eval_ts_ns: Option<i64>,
    #[allow(dead_code)]
    start_ns: i64,
    #[allow(dead_code)]
    end_ns: i64,
    #[allow(dead_code)]
    step_ns: i64,
    lookback_ns: i64,
    label_columns: Vec<String>,
    properties: Arc<PlanProperties>,
}

impl InstantVectorExec {
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        eval_ts_ns: Option<i64>,
        start_ns: i64,
        end_ns: i64,
        step_ns: i64,
        lookback_ns: i64,
        label_columns: Vec<String>,
    ) -> Self {
        let schema = child.schema();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            child,
            eval_ts_ns,
            start_ns,
            end_ns,
            step_ns,
            lookback_ns,
            label_columns,
            properties,
        }
    }

    /// Generate the list of evaluation timestamps.
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

impl DisplayAs for InstantVectorExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InstantVectorExec")
    }
}

impl ExecutionPlan for InstantVectorExec {
    fn name(&self) -> &str {
        "InstantVectorExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.child.schema()
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
            self.eval_ts_ns,
            self.start_ns,
            self.end_ns,
            self.step_ns,
            self.lookback_ns,
            self.label_columns.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, Arc::clone(&context))?;
        let schema = self.schema();
        let eval_timestamps = self.eval_timestamps();
        let lookback_ns = self.lookback_ns;
        let label_columns = self.label_columns.clone();

        let stream = futures::stream::once(async move {
            // Collect all batches from the child stream.
            use futures::StreamExt;
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

                // Get label column arrays.
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

            // For each eval timestamp and each series, find the most recent
            // sample within [eval_ts - lookback, eval_ts].
            let mut out_ts = Int64Builder::new();
            let mut out_val = Float64Builder::new();
            let mut out_labels: Vec<StringBuilder> =
                label_columns.iter().map(|_| StringBuilder::new()).collect();

            for &eval_ts in &eval_timestamps {
                let window_start = eval_ts - lookback_ns;
                for (key, samples) in &series_map {
                    // Binary search for the last sample <= eval_ts.
                    let pos = samples.partition_point(|(ts, _)| *ts <= eval_ts);
                    if pos == 0 {
                        continue; // No sample at or before eval_ts.
                    }
                    let (sample_ts, sample_val) = samples[pos - 1];
                    if sample_ts < window_start {
                        continue; // Sample is outside the lookback window.
                    }

                    out_ts.append_value(eval_ts);
                    out_val.append_value(sample_val);
                    for (i, label_val) in key.iter().enumerate() {
                        out_labels[i].append_value(label_val);
                    }
                }
            }

            // Build output RecordBatch with the same schema as input.
            let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();

            for field in schema.fields() {
                let name = field.name().as_str();
                if name == "timestamp" {
                    columns.push(Arc::new(out_ts.finish()));
                } else if name == "value" {
                    columns.push(Arc::new(out_val.finish()));
                } else if let Some(idx) = label_columns.iter().position(|n| n == name) {
                    columns.push(Arc::new(out_labels[idx].finish()));
                } else {
                    columns.push(arrow::array::new_empty_array(field.data_type()));
                }
            }

            let batch = RecordBatch::try_new(schema, columns)?;
            Ok(batch)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}
