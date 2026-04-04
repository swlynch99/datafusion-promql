use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, OrderingRequirements, Partitioning};
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use super::label_timestamp_ordering;

/// Physical plan node that aligns raw samples to a single evaluation timestamp
/// using the lookback window.
///
/// For each series (grouped by label columns), picks the most recent sample
/// within `[eval_ts - lookback, eval_ts]`.
///
/// This is used for instant queries. For range queries that evaluate over
/// multiple step timestamps, see [`super::StepVectorExec`].
#[derive(Debug)]
pub(crate) struct InstantVectorExec {
    child: Arc<dyn ExecutionPlan>,
    timestamp_ns: u64,
    lookback_ns: u64,
    offset_ns: i64,
    label_columns: Vec<String>,
    properties: Arc<PlanProperties>,
}

impl InstantVectorExec {
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        timestamp_ns: u64,
        lookback_ns: u64,
        offset_ns: i64,
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
            timestamp_ns,
            lookback_ns,
            offset_ns,
            label_columns,
            properties,
        }
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
            self.timestamp_ns,
            self.lookback_ns,
            self.offset_ns,
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
        let eval_ts = self.timestamp_ns;
        let lookback_ns = self.lookback_ns;
        let offset_ns = self.offset_ns;
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
            let mut series_map: HashMap<Vec<String>, Vec<(u64, f64)>> = HashMap::new();

            for batch in &batches {
                let ts_col = batch
                    .column_by_name("timestamp")
                    .expect("missing timestamp column");
                let val_col = batch.column_by_name("value").expect("missing value column");

                let ts_arr = ts_col
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .expect("timestamp must be UInt64");
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

            // For each series, find the most recent sample within
            // [eval_ts - offset - lookback, eval_ts - offset].
            let mut out_ts = UInt64Builder::new();
            let mut out_val = Float64Builder::new();
            let mut out_labels: Vec<StringBuilder> =
                label_columns.iter().map(|_| StringBuilder::new()).collect();

            // Apply offset: shift the lookup window into the past by offset_ns.
            // The effective lookup time is eval_ts - offset_ns, but the
            // result is reported at eval_ts.
            let effective_ts = (eval_ts as i64 - offset_ns) as u64;
            let window_start = effective_ts.saturating_sub(lookback_ns);
            for (key, samples) in &series_map {
                // Binary search for the last sample <= effective_ts.
                let pos = samples.partition_point(|(ts, _)| *ts <= effective_ts);
                if pos == 0 {
                    continue; // No sample at or before effective_ts.
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
