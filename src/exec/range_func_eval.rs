use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{AsArray, Float64Builder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::func::RangeFunction;

/// Physical plan node that applies a range function to pre-windowed data.
///
/// Consumes batches with `timestamp` (Int64), `timestamps` (List<Int64>),
/// `values` (List<Float64>), and label columns. For each row, applies the
/// range function to the (timestamps, values) arrays and outputs a scalar
/// result.
#[derive(Debug)]
pub(crate) struct RangeFunctionExec {
    child: Arc<dyn ExecutionPlan>,
    func: RangeFunction,
    label_columns: Vec<String>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

/// Build the output schema: timestamp, value, label columns.
fn compute_output_schema(label_columns: &[String]) -> SchemaRef {
    let mut fields = vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
    ];
    for label in label_columns {
        fields.push(Field::new(label, DataType::Utf8, true));
    }
    Arc::new(Schema::new(fields))
}

impl RangeFunctionExec {
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        func: RangeFunction,
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
            func,
            label_columns,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for RangeFunctionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RangeFunctionExec: func={}", self.func)
    }
}

impl ExecutionPlan for RangeFunctionExec {
    fn name(&self) -> &str {
        "RangeFunctionExec"
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
            self.func,
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
        let label_columns = self.label_columns.clone();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            // Collect all batches from the child stream.
            let mut batches = Vec::new();
            let mut stream = child_stream;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }

            let mut out_ts = Int64Builder::new();
            let mut out_val = Float64Builder::new();
            let mut out_labels: Vec<StringBuilder> =
                label_columns.iter().map(|_| StringBuilder::new()).collect();

            for batch in &batches {
                let ts_col = batch
                    .column_by_name("timestamp")
                    .expect("missing timestamp column");
                let timestamps_col = batch
                    .column_by_name("timestamps")
                    .expect("missing timestamps column");
                let values_col = batch
                    .column_by_name("values")
                    .expect("missing values column");

                let ts_arr = ts_col
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .expect("timestamp must be Int64");

                let timestamps_list = timestamps_col.as_list::<i32>();
                let values_list = values_col.as_list::<i32>();

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
                    // Extract the window arrays for this row.
                    let ts_list = timestamps_list.value(row);
                    let val_list = values_list.value(row);
                    let window_ts = ts_list.as_primitive::<Int64Type>();
                    let window_val = val_list.as_primitive::<Float64Type>();

                    // Build the (timestamp, value) pairs for the function.
                    let samples: Vec<(i64, f64)> = window_ts
                        .values()
                        .iter()
                        .zip(window_val.values().iter())
                        .map(|(&t, &v)| (t, v))
                        .collect();

                    if let Some(value) = func.evaluate(&samples) {
                        out_ts.append_value(ts_arr.value(row));
                        out_val.append_value(value);
                        for (i, label_arr) in label_arrays.iter().enumerate() {
                            out_labels[i].append_value(label_arr.value(row));
                        }
                    }
                }
            }

            // Build output columns in schema order.
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

            let batch = RecordBatch::try_new(output_schema, columns)?;
            Ok(batch)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}
