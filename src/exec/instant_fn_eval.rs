use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int64Builder, StringBuilder};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::func::InstantFunction;

/// Physical plan node that applies an element-wise instant function to each sample value.
///
/// Passes timestamp and all label columns through unchanged; only the `value` column is
/// transformed by the function.
#[derive(Debug)]
pub(crate) struct InstantFnExec {
    child: Arc<dyn ExecutionPlan>,
    func: InstantFunction,
    properties: Arc<PlanProperties>,
}

impl InstantFnExec {
    pub fn new(child: Arc<dyn ExecutionPlan>, func: InstantFunction) -> Self {
        let schema = child.schema();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            child,
            func,
            properties,
        }
    }
}

impl DisplayAs for InstantFnExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InstantFnExec: func={}", self.func)
    }
}

impl ExecutionPlan for InstantFnExec {
    fn name(&self) -> &str {
        "InstantFnExec"
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

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(Arc::clone(&children[0]), self.func)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, Arc::clone(&context))?;
        let schema = self.schema();
        let func = self.func;

        // Collect label column names (everything except timestamp and value).
        let label_columns: Vec<String> = schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            let mut batches = Vec::new();
            let mut s = child_stream;
            while let Some(b) = s.next().await {
                batches.push(b?);
            }

            let mut out_ts = Int64Builder::new();
            let mut out_val = Float64Builder::new();
            let mut out_labels: Vec<StringBuilder> =
                label_columns.iter().map(|_| StringBuilder::new()).collect();

            for batch in &batches {
                let ts_arr = batch
                    .column_by_name("timestamp")
                    .expect("missing timestamp column")
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .expect("timestamp must be Int64");
                let val_arr = batch
                    .column_by_name("value")
                    .expect("missing value column")
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .expect("value must be Float64");

                let label_arrays: Vec<&arrow::array::StringArray> = label_columns
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
                    out_ts.append_value(ts_arr.value(row));
                    out_val.append_value(func.evaluate(val_arr.value(row)));
                    for (i, arr) in label_arrays.iter().enumerate() {
                        out_labels[i].append_value(arr.value(row));
                    }
                }
            }

            // Build output RecordBatch preserving the input schema column order.
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
