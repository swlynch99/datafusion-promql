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

/// Physical plan node for instant vector functions (e.g., `abs`, `ceil`, `floor`, `ln`).
///
/// Applies `func` to the `value` column of each row from the child plan,
/// forwarding timestamps and label columns unchanged. Columns absent from the
/// output schema (e.g. `__name__`) are dropped.
#[derive(Debug)]
pub(crate) struct InstantFuncExec {
    child: Arc<dyn ExecutionPlan>,
    func: InstantFunction,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl InstantFuncExec {
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        func: InstantFunction,
        output_schema: SchemaRef,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            child,
            func,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for InstantFuncExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InstantFuncExec: func={}", self.func)
    }
}

impl ExecutionPlan for InstantFuncExec {
    fn name(&self) -> &str {
        "InstantFuncExec"
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
            Arc::clone(&self.output_schema),
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

        // Label columns in the output (everything except timestamp/value).
        let output_labels: Vec<String> = output_schema
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
            let mut out_label_builders: Vec<StringBuilder> =
                output_labels.iter().map(|_| StringBuilder::new()).collect();

            for batch in &batches {
                let ts_arr = batch
                    .column_by_name("timestamp")
                    .expect("missing timestamp")
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .expect("timestamp must be Int64");
                let val_arr = batch
                    .column_by_name("value")
                    .expect("missing value")
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .expect("value must be Float64");

                let label_arrays: Vec<Option<&arrow::array::StringArray>> = output_labels
                    .iter()
                    .map(|name| {
                        batch.column_by_name(name).map(|col| {
                            col.as_any()
                                .downcast_ref::<arrow::array::StringArray>()
                                .unwrap_or_else(|| panic!("label column {name} must be Utf8"))
                        })
                    })
                    .collect();

                for row in 0..batch.num_rows() {
                    let ts = ts_arr.value(row);
                    let val = val_arr.value(row);

                    out_ts.append_value(ts);
                    out_val.append_value(func.evaluate(val));
                    for (i, arr) in label_arrays.iter().enumerate() {
                        let v = arr.map(|a| a.value(row)).unwrap_or("");
                        out_label_builders[i].append_value(v);
                    }
                }
            }

            let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();
            columns.push(Arc::new(out_ts.finish()));
            columns.push(Arc::new(out_val.finish()));
            for builder in &mut out_label_builders {
                columns.push(Arc::new(builder.finish()));
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
