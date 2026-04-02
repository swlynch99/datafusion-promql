use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::func::AggregateFunction;

/// Physical plan node that applies a PromQL aggregation operator.
///
/// Groups input rows by (timestamp, grouping_labels) and applies the
/// aggregation function to each group's values.
#[derive(Debug)]
pub(crate) struct AggregateExec {
    child: Arc<dyn ExecutionPlan>,
    func: AggregateFunction,
    grouping_labels: Vec<String>,
    /// All label columns from the child schema (for reading input rows).
    input_label_columns: Vec<String>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl AggregateExec {
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        func: AggregateFunction,
        grouping_labels: Vec<String>,
    ) -> Self {
        // Determine input label columns from child schema.
        let child_schema = child.schema();
        let input_label_columns: Vec<String> = child_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        // Build output schema.
        let mut fields = vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ];
        for label in &grouping_labels {
            fields.push(Field::new(label, DataType::Utf8, false));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));

        Self {
            child,
            func,
            grouping_labels,
            input_label_columns,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for AggregateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AggregateExec: func={}, by=[{}]",
            self.func,
            self.grouping_labels.join(", ")
        )
    }
}

impl ExecutionPlan for AggregateExec {
    fn name(&self) -> &str {
        "AggregateExec"
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
            self.grouping_labels.clone(),
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
        let grouping_labels = self.grouping_labels.clone();
        let _input_label_columns = self.input_label_columns.clone();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            // Collect all batches from the child stream.
            let mut batches = Vec::new();
            let mut stream = child_stream;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }

            // Build a map: (timestamp, grouping_key) -> Vec<f64>
            // grouping_key is a Vec<String> of grouping label values.
            let mut group_map: HashMap<(i64, Vec<String>), Vec<f64>> = HashMap::new();

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

                // Build label column arrays lookup.
                // For each grouping label, find its position in the input.
                let grouping_arrays: Vec<Option<&arrow::array::StringArray>> = grouping_labels
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

                    let key: Vec<String> = grouping_arrays
                        .iter()
                        .map(|arr| arr.map(|a| a.value(row).to_string()).unwrap_or_default())
                        .collect();

                    group_map.entry((ts, key)).or_default().push(val);
                }
            }

            // Apply aggregation to each group and build output.
            let mut out_ts = Int64Builder::new();
            let mut out_val = Float64Builder::new();
            let mut out_labels: Vec<StringBuilder> = grouping_labels
                .iter()
                .map(|_| StringBuilder::new())
                .collect();

            // Sort keys for deterministic output.
            let mut entries: Vec<_> = group_map.into_iter().collect();
            entries.sort_by(|a, b| a.0.0.cmp(&b.0.0).then(a.0.1.cmp(&b.0.1)));

            for ((ts, key), values) in entries {
                if values.is_empty() {
                    continue;
                }
                let agg_value = func.evaluate(&values);
                out_ts.append_value(ts);
                out_val.append_value(agg_value);
                for (i, label_val) in key.iter().enumerate() {
                    out_labels[i].append_value(label_val);
                }
            }

            // Build output RecordBatch.
            let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();
            columns.push(Arc::new(out_ts.finish()));
            columns.push(Arc::new(out_val.finish()));
            for builder in &mut out_labels {
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
