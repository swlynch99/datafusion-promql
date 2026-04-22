use std::any::Any;
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, Float64Builder, StringBuilder, UInt64Builder};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    ConstExpr, EquivalenceProperties, Partitioning, PhysicalSortExpr,
};
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::node::WideColumnMeta;

/// Physical plan node that unpacks wide-format data into long format.
///
/// Reads `RecordBatch`es from a single child (a wide-format table scan with
/// columns `[timestamp, col_0, col_1, ..., col_N]`) and produces long-format
/// output `[timestamp, value, __name__, label_key_0, ...]`.
///
/// For each input row, emits N output rows (one per value column). Columns
/// are processed in order, so the output is grouped by series with timestamps
/// in the same order as the input within each group.
#[derive(Debug)]
pub(crate) struct WideUnpackExec {
    child: Arc<dyn ExecutionPlan>,
    columns: Vec<WideColumnMeta>,
    label_keys: Vec<String>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

fn compute_output_schema(label_keys: &[String]) -> SchemaRef {
    let mut fields = vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, true),
        Field::new("__name__", DataType::Utf8, false),
    ];
    for key in label_keys {
        fields.push(Field::new(key, DataType::Utf8, true));
    }
    Arc::new(Schema::new(fields))
}

impl WideUnpackExec {
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        mut columns: Vec<WideColumnMeta>,
        label_keys: Vec<String>,
    ) -> Self {
        let output_schema = compute_output_schema(&label_keys);

        // Sort so the per-column emission loop produces label-sorted output.
        columns.sort_by(|a, b| {
            for key in &label_keys {
                let av = a.labels.get(key).map(|s| s.as_str()).unwrap_or("");
                let bv = b.labels.get(key).map(|s| s.as_str()).unwrap_or("");
                match av.cmp(bv) {
                    Ordering::Equal => continue,
                    ord => return ord,
                }
            }
            a.metric_name.cmp(&b.metric_name)
        });

        let asc_nulls_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let mut ordering: Vec<PhysicalSortExpr> = Vec::with_capacity(label_keys.len() + 1);
        for key in &label_keys {
            if let Ok(col) = Column::new_with_schema(key, output_schema.as_ref()) {
                ordering.push(PhysicalSortExpr::new(Arc::new(col), asc_nulls_last));
            }
        }
        if let Ok(ts_col) = Column::new_with_schema("timestamp", output_schema.as_ref()) {
            ordering.push(PhysicalSortExpr::new(Arc::new(ts_col), asc_nulls_last));
        }

        let mut eq_properties = if ordering.is_empty() {
            EquivalenceProperties::new(Arc::clone(&output_schema))
        } else {
            EquivalenceProperties::new_with_orderings(Arc::clone(&output_schema), [ordering])
        };
        // A single WideUnpack handles one metric, so __name__ is constant.
        if let Ok(name_col) = Column::new_with_schema("__name__", output_schema.as_ref()) {
            let _ = eq_properties.add_constants([ConstExpr::from(
                Arc::new(name_col) as Arc<dyn datafusion::physical_expr::PhysicalExpr>
            )]);
        }

        let properties = Arc::new(PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            child,
            columns,
            label_keys,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for WideUnpackExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WideUnpackExec: {} columns, labels=[{}]",
            self.columns.len(),
            self.label_keys.join(", ")
        )
    }
}

impl ExecutionPlan for WideUnpackExec {
    fn name(&self) -> &str {
        "WideUnpackExec"
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

    fn required_input_ordering(
        &self,
    ) -> Vec<Option<datafusion::physical_expr::OrderingRequirements>> {
        // Require input sorted by timestamp so that each column group in the
        // output is also timestamp-sorted.
        vec![super::label_timestamp_ordering(&[], &self.child.schema())]
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
            self.columns.clone(),
            self.label_keys.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, Arc::clone(&context))?;
        let output_schema = Arc::clone(&self.output_schema);
        let columns_meta = self.columns.clone();
        let label_keys = self.label_keys.clone();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            // Collect all batches from the child stream.
            let mut batches = Vec::new();
            let mut stream = child_stream;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }

            // Count total input rows to pre-allocate.
            let total_input_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            let total_output_rows = total_input_rows * columns_meta.len();

            let mut out_ts = UInt64Builder::with_capacity(total_output_rows);
            let mut out_val = Float64Builder::with_capacity(total_output_rows);
            let mut out_name = StringBuilder::with_capacity(total_output_rows, 0);
            let mut out_labels: Vec<StringBuilder> = label_keys
                .iter()
                .map(|_| StringBuilder::with_capacity(total_output_rows, 0))
                .collect();

            // Process one column at a time. For each column, iterate over all
            // input rows and emit one output row per row. This produces output
            // grouped by series (column), with timestamps in input order within
            // each group.
            for col_meta in &columns_meta {
                // Pre-compute label values for this column (constant for all rows).
                let label_values: Vec<&str> = label_keys
                    .iter()
                    .map(|key| col_meta.labels.get(key).map(|s| s.as_str()).unwrap_or(""))
                    .collect();

                for batch in &batches {
                    let ts_col = batch
                        .column_by_name("timestamp")
                        .expect("missing timestamp column");
                    let ts_arr = ts_col
                        .as_any()
                        .downcast_ref::<arrow::array::UInt64Array>()
                        .expect("timestamp must be UInt64");

                    // Find the value column by name in the input batch.
                    let val_col = batch
                        .column_by_name(&col_meta.col_name)
                        .unwrap_or_else(|| panic!("missing value column: {}", col_meta.col_name));
                    let val_arr = val_col
                        .as_any()
                        .downcast_ref::<arrow::array::Float64Array>()
                        .unwrap_or_else(|| {
                            panic!("value column {} must be Float64", col_meta.col_name)
                        });

                    for row in 0..batch.num_rows() {
                        out_ts.append_value(ts_arr.value(row));
                        if val_arr.is_null(row) {
                            out_val.append_null();
                        } else {
                            out_val.append_value(val_arr.value(row));
                        }
                        out_name.append_value(&col_meta.metric_name);
                        for (i, label_val) in label_values.iter().enumerate() {
                            out_labels[i].append_value(label_val);
                        }
                    }
                }
            }

            // Build output RecordBatch.
            let mut arrays: Vec<arrow::array::ArrayRef> = vec![
                Arc::new(out_ts.finish()),
                Arc::new(out_val.finish()),
                Arc::new(out_name.finish()),
            ];
            for builder in &mut out_labels {
                arrays.push(Arc::new(builder.finish()));
            }

            let batch = RecordBatch::try_new(Arc::clone(&output_schema), arrays)?;
            Ok(batch)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            stream,
        )))
    }
}
