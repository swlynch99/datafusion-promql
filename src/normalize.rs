use std::any::Any;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int64Builder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::datasource::provider_as_source;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder, TableType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use crate::datasource::{ColumnMapping, MatchOp, Matcher};
use crate::error::{PromqlError, Result};
use crate::types::Labels;

/// A column that matched the requested metric, with its parsed labels.
#[derive(Debug, Clone)]
struct MatchedColumn {
    /// The original column name in the wide-format table.
    col_name: String,
    /// The labels parsed from the column name.
    labels: Labels,
}

/// Analyze the wide-format schema and find columns matching the given metric.
fn find_matching_columns(
    provider: &dyn TableProvider,
    mapping: &ColumnMapping,
    metric_name: &str,
    matchers: &[Matcher],
) -> Result<(Vec<MatchedColumn>, BTreeSet<String>)> {
    let schema = provider.schema();
    let ignore: BTreeSet<&str> = mapping
        .ignore_columns
        .iter()
        .map(|s| s.as_str())
        .collect();

    let mut matched: Vec<MatchedColumn> = Vec::new();

    for field in schema.fields() {
        let col_name = field.name().as_str();

        if col_name == mapping.timestamp_column || ignore.contains(col_name) {
            continue;
        }

        // Skip non-numeric columns.
        match field.data_type() {
            DataType::UInt64 | DataType::Int64 | DataType::Float64 => {}
            _ => continue,
        }

        let (parsed_metric, labels) = match (mapping.parse_column)(col_name) {
            Some(pair) => pair,
            None => continue,
        };

        if parsed_metric != metric_name {
            continue;
        }

        if !labels_match_matchers(&labels, matchers) {
            continue;
        }

        matched.push(MatchedColumn {
            col_name: col_name.to_string(),
            labels,
        });
    }

    if matched.is_empty() {
        return Err(PromqlError::DataSource(format!(
            "metric '{metric_name}' not found in wide-format table"
        )));
    }

    let all_label_keys: BTreeSet<String> = matched
        .iter()
        .flat_map(|m| m.labels.keys().cloned())
        .collect();

    Ok((matched, all_label_keys))
}

/// Build the long-format output schema: __name__, timestamp, value, plus label columns.
fn build_long_schema(metric_name: &str, all_label_keys: &BTreeSet<String>) -> SchemaRef {
    let _ = metric_name;
    let mut fields = vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ];
    for key in all_label_keys {
        fields.push(Field::new(key, DataType::Utf8, false));
    }
    Arc::new(Schema::new(fields))
}

/// Convert a wide-format `TableProvider` into a long-format one that can be
/// used directly by the selector.
///
/// Returns `(LogicalPlan, label_column_names)` where the plan scans a
/// [`NormalizedTableProvider`] that internally reads the wide table and
/// unpivots matching columns into long format.
pub(crate) fn normalize_wide_to_long(
    provider: Arc<dyn TableProvider>,
    mapping: &ColumnMapping,
    metric_name: &str,
    matchers: &[Matcher],
) -> Result<(LogicalPlan, Vec<String>)> {
    let (matched, all_label_keys) =
        find_matching_columns(provider.as_ref(), mapping, metric_name, matchers)?;

    let long_schema = build_long_schema(metric_name, &all_label_keys);

    // Compute which wide-format column indices we need to read.
    let wide_schema = provider.schema();
    let ts_idx = wide_schema
        .index_of(&mapping.timestamp_column)
        .map_err(|e| PromqlError::Plan(format!("timestamp column not found: {e}")))?;

    let mut projection_indices: Vec<usize> = vec![ts_idx];
    for mc in &matched {
        let idx = wide_schema
            .index_of(&mc.col_name)
            .map_err(|e| PromqlError::Plan(format!("metric column not found: {e}")))?;
        projection_indices.push(idx);
    }

    let normalized = NormalizedTableProvider {
        wide_provider: provider,
        wide_projection: projection_indices,
        long_schema: Arc::clone(&long_schema),
        metric_name: metric_name.to_string(),
        matched_columns: matched,
        all_label_keys: all_label_keys.iter().cloned().collect(),
        ts_column_name: mapping.timestamp_column.clone(),
    };

    let table_source = provider_as_source(Arc::new(normalized));
    let plan = LogicalPlanBuilder::scan_with_filters(metric_name, table_source, None, vec![])
        .map_err(|e| PromqlError::Plan(format!("failed to build table scan: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("failed to build plan: {e}")))?;

    let mut label_columns: Vec<String> = vec!["__name__".to_string()];
    label_columns.extend(all_label_keys);

    Ok((plan, label_columns))
}

/// A [`TableProvider`] that reads from a wide-format table and returns
/// long-format (unpivoted) data.
#[derive(Debug)]
struct NormalizedTableProvider {
    wide_provider: Arc<dyn TableProvider>,
    wide_projection: Vec<usize>,
    long_schema: SchemaRef,
    metric_name: String,
    matched_columns: Vec<MatchedColumn>,
    all_label_keys: Vec<String>,
    ts_column_name: String,
}

#[async_trait]
impl TableProvider for NormalizedTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.long_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn datafusion::catalog::Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[datafusion::prelude::Expr],
        _limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        // Scan the wide table with only the columns we need.
        let wide_exec = self
            .wide_provider
            .scan(state, Some(&self.wide_projection), &[], None)
            .await?;

        Ok(Arc::new(UnpivotExec {
            child: wide_exec,
            long_schema: Arc::clone(&self.long_schema),
            metric_name: self.metric_name.clone(),
            matched_columns: self.matched_columns.clone(),
            all_label_keys: self.all_label_keys.clone(),
            ts_column_name: self.ts_column_name.clone(),
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(Arc::clone(&self.long_schema)),
                Partitioning::UnknownPartitioning(1),
                datafusion::physical_plan::execution_plan::EmissionType::Final,
                datafusion::physical_plan::execution_plan::Boundedness::Bounded,
            )),
        }))
    }
}

/// Physical plan node that unpivots wide-format RecordBatches into long format.
#[derive(Debug)]
struct UnpivotExec {
    child: Arc<dyn ExecutionPlan>,
    long_schema: SchemaRef,
    metric_name: String,
    matched_columns: Vec<MatchedColumn>,
    all_label_keys: Vec<String>,
    ts_column_name: String,
    properties: Arc<PlanProperties>,
}

impl DisplayAs for UnpivotExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UnpivotExec: metric={}, columns={}",
            self.metric_name,
            self.matched_columns.len()
        )
    }
}

impl ExecutionPlan for UnpivotExec {
    fn name(&self) -> &str {
        "UnpivotExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.long_schema)
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
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            child: Arc::clone(&children[0]),
            long_schema: Arc::clone(&self.long_schema),
            metric_name: self.metric_name.clone(),
            matched_columns: self.matched_columns.clone(),
            all_label_keys: self.all_label_keys.clone(),
            ts_column_name: self.ts_column_name.clone(),
            properties: Arc::clone(&self.properties),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<datafusion::execution::SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, context)?;
        let long_schema = Arc::clone(&self.long_schema);
        let stream_schema = Arc::clone(&long_schema);
        let metric_name = self.metric_name.clone();
        let matched_columns = self.matched_columns.clone();
        let all_label_keys = self.all_label_keys.clone();
        let ts_column_name = self.ts_column_name.clone();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;
            let mut batches = Vec::new();
            let mut stream = child_stream;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }

            unpivot_batches(
                &batches,
                &long_schema,
                &metric_name,
                &matched_columns,
                &all_label_keys,
                &ts_column_name,
            )
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(stream_schema, stream)))
    }
}

/// Unpivot wide-format RecordBatches into a single long-format RecordBatch.
fn unpivot_batches(
    batches: &[RecordBatch],
    long_schema: &SchemaRef,
    metric_name: &str,
    matched_columns: &[MatchedColumn],
    all_label_keys: &[String],
    ts_column_name: &str,
) -> datafusion::common::Result<RecordBatch> {
    // Count total output rows: num_input_rows * num_matched_columns.
    let total_input_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let total_output_rows = total_input_rows * matched_columns.len();

    let mut name_builder = StringBuilder::with_capacity(total_output_rows, metric_name.len() * total_output_rows);
    let mut ts_builder = Int64Builder::with_capacity(total_output_rows);
    let mut val_builder = Float64Builder::with_capacity(total_output_rows);

    // One builder per label column.
    let mut label_builders: Vec<StringBuilder> = all_label_keys
        .iter()
        .map(|_| StringBuilder::with_capacity(total_output_rows, 32))
        .collect();

    for batch in batches {
        // Find the timestamp column in this batch.
        let ts_col = batch
            .column_by_name(ts_column_name)
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "missing timestamp column '{ts_column_name}'"
                ))
            })?;

        // Timestamp may be UInt64 or Int64 — handle both.
        let ts_values: Vec<i64> = if let Some(arr) = ts_col
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
        {
            (0..batch.num_rows())
                .map(|i| (arr.value(i) / 1_000_000) as i64)
                .collect()
        } else if let Some(arr) = ts_col
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
        {
            (0..batch.num_rows())
                .map(|i| arr.value(i) / 1_000_000)
                .collect()
        } else {
            return Err(datafusion::error::DataFusionError::Internal(
                "timestamp column must be UInt64 or Int64".into(),
            ));
        };

        // For each matched metric column, emit one row per input row.
        for mc in matched_columns {
            let val_col = batch.column_by_name(&mc.col_name).ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "missing metric column '{}'",
                    mc.col_name
                ))
            })?;

            // Cast to f64 values.
            let f64_values: Vec<f64> =
                if let Some(arr) = val_col.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                    (0..batch.num_rows())
                        .map(|i| arr.value(i) as f64)
                        .collect()
                } else if let Some(arr) =
                    val_col.as_any().downcast_ref::<arrow::array::Int64Array>()
                {
                    (0..batch.num_rows())
                        .map(|i| arr.value(i) as f64)
                        .collect()
                } else if let Some(arr) =
                    val_col.as_any().downcast_ref::<arrow::array::Float64Array>()
                {
                    (0..batch.num_rows())
                        .map(|i| arr.value(i))
                        .collect()
                } else {
                    return Err(datafusion::error::DataFusionError::Internal(format!(
                        "metric column '{}' has unsupported type",
                        mc.col_name
                    )));
                };

            for row in 0..batch.num_rows() {
                name_builder.append_value(metric_name);
                ts_builder.append_value(ts_values[row]);
                val_builder.append_value(f64_values[row]);

                for (i, key) in all_label_keys.iter().enumerate() {
                    let val = mc.labels.get(key).map(|s| s.as_str()).unwrap_or("");
                    label_builders[i].append_value(val);
                }
            }
        }
    }

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(name_builder.finish()),
        Arc::new(ts_builder.finish()),
        Arc::new(val_builder.finish()),
    ];
    for builder in &mut label_builders {
        columns.push(Arc::new(builder.finish()));
    }

    RecordBatch::try_new(Arc::clone(long_schema), columns).map_err(|e| {
        datafusion::error::DataFusionError::Internal(format!("failed to build batch: {e}"))
    })
}

/// Check whether parsed labels satisfy all matchers.
fn labels_match_matchers(labels: &Labels, matchers: &[Matcher]) -> bool {
    for m in matchers {
        let label_val = labels.get(&m.name).map(|s| s.as_str()).unwrap_or("");
        match m.op {
            MatchOp::Equal => {
                if label_val != m.value {
                    return false;
                }
            }
            MatchOp::NotEqual => {
                if label_val == m.value {
                    return false;
                }
            }
            MatchOp::RegexMatch | MatchOp::RegexNotMatch => {}
        }
    }
    true
}
