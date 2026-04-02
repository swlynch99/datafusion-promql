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
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::node::{BinaryOp, VectorMatching};

// ---------------------------------------------------------------------------
// BinaryExec: vector op vector
// ---------------------------------------------------------------------------

/// Physical plan node for binary operations between two instant vectors.
#[derive(Debug)]
pub(crate) struct BinaryExec {
    lhs: Arc<dyn ExecutionPlan>,
    rhs: Arc<dyn ExecutionPlan>,
    op: BinaryOp,
    return_bool: bool,
    matching: VectorMatching,
    lhs_label_columns: Vec<String>,
    rhs_label_columns: Vec<String>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl BinaryExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lhs: Arc<dyn ExecutionPlan>,
        rhs: Arc<dyn ExecutionPlan>,
        op: BinaryOp,
        return_bool: bool,
        matching: VectorMatching,
        output_schema: SchemaRef,
    ) -> Self {
        let lhs_label_columns: Vec<String> = lhs
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        let rhs_label_columns: Vec<String> = rhs
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));

        Self {
            lhs,
            rhs,
            op,
            return_bool,
            matching,
            lhs_label_columns,
            rhs_label_columns,
            output_schema,
            properties,
        }
    }
}

/// Extract the matching key from a row based on VectorMatching config.
fn matching_key(
    label_columns: &[String],
    label_values: &[String],
    matching: &VectorMatching,
) -> Vec<String> {
    match (&matching.on_labels, &matching.ignoring_labels) {
        (Some(on), _) => {
            // on(...): only specified labels
            on.iter()
                .map(|l| {
                    label_columns
                        .iter()
                        .position(|c| c == l)
                        .map(|i| label_values[i].clone())
                        .unwrap_or_default()
                })
                .collect()
        }
        (_, Some(ignoring)) => {
            // ignoring(...): all except specified and __name__
            label_columns
                .iter()
                .zip(label_values.iter())
                .filter(|(name, _)| {
                    !ignoring.contains(name) && name.as_str() != "__name__"
                })
                .map(|(_, v)| v.clone())
                .collect()
        }
        (None, None) => {
            // Default: all labels except __name__
            label_columns
                .iter()
                .zip(label_values.iter())
                .filter(|(name, _)| name.as_str() != "__name__")
                .map(|(_, v)| v.clone())
                .collect()
        }
    }
}

/// Collect series from batches: returns map from full_label_key -> Vec<(timestamp, value)>
/// and for each row also records the matching key.
fn collect_series(
    batches: &[RecordBatch],
    label_columns: &[String],
    matching: &VectorMatching,
) -> (
    HashMap<Vec<String>, Vec<(i64, f64)>>,
    HashMap<Vec<String>, Vec<Vec<String>>>,
) {
    // full_key -> samples
    let mut series_map: HashMap<Vec<String>, Vec<(i64, f64)>> = HashMap::new();
    // match_key -> list of full_keys that map to it
    let mut match_to_full: HashMap<Vec<String>, Vec<Vec<String>>> = HashMap::new();

    for batch in batches {
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

        let label_arrays: Vec<Option<&arrow::array::StringArray>> = label_columns
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
            let full_key: Vec<String> = label_arrays
                .iter()
                .map(|arr| arr.map(|a| a.value(row).to_string()).unwrap_or_default())
                .collect();
            let ts = ts_arr.value(row);
            let val = val_arr.value(row);

            let mk = matching_key(label_columns, &full_key, matching);

            series_map.entry(full_key.clone()).or_default().push((ts, val));

            let full_keys = match_to_full.entry(mk).or_default();
            if !full_keys.contains(&full_key) {
                full_keys.push(full_key);
            }
        }
    }

    (series_map, match_to_full)
}

/// Determine output label values from the left-side full key and matching config.
fn output_label_values(
    full_key: &[String],
    label_columns: &[String],
    output_labels: &[String],
) -> Vec<String> {
    output_labels
        .iter()
        .map(|label| {
            label_columns
                .iter()
                .position(|c| c == label)
                .map(|i| full_key[i].clone())
                .unwrap_or_default()
        })
        .collect()
}

impl DisplayAs for BinaryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryExec: op={}", self.op)
    }
}

impl ExecutionPlan for BinaryExec {
    fn name(&self) -> &str {
        "BinaryExec"
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
        vec![&self.lhs, &self.rhs]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            Arc::clone(&children[1]),
            self.op,
            self.return_bool,
            self.matching.clone(),
            Arc::clone(&self.output_schema),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let lhs_stream = self.lhs.execute(partition, Arc::clone(&context))?;
        let rhs_stream = self.rhs.execute(partition, Arc::clone(&context))?;
        let output_schema = Arc::clone(&self.output_schema);
        let op = self.op;
        let return_bool = self.return_bool;
        let matching = self.matching.clone();
        let lhs_label_columns = self.lhs_label_columns.clone();
        let rhs_label_columns = self.rhs_label_columns.clone();

        // Determine output label names (everything except timestamp/value in output schema).
        let output_labels: Vec<String> = output_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            // Collect all batches from both sides.
            let mut lhs_batches = Vec::new();
            let mut s = lhs_stream;
            while let Some(b) = s.next().await {
                lhs_batches.push(b?);
            }

            let mut rhs_batches = Vec::new();
            let mut s = rhs_stream;
            while let Some(b) = s.next().await {
                rhs_batches.push(b?);
            }

            let (lhs_series, lhs_match_to_full) =
                collect_series(&lhs_batches, &lhs_label_columns, &matching);
            let (rhs_series, rhs_match_to_full) =
                collect_series(&rhs_batches, &rhs_label_columns, &matching);

            let mut out_ts = Int64Builder::new();
            let mut out_val = Float64Builder::new();
            let mut out_labels: Vec<StringBuilder> =
                output_labels.iter().map(|_| StringBuilder::new()).collect();

            if op.is_set_operator() {
                // Set operators work at the series level.
                match op {
                    BinaryOp::Land => {
                        // Return LHS series that have a match in RHS.
                        for (mk, lhs_keys) in &lhs_match_to_full {
                            if rhs_match_to_full.contains_key(mk) {
                                for lhs_key in lhs_keys {
                                    let samples = &lhs_series[lhs_key];
                                    let out_vals =
                                        output_label_values(lhs_key, &lhs_label_columns, &output_labels);
                                    for &(ts, val) in samples {
                                        out_ts.append_value(ts);
                                        out_val.append_value(val);
                                        for (i, v) in out_vals.iter().enumerate() {
                                            out_labels[i].append_value(v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    BinaryOp::Lor => {
                        // Return all LHS series + RHS series without a match in LHS.
                        for (_, lhs_keys) in &lhs_match_to_full {
                            for lhs_key in lhs_keys {
                                let samples = &lhs_series[lhs_key];
                                let out_vals =
                                    output_label_values(lhs_key, &lhs_label_columns, &output_labels);
                                for &(ts, val) in samples {
                                    out_ts.append_value(ts);
                                    out_val.append_value(val);
                                    for (i, v) in out_vals.iter().enumerate() {
                                        out_labels[i].append_value(v);
                                    }
                                }
                            }
                        }
                        for (mk, rhs_keys) in &rhs_match_to_full {
                            if !lhs_match_to_full.contains_key(mk) {
                                for rhs_key in rhs_keys {
                                    let samples = &rhs_series[rhs_key];
                                    let out_vals =
                                        output_label_values(rhs_key, &rhs_label_columns, &output_labels);
                                    for &(ts, val) in samples {
                                        out_ts.append_value(ts);
                                        out_val.append_value(val);
                                        for (i, v) in out_vals.iter().enumerate() {
                                            out_labels[i].append_value(v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    BinaryOp::Lunless => {
                        // Return LHS series that do NOT have a match in RHS.
                        for (mk, lhs_keys) in &lhs_match_to_full {
                            if !rhs_match_to_full.contains_key(mk) {
                                for lhs_key in lhs_keys {
                                    let samples = &lhs_series[lhs_key];
                                    let out_vals =
                                        output_label_values(lhs_key, &lhs_label_columns, &output_labels);
                                    for &(ts, val) in samples {
                                        out_ts.append_value(ts);
                                        out_val.append_value(val);
                                        for (i, v) in out_vals.iter().enumerate() {
                                            out_labels[i].append_value(v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                // Arithmetic and comparison operators: match per timestamp.
                // Build timestamp-indexed maps for RHS.
                let mut rhs_by_match_and_ts: HashMap<Vec<String>, HashMap<i64, (f64, Vec<String>)>> =
                    HashMap::new();

                for (mk, rhs_keys) in &rhs_match_to_full {
                    for rhs_key in rhs_keys {
                        if let Some(samples) = rhs_series.get(rhs_key) {
                            let entry = rhs_by_match_and_ts.entry(mk.clone()).or_default();
                            for &(ts, val) in samples {
                                entry.insert(ts, (val, rhs_key.clone()));
                            }
                        }
                    }
                }

                // For each LHS series, find matching RHS and apply operation.
                // Sort LHS entries for deterministic output.
                let mut lhs_entries: Vec<_> = lhs_match_to_full.iter().collect();
                lhs_entries.sort_by(|a, b| a.0.cmp(b.0));

                for (mk, lhs_keys) in lhs_entries {
                    let rhs_ts_map = match rhs_by_match_and_ts.get(mk) {
                        Some(m) => m,
                        None => continue, // No match in RHS
                    };

                    for lhs_key in lhs_keys {
                        if let Some(samples) = lhs_series.get(lhs_key) {
                            let out_vals =
                                output_label_values(lhs_key, &lhs_label_columns, &output_labels);

                            let mut sorted_samples = samples.clone();
                            sorted_samples.sort_by_key(|(ts, _)| *ts);

                            for (ts, lhs_val) in sorted_samples {
                                if let Some((rhs_val, _)) = rhs_ts_map.get(&ts) {
                                    if return_bool && op.is_comparison() {
                                        let result = op.evaluate_bool(lhs_val, *rhs_val);
                                        out_ts.append_value(ts);
                                        out_val.append_value(result);
                                        for (i, v) in out_vals.iter().enumerate() {
                                            out_labels[i].append_value(v);
                                        }
                                    } else if let Some(result) = op.evaluate(lhs_val, *rhs_val) {
                                        out_ts.append_value(ts);
                                        out_val.append_value(result);
                                        for (i, v) in out_vals.iter().enumerate() {
                                            out_labels[i].append_value(v);
                                        }
                                    }
                                }
                            }
                        }
                    }
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

// ---------------------------------------------------------------------------
// ScalarBinaryExec: vector op scalar (or scalar op vector)
// ---------------------------------------------------------------------------

/// Physical plan node for binary operations between a vector and a scalar.
#[derive(Debug)]
pub(crate) struct ScalarBinaryExec {
    child: Arc<dyn ExecutionPlan>,
    scalar_value: f64,
    op: BinaryOp,
    scalar_is_lhs: bool,
    return_bool: bool,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl ScalarBinaryExec {
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        scalar_value: f64,
        op: BinaryOp,
        scalar_is_lhs: bool,
        return_bool: bool,
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
            scalar_value,
            op,
            scalar_is_lhs,
            return_bool,
            output_schema,
            properties,
        }
    }
}

impl DisplayAs for ScalarBinaryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scalar_is_lhs {
            write!(f, "ScalarBinaryExec: {} {} vector", self.scalar_value, self.op)
        } else {
            write!(f, "ScalarBinaryExec: vector {} {}", self.op, self.scalar_value)
        }
    }
}

impl ExecutionPlan for ScalarBinaryExec {
    fn name(&self) -> &str {
        "ScalarBinaryExec"
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
            self.scalar_value,
            self.op,
            self.scalar_is_lhs,
            self.return_bool,
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
        let op = self.op;
        let scalar_value = self.scalar_value;
        let scalar_is_lhs = self.scalar_is_lhs;
        let return_bool = self.return_bool;

        // Output label columns (everything except timestamp/value in output schema).
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
            let mut out_labels_builders: Vec<StringBuilder> =
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
                    let vec_val = val_arr.value(row);

                    let (lhs, rhs) = if scalar_is_lhs {
                        (scalar_value, vec_val)
                    } else {
                        (vec_val, scalar_value)
                    };

                    let result = if return_bool && op.is_comparison() {
                        Some(op.evaluate_bool(lhs, rhs))
                    } else {
                        op.evaluate(lhs, rhs)
                    };

                    if let Some(val) = result {
                        out_ts.append_value(ts);
                        out_val.append_value(val);
                        for (i, arr) in label_arrays.iter().enumerate() {
                            let v = arr.map(|a| a.value(row)).unwrap_or("");
                            out_labels_builders[i].append_value(v);
                        }
                    }
                }
            }

            let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();
            columns.push(Arc::new(out_ts.finish()));
            columns.push(Arc::new(out_val.finish()));
            for builder in &mut out_labels_builders {
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
