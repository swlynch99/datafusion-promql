use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::error::{PromqlError, Result};
use crate::types::Labels;

/// Metadata for a single wide-format value column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WideColumnMeta {
    /// The original column name in the wide-format table.
    pub col_name: String,
    /// The PromQL metric name parsed from this column.
    pub metric_name: String,
    /// Label key/value pairs parsed from this column's name or metadata.
    pub labels: Labels,
}

/// Custom logical node that unpacks a wide-format plan into long format.
///
/// Input: a plan whose schema has `(timestamp: UInt64, col_0: F64, col_1: F64, ...)`
/// — one timestamp column and N value columns, all from a single table scan.
///
/// Output: `(timestamp: UInt64, value: Float64, __name__: Utf8, label_key_0: Utf8, ...)`
/// — standard long format with one row per (timestamp, series) pair.
///
/// For each input row, this node logically produces N output rows, one per
/// value column. Each output row carries the value from that column and
/// constant string literals for the metric name and labels (derived from
/// [`WideColumnMeta`]).
///
/// Output ordering: rows are grouped by column (i.e., all rows for column 0,
/// then all rows for column 1, etc.). Within each column group, rows are in
/// the same timestamp order as the input. Since each column's labels are
/// constant, this is equivalent to `(label_columns ASC, timestamp ASC)`.
#[derive(Debug, Clone)]
pub struct WideUnpack {
    /// The wide-format input plan.
    pub input: LogicalPlan,
    /// Metadata for each value column to unpack.
    pub columns: Arc<Vec<WideColumnMeta>>,
    /// The union of all label keys across all columns, sorted.
    pub label_keys: Arc<Vec<String>>,
    /// Output schema: (timestamp, value, __name__, label_key_0, ...).
    pub output_schema: DFSchemaRef,
}

impl WideUnpack {
    pub fn new(
        input: LogicalPlan,
        columns: Arc<Vec<WideColumnMeta>>,
        label_keys: Arc<Vec<String>>,
    ) -> Result<Self> {
        let output_schema = compute_output_schema(&input, &label_keys)?;
        Ok(Self {
            input,
            columns,
            label_keys,
            output_schema,
        })
    }
}

fn compute_output_schema(input: &LogicalPlan, label_keys: &[String]) -> Result<DFSchemaRef> {
    // Verify the input has a timestamp column.
    let _ts = input
        .schema()
        .field_with_unqualified_name("timestamp")
        .map_err(|_| PromqlError::Plan("WideUnpack input must have a 'timestamp' column".into()))?;

    let mut fields = vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, true),
        Field::new("__name__", DataType::Utf8, false),
    ];
    for key in label_keys {
        fields.push(Field::new(key, DataType::Utf8, true));
    }
    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl UserDefinedLogicalNodeCore for WideUnpack {
    fn name(&self) -> &str {
        "WideUnpack"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.output_schema
    }

    fn expressions(&self) -> Vec<datafusion::logical_expr::Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WideUnpack: {} columns, labels=[{}]",
            self.columns.len(),
            self.label_keys.join(", ")
        )
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            columns: self.columns.clone(),
            label_keys: self.label_keys.clone(),
            output_schema: self.output_schema.clone(),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        // Don't push filters on value/__name__/label columns past this node,
        // since those columns don't exist in the wide-format input.
        let mut cols = HashSet::new();
        cols.insert("value".to_string());
        cols.insert("__name__".to_string());
        for key in self.label_keys.iter() {
            cols.insert(key.clone());
        }
        cols
    }
}

impl PartialEq for WideUnpack {
    fn eq(&self, other: &Self) -> bool {
        self.columns == other.columns && self.label_keys == other.label_keys
    }
}

impl Eq for WideUnpack {}

impl Hash for WideUnpack {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.columns.hash(state);
        self.label_keys.hash(state);
    }
}

impl PartialOrd for WideUnpack {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WideUnpack {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}
