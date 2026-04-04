use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::error::{PromqlError, Result};

/// Custom logical node that groups samples by metric series and collects
/// them into per-window arrays at each evaluation timestamp.
///
/// For each step timestamp `t` and each series, this node collects all samples
/// in `[t - offset - range_ns, t - offset]` and outputs them as `List<Int64>`
/// (timestamps) and `List<Float64>` (values) arrays. The result is reported at
/// timestamp `t`.
#[derive(Debug, Clone)]
pub(crate) struct RangeVectorEval {
    /// The child plan that produces raw samples in long format.
    pub input: LogicalPlan,
    /// The range window duration in nanoseconds (e.g. 5m = 300_000_000_000).
    pub range_ns: i64,
    /// For an instant query, the single evaluation timestamp (ns).
    pub eval_ts_ns: Option<i64>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub step_ns: i64,
    /// Offset in nanoseconds. Positive shifts the lookup window into the past.
    pub offset_ns: i64,
    /// Label column names used for grouping series.
    pub label_columns: Vec<String>,
    /// Output schema: timestamp, timestamps (list), values (list), labels.
    pub output_schema: DFSchemaRef,
}

/// Compute the output schema for a range vector windowing node.
///
/// Output columns:
/// - `timestamp: Int64` — the evaluation timestamp
/// - `timestamps: List<Int64>` — sample timestamps within the window
/// - `values: List<Float64>` — sample values within the window
/// - label columns from the input (Utf8)
pub(crate) fn compute_range_vector_schema(
    input: &LogicalPlan,
    label_columns: &[String],
) -> Result<DFSchemaRef> {
    let mut fields = vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new(
            "timestamps",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ),
        Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
    ];

    // Carry over label columns from the input schema.
    for label in label_columns {
        let input_field = input
            .schema()
            .field_with_unqualified_name(label)
            .map_err(|e| PromqlError::Plan(format!("missing label column {label}: {e}")))?;
        fields.push(input_field.as_ref().clone());
    }

    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl RangeVectorEval {
    /// Create a node for an instant query at a single timestamp.
    pub fn instant(
        input: LogicalPlan,
        timestamp_ns: i64,
        range_ns: i64,
        offset_ns: i64,
        label_columns: Vec<String>,
    ) -> Result<Self> {
        let output_schema = compute_range_vector_schema(&input, &label_columns)?;
        Ok(Self {
            input,
            range_ns,
            eval_ts_ns: Some(timestamp_ns),
            start_ns: timestamp_ns,
            end_ns: timestamp_ns,
            step_ns: 1,
            offset_ns,
            label_columns,
            output_schema,
        })
    }

    /// Create a node for a range query over `[start, end]` with step.
    pub fn range(
        input: LogicalPlan,
        start_ns: i64,
        end_ns: i64,
        step_ns: i64,
        range_ns: i64,
        offset_ns: i64,
        label_columns: Vec<String>,
    ) -> Result<Self> {
        let output_schema = compute_range_vector_schema(&input, &label_columns)?;
        Ok(Self {
            input,
            range_ns,
            eval_ts_ns: None,
            start_ns,
            end_ns,
            step_ns,
            offset_ns,
            label_columns,
            output_schema,
        })
    }
}

impl UserDefinedLogicalNodeCore for RangeVectorEval {
    fn name(&self) -> &str {
        "RangeVectorEval"
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
        if let Some(ts) = self.eval_ts_ns {
            write!(
                f,
                "RangeVectorEval: ts={ts}, range={}ns, offset={}ns, group_by=[{}]",
                self.range_ns,
                self.offset_ns,
                self.label_columns.join(", ")
            )
        } else {
            write!(
                f,
                "RangeVectorEval: range=[{}, {}], step={}ns, window={}ns, offset={}ns, group_by=[{}]",
                self.start_ns,
                self.end_ns,
                self.step_ns,
                self.range_ns,
                self.offset_ns,
                self.label_columns.join(", ")
            )
        }
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            range_ns: self.range_ns,
            eval_ts_ns: self.eval_ts_ns,
            start_ns: self.start_ns,
            end_ns: self.end_ns,
            step_ns: self.step_ns,
            offset_ns: self.offset_ns,
            label_columns: self.label_columns.clone(),
            output_schema: Arc::clone(&self.output_schema),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for RangeVectorEval {
    fn eq(&self, other: &Self) -> bool {
        self.range_ns == other.range_ns
            && self.eval_ts_ns == other.eval_ts_ns
            && self.start_ns == other.start_ns
            && self.end_ns == other.end_ns
            && self.step_ns == other.step_ns
            && self.offset_ns == other.offset_ns
            && self.label_columns == other.label_columns
    }
}

impl Eq for RangeVectorEval {}

impl Hash for RangeVectorEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.range_ns.hash(state);
        self.eval_ts_ns.hash(state);
        self.start_ns.hash(state);
        self.end_ns.hash(state);
        self.step_ns.hash(state);
        self.offset_ns.hash(state);
        self.label_columns.hash(state);
    }
}

impl PartialOrd for RangeVectorEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RangeVectorEval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.start_ns
            .cmp(&other.start_ns)
            .then(self.end_ns.cmp(&other.end_ns))
            .then(self.step_ns.cmp(&other.step_ns))
    }
}
