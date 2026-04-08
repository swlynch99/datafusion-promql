use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::error::{PromqlError, Result};
use crate::func::RangeFunction;

/// Combined logical node that applies a range function to raw sorted samples
/// using a streaming sliding-window, bypassing the intermediate
/// `List<UInt64>` / `List<Float64>` arrays that `RangeVectorEval` produces.
///
/// Input: raw samples in long format (`timestamp: UInt64`, `value: Float64`,
/// label columns), sorted by `(label_cols ASC, timestamp ASC)`.
///
/// Output: `(timestamp: UInt64, value: Float64, label_cols...)` — one row per
/// (eval_timestamp, series) pair where the range function returns a value.
///
/// This node is produced by the `RangeVectorToAggregation` optimizer rule for
/// functions that work on only the last few samples in a window (currently
/// `irate` and `idelta`), where the overhead of materialising full window
/// arrays is not worthwhile.
#[derive(Debug, Clone)]
pub(crate) struct StreamingRangeFunctionEval {
    /// Raw-sample input plan (same source as `RangeVectorEval`'s input).
    pub input: LogicalPlan,
    /// The range function to apply at each evaluation timestamp.
    pub func: RangeFunction,
    /// Optional scalar argument (`predict_linear` duration, etc.).
    pub scalar_arg: Option<f64>,
    /// Sliding window width in nanoseconds.
    pub range_ns: u64,
    /// Single evaluation timestamp for instant queries (ns); `None` for range
    /// queries which use `start_ns / end_ns / step_ns`.
    pub eval_ts_ns: Option<u64>,
    pub start_ns: u64,
    pub end_ns: u64,
    pub step_ns: u64,
    /// Offset in nanoseconds (positive = shift window into the past).
    pub offset_ns: i64,
    /// Label columns used to group series.
    pub label_columns: Vec<String>,
    /// Fixed `@`-modifier timestamp (ns). When set, every eval timestamp uses
    /// this as the window anchor instead of the eval timestamp itself.
    pub at_timestamp_ns: Option<u64>,
    /// Output schema: `timestamp`, `value`, label columns.
    pub output_schema: DFSchemaRef,
}

impl StreamingRangeFunctionEval {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: LogicalPlan,
        func: RangeFunction,
        scalar_arg: Option<f64>,
        range_ns: u64,
        eval_ts_ns: Option<u64>,
        start_ns: u64,
        end_ns: u64,
        step_ns: u64,
        offset_ns: i64,
        label_columns: Vec<String>,
        at_timestamp_ns: Option<u64>,
    ) -> Result<Self> {
        let output_schema = compute_output_schema(&input, &label_columns)?;
        Ok(Self {
            input,
            func,
            scalar_arg,
            range_ns,
            eval_ts_ns,
            start_ns,
            end_ns,
            step_ns,
            offset_ns,
            label_columns,
            at_timestamp_ns,
            output_schema,
        })
    }

    /// Generate the sorted list of evaluation timestamps.
    pub fn eval_timestamps(&self) -> Vec<u64> {
        if let Some(ts) = self.eval_ts_ns {
            return vec![ts];
        }
        let mut timestamps = Vec::new();
        let mut t = self.start_ns;
        while t <= self.end_ns {
            timestamps.push(t);
            t += self.step_ns;
        }
        timestamps
    }
}

fn compute_output_schema(input: &LogicalPlan, label_columns: &[String]) -> Result<DFSchemaRef> {
    let input_schema = input.schema();
    let mut fields = vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, true),
    ];
    for label in label_columns {
        // Inherit nullability from the input schema if the column exists there.
        let nullable = input_schema
            .field_with_unqualified_name(label)
            .map(|f| f.is_nullable())
            .unwrap_or(true);
        fields.push(Field::new(label, DataType::Utf8, nullable));
    }
    let schema = Schema::new(fields);
    let df_schema = DFSchema::try_from(schema)
        .map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl UserDefinedLogicalNodeCore for StreamingRangeFunctionEval {
    fn name(&self) -> &str {
        "StreamingRangeFunctionEval"
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
            "StreamingRangeFunctionEval: func={}, range={}ns",
            self.func, self.range_ns
        )
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            ..self.clone()
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for StreamingRangeFunctionEval {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func && self.range_ns == other.range_ns
    }
}

impl Eq for StreamingRangeFunctionEval {}

impl Hash for StreamingRangeFunctionEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state);
        self.range_ns.hash(state);
    }
}

impl PartialOrd for StreamingRangeFunctionEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StreamingRangeFunctionEval {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}
