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

/// Wide-format counterpart to [`StreamingRangeFunctionEval`].
///
/// Input: wide-format samples `(timestamp: UInt64, col_0: Float64, …,
/// col_N: Float64)` sorted by `timestamp ASC`. Each value column represents
/// a distinct series.
///
/// Output: wide-format rows with the same schema as the input, but one row
/// per evaluation timestamp: `(timestamp: UInt64, col_0: Float64, …,
/// col_N: Float64)`, where each `col_i` holds the range-function result for
/// that column's series at the corresponding eval timestamp. When a series
/// has no result at a given eval timestamp (e.g. too few samples), its cell
/// is null.
///
/// The transformation over this node, combined with a [`WideUnpack`] above
/// it, is equivalent to running [`StreamingRangeFunctionEval`] on the
/// long-format unpacked output — but avoids materialising N × #rows
/// long-format rows before the range reduction.
///
/// [`StreamingRangeFunctionEval`]: super::StreamingRangeFunctionEval
/// [`WideUnpack`]: super::WideUnpack
#[derive(Debug, Clone)]
pub struct WideStreamingRangeFunctionEval {
    /// Wide-format input plan.
    pub input: LogicalPlan,
    /// The range function to apply at each evaluation timestamp.
    pub func: RangeFunction,
    /// Optional scalar argument (`predict_linear` duration, etc.).
    pub scalar_arg: Option<f64>,
    /// Sliding window width in nanoseconds.
    pub range_ns: u64,
    /// Single evaluation timestamp for instant queries (ns); `None` for
    /// range queries which use `start_ns / end_ns / step_ns`.
    pub eval_ts_ns: Option<u64>,
    pub start_ns: u64,
    pub end_ns: u64,
    pub step_ns: u64,
    /// Offset in nanoseconds (positive = shift window into the past).
    pub offset_ns: i64,
    /// Names of the value columns to process. The node iterates these
    /// in-order and maintains one sliding window per column.
    pub value_columns: Arc<Vec<String>>,
    /// Fixed `@`-modifier timestamp (ns).
    pub at_timestamp_ns: Option<u64>,
    /// Output schema: `(timestamp, value_columns…)`.
    pub output_schema: DFSchemaRef,
}

impl WideStreamingRangeFunctionEval {
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
        value_columns: Arc<Vec<String>>,
        at_timestamp_ns: Option<u64>,
    ) -> Result<Self> {
        let output_schema = compute_output_schema(&input, &value_columns)?;
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
            value_columns,
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

fn compute_output_schema(input: &LogicalPlan, value_columns: &[String]) -> Result<DFSchemaRef> {
    let input_schema = input.schema();
    // Validate the input schema has a timestamp column.
    let _ = input_schema
        .field_with_unqualified_name("timestamp")
        .map_err(|_| {
            PromqlError::Plan(
                "WideStreamingRangeFunctionEval input must have a 'timestamp' column".into(),
            )
        })?;

    let mut fields = vec![Field::new("timestamp", DataType::UInt64, false)];
    for col in value_columns {
        // All value columns are Float64 and nullable after the range function
        // (null = no value at this eval timestamp for that series).
        fields.push(Field::new(col.as_str(), DataType::Float64, true));
    }
    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl UserDefinedLogicalNodeCore for WideStreamingRangeFunctionEval {
    fn name(&self) -> &str {
        "WideStreamingRangeFunctionEval"
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
            "WideStreamingRangeFunctionEval: func={}, range={}ns, columns={}",
            self.func,
            self.range_ns,
            self.value_columns.len()
        )
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            func: self.func,
            scalar_arg: self.scalar_arg,
            range_ns: self.range_ns,
            eval_ts_ns: self.eval_ts_ns,
            start_ns: self.start_ns,
            end_ns: self.end_ns,
            step_ns: self.step_ns,
            offset_ns: self.offset_ns,
            value_columns: Arc::clone(&self.value_columns),
            at_timestamp_ns: self.at_timestamp_ns,
            output_schema: Arc::clone(&self.output_schema),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        // The output timestamps are the eval timestamps, not the input sample
        // timestamps, so pushing a timestamp filter through this node would
        // change the results.
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for WideStreamingRangeFunctionEval {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
            && self.range_ns == other.range_ns
            && self.value_columns == other.value_columns
    }
}

impl Eq for WideStreamingRangeFunctionEval {}

impl Hash for WideStreamingRangeFunctionEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state);
        self.range_ns.hash(state);
        self.value_columns.hash(state);
    }
}

impl PartialOrd for WideStreamingRangeFunctionEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WideStreamingRangeFunctionEval {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}
