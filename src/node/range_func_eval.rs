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

/// Custom logical node that applies a range function to pre-windowed data.
///
/// Consumes the output of a `RangeVectorEval` node (which provides
/// `List<Int64>` timestamps and `List<Float64>` values per window) and
/// applies a `RangeFunction` (e.g. rate, delta, idelta) to collapse each
/// window into a single scalar value.
#[derive(Debug, Clone)]
pub(crate) struct RangeFunctionEval {
    /// The child plan (typically a `RangeVectorEval`).
    pub input: LogicalPlan,
    /// The range function to apply to each window.
    pub func: RangeFunction,
    /// Output schema: timestamp, value, label columns.
    pub output_schema: DFSchemaRef,
}

/// Compute the output schema for a range function node.
///
/// Input schema is expected to contain `timestamp`, `timestamps` (list),
/// `values` (list), and label columns. Output replaces the list columns
/// with a scalar `value: Float64`.
fn compute_output_schema(input: &LogicalPlan) -> Result<DFSchemaRef> {
    let input_schema = input.schema();
    let mut fields = Vec::new();

    for field in input_schema.fields() {
        let name = field.name().as_str();
        match name {
            "timestamp" => fields.push(Field::new("timestamp", DataType::Int64, false)),
            "timestamps" | "values" => {
                // Skip the list columns; we'll add a scalar "value" column.
            }
            _ => fields.push(field.as_ref().clone()),
        }
    }

    // Insert the scalar value column after timestamp.
    fields.insert(1, Field::new("value", DataType::Float64, true));

    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl RangeFunctionEval {
    pub fn new(input: LogicalPlan, func: RangeFunction) -> Result<Self> {
        let output_schema = compute_output_schema(&input)?;
        Ok(Self {
            input,
            func,
            output_schema,
        })
    }
}

impl UserDefinedLogicalNodeCore for RangeFunctionEval {
    fn name(&self) -> &str {
        "RangeFunctionEval"
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
        write!(f, "RangeFunctionEval: func={}", self.func)
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            func: self.func,
            output_schema: Arc::clone(&self.output_schema),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for RangeFunctionEval {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}

impl Eq for RangeFunctionEval {}

impl Hash for RangeFunctionEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state);
    }
}

impl PartialOrd for RangeFunctionEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RangeFunctionEval {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}
