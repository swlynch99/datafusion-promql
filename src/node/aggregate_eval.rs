use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::error::{PromqlError, Result};
use crate::func::AggregateFunction;

/// Custom logical node that applies a PromQL aggregation operator.
///
/// Groups series by `grouping_labels` and applies the aggregation function
/// to each group at each timestamp.
#[derive(Debug, Clone)]
pub(crate) struct AggregateEval {
    pub input: LogicalPlan,
    pub func: AggregateFunction,
    /// The label columns to GROUP BY in the output.
    pub grouping_labels: Vec<String>,
    /// Precomputed output schema: timestamp + value + grouping label columns.
    pub output_schema: DFSchemaRef,
}

impl AggregateEval {
    pub fn new(
        input: LogicalPlan,
        func: AggregateFunction,
        grouping_labels: Vec<String>,
    ) -> Result<Self> {
        let output_schema = compute_aggregate_output_schema(&grouping_labels)?;
        Ok(Self {
            input,
            func,
            grouping_labels,
            output_schema,
        })
    }
}

fn compute_aggregate_output_schema(grouping_labels: &[String]) -> Result<DFSchemaRef> {
    let mut fields = vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ];
    for label in grouping_labels {
        fields.push(Field::new(label, DataType::Utf8, false));
    }

    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl UserDefinedLogicalNodeCore for AggregateEval {
    fn name(&self) -> &str {
        "AggregateEval"
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
            "AggregateEval: func={}, by=[{}]",
            self.func,
            self.grouping_labels.join(", ")
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
            grouping_labels: self.grouping_labels.clone(),
            output_schema: Arc::clone(&self.output_schema),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for AggregateEval {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func && self.grouping_labels == other.grouping_labels
    }
}
impl Eq for AggregateEval {}

impl Hash for AggregateEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state);
        self.grouping_labels.hash(state);
    }
}

impl PartialOrd for AggregateEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AggregateEval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.grouping_labels.cmp(&other.grouping_labels)
    }
}
