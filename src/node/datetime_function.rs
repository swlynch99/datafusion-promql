use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::datatypes::{Field, Schema};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::error::{PromqlError, Result};

/// Custom logical node that applies a datetime scalar UDF to the `timestamp`
/// column, producing a new `value`.
///
/// Unlike `InstantFunction` (which transforms `value`), this transforms the
/// `timestamp` column. The `__name__` label is dropped from the output,
/// matching Prometheus semantics.
///
/// This node has no corresponding physical node. It must always be lowered
/// to a projection by the `DateTimeFuncToProjection` optimizer rule.
#[derive(Debug, Clone)]
pub(crate) struct DateTimeFunctionNode {
    pub input: LogicalPlan,
    /// The scalar expression to apply to the `timestamp` column.
    /// This should be a DataFusion expression that takes a column reference
    /// as input (e.g. `promql_hour(col("timestamp"))`).
    pub func_expr: datafusion::logical_expr::Expr,
    /// Display name for the function (used for explain plans and equality).
    pub func_name: String,
    pub output_schema: DFSchemaRef,
}

impl DateTimeFunctionNode {
    pub fn new(
        input: LogicalPlan,
        func_expr: datafusion::logical_expr::Expr,
        func_name: String,
    ) -> Result<Self> {
        let output_schema = compute_output_schema(&input)?;
        Ok(Self {
            input,
            func_expr,
            func_name,
            output_schema,
        })
    }
}

/// Build the output schema: same as input but with `__name__` dropped.
fn compute_output_schema(input: &LogicalPlan) -> Result<DFSchemaRef> {
    let fields: Vec<Field> = input
        .schema()
        .fields()
        .iter()
        .filter(|f| f.name() != "__name__")
        .map(|f| f.as_ref().clone())
        .collect();

    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl UserDefinedLogicalNodeCore for DateTimeFunctionNode {
    fn name(&self) -> &str {
        "DateTimeFunction"
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
        write!(f, "DateTimeFunction: func={}", self.func_name)
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            func_expr: self.func_expr.clone(),
            func_name: self.func_name.clone(),
            output_schema: Arc::clone(&self.output_schema),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for DateTimeFunctionNode {
    fn eq(&self, other: &Self) -> bool {
        self.func_name == other.func_name
    }
}
impl Eq for DateTimeFunctionNode {}

impl Hash for DateTimeFunctionNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func_name.hash(state);
    }
}

impl PartialOrd for DateTimeFunctionNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DateTimeFunctionNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.func_name.cmp(&other.func_name)
    }
}
