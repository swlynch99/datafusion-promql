use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::datatypes::{Field, Schema};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::error::{PromqlError, Result};
use crate::func::InstantFunction;

/// Custom logical node for instant vector functions (e.g., `ln`, `log2`, `round`).
///
/// Wraps a child instant vector plan and applies `func` to each sample value.
/// The `__name__` label is dropped from the output, matching Prometheus semantics.
#[derive(Debug, Clone)]
pub(crate) struct InstantFuncEval {
    pub input: LogicalPlan,
    pub func: InstantFunction,
    pub output_schema: DFSchemaRef,
}

impl InstantFuncEval {
    pub fn new(input: LogicalPlan, func: InstantFunction) -> Result<Self> {
        let output_schema = compute_output_schema(&input, func)?;
        Ok(Self {
            input,
            func,
            output_schema,
        })
    }
}

fn compute_output_schema(input: &LogicalPlan, func: InstantFunction) -> Result<DFSchemaRef> {
    let input_schema = input.schema();
    let fields: Vec<Field> = input_schema
        .fields()
        .iter()
        .filter(|f| !func.drops_metric_name() || f.name() != "__name__")
        .map(|f| f.as_ref().clone())
        .collect();

    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl UserDefinedLogicalNodeCore for InstantFuncEval {
    fn name(&self) -> &str {
        "InstantFuncEval"
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
        write!(f, "InstantFuncEval: func={}", self.func)
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

impl PartialEq for InstantFuncEval {
    fn eq(&self, other: &Self) -> bool {
        match (self.func, other.func) {
            (InstantFunction::Ceil, InstantFunction::Ceil) => true,
            (InstantFunction::Ln, InstantFunction::Ln) => true,
            (InstantFunction::Log2, InstantFunction::Log2) => true,
            (
                InstantFunction::Round { to_nearest: a },
                InstantFunction::Round { to_nearest: b },
            ) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}
impl Eq for InstantFuncEval {}

impl Hash for InstantFuncEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.func {
            InstantFunction::Ceil => "ceil".hash(state),
            InstantFunction::Ln => "ln".hash(state),
            InstantFunction::Log2 => "log2".hash(state),
            InstantFunction::Round { to_nearest } => {
                "round".hash(state);
                to_nearest.to_bits().hash(state);
            }
        }
    }
}

impl PartialOrd for InstantFuncEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InstantFuncEval {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}
