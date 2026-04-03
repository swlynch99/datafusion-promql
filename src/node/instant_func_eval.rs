use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::func::InstantFunction;

/// Custom logical node for applying an instant vector function to an instant vector.
///
/// Transforms the `value` column according to the function, passing all other columns
/// through unchanged.
#[derive(Debug, Clone)]
pub(crate) struct InstantFuncEval {
    pub input: LogicalPlan,
    pub func: InstantFunction,
}

impl InstantFuncEval {
    pub fn new(input: LogicalPlan, func: InstantFunction) -> Self {
        Self { input, func }
    }
}

impl UserDefinedLogicalNodeCore for InstantFuncEval {
    fn name(&self) -> &str {
        "InstantFuncEval"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        // Same schema as input: we only transform the value column.
        self.input.schema()
    }

    fn expressions(&self) -> Vec<datafusion::logical_expr::Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InstantFuncEval: {}", self.func)
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            func: self.func,
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
            (
                InstantFunction::Round { to_nearest: a },
                InstantFunction::Round { to_nearest: b },
            ) => a.to_bits() == b.to_bits(),
        }
    }
}

impl Eq for InstantFuncEval {}

impl Hash for InstantFuncEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.func {
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
