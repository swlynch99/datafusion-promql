use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::func::InstantFunction;

/// Custom logical node that applies an element-wise instant function to each sample value.
///
/// The output schema is identical to the input schema; only the `value` column is transformed.
#[derive(Debug, Clone)]
pub(crate) struct InstantFnEval {
    /// The child plan producing the instant vector.
    pub input: LogicalPlan,
    /// The function to apply to each sample value.
    pub func: InstantFunction,
}

impl InstantFnEval {
    pub fn new(input: LogicalPlan, func: InstantFunction) -> Self {
        Self { input, func }
    }
}

impl UserDefinedLogicalNodeCore for InstantFnEval {
    fn name(&self) -> &str {
        "InstantFnEval"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        // Output schema is the same as the input schema.
        self.input.schema()
    }

    fn expressions(&self) -> Vec<datafusion::logical_expr::Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InstantFnEval: func={}", self.func)
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

impl PartialEq for InstantFnEval {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}
impl Eq for InstantFnEval {}

impl Hash for InstantFnEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state);
    }
}

impl PartialOrd for InstantFnEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InstantFnEval {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}
