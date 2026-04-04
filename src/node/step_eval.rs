use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

/// Custom logical node that aligns raw samples to a range of step timestamps.
///
/// For each step timestamp `t` in `[start_ns, end_ns]` with `step_ns`, this
/// node picks the most recent sample within the lookback window
/// `[t - offset - lookback, t - offset]` for each series. The result is
/// reported at timestamp `t` (the original eval timestamp).
///
/// This is used for range queries. For instant (single-timestamp) queries, see
/// [`super::InstantVectorEval`].
#[derive(Debug, Clone)]
pub(crate) struct StepVectorEval {
    /// The child plan that produces raw samples in long format.
    pub input: LogicalPlan,
    pub start_ns: u64,
    pub end_ns: u64,
    pub step_ns: u64,
    /// Lookback window in nanoseconds.
    pub lookback_ns: u64,
    /// Offset in nanoseconds. Positive shifts the lookup window into the past.
    pub offset_ns: i64,
    /// Label column names used for grouping series (excludes timestamp/value).
    pub label_columns: Vec<String>,
}

impl StepVectorEval {
    pub fn new(
        input: LogicalPlan,
        start_ns: u64,
        end_ns: u64,
        step_ns: u64,
        lookback_ns: u64,
        offset_ns: i64,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            start_ns,
            end_ns,
            step_ns,
            lookback_ns,
            offset_ns,
            label_columns,
        }
    }
}

impl UserDefinedLogicalNodeCore for StepVectorEval {
    fn name(&self) -> &str {
        "StepVectorEval"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<datafusion::logical_expr::Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StepVectorEval: range=[{}, {}], step={}ns, lookback={}ns, offset={}ns, group_by=[{}]",
            self.start_ns,
            self.end_ns,
            self.step_ns,
            self.lookback_ns,
            self.offset_ns,
            self.label_columns.join(", ")
        )
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            start_ns: self.start_ns,
            end_ns: self.end_ns,
            step_ns: self.step_ns,
            lookback_ns: self.lookback_ns,
            offset_ns: self.offset_ns,
            label_columns: self.label_columns.clone(),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        // Don't push timestamp filters past this node; we handle time alignment.
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for StepVectorEval {
    fn eq(&self, other: &Self) -> bool {
        self.start_ns == other.start_ns
            && self.end_ns == other.end_ns
            && self.step_ns == other.step_ns
            && self.lookback_ns == other.lookback_ns
            && self.offset_ns == other.offset_ns
            && self.label_columns == other.label_columns
    }
}

impl Eq for StepVectorEval {}

impl Hash for StepVectorEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start_ns.hash(state);
        self.end_ns.hash(state);
        self.step_ns.hash(state);
        self.lookback_ns.hash(state);
        self.offset_ns.hash(state);
        self.label_columns.hash(state);
    }
}

impl PartialOrd for StepVectorEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StepVectorEval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.start_ns
            .cmp(&other.start_ns)
            .then(self.end_ns.cmp(&other.end_ns))
            .then(self.step_ns.cmp(&other.step_ns))
    }
}
