use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

/// Custom logical node that aligns raw samples to evaluation timestamps.
///
/// For each step timestamp `t`, this node picks the most recent sample
/// within the lookback window `[t - lookback, t]` for each series.
#[derive(Debug, Clone)]
pub(crate) struct InstantVectorEval {
    /// The child plan that produces raw samples in long format.
    pub input: LogicalPlan,
    /// For an instant query, the single evaluation timestamp (ns).
    /// For a range query, the step timestamps are generated from
    /// `start_ns..=end_ns` with `step_ns`.
    pub eval_ts_ns: Option<i64>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub step_ns: i64,
    /// Lookback window in nanoseconds.
    pub lookback_ns: i64,
    /// Label column names used for grouping series (excludes timestamp/value).
    pub label_columns: Vec<String>,
}

impl InstantVectorEval {
    /// Create a node for an instant query at a single timestamp.
    pub fn instant(
        input: LogicalPlan,
        timestamp_ns: i64,
        lookback_ns: i64,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            eval_ts_ns: Some(timestamp_ns),
            start_ns: timestamp_ns,
            end_ns: timestamp_ns,
            step_ns: 1, // single step
            lookback_ns,
            label_columns,
        }
    }

    /// Create a node for a range query over `[start, end]` with step.
    pub fn range(
        input: LogicalPlan,
        start_ns: i64,
        end_ns: i64,
        step_ns: i64,
        lookback_ns: i64,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            eval_ts_ns: None,
            start_ns,
            end_ns,
            step_ns,
            lookback_ns,
            label_columns,
        }
    }
}

impl UserDefinedLogicalNodeCore for InstantVectorEval {
    fn name(&self) -> &str {
        "InstantVectorEval"
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
        if let Some(ts) = self.eval_ts_ns {
            write!(
                f,
                "InstantVectorEval: ts={ts}, lookback={}ns, group_by=[{}]",
                self.lookback_ns,
                self.label_columns.join(", ")
            )
        } else {
            write!(
                f,
                "InstantVectorEval: range=[{}, {}], step={}ns, lookback={}ns, group_by=[{}]",
                self.start_ns, self.end_ns, self.step_ns, self.lookback_ns,
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
            eval_ts_ns: self.eval_ts_ns,
            start_ns: self.start_ns,
            end_ns: self.end_ns,
            step_ns: self.step_ns,
            lookback_ns: self.lookback_ns,
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

impl PartialEq for InstantVectorEval {
    fn eq(&self, other: &Self) -> bool {
        self.eval_ts_ns == other.eval_ts_ns
            && self.start_ns == other.start_ns
            && self.end_ns == other.end_ns
            && self.step_ns == other.step_ns
            && self.lookback_ns == other.lookback_ns
            && self.label_columns == other.label_columns
    }
}

impl Eq for InstantVectorEval {}

impl Hash for InstantVectorEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.eval_ts_ns.hash(state);
        self.start_ns.hash(state);
        self.end_ns.hash(state);
        self.step_ns.hash(state);
        self.lookback_ns.hash(state);
        self.label_columns.hash(state);
    }
}

impl PartialOrd for InstantVectorEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InstantVectorEval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.start_ns
            .cmp(&other.start_ns)
            .then(self.end_ns.cmp(&other.end_ns))
            .then(self.step_ns.cmp(&other.step_ns))
    }
}
