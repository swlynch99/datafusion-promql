use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

/// Custom logical node that aligns raw samples to a single evaluation timestamp.
///
/// For the evaluation timestamp `t`, this node picks the most recent sample
/// within the lookback window `[t - offset - lookback, t - offset]` for each series.
/// The result is reported at timestamp `t` (the original eval timestamp).
///
/// This is used for instant queries. For range queries that evaluate over
/// multiple step timestamps, see [`super::StepVectorEval`].
#[derive(Debug, Clone)]
pub(crate) struct InstantVectorEval {
    /// The child plan that produces raw samples in long format.
    pub input: LogicalPlan,
    /// The single evaluation timestamp (ns).
    pub timestamp_ns: u64,
    /// Lookback window in nanoseconds.
    pub lookback_ns: u64,
    /// Offset in nanoseconds. Positive shifts the lookup window into the past.
    pub offset_ns: i64,
    /// Label column names used for grouping series (excludes timestamp/value).
    pub label_columns: Vec<String>,
}

impl InstantVectorEval {
    pub fn new(
        input: LogicalPlan,
        timestamp_ns: u64,
        lookback_ns: u64,
        offset_ns: i64,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            timestamp_ns,
            lookback_ns,
            offset_ns,
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
        write!(
            f,
            "InstantVectorEval: ts={}, lookback={}ns, offset={}ns, group_by=[{}]",
            self.timestamp_ns,
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
            timestamp_ns: self.timestamp_ns,
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

impl PartialEq for InstantVectorEval {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp_ns == other.timestamp_ns
            && self.lookback_ns == other.lookback_ns
            && self.offset_ns == other.offset_ns
            && self.label_columns == other.label_columns
    }
}

impl Eq for InstantVectorEval {}

impl Hash for InstantVectorEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.timestamp_ns.hash(state);
        self.lookback_ns.hash(state);
        self.offset_ns.hash(state);
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
        self.timestamp_ns.cmp(&other.timestamp_ns)
    }
}
