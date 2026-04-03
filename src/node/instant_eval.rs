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
    /// For an instant query, the single evaluation timestamp (ms).
    /// For a range query, the step timestamps are generated from
    /// `start_ms..=end_ms` with `step_ms`.
    pub eval_ts_ms: Option<i64>,
    pub start_ms: i64,
    pub end_ms: i64,
    pub step_ms: i64,
    /// Lookback window in milliseconds.
    pub lookback_ms: i64,
    /// Label column names used for grouping series (excludes timestamp/value).
    pub label_columns: Vec<String>,
}

impl InstantVectorEval {
    /// Create a node for an instant query at a single timestamp.
    pub fn instant(
        input: LogicalPlan,
        timestamp_ms: i64,
        lookback_ms: i64,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            eval_ts_ms: Some(timestamp_ms),
            start_ms: timestamp_ms,
            end_ms: timestamp_ms,
            step_ms: 1, // single step
            lookback_ms,
            label_columns,
        }
    }

    /// Create a node for a range query over `[start, end]` with step.
    pub fn range(
        input: LogicalPlan,
        start_ms: i64,
        end_ms: i64,
        step_ms: i64,
        lookback_ms: i64,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            eval_ts_ms: None,
            start_ms,
            end_ms,
            step_ms,
            lookback_ms,
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
        if let Some(ts) = self.eval_ts_ms {
            write!(
                f,
                "InstantVectorEval: ts={ts}, lookback={}ms",
                self.lookback_ms
            )
        } else {
            write!(
                f,
                "InstantVectorEval: range=[{}, {}], step={}ms, lookback={}ms",
                self.start_ms, self.end_ms, self.step_ms, self.lookback_ms
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
            eval_ts_ms: self.eval_ts_ms,
            start_ms: self.start_ms,
            end_ms: self.end_ms,
            step_ms: self.step_ms,
            lookback_ms: self.lookback_ms,
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
        self.eval_ts_ms == other.eval_ts_ms
            && self.start_ms == other.start_ms
            && self.end_ms == other.end_ms
            && self.step_ms == other.step_ms
            && self.lookback_ms == other.lookback_ms
            && self.label_columns == other.label_columns
    }
}

impl Eq for InstantVectorEval {}

impl Hash for InstantVectorEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.eval_ts_ms.hash(state);
        self.start_ms.hash(state);
        self.end_ms.hash(state);
        self.step_ms.hash(state);
        self.lookback_ms.hash(state);
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
        self.start_ms
            .cmp(&other.start_ms)
            .then(self.end_ms.cmp(&other.end_ms))
            .then(self.step_ms.cmp(&other.step_ms))
    }
}
