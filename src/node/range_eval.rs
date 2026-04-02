use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};

use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::func::RangeFunction;

/// Custom logical node that applies a range vector function over a sliding
/// window of samples at each evaluation timestamp.
///
/// For each step timestamp `t` and each series, this node collects all samples
/// in `[t - range_ms, t]` and applies the range function (e.g. rate, delta).
#[derive(Debug, Clone)]
pub(crate) struct RangeVectorEval {
    /// The child plan that produces raw samples in long format.
    pub input: LogicalPlan,
    /// The range window duration in milliseconds (e.g. 5m = 300_000).
    pub range_ms: i64,
    /// The range function to apply.
    pub func: RangeFunction,
    /// For an instant query, the single evaluation timestamp (ms).
    pub eval_ts_ms: Option<i64>,
    pub start_ms: i64,
    pub end_ms: i64,
    pub step_ms: i64,
    /// Label column names used for grouping series.
    pub label_columns: Vec<String>,
}

impl RangeVectorEval {
    /// Create a node for an instant query at a single timestamp.
    pub fn instant(
        input: LogicalPlan,
        timestamp_ms: i64,
        range_ms: i64,
        func: RangeFunction,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            range_ms,
            func,
            eval_ts_ms: Some(timestamp_ms),
            start_ms: timestamp_ms,
            end_ms: timestamp_ms,
            step_ms: 1,
            label_columns,
        }
    }

    /// Create a node for a range query over `[start, end]` with step.
    pub fn range(
        input: LogicalPlan,
        start_ms: i64,
        end_ms: i64,
        step_ms: i64,
        range_ms: i64,
        func: RangeFunction,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            range_ms,
            func,
            eval_ts_ms: None,
            start_ms,
            end_ms,
            step_ms,
            label_columns,
        }
    }
}

impl UserDefinedLogicalNodeCore for RangeVectorEval {
    fn name(&self) -> &str {
        "RangeVectorEval"
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
                "RangeVectorEval: func={}, ts={ts}, range={}ms",
                self.func, self.range_ms
            )
        } else {
            write!(
                f,
                "RangeVectorEval: func={}, range=[{}, {}], step={}ms, window={}ms",
                self.func, self.start_ms, self.end_ms, self.step_ms, self.range_ms
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
            range_ms: self.range_ms,
            func: self.func,
            eval_ts_ms: self.eval_ts_ms,
            start_ms: self.start_ms,
            end_ms: self.end_ms,
            step_ms: self.step_ms,
            label_columns: self.label_columns.clone(),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for RangeVectorEval {
    fn eq(&self, other: &Self) -> bool {
        self.range_ms == other.range_ms
            && self.func == other.func
            && self.eval_ts_ms == other.eval_ts_ms
            && self.start_ms == other.start_ms
            && self.end_ms == other.end_ms
            && self.step_ms == other.step_ms
            && self.label_columns == other.label_columns
    }
}

impl Eq for RangeVectorEval {}

impl Hash for RangeVectorEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.range_ms.hash(state);
        self.func.hash(state);
        self.eval_ts_ms.hash(state);
        self.start_ms.hash(state);
        self.end_ms.hash(state);
        self.step_ms.hash(state);
        self.label_columns.hash(state);
    }
}

impl PartialOrd for RangeVectorEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RangeVectorEval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.start_ms
            .cmp(&other.start_ms)
            .then(self.end_ms.cmp(&other.end_ms))
            .then(self.step_ms.cmp(&other.step_ms))
    }
}
