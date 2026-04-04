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
/// in `[t - range_ns, t]` and applies the range function (e.g. rate, delta).
#[derive(Debug, Clone)]
pub(crate) struct RangeVectorEval {
    /// The child plan that produces raw samples in long format.
    pub input: LogicalPlan,
    /// The range window duration in nanoseconds (e.g. 5m = 300_000_000_000).
    pub range_ns: i64,
    /// The range function to apply.
    pub func: RangeFunction,
    /// For an instant query, the single evaluation timestamp (ns).
    pub eval_ts_ns: Option<i64>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub step_ns: i64,
    /// Label column names used for grouping series.
    pub label_columns: Vec<String>,
}

impl RangeVectorEval {
    /// Create a node for an instant query at a single timestamp.
    pub fn instant(
        input: LogicalPlan,
        timestamp_ns: i64,
        range_ns: i64,
        func: RangeFunction,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            range_ns,
            func,
            eval_ts_ns: Some(timestamp_ns),
            start_ns: timestamp_ns,
            end_ns: timestamp_ns,
            step_ns: 1,
            label_columns,
        }
    }

    /// Create a node for a range query over `[start, end]` with step.
    pub fn range(
        input: LogicalPlan,
        start_ns: i64,
        end_ns: i64,
        step_ns: i64,
        range_ns: i64,
        func: RangeFunction,
        label_columns: Vec<String>,
    ) -> Self {
        Self {
            input,
            range_ns,
            func,
            eval_ts_ns: None,
            start_ns,
            end_ns,
            step_ns,
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
        if let Some(ts) = self.eval_ts_ns {
            write!(
                f,
                "RangeVectorEval: func={}, ts={ts}, range={}ns",
                self.func, self.range_ns
            )
        } else {
            write!(
                f,
                "RangeVectorEval: func={}, range=[{}, {}], step={}ns, window={}ns",
                self.func, self.start_ns, self.end_ns, self.step_ns, self.range_ns
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
            range_ns: self.range_ns,
            func: self.func,
            eval_ts_ns: self.eval_ts_ns,
            start_ns: self.start_ns,
            end_ns: self.end_ns,
            step_ns: self.step_ns,
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
        self.range_ns == other.range_ns
            && self.func == other.func
            && self.eval_ts_ns == other.eval_ts_ns
            && self.start_ns == other.start_ns
            && self.end_ns == other.end_ns
            && self.step_ns == other.step_ns
            && self.label_columns == other.label_columns
    }
}

impl Eq for RangeVectorEval {}

impl Hash for RangeVectorEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.range_ns.hash(state);
        self.func.hash(state);
        self.eval_ts_ns.hash(state);
        self.start_ns.hash(state);
        self.end_ns.hash(state);
        self.step_ns.hash(state);
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
        self.start_ns
            .cmp(&other.start_ns)
            .then(self.end_ns.cmp(&other.end_ns))
            .then(self.step_ns.cmp(&other.step_ns))
    }
}
