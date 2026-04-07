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
pub struct InstantVectorEval {
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
    /// Fixed lookup timestamp from the `@` modifier (ns). When set, the lookup
    /// uses this timestamp instead of `timestamp_ns`, but the output is still
    /// reported at `timestamp_ns`.
    pub at_timestamp_ns: Option<u64>,
}

impl InstantVectorEval {
    pub fn new(
        input: LogicalPlan,
        timestamp_ns: u64,
        lookback_ns: u64,
        offset_ns: i64,
        label_columns: Vec<String>,
        at_timestamp_ns: Option<u64>,
    ) -> Self {
        Self {
            input,
            timestamp_ns,
            lookback_ns,
            offset_ns,
            label_columns,
            at_timestamp_ns,
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
            "InstantVectorEval: ts={}, lookback={}ns, offset={}ns",
            self.timestamp_ns, self.lookback_ns, self.offset_ns,
        )?;
        if let Some(at) = self.at_timestamp_ns {
            write!(f, ", @={at}")?;
        }
        write!(f, ", group_by=[{}]", self.label_columns.join(", "))
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
            at_timestamp_ns: self.at_timestamp_ns,
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        // Don't push timestamp filters past this node; we handle time alignment.
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let schema = self.input.schema();

        // Start with what the parent needs.
        let mut needed: HashSet<usize> = output_columns.iter().copied().collect();

        // Always require timestamp, value, and all label columns — these are
        // needed for the lookback-window alignment even if the parent doesn't
        // reference them.
        for (i, field) in schema.fields().iter().enumerate() {
            let name = field.name().as_str();
            if name == "timestamp"
                || name == "value"
                || self.label_columns.iter().any(|lc| lc == name)
            {
                needed.insert(i);
            }
        }

        let mut indices: Vec<usize> = needed.into_iter().collect();
        indices.sort();
        // Single child.
        Some(vec![indices])
    }
}

impl PartialEq for InstantVectorEval {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp_ns == other.timestamp_ns
            && self.lookback_ns == other.lookback_ns
            && self.offset_ns == other.offset_ns
            && self.label_columns == other.label_columns
            && self.at_timestamp_ns == other.at_timestamp_ns
    }
}

impl Eq for InstantVectorEval {}

impl Hash for InstantVectorEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.timestamp_ns.hash(state);
        self.lookback_ns.hash(state);
        self.offset_ns.hash(state);
        self.label_columns.hash(state);
        self.at_timestamp_ns.hash(state);
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
