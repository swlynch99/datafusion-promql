use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNodeCore};

use crate::error::{PromqlError, Result};

/// Binary operators supported in PromQL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eql,
    Neq,
    Lss,
    Gtr,
    Lte,
    Gte,
    Land,
    Lor,
    Lunless,
}

impl BinaryOp {
    /// Apply the binary operation to two values.
    /// Returns `None` for comparison operators when the comparison is false
    /// (used for filtering mode, i.e. without `bool` modifier).
    pub fn evaluate(&self, lhs: f64, rhs: f64) -> Option<f64> {
        match self {
            Self::Add => Some(lhs + rhs),
            Self::Sub => Some(lhs - rhs),
            Self::Mul => Some(lhs * rhs),
            Self::Div => Some(lhs / rhs),
            Self::Mod => Some(lhs % rhs),
            Self::Pow => Some(lhs.powf(rhs)),
            Self::Eql => {
                if lhs == rhs {
                    Some(lhs)
                } else {
                    None
                }
            }
            Self::Neq => {
                if lhs != rhs {
                    Some(lhs)
                } else {
                    None
                }
            }
            Self::Lss => {
                if lhs < rhs {
                    Some(lhs)
                } else {
                    None
                }
            }
            Self::Gtr => {
                if lhs > rhs {
                    Some(lhs)
                } else {
                    None
                }
            }
            Self::Lte => {
                if lhs <= rhs {
                    Some(lhs)
                } else {
                    None
                }
            }
            Self::Gte => {
                if lhs >= rhs {
                    Some(lhs)
                } else {
                    None
                }
            }
            // Set operators are handled at the series level, not per-value.
            Self::Land | Self::Lor | Self::Lunless => Some(lhs),
        }
    }

    /// Apply with `bool` modifier: return 1.0 if comparison is true, 0.0 if false.
    pub fn evaluate_bool(&self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Eql => {
                if lhs == rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Neq => {
                if lhs != rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Lss => {
                if lhs < rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Gtr => {
                if lhs > rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Lte => {
                if lhs <= rhs {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Gte => {
                if lhs >= rhs {
                    1.0
                } else {
                    0.0
                }
            }
            _ => self.evaluate(lhs, rhs).unwrap_or(0.0),
        }
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Eql | Self::Neq | Self::Lss | Self::Gtr | Self::Lte | Self::Gte
        )
    }

    pub fn is_set_operator(&self) -> bool {
        matches!(self, Self::Land | Self::Lor | Self::Lunless)
    }

    /// Whether this operator should drop __name__ from the result.
    pub fn drops_metric_name(&self) -> bool {
        // Arithmetic and comparison operators drop __name__
        !self.is_set_operator()
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Mod => write!(f, "%"),
            Self::Pow => write!(f, "^"),
            Self::Eql => write!(f, "=="),
            Self::Neq => write!(f, "!="),
            Self::Lss => write!(f, "<"),
            Self::Gtr => write!(f, ">"),
            Self::Lte => write!(f, "<="),
            Self::Gte => write!(f, ">="),
            Self::Land => write!(f, "and"),
            Self::Lor => write!(f, "or"),
            Self::Lunless => write!(f, "unless"),
        }
    }
}

/// Convert a promql-parser TokenType to our BinaryOp.
pub(crate) fn convert_binary_op(op: promql_parser::parser::token::TokenType) -> Result<BinaryOp> {
    match op.to_string().as_str() {
        "+" => Ok(BinaryOp::Add),
        "-" => Ok(BinaryOp::Sub),
        "*" => Ok(BinaryOp::Mul),
        "/" => Ok(BinaryOp::Div),
        "%" => Ok(BinaryOp::Mod),
        "^" => Ok(BinaryOp::Pow),
        "==" => Ok(BinaryOp::Eql),
        "!=" => Ok(BinaryOp::Neq),
        "<" => Ok(BinaryOp::Lss),
        ">" => Ok(BinaryOp::Gtr),
        "<=" => Ok(BinaryOp::Lte),
        ">=" => Ok(BinaryOp::Gte),
        "and" => Ok(BinaryOp::Land),
        "or" => Ok(BinaryOp::Lor),
        "unless" => Ok(BinaryOp::Lunless),
        other => Err(PromqlError::NotImplemented(format!(
            "binary operator not yet supported: {other}"
        ))),
    }
}

/// How series are matched between two vectors in a binary operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct VectorMatching {
    pub card: MatchCardinality,
    /// If `Some`, match only on these labels (`on(...)`).
    pub on_labels: Option<Vec<String>>,
    /// If `Some`, match on all labels except these (`ignoring(...)`).
    pub ignoring_labels: Option<Vec<String>>,
}

impl VectorMatching {
    /// Default matching: match on all labels.
    pub fn default_matching() -> Self {
        Self {
            card: MatchCardinality::OneToOne,
            on_labels: None,
            ignoring_labels: None,
        }
    }
}

/// Cardinality of binary vector matching.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MatchCardinality {
    OneToOne,
    ManyToOne(Vec<String>),
    OneToMany(Vec<String>),
}

// ---------------------------------------------------------------------------
// BinaryEval: vector op vector
// ---------------------------------------------------------------------------

/// Custom logical node for binary operations between two instant vectors.
#[derive(Debug, Clone)]
pub(crate) struct BinaryEval {
    pub lhs: LogicalPlan,
    pub rhs: LogicalPlan,
    pub op: BinaryOp,
    pub return_bool: bool,
    pub matching: VectorMatching,
    pub output_schema: DFSchemaRef,
}

impl BinaryEval {
    pub fn new(
        lhs: LogicalPlan,
        rhs: LogicalPlan,
        op: BinaryOp,
        return_bool: bool,
        matching: VectorMatching,
    ) -> Result<Self> {
        let output_schema = compute_binary_output_schema(&lhs, &rhs, &op, &matching)?;
        Ok(Self {
            lhs,
            rhs,
            op,
            return_bool,
            matching,
            output_schema,
        })
    }
}

/// Compute the output schema for a vector-vector binary operation.
fn compute_binary_output_schema(
    lhs: &LogicalPlan,
    _rhs: &LogicalPlan,
    op: &BinaryOp,
    matching: &VectorMatching,
) -> Result<DFSchemaRef> {
    let lhs_schema = lhs.schema();

    // Get label columns from the LHS (everything except timestamp/value).
    let lhs_labels: Vec<String> = lhs_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .filter(|n| n != "timestamp" && n != "value")
        .collect();

    // Determine output label columns.
    let output_labels: Vec<String> = match (&matching.on_labels, &matching.ignoring_labels) {
        (Some(on), _) => {
            // on(...): keep only specified labels
            on.iter()
                .filter(|l| lhs_labels.contains(l))
                .cloned()
                .collect()
        }
        (_, Some(ignoring)) => {
            // ignoring(...): keep all except specified
            let ignore_set: HashSet<&str> = ignoring.iter().map(|s| s.as_str()).collect();
            lhs_labels
                .iter()
                .filter(|l| !ignore_set.contains(l.as_str()))
                .filter(|l| !op.drops_metric_name() || l.as_str() != "__name__")
                .cloned()
                .collect()
        }
        (None, None) => {
            // Default: keep all labels (minus __name__ for arithmetic/comparison)
            lhs_labels
                .iter()
                .filter(|l| !op.drops_metric_name() || l.as_str() != "__name__")
                .cloned()
                .collect()
        }
    };

    let mut fields = vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ];
    for label in &output_labels {
        fields.push(Field::new(label, DataType::Utf8, false));
    }

    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl UserDefinedLogicalNodeCore for BinaryEval {
    fn name(&self) -> &str {
        "BinaryEval"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.lhs, &self.rhs]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.output_schema
    }

    fn expressions(&self) -> Vec<datafusion::logical_expr::Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryEval: op={}", self.op)
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        let rhs = inputs.pop().unwrap();
        let lhs = inputs.pop().unwrap();
        Ok(Self {
            lhs,
            rhs,
            op: self.op,
            return_bool: self.return_bool,
            matching: self.matching.clone(),
            output_schema: Arc::clone(&self.output_schema),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

impl PartialEq for BinaryEval {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op
            && self.return_bool == other.return_bool
            && self.matching == other.matching
    }
}
impl Eq for BinaryEval {}

impl Hash for BinaryEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        self.return_bool.hash(state);
        self.matching.hash(state);
    }
}

impl PartialOrd for BinaryEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BinaryEval {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

// ---------------------------------------------------------------------------
// ScalarBinaryEval: vector op scalar (or scalar op vector)
// ---------------------------------------------------------------------------

/// Custom logical node for binary operations between a vector and a scalar.
#[derive(Debug, Clone)]
pub(crate) struct ScalarBinaryEval {
    pub input: LogicalPlan,
    pub scalar_value: f64,
    pub op: BinaryOp,
    /// `true` if the scalar is on the left-hand side: `scalar op vector`.
    pub scalar_is_lhs: bool,
    pub return_bool: bool,
    pub output_schema: DFSchemaRef,
}

impl ScalarBinaryEval {
    pub fn new(
        input: LogicalPlan,
        scalar_value: f64,
        op: BinaryOp,
        scalar_is_lhs: bool,
        return_bool: bool,
    ) -> Result<Self> {
        let output_schema = compute_scalar_binary_output_schema(&input, &op)?;
        Ok(Self {
            input,
            scalar_value,
            op,
            scalar_is_lhs,
            return_bool,
            output_schema,
        })
    }
}

/// For scalar-vector binary ops, output schema is same as input but may drop __name__.
fn compute_scalar_binary_output_schema(
    input: &LogicalPlan,
    op: &BinaryOp,
) -> Result<DFSchemaRef> {
    if !op.drops_metric_name() {
        return Ok(Arc::clone(input.schema()));
    }

    let input_schema = input.schema();
    let fields: Vec<Field> = input_schema
        .fields()
        .iter()
        .filter(|f| f.name() != "__name__")
        .map(|f| f.as_ref().clone())
        .collect();

    let schema = Schema::new(fields);
    let df_schema =
        DFSchema::try_from(schema).map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
    Ok(Arc::new(df_schema))
}

impl UserDefinedLogicalNodeCore for ScalarBinaryEval {
    fn name(&self) -> &str {
        "ScalarBinaryEval"
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
        if self.scalar_is_lhs {
            write!(f, "ScalarBinaryEval: {} {} vector", self.scalar_value, self.op)
        } else {
            write!(f, "ScalarBinaryEval: vector {} {}", self.op, self.scalar_value)
        }
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            scalar_value: self.scalar_value,
            op: self.op,
            scalar_is_lhs: self.scalar_is_lhs,
            return_bool: self.return_bool,
            output_schema: Arc::clone(&self.output_schema),
        })
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        let mut cols = HashSet::new();
        cols.insert("timestamp".to_string());
        cols
    }
}

// ScalarBinaryEval needs Hash for f64 which doesn't impl Hash.
// Use bit representation for hashing.
impl PartialEq for ScalarBinaryEval {
    fn eq(&self, other: &Self) -> bool {
        self.scalar_value.to_bits() == other.scalar_value.to_bits()
            && self.op == other.op
            && self.scalar_is_lhs == other.scalar_is_lhs
            && self.return_bool == other.return_bool
    }
}
impl Eq for ScalarBinaryEval {}

impl Hash for ScalarBinaryEval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scalar_value.to_bits().hash(state);
        self.op.hash(state);
        self.scalar_is_lhs.hash(state);
        self.return_bool.hash(state);
    }
}

impl PartialOrd for ScalarBinaryEval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScalarBinaryEval {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}
