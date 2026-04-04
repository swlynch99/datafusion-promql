use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::{Extension, LogicalPlan, LogicalPlanBuilder, cast, col};
use promql_parser::parser::ast::Offset;
use promql_parser::parser::{self, Expr, LabelModifier};

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::Field;
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;

use crate::datasource::MetricSource;
use crate::error::{PromqlError, Result};
use crate::func::{
    AggregateFunction, datetime_func_to_expr, is_time_function, lookup_aggregate_function,
    lookup_datetime_function, lookup_instant_function, lookup_range_function,
};
use crate::node::{
    BinaryEval, InstantFuncEval, InstantVectorEval, MatchCardinality, RangeFunctionEval,
    RangeVectorEval, ScalarBinaryEval, VectorMatching, convert_binary_op,
};
use crate::types::{DEFAULT_LOOKBACK_NS, TimeRange};

use super::selector::plan_vector_selector;

/// Convert a promql-parser `Offset` to a signed nanoseconds value.
/// Positive = shift lookup window into the past, negative = into the future.
fn offset_to_ns(offset: &Option<Offset>) -> i64 {
    match offset {
        Some(Offset::Pos(dur)) => dur.as_nanos() as i64,
        Some(Offset::Neg(dur)) => -(dur.as_nanos() as i64),
        None => 0,
    }
}

/// Parameters controlling how evaluation timestamps are generated.
#[derive(Debug, Clone, Copy)]
pub struct EvalParams {
    /// For instant queries: the single evaluation timestamp (ns).
    /// `None` for range queries (timestamps generated from start/end/step).
    pub eval_ts_ns: Option<i64>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub step_ns: i64,
}

/// Extract label column names from a schema (everything except timestamp/value).
fn label_columns_from_schema(schema: &datafusion::common::DFSchemaRef) -> Vec<String> {
    schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .filter(|n| n != "timestamp" && n != "value")
        .collect()
}

/// Translate a promql-parser AST `Expr` into a DataFusion `LogicalPlan`.
pub async fn plan_expr(
    expr: &Expr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    match expr {
        Expr::VectorSelector(vs) => {
            let offset_ns = offset_to_ns(&vs.offset);
            let (child_plan, label_columns) =
                plan_vector_selector(vs, source, time_range, 0, offset_ns).await?;

            let node = if let Some(ts) = params.eval_ts_ns {
                InstantVectorEval::instant(
                    child_plan,
                    ts,
                    DEFAULT_LOOKBACK_NS,
                    offset_ns,
                    label_columns,
                )
            } else {
                InstantVectorEval::range(
                    child_plan,
                    params.start_ns,
                    params.end_ns,
                    params.step_ns,
                    DEFAULT_LOOKBACK_NS,
                    offset_ns,
                    label_columns,
                )
            };

            Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            }))
        }

        Expr::Call(call) => plan_call(call, source, time_range, params).await,

        Expr::MatrixSelector(_) => Err(PromqlError::Plan(
            "bare matrix selector is not allowed as a top-level expression; \
             use it inside a range function like rate()"
                .into(),
        )),

        Expr::NumberLiteral(_) | Expr::StringLiteral(_) => Err(PromqlError::NotImplemented(
            "scalar/string literals as top-level query not yet implemented".into(),
        )),

        Expr::Paren(paren) => Box::pin(plan_expr(&paren.expr, source, time_range, params)).await,

        Expr::Aggregate(agg) => plan_aggregate(agg, source, time_range, params).await,

        Expr::Binary(bin) => plan_binary(bin, source, time_range, params).await,

        Expr::Unary(unary) => plan_unary(unary, source, time_range, params).await,

        _ => Err(PromqlError::NotImplemented(format!(
            "expression type not yet supported: {expr:?}"
        ))),
    }
}

/// Plan a function call expression.
async fn plan_call(
    call: &parser::Call,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    let func_name = call.func.name;

    // Extra scalar arguments (args after the first vector arg) for functions like round().
    let extra_scalar_args: Vec<f64> = call
        .args
        .args
        .iter()
        .skip(1)
        .filter_map(|arg| {
            if let Expr::NumberLiteral(lit) = arg.as_ref() {
                Some(lit.val)
            } else {
                None
            }
        })
        .collect();

    // Check if this is an instant vector function.
    if let Some(func) = lookup_instant_function(func_name, &extra_scalar_args) {
        if call.args.args.is_empty() {
            return Err(PromqlError::Plan(format!(
                "{func_name}() requires at least 1 argument"
            )));
        }
        let vector_arg = &call.args.args[0];
        let child_plan = Box::pin(plan_expr(vector_arg, source, time_range, params)).await?;
        let node = InstantFuncEval::new(child_plan, func)?;
        return Ok(LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        }));
    }

    // Check if this is a range vector function.
    if let Some(range_func) = lookup_range_function(func_name) {
        // Range functions expect exactly one argument: a MatrixSelector.
        if call.args.args.len() != 1 {
            return Err(PromqlError::Plan(format!(
                "{func_name}() requires exactly 1 argument, got {}",
                call.args.args.len()
            )));
        }

        let arg = &call.args.args[0];
        let matrix = match arg.as_ref() {
            Expr::MatrixSelector(ms) => ms,
            _ => {
                return Err(PromqlError::Plan(format!(
                    "{func_name}() requires a range vector (matrix selector) argument"
                )));
            }
        };

        let range_ns = matrix.range.as_nanos() as i64;
        let offset_ns = offset_to_ns(&matrix.vs.offset);

        // Plan the inner vector selector with extra range expansion.
        let (child_plan, label_columns) =
            plan_vector_selector(&matrix.vs, source, time_range, range_ns, offset_ns).await?;

        // Wrap in RangeVectorEval (windowing) then RangeFunctionEval (function).
        let window_node = if let Some(ts) = params.eval_ts_ns {
            RangeVectorEval::instant(child_plan, ts, range_ns, offset_ns, label_columns)?
        } else {
            RangeVectorEval::range(
                child_plan,
                params.start_ns,
                params.end_ns,
                params.step_ns,
                range_ns,
                offset_ns,
                label_columns,
            )?
        };

        let window_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(window_node),
        });

        let func_node = RangeFunctionEval::new(window_plan, range_func)?;

        return Ok(LogicalPlan::Extension(Extension {
            node: Arc::new(func_node),
        }));
    }

    // Check if this is the time() function (no arguments, returns eval timestamp).
    if is_time_function(func_name) {
        if !call.args.args.is_empty() {
            return Err(PromqlError::Plan("time() takes no arguments".into()));
        }
        return plan_time_function(params);
    }

    // Check if this is a datetime function (timestamp, day_of_month, etc.).
    if let Some(dt_func) = lookup_datetime_function(func_name) {
        return plan_datetime_function(dt_func, call, source, time_range, params).await;
    }

    Err(PromqlError::NotImplemented(format!(
        "function not yet supported: {func_name}"
    )))
}

/// Plan an aggregation expression (sum, avg, count, min, max).
///
/// Produces a native DataFusion `Aggregate` logical plan so that the
/// optimizer can push projections and filters through it.
async fn plan_aggregate(
    agg: &parser::AggregateExpr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    let func = lookup_aggregate_function(agg.op)?;

    // Recursively plan the inner expression.
    let child_plan = Box::pin(plan_expr(&agg.expr, source, time_range, params)).await?;

    // Determine grouping labels from the modifier and child schema.
    let child_label_cols = label_columns_from_schema(child_plan.schema());
    let grouping_labels = compute_grouping_labels(&agg.modifier, &child_label_cols);

    // Build group-by expressions: always include timestamp, plus grouping labels.
    let mut group_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in &grouping_labels {
        group_exprs.push(col(label.as_str()));
    }

    // Map our aggregate function to a DataFusion aggregate expression on the
    // "value" column.
    let value_col = col("value");
    let agg_expr = match func {
        AggregateFunction::Sum => datafusion::functions_aggregate::sum::sum(value_col),
        AggregateFunction::Avg => datafusion::functions_aggregate::average::avg(value_col),
        AggregateFunction::Count => datafusion::functions_aggregate::count::count(value_col),
        AggregateFunction::Min => datafusion::functions_aggregate::min_max::min(value_col),
        AggregateFunction::Max => datafusion::functions_aggregate::min_max::max(value_col),
    }
    .alias("value");

    let mut builder = LogicalPlanBuilder::from(child_plan)
        .aggregate(group_exprs, vec![agg_expr])
        .map_err(|e| PromqlError::Plan(format!("aggregate plan error: {e}")))?;

    // COUNT returns Int64 but downstream expects Float64 for the "value"
    // column, so add a projection to cast.
    if func == AggregateFunction::Count {
        let mut proj_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
        for label in &grouping_labels {
            proj_exprs.push(col(label.as_str()));
        }
        proj_exprs.push(cast(col("value"), DataType::Float64).alias("value"));
        builder = builder
            .project(proj_exprs)
            .map_err(|e| PromqlError::Plan(format!("count cast projection error: {e}")))?;
    }

    let plan = builder
        .build()
        .map_err(|e| PromqlError::Plan(format!("aggregate build error: {e}")))?;

    Ok(plan)
}

/// Compute grouping labels from a LabelModifier and the available child label columns.
fn compute_grouping_labels(
    modifier: &Option<LabelModifier>,
    child_label_cols: &[String],
) -> Vec<String> {
    match modifier {
        Some(LabelModifier::Include(labels)) => {
            // by(...): keep only specified labels that exist in the child schema.
            // Exclude __name__ from grouping by default.
            labels
                .labels
                .iter()
                .filter(|l| child_label_cols.contains(l) && l.as_str() != "__name__")
                .cloned()
                .collect()
        }
        Some(LabelModifier::Exclude(labels)) => {
            // without(...): keep all child labels except specified ones and __name__.
            let exclude: HashSet<&str> = labels.labels.iter().map(|s| s.as_str()).collect();
            child_label_cols
                .iter()
                .filter(|l| !exclude.contains(l.as_str()) && l.as_str() != "__name__")
                .cloned()
                .collect()
        }
        None => {
            // No modifier: aggregate all into one group (no grouping labels).
            vec![]
        }
    }
}

/// Plan a binary expression.
async fn plan_binary(
    bin: &parser::BinaryExpr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    let op = convert_binary_op(bin.op)?;

    let lhs_is_scalar = matches!(bin.lhs.as_ref(), Expr::NumberLiteral(_));
    let rhs_is_scalar = matches!(bin.rhs.as_ref(), Expr::NumberLiteral(_));

    let return_bool = bin
        .modifier
        .as_ref()
        .map(|m| m.return_bool)
        .unwrap_or(false);

    match (lhs_is_scalar, rhs_is_scalar) {
        (true, true) => {
            // scalar op scalar: not yet implemented
            Err(PromqlError::NotImplemented(
                "scalar op scalar not yet supported".into(),
            ))
        }
        (true, false) => {
            // scalar op vector
            let scalar_val = extract_scalar(&bin.lhs)?;
            let rhs_plan = Box::pin(plan_expr(&bin.rhs, source, time_range, params)).await?;
            let node = ScalarBinaryEval::new(rhs_plan, scalar_val, op, true, return_bool)?;
            Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            }))
        }
        (false, true) => {
            // vector op scalar
            let scalar_val = extract_scalar(&bin.rhs)?;
            let lhs_plan = Box::pin(plan_expr(&bin.lhs, source, time_range, params)).await?;
            let node = ScalarBinaryEval::new(lhs_plan, scalar_val, op, false, return_bool)?;
            Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            }))
        }
        (false, false) => {
            // vector op vector
            let lhs_plan = Box::pin(plan_expr(&bin.lhs, source, time_range, params)).await?;
            let rhs_plan = Box::pin(plan_expr(&bin.rhs, source, time_range, params)).await?;

            let matching = extract_vector_matching(bin)?;
            let node = BinaryEval::new(lhs_plan, rhs_plan, op, return_bool, matching)?;

            Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            }))
        }
    }
}

/// Extract the scalar value from a NumberLiteral expression.
fn extract_scalar(expr: &Expr) -> Result<f64> {
    match expr {
        Expr::NumberLiteral(lit) => Ok(lit.val),
        _ => Err(PromqlError::Plan(
            "expected a number literal for scalar operand".into(),
        )),
    }
}

/// Extract VectorMatching from a BinaryExpr's modifier.
fn extract_vector_matching(bin: &parser::BinaryExpr) -> Result<VectorMatching> {
    let modifier = match &bin.modifier {
        Some(m) => m,
        None => return Ok(VectorMatching::default_matching()),
    };

    let card = match &modifier.card {
        parser::VectorMatchCardinality::OneToOne => MatchCardinality::OneToOne,
        parser::VectorMatchCardinality::ManyToOne(labels) => {
            MatchCardinality::ManyToOne(labels.labels.clone())
        }
        parser::VectorMatchCardinality::OneToMany(labels) => {
            MatchCardinality::OneToMany(labels.labels.clone())
        }
        parser::VectorMatchCardinality::ManyToMany => MatchCardinality::OneToOne,
    };

    let (on_labels, ignoring_labels) = match &modifier.matching {
        Some(LabelModifier::Include(labels)) => (Some(labels.labels.clone()), None),
        Some(LabelModifier::Exclude(labels)) => (None, Some(labels.labels.clone())),
        None => (None, None),
    };

    Ok(VectorMatching {
        card,
        on_labels,
        ignoring_labels,
    })
}

/// Plan a unary expression (negation).
async fn plan_unary(
    unary: &parser::UnaryExpr,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    // Unary negation: multiply by -1
    let child_plan = Box::pin(plan_expr(&unary.expr, source, time_range, params)).await?;

    use crate::node::BinaryOp;
    let node = ScalarBinaryEval::new(child_plan, -1.0, BinaryOp::Mul, true, false)?;
    Ok(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    }))
}

/// Plan the `time()` function: returns evaluation timestamps as float64 seconds.
///
/// Generates a synthetic series with no labels, where each row has:
/// - `timestamp` = step timestamp (ns)
/// - `value` = step timestamp in seconds (float64)
fn plan_time_function(params: EvalParams) -> Result<LogicalPlan> {
    use crate::func::DateTimeFunction;
    plan_synthetic_datetime(DateTimeFunction::Timestamp, params)
}

/// Plan a datetime function.
///
/// When called with a vector argument, applies the function to each sample's timestamp.
/// When called without arguments, generates a synthetic series using eval timestamps.
async fn plan_datetime_function(
    dt_func: crate::func::DateTimeFunction,
    call: &parser::Call,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    if call.args.args.is_empty() {
        // No arguments: apply function to evaluation timestamps.
        return plan_synthetic_datetime(dt_func, params);
    }

    // Has a vector argument: plan the child, then project timestamp → value.
    let vector_arg = &call.args.args[0];
    let child_plan = Box::pin(plan_expr(vector_arg, source, time_range, params)).await?;

    // Build a projection that replaces `value` with dt_func(timestamp)
    // and drops __name__ (since datetime functions change the meaning of the value).
    let child_schema = child_plan.schema();
    let mut exprs: Vec<datafusion::logical_expr::Expr> = Vec::new();

    for field in child_schema.fields() {
        let name = field.name();
        if name == "__name__" {
            continue;
        }
        let (qualifier, child_field) = child_schema
            .qualified_field_with_name(None, name.as_str())
            .map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
        let col_expr = datafusion::logical_expr::Expr::Column(datafusion::common::Column::from((
            qualifier,
            child_field,
        )));

        if name == "value" {
            // Replace value with the datetime function applied to the timestamp column.
            let (ts_qualifier, ts_field) = child_schema
                .qualified_field_with_name(None, "timestamp")
                .map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
            let ts_expr = datafusion::logical_expr::Expr::Column(datafusion::common::Column::from(
                (ts_qualifier, ts_field),
            ));
            exprs.push(datetime_func_to_expr(dt_func, ts_expr));
        } else {
            exprs.push(col_expr.alias(name.as_str()));
        }
    }

    let plan = LogicalPlanBuilder::from(child_plan)
        .project(exprs)
        .map_err(|e| PromqlError::Plan(format!("datetime projection error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("datetime build error: {e}")))?;

    Ok(plan)
}

/// Generate a synthetic series for no-argument datetime functions (and `time()`).
///
/// Creates a MemTable with one row per step timestamp, then applies the datetime
/// function to compute the value.
fn plan_synthetic_datetime(
    dt_func: crate::func::DateTimeFunction,
    params: EvalParams,
) -> Result<LogicalPlan> {
    // Generate step timestamps.
    let timestamps: Vec<i64> = if let Some(ts) = params.eval_ts_ns {
        vec![ts]
    } else {
        let mut ts_vec = Vec::new();
        let mut t = params.start_ns;
        while t <= params.end_ns {
            ts_vec.push(t);
            t += params.step_ns;
        }
        ts_vec
    };

    // Compute values by applying the datetime function to each timestamp.
    let values: Vec<f64> = timestamps
        .iter()
        .map(|&ts| dt_func.evaluate_ns(ts))
        .collect();
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .map_err(|e| PromqlError::Plan(format!("failed to create time() batch: {e}")))?;

    let mem_table = MemTable::try_new(Arc::clone(&schema), vec![vec![batch]])
        .map_err(|e| PromqlError::Plan(format!("failed to create time() table: {e}")))?;

    let table_source =
        provider_as_source(Arc::new(mem_table) as Arc<dyn datafusion::catalog::TableProvider>);
    let plan = LogicalPlanBuilder::scan("time_series", table_source, None)
        .map_err(|e| PromqlError::Plan(format!("time() scan error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("time() build error: {e}")))?;

    Ok(plan)
}
