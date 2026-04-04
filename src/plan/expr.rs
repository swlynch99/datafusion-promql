use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::common::Column;
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::expr::Sort;
use datafusion::logical_expr::{
    Extension, LogicalPlan, LogicalPlanBuilder, WindowFrame, WindowFrameBound, WindowFrameUnits,
    cast, col, lit,
};
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
    make_label_join_udf, make_label_replace_udf,
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

    // Check if this is a label manipulation function.
    if func_name == "label_replace" || func_name == "label_join" {
        return plan_label_function(call, source, time_range, params).await;
    }

    Err(PromqlError::NotImplemented(format!(
        "function not yet supported: {func_name}"
    )))
}

/// Extract a string literal from a PromQL expression.
fn extract_string_literal(expr: &Expr) -> Result<String> {
    match expr {
        Expr::StringLiteral(lit) => Ok(lit.val.clone()),
        _ => Err(PromqlError::Plan(
            "expected a string literal argument".into(),
        )),
    }
}

/// Plan a `label_replace` or `label_join` function call.
///
/// These functions modify label columns rather than the value column, so they
/// are implemented as projections with UDFs over string columns.
async fn plan_label_function(
    call: &parser::Call,
    source: &dyn MetricSource,
    time_range: TimeRange,
    params: EvalParams,
) -> Result<LogicalPlan> {
    let func_name = call.func.name;

    match func_name {
        "label_replace" => {
            // label_replace(v, dst_label, replacement, src_label, regex)
            if call.args.args.len() != 5 {
                return Err(PromqlError::Plan(format!(
                    "label_replace() requires exactly 5 arguments, got {}",
                    call.args.args.len()
                )));
            }

            let vector_arg = &call.args.args[0];
            let dst_label = extract_string_literal(&call.args.args[1])?;
            let replacement = extract_string_literal(&call.args.args[2])?;
            let src_label = extract_string_literal(&call.args.args[3])?;
            let regex_pattern = extract_string_literal(&call.args.args[4])?;

            let child_plan =
                Box::pin(plan_expr(vector_arg, source, time_range, params)).await?;

            build_label_replace_projection(
                child_plan,
                &dst_label,
                &replacement,
                &src_label,
                &regex_pattern,
            )
        }
        "label_join" => {
            // label_join(v, dst_label, separator, src_label_1, src_label_2, ...)
            if call.args.args.len() < 4 {
                return Err(PromqlError::Plan(format!(
                    "label_join() requires at least 4 arguments, got {}",
                    call.args.args.len()
                )));
            }

            let vector_arg = &call.args.args[0];
            let dst_label = extract_string_literal(&call.args.args[1])?;
            let separator = extract_string_literal(&call.args.args[2])?;
            let src_labels: Vec<String> = call.args.args[3..]
                .iter()
                .map(|a| extract_string_literal(a))
                .collect::<Result<_>>()?;

            let child_plan =
                Box::pin(plan_expr(vector_arg, source, time_range, params)).await?;

            build_label_join_projection(child_plan, &dst_label, &separator, &src_labels)
        }
        _ => unreachable!(),
    }
}

/// Build a projection plan for `label_replace`.
///
/// Passes through all existing columns and adds/replaces `dst_label` with the
/// result of applying the regex replacement on `src_label`.
fn build_label_replace_projection(
    child_plan: LogicalPlan,
    dst_label: &str,
    replacement: &str,
    src_label: &str,
    regex_pattern: &str,
) -> Result<LogicalPlan> {
    let child_schema = child_plan.schema();
    let udf = make_label_replace_udf(replacement.to_string(), regex_pattern.to_string());

    // Determine the src_label expression. If the column doesn't exist, use empty string.
    let src_expr = if child_schema
        .fields()
        .iter()
        .any(|f| f.name() == src_label)
    {
        col(src_label)
    } else {
        lit("").alias(src_label)
    };

    // Determine if dst_label already exists.
    let dst_exists = child_schema
        .fields()
        .iter()
        .any(|f| f.name() == dst_label);

    // The current dst value: if column exists, use it; otherwise use empty string.
    let current_dst_expr = if dst_exists {
        col(dst_label)
    } else {
        lit("")
    };

    // Build projection: pass through all columns, add/replace dst_label.
    let mut exprs: Vec<datafusion::logical_expr::Expr> = Vec::new();
    for field in child_schema.fields() {
        let name = field.name();
        if name == dst_label {
            // Replace with UDF result.
            let (qualifier, child_field) = child_schema
                .qualified_field_with_name(None, name)
                .map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
            let _col_expr =
                datafusion::logical_expr::Expr::Column(Column::from((qualifier, child_field)));
            let replace_expr = udf.call(vec![src_expr.clone(), current_dst_expr.clone()]);
            exprs.push(replace_expr.alias(dst_label));
        } else {
            let (qualifier, child_field) = child_schema
                .qualified_field_with_name(None, name)
                .map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
            let col_expr =
                datafusion::logical_expr::Expr::Column(Column::from((qualifier, child_field)));
            exprs.push(col_expr.alias(name.as_str()));
        }
    }

    // If dst_label doesn't exist, add it as a new column.
    if !dst_exists {
        let replace_expr = udf.call(vec![src_expr, current_dst_expr]);
        exprs.push(replace_expr.alias(dst_label));
    }

    LogicalPlanBuilder::from(child_plan)
        .project(exprs)
        .map_err(|e| PromqlError::Plan(format!("label_replace projection error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("label_replace build error: {e}")))
}

/// Build a projection plan for `label_join`.
///
/// Passes through all existing columns and adds/replaces `dst_label` with the
/// concatenation of source label values joined by `separator`.
fn build_label_join_projection(
    child_plan: LogicalPlan,
    dst_label: &str,
    separator: &str,
    src_labels: &[String],
) -> Result<LogicalPlan> {
    let child_schema = child_plan.schema();
    let udf = make_label_join_udf(separator.to_string(), src_labels.len());

    // Build the UDF arguments: one expression per source label.
    let src_exprs: Vec<datafusion::logical_expr::Expr> = src_labels
        .iter()
        .map(|label| {
            if child_schema
                .fields()
                .iter()
                .any(|f| f.name() == label.as_str())
            {
                col(label.as_str())
            } else {
                lit("")
            }
        })
        .collect();

    let dst_exists = child_schema
        .fields()
        .iter()
        .any(|f| f.name() == dst_label);

    // Build projection.
    let mut exprs: Vec<datafusion::logical_expr::Expr> = Vec::new();
    for field in child_schema.fields() {
        let name = field.name();
        if name == dst_label {
            let join_expr = udf.call(src_exprs.clone());
            exprs.push(join_expr.alias(dst_label));
        } else {
            let (qualifier, child_field) = child_schema
                .qualified_field_with_name(None, name)
                .map_err(|e| PromqlError::Plan(format!("schema error: {e}")))?;
            let col_expr =
                datafusion::logical_expr::Expr::Column(Column::from((qualifier, child_field)));
            exprs.push(col_expr.alias(name.as_str()));
        }
    }

    // If dst_label doesn't exist, add it.
    if !dst_exists {
        let join_expr = udf.call(src_exprs);
        exprs.push(join_expr.alias(dst_label));
    }

    LogicalPlanBuilder::from(child_plan)
        .project(exprs)
        .map_err(|e| PromqlError::Plan(format!("label_join projection error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("label_join build error: {e}")))
}

/// Plan an aggregation expression.
///
/// For simple aggregations (sum, avg, count, min, max, stddev, stdvar, group),
/// produces a native DataFusion `Aggregate` logical plan.
/// For complex aggregations (topk, bottomk, quantile, count_values, limitk, limit_ratio),
/// uses window functions or custom plan structures.
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

    match func {
        AggregateFunction::TopK | AggregateFunction::BottomK => {
            let k = extract_int_param(agg, &func)?;
            plan_topk_bottomk(child_plan, func, k, &grouping_labels, &child_label_cols)
        }
        AggregateFunction::LimitK => {
            let k = extract_int_param(agg, &func)?;
            plan_limitk(child_plan, k, &grouping_labels, &child_label_cols)
        }
        AggregateFunction::LimitRatio => {
            let ratio = extract_float_param(agg, &func)?;
            plan_limit_ratio(child_plan, ratio, &grouping_labels, &child_label_cols)
        }
        AggregateFunction::Quantile => {
            let q = extract_float_param(agg, &func)?;
            plan_quantile(child_plan, q, &grouping_labels)
        }
        AggregateFunction::CountValues => {
            let label_name = extract_string_param(agg)?;
            plan_count_values(child_plan, &label_name, &grouping_labels)
        }
        _ => plan_simple_aggregate(child_plan, func, &grouping_labels),
    }
}

/// Extract an integer parameter from an aggregate expression (e.g. k in topk).
fn extract_int_param(agg: &parser::AggregateExpr, func: &AggregateFunction) -> Result<i64> {
    match &agg.param {
        Some(param) => match param.as_ref() {
            Expr::NumberLiteral(lit) => Ok(lit.val as i64),
            _ => Err(PromqlError::Plan(format!(
                "{func}() requires a scalar integer parameter"
            ))),
        },
        None => Err(PromqlError::Plan(format!("{func}() requires a parameter"))),
    }
}

/// Extract a float parameter from an aggregate expression (e.g. φ in quantile).
fn extract_float_param(agg: &parser::AggregateExpr, func: &AggregateFunction) -> Result<f64> {
    match &agg.param {
        Some(param) => match param.as_ref() {
            Expr::NumberLiteral(lit) => Ok(lit.val),
            _ => Err(PromqlError::Plan(format!(
                "{func}() requires a scalar parameter"
            ))),
        },
        None => Err(PromqlError::Plan(format!("{func}() requires a parameter"))),
    }
}

/// Extract a string parameter from an aggregate expression (e.g. label in count_values).
fn extract_string_param(agg: &parser::AggregateExpr) -> Result<String> {
    match &agg.param {
        Some(param) => match param.as_ref() {
            Expr::StringLiteral(s) => Ok(s.val.clone()),
            _ => Err(PromqlError::Plan(
                "count_values() requires a string parameter".into(),
            )),
        },
        None => Err(PromqlError::Plan(
            "count_values() requires a parameter".into(),
        )),
    }
}

/// Plan simple aggregations that map directly to DataFusion aggregate expressions.
fn plan_simple_aggregate(
    child_plan: LogicalPlan,
    func: AggregateFunction,
    grouping_labels: &[String],
) -> Result<LogicalPlan> {
    // Build group-by expressions: always include timestamp, plus grouping labels.
    let mut group_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in grouping_labels {
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
        AggregateFunction::Stddev => datafusion::functions_aggregate::stddev::stddev_pop(value_col),
        AggregateFunction::Stdvar => datafusion::functions_aggregate::variance::var_pop(value_col),
        AggregateFunction::Group => datafusion::functions_aggregate::count::count(value_col),
        _ => unreachable!("complex aggregations handled separately"),
    }
    .alias("value");

    let mut builder = LogicalPlanBuilder::from(child_plan)
        .aggregate(group_exprs, vec![agg_expr])
        .map_err(|e| PromqlError::Plan(format!("aggregate plan error: {e}")))?;

    // COUNT and GROUP return Int64 but downstream expects Float64 for the "value"
    // column. GROUP additionally needs to project all values as 1.0.
    let needs_cast = matches!(func, AggregateFunction::Count | AggregateFunction::Group);
    if needs_cast {
        let mut proj_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
        for label in grouping_labels {
            proj_exprs.push(col(label.as_str()));
        }
        if func == AggregateFunction::Group {
            // group() always returns 1.0
            proj_exprs.push(lit(1.0_f64).alias("value"));
        } else {
            proj_exprs.push(cast(col("value"), DataType::Float64).alias("value"));
        }
        builder = builder
            .project(proj_exprs)
            .map_err(|e| PromqlError::Plan(format!("cast projection error: {e}")))?;
    }

    let plan = builder
        .build()
        .map_err(|e| PromqlError::Plan(format!("aggregate build error: {e}")))?;

    Ok(plan)
}

/// Plan topk/bottomk aggregation using window functions.
///
/// These preserve original labels (unlike sum/avg which collapse labels).
/// Uses ROW_NUMBER() OVER (PARTITION BY timestamp, group_labels ORDER BY value DESC/ASC)
/// then filters to keep only rows where row_number <= k.
fn plan_topk_bottomk(
    child_plan: LogicalPlan,
    func: AggregateFunction,
    k: i64,
    grouping_labels: &[String],
    all_label_cols: &[String],
) -> Result<LogicalPlan> {
    use datafusion::logical_expr::expr::WindowFunction;
    use datafusion::logical_expr::{WindowFunctionDefinition, expr::WindowFunctionParams};

    // Partition by timestamp + grouping labels
    let mut partition_by: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in grouping_labels {
        partition_by.push(col(label.as_str()));
    }

    // Order by value DESC for topk, ASC for bottomk
    let asc = func == AggregateFunction::BottomK;
    let order_by = vec![Sort {
        expr: col("value"),
        asc,
        nulls_first: false,
    }];

    let row_num_expr = datafusion::logical_expr::Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(Arc::new(
            datafusion::functions_window::row_number::RowNumber::new().into(),
        )),
        params: WindowFunctionParams {
            args: vec![],
            partition_by,
            order_by,
            window_frame: WindowFrame::new_bounds(
                WindowFrameUnits::Rows,
                WindowFrameBound::Preceding(datafusion::common::ScalarValue::UInt64(None)),
                WindowFrameBound::CurrentRow,
            ),
            filter: None,
            null_treatment: None,
            distinct: false,
        },
    }))
    .alias("__row_num");

    // Add window function
    let windowed = LogicalPlanBuilder::from(child_plan)
        .window(vec![row_num_expr])
        .map_err(|e| PromqlError::Plan(format!("topk/bottomk window error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("topk/bottomk window build error: {e}")))?;

    // Filter: __row_num <= k
    let filtered = LogicalPlanBuilder::from(windowed)
        .filter(col("__row_num").lt_eq(lit(k)))
        .map_err(|e| PromqlError::Plan(format!("topk/bottomk filter error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("topk/bottomk filter build error: {e}")))?;

    // Project to drop __row_num, keeping timestamp + all original label columns + value
    let mut proj_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in all_label_cols {
        proj_exprs.push(col(label.as_str()));
    }
    proj_exprs.push(col("value"));

    let plan = LogicalPlanBuilder::from(filtered)
        .project(proj_exprs)
        .map_err(|e| PromqlError::Plan(format!("topk/bottomk project error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("topk/bottomk project build error: {e}")))?;

    Ok(plan)
}

/// Plan limitk aggregation: take K arbitrary series per group.
fn plan_limitk(
    child_plan: LogicalPlan,
    k: i64,
    grouping_labels: &[String],
    all_label_cols: &[String],
) -> Result<LogicalPlan> {
    use datafusion::logical_expr::expr::WindowFunction;
    use datafusion::logical_expr::{WindowFunctionDefinition, expr::WindowFunctionParams};

    let mut partition_by: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in grouping_labels {
        partition_by.push(col(label.as_str()));
    }

    let row_num_expr = datafusion::logical_expr::Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(Arc::new(
            datafusion::functions_window::row_number::RowNumber::new().into(),
        )),
        params: WindowFunctionParams {
            args: vec![],
            partition_by,
            order_by: vec![],
            window_frame: WindowFrame::new_bounds(
                WindowFrameUnits::Rows,
                WindowFrameBound::Preceding(datafusion::common::ScalarValue::UInt64(None)),
                WindowFrameBound::CurrentRow,
            ),
            filter: None,
            null_treatment: None,
            distinct: false,
        },
    }))
    .alias("__row_num");

    let windowed = LogicalPlanBuilder::from(child_plan)
        .window(vec![row_num_expr])
        .map_err(|e| PromqlError::Plan(format!("limitk window error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("limitk window build error: {e}")))?;

    let filtered = LogicalPlanBuilder::from(windowed)
        .filter(col("__row_num").lt_eq(lit(k)))
        .map_err(|e| PromqlError::Plan(format!("limitk filter error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("limitk filter build error: {e}")))?;

    let mut proj_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in all_label_cols {
        proj_exprs.push(col(label.as_str()));
    }
    proj_exprs.push(col("value"));

    let plan = LogicalPlanBuilder::from(filtered)
        .project(proj_exprs)
        .map_err(|e| PromqlError::Plan(format!("limitk project error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("limitk project build error: {e}")))?;

    Ok(plan)
}

/// Plan limit_ratio aggregation: sample a ratio of series per group.
fn plan_limit_ratio(
    child_plan: LogicalPlan,
    ratio: f64,
    grouping_labels: &[String],
    all_label_cols: &[String],
) -> Result<LogicalPlan> {
    use datafusion::logical_expr::expr::WindowFunction;
    use datafusion::logical_expr::{WindowFunctionDefinition, expr::WindowFunctionParams};

    let mut partition_by: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in grouping_labels {
        partition_by.push(col(label.as_str()));
    }

    // Add both ROW_NUMBER and COUNT window functions
    let row_num_expr = datafusion::logical_expr::Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::WindowUDF(Arc::new(
            datafusion::functions_window::row_number::RowNumber::new().into(),
        )),
        params: WindowFunctionParams {
            args: vec![],
            partition_by: partition_by.clone(),
            order_by: vec![],
            window_frame: WindowFrame::new_bounds(
                WindowFrameUnits::Rows,
                WindowFrameBound::Preceding(datafusion::common::ScalarValue::UInt64(None)),
                WindowFrameBound::CurrentRow,
            ),
            filter: None,
            null_treatment: None,
            distinct: false,
        },
    }))
    .alias("__row_num");

    let count_expr = datafusion::logical_expr::Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::AggregateUDF(Arc::new(
            datafusion::functions_aggregate::count::Count::new().into(),
        )),
        params: WindowFunctionParams {
            args: vec![col("value")],
            partition_by,
            order_by: vec![],
            window_frame: WindowFrame::new_bounds(
                WindowFrameUnits::Rows,
                WindowFrameBound::Preceding(datafusion::common::ScalarValue::UInt64(None)),
                WindowFrameBound::Following(datafusion::common::ScalarValue::UInt64(None)),
            ),
            filter: None,
            null_treatment: None,
            distinct: false,
        },
    }))
    .alias("__group_count");

    let windowed = LogicalPlanBuilder::from(child_plan)
        .window(vec![row_num_expr, count_expr])
        .map_err(|e| PromqlError::Plan(format!("limit_ratio window error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("limit_ratio window build error: {e}")))?;

    // Filter: __row_num <= ceil(__group_count * ratio)
    // Implement ceil(x) as: CASE WHEN x = CAST(x AS INT64) THEN x ELSE CAST(x AS INT64) + 1 END
    // Simpler: cast row_num and group_count to float64 and compare.
    // We use: __row_num <= CAST((__group_count * ratio + 0.9999999999) AS INT64)
    // which effectively performs ceiling for practical purposes.
    let raw_threshold = cast(col("__group_count"), DataType::Float64) * lit(ratio);
    // ceil via: cast(x + 1.0 - epsilon) but simpler: just use (x - floor(x) > 0 ? floor(x)+1 : x)
    // Easiest: cast to int64 rounds towards zero, so for positive: if fractional part > 0, add 1
    // Use: CAST(group_count AS FLOAT64) * ratio, then compare row_num as float64
    // Actually simplest: row_num (1-indexed) <= group_count * ratio means we keep at least 1
    // when ratio > 0. For ceil: row_num <= floor(group_count * ratio) + 1 when fractional
    // But let's just do: CAST(__row_num AS FLOAT64) <= CEIL(__group_count * ratio)
    // Use ScalarFunction directly with the UDF:
    let ceil_udf = datafusion::functions::math::ceil();
    let threshold = datafusion::logical_expr::Expr::ScalarFunction(
        datafusion::logical_expr::expr::ScalarFunction::new_udf(ceil_udf, vec![raw_threshold]),
    );
    let filtered = LogicalPlanBuilder::from(windowed)
        .filter(cast(col("__row_num"), DataType::Float64).lt_eq(threshold))
        .map_err(|e| PromqlError::Plan(format!("limit_ratio filter error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("limit_ratio filter build error: {e}")))?;

    let mut proj_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in all_label_cols {
        proj_exprs.push(col(label.as_str()));
    }
    proj_exprs.push(col("value"));

    let plan = LogicalPlanBuilder::from(filtered)
        .project(proj_exprs)
        .map_err(|e| PromqlError::Plan(format!("limit_ratio project error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("limit_ratio project build error: {e}")))?;

    Ok(plan)
}

/// Plan quantile aggregation using percentile_cont.
fn plan_quantile(
    child_plan: LogicalPlan,
    q: f64,
    grouping_labels: &[String],
) -> Result<LogicalPlan> {
    let mut group_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in grouping_labels {
        group_exprs.push(col(label.as_str()));
    }

    let agg_expr = datafusion::functions_aggregate::percentile_cont::percentile_cont(
        Sort {
            expr: col("value"),
            asc: true,
            nulls_first: false,
        },
        lit(q),
    )
    .alias("value");

    let plan = LogicalPlanBuilder::from(child_plan)
        .aggregate(group_exprs, vec![agg_expr])
        .map_err(|e| PromqlError::Plan(format!("quantile aggregate error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("quantile build error: {e}")))?;

    Ok(plan)
}

/// Plan count_values aggregation.
///
/// count_values("label_name", vector) groups by the value column (cast to string)
/// and counts occurrences, with the value becoming a new label.
fn plan_count_values(
    child_plan: LogicalPlan,
    label_name: &str,
    grouping_labels: &[String],
) -> Result<LogicalPlan> {
    // Project to add a new column: cast value to string as the new label.
    let schema_fields: Vec<String> = child_plan
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    let mut proj_exprs: Vec<datafusion::logical_expr::Expr> = Vec::new();
    for field_name in &schema_fields {
        proj_exprs.push(col(field_name.as_str()));
    }
    proj_exprs.push(cast(col("value"), DataType::Utf8).alias(label_name));

    let projected = LogicalPlanBuilder::from(child_plan)
        .project(proj_exprs)
        .map_err(|e| PromqlError::Plan(format!("count_values project error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("count_values project build error: {e}")))?;

    // Group by timestamp + grouping labels + new label column, count
    let mut group_exprs: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in grouping_labels {
        group_exprs.push(col(label.as_str()));
    }
    group_exprs.push(col(label_name));

    let count_expr = datafusion::functions_aggregate::count::count(col(label_name)).alias("value");

    let aggregated = LogicalPlanBuilder::from(projected)
        .aggregate(group_exprs, vec![count_expr])
        .map_err(|e| PromqlError::Plan(format!("count_values aggregate error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("count_values aggregate build error: {e}")))?;

    // Cast count (Int64) to Float64
    let mut final_proj: Vec<datafusion::logical_expr::Expr> = vec![col("timestamp")];
    for label in grouping_labels {
        final_proj.push(col(label.as_str()));
    }
    final_proj.push(col(label_name));
    final_proj.push(cast(col("value"), DataType::Float64).alias("value"));

    let plan = LogicalPlanBuilder::from(aggregated)
        .project(final_proj)
        .map_err(|e| PromqlError::Plan(format!("count_values cast error: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("count_values build error: {e}")))?;

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
