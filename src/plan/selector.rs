use arrow::datatypes::DataType;
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::{Like, LogicalPlan, LogicalPlanBuilder};
use datafusion::prelude::*;
use promql_parser::label::{METRIC_NAME, MatchOp, Matcher};
use promql_parser::parser::VectorSelector;

use crate::datasource::{self, MetricSource, TableFormat};
use crate::error::{PromqlError, Result};
use crate::types::TimeRange;

/// Create a timestamp literal that matches the timestamp column's data type.
///
/// The `timestamp` column may be either `UInt64` or `Int64` depending on the
/// data source. This helper inspects the schema and produces a literal of the
/// matching type so that DataFusion filter comparisons don't fail with a type
/// mismatch.
fn timestamp_lit(schema: &arrow::datatypes::Schema, value: i64) -> datafusion::prelude::Expr {
    let ts_type = schema
        .column_with_name("timestamp")
        .map(|(_, f)| f.data_type().clone());
    match ts_type {
        Some(DataType::UInt64) => lit(value.max(0) as u64),
        _ => lit(value),
    }
}

/// Convert a promql-parser `Matcher` to a DataFusion filter expression.
fn matcher_to_filter_expr(m: &Matcher) -> datafusion::logical_expr::Expr {
    let col_expr = col(&m.name);
    match &m.op {
        MatchOp::Equal => col_expr.eq(lit(&m.value)),
        MatchOp::NotEqual => col_expr.not_eq(lit(&m.value)),
        MatchOp::Re(re) => datafusion::prelude::Expr::SimilarTo(Like {
            negated: false,
            expr: Box::new(col_expr),
            pattern: Box::new(lit(re.as_str())),
            escape_char: None,
            case_insensitive: false,
        }),
        MatchOp::NotRe(re) => datafusion::prelude::Expr::SimilarTo(Like {
            negated: true,
            expr: Box::new(col_expr),
            pattern: Box::new(lit(re.as_str())),
            escape_char: None,
            case_insensitive: false,
        }),
    }
}

/// Convert promql-parser matchers to our datasource matchers for pushdown.
fn convert_matchers(matchers: &[Matcher]) -> Vec<datasource::Matcher> {
    matchers
        .iter()
        .map(|m| datasource::Matcher {
            name: m.name.clone(),
            op: match &m.op {
                MatchOp::Equal => datasource::MatchOp::Equal,
                MatchOp::NotEqual => datasource::MatchOp::NotEqual,
                MatchOp::Re(_) => datasource::MatchOp::RegexMatch,
                MatchOp::NotRe(_) => datasource::MatchOp::RegexNotMatch,
            },
            value: m.value.clone(),
        })
        .collect()
}

/// Plan a VectorSelector: table scan + label filters + time range filter.
///
/// Returns the logical plan and the list of label column names (for grouping
/// in the InstantVectorEval node).
///
/// `extra_range_ns` extends the time range expansion beyond the default lookback.
/// This is used for range vectors where the window may be larger than the
/// default 5-minute lookback.
///
/// `offset_ns` shifts the data fetch window. Positive values shift the window
/// into the past (need older data), negative values shift toward the future.
pub(crate) async fn plan_vector_selector(
    vs: &VectorSelector,
    source: &dyn MetricSource,
    time_range: TimeRange,
    extra_range_ns: i64,
    offset_ns: i64,
) -> Result<(LogicalPlan, Vec<String>)> {
    let metric_name = vs
        .name
        .as_deref()
        .or_else(|| {
            vs.matchers.matchers.iter().find_map(|m| {
                if m.name == METRIC_NAME && matches!(m.op, MatchOp::Equal) {
                    Some(m.value.as_str())
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| PromqlError::Plan("vector selector must have a metric name".into()))?;

    // Convert matchers for pushdown (exclude __name__ matcher).
    let non_name_matchers: Vec<_> = vs
        .matchers
        .matchers
        .iter()
        .filter(|m| m.name != METRIC_NAME)
        .cloned()
        .collect();
    let ds_matchers = convert_matchers(&non_name_matchers);

    // Expand the time range to include the lookback window, any extra range
    // duration for range vectors, and the offset so downstream nodes have
    // enough data.
    //
    // A positive offset means we look further into the past, so we expand
    // start_ns. A negative offset means we look toward the future, so we
    // expand end_ns.
    let offset_expand_start = offset_ns.max(0);
    let offset_expand_end = (-offset_ns).max(0);
    let expanded_range = TimeRange {
        start_ns: time_range.start_ns.map(|s| {
            s.saturating_sub(crate::types::DEFAULT_LOOKBACK_NS)
                .saturating_sub(extra_range_ns)
                .saturating_sub(offset_expand_start)
        }),
        end_ns: time_range.end_ns.map(|e| e.saturating_add(offset_expand_end)),
    };

    let (provider, format) = source
        .table_for_metric(metric_name, &ds_matchers, expanded_range)
        .await?;

    let provider_schema = provider.schema();

    match format {
        TableFormat::Wide(mapping) => {
            // Normalize wide format to long format via UNION ALL projections.
            // After normalization, `timestamp` is always Int64 nanoseconds.
            let (plan, label_columns) = crate::normalize::normalize_wide_to_long(
                provider,
                &mapping,
                metric_name,
                &ds_matchers,
            )?;

            // Apply time range filter (if bounded) and sort on the normalized output.
            // Use Int64 literals directly since the normalized timestamp is always Int64 ns.
            let mut builder = LogicalPlanBuilder::from(plan);
            if let Some(start) = expanded_range.start_ns {
                builder = builder
                    .filter(col("timestamp").gt_eq(lit(start)))
                    .map_err(|e| PromqlError::Plan(format!("failed to apply time filter: {e}")))?;
            }
            if let Some(end) = expanded_range.end_ns {
                builder = builder
                    .filter(col("timestamp").lt_eq(lit(end)))
                    .map_err(|e| PromqlError::Plan(format!("failed to apply time filter: {e}")))?;
            }
            let plan = builder
                .sort(vec![col("timestamp").sort(true, false)])
                .map_err(|e| PromqlError::Plan(format!("failed to add sort: {e}")))?
                .build()
                .map_err(|e| PromqlError::Plan(format!("failed to build plan: {e}")))?;

            return Ok((plan, label_columns));
        }
        TableFormat::Long => {}
    }

    // Determine label columns from the provider schema.
    // In long format, label columns are everything except "timestamp", "value".
    let label_columns: Vec<String> = provider_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .filter(|n| n != "timestamp" && n != "value")
        .collect();

    // Build a table scan using provider_as_source to convert TableProvider -> TableSource.
    let table_source = provider_as_source(provider);
    let plan = LogicalPlanBuilder::scan_with_filters(metric_name, table_source, None, vec![])
        .map_err(|e| PromqlError::Plan(format!("failed to build table scan: {e}")))?;

    // Apply label matchers as filters.
    let plan = non_name_matchers
        .iter()
        .try_fold(plan, |plan, matcher| {
            plan.filter(matcher_to_filter_expr(matcher))
        })
        .map_err(|e| PromqlError::Plan(format!("failed to apply filter: {e}")))?;

    // Apply time range filter on the expanded range (skip if unbounded).
    let plan = if let Some(start) = expanded_range.start_ns {
        plan.filter(col("timestamp").gt_eq(timestamp_lit(&provider_schema, start)))
            .map_err(|e| PromqlError::Plan(format!("failed to apply time filter: {e}")))?
    } else {
        plan
    };
    let plan = if let Some(end) = expanded_range.end_ns {
        plan.filter(col("timestamp").lt_eq(timestamp_lit(&provider_schema, end)))
            .map_err(|e| PromqlError::Plan(format!("failed to apply time filter: {e}")))?
    } else {
        plan
    };

    // Sort by timestamp for the InstantVectorEval node.
    let plan = plan
        .sort(vec![col("timestamp").sort(true, false)])
        .map_err(|e| PromqlError::Plan(format!("failed to add sort: {e}")))?;

    let plan = plan
        .build()
        .map_err(|e| PromqlError::Plan(format!("failed to build plan: {e}")))?;

    Ok((plan, label_columns))
}
