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
/// `extra_range_ms` extends the time range expansion beyond the default lookback.
/// This is used for range vectors where the window may be larger than the
/// default 5-minute lookback.
pub(crate) async fn plan_vector_selector(
    vs: &VectorSelector,
    source: &dyn MetricSource,
    time_range: TimeRange,
    extra_range_ms: i64,
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

    // Expand the time range to include the lookback window (and any extra
    // range duration for range vectors) so downstream nodes have enough data.
    let expanded_range = TimeRange {
        start_ms: time_range.start_ms - crate::types::DEFAULT_LOOKBACK_MS - extra_range_ms,
        end_ms: time_range.end_ms,
    };

    let (provider, format) = source
        .table_for_metric(metric_name, &ds_matchers, expanded_range)
        .await?;

    let provider_schema = provider.schema();

    match format {
        TableFormat::Wide(mapping) => {
            // Normalize wide format to long format via UNION ALL projections.
            let (plan, label_columns) = crate::normalize::normalize_wide_to_long(
                provider,
                &mapping,
                metric_name,
                &ds_matchers,
            )?;

            // Apply time range filter and sort on the normalized output.
            let plan = LogicalPlanBuilder::from(plan)
                .filter(
                    col("timestamp")
                        .gt_eq(timestamp_lit(&provider_schema, expanded_range.start_ms))
                        .and(col("timestamp").lt_eq(timestamp_lit(&provider_schema, expanded_range.end_ms))),
                )
                .map_err(|e| PromqlError::Plan(format!("failed to apply time filter: {e}")))?
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

    // Apply time range filter on the expanded range.
    let plan = plan
        .filter(
            col("timestamp")
                .gt_eq(timestamp_lit(&provider_schema, expanded_range.start_ms))
                .and(col("timestamp").lt_eq(timestamp_lit(&provider_schema, expanded_range.end_ms))),
        )
        .map_err(|e| PromqlError::Plan(format!("failed to apply time filter: {e}")))?;

    // Sort by timestamp for the InstantVectorEval node.
    let plan = plan
        .sort(vec![col("timestamp").sort(true, false)])
        .map_err(|e| PromqlError::Plan(format!("failed to add sort: {e}")))?;

    let plan = plan
        .build()
        .map_err(|e| PromqlError::Plan(format!("failed to build plan: {e}")))?;

    Ok((plan, label_columns))
}
