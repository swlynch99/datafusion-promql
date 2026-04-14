use std::collections::BTreeSet;
use std::sync::Arc;

use crate::datasource::{ColumnMapping, MatchOp, Matcher};
use crate::error::{PromqlError, Result};
use crate::node::{WideColumnMeta, WideUnpack};
use crate::types::{Labels, TimeRange};
use arrow::datatypes::DataType;
use datafusion::catalog::TableProvider;
use datafusion::common::Column;
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, LogicalPlanBuilder};
use datafusion::prelude::{cast, col, lit};
use regex::Regex;

/// A column that matched the requested metric, with its parsed labels.
#[derive(Debug, Clone)]
pub(crate) struct MatchedColumn {
    /// The original column name in the wide-format table.
    pub col_name: String,
    /// The labels parsed from the column field metadata.
    pub labels: Labels,
}

/// Analyze the wide-format schema and find columns matching the given metric.
pub(crate) fn find_matching_columns(
    provider: &dyn TableProvider,
    mapping: &ColumnMapping,
    metric_name: &str,
    matchers: &[Matcher],
) -> Result<(Vec<MatchedColumn>, BTreeSet<String>)> {
    let schema = provider.schema();
    let ignore: BTreeSet<&str> = mapping.ignore_columns.iter().map(|s| s.as_str()).collect();

    let mut matched: Vec<MatchedColumn> = Vec::new();

    for field in schema.fields() {
        let col_name = field.name().as_str();

        if col_name == mapping.timestamp_column || ignore.contains(col_name) {
            continue;
        }

        // Skip non-numeric columns.
        match field.data_type() {
            DataType::UInt64 | DataType::Int64 | DataType::Float64 => {}
            _ => continue,
        }

        let (parsed_metric, labels) = match (mapping.parse_column)(field.as_ref()) {
            Some(pair) => pair,
            None => continue,
        };

        if parsed_metric != metric_name {
            continue;
        }

        if !labels_match_matchers(&labels, matchers) {
            continue;
        }

        matched.push(MatchedColumn {
            col_name: col_name.to_string(),
            labels,
        });
    }

    if matched.is_empty() {
        return Err(PromqlError::DataSource(format!(
            "metric '{metric_name}' not found in wide-format table"
        )));
    }

    let all_label_keys: BTreeSet<String> = matched
        .iter()
        .flat_map(|m| m.labels.keys().cloned())
        .collect();

    Ok((matched, all_label_keys))
}

/// Convert a wide-format `TableProvider` into a long-format logical plan via
/// a single table scan followed by a [`WideUnpack`] node.
///
/// Unlike [`normalize_wide_to_long`], this function reads ALL matched value
/// columns in a single scan (timestamp + N value columns) and relies on the
/// `WideUnpack` execution node to expand each input row into N output rows
/// (one per value column).
///
/// This eliminates the N independent file scans that the UNION ALL approach
/// requires. The output schema and semantics are identical.
///
/// Returns `(LogicalPlan, label_column_names)`.
pub(crate) fn plan_wide_single_scan(
    provider: Arc<dyn TableProvider>,
    mapping: &ColumnMapping,
    metric_name: &str,
    matchers: &[Matcher],
    time_range: &TimeRange,
) -> Result<(LogicalPlan, Vec<String>)> {
    let (matched, all_label_keys) =
        find_matching_columns(provider.as_ref(), mapping, metric_name, matchers)?;

    let all_label_keys: Vec<String> = all_label_keys.into_iter().collect();

    let schema = provider.schema();
    let ts_field = schema
        .field_with_name(mapping.timestamp_column.as_str())
        .map_err(|e| PromqlError::Plan(format!("timestamp column not found: {e}")))?;

    // Pre-compute the timestamp column index.
    let ts_col_idx = schema
        .index_of(mapping.timestamp_column.as_str())
        .map_err(|e| PromqlError::Plan(format!("timestamp column not found: {e}")))?;

    // Build column projection: timestamp + all matched value columns.
    let mut projection_indices = vec![ts_col_idx];
    for mc in &matched {
        let idx = schema
            .index_of(mc.col_name.as_str())
            .map_err(|e| PromqlError::Plan(format!("column '{}' not found: {e}", mc.col_name)))?;
        projection_indices.push(idx);
    }

    // Build scan-level time-range filters for parquet row-group pruning.
    let scan_time_filters: Vec<Expr> = {
        let ts_col = col(mapping.timestamp_column.as_str());
        let is_int64 = ts_field.data_type() == &DataType::Int64;
        let mut filters = Vec::new();
        if let Some(start) = time_range.start_ns {
            let bound = if is_int64 {
                lit(start as i64)
            } else {
                lit(start)
            };
            filters.push(ts_col.clone().gt_eq(bound));
        }
        if let Some(end) = time_range.end_ns {
            let bound = if is_int64 { lit(end as i64) } else { lit(end) };
            filters.push(ts_col.lt_eq(bound));
        }
        filters
    };

    // Build a single scan reading timestamp + all matched value columns.
    let scan_plan = LogicalPlanBuilder::scan_with_filters(
        metric_name,
        provider_as_source(provider),
        Some(projection_indices),
        scan_time_filters,
    )
    .map_err(|e| PromqlError::Plan(format!("failed to build scan: {e}")))?;

    // Project: cast timestamp to UInt64 if needed, cast value columns to Float64.
    let ts_expr = if ts_field.data_type() == &DataType::UInt64 {
        col(mapping.timestamp_column.as_str()).alias("timestamp")
    } else {
        cast(col(mapping.timestamp_column.as_str()), DataType::UInt64).alias("timestamp")
    };

    let mut proj_exprs = vec![ts_expr];
    for mc in &matched {
        // Cast each value column to Float64, keeping its original name so
        // WideUnpackExec can look it up by name.
        proj_exprs.push(
            cast(
                Expr::Column(Column::new_unqualified(mc.col_name.as_str())),
                DataType::Float64,
            )
            .alias(mc.col_name.as_str()),
        );
    }

    let plan = scan_plan
        .project(proj_exprs)
        .map_err(|e| PromqlError::Plan(format!("failed to build projection: {e}")))?
        // Sort by timestamp so the WideUnpack output (which processes one
        // column at a time) is sorted by (label_columns, timestamp).
        .sort(vec![col("timestamp").sort(true, false)])
        .map_err(|e| PromqlError::Plan(format!("failed to add sort: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("failed to build plan: {e}")))?;

    // Build WideColumnMeta for each matched column.
    let column_metas: Vec<WideColumnMeta> = matched
        .iter()
        .map(|mc| WideColumnMeta {
            col_name: mc.col_name.clone(),
            metric_name: metric_name.to_string(),
            labels: mc.labels.clone(),
        })
        .collect();

    let unpack = WideUnpack::new(plan, column_metas, all_label_keys.clone())?;
    let unpack_plan = LogicalPlan::Extension(Extension {
        node: Arc::new(unpack),
    });

    let mut label_columns = vec!["__name__".to_string()];
    label_columns.extend(all_label_keys);

    Ok((unpack_plan, label_columns))
}

/// Check whether parsed labels satisfy all matchers.
fn labels_match_matchers(labels: &Labels, matchers: &[Matcher]) -> bool {
    for m in matchers {
        let label_val = labels.get(&m.name).map(|s| s.as_str()).unwrap_or("");
        match m.op {
            MatchOp::Equal => {
                if label_val != m.value {
                    return false;
                }
            }
            MatchOp::NotEqual => {
                if label_val == m.value {
                    return false;
                }
            }
            MatchOp::RegexMatch => {
                let anchored = format!("^(?:{})$", m.value);
                if let Ok(re) = Regex::new(&anchored)
                    && !re.is_match(label_val)
                {
                    return false;
                }
            }
            MatchOp::RegexNotMatch => {
                let anchored = format!("^(?:{})$", m.value);
                if let Ok(re) = Regex::new(&anchored)
                    && re.is_match(label_val)
                {
                    return false;
                }
            }
        }
    }
    true
}
