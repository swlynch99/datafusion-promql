use std::collections::BTreeSet;
use std::sync::Arc;

use crate::datasource::{ColumnMapping, MatchOp, Matcher};
use crate::error::{PromqlError, Result};
use crate::types::Labels;
use arrow::datatypes::DataType;
use datafusion::catalog::TableProvider;
use datafusion::common::Column;
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Union};
use datafusion::prelude::{cast, col, lit};

/// A column that matched the requested metric, with its parsed labels.
#[derive(Debug, Clone)]
struct MatchedColumn {
    /// The original column name in the wide-format table.
    col_name: String,
    /// The labels parsed from the column field metadata.
    labels: Labels,
}

/// Analyze the wide-format schema and find columns matching the given metric.
fn find_matching_columns(
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
/// a UNION ALL of per-column projections.
///
/// Each matched column produces one branch:
/// ```sql
/// SELECT
///     CAST(ts AS BIGINT) AS timestamp,
///     CAST(<col> AS DOUBLE)        AS value,
///     '<metric>'                   AS __name__,
///     '<label_val>'                AS <label_key>, ...
/// FROM wide_table
/// ```
///
/// All branches are combined with UNION ALL. The caller applies time-range
/// filtering and sorting on top.
///
/// Returns `(LogicalPlan, label_column_names)`.
pub(crate) fn normalize_wide_to_long(
    provider: Arc<dyn TableProvider>,
    mapping: &ColumnMapping,
    metric_name: &str,
    matchers: &[Matcher],
) -> Result<(LogicalPlan, Vec<String>)> {
    let (matched, all_label_keys) =
        find_matching_columns(provider.as_ref(), mapping, metric_name, matchers)?;

    let all_label_keys: Vec<String> = all_label_keys.into_iter().collect();

    // Build one SELECT branch per matched column.
    // Each branch gets a unique scan alias so the DataFusion optimizer does not
    // collapse the identical table scans into a single one via CSE.
    // Check if the timestamp column already has the target type.
    let schema = provider.schema();
    let ts_field = schema
        .field_with_name(mapping.timestamp_column.as_str())
        .map_err(|e| PromqlError::Plan(format!("timestamp column not found: {e}")))?;
    let ts_expr = if ts_field.data_type() == &DataType::UInt64 {
        col(mapping.timestamp_column.as_str()).alias("timestamp")
    } else {
        cast(col(mapping.timestamp_column.as_str()), DataType::UInt64).alias("timestamp")
    };

    let mut branch_plans: Vec<LogicalPlan> = Vec::with_capacity(matched.len());
    for (idx, mc) in matched.iter().enumerate() {
        let mut exprs = vec![
            // Timestamp: cast to UInt64 (nanoseconds) if needed.
            ts_expr.clone(),
            // Value: cast to Float64.
            // Use Column::new_unqualified to bypass the SQL parser, which would
            // strip anything after `/` from the column name.
            cast(
                Expr::Column(Column::new_unqualified(mc.col_name.as_str())),
                DataType::Float64,
            )
            .alias("value"),
            // Metric name literal.
            lit(metric_name).alias("__name__"),
        ];
        // One string-literal column per label key.
        for key in &all_label_keys {
            let val = mc.labels.get(key).map(|s| s.as_str()).unwrap_or("");
            exprs.push(lit(val).alias(key.as_str()));
        }

        // Use a unique alias per branch so the optimizer treats each scan as
        // distinct (avoids CSE merging identical-looking table scans).
        let scan_alias = format!("{metric_name}_{idx}");
        let plan = LogicalPlanBuilder::scan_with_filters(
            scan_alias,
            provider_as_source(Arc::clone(&provider)),
            None,
            vec![],
        )
        .map_err(|e| PromqlError::Plan(format!("failed to build scan: {e}")))?
        .project(exprs)
        .map_err(|e| PromqlError::Plan(format!("failed to build projection: {e}")))?
        .build()
        .map_err(|e| PromqlError::Plan(format!("failed to build plan: {e}")))?;

        branch_plans.push(plan);
    }

    // UNION ALL all branches into a single flat Union node.
    let union_plan = if branch_plans.len() == 1 {
        branch_plans.into_iter().next().unwrap()
    } else {
        let inputs: Vec<Arc<LogicalPlan>> = branch_plans.into_iter().map(Arc::new).collect();
        LogicalPlan::Union(
            Union::try_new_with_loose_types(inputs)
                .map_err(|e| PromqlError::Plan(format!("failed to build union: {e}")))?,
        )
    };

    let mut label_columns = vec!["__name__".to_string()];
    label_columns.extend(all_label_keys);

    Ok((union_plan, label_columns))
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
            MatchOp::RegexMatch | MatchOp::RegexNotMatch => {}
        }
    }
    true
}
