use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::{Column, ScalarValue};
use datafusion::error::Result;
use datafusion::logical_expr::expr::Alias;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, LogicalPlanBuilder, when};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::{col, lit};

use crate::node::{WideColumnMeta, WideUnpack};
use crate::types::Labels;

/// Optimizer rule that rewrites `Aggregate(sum)` directly on top of a
/// [`WideUnpack`] node into a column-wise sum projection followed by a
/// `WideUnpack` over the resulting (much narrower) wide table.
///
/// ```text
/// Aggregate(group=[timestamp, g_1, …, g_k], sum(value))         WideUnpack(columns=[s_1,…,s_M], label_keys=[g_1,…,g_k])
///   WideUnpack(columns=[c_1, …, c_N], label_keys=[…])      →      Projection(timestamp, sum(group_1) AS s_1, …, sum(group_M) AS s_M)
///     WideInput                                                       WideInput
/// ```
///
/// Each output column `s_i` corresponds to a unique combination of grouping
/// label values across the original wide columns. Wide columns whose grouping
/// labels match are summed column-wise. The N original columns collapse into
/// exactly M ≤ N columns (one per unique group) before `WideUnpack` expands
/// them into long format.
///
/// The transformation is valid because:
///
///   * Each wide value column maps 1:1 to a long-format series carrying the
///     constant labels recorded in its [`WideColumnMeta`].
///   * The aggregate groups by `timestamp` plus a subset of those constant
///     labels, so two wide columns that agree on the grouping labels always
///     fall into the same aggregation group at every timestamp.
///   * Summing the wide columns row-wise (one row per timestamp) and then
///     unpacking yields the same `(timestamp, group-labels, value)` tuples
///     as unpacking first and then aggregating.
///
/// The two main wins are:
///
///   * The expensive long-format expansion runs on M ≤ N (often ≪) columns
///     instead of the full N.
///   * The sum becomes a per-row arithmetic projection rather than a hash
///     aggregate.
///
/// Only `sum` is currently handled. `count`, `min`, `max`, etc. would require
/// different per-row expressions and are left for follow-up rules.
#[derive(Debug)]
pub struct PushSumThroughWideUnpack;

impl OptimizerRule for PushSumThroughWideUnpack {
    fn name(&self) -> &str {
        "push_sum_through_wide_unpack"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Match: Aggregate whose input is Extension(WideUnpack).
        let LogicalPlan::Aggregate(ref agg) = plan else {
            return Ok(Transformed::no(plan));
        };

        let LogicalPlan::Extension(ref ext) = *agg.input else {
            return Ok(Transformed::no(plan));
        };

        let Some(unpack) = ext.node.as_any().downcast_ref::<WideUnpack>() else {
            return Ok(Transformed::no(plan));
        };

        // Exactly one aggregate expression — sum(col("value")) optionally aliased.
        if agg.aggr_expr.len() != 1 {
            return Ok(Transformed::no(plan));
        }
        let Some(agg_output_name) = sum_value_alias(&agg.aggr_expr[0]) else {
            return Ok(Transformed::no(plan));
        };

        // Each group expression must be a plain column reference. The grouping
        // must include `timestamp`; the remaining columns must all be labels
        // produced by the WideUnpack (`__name__` or one of `label_keys`).
        let mut group_col_names: Vec<String> = Vec::with_capacity(agg.group_expr.len());
        for ge in &agg.group_expr {
            let Some(name) = simple_column_name(ge) else {
                return Ok(Transformed::no(plan));
            };
            group_col_names.push(name);
        }
        if !group_col_names.iter().any(|n| n == "timestamp") {
            return Ok(Transformed::no(plan));
        }
        let grouping_labels: Vec<String> = group_col_names
            .iter()
            .filter(|n| n.as_str() != "timestamp")
            .cloned()
            .collect();
        for g in &grouping_labels {
            if g != "__name__" && !unpack.label_keys.iter().any(|k| k == g) {
                return Ok(Transformed::no(plan));
            }
        }

        // Bucket the wide columns by the values of the grouping labels.
        // Use a BTreeMap so the iteration order is deterministic, which keeps
        // the rewritten projection stable under repeated runs.
        let mut groups: BTreeMap<Vec<String>, Vec<usize>> = BTreeMap::new();
        for (i, c) in unpack.columns.iter().enumerate() {
            let key: Vec<String> = grouping_labels
                .iter()
                .map(|g| {
                    if g == "__name__" {
                        c.metric_name.clone()
                    } else {
                        c.labels.get(g).cloned().unwrap_or_default()
                    }
                })
                .collect();
            groups.entry(key).or_default().push(i);
        }

        // Build the column-wise sum projection. Each group contributes one
        // synthetic value column whose name is unique within the projection.
        // The new `WideColumnMeta` carries only the grouping labels so the
        // downstream `WideUnpack` emits the right schema.
        let mut new_columns: Vec<WideColumnMeta> = Vec::with_capacity(groups.len());
        let mut proj_exprs: Vec<Expr> = Vec::with_capacity(groups.len() + 1);
        proj_exprs.push(col("timestamp"));

        let metric_name_grouped = grouping_labels.iter().any(|g| g == "__name__");
        let default_metric_name = unpack
            .columns
            .first()
            .map(|c| c.metric_name.clone())
            .unwrap_or_default();

        for (group_idx, (group_key, col_indices)) in groups.into_iter().enumerate() {
            let new_col_name = format!("__sum_{group_idx}");
            let cols_to_sum: Vec<&str> = col_indices
                .iter()
                .map(|i| unpack.columns[*i].col_name.as_str())
                .collect();

            let sum_expr = build_nullable_sum(&cols_to_sum)?;
            proj_exprs.push(sum_expr.alias(new_col_name.clone()));

            let mut new_labels = Labels::new();
            let mut metric_name = default_metric_name.clone();
            for (g_name, g_value) in grouping_labels.iter().zip(group_key.iter()) {
                if g_name == "__name__" {
                    metric_name = g_value.clone();
                } else if !g_value.is_empty() {
                    new_labels.insert(g_name.clone(), g_value.clone());
                }
            }
            // If `__name__` is not part of the grouping, the `WideUnpack`'s
            // `__name__` column is dropped by the final projection below, so
            // the value here is irrelevant — but it must still be a valid
            // (non-empty) string because the schema marks `__name__` as
            // non-nullable.
            if !metric_name_grouped {
                metric_name = default_metric_name.clone();
            }
            new_columns.push(WideColumnMeta {
                col_name: new_col_name,
                metric_name,
                labels: new_labels,
            });
        }

        let summed_input = LogicalPlanBuilder::from(unpack.input.clone())
            .project(proj_exprs)?
            .build()?;

        // The `WideUnpack`'s `label_keys` always exclude `__name__` (it has
        // its own dedicated column). When the aggregate groups by
        // `__name__`, the value still flows through `WideColumnMeta::metric_name`
        // and surfaces in the unpack output's `__name__` column.
        let new_label_keys: Vec<String> = grouping_labels
            .iter()
            .filter(|g| g.as_str() != "__name__")
            .cloned()
            .collect();

        let new_unpack = WideUnpack::new(
            summed_input,
            Arc::new(new_columns),
            Arc::new(new_label_keys),
        )
        .map_err(|e| datafusion::error::DataFusionError::Plan(e.to_string()))?;

        let new_unpack_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(new_unpack),
        });

        // The Aggregate's output schema is `[timestamp, g_1, …, g_k, value]`.
        // The new `WideUnpack`'s output is `[timestamp, value, __name__,
        // <new_label_keys>…]`. Project to drop `__name__` (when not grouped
        // by it) and reorder to match the original schema, preserving
        // qualifiers so the optimizer's schema-equivalence check passes.
        let original_schema = agg.schema.clone();
        let mut final_proj_exprs: Vec<Expr> = Vec::with_capacity(original_schema.fields().len());
        for (qualifier, field) in original_schema.iter() {
            let name = field.name();
            let expr = if name == agg_output_name.as_str() {
                col("value")
            } else {
                col(name.as_str())
            };
            final_proj_exprs.push(Expr::Alias(Alias::new(
                expr,
                qualifier.cloned(),
                name.as_str(),
            )));
        }

        let final_plan = LogicalPlanBuilder::from(new_unpack_plan)
            .project(final_proj_exprs)?
            .build()?;

        Ok(Transformed::yes(final_plan))
    }
}

/// If `expr` is `sum(col("value"))` (optionally aliased), return the output
/// column name (the alias, or `"sum(value)"` if unaliased). Otherwise `None`.
fn sum_value_alias(expr: &Expr) -> Option<String> {
    let (alias_name, inner) = match expr {
        Expr::Alias(a) => (Some(a.name.clone()), a.expr.as_ref()),
        other => (None, other),
    };
    let Expr::AggregateFunction(af) = inner else {
        return None;
    };
    if af.func.name() != "sum" {
        return None;
    }
    if af.params.distinct
        || af.params.filter.is_some()
        || !af.params.order_by.is_empty()
        || af.params.null_treatment.is_some()
    {
        return None;
    }
    if af.params.args.len() != 1 {
        return None;
    }
    let arg_name = simple_column_name(&af.params.args[0])?;
    if arg_name != "value" {
        return None;
    }
    Some(alias_name.unwrap_or_else(|| "sum(value)".to_string()))
}

/// If `expr` is a bare column reference, return its (unqualified) name.
fn simple_column_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(Column { name, .. }) => Some(name.clone()),
        _ => None,
    }
}

/// Build an expression that sums the given columns with SQL-style NULL
/// semantics: NULL inputs are ignored, but if every input is NULL the
/// result is NULL (so the row stays absent from the long-format output
/// after the downstream `IS NOT NULL` filter).
fn build_nullable_sum(cols: &[&str]) -> Result<Expr> {
    debug_assert!(!cols.is_empty(), "build_nullable_sum requires ≥1 column");
    if cols.len() == 1 {
        // A single column trivially preserves NULL/non-NULL semantics.
        return Ok(col(cols[0]));
    }

    // all-null condition
    let null_cond = cols
        .iter()
        .map(|c| col(*c).is_null())
        .reduce(|a, b| a.and(b))
        .expect("non-empty cols");

    // sum of coalesce(c_i, 0.0) — equivalent to summing only the non-NULL values
    let sum_expr = cols
        .iter()
        .map(|c| datafusion::functions::expr_fn::coalesce(vec![col(*c), lit(0.0_f64)]))
        .reduce(|a, b| a + b)
        .expect("non-empty cols");

    when(null_cond, lit(ScalarValue::Float64(None))).otherwise(sum_expr)
}
