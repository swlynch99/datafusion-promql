use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use datafusion::common::ScalarValue;
use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, LogicalPlanBuilder, when};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::{col, lit};

use crate::node::{
    BinaryEval, BinaryOp, MatchCardinality, VectorMatching, WideColumnMeta, WideUnpack,
};
use crate::opt::logical::subplan_fingerprint;
use crate::types::Labels;

/// Optimizer rule that pushes a [`BinaryEval`] (vector op vector) down through
/// two sibling [`WideUnpack`] nodes by rewriting the operation as a column-wise
/// projection on the shared wide-format input, wrapped by a single
/// [`WideUnpack`].
///
/// ```text
/// BinaryEval(op, matching)                          WideUnpack(new_columns, output_label_keys)
///   WideUnpack(L_columns, label_keys)        →        Projection(timestamp, op(l_i, r_j) AS __bin_k …)
///     W                                                 W
///   WideUnpack(R_columns, label_keys)
///     W
/// ```
///
/// Only fires when both [`WideUnpack`] inputs are structurally identical (same
/// underlying wide subplan) and share the same `label_keys` and `include_name`.
/// `DeduplicateSubplans` runs earlier, so identical subtrees are typically
/// already represented by the same `Arc<LogicalPlan>`; we additionally compare
/// fingerprints to handle the post-clone case.
///
/// The rewrite is valid because:
///
///   * Each value column of either wide input maps 1:1 to a long-format series
///     in the corresponding `WideUnpack`'s output, with constant labels recorded
///     in its [`WideColumnMeta`].
///   * For a given vector matching (`on(...)` / `ignoring(...)` / default), the
///     matching key of every wide column is determined entirely by those
///     constants, so columns can be bucketed into match-key groups at plan
///     time.
///   * Within a bucket, applying `op` column-wise to a `(lhs_col, rhs_col)`
///     pair, then unpacking, produces the same `(timestamp, labels, value)`
///     tuples as unpacking first and applying [`BinaryExec`](crate::exec).
///   * Comparison ops without `bool` map to `CASE WHEN cmp THEN lhs ELSE NULL`
///     so the downstream `value IS NOT NULL` filter drops false rows. With
///     `bool`, they map to `CASE WHEN cmp THEN 1.0 ELSE 0.0`. Arithmetic and
///     `^` map to direct expressions / `power(...)`.
///   * Set operators (`and` / `or` / `unless`) have row-level semantics that do
///     not commute with the unpack and are skipped.
#[derive(Debug)]
pub struct PushBinaryThroughWideUnpack;

impl OptimizerRule for PushBinaryThroughWideUnpack {
    fn name(&self) -> &str {
        "push_binary_through_wide_unpack"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // ─── Borrowed-phase: validate the pattern and build the rewrite plan
        // (projection exprs + new column metas) without consuming `plan`.

        let LogicalPlan::Extension(ref ext) = plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(eval) = ext.node.as_any().downcast_ref::<BinaryEval>() else {
            return Ok(Transformed::no(plan));
        };

        // Set operators are out of scope (row-level semantics don't commute
        // with the unpack).
        if eval.op.is_set_operator() {
            return Ok(Transformed::no(plan));
        }

        let LogicalPlan::Extension(ref lhs_ext) = eval.lhs else {
            return Ok(Transformed::no(plan));
        };
        let LogicalPlan::Extension(ref rhs_ext) = eval.rhs else {
            return Ok(Transformed::no(plan));
        };
        let Some(lhs_unpack) = lhs_ext.node.as_any().downcast_ref::<WideUnpack>() else {
            return Ok(Transformed::no(plan));
        };
        let Some(rhs_unpack) = rhs_ext.node.as_any().downcast_ref::<WideUnpack>() else {
            return Ok(Transformed::no(plan));
        };

        // Both unpacks must wrap the same wide subplan and produce the same
        // long-format schema. Anything else is "case 2/3" territory.
        if lhs_unpack.label_keys != rhs_unpack.label_keys
            || lhs_unpack.include_name != rhs_unpack.include_name
            || subplan_fingerprint(&lhs_unpack.input) != subplan_fingerprint(&rhs_unpack.input)
        {
            return Ok(Transformed::no(plan));
        }

        let label_keys: &[String] = lhs_unpack.label_keys.as_slice();

        // Bucket each side's columns by matching key.
        let lhs_buckets = bucket_by_match_key(&lhs_unpack.columns, label_keys, &eval.matching);
        let rhs_buckets = bucket_by_match_key(&rhs_unpack.columns, label_keys, &eval.matching);

        // Determine the output WideUnpack's label_keys / include_name. This
        // mirrors `compute_binary_output_schema` so the rewritten plan's schema
        // is identical to the original `BinaryEval` schema.
        let (output_label_keys, output_include_name) =
            compute_output_label_keys(label_keys, lhs_unpack.include_name, eval.op, &eval.matching);
        let output_label_set: HashSet<&str> =
            output_label_keys.iter().map(|s| s.as_str()).collect();

        // Iterate buckets in deterministic order (BTreeMap) and assemble pairs.
        let mut new_columns: Vec<WideColumnMeta> = Vec::new();
        let mut proj_exprs: Vec<Expr> = vec![col("timestamp")];

        for (key, lhs_cols) in &lhs_buckets {
            let Some(rhs_cols) = rhs_buckets.get(key) else {
                continue;
            };

            let pairs: Vec<(&WideColumnMeta, &WideColumnMeta, MetaSide)> = match &eval.matching.card
            {
                MatchCardinality::OneToOne => {
                    if lhs_cols.len() != 1 || rhs_cols.len() != 1 {
                        // Ambiguous matching at plan time. Bail to preserve
                        // BinaryExec's runtime error semantics.
                        return Ok(Transformed::no(plan));
                    }
                    vec![(lhs_cols[0], rhs_cols[0], MetaSide::Lhs)]
                }
                MatchCardinality::ManyToOne(_include) => {
                    if rhs_cols.len() != 1 {
                        return Ok(Transformed::no(plan));
                    }
                    let rhs_col = rhs_cols[0];
                    lhs_cols
                        .iter()
                        .map(|lhs_col| (*lhs_col, rhs_col, MetaSide::Lhs))
                        .collect()
                }
                MatchCardinality::OneToMany(_include) => {
                    if lhs_cols.len() != 1 {
                        return Ok(Transformed::no(plan));
                    }
                    let lhs_col = lhs_cols[0];
                    rhs_cols
                        .iter()
                        .map(|rhs_col| (lhs_col, *rhs_col, MetaSide::Rhs))
                        .collect()
                }
            };

            for (lhs_col, rhs_col, meta_side) in pairs {
                let new_col_name = format!("__bin_{}", new_columns.len());
                let expr = build_binary_op_expr(
                    col(lhs_col.col_name.as_str()),
                    col(rhs_col.col_name.as_str()),
                    eval.op,
                    eval.return_bool,
                )?;
                proj_exprs.push(expr.alias(new_col_name.as_str()));

                let source_meta = match meta_side {
                    MetaSide::Lhs => lhs_col,
                    MetaSide::Rhs => rhs_col,
                };
                new_columns.push(make_output_meta(
                    source_meta,
                    &output_label_set,
                    output_include_name,
                    new_col_name,
                ));
            }
        }

        if new_columns.is_empty() {
            return Ok(Transformed::no(plan));
        }

        // ─── Owned-phase: take ownership of the inner unpack's wide input.
        let LogicalPlan::Extension(ext) = plan else {
            unreachable!();
        };
        let eval = ext
            .node
            .as_any()
            .downcast_ref::<BinaryEval>()
            .unwrap()
            .clone();
        let LogicalPlan::Extension(lhs_ext) = eval.lhs else {
            unreachable!();
        };
        let lhs_unpack = lhs_ext
            .node
            .as_any()
            .downcast_ref::<WideUnpack>()
            .unwrap()
            .clone();

        let projected = LogicalPlanBuilder::from(lhs_unpack.input)
            .project(proj_exprs)?
            .build()?;

        let new_unpack = WideUnpack::new_with_options(
            projected,
            Arc::new(new_columns),
            Arc::new(output_label_keys),
            output_include_name,
        )
        .map_err(|e| datafusion::error::DataFusionError::Plan(e.to_string()))?;

        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(new_unpack),
        })))
    }
}

/// Marker used to remember which side a paired column's labels should come
/// from when constructing the output [`WideColumnMeta`]. The "many" side wins
/// (matches the behavior of `compute_binary_output_schema` in the OneToOne and
/// ManyToOne cases — and OneToMany picks RHS so each emitted series carries
/// a distinct label set).
#[derive(Copy, Clone)]
enum MetaSide {
    Lhs,
    Rhs,
}

/// Group `columns` into buckets keyed by their matching key. Iteration order
/// of the returned `BTreeMap` is stable, which keeps the rewritten plan
/// deterministic across runs.
fn bucket_by_match_key<'a>(
    columns: &'a [WideColumnMeta],
    label_keys: &[String],
    matching: &VectorMatching,
) -> BTreeMap<Vec<String>, Vec<&'a WideColumnMeta>> {
    let mut buckets: BTreeMap<Vec<String>, Vec<&WideColumnMeta>> = BTreeMap::new();
    for c in columns {
        let key = matching_key(c, label_keys, matching);
        buckets.entry(key).or_default().push(c);
    }
    buckets
}

/// Build the matching key for a single wide column, mirroring
/// `match_selector` in [`crate::exec::binary_eval`]:
///
///   * `on(L1, …)` — values for those labels in order. `__name__` is only
///     included if listed.
///   * `ignoring(L1, …)` — values for `label_keys \ ignoring \ {__name__}`.
///   * default — values for `label_keys \ {__name__}`.
///
/// Missing labels resolve to the empty string (matching the empty-id behavior
/// in the executor).
fn matching_key(
    c: &WideColumnMeta,
    label_keys: &[String],
    matching: &VectorMatching,
) -> Vec<String> {
    match (&matching.on_labels, &matching.ignoring_labels) {
        (Some(on), _) => on
            .iter()
            .map(|l| {
                if l == "__name__" {
                    c.metric_name.clone()
                } else {
                    c.labels.get(l.as_str()).cloned().unwrap_or_default()
                }
            })
            .collect(),
        (_, Some(ignoring)) => {
            let ignore_set: HashSet<&str> = ignoring.iter().map(|s| s.as_str()).collect();
            label_keys
                .iter()
                .filter(|k| k.as_str() != "__name__" && !ignore_set.contains(k.as_str()))
                .map(|k| c.labels.get(k.as_str()).cloned().unwrap_or_default())
                .collect()
        }
        (None, None) => label_keys
            .iter()
            .filter(|k| k.as_str() != "__name__")
            .map(|k| c.labels.get(k.as_str()).cloned().unwrap_or_default())
            .collect(),
    }
}

/// Compute the output `WideUnpack`'s `(label_keys, include_name)` so that the
/// emitted long-format schema matches the schema produced by
/// `compute_binary_output_schema` for the same input/operator/matching.
fn compute_output_label_keys(
    lhs_label_keys: &[String],
    lhs_include_name: bool,
    op: BinaryOp,
    matching: &VectorMatching,
) -> (Vec<String>, bool) {
    match (&matching.on_labels, &matching.ignoring_labels) {
        (Some(on), _) => {
            let lhs_lookup: HashSet<&str> = lhs_label_keys.iter().map(|s| s.as_str()).collect();
            let include_name = lhs_include_name && on.iter().any(|l| l == "__name__");
            let label_keys: Vec<String> = on
                .iter()
                .filter(|l| l.as_str() != "__name__" && lhs_lookup.contains(l.as_str()))
                .cloned()
                .collect();
            (label_keys, include_name)
        }
        (_, Some(ignoring)) => {
            let ignore_set: HashSet<&str> = ignoring.iter().map(|s| s.as_str()).collect();
            let label_keys: Vec<String> = lhs_label_keys
                .iter()
                .filter(|k| !ignore_set.contains(k.as_str()))
                .cloned()
                .collect();
            let include_name =
                lhs_include_name && !ignore_set.contains("__name__") && !op.drops_metric_name();
            (label_keys, include_name)
        }
        (None, None) => {
            let include_name = lhs_include_name && !op.drops_metric_name();
            (lhs_label_keys.to_vec(), include_name)
        }
    }
}

/// Build a `WideColumnMeta` for one emitted column. Labels are the source
/// column's labels filtered to the output label set; `metric_name` is reused
/// (it's only surfaced when `include_name` is true).
fn make_output_meta(
    source: &WideColumnMeta,
    output_label_set: &HashSet<&str>,
    _include_name: bool,
    col_name: String,
) -> WideColumnMeta {
    let mut labels = Labels::new();
    for (k, v) in source.labels.iter() {
        if output_label_set.contains(k.as_str()) {
            labels.insert(k.clone(), v.clone());
        }
    }
    WideColumnMeta {
        col_name,
        metric_name: source.metric_name.clone(),
        labels,
        value_kind: source.value_kind,
    }
}

/// Build the per-row expression for `op` applied to `(lhs_col, rhs_col)`,
/// honoring `return_bool`. Mirrors the row-level logic in `BinaryExec` for the
/// non-set-op cases.
fn build_binary_op_expr(
    lhs_col: Expr,
    rhs_col: Expr,
    op: BinaryOp,
    return_bool: bool,
) -> Result<Expr> {
    match op {
        BinaryOp::Add => Ok(lhs_col + rhs_col),
        BinaryOp::Sub => Ok(lhs_col - rhs_col),
        BinaryOp::Mul => Ok(lhs_col * rhs_col),
        BinaryOp::Div => Ok(lhs_col / rhs_col),
        BinaryOp::Mod => Ok(lhs_col % rhs_col),
        BinaryOp::Pow => Ok(datafusion::functions::math::expr_fn::power(
            lhs_col, rhs_col,
        )),
        BinaryOp::Eql
        | BinaryOp::Neq
        | BinaryOp::Lss
        | BinaryOp::Gtr
        | BinaryOp::Lte
        | BinaryOp::Gte => {
            let cmp = match op {
                BinaryOp::Eql => lhs_col.clone().eq(rhs_col),
                BinaryOp::Neq => lhs_col.clone().not_eq(rhs_col),
                BinaryOp::Lss => lhs_col.clone().lt(rhs_col),
                BinaryOp::Gtr => lhs_col.clone().gt(rhs_col),
                BinaryOp::Lte => lhs_col.clone().lt_eq(rhs_col),
                BinaryOp::Gte => lhs_col.clone().gt_eq(rhs_col),
                _ => unreachable!(),
            };
            if return_bool {
                when(cmp, lit(1.0_f64)).otherwise(lit(0.0_f64))
            } else {
                when(cmp, lhs_col).otherwise(lit(ScalarValue::Float64(None)))
            }
        }
        BinaryOp::Land | BinaryOp::Lor | BinaryOp::Lunless => {
            unreachable!("set operators are filtered out at the top of `rewrite`")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::DFSchema;
    use datafusion::logical_expr::{EmptyRelation, Extension, LogicalPlan};
    use datafusion::optimizer::{Optimizer, OptimizerContext};

    use super::*;
    use crate::datasource::ValueKind;
    use crate::node::{
        BinaryEval, BinaryOp, MatchCardinality, VectorMatching, WideColumnMeta, WideUnpack,
    };
    use crate::types::Labels;

    /// Build a wide-format input plan with `(timestamp, c0, c1, c2, c3)` cols.
    fn make_wide_input() -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("c0", DataType::Float64, true),
            Field::new("c1", DataType::Float64, true),
            Field::new("c2", DataType::Float64, true),
            Field::new("c3", DataType::Float64, true),
        ]);
        let df_schema = DFSchema::try_from(schema).unwrap();
        LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        })
    }

    fn meta(col_name: &str, metric: &str, labels: &[(&str, &str)]) -> WideColumnMeta {
        let mut l = Labels::new();
        for (k, v) in labels {
            l.insert((*k).to_string(), (*v).to_string());
        }
        WideColumnMeta {
            col_name: col_name.into(),
            metric_name: metric.into(),
            labels: l,
            value_kind: ValueKind::Scalar,
        }
    }

    fn wrap_unpack(
        input: LogicalPlan,
        columns: Vec<WideColumnMeta>,
        label_keys: Vec<String>,
        include_name: bool,
    ) -> LogicalPlan {
        let unpack = WideUnpack::new_with_options(
            input,
            Arc::new(columns),
            Arc::new(label_keys),
            include_name,
        )
        .unwrap();
        LogicalPlan::Extension(Extension {
            node: Arc::new(unpack),
        })
    }

    fn wrap_binary(
        lhs: LogicalPlan,
        rhs: LogicalPlan,
        op: BinaryOp,
        return_bool: bool,
        matching: VectorMatching,
    ) -> LogicalPlan {
        let eval = BinaryEval::new(lhs, rhs, op, return_bool, matching).unwrap();
        LogicalPlan::Extension(Extension {
            node: Arc::new(eval),
        })
    }

    fn run(plan: LogicalPlan) -> LogicalPlan {
        let optimizer = Optimizer::with_rules(vec![Arc::new(PushBinaryThroughWideUnpack)]);
        optimizer
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
    }

    /// Top of the optimized plan must be a `WideUnpack` over a `Projection`
    /// over the original wide input. Returns the inner projection plan.
    fn assert_pushed(plan: &LogicalPlan) -> &LogicalPlan {
        let LogicalPlan::Extension(ext) = plan else {
            panic!("expected Extension at top, got:\n{plan}");
        };
        assert!(
            ext.node.as_any().downcast_ref::<WideUnpack>().is_some(),
            "top should be WideUnpack, got {}",
            ext.node.name()
        );
        let inputs = ext.node.inputs();
        assert_eq!(inputs.len(), 1);
        let LogicalPlan::Projection(_) = inputs[0] else {
            panic!("expected Projection, got:\n{}", inputs[0]);
        };
        inputs[0]
    }

    fn unpack_of(plan: &LogicalPlan) -> &WideUnpack {
        let LogicalPlan::Extension(ext) = plan else {
            panic!("expected Extension, got:\n{plan}");
        };
        ext.node.as_any().downcast_ref::<WideUnpack>().unwrap()
    }

    /// Two unpacks over a shared wide input:
    ///   LHS columns: c0(op=user), c1(op=system)
    ///   RHS columns: c2(op=user), c3(op=system)
    /// `on(op)` should pair (c0,c2) and (c1,c3).
    #[test]
    fn one_to_one_with_on_label() {
        let input = make_wide_input();
        let lhs = wrap_unpack(
            input.clone(),
            vec![
                meta("c0", "metric_a", &[("op", "user")]),
                meta("c1", "metric_a", &[("op", "system")]),
            ],
            vec!["op".into()],
            true,
        );
        let rhs = wrap_unpack(
            input,
            vec![
                meta("c2", "metric_b", &[("op", "user")]),
                meta("c3", "metric_b", &[("op", "system")]),
            ],
            vec!["op".into()],
            true,
        );

        let matching = VectorMatching {
            card: MatchCardinality::OneToOne,
            on_labels: Some(vec!["op".into()]),
            ignoring_labels: None,
        };
        let plan = wrap_binary(lhs, rhs, BinaryOp::Div, false, matching);
        let optimized = run(plan);

        assert_pushed(&optimized);
        let unpack = unpack_of(&optimized);
        // `on(op)` keeps only `op`; arithmetic op drops __name__ regardless.
        assert_eq!(unpack.label_keys.as_slice(), &["op".to_string()]);
        assert!(!unpack.include_name);
        // Two pairs → two emitted columns.
        assert_eq!(unpack.columns.len(), 2);
    }

    /// Default matching with `ignoring(role)` on label_keys=[op, role]:
    /// matching key uses `op` only.
    #[test]
    fn one_to_one_with_ignoring_label() {
        let input = make_wide_input();
        let lhs = wrap_unpack(
            input.clone(),
            vec![
                meta("c0", "m", &[("op", "user"), ("role", "primary")]),
                meta("c1", "m", &[("op", "system"), ("role", "primary")]),
            ],
            vec!["op".into(), "role".into()],
            true,
        );
        let rhs = wrap_unpack(
            input,
            vec![
                meta("c2", "m", &[("op", "user"), ("role", "secondary")]),
                meta("c3", "m", &[("op", "system"), ("role", "secondary")]),
            ],
            vec!["op".into(), "role".into()],
            true,
        );

        let matching = VectorMatching {
            card: MatchCardinality::OneToOne,
            on_labels: None,
            ignoring_labels: Some(vec!["role".into()]),
        };
        let plan = wrap_binary(lhs, rhs, BinaryOp::Mul, false, matching);
        let optimized = run(plan);

        assert_pushed(&optimized);
        let unpack = unpack_of(&optimized);
        // `ignoring(role)` keeps `op` (and would keep `role` too, since
        // ignoring filters from the schema's label set; but `role` is filtered).
        assert_eq!(unpack.label_keys.as_slice(), &["op".to_string()]);
        assert_eq!(unpack.columns.len(), 2);
    }

    /// ManyToOne: LHS has many columns per matching key, RHS has one.
    #[test]
    fn many_to_one_pairs_each_lhs_with_single_rhs() {
        let input = make_wide_input();
        let lhs = wrap_unpack(
            input.clone(),
            vec![
                // bucket "user": c0, c1
                meta("c0", "m", &[("op", "user"), ("inst", "a")]),
                meta("c1", "m", &[("op", "user"), ("inst", "b")]),
                // bucket "system": c2
                meta("c2", "m", &[("op", "system"), ("inst", "a")]),
            ],
            vec!["op".into(), "inst".into()],
            true,
        );
        let rhs = wrap_unpack(
            input,
            vec![
                // bucket "user": c3
                meta("c3", "m", &[("op", "user"), ("inst", "x")]),
                // bucket "system": no match (skipped)
            ],
            vec!["op".into(), "inst".into()],
            true,
        );

        let matching = VectorMatching {
            card: MatchCardinality::ManyToOne(vec![]),
            on_labels: Some(vec!["op".into()]),
            ignoring_labels: None,
        };
        let plan = wrap_binary(lhs, rhs, BinaryOp::Div, false, matching);
        let optimized = run(plan);

        let proj_plan = assert_pushed(&optimized);
        let unpack = unpack_of(&optimized);
        // Two LHS columns matched the single RHS column in bucket "user".
        // The "system" bucket is dropped because RHS has no match.
        assert_eq!(unpack.columns.len(), 2);
        // `on(op)` keeps only `op`; arithmetic op drops __name__.
        assert_eq!(unpack.label_keys.as_slice(), &["op".to_string()]);
        assert!(!unpack.include_name);
        // Both emitted columns belong to the "user" bucket.
        for c in unpack.columns.iter() {
            assert_eq!(c.labels.get("op").map(|s| s.as_str()), Some("user"));
        }
        // Each emitted projection expression must reference the single RHS
        // column (c3) and a distinct LHS column.
        let LogicalPlan::Projection(proj) = proj_plan else {
            unreachable!()
        };
        let bin_exprs: Vec<String> = proj.expr.iter().skip(1).map(|e| format!("{e}")).collect();
        assert_eq!(bin_exprs.len(), 2);
        let mut saw_c0 = false;
        let mut saw_c1 = false;
        for e in &bin_exprs {
            assert!(e.contains("c3"), "expr should reference RHS c3: {e}");
            if e.contains("c0") {
                saw_c0 = true;
            }
            if e.contains("c1") {
                saw_c1 = true;
            }
        }
        assert!(saw_c0 && saw_c1, "both LHS columns should be paired");
    }

    /// OneToMany: LHS has one column per matching key, RHS has many. The output
    /// labels come from the RHS side so the emitted series have distinct keys.
    #[test]
    fn one_to_many_pairs_single_lhs_with_each_rhs() {
        let input = make_wide_input();
        let lhs = wrap_unpack(
            input.clone(),
            vec![meta("c0", "m", &[("op", "user"), ("inst", "lhs")])],
            vec!["op".into(), "inst".into()],
            true,
        );
        let rhs = wrap_unpack(
            input,
            vec![
                meta("c1", "m", &[("op", "user"), ("inst", "x")]),
                meta("c2", "m", &[("op", "user"), ("inst", "y")]),
            ],
            vec!["op".into(), "inst".into()],
            true,
        );

        let matching = VectorMatching {
            card: MatchCardinality::OneToMany(vec![]),
            on_labels: Some(vec!["op".into()]),
            ignoring_labels: None,
        };
        let plan = wrap_binary(lhs, rhs, BinaryOp::Div, false, matching);
        let optimized = run(plan);

        let proj_plan = assert_pushed(&optimized);
        let unpack = unpack_of(&optimized);
        // Two RHS columns matched the single LHS column in bucket "user".
        assert_eq!(unpack.columns.len(), 2);
        assert_eq!(unpack.label_keys.as_slice(), &["op".to_string()]);
        assert!(!unpack.include_name);
        for c in unpack.columns.iter() {
            assert_eq!(c.labels.get("op").map(|s| s.as_str()), Some("user"));
        }
        // The single LHS column (c0) appears in every emitted expression, and
        // each RHS column appears in exactly one emitted expression.
        let LogicalPlan::Projection(proj) = proj_plan else {
            unreachable!()
        };
        let bin_exprs: Vec<String> = proj.expr.iter().skip(1).map(|e| format!("{e}")).collect();
        assert_eq!(bin_exprs.len(), 2);
        let mut saw_c1 = false;
        let mut saw_c2 = false;
        for e in &bin_exprs {
            assert!(e.contains("c0"), "expr should reference LHS c0: {e}");
            if e.contains("c1") {
                saw_c1 = true;
            }
            if e.contains("c2") {
                saw_c2 = true;
            }
        }
        assert!(saw_c1 && saw_c2, "both RHS columns should be paired");
    }

    /// Arithmetic op drops `__name__` from the new `WideUnpack`.
    #[test]
    fn arithmetic_drops_metric_name() {
        let input = make_wide_input();
        let lhs = wrap_unpack(
            input.clone(),
            vec![meta("c0", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let rhs = wrap_unpack(
            input,
            vec![meta("c1", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let plan = wrap_binary(
            lhs,
            rhs,
            BinaryOp::Add,
            false,
            VectorMatching::default_matching(),
        );
        let optimized = run(plan);
        let unpack = unpack_of(&optimized);
        assert!(!unpack.include_name);
    }

    /// Filtering comparison without `bool` produces a CASE … ELSE NULL so that
    /// the downstream `value IS NOT NULL` filter drops false rows.
    #[test]
    fn filtering_comparison_emits_case_else_null() {
        let input = make_wide_input();
        let lhs = wrap_unpack(
            input.clone(),
            vec![meta("c0", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let rhs = wrap_unpack(
            input,
            vec![meta("c1", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let plan = wrap_binary(
            lhs,
            rhs,
            BinaryOp::Gtr,
            false,
            VectorMatching::default_matching(),
        );
        let optimized = run(plan);

        let proj_plan = assert_pushed(&optimized);
        let LogicalPlan::Projection(proj) = proj_plan else {
            unreachable!()
        };
        // First expr is `timestamp`; the second is the binary op.
        let bin_expr = format!("{}", proj.expr[1]);
        assert!(
            bin_expr.contains("CASE") && bin_expr.contains("NULL"),
            "expected CASE … ELSE NULL, got: {bin_expr}"
        );
    }

    /// Comparison with `bool` produces 1.0 / 0.0.
    #[test]
    fn bool_comparison_emits_one_zero() {
        let input = make_wide_input();
        let lhs = wrap_unpack(
            input.clone(),
            vec![meta("c0", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let rhs = wrap_unpack(
            input,
            vec![meta("c1", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let plan = wrap_binary(
            lhs,
            rhs,
            BinaryOp::Eql,
            true,
            VectorMatching::default_matching(),
        );
        let optimized = run(plan);

        let proj_plan = assert_pushed(&optimized);
        let LogicalPlan::Projection(proj) = proj_plan else {
            unreachable!()
        };
        let bin_expr = format!("{}", proj.expr[1]);
        assert!(
            bin_expr.contains("CASE")
                && bin_expr.contains("Float64(1)")
                && bin_expr.contains("Float64(0)"),
            "expected CASE … 1.0 ELSE 0.0, got: {bin_expr}"
        );
    }

    /// When neither side is a `WideUnpack`, the rule must not fire.
    #[test]
    fn does_not_fire_without_wide_unpack_inputs() {
        // Build two long-format-shaped EmptyRelations directly.
        let long_schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("value", DataType::Float64, true),
            Field::new("op", DataType::Utf8, false),
        ]);
        let df = Arc::new(DFSchema::try_from(long_schema).unwrap());
        let lhs = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: df.clone(),
        });
        let rhs = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: df,
        });

        let plan = wrap_binary(
            lhs,
            rhs,
            BinaryOp::Add,
            false,
            VectorMatching::default_matching(),
        );
        let optimized = run(plan);

        // Top should still be BinaryEval.
        let LogicalPlan::Extension(ext) = &optimized else {
            panic!("expected Extension at top");
        };
        assert!(ext.node.as_any().downcast_ref::<BinaryEval>().is_some());
    }

    /// Set operators (`and`/`or`/`unless`) are out of scope.
    #[test]
    fn does_not_fire_for_set_operators() {
        let input = make_wide_input();
        let lhs = wrap_unpack(
            input.clone(),
            vec![meta("c0", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let rhs = wrap_unpack(
            input,
            vec![meta("c1", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );

        for op in [BinaryOp::Land, BinaryOp::Lor, BinaryOp::Lunless] {
            let plan = wrap_binary(
                lhs.clone(),
                rhs.clone(),
                op,
                false,
                VectorMatching::default_matching(),
            );
            let optimized = run(plan);
            let LogicalPlan::Extension(ext) = &optimized else {
                panic!("expected Extension at top, op={op:?}");
            };
            assert!(
                ext.node.as_any().downcast_ref::<BinaryEval>().is_some(),
                "set op {op:?} should not be pushed",
            );
        }
    }

    /// Schema of the pushed-down plan equals the schema before pushdown.
    #[test]
    fn output_schema_matches_original() {
        for op in [
            BinaryOp::Add,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
            BinaryOp::Lss,
            BinaryOp::Gte,
        ] {
            for return_bool in [false, true] {
                if return_bool && !op.is_comparison() {
                    continue;
                }
                let input = make_wide_input();
                let lhs = wrap_unpack(
                    input.clone(),
                    vec![meta("c0", "m", &[("op", "user")])],
                    vec!["op".into()],
                    true,
                );
                let rhs = wrap_unpack(
                    input,
                    vec![meta("c1", "m", &[("op", "user")])],
                    vec!["op".into()],
                    true,
                );
                let plan = wrap_binary(
                    lhs,
                    rhs,
                    op,
                    return_bool,
                    VectorMatching::default_matching(),
                );
                let original_schema = plan.schema().clone();

                let optimized = run(plan);
                let optimized_schema = optimized.schema().clone();

                let original_names: Vec<&str> = original_schema
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .collect();
                let optimized_names: Vec<&str> = optimized_schema
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .collect();
                assert_eq!(
                    original_names, optimized_names,
                    "schema mismatch for op={op:?} return_bool={return_bool}"
                );
            }
        }
    }

    /// When the two `WideUnpack`s wrap structurally different wide subplans,
    /// the rule must not fire (case 2 territory).
    #[test]
    fn does_not_fire_with_distinct_wide_inputs() {
        // Build two distinct EmptyRelations (different schemas) so the
        // fingerprints don't match.
        let lhs_input = make_wide_input();
        let rhs_input = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(
                DFSchema::try_from(Schema::new(vec![
                    Field::new("timestamp", DataType::UInt64, false),
                    Field::new("d0", DataType::Float64, true),
                ]))
                .unwrap(),
            ),
        });

        let lhs = wrap_unpack(
            lhs_input,
            vec![meta("c0", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let rhs = wrap_unpack(
            rhs_input,
            vec![meta("d0", "m", &[("op", "user")])],
            vec!["op".into()],
            true,
        );
        let plan = wrap_binary(
            lhs,
            rhs,
            BinaryOp::Add,
            false,
            VectorMatching::default_matching(),
        );
        let optimized = run(plan);
        let LogicalPlan::Extension(ext) = &optimized else {
            panic!("expected Extension at top");
        };
        assert!(ext.node.as_any().downcast_ref::<BinaryEval>().is_some());
    }
}
