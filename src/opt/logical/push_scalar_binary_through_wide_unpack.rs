use std::sync::Arc;

use datafusion::common::ScalarValue;
use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, LogicalPlanBuilder, when};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::{col, lit};

use crate::node::{BinaryOp, ScalarBinaryEval, WideUnpack};

/// Optimizer rule that pushes a [`ScalarBinaryEval`] down through a
/// [`WideUnpack`] node by rewriting the operation as a per-column projection
/// applied to the wide-format input.
///
/// ```text
/// ScalarBinaryEval(scalar, op, return_bool)        WideUnpack(columns, label_keys, include_name=keep_name)
///   WideUnpack(columns, label_keys)           →      Projection(timestamp, op(col_0), …, op(col_N))
///     WideInput                                        WideInput
/// ```
///
/// This is valid because:
///
///   * Each value column of the wide input maps 1:1 to a long-format series in
///     [`WideUnpack`]'s output, with constant labels recorded in its
///     [`WideColumnMeta`](crate::node::WideColumnMeta).
///   * The scalar binary op is evaluated independently for every
///     `(timestamp, series)` pair. Applying it column-wise on the wide
///     representation, then unpacking, produces the same long-format tuples
///     as unpacking first and then applying.
///   * For filtering comparison ops (no `bool` modifier), rows where the
///     comparison is false become `NULL` in the projected wide column. The
///     downstream `value IS NOT NULL` filter (added by the engine after
///     optimization) drops those rows, matching the row-dropping behavior of
///     [`ScalarBinaryExec`](crate::exec).
///   * Arithmetic operators drop `__name__` from the output. The new
///     [`WideUnpack`] honors this by setting `include_name = false` whenever
///     `op.drops_metric_name()` is true.
///
/// The two main wins are:
///
///   * The scalar arithmetic runs on N (typically ~900) slim value columns
///     rather than on N × #timestamps long-format rows.
///   * For filtering comparisons, the long-format expansion runs after rows
///     have been nulled out, so it materializes fewer rows.
#[derive(Debug)]
pub struct PushScalarBinaryThroughWideUnpack;

impl OptimizerRule for PushScalarBinaryThroughWideUnpack {
    fn name(&self) -> &str {
        "push_scalar_binary_through_wide_unpack"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Match: Extension(ScalarBinaryEval) whose input is Extension(WideUnpack).
        let LogicalPlan::Extension(ref ext) = plan else {
            return Ok(Transformed::no(plan));
        };

        let Some(eval) = ext.node.as_any().downcast_ref::<ScalarBinaryEval>() else {
            return Ok(Transformed::no(plan));
        };

        // Set operators (and/or/unless) are only valid between two vectors and
        // should never appear on a `ScalarBinaryEval`. Bail defensively rather
        // than crash if one slips through.
        if eval.op.is_set_operator() {
            return Ok(Transformed::no(plan));
        }

        let LogicalPlan::Extension(ref inner_ext) = eval.input else {
            return Ok(Transformed::no(plan));
        };

        if inner_ext
            .node
            .as_any()
            .downcast_ref::<WideUnpack>()
            .is_none()
        {
            return Ok(Transformed::no(plan));
        };

        // Destructure to take ownership of the inner nodes without cloning
        // the underlying sub-plan.
        let LogicalPlan::Extension(ext) = plan else {
            unreachable!();
        };
        let eval = ext
            .node
            .as_any()
            .downcast_ref::<ScalarBinaryEval>()
            .unwrap()
            .clone();
        let LogicalPlan::Extension(inner_ext) = eval.input else {
            unreachable!();
        };
        let unpack = inner_ext
            .node
            .as_any()
            .downcast_ref::<WideUnpack>()
            .unwrap()
            .clone();

        // Build the column-wise projection: pass `timestamp` through and
        // replace each value column with the scalar op applied to it.
        let mut proj_exprs: Vec<Expr> = Vec::with_capacity(unpack.columns.len() + 1);
        proj_exprs.push(col("timestamp"));
        for c in unpack.columns.iter() {
            let new_expr = build_scalar_op_expr(
                col(c.col_name.as_str()),
                eval.scalar_value,
                eval.op,
                eval.scalar_is_lhs,
                eval.return_bool,
            )?;
            proj_exprs.push(new_expr.alias(c.col_name.as_str()));
        }

        let projected = LogicalPlanBuilder::from(unpack.input)
            .project(proj_exprs)?
            .build()?;

        // Drop `__name__` from the new `WideUnpack` whenever the operator
        // drops it, matching the schema produced by the original
        // `ScalarBinaryEval`.
        let include_name = unpack.include_name && !eval.op.drops_metric_name();
        let new_unpack = WideUnpack::new_with_options(
            projected,
            unpack.columns,
            unpack.label_keys,
            include_name,
        )
        .map_err(|e| datafusion::error::DataFusionError::Plan(e.to_string()))?;

        let new_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(new_unpack),
        });

        Ok(Transformed::yes(new_plan))
    }
}

/// Build the per-row expression for `op` applied to `col_expr` and `scalar`,
/// honoring `scalar_is_lhs` and `return_bool`. Mirrors the row-level logic in
/// `ScalarBinaryExec`.
fn build_scalar_op_expr(
    col_expr: Expr,
    scalar: f64,
    op: BinaryOp,
    scalar_is_lhs: bool,
    return_bool: bool,
) -> Result<Expr> {
    let scalar_expr = lit(scalar);
    let (lhs, rhs) = if scalar_is_lhs {
        (scalar_expr, col_expr.clone())
    } else {
        (col_expr.clone(), scalar_expr)
    };

    match op {
        BinaryOp::Add => Ok(lhs + rhs),
        BinaryOp::Sub => Ok(lhs - rhs),
        BinaryOp::Mul => Ok(lhs * rhs),
        BinaryOp::Div => Ok(lhs / rhs),
        BinaryOp::Mod => Ok(lhs % rhs),
        BinaryOp::Pow => Ok(datafusion::functions::math::expr_fn::power(lhs, rhs)),
        BinaryOp::Eql
        | BinaryOp::Neq
        | BinaryOp::Lss
        | BinaryOp::Gtr
        | BinaryOp::Lte
        | BinaryOp::Gte => {
            let cmp = match op {
                BinaryOp::Eql => lhs.eq(rhs),
                BinaryOp::Neq => lhs.not_eq(rhs),
                BinaryOp::Lss => lhs.lt(rhs),
                BinaryOp::Gtr => lhs.gt(rhs),
                BinaryOp::Lte => lhs.lt_eq(rhs),
                BinaryOp::Gte => lhs.gt_eq(rhs),
                _ => unreachable!(),
            };
            if return_bool {
                // bool modifier: emit 1.0 / 0.0 (no row dropping).
                when(cmp, lit(1.0_f64)).otherwise(lit(0.0_f64))
            } else {
                // Filtering comparison: emit the column value when the
                // comparison is true, NULL otherwise. Downstream the
                // `value IS NOT NULL` filter drops the false rows.
                when(cmp, col_expr).otherwise(lit(ScalarValue::Float64(None)))
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
    use crate::node::{ScalarBinaryEval, WideColumnMeta, WideUnpack};
    use crate::types::Labels;

    /// Build a wide-format input plan with `(timestamp, c0, c1, c2)` columns.
    fn make_wide_input() -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("c0", DataType::Float64, true),
            Field::new("c1", DataType::Float64, true),
            Field::new("c2", DataType::Float64, true),
        ]);
        let df_schema = DFSchema::try_from(schema).unwrap();
        LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        })
    }

    /// Wrap `input` in a WideUnpack with the given options.
    fn wrap_unpack(input: LogicalPlan, include_name: bool) -> LogicalPlan {
        let mk = |col_name: &str, host: &str| -> WideColumnMeta {
            let mut labels = Labels::new();
            labels.insert("instance".into(), host.into());
            WideColumnMeta {
                col_name: col_name.into(),
                metric_name: "cpu".into(),
                labels,
                value_kind: ValueKind::Scalar,
            }
        };
        let columns = Arc::new(vec![
            mk("c0", "host-a"),
            mk("c1", "host-b"),
            mk("c2", "host-c"),
        ]);
        let label_keys = Arc::new(vec!["instance".to_string()]);
        let unpack =
            WideUnpack::new_with_options(input, columns, label_keys, include_name).unwrap();
        LogicalPlan::Extension(Extension {
            node: Arc::new(unpack),
        })
    }

    fn wrap_scalar_eval(
        input: LogicalPlan,
        scalar: f64,
        op: BinaryOp,
        scalar_is_lhs: bool,
        return_bool: bool,
    ) -> LogicalPlan {
        let eval = ScalarBinaryEval::new(input, scalar, op, scalar_is_lhs, return_bool).unwrap();
        LogicalPlan::Extension(Extension {
            node: Arc::new(eval),
        })
    }

    fn run(plan: LogicalPlan) -> LogicalPlan {
        let optimizer = Optimizer::with_rules(vec![Arc::new(PushScalarBinaryThroughWideUnpack)]);
        optimizer
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap()
    }

    /// After pushdown, the plan must be `WideUnpack > Projection > WideInput`,
    /// with the projection containing one expression per wide value column
    /// plus the timestamp pass-through.
    fn assert_pushed(plan: &LogicalPlan, expected_cols: usize) -> &LogicalPlan {
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
        let LogicalPlan::Projection(proj) = inputs[0] else {
            panic!("expected Projection, got:\n{}", inputs[0]);
        };
        assert_eq!(proj.expr.len(), expected_cols + 1);
        inputs[0]
    }

    #[test]
    fn pushes_addition() {
        let input = make_wide_input();
        let unpack = wrap_unpack(input, true);
        let plan = wrap_scalar_eval(unpack, 5.0, BinaryOp::Add, false, false);

        let optimized = run(plan);
        assert_pushed(&optimized, 3);

        // Arithmetic ops drop __name__.
        let LogicalPlan::Extension(ref ext) = optimized else {
            unreachable!()
        };
        let unpack = ext.node.as_any().downcast_ref::<WideUnpack>().unwrap();
        assert!(!unpack.include_name);
    }

    #[test]
    fn pushes_subtraction_with_scalar_lhs() {
        let input = make_wide_input();
        let unpack = wrap_unpack(input, true);
        let plan = wrap_scalar_eval(unpack, 10.0, BinaryOp::Sub, true, false);

        let optimized = run(plan);
        assert_pushed(&optimized, 3);
    }

    #[test]
    fn pushes_division_and_pow() {
        for op in [BinaryOp::Div, BinaryOp::Mul, BinaryOp::Mod, BinaryOp::Pow] {
            let input = make_wide_input();
            let unpack = wrap_unpack(input, true);
            let plan = wrap_scalar_eval(unpack, 2.0, op, false, false);

            let optimized = run(plan);
            assert_pushed(&optimized, 3);
        }
    }

    #[test]
    fn pushes_filtering_comparison() {
        // `value > 5` (no bool modifier) — false rows must be NULL after the
        // projection so the downstream `value IS NOT NULL` filter drops them.
        let input = make_wide_input();
        let unpack = wrap_unpack(input, true);
        let plan = wrap_scalar_eval(unpack, 5.0, BinaryOp::Gtr, false, false);

        let optimized = run(plan);
        assert_pushed(&optimized, 3);
    }

    #[test]
    fn pushes_bool_comparison() {
        // `value == 5` with bool modifier — produces 1.0 / 0.0.
        let input = make_wide_input();
        let unpack = wrap_unpack(input, true);
        let plan = wrap_scalar_eval(unpack, 5.0, BinaryOp::Eql, false, true);

        let optimized = run(plan);
        assert_pushed(&optimized, 3);
    }

    #[test]
    fn output_drops_name_for_arithmetic() {
        let input = make_wide_input();
        let unpack = wrap_unpack(input, true);
        let plan = wrap_scalar_eval(unpack, 1.0, BinaryOp::Add, false, false);

        let optimized = run(plan);
        let names: Vec<&str> = optimized
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["timestamp", "value", "instance"]);
    }

    #[test]
    fn propagates_existing_no_name_input() {
        // If the input WideUnpack already has include_name=false (e.g., set by
        // PruneWideUnpackColumns), the rewritten WideUnpack must keep it off.
        let input = make_wide_input();
        let unpack = wrap_unpack(input, false);
        let plan = wrap_scalar_eval(unpack, 1.0, BinaryOp::Add, false, false);

        let optimized = run(plan);
        let LogicalPlan::Extension(ref ext) = optimized else {
            panic!("expected Extension");
        };
        let unpack = ext.node.as_any().downcast_ref::<WideUnpack>().unwrap();
        assert!(!unpack.include_name);
    }

    #[test]
    fn does_not_fire_without_unpack_input() {
        // Plain Projection input — nothing for the rule to grab onto.
        let input = make_wide_input();
        let plan = wrap_scalar_eval(input, 1.0, BinaryOp::Add, false, false);

        let optimized = run(plan);
        let LogicalPlan::Extension(ref ext) = optimized else {
            panic!("expected Extension at top");
        };
        // Top should still be ScalarBinaryEval.
        assert!(
            ext.node
                .as_any()
                .downcast_ref::<ScalarBinaryEval>()
                .is_some()
        );
    }

    #[test]
    fn output_schema_matches_original() {
        // The pushed-down plan must have the same output schema as the
        // pre-rewrite plan (modulo the constraint that ScalarBinaryEval drops
        // __name__ for arithmetic ops).
        for op in [
            BinaryOp::Add,
            BinaryOp::Mul,
            BinaryOp::Pow,
            BinaryOp::Lss,
            BinaryOp::Gte,
        ] {
            for return_bool in [false, true] {
                if return_bool && !op.is_comparison() {
                    continue;
                }
                let input = make_wide_input();
                let unpack = wrap_unpack(input, true);
                let original = wrap_scalar_eval(unpack, 5.0, op, false, return_bool);
                let original_schema = original.schema().clone();

                let optimized = run(original);
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
}
