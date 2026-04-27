use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::{col, lit};

use crate::node::WideUnpack;

/// Optimizer rule that lowers a trivial [`WideUnpack`] (one value column, no
/// label keys) into a plain `Projection`.
///
/// When `WideUnpack` only has a single value column and no label keys to emit,
/// it does no real per-row work: it renames the input value column to `value`
/// and (optionally) appends a constant `__name__` literal. A standard
/// projection expresses the same semantics:
///
/// ```text
/// WideUnpack: 1 columns, labels=[]               Projection: timestamp, col_0 AS value, lit(name) AS __name__
///   <wide input: timestamp, col_0>          →      <wide input: timestamp, col_0>
/// ```
///
/// The output schema is preserved: `(timestamp, value, [__name__])`. The
/// `__name__` literal is only emitted when the unpack had `include_name=true`
/// (e.g., the column-pruning rule may have already disabled it).
///
/// The rule deliberately does *not* apply when the unpack has multiple columns
/// or any label keys, because in that case the unpack actually expands rows or
/// projects per-column label literals — work that a single projection cannot
/// express.
#[derive(Debug)]
pub struct LowerTrivialWideUnpack;

impl OptimizerRule for LowerTrivialWideUnpack {
    fn name(&self) -> &str {
        "lower_trivial_wide_unpack"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(ref ext) = plan else {
            return Ok(Transformed::no(plan));
        };

        let Some(unpack) = ext.node.as_any().downcast_ref::<WideUnpack>() else {
            return Ok(Transformed::no(plan));
        };

        if unpack.columns.len() != 1 || !unpack.label_keys.is_empty() {
            return Ok(Transformed::no(plan));
        }

        let LogicalPlan::Extension(ext) = plan else {
            unreachable!();
        };
        let unpack = ext
            .node
            .as_any()
            .downcast_ref::<WideUnpack>()
            .unwrap()
            .clone();

        let col_meta = &unpack.columns[0];

        let mut exprs: Vec<Expr> = Vec::with_capacity(3);
        exprs.push(col("timestamp"));
        exprs.push(col(col_meta.col_name.as_str()).alias("value"));
        if unpack.include_name {
            exprs.push(lit(col_meta.metric_name.clone()).alias("__name__"));
        }

        let WideUnpack { input, .. } = unpack;
        let new_plan = LogicalPlanBuilder::from(input).project(exprs)?.build()?;

        Ok(Transformed::yes(new_plan))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::DFSchema;
    use datafusion::logical_expr::{EmptyRelation, Extension, LogicalPlan};
    use datafusion::optimizer::Optimizer;
    use datafusion::optimizer::OptimizerContext;

    use super::*;
    use crate::node::{WideColumnMeta, WideUnpack};
    use crate::types::Labels;

    fn make_wide_input() -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("col_0", DataType::Float64, true),
        ]);
        let df_schema = DFSchema::try_from(schema).unwrap();
        LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        })
    }

    #[test]
    fn lowers_single_column_no_labels() {
        let input = make_wide_input();
        let columns = Arc::new(vec![WideColumnMeta {
            col_name: "col_0".to_string(),
            metric_name: "cpu_cores".to_string(),
            labels: Labels::new(),
        }]);
        let unpack = WideUnpack::new(input, columns, Arc::new(vec![])).unwrap();
        let plan = LogicalPlan::Extension(Extension {
            node: Arc::new(unpack),
        });

        let optimizer = Optimizer::with_rules(vec![Arc::new(LowerTrivialWideUnpack)]);
        let optimized = optimizer
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap();

        // The extension node should be gone, replaced with a Projection.
        let LogicalPlan::Projection(proj) = &optimized else {
            panic!("expected Projection, got: {optimized:?}");
        };
        assert_eq!(proj.expr.len(), 3);

        let names: Vec<&str> = proj
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["timestamp", "value", "__name__"]);
    }

    #[test]
    fn lowers_single_column_no_labels_no_name() {
        let input = make_wide_input();
        let columns = Arc::new(vec![WideColumnMeta {
            col_name: "col_0".to_string(),
            metric_name: "cpu_cores".to_string(),
            labels: Labels::new(),
        }]);
        let unpack = WideUnpack::new_with_options(input, columns, Arc::new(vec![]), false).unwrap();
        let plan = LogicalPlan::Extension(Extension {
            node: Arc::new(unpack),
        });

        let optimizer = Optimizer::with_rules(vec![Arc::new(LowerTrivialWideUnpack)]);
        let optimized = optimizer
            .optimize(plan, &OptimizerContext::new(), |_, _| {})
            .unwrap();

        let LogicalPlan::Projection(proj) = &optimized else {
            panic!("expected Projection, got: {optimized:?}");
        };
        assert_eq!(proj.expr.len(), 2);

        let names: Vec<&str> = proj
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["timestamp", "value"]);
    }

    #[test]
    fn skips_multi_column_unpack() {
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("col_0", DataType::Float64, true),
            Field::new("col_1", DataType::Float64, true),
        ]);
        let df_schema = DFSchema::try_from(schema).unwrap();
        let input = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        });
        let columns = Arc::new(vec![
            WideColumnMeta {
                col_name: "col_0".to_string(),
                metric_name: "cpu".to_string(),
                labels: Labels::new(),
            },
            WideColumnMeta {
                col_name: "col_1".to_string(),
                metric_name: "cpu".to_string(),
                labels: Labels::new(),
            },
        ]);
        let unpack = WideUnpack::new(input, columns, Arc::new(vec![])).unwrap();
        let plan = LogicalPlan::Extension(Extension {
            node: Arc::new(unpack),
        });

        let optimizer = Optimizer::with_rules(vec![Arc::new(LowerTrivialWideUnpack)]);
        let optimized = optimizer
            .optimize(plan.clone(), &OptimizerContext::new(), |_, _| {})
            .unwrap();

        // Should still be an Extension node — no rewrite.
        assert!(matches!(optimized, LogicalPlan::Extension(_)));
    }

    #[test]
    fn skips_unpack_with_label_keys() {
        let input = make_wide_input();
        let mut labels = Labels::new();
        labels.insert("instance".to_string(), "host1".to_string());
        let columns = Arc::new(vec![WideColumnMeta {
            col_name: "col_0".to_string(),
            metric_name: "cpu_cores".to_string(),
            labels,
        }]);
        let unpack =
            WideUnpack::new(input, columns, Arc::new(vec!["instance".to_string()])).unwrap();
        let plan = LogicalPlan::Extension(Extension {
            node: Arc::new(unpack),
        });

        let optimizer = Optimizer::with_rules(vec![Arc::new(LowerTrivialWideUnpack)]);
        let optimized = optimizer
            .optimize(plan.clone(), &OptimizerContext::new(), |_, _| {})
            .unwrap();

        assert!(matches!(optimized, LogicalPlan::Extension(_)));
    }
}
