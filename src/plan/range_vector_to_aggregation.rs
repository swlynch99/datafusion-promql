use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use datafusion::common::tree_node::Transformed;
use datafusion::common::{Column, DFSchema};
use datafusion::error::Result;
use datafusion::logical_expr::expr::Alias;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Values, col, lit};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::func::range_udaf::make_range_udaf;
use crate::node::RangeVectorEval;

/// Optimizer rule that rewrites `RangeVectorEval` extension nodes into
/// standard DataFusion plans: a cross join with evaluation timestamps,
/// a time-window filter, and a group-by aggregate using a range-function UDAF.
///
/// This decomposition lets DataFusion's built-in optimizers handle sort
/// elimination, predicate push-down, and hash-aggregate parallelism.
#[derive(Debug)]
pub struct RangeVectorToAggregation;

impl OptimizerRule for RangeVectorToAggregation {
    fn name(&self) -> &str {
        "range_vector_to_aggregation"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Extension(ext) = &plan else {
            return Ok(Transformed::no(plan));
        };

        let Some(eval) = ext.node.as_any().downcast_ref::<RangeVectorEval>() else {
            return Ok(Transformed::no(plan));
        };

        let input = eval.input.clone();
        let range_ns = eval.range_ns;
        let func = eval.func;
        let label_columns = &eval.label_columns;
        let input_schema = input.schema().clone();

        // Capture the original output schema so we can reproduce its qualifiers.
        let original_schema = input_schema.clone();

        // Generate evaluation timestamps.
        let eval_timestamps =
            generate_eval_timestamps(eval.eval_ts_ns, eval.start_ns, eval.end_ns, eval.step_ns);

        // Step 1: Rename "timestamp" in input to "sample_ts" to avoid collision
        // with the eval_ts column we're about to introduce.
        let mut rename_exprs: Vec<Expr> = Vec::new();
        for field in input_schema.fields() {
            let name = field.name();
            let (qualifier, child_field) =
                input_schema.qualified_field_with_name(None, name.as_str())?;
            let col_expr = Expr::Column(Column::from((qualifier, child_field)));
            if name == "timestamp" {
                rename_exprs.push(col_expr.alias("sample_ts"));
            } else {
                rename_exprs.push(col_expr.alias(name.as_str()));
            }
        }
        let renamed_input = LogicalPlanBuilder::from(input)
            .project(rename_exprs)?
            .build()?;

        // Step 2: Create a Values scan for the evaluation timestamps.
        let values_schema = Arc::new(DFSchema::from_unqualified_fields(
            vec![Field::new("eval_ts", DataType::Int64, false)].into(),
            std::collections::HashMap::new(),
        )?);
        let values_rows: Vec<Vec<Expr>> = eval_timestamps.iter().map(|ts| vec![lit(*ts)]).collect();
        let values_plan = LogicalPlan::Values(Values {
            schema: values_schema,
            values: values_rows,
        });

        // Step 3: Cross join input samples with eval timestamps.
        let cross_joined = LogicalPlanBuilder::from(renamed_input)
            .cross_join(values_plan)?
            .build()?;

        // Step 4: Filter to keep only samples within the range window
        // [eval_ts - range_ns, eval_ts] for each evaluation timestamp.
        let filtered = LogicalPlanBuilder::from(cross_joined)
            .filter(
                col("sample_ts")
                    .gt_eq(col("eval_ts") - lit(range_ns))
                    .and(col("sample_ts").lt_eq(col("eval_ts"))),
            )?
            .build()?;

        // Step 5: Aggregate by (eval_ts, label_columns...) applying the
        // range function UDAF over (sample_ts, value).
        let udaf = make_range_udaf(func);
        let mut group_exprs: Vec<Expr> = vec![col("eval_ts")];
        for label in label_columns {
            group_exprs.push(col(label.as_str()));
        }
        let agg_expr = udaf
            .call(vec![col("sample_ts"), col("value")])
            .alias("value");
        let aggregated = LogicalPlanBuilder::from(filtered)
            .aggregate(group_exprs, vec![agg_expr])?
            .build()?;

        // Step 6: Filter out NULL results (range functions return None for
        // insufficient samples) and project to match the original output schema.
        let with_filter = LogicalPlanBuilder::from(aggregated)
            .filter(col("value").is_not_null())?
            .build()?;

        // Build the final projection, preserving the original schema's
        // field qualifiers so the optimizer's schema-equivalence check passes.
        let mut proj_exprs: Vec<Expr> = Vec::new();
        for (idx, field) in original_schema.fields().iter().enumerate() {
            let name = field.name();
            let qualifier = original_schema.iter().nth(idx).map(|(q, _)| q).unwrap();

            let expr = if name == "timestamp" {
                col("eval_ts")
            } else if name == "value" {
                col("value")
            } else if label_columns.contains(name) {
                col(name.as_str())
            } else {
                // Columns in the schema but not tracked as labels (shouldn't
                // normally happen, but handle gracefully with an empty literal).
                lit("")
            };

            proj_exprs.push(Expr::Alias(Alias::new(
                expr,
                qualifier.cloned(),
                name.as_str(),
            )));
        }

        let projected = LogicalPlanBuilder::from(with_filter)
            .project(proj_exprs)?
            .build()?;

        Ok(Transformed::yes(projected))
    }
}

/// Generate the list of evaluation timestamps from the node parameters.
fn generate_eval_timestamps(
    eval_ts_ns: Option<i64>,
    start_ns: i64,
    end_ns: i64,
    step_ns: i64,
) -> Vec<i64> {
    if let Some(ts) = eval_ts_ns {
        return vec![ts];
    }
    let mut timestamps = Vec::new();
    let mut t = start_ns;
    while t <= end_ns {
        timestamps.push(t);
        t += step_ns;
    }
    timestamps
}
