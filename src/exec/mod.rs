mod binary_eval;
mod instant_eval;
mod range_eval;
mod range_func_eval;
mod step_eval;

pub(crate) use binary_eval::{BinaryExec, ScalarBinaryExec};
pub(crate) use instant_eval::InstantVectorExec;
pub(crate) use range_eval::RangeVectorExec;
pub(crate) use range_func_eval::RangeFunctionExec;
pub(crate) use step_eval::StepVectorExec;

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::error::Result;
use datafusion::execution::SessionState;
use datafusion::logical_expr::UserDefinedLogicalNode;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};

use crate::node::{
    BinaryEval, InstantVectorEval, RangeFunctionEval, RangeVectorEval, ScalarBinaryEval,
    StepVectorEval,
};

/// Extension planner that converts our custom logical nodes into physical plans.
pub struct PromqlExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for PromqlExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&datafusion::logical_expr::LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(eval) = node.as_any().downcast_ref::<InstantVectorEval>() {
            let child = coalesce_if_needed(Arc::clone(&physical_inputs[0]));
            let exec = InstantVectorExec::new(
                child,
                eval.timestamp_ns,
                eval.lookback_ns,
                eval.offset_ns,
                eval.label_columns.clone(),
            );
            return Ok(Some(Arc::new(exec)));
        }

        if let Some(eval) = node.as_any().downcast_ref::<StepVectorEval>() {
            let child = coalesce_if_needed(Arc::clone(&physical_inputs[0]));
            let exec = StepVectorExec::new(
                child,
                eval.start_ns,
                eval.end_ns,
                eval.step_ns,
                eval.lookback_ns,
                eval.offset_ns,
                eval.label_columns.clone(),
            );
            return Ok(Some(Arc::new(exec)));
        }

        if let Some(eval) = node.as_any().downcast_ref::<RangeVectorEval>() {
            let child = coalesce_if_needed(Arc::clone(&physical_inputs[0]));
            let exec = RangeVectorExec::new(
                child,
                eval.range_ns,
                eval.eval_ts_ns,
                eval.start_ns,
                eval.end_ns,
                eval.step_ns,
                eval.offset_ns,
                eval.label_columns.clone(),
            );
            return Ok(Some(Arc::new(exec)));
        }

        if let Some(eval) = node.as_any().downcast_ref::<RangeFunctionEval>() {
            let child = coalesce_if_needed(Arc::clone(&physical_inputs[0]));
            // Extract label columns from the output schema (everything except
            // timestamp and value).
            let label_columns: Vec<String> = eval
                .output_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| n != "timestamp" && n != "value")
                .collect();
            let exec = RangeFunctionExec::new(child, eval.func, label_columns);
            return Ok(Some(Arc::new(exec)));
        }

        if let Some(eval) = node.as_any().downcast_ref::<BinaryEval>() {
            let lhs = Arc::clone(&physical_inputs[0]);
            let rhs = Arc::clone(&physical_inputs[1]);
            // Convert the logical node's DFSchemaRef to a physical SchemaRef.
            let output_schema: arrow::datatypes::SchemaRef =
                Arc::new(eval.output_schema.as_arrow().clone());
            let exec = BinaryExec::new(
                lhs,
                rhs,
                eval.op,
                eval.return_bool,
                eval.matching.clone(),
                output_schema,
            );
            return Ok(Some(Arc::new(exec)));
        }

        if let Some(eval) = node.as_any().downcast_ref::<ScalarBinaryEval>() {
            let child = Arc::clone(&physical_inputs[0]);
            let output_schema: arrow::datatypes::SchemaRef =
                Arc::new(eval.output_schema.as_arrow().clone());
            let exec = ScalarBinaryExec::new(
                child,
                eval.scalar_value,
                eval.op,
                eval.scalar_is_lhs,
                eval.return_bool,
                output_schema,
            );
            return Ok(Some(Arc::new(exec)));
        }

        Ok(None)
    }
}

/// Wrap `plan` in a [`CoalescePartitionsExec`] if it has more than one
/// partition, so downstream single-partition exec nodes (InstantVectorExec,
/// RangeVectorExec) see all rows in a single `execute(0)` call.
fn coalesce_if_needed(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.output_partitioning().partition_count() > 1 {
        Arc::new(CoalescePartitionsExec::new(plan))
    } else {
        plan
    }
}
