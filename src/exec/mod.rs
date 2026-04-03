mod aggregate_eval;
mod binary_eval;
mod instant_eval;
mod instant_func_eval;
mod range_eval;

pub(crate) use aggregate_eval::AggregateExec;
pub(crate) use binary_eval::{BinaryExec, ScalarBinaryExec};
pub(crate) use instant_eval::InstantVectorExec;
pub(crate) use instant_func_eval::InstantFuncExec;
pub(crate) use range_eval::RangeVectorExec;

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::error::Result;
use datafusion::execution::SessionState;
use datafusion::logical_expr::UserDefinedLogicalNode;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};

use crate::node::{
    AggregateEval, BinaryEval, InstantFuncEval, InstantVectorEval, RangeVectorEval,
    ScalarBinaryEval,
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
            let child = Arc::clone(&physical_inputs[0]);
            let exec = InstantVectorExec::new(
                child,
                eval.eval_ts_ms,
                eval.start_ms,
                eval.end_ms,
                eval.step_ms,
                eval.lookback_ms,
                eval.label_columns.clone(),
            );
            return Ok(Some(Arc::new(exec)));
        }

        if let Some(eval) = node.as_any().downcast_ref::<RangeVectorEval>() {
            let child = Arc::clone(&physical_inputs[0]);
            let exec = RangeVectorExec::new(
                child,
                eval.range_ms,
                eval.func,
                eval.eval_ts_ms,
                eval.start_ms,
                eval.end_ms,
                eval.step_ms,
                eval.label_columns.clone(),
            );
            return Ok(Some(Arc::new(exec)));
        }

        if let Some(eval) = node.as_any().downcast_ref::<AggregateEval>() {
            let child = Arc::clone(&physical_inputs[0]);
            let exec = AggregateExec::new(child, eval.func, eval.grouping_labels.clone());
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

        if let Some(eval) = node.as_any().downcast_ref::<InstantFuncEval>() {
            let child = Arc::clone(&physical_inputs[0]);
            let output_schema: arrow::datatypes::SchemaRef =
                Arc::new(eval.output_schema.as_arrow().clone());
            let exec = InstantFuncExec::new(child, eval.func, output_schema);
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
