mod instant_eval;
mod range_eval;

pub(crate) use instant_eval::InstantVectorExec;
pub(crate) use range_eval::RangeVectorExec;

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::error::Result;
use datafusion::execution::SessionState;
use datafusion::logical_expr::UserDefinedLogicalNode;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};

use crate::node::{InstantVectorEval, RangeVectorEval};

/// Extension planner that converts our custom logical nodes into physical plans.
pub(crate) struct PromqlExtensionPlanner;

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

        Ok(None)
    }
}
