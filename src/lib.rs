pub mod datasource;
pub mod error;
pub mod types;

pub mod exec;
mod func;
pub mod node;
mod normalize;
pub mod opt;
pub mod plan;

#[cfg(feature = "parquet")]
pub mod parquet;

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion::prelude::*;

use crate::datasource::MetricSource;
use crate::error::{PromqlError, Result};
use crate::exec::PromqlExtensionPlanner;
use crate::plan::{EvalParams, plan_expr};
use crate::types::{InstantSample, Labels, QueryResult, RangeSamples, TimeRange};

/// A step-by-step PromQL planner that exposes each stage of the query pipeline.
///
/// The typical usage is:
///
/// ```text
/// let logical  = planner.instant_logical_plan(query, ts)?;
/// let logical  = planner.optimize_logical_plan(logical)?;
/// let physical = planner.create_physical_plan(&logical).await?;
/// // optionally: let physical = planner.optimize_physical_plan(physical)?;
/// let batches  = planner.execute(physical).await?;
/// let result   = PromqlPlanner::batches_to_vector(&batches)?;
/// ```
pub struct PromqlPlanner {
    ctx: SessionContext,
    source: Arc<dyn MetricSource>,
}

impl PromqlPlanner {
    /// Create a new planner with the given metric source.
    pub fn new(source: Arc<dyn MetricSource>) -> Self {
        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_optimizer_rule(Arc::new(crate::opt::logical::InstantFuncToProjection))
            .with_optimizer_rule(Arc::new(crate::opt::logical::DateTimeFuncToProjection))
            .with_optimizer_rule(Arc::new(crate::opt::logical::RangeVectorToAggregation))
            .with_optimizer_rule(Arc::new(crate::opt::logical::LiftConstantProjections))
            .with_optimizer_rule(Arc::new(crate::opt::logical::RemoveNoopProjections))
            .build();
        let ctx = SessionContext::new_with_state(state);
        Self { ctx, source }
    }

    /// Create an unoptimized logical plan for an instant query at a single timestamp.
    pub async fn instant_logical_plan(
        &self,
        query: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<datafusion::logical_expr::LogicalPlan> {
        let expr = promql_parser::parser::parse(query).map_err(PromqlError::Parse)?;
        let ts_ns = timestamp
            .timestamp_nanos_opt()
            .expect("timestamp out of range for nanoseconds") as u64;
        let time_range = TimeRange {
            start_ns: Some(ts_ns),
            end_ns: Some(ts_ns),
        };
        let params = EvalParams {
            eval_ts_ns: Some(ts_ns),
            start_ns: ts_ns,
            end_ns: ts_ns,
            step_ns: 1,
        };
        plan_expr(&expr, self.source.as_ref(), time_range, params).await
    }

    /// Create an unoptimized logical plan for a range query over `[start, end]` with step.
    pub async fn range_logical_plan(
        &self,
        query: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        step: std::time::Duration,
    ) -> Result<datafusion::logical_expr::LogicalPlan> {
        let expr = promql_parser::parser::parse(query).map_err(PromqlError::Parse)?;
        let start_ns = start
            .timestamp_nanos_opt()
            .expect("start timestamp out of range for nanoseconds") as u64;
        let end_ns = end
            .timestamp_nanos_opt()
            .expect("end timestamp out of range for nanoseconds") as u64;
        let step_ns = step.as_nanos() as u64;
        let time_range = TimeRange {
            start_ns: Some(start_ns),
            end_ns: Some(end_ns),
        };
        let params = EvalParams {
            eval_ts_ns: None,
            start_ns,
            end_ns,
            step_ns,
        };
        plan_expr(&expr, self.source.as_ref(), time_range, params).await
    }

    /// Apply DataFusion's logical optimizers to the plan.
    pub fn optimize_logical_plan(
        &self,
        plan: datafusion::logical_expr::LogicalPlan,
    ) -> Result<datafusion::logical_expr::LogicalPlan> {
        self.ctx
            .state()
            .optimize(&plan)
            .map_err(PromqlError::Execution)
    }

    /// Convert an optimized logical plan into a physical execution plan.
    ///
    /// The returned plan has already had DataFusion's physical optimizer applied.
    /// Use [`optimize_physical_plan`](Self::optimize_physical_plan) to re-apply
    /// physical optimization after any manual modifications to the plan.
    pub async fn create_physical_plan(
        &self,
        logical_plan: &datafusion::logical_expr::LogicalPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(PromqlExtensionPlanner)]);
        planner
            .create_physical_plan(logical_plan, &self.ctx.state())
            .await
            .map_err(PromqlError::Execution)
    }

    /// Apply DataFusion's physical optimizers to a physical plan.
    ///
    /// [`create_physical_plan`](Self::create_physical_plan) already applies
    /// physical optimization, so this method is only needed after manually
    /// modifying the plan produced by that method.
    pub fn optimize_physical_plan(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(PromqlExtensionPlanner)]);
        planner
            .optimize_physical_plan(plan, &self.ctx.state(), |_, _| {})
            .map_err(PromqlError::Execution)
    }

    /// Execute a physical plan and collect all result batches.
    pub async fn execute(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        collect(plan, self.ctx.task_ctx())
            .await
            .map_err(PromqlError::Execution)
    }

    /// Convert collected batches from an instant query into a [`QueryResult::Vector`].
    pub fn batches_to_vector(batches: &[arrow::record_batch::RecordBatch]) -> Result<QueryResult> {
        let mut samples = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                let (ts, val, labels) = extract_row(batch, row)?;
                samples.push(InstantSample {
                    labels,
                    timestamp_ns: ts,
                    value: val,
                });
            }
        }
        Ok(QueryResult::Vector(samples))
    }

    /// Convert collected batches from a range query into a [`QueryResult::Matrix`].
    ///
    /// Rows are grouped by their label set and collected as `(timestamp, value)` pairs.
    pub fn batches_to_matrix(batches: &[arrow::record_batch::RecordBatch]) -> Result<QueryResult> {
        let mut series_map: BTreeMap<Labels, Vec<(u64, f64)>> = BTreeMap::new();

        for batch in batches {
            for row in 0..batch.num_rows() {
                let (ts, val, labels) = extract_row(batch, row)?;
                series_map.entry(labels).or_default().push((ts, val));
            }
        }

        let result: Vec<RangeSamples> = series_map
            .into_iter()
            .map(|(labels, mut samples)| {
                samples.sort_by_key(|(ts, _)| *ts);
                RangeSamples { labels, samples }
            })
            .collect();

        Ok(QueryResult::Matrix(result))
    }
}

/// The main PromQL query engine.
///
/// Wraps a [`PromqlPlanner`] and provides a simple high-level API for executing
/// instant and range queries end-to-end.
pub struct PromqlEngine {
    planner: PromqlPlanner,
}

impl PromqlEngine {
    /// Create a new engine with the given metric source.
    pub fn new(source: Arc<dyn MetricSource>) -> Self {
        Self {
            planner: PromqlPlanner::new(source),
        }
    }

    /// Execute an instant query at a single timestamp.
    pub async fn instant_query(
        &self,
        query: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<QueryResult> {
        let logical = self.planner.instant_logical_plan(query, timestamp).await?;
        let optimized = self.planner.optimize_logical_plan(logical)?;
        let physical = self.planner.create_physical_plan(&optimized).await?;
        let batches = self.planner.execute(physical).await?;
        PromqlPlanner::batches_to_vector(&batches)
    }

    /// Execute a range query over `[start, end]` with step.
    pub async fn range_query(
        &self,
        query: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        step: std::time::Duration,
    ) -> Result<QueryResult> {
        let logical = self
            .planner
            .range_logical_plan(query, start, end, step)
            .await?;
        let optimized = self.planner.optimize_logical_plan(logical)?;
        let physical = self.planner.create_physical_plan(&optimized).await?;
        let batches = self.planner.execute(physical).await?;
        PromqlPlanner::batches_to_matrix(&batches)
    }
}

/// Extract labels and sample data from a batch for a given row.
fn extract_row(batch: &arrow::record_batch::RecordBatch, row: usize) -> Result<(u64, f64, Labels)> {
    let ts_col = batch.column_by_name("timestamp").ok_or_else(|| {
        PromqlError::Execution(datafusion::error::DataFusionError::Internal(
            "missing timestamp column".into(),
        ))
    })?;
    let val_col = batch.column_by_name("value").ok_or_else(|| {
        PromqlError::Execution(datafusion::error::DataFusionError::Internal(
            "missing value column".into(),
        ))
    })?;

    let ts_arr = ts_col
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .ok_or_else(|| {
            PromqlError::Execution(datafusion::error::DataFusionError::Internal(
                "timestamp must be UInt64".into(),
            ))
        })?;
    let val_arr = val_col
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| {
            PromqlError::Execution(datafusion::error::DataFusionError::Internal(
                "value must be Float64".into(),
            ))
        })?;

    let schema = batch.schema();
    let mut labels = Labels::new();
    for field in schema.fields() {
        let name = field.name();
        if name == "timestamp" || name == "value" {
            continue;
        }
        let col = batch.column_by_name(name).unwrap();
        let arr = col
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let val = arr.value(row);
        if !val.is_empty() {
            labels.insert(name.to_string(), val.to_string());
        }
    }

    Ok((ts_arr.value(row), val_arr.value(row), labels))
}
