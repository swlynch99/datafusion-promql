pub mod datasource;
pub mod error;
pub mod types;

mod exec;
mod func;
mod node;
mod plan;

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion::prelude::*;

use crate::datasource::MetricSource;
use crate::error::{PromqlError, Result};
use crate::exec::PromqlExtensionPlanner;
use crate::plan::{EvalParams, plan_expr};
use crate::types::{InstantSample, Labels, QueryResult, RangeSamples, TimeRange};

/// Custom query planner that delegates to DefaultPhysicalPlanner with our
/// extension planner registered.
#[derive(Debug)]
struct PromqlQueryPlanner;

#[async_trait]
impl datafusion::execution::context::QueryPlanner for PromqlQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &datafusion::logical_expr::LogicalPlan,
        session_state: &datafusion::execution::SessionState,
    ) -> datafusion::common::Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        let planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(PromqlExtensionPlanner)]);
        planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

/// The main PromQL query engine.
///
/// Wraps a DataFusion `SessionContext` and a pluggable [`MetricSource`] backend.
pub struct PromqlEngine {
    ctx: SessionContext,
    source: Arc<dyn MetricSource>,
}

impl PromqlEngine {
    /// Create a new engine with the given metric source.
    pub fn new(source: Arc<dyn MetricSource>) -> Self {
        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_query_planner(Arc::new(PromqlQueryPlanner))
            .build();
        let ctx = SessionContext::new_with_state(state);

        Self { ctx, source }
    }

    /// Execute an instant query at a single timestamp.
    pub async fn instant_query(
        &self,
        query: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<QueryResult> {
        let expr = promql_parser::parser::parse(query).map_err(PromqlError::Parse)?;

        let ts_ms = timestamp.timestamp_millis();
        let time_range = TimeRange {
            start_ms: ts_ms,
            end_ms: ts_ms,
        };
        let params = EvalParams {
            eval_ts_ms: Some(ts_ms),
            start_ms: ts_ms,
            end_ms: ts_ms,
            step_ms: 1,
        };

        let logical_plan = plan_expr(&expr, self.source.as_ref(), time_range, params).await?;

        let df = self.ctx.execute_logical_plan(logical_plan).await?;
        let batches = df.collect().await?;

        batches_to_vector(&batches)
    }

    /// Execute a range query over `[start, end]` with step.
    pub async fn range_query(
        &self,
        query: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        step: std::time::Duration,
    ) -> Result<QueryResult> {
        let expr = promql_parser::parser::parse(query).map_err(PromqlError::Parse)?;

        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();
        let step_ms = step.as_millis() as i64;

        let time_range = TimeRange { start_ms, end_ms };
        let params = EvalParams {
            eval_ts_ms: None,
            start_ms,
            end_ms,
            step_ms,
        };

        let logical_plan = plan_expr(&expr, self.source.as_ref(), time_range, params).await?;

        let df = self.ctx.execute_logical_plan(logical_plan).await?;
        let batches = df.collect().await?;

        batches_to_matrix(&batches)
    }
}

/// Convert RecordBatches to `QueryResult::Vector` (instant query result).
fn batches_to_vector(batches: &[arrow::record_batch::RecordBatch]) -> Result<QueryResult> {
    let mut samples = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let (ts, val, labels) = extract_row(batch, row)?;
            samples.push(InstantSample {
                labels,
                timestamp_ms: ts,
                value: val,
            });
        }
    }

    Ok(QueryResult::Vector(samples))
}

/// Convert RecordBatches to `QueryResult::Matrix` (range query result).
///
/// Groups rows by their label set and collects `(timestamp, value)` pairs.
fn batches_to_matrix(batches: &[arrow::record_batch::RecordBatch]) -> Result<QueryResult> {
    let mut series_map: BTreeMap<Labels, Vec<(i64, f64)>> = BTreeMap::new();

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

/// Extract labels and sample data from a batch for a given row.
fn extract_row(batch: &arrow::record_batch::RecordBatch, row: usize) -> Result<(i64, f64, Labels)> {
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
        .downcast_ref::<arrow::array::Int64Array>()
        .ok_or_else(|| {
            PromqlError::Execution(datafusion::error::DataFusionError::Internal(
                "timestamp must be Int64".into(),
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
