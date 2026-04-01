pub mod datasource;
pub mod error;
pub mod types;

mod exec;
mod node;
mod plan;

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion::prelude::*;

use crate::datasource::MetricSource;
use crate::error::{PromqlError, Result};
use crate::exec::PromqlExtensionPlanner;
use crate::plan::plan_expr;
use crate::types::{InstantSample, Labels, QueryResult, TimeRange};

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
        let planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            PromqlExtensionPlanner,
        )]);
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
        let expr = promql_parser::parser::parse(query)
            .map_err(PromqlError::Parse)?;

        let ts_ms = timestamp.timestamp_millis();
        let time_range = TimeRange {
            start_ms: ts_ms,
            end_ms: ts_ms,
        };

        let logical_plan = plan_expr(&expr, self.source.as_ref(), time_range, Some(ts_ms)).await?;

        let df = self.ctx.execute_logical_plan(logical_plan).await?;
        let batches = df.collect().await?;

        // Convert RecordBatches to QueryResult::Vector.
        let mut samples = Vec::new();
        for batch in &batches {
            let ts_col = batch
                .column_by_name("timestamp")
                .ok_or_else(|| PromqlError::Execution(
                    datafusion::error::DataFusionError::Internal("missing timestamp column".into()),
                ))?;
            let val_col = batch
                .column_by_name("value")
                .ok_or_else(|| PromqlError::Execution(
                    datafusion::error::DataFusionError::Internal("missing value column".into()),
                ))?;

            let ts_arr = ts_col
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| PromqlError::Execution(
                    datafusion::error::DataFusionError::Internal("timestamp must be Int64".into()),
                ))?;
            let val_arr = val_col
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| PromqlError::Execution(
                    datafusion::error::DataFusionError::Internal("value must be Float64".into()),
                ))?;

            // Collect label columns (everything except timestamp and value).
            let schema = batch.schema();
            let label_col_names: Vec<String> = schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| n != "timestamp" && n != "value")
                .collect();

            let label_arrays: Vec<(&str, &arrow::array::StringArray)> = label_col_names
                .iter()
                .map(|name| {
                    let col = batch.column_by_name(name).unwrap();
                    let arr = col
                        .as_any()
                        .downcast_ref::<arrow::array::StringArray>()
                        .unwrap();
                    (name.as_str(), arr)
                })
                .collect();

            for row in 0..batch.num_rows() {
                let mut labels = Labels::new();
                for &(name, arr) in &label_arrays {
                    let val = arr.value(row);
                    if !val.is_empty() {
                        labels.insert(name.to_string(), val.to_string());
                    }
                }

                samples.push(InstantSample {
                    labels,
                    timestamp_ms: ts_arr.value(row),
                    value: val_arr.value(row),
                });
            }
        }

        Ok(QueryResult::Vector(samples))
    }

    /// Execute a range query over `[start, end]` with step.
    pub async fn range_query(
        &self,
        _query: &str,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
        _step: std::time::Duration,
    ) -> Result<QueryResult> {
        Err(PromqlError::NotImplemented(
            "range_query is not yet implemented".into(),
        ))
    }
}
