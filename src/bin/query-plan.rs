use std::sync::Arc;

use async_trait::async_trait;
use chrono::DateTime;
use clap::Parser;
use datafusion::catalog::TableProvider;
use datafusion::prelude::*;

use datafusion_promql::PromqlPlanner;
use datafusion_promql::datasource::{Matcher, MetricMeta, MetricSource, TableFormat};
use datafusion_promql::error::{PromqlError, Result};
use datafusion_promql::types::TimeRange;

#[derive(Parser)]
#[command(
    about = "Show DataFusion logical and physical plans for a PromQL query against a parquet file"
)]
struct Cli {
    /// Path to the parquet file
    #[arg(short, long)]
    file: String,

    /// The PromQL query
    query: String,

    /// Evaluation timestamp in milliseconds (omit for whole-range query)
    #[arg(short, long)]
    timestamp: Option<i64>,

    /// Show the logical plan (before optimization)
    #[arg(long)]
    logical: bool,

    /// Show the optimized logical plan
    #[arg(long)]
    optimized: bool,

    /// Show the physical plan
    #[arg(long)]
    physical: bool,
}

/// A metric source backed by a single parquet file.
struct ParquetSource {
    ctx: SessionContext,
}

impl ParquetSource {
    async fn new(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let ctx = SessionContext::new();
        // Register the parquet file so we can derive schemas.
        ctx.register_parquet("metrics", path, Default::default())
            .await?;
        Ok(Self { ctx })
    }
}

#[async_trait]
impl MetricSource for ParquetSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        let table = self
            .ctx
            .table_provider("metrics")
            .await
            .map_err(|e| PromqlError::DataSource(e.to_string()))?;
        Ok((table, TableFormat::Long))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        let table = self
            .ctx
            .table_provider("metrics")
            .await
            .map_err(|e| PromqlError::DataSource(e.to_string()))?;
        let schema = table.schema();

        // Derive label names from the schema: all Utf8 columns except __name__.
        let label_names: Vec<String> = schema
            .fields()
            .iter()
            .filter(|f| {
                f.data_type() == &arrow::datatypes::DataType::Utf8 && f.name() != "__name__"
            })
            .map(|f| f.name().clone())
            .collect();

        Ok(vec![MetricMeta {
            name: "unknown".into(),
            label_names,
            extra_columns: vec![],
        }])
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // If neither flag is specified, show all three.
    let (show_logical, show_optimized, show_physical) =
        if !cli.logical && !cli.optimized && !cli.physical {
            (true, true, true)
        } else {
            (cli.logical, cli.optimized, cli.physical)
        };

    let source = Arc::new(ParquetSource::new(&cli.file).await?);
    let planner = PromqlPlanner::new(source.clone());

    let logical_plan = if let Some(ts_ms) = cli.timestamp {
        let timestamp = DateTime::from_timestamp_millis(ts_ms)
            .ok_or_else(|| format!("invalid timestamp: {ts_ms}"))?;
        planner.instant_logical_plan(&cli.query, timestamp).await?
    } else {
        // Whole-range query: no timestamp, use plan_expr directly.
        let expr =
            promql_parser::parser::parse(&cli.query).map_err(|e| format!("parse error: {e}"))?;
        let time_range = TimeRange {
            start_ms: i64::MIN,
            end_ms: i64::MAX,
        };
        let params = datafusion_promql::plan::EvalParams {
            eval_ts_ms: None,
            start_ms: i64::MIN,
            end_ms: i64::MAX,
            step_ms: 1,
        };
        datafusion_promql::plan::plan_expr(&expr, source.as_ref(), time_range, params).await?
    };

    if show_logical {
        println!("=== Logical Plan ===");
        println!("{}", logical_plan.display_indent_schema());
        println!();
    }

    if show_optimized {
        let optimized_plan = planner.optimize_logical_plan(logical_plan.clone())?;
        println!("=== Optimized Logical Plan ===");
        println!("{}", optimized_plan.display_indent_schema());
        println!();
    }

    if show_physical {
        let physical_plan = planner.create_physical_plan(&logical_plan).await?;
        println!("=== Physical Plan ===");
        println!(
            "{}",
            datafusion::physical_plan::displayable(physical_plan.as_ref()).indent(true)
        );
    }

    Ok(())
}
