use std::sync::Arc;

use async_trait::async_trait;
use clap::Parser;
use datafusion::catalog::TableProvider;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion::prelude::*;

use datafusion_promql::datasource::{Matcher, MetricMeta, MetricSource, TableFormat};
use datafusion_promql::error::{PromqlError, Result};
use datafusion_promql::types::TimeRange;

#[derive(Parser)]
#[command(about = "Show DataFusion logical and physical plans for a PromQL query against a parquet file")]
struct Cli {
    /// Path to the parquet file
    #[arg(short, long)]
    file: String,

    /// The PromQL query
    query: String,

    /// Evaluation timestamp in milliseconds (default: 0)
    #[arg(short, long, default_value_t = 0)]
    timestamp: i64,

    /// Show the logical plan
    #[arg(long)]
    logical: bool,

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

    // If neither flag is specified, show both.
    let (show_logical, show_physical) = if !cli.logical && !cli.physical {
        (true, true)
    } else {
        (cli.logical, cli.physical)
    };

    let source = ParquetSource::new(&cli.file).await?;
    let source = Arc::new(source);

    // Parse and plan
    let expr =
        promql_parser::parser::parse(&cli.query).map_err(|e| format!("parse error: {e}"))?;

    let ts_ms = cli.timestamp;
    let time_range = TimeRange {
        start_ms: ts_ms,
        end_ms: ts_ms,
    };
    let params = datafusion_promql::plan::EvalParams {
        eval_ts_ms: Some(ts_ms),
        start_ms: ts_ms,
        end_ms: ts_ms,
        step_ms: 1,
    };

    let logical_plan =
        datafusion_promql::plan::plan_expr(&expr, source.as_ref(), time_range, params).await?;

    if show_logical {
        println!("=== Logical Plan ===");
        println!("{}", logical_plan.display_indent_schema());
        println!();
    }

    if show_physical {
        // Build a session with the PromQL extension planner.
        let state = SessionStateBuilder::new()
            .with_default_features()
            .build();
        let planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            datafusion_promql::exec::PromqlExtensionPlanner,
        )]);
        let physical_plan = planner
            .create_physical_plan(&logical_plan, &state)
            .await?;

        println!("=== Physical Plan ===");
        println!(
            "{}",
            datafusion::physical_plan::displayable(physical_plan.as_ref())
                .indent(true)
        );
    }

    Ok(())
}
