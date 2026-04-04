use std::sync::Arc;

use chrono::DateTime;
use clap::Parser;

use datafusion_promql::PromqlPlanner;
use datafusion_promql::parquet::ParquetMetricSource;
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

    /// Evaluation timestamp in nanoseconds (omit for whole-range query)
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

    let source = Arc::new(ParquetMetricSource::try_new(&cli.file).await?);
    let planner = PromqlPlanner::new(source.clone());

    let logical_plan = if let Some(ts_ns) = cli.timestamp {
        let timestamp = DateTime::from_timestamp_nanos(ts_ns);
        planner.instant_logical_plan(&cli.query, timestamp).await?
    } else {
        // Whole-range query: no timestamp, use plan_expr directly.
        let expr =
            promql_parser::parser::parse(&cli.query).map_err(|e| format!("parse error: {e}"))?;
        let time_range = TimeRange {
            start_ns: i64::MIN,
            end_ns: i64::MAX,
        };
        let params = datafusion_promql::plan::EvalParams {
            eval_ts_ns: None,
            start_ns: i64::MIN,
            end_ns: i64::MAX,
            step_ns: 1,
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
