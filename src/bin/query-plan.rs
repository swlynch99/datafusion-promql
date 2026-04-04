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

    /// Evaluation timestamp as unix seconds (for instant queries)
    #[arg(short, long, conflicts_with_all = ["start", "end", "step"])]
    timestamp: Option<i64>,

    /// Start timestamp as unix seconds (for range queries)
    #[arg(long, conflicts_with = "timestamp")]
    start: Option<i64>,

    /// End timestamp as unix seconds (for range queries)
    #[arg(long, conflicts_with = "timestamp")]
    end: Option<i64>,

    /// Step duration in seconds (for range queries, default: 1)
    #[arg(long, conflicts_with = "timestamp")]
    step: Option<u64>,

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

    const NS_PER_SEC: i64 = 1_000_000_000;

    let logical_plan = if let Some(ts) = cli.timestamp {
        let ts_ns = ts * NS_PER_SEC;
        let timestamp = DateTime::from_timestamp_nanos(ts_ns);
        planner.instant_logical_plan(&cli.query, timestamp).await?
    } else {
        let expr =
            promql_parser::parser::parse(&cli.query).map_err(|e| format!("parse error: {e}"))?;

        let start_ns = cli
            .start
            .ok_or("--start is required for range queries (or use --timestamp for instant)")?
            * NS_PER_SEC;
        let end_ns = cli
            .end
            .ok_or("--end is required for range queries (or use --timestamp for instant)")?
            * NS_PER_SEC;
        let step_ns = cli.step.unwrap_or(1) as i64 * NS_PER_SEC;

        let time_range = TimeRange {
            start_ns: Some(start_ns),
            end_ns: Some(end_ns),
        };
        let params = datafusion_promql::plan::EvalParams {
            eval_ts_ns: None,
            start_ns,
            end_ns,
            step_ns,
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
