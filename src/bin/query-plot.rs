use std::sync::Arc;

use chrono::DateTime;
use clap::Parser;

use datafusion_promql::PromqlEngine;
use datafusion_promql::parquet::ParquetMetricSource;
use datafusion_promql::types::QueryResult;

use textplots::{Chart, ColorPlot, Shape};

#[derive(Parser)]
#[command(about = "Execute a PromQL query against a parquet file and plot results in the terminal")]
struct Cli {
    /// Path to the parquet file
    #[arg(short, long)]
    file: String,

    /// The PromQL query
    query: String,

    /// Start timestamp in nanoseconds (for range queries)
    #[arg(long)]
    start: Option<i64>,

    /// End timestamp in nanoseconds (for range queries)
    #[arg(long)]
    end: Option<i64>,

    /// Step duration in seconds (default: 15)
    #[arg(long, default_value = "15")]
    step: u64,

    /// Evaluation timestamp in nanoseconds (for instant queries)
    #[arg(short, long)]
    timestamp: Option<i64>,

    /// Chart width in terminal columns (defaults to terminal width)
    #[arg(long)]
    width: Option<u32>,

    /// Chart height in terminal rows (defaults to terminal height minus 10)
    #[arg(long)]
    height: Option<u32>,
}

fn terminal_dimensions() -> (u32, u32) {
    let (w, h) = terminal_size::terminal_size()
        .map(|(w, h)| (w.0 as u32, h.0 as u32))
        .unwrap_or((120, 40));
    // Reserve some rows for the legend/labels printed below the chart.
    (w, h.saturating_sub(10).max(10))
}

/// Format a label set into a compact series name like `{instance="host1", job="node"}`.
fn format_labels(labels: &std::collections::BTreeMap<String, String>) -> String {
    if labels.is_empty() {
        return "{}".to_string();
    }
    let parts: Vec<String> = labels
        .iter()
        .filter(|(k, _)| k.as_str() != "__name__")
        .map(|(k, v)| format!("{k}=\"{v}\""))
        .collect();
    if parts.is_empty() {
        labels
            .get("__name__")
            .cloned()
            .unwrap_or_else(|| "{}".to_string())
    } else {
        format!("{{{}}}", parts.join(", "))
    }
}

/// Pick a color for series index `i` from a fixed palette.
fn series_color(i: usize) -> rgb::RGB8 {
    const PALETTE: &[rgb::RGB8] = &[
        rgb::RGB8::new(31, 119, 180),  // blue
        rgb::RGB8::new(255, 127, 14),  // orange
        rgb::RGB8::new(44, 160, 44),   // green
        rgb::RGB8::new(214, 39, 40),   // red
        rgb::RGB8::new(148, 103, 189), // purple
        rgb::RGB8::new(140, 86, 75),   // brown
        rgb::RGB8::new(227, 119, 194), // pink
        rgb::RGB8::new(127, 127, 127), // gray
        rgb::RGB8::new(188, 189, 34),  // olive
        rgb::RGB8::new(23, 190, 207),  // cyan
    ];
    PALETTE[i % PALETTE.len()]
}

fn plot_matrix(
    result: &QueryResult,
    query: &str,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let series = match result {
        QueryResult::Matrix(s) => s,
        _ => return Err("expected matrix result for range query".into()),
    };

    if series.is_empty() {
        return Err("query returned no data".into());
    }

    // Compute global min/max timestamps to use as x-axis bounds.
    let mut ts_min = i64::MAX;
    let mut ts_max = i64::MIN;

    for s in series {
        for &(ts, _) in &s.samples {
            ts_min = ts_min.min(ts);
            ts_max = ts_max.max(ts);
        }
    }

    // Convert to seconds offset from ts_min for f32 plotting.
    let ts_to_x = |ts: i64| -> f32 { ((ts - ts_min) as f64 / 1e9) as f32 };
    let x_max = ts_to_x(ts_max);

    println!("{query}");
    println!();

    // Build point data for all series.
    let all_points: Vec<Vec<(f32, f32)>> = series
        .iter()
        .map(|s| {
            s.samples
                .iter()
                .filter(|(_, v)| v.is_finite())
                .map(|&(ts, val)| (ts_to_x(ts), val as f32))
                .collect()
        })
        .collect();

    // Build shapes upfront so borrows live long enough.
    let shapes: Vec<Shape> = all_points.iter().map(|p| Shape::Lines(p)).collect();
    let colors: Vec<rgb::RGB8> = (0..shapes.len()).map(series_color).collect();

    let mut chart = Chart::new(width, height, 0.0, x_max);
    let chart_ref = &mut chart;
    // Chain all linecolorplot calls so borrow checker sees one contiguous borrow.
    let mut c = &mut *chart_ref;
    for (shape, &color) in shapes.iter().zip(colors.iter()) {
        c = c.linecolorplot(shape, color);
    }
    c.nice();

    // Print time range.
    let dt_min = DateTime::from_timestamp_nanos(ts_min);
    let dt_max = DateTime::from_timestamp_nanos(ts_max);
    println!(
        "  x: {:.1}s  [{} .. {}]",
        x_max,
        dt_min.format("%H:%M:%S"),
        dt_max.format("%H:%M:%S")
    );

    // Print legend.
    if series.len() > 1 {
        println!();
        for (i, s) in series.iter().enumerate() {
            let color = series_color(i);
            // Use ANSI true color to show the color swatch.
            let label = format_labels(&s.labels);
            println!(
                "  \x1b[38;2;{};{};{}m●\x1b[0m {}",
                color.r, color.g, color.b, label
            );
        }
    } else {
        let label = format_labels(&series[0].labels);
        if label != "{}" {
            println!("  {label}");
        }
    }

    Ok(())
}

fn plot_vector(
    result: &QueryResult,
    query: &str,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples = match result {
        QueryResult::Vector(s) => s,
        _ => return Err("expected vector result for instant query".into()),
    };

    if samples.is_empty() {
        return Err("query returned no data".into());
    }

    println!("{query}");
    println!();

    // For instant vectors, plot as points along x = index.
    let points: Vec<(f32, f32)> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| (i as f32, s.value as f32))
        .collect();

    let x_max = (samples.len() as f32).max(1.0);

    let shape = Shape::Points(&points);
    let color = series_color(0);
    Chart::new(width, height, -0.5, x_max - 0.5)
        .linecolorplot(&shape, color)
        .nice();

    // Print labels below.
    for (i, s) in samples.iter().enumerate() {
        let label = format_labels(&s.labels);
        println!("  [{i}] {label} = {}", s.value);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let (term_w, term_h) = terminal_dimensions();
    let width = cli.width.unwrap_or(term_w);
    let height = cli.height.unwrap_or(term_h);

    let source = Arc::new(ParquetMetricSource::try_new(&cli.file).await?);
    let engine = PromqlEngine::new(source);

    if let Some(ts_ns) = cli.timestamp {
        // Instant query.
        let ts = DateTime::from_timestamp_nanos(ts_ns);
        eprintln!("Executing instant query at {ts}...");
        let result = engine.instant_query(&cli.query, ts).await?;
        plot_vector(&result, &cli.query, width, height)?;
    } else if let (Some(start_ns), Some(end_ns)) = (cli.start, cli.end) {
        // Range query.
        let start = DateTime::from_timestamp_nanos(start_ns);
        let end = DateTime::from_timestamp_nanos(end_ns);
        let step = std::time::Duration::from_secs(cli.step);
        eprintln!("Executing range query [{start} .. {end}] step {step:?}...");
        let result = engine.range_query(&cli.query, start, end, step).await?;
        plot_matrix(&result, &cli.query, width, height)?;
    } else {
        return Err(
            "provide either --timestamp for instant queries, or --start and --end for range queries"
                .into(),
        );
    };

    Ok(())
}
