use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use clap::Parser;

use datafusion_promql::parquet::ParquetMetricSource;
use datafusion_promql::types::QueryResult;
use datafusion_promql::PromqlEngine;

use plotters::prelude::*;

#[derive(Parser)]
#[command(about = "Execute a PromQL query against a parquet file and plot the results")]
struct Cli {
    /// Path to the parquet file
    #[arg(short, long)]
    file: String,

    /// The PromQL query
    query: String,

    /// Start timestamp in milliseconds (for range queries)
    #[arg(long)]
    start: Option<i64>,

    /// End timestamp in milliseconds (for range queries)
    #[arg(long)]
    end: Option<i64>,

    /// Step duration in seconds (default: 15)
    #[arg(long, default_value = "15")]
    step: u64,

    /// Evaluation timestamp in milliseconds (for instant queries)
    #[arg(short, long)]
    timestamp: Option<i64>,

    /// Output file path (PNG)
    #[arg(short, long, default_value = "plot.png")]
    output: String,

    /// Image width in pixels
    #[arg(long, default_value = "1024")]
    width: u32,

    /// Image height in pixels
    #[arg(long, default_value = "768")]
    height: u32,
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
fn series_color(i: usize) -> RGBColor {
    const PALETTE: &[RGBColor] = &[
        RGBColor(31, 119, 180),
        RGBColor(255, 127, 14),
        RGBColor(44, 160, 44),
        RGBColor(214, 39, 40),
        RGBColor(148, 103, 189),
        RGBColor(140, 86, 75),
        RGBColor(227, 119, 194),
        RGBColor(127, 127, 127),
        RGBColor(188, 189, 34),
        RGBColor(23, 190, 207),
    ];
    PALETTE[i % PALETTE.len()]
}

fn plot_matrix(
    result: &QueryResult,
    query: &str,
    output: &str,
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

    // Compute global min/max for axes.
    let mut ts_min = i64::MAX;
    let mut ts_max = i64::MIN;
    let mut val_min = f64::INFINITY;
    let mut val_max = f64::NEG_INFINITY;

    for s in series {
        for &(ts, val) in &s.samples {
            ts_min = ts_min.min(ts);
            ts_max = ts_max.max(ts);
            if val.is_finite() {
                val_min = val_min.min(val);
                val_max = val_max.max(val);
            }
        }
    }

    // Add a little vertical padding.
    let val_range = val_max - val_min;
    let padding = if val_range == 0.0 { 1.0 } else { val_range * 0.05 };
    val_min -= padding;
    val_max += padding;

    let dt_min = Utc.timestamp_millis_opt(ts_min).unwrap();
    let dt_max = Utc.timestamp_millis_opt(ts_max).unwrap();

    let root = BitMapBackend::new(output, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(query, ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(dt_min..dt_max, val_min..val_max)?;

    chart
        .configure_mesh()
        .x_label_formatter(&|dt| dt.format("%H:%M:%S").to_string())
        .x_desc("Time (UTC)")
        .y_desc("Value")
        .draw()?;

    for (i, s) in series.iter().enumerate() {
        let color = series_color(i);
        let label = format_labels(&s.labels);
        let points: Vec<(DateTime<Utc>, f64)> = s
            .samples
            .iter()
            .map(|&(ts, val)| (Utc.timestamp_millis_opt(ts).unwrap(), val))
            .collect();

        chart
            .draw_series(LineSeries::new(points.clone(), color.stroke_width(2)))?
            .label(&label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));
    }

    if series.len() > 1 {
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

fn plot_vector(
    result: &QueryResult,
    query: &str,
    output: &str,
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

    let labels: Vec<String> = samples.iter().map(|s| format_labels(&s.labels)).collect();
    let values: Vec<f64> = samples.iter().map(|s| s.value).collect();

    let val_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let val_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = val_max - val_min;
    let padding = if range == 0.0 { 1.0 } else { range * 0.1 };
    let y_min = (val_min - padding).min(0.0);
    let y_max = val_max + padding;

    let n = samples.len();

    let root = BitMapBackend::new(output, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(query, ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(80)
        .y_label_area_size(70)
        .build_cartesian_2d(0..n, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_label_formatter(&|idx| {
            labels.get(*idx).cloned().unwrap_or_default()
        })
        .x_labels(n.min(20))
        .y_desc("Value")
        .draw()?;

    chart.draw_series(
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let color = series_color(i);
                Rectangle::new([(i, 0.0f64.max(y_min)), (i + 1, v)], color.filled())
            }),
    )?;

    root.present()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let source = Arc::new(ParquetMetricSource::try_new(&cli.file).await?);
    let engine = PromqlEngine::new(source);

    let result = if let Some(ts_ms) = cli.timestamp {
        // Instant query.
        let ts = Utc
            .timestamp_millis_opt(ts_ms)
            .single()
            .ok_or_else(|| format!("invalid timestamp: {ts_ms}"))?;
        println!("Executing instant query at {ts}...");
        let r = engine.instant_query(&cli.query, ts).await?;
        println!("Plotting instant vector...");
        plot_vector(&r, &cli.query, &cli.output, cli.width, cli.height)?;
        r
    } else if let (Some(start_ms), Some(end_ms)) = (cli.start, cli.end) {
        // Range query.
        let start = Utc
            .timestamp_millis_opt(start_ms)
            .single()
            .ok_or_else(|| format!("invalid start timestamp: {start_ms}"))?;
        let end = Utc
            .timestamp_millis_opt(end_ms)
            .single()
            .ok_or_else(|| format!("invalid end timestamp: {end_ms}"))?;
        let step = std::time::Duration::from_secs(cli.step);
        println!("Executing range query [{start} .. {end}] step {step:?}...");
        let r = engine.range_query(&cli.query, start, end, step).await?;
        println!("Plotting range matrix...");
        plot_matrix(&r, &cli.query, &cli.output, cli.width, cli.height)?;
        r
    } else {
        return Err(
            "provide either --timestamp for instant queries, or --start and --end for range queries"
                .into(),
        );
    };

    // Print a summary.
    match &result {
        QueryResult::Vector(samples) => {
            println!("Result: {} sample(s)", samples.len());
            for s in samples {
                println!(
                    "  {} @ {} = {}",
                    format_labels(&s.labels),
                    s.timestamp_ms,
                    s.value
                );
            }
        }
        QueryResult::Matrix(series) => {
            println!("Result: {} series", series.len());
            for s in series {
                println!(
                    "  {} ({} samples)",
                    format_labels(&s.labels),
                    s.samples.len()
                );
            }
        }
        QueryResult::Scalar(val, ts) => {
            println!("Scalar: {val} @ {ts}");
        }
        QueryResult::String(val, ts) => {
            println!("String: {val} @ {ts}");
        }
    }

    println!("Plot written to {}", cli.output);
    Ok(())
}
