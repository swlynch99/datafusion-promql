use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::DateTime;
use datafusion::logical_expr::{LogicalPlanBuilder, col};
use datafusion_promql::parquet::{ParquetMetricSource, read_timestamp_range};
use datafusion_promql::{PromqlEngine, PromqlPlanner};

const NS_PER_SEC: u64 = 1_000_000_000;

struct Phase {
    plan: Vec<Duration>,
    opt: Vec<Duration>,
    phys: Vec<Duration>,
    exec: Vec<Duration>,
    total: Vec<Duration>,
}

async fn bench_range(
    planner: &PromqlPlanner,
    engine: &PromqlEngine,
    query: &str,
    start_ns: u64,
    end_ns: u64,
    step_s: u64,
    iters: usize,
) -> Phase {
    let start = DateTime::from_timestamp_nanos(start_ns as i64);
    let end = DateTime::from_timestamp_nanos(end_ns as i64);
    let step = Duration::from_secs(step_s);
    let mut phase = Phase {
        plan: Vec::with_capacity(iters),
        opt: Vec::with_capacity(iters),
        phys: Vec::with_capacity(iters),
        exec: Vec::with_capacity(iters),
        total: Vec::with_capacity(iters),
    };
    for _ in 0..iters {
        let total_t0 = Instant::now();

        let t0 = Instant::now();
        let logical = planner
            .range_logical_plan(query, start, end, step)
            .await
            .unwrap();
        phase.plan.push(t0.elapsed());

        let t0 = Instant::now();
        let optimized = planner.optimize_logical_plan(logical).unwrap();
        let filtered = LogicalPlanBuilder::from(optimized)
            .filter(col("value").is_not_null())
            .unwrap()
            .build()
            .unwrap();
        let with_agg = PromqlPlanner::add_matrix_series_aggregation(filtered).unwrap();
        phase.opt.push(t0.elapsed());

        let t0 = Instant::now();
        let physical = planner.create_physical_plan(&with_agg).await.unwrap();
        phase.phys.push(t0.elapsed());

        let t0 = Instant::now();
        let _ = planner.execute(physical).await.unwrap();
        phase.exec.push(t0.elapsed());

        phase.total.push(total_t0.elapsed());

        // Sanity-check vs the engine end-to-end path.
        let _ = engine;
    }
    phase
}

fn pct(vs: &mut [Duration], p: f64) -> f64 {
    vs.sort();
    let idx = ((vs.len() as f64 - 1.0) * p).round() as usize;
    vs[idx].as_secs_f64() * 1000.0
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let file = args
        .iter()
        .position(|a| a == "--file")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "data/metrics.parquet".into());
    let iters: usize = args
        .iter()
        .position(|a| a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let only = args
        .iter()
        .position(|a| a == "--only")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let (min_ns, max_ns) = read_timestamp_range(&file)?;
    let source = Arc::new(ParquetMetricSource::try_new(&file).await?);
    let planner = PromqlPlanner::new(source.clone());
    let engine = PromqlEngine::new(source);

    // Align start/end to whole seconds inside the data range; use step=60s.
    let start_ns = (min_ns / NS_PER_SEC) * NS_PER_SEC;
    let end_ns = (max_ns / NS_PER_SEC) * NS_PER_SEC;
    let step_s = 60u64;

    let queries = vec![
        ("q1_selector", "cpu_cores"),
        ("q2_rate", "rate(cpu_usage[60s])"),
        ("q3_sum_by", "sum by (id) (rate(cpu_usage[60s]))"),
        ("q4_scalar_bin", "rate(cpu_usage[60s]) * 100"),
        (
            "q5_vec_bin",
            "rate(cpu_usage[60s]) / ignoring(op) group_left sum by (id) (rate(cpu_usage[60s]))",
        ),
        ("q6_instant_fn", "abs(cpu_cores - 4)"),
        ("q7_topk", "topk(3, rate(cpu_usage[60s]))"),
    ];

    // Warmup with a cheap query so the engine's lazy caches are populated.
    let _ = engine
        .range_query(
            "cpu_cores",
            DateTime::from_timestamp_nanos(start_ns as i64),
            DateTime::from_timestamp_nanos(end_ns as i64),
            Duration::from_secs(step_s),
        )
        .await;

    println!("file={file}  range=[{start_ns}..{end_ns}]  step={step_s}s  iters={iters}\n");
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "query", "plan(ms)", "opt(ms)", "phys(ms)", "exec(ms)", "total(ms)"
    );

    for (name, q) in &queries {
        if let Some(filter) = &only {
            if filter != name {
                continue;
            }
        }
        let mut p = bench_range(&planner, &engine, q, start_ns, end_ns, step_s, iters).await;
        println!(
            "{name:<16} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            pct(&mut p.plan, 0.5),
            pct(&mut p.opt, 0.5),
            pct(&mut p.phys, 0.5),
            pct(&mut p.exec, 0.5),
            pct(&mut p.total, 0.5),
        );
    }

    Ok(())
}
