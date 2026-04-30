//! A micro-benchmark that profiles every stage of the PromQL pipeline.
//!
//! Stages reported (median across `--iters` runs):
//!   parse   — promql_parser lex/parse
//!   plan    — translate AST → DataFusion LogicalPlan
//!   opt     — logical optimizer (includes the custom PromQL rules)
//!   phys    — logical → physical + physical optimizer
//!   exec    — stream the plan to completion
//!
//! With `--profile`, the tool runs each query once more after the timing
//! loop, walks the final physical plan, and reports per-node
//! elapsed_compute / output_rows plus the plan tree. Only DataFusion's
//! built-in operators record metrics; the crate's custom `*Exec` nodes
//! (InstantVectorExec, RangeFuncExec, ...) show `elapsed=?` so we can still
//! account their share of total `exec` time.
//!
//! The query set mixes the shapes from the original bench with harder
//! patterns: deep instant-function stacks, nested aggregations,
//! group_left / group_right many-to-one binops, label_replace-driven
//! relabelling, ratios between rate()s, and quantile_over_time on a wide
//! table.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::DateTime;
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder, col};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{ExecutionPlanVisitor, accept};
use datafusion_promql::parquet::{ParquetMetricSource, read_timestamp_range};
use datafusion_promql::{PromqlEngine, PromqlPlanner};

const NS_PER_SEC: u64 = 1_000_000_000;

/// All timing data we collect for a single query across N iterations.
struct Phase {
    parse: Vec<Duration>,
    plan: Vec<Duration>,
    opt: Vec<Duration>,
    phys: Vec<Duration>,
    exec: Vec<Duration>,
    total: Vec<Duration>,
    /// Node counts, captured once (they're deterministic across runs).
    logical_nodes: usize,
    optimized_nodes: usize,
    physical_nodes: usize,
    result_rows: usize,
}

fn logical_node_count(plan: &LogicalPlan) -> usize {
    1 + plan
        .inputs()
        .iter()
        .map(|p| logical_node_count(p))
        .sum::<usize>()
}

fn physical_node_count(plan: &dyn ExecutionPlan) -> usize {
    1 + plan
        .children()
        .iter()
        .map(|p| physical_node_count(p.as_ref()))
        .sum::<usize>()
}

async fn bench_range(
    planner: &PromqlPlanner,
    query: &str,
    start_ns: u64,
    end_ns: u64,
    step_s: u64,
    iters: usize,
) -> Result<Phase, Box<dyn std::error::Error>> {
    let start = DateTime::from_timestamp_nanos(start_ns as i64);
    let end = DateTime::from_timestamp_nanos(end_ns as i64);
    let step = Duration::from_secs(step_s);
    let mut phase = Phase {
        parse: Vec::with_capacity(iters),
        plan: Vec::with_capacity(iters),
        opt: Vec::with_capacity(iters),
        phys: Vec::with_capacity(iters),
        exec: Vec::with_capacity(iters),
        total: Vec::with_capacity(iters),
        logical_nodes: 0,
        optimized_nodes: 0,
        physical_nodes: 0,
        result_rows: 0,
    };
    for i in 0..iters {
        let total_t0 = Instant::now();

        // Parse in isolation so we can separate AST construction from planning.
        let t0 = Instant::now();
        promql_parser::parser::parse(query).map_err(|e| format!("parse: {e}"))?;
        phase.parse.push(t0.elapsed());

        let t0 = Instant::now();
        let logical = planner.range_logical_plan(query, start, end, step).await?;
        phase.plan.push(t0.elapsed());
        if i == 0 {
            phase.logical_nodes = logical_node_count(&logical);
        }

        let t0 = Instant::now();
        let optimized = planner.optimize_logical_plan(logical)?;
        let filtered = LogicalPlanBuilder::from(optimized)
            .filter(col("value").is_not_null())?
            .build()?;
        let with_agg = PromqlPlanner::add_matrix_series_aggregation(filtered)?;
        phase.opt.push(t0.elapsed());
        if i == 0 {
            phase.optimized_nodes = logical_node_count(&with_agg);
        }

        let t0 = Instant::now();
        let physical = planner.create_physical_plan(&with_agg).await?;
        phase.phys.push(t0.elapsed());
        if i == 0 {
            phase.physical_nodes = physical_node_count(physical.as_ref());
        }

        let t0 = Instant::now();
        let batches = planner.execute(physical).await?;
        phase.exec.push(t0.elapsed());
        if i == 0 {
            phase.result_rows = batches.iter().map(|b| b.num_rows()).sum();
        }

        phase.total.push(total_t0.elapsed());
    }
    Ok(phase)
}

fn pct(vs: &mut [Duration], p: f64) -> f64 {
    vs.sort();
    let idx = ((vs.len() as f64 - 1.0) * p).round() as usize;
    vs[idx].as_secs_f64() * 1000.0
}

/// Collect per-node metrics from an executed physical plan and print a
/// flat, sorted-by-elapsed-compute summary plus the tree with metrics inline.
struct NodeStat {
    name: String,
    elapsed_ns: u64,
    output_rows: u64,
    has_metrics: bool,
}

struct MetricCollector {
    nodes: Vec<NodeStat>,
}

impl ExecutionPlanVisitor for MetricCollector {
    type Error = datafusion::error::DataFusionError;
    fn pre_visit(&mut self, plan: &dyn ExecutionPlan) -> Result<bool, Self::Error> {
        let name = plan.name().to_string();
        let (elapsed_ns, output_rows, has_metrics) = match plan.metrics() {
            Some(ms) => {
                let agg = ms.aggregate_by_name();
                let mut elapsed = 0u64;
                let mut rows = 0u64;
                let mut saw = false;
                for m in agg.iter() {
                    match m.value() {
                        MetricValue::ElapsedCompute(t) => {
                            elapsed = elapsed.saturating_add(t.value() as u64);
                            saw = true;
                        }
                        MetricValue::OutputRows(c) => {
                            rows = rows.saturating_add(c.value() as u64);
                            saw = true;
                        }
                        _ => {}
                    }
                }
                (elapsed, rows, saw)
            }
            None => (0, 0, false),
        };
        self.nodes.push(NodeStat {
            name,
            elapsed_ns,
            output_rows,
            has_metrics,
        });
        Ok(true)
    }
}

async fn profile_one(
    planner: &PromqlPlanner,
    name: &str,
    query: &str,
    start_ns: u64,
    end_ns: u64,
    step_s: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = DateTime::from_timestamp_nanos(start_ns as i64);
    let end = DateTime::from_timestamp_nanos(end_ns as i64);
    let step = Duration::from_secs(step_s);

    let logical = planner.range_logical_plan(query, start, end, step).await?;
    let optimized = planner.optimize_logical_plan(logical)?;
    let filtered = LogicalPlanBuilder::from(optimized)
        .filter(col("value").is_not_null())?
        .build()?;
    let with_agg = PromqlPlanner::add_matrix_series_aggregation(filtered)?;
    let physical = planner.create_physical_plan(&with_agg).await?;

    let t0 = Instant::now();
    let _ = planner.execute(Arc::clone(&physical)).await?;
    let exec_elapsed = t0.elapsed();

    // Walk the executed plan to harvest metrics.
    let mut collector = MetricCollector { nodes: Vec::new() };
    accept(physical.as_ref(), &mut collector)?;

    // Rollup by operator name.
    let mut rollup: BTreeMap<String, (u64, u64, usize, usize)> = BTreeMap::new();
    let mut accounted_ns: u64 = 0;
    let mut missing_metric_nodes = 0usize;
    for n in &collector.nodes {
        let e = rollup.entry(n.name.clone()).or_insert((0, 0, 0, 0));
        e.0 += n.elapsed_ns;
        e.1 += n.output_rows;
        e.2 += 1;
        if !n.has_metrics {
            e.3 += 1;
            missing_metric_nodes += 1;
        }
        accounted_ns += n.elapsed_ns;
    }

    println!("\n=== PROFILE {name}: {query} ===");
    println!(
        "wall exec = {:.2} ms   sum(elapsed_compute) = {:.2} ms",
        exec_elapsed.as_secs_f64() * 1000.0,
        accounted_ns as f64 / 1e6,
    );
    if missing_metric_nodes > 0 {
        println!(
            "note: {missing_metric_nodes} node(s) do not report metrics (custom PromQL execs). \
             Their cost is visible only in the wall time delta.",
        );
    }

    println!(
        "\n{:<40} {:>6} {:>6} {:>14} {:>14}",
        "operator", "count", "nometr", "elapsed(ms)", "output_rows"
    );
    let mut rows: Vec<_> = rollup.into_iter().collect();
    rows.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    for (op, (ns, rows_out, count, nometr)) in rows {
        println!(
            "{:<40} {:>6} {:>6} {:>14.3} {:>14}",
            op,
            count,
            nometr,
            ns as f64 / 1e6,
            rows_out
        );
    }

    println!("\nphysical plan (metrics inline):");
    println!(
        "{}",
        DisplayableExecutionPlan::with_metrics(physical.as_ref()).indent(false)
    );
    Ok(())
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn flag_present(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let file = arg_value(&args, "--file").unwrap_or_else(|| "data/metrics.parquet".into());
    let iters: usize = arg_value(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let step_s: u64 = arg_value(&args, "--step")
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let only = arg_value(&args, "--only");
    let profile = flag_present(&args, "--profile");

    let (min_ns, max_ns) = read_timestamp_range(&file)?;
    let source = Arc::new(ParquetMetricSource::try_new(&file).await?);
    let planner = PromqlPlanner::new(source.clone());
    let engine = PromqlEngine::new(source);

    let start_ns = (min_ns / NS_PER_SEC) * NS_PER_SEC;
    let end_ns = (max_ns / NS_PER_SEC) * NS_PER_SEC;

    let queries: Vec<(&str, &str)> = vec![
        ("q01_selector", "cpu_cores"),
        ("q02_rate", "rate(cpu_usage[60s])"),
        ("q03_irate", "irate(cpu_usage[60s])"),
        ("q04_increase", "increase(cpu_usage[60s])"),
        ("q05_delta", "delta(cpu_usage[60s])"),
        ("q06_idelta", "idelta(cpu_usage[60s])"),
        ("q07_deriv", "deriv(cpu_usage[60s])"),
        ("q08_predict_linear", "predict_linear(cpu_usage[60s], 30)"),
        ("q09_sum_over_time", "sum_over_time(cpu_usage[60s])"),
        (
            "q10_avg_over_time_wide",
            "avg_over_time(cgroup_cpu_usage[60s])",
        ),
        ("q11_max_over_time", "max_over_time(cpu_usage[60s])"),
        ("q12_stddev_over_time", "stddev_over_time(cpu_usage[60s])"),
        (
            "q13_quantile_over_time",
            "quantile_over_time(0.95, cpu_usage[60s])",
        ),
        ("q14_last_over_time", "last_over_time(cpu_usage[60s])"),
        ("q15_count_over_time", "count_over_time(cpu_usage[60s])"),
        ("q16_sqrt_rate", "sqrt(rate(cpu_usage[60s]))"),
        ("q17_ln_plus1", "ln(rate(cpu_usage[60s]) + 1)"),
        ("q18_clamp", "clamp(rate(cpu_usage[60s]), 0, 1)"),
        ("q19_trig_combo", "sin(cpu_cores) + cos(cpu_cores)"),
        (
            "q20_deep_instant",
            "abs(ceil(rate(cpu_usage[60s]) * 1000 - 500)) / 1000",
        ),
        ("q21_sum_by", "sum by (id) (rate(cpu_usage[60s]))"),
        ("q22_sum_without", "sum without (op) (rate(cpu_usage[60s]))"),
        ("q23_stddev_agg", "stddev by (id) (rate(cpu_usage[60s]))"),
        (
            "q24_quantile_agg",
            "quantile by (id) (0.9, rate(cpu_usage[60s]))",
        ),
        ("q25_topk", "topk(3, rate(cpu_usage[60s]))"),
        ("q26_bottomk", "bottomk(3, rate(cpu_usage[60s]))"),
        (
            "q27_count_values",
            "count_values(\"v\", round(rate(cpu_usage[60s]) * 100))",
        ),
        (
            "q28_vec_bin_on",
            "sum by (id) (rate(cpu_usage[60s])) / on(id) \
             sum by (id) (rate(cpu_migrations[60s]))",
        ),
        (
            "q29_groupleft_ignoring",
            "rate(cpu_usage[60s]) / ignoring(op) group_left \
             sum by (id) (rate(cpu_usage[60s]))",
        ),
        (
            "q30_groupleft_on",
            "rate(cpu_usage[60s]) / on(id) group_left \
             sum by (id) (rate(cpu_usage[60s]))",
        ),
        ("q31_bool_compare", "rate(cpu_usage[60s]) > bool 0.5"),
        (
            "q32_and_op",
            "sum by (id) (rate(cpu_usage[60s])) and on(id) \
             sum by (id) (rate(cpu_migrations[60s]))",
        ),
        (
            "q33_unless_op",
            "(sum by (id) (rate(cpu_usage[60s])) > 0) unless \
             (sum by (id) (rate(cpu_migrations[60s])) > 0)",
        ),
        (
            "q34_label_replace",
            "label_replace(rate(cpu_usage[60s]), \"cpu\", \"cpu_$1\", \"id\", \"(.*)\")",
        ),
        (
            "q35_label_join",
            "label_join(rate(cpu_usage[60s]), \"key\", \"_\", \"id\", \"op\")",
        ),
        ("q36_datetime", "hour(timestamp(cpu_cores))"),
        ("q37_sort_desc", "sort_desc(rate(cpu_usage[60s]))"),
        (
            "q38_cross_metric_ratio",
            "sum by (cgroup) (rate(cgroup_cpu_usage[60s])) / on(cgroup) \
             sum by (cgroup) (rate(cgroup_cpu_cycles[60s]))",
        ),
    ];

    // Warmup (populates lazy caches, parquet row-group stats, JIT-like caches).
    let _ = engine
        .range_query(
            "cpu_cores",
            DateTime::from_timestamp_nanos(start_ns as i64),
            DateTime::from_timestamp_nanos(end_ns as i64),
            Duration::from_secs(step_s),
        )
        .await;

    let steps = (end_ns.saturating_sub(start_ns)) / (step_s * NS_PER_SEC) + 1;
    println!(
        "file={file}  range=[{start_ns}..{end_ns}]  step={step_s}s  steps={steps}  iters={iters}\n"
    );
    println!(
        "{:<24} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6} {:>6} {:>6}",
        "query", "parse", "plan", "opt", "phys", "exec(ms)", "total", "lnode", "onode", "pnode",
    );

    let filtered: Vec<_> = match &only {
        Some(name) => queries.iter().filter(|(n, _)| n == name).copied().collect(),
        None => queries.clone(),
    };

    for (name, q) in &filtered {
        match bench_range(&planner, q, start_ns, end_ns, step_s, iters).await {
            Ok(mut p) => println!(
                "{:<24} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>8.2} {:>6.2} {:>6} {:>6} {:>6}",
                name,
                pct(&mut p.parse, 0.5),
                pct(&mut p.plan, 0.5),
                pct(&mut p.opt, 0.5),
                pct(&mut p.phys, 0.5),
                pct(&mut p.exec, 0.5),
                pct(&mut p.total, 0.5),
                p.logical_nodes,
                p.optimized_nodes,
                p.physical_nodes,
            ),
            Err(e) => println!("{name:<24} ERROR: {e}"),
        }
    }

    if profile {
        println!("\n--- per-operator profiling (one run each) ---");
        for (name, q) in &filtered {
            profile_one(&planner, name, q, start_ns, end_ns, step_s).await?;
        }
    }

    Ok(())
}
