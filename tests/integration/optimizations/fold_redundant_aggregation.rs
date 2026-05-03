use std::sync::Arc;

use arrow::array::{Float64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::TimeZone;
use datafusion::catalog::TableProvider;
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::Transformed;
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::OptimizerRule;
use datafusion::prelude::*;

use datafusion_promql::PromqlEngine;
use datafusion_promql::datasource::{Matcher, MetricMeta, MetricSource, TableFormat, ValueKind};
use datafusion_promql::error::Result;
use datafusion_promql::opt::logical::FoldRedundantAggregation;
use datafusion_promql::types::{QueryResult, TimeRange};

// ─── Helpers for unit tests ────────────────────────────────────────────────

/// Minimal OptimizerConfig that does nothing.
struct NoopConfig;

impl datafusion::optimizer::OptimizerConfig for NoopConfig {
    fn query_execution_start_time(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        None
    }

    fn alias_generator(&self) -> &Arc<AliasGenerator> {
        static GEN: std::sync::LazyLock<Arc<AliasGenerator>> =
            std::sync::LazyLock::new(|| Arc::new(AliasGenerator::default()));
        &GEN
    }

    fn options(&self) -> Arc<ConfigOptions> {
        Arc::new(ConfigOptions::default())
    }
}

/// Build a trivial single-row MemTable scan.
fn make_scan() -> LogicalPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let table = MemTable::try_new(schema, vec![vec![]]).expect("failed to create MemTable");
    LogicalPlanBuilder::scan(
        "t",
        datafusion::datasource::provider_as_source(Arc::new(table)),
        None,
    )
    .unwrap()
    .build()
    .unwrap()
}

/// Build a nested aggregate: outer_fn(inner_fn(value)) GROUP BY timestamp.
fn make_nested_aggregate(inner_fn_name: &str, outer_fn_name: &str) -> LogicalPlan {
    let scan = make_scan();

    let value_col = col("value");
    let group_expr = vec![col("timestamp")];

    let inner_agg = match inner_fn_name {
        "sum" => datafusion::functions_aggregate::sum::sum(value_col.clone()),
        "min" => datafusion::functions_aggregate::min_max::min(value_col.clone()),
        "max" => datafusion::functions_aggregate::min_max::max(value_col.clone()),
        "avg" => datafusion::functions_aggregate::average::avg(value_col.clone()),
        "count" => datafusion::functions_aggregate::count::count(value_col.clone()),
        _ => panic!("unsupported function: {inner_fn_name}"),
    };

    let inner = LogicalPlanBuilder::from(scan)
        .aggregate(group_expr.clone(), vec![inner_agg.alias("value")])
        .unwrap()
        .build()
        .unwrap();

    let outer_value_col = col("value");
    let outer_agg = match outer_fn_name {
        "sum" => datafusion::functions_aggregate::sum::sum(outer_value_col),
        "min" => datafusion::functions_aggregate::min_max::min(outer_value_col),
        "max" => datafusion::functions_aggregate::min_max::max(outer_value_col),
        "avg" => datafusion::functions_aggregate::average::avg(outer_value_col),
        "count" => datafusion::functions_aggregate::count::count(outer_value_col),
        _ => panic!("unsupported function: {outer_fn_name}"),
    };

    LogicalPlanBuilder::from(inner)
        .aggregate(group_expr, vec![outer_agg.alias("value")])
        .unwrap()
        .build()
        .unwrap()
}

fn apply_rule(plan: LogicalPlan) -> (LogicalPlan, bool) {
    let rule = FoldRedundantAggregation;
    let Transformed {
        data, transformed, ..
    } = rule.rewrite(plan, &NoopConfig).unwrap();
    (data, transformed)
}

fn count_aggregate_nodes(plan: &LogicalPlan) -> usize {
    let mut count = 0;
    if matches!(plan, LogicalPlan::Aggregate(_)) {
        count += 1;
    }
    for child in plan.inputs() {
        count += count_aggregate_nodes(child);
    }
    count
}

// ─── Unit tests: rule fires for idempotent functions ───────────────────────

#[test]
fn sum_sum_folds() {
    let plan = make_nested_aggregate("sum", "sum");
    assert_eq!(count_aggregate_nodes(&plan), 2);
    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should fire for sum(sum(...))");
    assert_eq!(count_aggregate_nodes(&result), 1);
}

#[test]
fn min_min_folds() {
    let plan = make_nested_aggregate("min", "min");
    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should fire for min(min(...))");
    assert_eq!(count_aggregate_nodes(&result), 1);
}

#[test]
fn max_max_folds() {
    let plan = make_nested_aggregate("max", "max");
    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should fire for max(max(...))");
    assert_eq!(count_aggregate_nodes(&result), 1);
}

// ─── Unit tests: rule does NOT fire ────────────────────────────────────────

#[test]
fn avg_avg_does_not_fold() {
    let plan = make_nested_aggregate("avg", "avg");
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should NOT fire for avg(avg(...))");
}

#[test]
fn count_count_does_not_fold() {
    let plan = make_nested_aggregate("count", "count");
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should NOT fire for count(count(...))");
}

#[test]
fn mismatched_functions_do_not_fold() {
    let plan = make_nested_aggregate("min", "sum");
    let (_, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "rule should NOT fire for sum(min(...)) — mismatched functions"
    );
}

#[test]
fn different_grouping_does_not_fold() {
    // Build manually with different group expressions.
    let scan = make_scan();
    let inner = LogicalPlanBuilder::from(scan)
        .aggregate(
            vec![col("timestamp")],
            vec![datafusion::functions_aggregate::sum::sum(col("value")).alias("value")],
        )
        .unwrap()
        .build()
        .unwrap();

    // Outer groups by nothing (empty group).
    let outer = LogicalPlanBuilder::from(inner)
        .aggregate(
            Vec::<Expr>::new(),
            vec![datafusion::functions_aggregate::sum::sum(col("value")).alias("value")],
        )
        .unwrap()
        .build()
        .unwrap();

    let (_, transformed) = apply_rule(outer);
    assert!(
        !transformed,
        "rule should NOT fire when grouping columns differ"
    );
}

// ─── End-to-end test infrastructure ────────────────────────────────────────

struct SimpleSource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

#[async_trait]
impl MetricSource for SimpleSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        let table = MemTable::try_new(Arc::clone(&self.schema), vec![self.batches.clone()])
            .map_err(|e| datafusion_promql::error::PromqlError::DataSource(e.to_string()))?;
        Ok((
            Arc::new(table),
            TableFormat::Long {
                value_kind: ValueKind::Scalar,
            },
        ))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![])
    }
}

fn make_source() -> SimpleSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    // Two hosts at t=2s: host1=10.0, host2=30.0
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["cpu_usage", "cpu_usage"])),
            Arc::new(UInt64Array::from(vec![2_000_000_000_u64, 2_000_000_000])),
            Arc::new(Float64Array::from(vec![10.0, 30.0])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
        ],
    )
    .unwrap();

    SimpleSource {
        schema,
        batches: vec![batch],
    }
}

// ─── End-to-end tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn e2e_sum_sum_returns_correct_value() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_opt(2, 0).unwrap();

    let result = engine
        .instant_query("sum(sum(cpu_usage))", ts)
        .await
        .unwrap();
    let QueryResult::Vector(samples) = result else {
        panic!("expected Vector result");
    };
    assert_eq!(samples.len(), 1);
    // sum of [10, 30] = 40, sum(40) = 40
    assert_eq!(samples[0].value, 40.0);
}

#[tokio::test]
async fn e2e_min_min_returns_correct_value() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_opt(2, 0).unwrap();

    let result = engine
        .instant_query("min(min(cpu_usage))", ts)
        .await
        .unwrap();
    let QueryResult::Vector(samples) = result else {
        panic!("expected Vector result");
    };
    assert_eq!(samples.len(), 1);
    // min of [10, 30] = 10, min(10) = 10
    assert_eq!(samples[0].value, 10.0);
}

#[tokio::test]
async fn e2e_max_max_returns_correct_value() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_opt(2, 0).unwrap();

    let result = engine
        .instant_query("max(max(cpu_usage))", ts)
        .await
        .unwrap();
    let QueryResult::Vector(samples) = result else {
        panic!("expected Vector result");
    };
    assert_eq!(samples.len(), 1);
    // max of [10, 30] = 30, max(30) = 30
    assert_eq!(samples[0].value, 30.0);
}
