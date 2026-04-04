use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::TimeZone;
use datafusion::catalog::TableProvider;
use datafusion::datasource::MemTable;

use datafusion_promql::PromqlEngine;
use datafusion_promql::datasource::{Matcher, MetricMeta, MetricSource, TableFormat};
use datafusion_promql::error::Result;
use datafusion_promql::types::{QueryResult, TimeRange};

// ---------------------------------------------------------------------------
// Test data source
// ---------------------------------------------------------------------------

struct MultiMetricSource {
    tables: Vec<(String, Arc<Schema>, Vec<RecordBatch>)>,
}

impl MultiMetricSource {
    fn new() -> Self {
        Self { tables: Vec::new() }
    }

    fn add_metric(mut self, name: &str, schema: Arc<Schema>, batches: Vec<RecordBatch>) -> Self {
        self.tables.push((name.to_string(), schema, batches));
        self
    }
}

#[async_trait]
impl MetricSource for MultiMetricSource {
    async fn table_for_metric(
        &self,
        metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        for (name, schema, batches) in &self.tables {
            if name == metric_name {
                let table =
                    MemTable::try_new(Arc::clone(schema), vec![batches.clone()]).map_err(|e| {
                        datafusion_promql::error::PromqlError::DataSource(e.to_string())
                    })?;
                return Ok((Arc::new(table), TableFormat::Long));
            }
        }
        Err(datafusion_promql::error::PromqlError::DataSource(format!(
            "metric not found: {metric_name}"
        )))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![])
    }
}

/// Create a source with a simple gauge metric for aggregation tests.
///
/// `cpu_usage` with label `instance`:
/// - host1: values [10.0, 20.0, 30.0, 40.0, 50.0] at t=0..4s
/// - host2: values [50.0, 40.0, 30.0, 20.0, 10.0] at t=0..4s
/// - host3: values [15.0, 25.0, 35.0, 45.0, 55.0] at t=0..4s
fn make_gauge_source() -> MultiMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let host1_vals = [10.0, 20.0, 30.0, 40.0, 50.0];
    let host2_vals = [50.0, 40.0, 30.0, 20.0, 10.0];
    let host3_vals = [15.0, 25.0, 35.0, 45.0, 55.0];

    let mut names = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut instances = Vec::new();

    for (i, &v) in host1_vals.iter().enumerate() {
        names.push("cpu_usage");
        timestamps.push((i as i64) * 1_000_000_000);
        values.push(v);
        instances.push("host1");
    }
    for (i, &v) in host2_vals.iter().enumerate() {
        names.push("cpu_usage");
        timestamps.push((i as i64) * 1_000_000_000);
        values.push(v);
        instances.push("host2");
    }
    for (i, &v) in host3_vals.iter().enumerate() {
        names.push("cpu_usage");
        timestamps.push((i as i64) * 1_000_000_000);
        values.push(v);
        instances.push("host3");
    }

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(instances)),
        ],
    )
    .unwrap();

    MultiMetricSource::new().add_metric("cpu_usage", schema, vec![batch])
}

/// Create a gauge source with two labels for more complex grouping tests.
///
/// `request_duration` with labels `instance` and `job`:
/// - (host1, web): value=10.0 at t=2s
/// - (host2, web): value=20.0 at t=2s
/// - (host3, api): value=30.0 at t=2s
/// - (host4, api): value=40.0 at t=2s
fn make_two_label_source() -> MultiMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
        Field::new("job", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "request_duration",
                "request_duration",
                "request_duration",
                "request_duration",
            ])),
            Arc::new(Int64Array::from(vec![
                2_000_000_000_i64,
                2_000_000_000,
                2_000_000_000,
                2_000_000_000,
            ])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            Arc::new(StringArray::from(vec!["host1", "host2", "host3", "host4"])),
            Arc::new(StringArray::from(vec!["web", "web", "api", "api"])),
        ],
    )
    .unwrap();

    MultiMetricSource::new().add_metric("request_duration", schema, vec![batch])
}

// ---------------------------------------------------------------------------
// stddev tests
// ---------------------------------------------------------------------------

/// Test: stddev(cpu_usage) at t=2000
/// At t=2000: host1=30, host2=30, host3=35
/// Population stddev = sqrt(((30-31.67)^2 + (30-31.67)^2 + (35-31.67)^2) / 3)
#[tokio::test]
async fn test_stddev_no_modifier() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine.instant_query("stddev(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            // mean = (30+30+35)/3 = 31.666...
            // pop variance = ((30-31.67)^2 + (30-31.67)^2 + (35-31.67)^2)/3
            //              = (2.778 + 2.778 + 11.111)/3 = 5.556
            // pop stddev = sqrt(5.556) ≈ 2.357
            let expected_mean: f64 = (30.0 + 30.0 + 35.0) / 3.0;
            let expected_var: f64 = ((30.0 - expected_mean).powi(2)
                + (30.0 - expected_mean).powi(2)
                + (35.0 - expected_mean).powi(2))
                / 3.0;
            let expected_stddev = expected_var.sqrt();
            assert!(
                (samples[0].value - expected_stddev).abs() < 1e-6,
                "expected stddev ≈ {expected_stddev}, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// stdvar tests
// ---------------------------------------------------------------------------

/// Test: stdvar(cpu_usage) at t=2000
#[tokio::test]
async fn test_stdvar_no_modifier() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine.instant_query("stdvar(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            let expected_mean: f64 = (30.0 + 30.0 + 35.0) / 3.0;
            let expected_var: f64 = ((30.0 - expected_mean).powi(2)
                + (30.0 - expected_mean).powi(2)
                + (35.0 - expected_mean).powi(2))
                / 3.0;
            assert!(
                (samples[0].value - expected_var).abs() < 1e-6,
                "expected variance ≈ {expected_var}, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// group tests
// ---------------------------------------------------------------------------

/// Test: group(cpu_usage) at t=2000 - all values should be 1.0
#[tokio::test]
async fn test_group_no_modifier() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine.instant_query("group(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 group");
            assert!(
                (samples[0].value - 1.0).abs() < f64::EPSILON,
                "expected 1.0, got {}",
                samples[0].value
            );
            assert!(samples[0].labels.is_empty());
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: group(cpu_usage) by (instance) - one group per instance, all values 1.0
#[tokio::test]
async fn test_group_by_instance() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("group(cpu_usage) by (instance)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3, "expected 3 groups");
            for s in &samples {
                assert!(
                    (s.value - 1.0).abs() < f64::EPSILON,
                    "expected 1.0, got {}",
                    s.value
                );
                assert!(s.labels.contains_key("instance"));
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// topk tests
// ---------------------------------------------------------------------------

/// Test: topk(2, cpu_usage) at t=2000
/// At t=2000: host1=30, host2=30, host3=35
/// Top 2 by value: host3=35, host1=30 (or host2=30)
#[tokio::test]
async fn test_topk_2() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("topk(2, cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 results for topk(2)");
            // All results should have instance label (labels preserved)
            for s in &samples {
                assert!(
                    s.labels.contains_key("instance"),
                    "topk should preserve labels"
                );
            }
            // The top value should be 35.0
            let max_val = samples
                .iter()
                .map(|s| s.value)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                (max_val - 35.0).abs() < f64::EPSILON,
                "expected max=35.0, got {max_val}"
            );
            // Both values should be >= 30.0
            for s in &samples {
                assert!(
                    s.value >= 30.0 - f64::EPSILON,
                    "expected value >= 30.0, got {}",
                    s.value
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: topk(1, cpu_usage) at t=2000 - should return only host3=35
#[tokio::test]
async fn test_topk_1() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("topk(1, cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 result for topk(1)");
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host3");
            assert!(
                (samples[0].value - 35.0).abs() < f64::EPSILON,
                "expected 35.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// bottomk tests
// ---------------------------------------------------------------------------

/// Test: bottomk(1, cpu_usage) at t=2000
/// At t=2000: host1=30, host2=30, host3=35
/// Bottom 1: host1=30 or host2=30
#[tokio::test]
async fn test_bottomk_1() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("bottomk(1, cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 result for bottomk(1)");
            assert!(
                (samples[0].value - 30.0).abs() < f64::EPSILON,
                "expected 30.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: bottomk(2, cpu_usage) at t=2000 - two lowest
#[tokio::test]
async fn test_bottomk_2() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("bottomk(2, cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 results for bottomk(2)");
            for s in &samples {
                assert!(
                    (s.value - 30.0).abs() < f64::EPSILON,
                    "expected 30.0, got {}",
                    s.value
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// topk with by() modifier
// ---------------------------------------------------------------------------

/// Test: topk(1, request_duration) by (job) at t=2000
/// job=web: host1=10, host2=20 -> topk(1) = host2=20
/// job=api: host3=30, host4=40 -> topk(1) = host4=40
#[tokio::test]
async fn test_topk_by_job() {
    let source = make_two_label_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("topk(1, request_duration) by (job)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 groups (1 per job)");
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("job").cmp(&b.labels.get("job")));

            // api: topk(1) = host4=40
            assert_eq!(samples[0].labels.get("job").unwrap(), "api");
            assert!(
                (samples[0].value - 40.0).abs() < f64::EPSILON,
                "expected 40.0, got {}",
                samples[0].value
            );

            // web: topk(1) = host2=20
            assert_eq!(samples[1].labels.get("job").unwrap(), "web");
            assert!(
                (samples[1].value - 20.0).abs() < f64::EPSILON,
                "expected 20.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// quantile tests
// ---------------------------------------------------------------------------

/// Test: quantile(0.5, cpu_usage) at t=2000
/// At t=2000: values are [30, 30, 35] -> median = 30.0
#[tokio::test]
async fn test_quantile_median() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("quantile(0.5, cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert!(
                (samples[0].value - 30.0).abs() < 1.0,
                "expected quantile(0.5) ≈ 30.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: quantile(1.0, cpu_usage) at t=2000 -> max value = 35.0
#[tokio::test]
async fn test_quantile_max() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("quantile(1.0, cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert!(
                (samples[0].value - 35.0).abs() < f64::EPSILON,
                "expected quantile(1.0) = 35.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: quantile(0.0, cpu_usage) at t=2000 -> min value = 30.0
#[tokio::test]
async fn test_quantile_min() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("quantile(0.0, cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert!(
                (samples[0].value - 30.0).abs() < f64::EPSILON,
                "expected quantile(0.0) = 30.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// count_values tests
// ---------------------------------------------------------------------------

/// Test: count_values("val", cpu_usage) at t=2000
/// At t=2000: host1=30, host2=30, host3=35
/// -> val="30": count=2, val="35": count=1
#[tokio::test]
async fn test_count_values_no_modifier() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("count_values(\"val\", cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 groups (2 distinct values)");
            let mut samples = samples;
            samples.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

            // val="35": count=1
            assert!(
                (samples[0].value - 1.0).abs() < f64::EPSILON,
                "expected count=1, got {}",
                samples[0].value
            );

            // val="30": count=2
            assert!(
                (samples[1].value - 2.0).abs() < f64::EPSILON,
                "expected count=2, got {}",
                samples[1].value
            );

            // Check that the "val" label exists
            for s in &samples {
                assert!(
                    s.labels.contains_key("val"),
                    "expected 'val' label, got {:?}",
                    s.labels
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// Note: limitk and limit_ratio are not yet supported by the promql-parser grammar,
// so they cannot be tested end-to-end. The planner code is ready for when the
// parser adds support.

// ---------------------------------------------------------------------------
// stddev/stdvar with by() modifier
// ---------------------------------------------------------------------------

/// Test: stddev(request_duration) by (job) at t=2000
/// job=web: [10, 20] -> mean=15, pop_stddev = sqrt(25) = 5
/// job=api: [30, 40] -> mean=35, pop_stddev = sqrt(25) = 5
#[tokio::test]
async fn test_stddev_by_job() {
    let source = make_two_label_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("stddev(request_duration) by (job)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            for s in &samples {
                assert!(
                    (s.value - 5.0).abs() < 1e-6,
                    "expected stddev=5.0, got {}",
                    s.value
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: stdvar(request_duration) by (job) at t=2000
/// job=web: [10, 20] -> mean=15, pop_var = 25
/// job=api: [30, 40] -> mean=35, pop_var = 25
#[tokio::test]
async fn test_stdvar_by_job() {
    let source = make_two_label_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("stdvar(request_duration) by (job)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            for s in &samples {
                assert!(
                    (s.value - 25.0).abs() < 1e-6,
                    "expected variance=25.0, got {}",
                    s.value
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}
