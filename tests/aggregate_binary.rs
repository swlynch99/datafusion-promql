use std::sync::Arc;
use std::time::Duration;

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
// Test data sources
// ---------------------------------------------------------------------------

/// A metric source that dispatches to different tables by metric name.
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
        Err(datafusion_promql::error::PromqlError::DataSource(
            format!("metric not found: {metric_name}"),
        ))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![])
    }
}

/// Create a counter metric source with 3 series for aggregation testing.
///
/// `http_requests_total` with labels `instance` and `job`:
/// - (host1, webserver): rate = 10/s (values: 0, 10, 20, ..., 100)
/// - (host2, webserver): rate = 20/s (values: 0, 20, 40, ..., 200)
/// - (host3, api):       rate = 5/s  (values: 0, 5, 10, ..., 50)
fn make_counter_source() -> MultiMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
        Field::new("job", DataType::Utf8, false),
    ]));

    let n = 11; // 0..=10 seconds
    let mut names = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut instances = Vec::new();
    let mut jobs = Vec::new();

    // Series 1: instance=host1, job=webserver, rate = 10 req/s
    for i in 0..n {
        names.push("http_requests_total");
        timestamps.push((i as i64) * 1000);
        values.push((i as f64) * 10.0);
        instances.push("host1");
        jobs.push("webserver");
    }

    // Series 2: instance=host2, job=webserver, rate = 20 req/s
    for i in 0..n {
        names.push("http_requests_total");
        timestamps.push((i as i64) * 1000);
        values.push((i as f64) * 20.0);
        instances.push("host2");
        jobs.push("webserver");
    }

    // Series 3: instance=host3, job=api, rate = 5 req/s
    for i in 0..n {
        names.push("http_requests_total");
        timestamps.push((i as i64) * 1000);
        values.push((i as f64) * 5.0);
        instances.push("host3");
        jobs.push("api");
    }

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(instances)),
            Arc::new(StringArray::from(jobs)),
        ],
    )
    .unwrap();

    MultiMetricSource::new().add_metric("http_requests_total", schema, vec![batch])
}

/// Create a source with a simple gauge metric for basic aggregation tests.
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
        timestamps.push((i as i64) * 1000);
        values.push(v);
        instances.push("host1");
    }
    for (i, &v) in host2_vals.iter().enumerate() {
        names.push("cpu_usage");
        timestamps.push((i as i64) * 1000);
        values.push(v);
        instances.push("host2");
    }
    for (i, &v) in host3_vals.iter().enumerate() {
        names.push("cpu_usage");
        timestamps.push((i as i64) * 1000);
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

/// Create a source with two different metrics for binary vector-vector tests.
///
/// `metric_a` and `metric_b` both with label `instance`:
/// - metric_a, host1: value=100 at t=5000
/// - metric_a, host2: value=200 at t=5000
/// - metric_b, host1: value=10 at t=5000
/// - metric_b, host2: value=20 at t=5000
fn make_two_metric_source() -> MultiMetricSource {
    let schema_a = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let batch_a = RecordBatch::try_new(
        Arc::clone(&schema_a),
        vec![
            Arc::new(StringArray::from(vec!["metric_a", "metric_a"])),
            Arc::new(Int64Array::from(vec![5000, 5000])),
            Arc::new(Float64Array::from(vec![100.0, 200.0])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
        ],
    )
    .unwrap();

    let schema_b = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let batch_b = RecordBatch::try_new(
        Arc::clone(&schema_b),
        vec![
            Arc::new(StringArray::from(vec!["metric_b", "metric_b"])),
            Arc::new(Int64Array::from(vec![5000, 5000])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
        ],
    )
    .unwrap();

    MultiMetricSource::new()
        .add_metric("metric_a", schema_a, vec![batch_a])
        .add_metric("metric_b", schema_b, vec![batch_b])
}

// ---------------------------------------------------------------------------
// Aggregation tests
// ---------------------------------------------------------------------------

/// Test: sum(rate(http_requests_total[5s])) by (instance)
/// The main target test from the architecture plan.
#[tokio::test]
async fn test_sum_rate_by_instance() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // rate(http_requests_total[5s]) at t=10000:
    //   host1,webserver: rate = 10.0/s
    //   host2,webserver: rate = 20.0/s
    //   host3,api:       rate = 5.0/s
    // sum by (instance):
    //   host1: 10.0
    //   host2: 20.0
    //   host3: 5.0
    // (Each instance has only one series, so sum is the rate itself.)
    let ts = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let result = engine
        .instant_query("sum(rate(http_requests_total[5s])) by (instance)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3, "expected 3 groups");
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert!(
                (samples[0].value - 10.0).abs() < f64::EPSILON,
                "expected 10.0, got {}",
                samples[0].value
            );

            assert_eq!(samples[1].labels.get("instance").unwrap(), "host2");
            assert!(
                (samples[1].value - 20.0).abs() < f64::EPSILON,
                "expected 20.0, got {}",
                samples[1].value
            );

            assert_eq!(samples[2].labels.get("instance").unwrap(), "host3");
            assert!(
                (samples[2].value - 5.0).abs() < f64::EPSILON,
                "expected 5.0, got {}",
                samples[2].value
            );

            // Verify labels only contain "instance" (no "job", no "__name__")
            for s in &samples {
                assert!(
                    !s.labels.contains_key("job"),
                    "labels should not contain 'job'"
                );
                assert!(
                    !s.labels.contains_key("__name__"),
                    "labels should not contain '__name__'"
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: sum(rate(http_requests_total[5s])) by (job)
/// Two groups: webserver (host1 + host2) = 30.0, api (host3) = 5.0
#[tokio::test]
async fn test_sum_rate_by_job() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let result = engine
        .instant_query("sum(rate(http_requests_total[5s])) by (job)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 groups");
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("job").cmp(&b.labels.get("job")));

            // api: 5.0
            assert_eq!(samples[0].labels.get("job").unwrap(), "api");
            assert!(
                (samples[0].value - 5.0).abs() < f64::EPSILON,
                "expected 5.0, got {}",
                samples[0].value
            );

            // webserver: 10.0 + 20.0 = 30.0
            assert_eq!(samples[1].labels.get("job").unwrap(), "webserver");
            assert!(
                (samples[1].value - 30.0).abs() < f64::EPSILON,
                "expected 30.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: sum(cpu_usage) - aggregate all into one group (no modifier)
#[tokio::test]
async fn test_sum_no_modifier() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // At t=2000: host1=30.0, host2=30.0, host3=35.0 -> sum = 95.0
    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine.instant_query("sum(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 group (sum of all)");
            assert!(
                (samples[0].value - 95.0).abs() < f64::EPSILON,
                "expected 95.0, got {}",
                samples[0].value
            );
            // No labels should be present
            assert!(
                samples[0].labels.is_empty(),
                "expected empty labels, got {:?}",
                samples[0].labels
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: avg(cpu_usage) by (instance) at t=2000
/// Each instance has one value at that time, so avg = value itself.
#[tokio::test]
async fn test_avg_by_instance() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("avg(cpu_usage) by (instance)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3);
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert!(
                (samples[0].value - 30.0).abs() < f64::EPSILON,
                "host1 avg expected 30.0, got {}",
                samples[0].value
            );
            assert!(
                (samples[1].value - 30.0).abs() < f64::EPSILON,
                "host2 avg expected 30.0, got {}",
                samples[1].value
            );
            assert!(
                (samples[2].value - 35.0).abs() < f64::EPSILON,
                "host3 avg expected 35.0, got {}",
                samples[2].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: count(cpu_usage) - count all series
#[tokio::test]
async fn test_count_no_modifier() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine.instant_query("count(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert!(
                (samples[0].value - 3.0).abs() < f64::EPSILON,
                "expected count=3, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: min(cpu_usage) and max(cpu_usage) at t=2000
#[tokio::test]
async fn test_min_max() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();

    // min: host1=30, host2=30, host3=35 -> min=30
    let result = engine.instant_query("min(cpu_usage)", ts).await.unwrap();
    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert!(
                (samples[0].value - 30.0).abs() < f64::EPSILON,
                "expected min=30.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }

    // max: 35.0
    let result = engine.instant_query("max(cpu_usage)", ts).await.unwrap();
    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert!(
                (samples[0].value - 35.0).abs() < f64::EPSILON,
                "expected max=35.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: sum without (instance) (cpu_usage) at t=2000
/// Without instance means group by all labels except instance and __name__.
/// Since the only label is instance, this aggregates everything.
#[tokio::test]
async fn test_sum_without() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("sum without (instance) (cpu_usage)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 group");
            assert!(
                (samples[0].value - 95.0).abs() < f64::EPSILON,
                "expected 95.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Aggregation range query test
// ---------------------------------------------------------------------------

/// Test: sum(rate(http_requests_total[5s])) by (job) as a range query
#[tokio::test]
async fn test_sum_rate_range_query() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(5000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let step = Duration::from_secs(5);

    let result = engine
        .range_query(
            "sum(rate(http_requests_total[5s])) by (job)",
            start,
            end,
            step,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 2, "expected 2 series (by job)");
            let mut series = series;
            series.sort_by(|a, b| a.labels.get("job").cmp(&b.labels.get("job")));

            // api job: rate=5.0 at both steps
            assert_eq!(series[0].labels.get("job").unwrap(), "api");
            for &(ts, val) in &series[0].samples {
                assert!(
                    (val - 5.0).abs() < f64::EPSILON,
                    "api: expected 5.0 at ts={ts}, got {val}"
                );
            }

            // webserver job: rate=30.0 at both steps
            assert_eq!(series[1].labels.get("job").unwrap(), "webserver");
            for &(ts, val) in &series[1].samples {
                assert!(
                    (val - 30.0).abs() < f64::EPSILON,
                    "webserver: expected 30.0 at ts={ts}, got {val}"
                );
            }
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Binary operation tests: vector + scalar
// ---------------------------------------------------------------------------

/// Test: cpu_usage + 100
#[tokio::test]
async fn test_vector_plus_scalar() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // At t=2000: host1=30, host2=30, host3=35
    // + 100 -> 130, 130, 135
    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("cpu_usage + 100", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3);
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert!(
                (samples[0].value - 130.0).abs() < f64::EPSILON,
                "host1: expected 130.0, got {}",
                samples[0].value
            );
            assert!(
                (samples[1].value - 130.0).abs() < f64::EPSILON,
                "host2: expected 130.0, got {}",
                samples[1].value
            );
            assert!(
                (samples[2].value - 135.0).abs() < f64::EPSILON,
                "host3: expected 135.0, got {}",
                samples[2].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: 100 + cpu_usage (commutative)
#[tokio::test]
async fn test_scalar_plus_vector() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("100 + cpu_usage", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3);
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert!(
                (samples[0].value - 130.0).abs() < f64::EPSILON,
                "host1: expected 130.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: cpu_usage * 2
#[tokio::test]
async fn test_vector_mul_scalar() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("cpu_usage * 2", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3);
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: 30*2=60, host2: 30*2=60, host3: 35*2=70
            assert!(
                (samples[0].value - 60.0).abs() < f64::EPSILON,
                "host1: expected 60.0, got {}",
                samples[0].value
            );
            assert!(
                (samples[2].value - 70.0).abs() < f64::EPSILON,
                "host3: expected 70.0, got {}",
                samples[2].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: cpu_usage > 30 (comparison operator, filtering mode)
#[tokio::test]
async fn test_comparison_filter() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // At t=2000: host1=30, host2=30, host3=35
    // cpu_usage > 30 should only return host3 (35 > 30)
    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine
        .instant_query("cpu_usage > 30", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 sample (only host3 > 30)");
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
// Binary operation tests: vector + vector
// ---------------------------------------------------------------------------

/// Test: metric_a + on(instance) metric_b
#[tokio::test]
async fn test_vector_vector_add() {
    let source = make_two_metric_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // metric_a: host1=100, host2=200
    // metric_b: host1=10, host2=20
    // metric_a + on(instance) metric_b -> host1=110, host2=220
    let ts = chrono::Utc.timestamp_millis_opt(5000).unwrap();
    let result = engine
        .instant_query("metric_a + on(instance) metric_b", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 samples");
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert!(
                (samples[0].value - 110.0).abs() < f64::EPSILON,
                "host1: expected 110.0, got {}",
                samples[0].value
            );

            assert_eq!(samples[1].labels.get("instance").unwrap(), "host2");
            assert!(
                (samples[1].value - 220.0).abs() < f64::EPSILON,
                "host2: expected 220.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Test: metric_a / on(instance) metric_b
#[tokio::test]
async fn test_vector_vector_div() {
    let source = make_two_metric_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // metric_a / metric_b: host1=100/10=10, host2=200/20=10
    let ts = chrono::Utc.timestamp_millis_opt(5000).unwrap();
    let result = engine
        .instant_query("metric_a / on(instance) metric_b", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            for s in &samples {
                assert!(
                    (s.value - 10.0).abs() < f64::EPSILON,
                    "expected 10.0, got {}",
                    s.value
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Unary negation test
// ---------------------------------------------------------------------------

/// Test: -cpu_usage
#[tokio::test]
async fn test_unary_negation() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // At t=2000: host1=30, host2=30, host3=35
    // -cpu_usage -> -30, -30, -35
    let ts = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let result = engine.instant_query("-cpu_usage", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3);
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert!(
                (samples[0].value - (-30.0)).abs() < f64::EPSILON,
                "host1: expected -30.0, got {}",
                samples[0].value
            );
            assert!(
                (samples[2].value - (-35.0)).abs() < f64::EPSILON,
                "host3: expected -35.0, got {}",
                samples[2].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}
