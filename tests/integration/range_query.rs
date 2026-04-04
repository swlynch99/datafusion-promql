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

/// In-memory metric source for testing range queries.
struct InMemoryMetricSource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

impl InMemoryMetricSource {
    fn new(schema: Arc<Schema>, batches: Vec<RecordBatch>) -> Self {
        Self { schema, batches }
    }
}

#[async_trait]
impl MetricSource for InMemoryMetricSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        let table = MemTable::try_new(Arc::clone(&self.schema), vec![self.batches.clone()])
            .map_err(|e| datafusion_promql::error::PromqlError::DataSource(e.to_string()))?;
        Ok((Arc::new(table), TableFormat::Long))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![])
    }
}

/// Create a counter metric source: `http_requests_total` with two series,
/// monotonically increasing at 10/s and 20/s respectively.
/// Samples every 1 second from t=0 to t=10s.
fn make_counter_source() -> InMemoryMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
        Field::new("job", DataType::Utf8, false),
    ]));

    let n = 11; // 0..=10 seconds
    let mut names = Vec::with_capacity(n * 2);
    let mut timestamps = Vec::with_capacity(n * 2);
    let mut values = Vec::with_capacity(n * 2);
    let mut instances = Vec::with_capacity(n * 2);
    let mut jobs = Vec::with_capacity(n * 2);

    // Series 1: instance=host1, rate = 10 req/s
    for i in 0..n {
        names.push("http_requests_total");
        timestamps.push((i as i64) * 1_000_000_000);
        values.push((i as f64) * 10.0); // 0, 10, 20, 30, ...
        instances.push("host1");
        jobs.push("webserver");
    }

    // Series 2: instance=host2, rate = 20 req/s
    for i in 0..n {
        names.push("http_requests_total");
        timestamps.push((i as i64) * 1_000_000_000);
        values.push((i as f64) * 20.0); // 0, 20, 40, 60, ...
        instances.push("host2");
        jobs.push("webserver");
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
    .expect("failed to create test batch");

    InMemoryMetricSource::new(schema, vec![batch])
}

/// Create a gauge metric source: `temperature` with one series,
/// values that go up and down.
fn make_gauge_source() -> InMemoryMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("sensor", DataType::Utf8, false),
    ]));

    let gauge_values = vec![
        20.0, 22.0, 25.0, 23.0, 21.0, 24.0, 26.0, 28.0, 27.0, 25.0, 23.0,
    ];
    let n = gauge_values.len();

    let mut names = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let mut sensors = Vec::with_capacity(n);

    for (i, &v) in gauge_values.iter().enumerate() {
        names.push("temperature");
        timestamps.push((i as i64) * 1_000_000_000);
        values.push(v);
        sensors.push("room1");
    }

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(sensors)),
        ],
    )
    .expect("failed to create test batch");

    InMemoryMetricSource::new(schema, vec![batch])
}

// ---- Instant query with rate() ----

#[tokio::test]
async fn test_instant_query_rate() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // rate(http_requests_total[5s]) at t=10s
    // Window for host1: [5s..10s] -> samples at 5s,6s,7s,8s,9s,10s
    // Values: 50,60,70,80,90,100 -> increase=50 over 5s -> rate=10/s
    let ts = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let result = engine
        .instant_query("rate(http_requests_total[5s])", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: rate = 10.0/s
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert!(
                (samples[0].value - 10.0).abs() < f64::EPSILON,
                "expected rate 10.0, got {}",
                samples[0].value
            );

            // host2: rate = 20.0/s
            assert_eq!(samples[1].labels.get("instance").unwrap(), "host2");
            assert!(
                (samples[1].value - 20.0).abs() < f64::EPSILON,
                "expected rate 20.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---- Range query with rate() ----

#[tokio::test]
async fn test_range_query_rate() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // rate(http_requests_total[5s]) over [5s, 10s] step 5s
    // At t=5s: window [0,5s], samples at 0s..5s, host1: (0->50)/5=10, host2: (0->100)/5=20
    // At t=10s: window [5s,10s], samples at 5s..10s, host1: (50->100)/5=10, host2: (100->200)/5=20
    let start = chrono::Utc.timestamp_millis_opt(5000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let step = Duration::from_secs(5);

    let result = engine
        .range_query("rate(http_requests_total[5s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 2, "expected 2 series");
            let mut series = series;
            series.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: rate should be 10.0 at both steps
            assert_eq!(series[0].labels.get("instance").unwrap(), "host1");
            assert_eq!(series[0].samples.len(), 2, "expected 2 steps for host1");
            for &(ts, val) in &series[0].samples {
                assert!(
                    (val - 10.0).abs() < f64::EPSILON,
                    "expected rate 10.0 at ts={ts}, got {val}"
                );
            }

            // host2: rate should be 20.0 at both steps
            assert_eq!(series[1].labels.get("instance").unwrap(), "host2");
            assert_eq!(series[1].samples.len(), 2, "expected 2 steps for host2");
            for &(ts, val) in &series[1].samples {
                assert!(
                    (val - 20.0).abs() < f64::EPSILON,
                    "expected rate 20.0 at ts={ts}, got {val}"
                );
            }
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}

// ---- Instant query with irate() ----

#[tokio::test]
async fn test_instant_query_irate() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // irate(http_requests_total[5s]) at t=10s
    // Uses last two samples: (9s, 90) and (10s, 100) for host1
    // irate = (100-90)/(1s) = 10.0
    let ts = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let result = engine
        .instant_query("irate(http_requests_total[5s])", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: irate = 10.0/s
            assert!(
                (samples[0].value - 10.0).abs() < f64::EPSILON,
                "expected irate 10.0, got {}",
                samples[0].value
            );

            // host2: irate = 20.0/s
            assert!(
                (samples[1].value - 20.0).abs() < f64::EPSILON,
                "expected irate 20.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---- Range query with increase() ----

#[tokio::test]
async fn test_range_query_increase() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // increase(http_requests_total[5s]) over [5s, 10s] step 5s
    // At t=5s: window [0,5s], host1 increase = 50-0 = 50
    // At t=10s: window [5s,10s], host1 increase = 100-50 = 50
    let start = chrono::Utc.timestamp_millis_opt(5000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let step = Duration::from_secs(5);

    let result = engine
        .range_query("increase(http_requests_total[5s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 2, "expected 2 series");
            let mut series = series;
            series.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: increase should be 50 at both steps
            assert_eq!(series[0].labels.get("instance").unwrap(), "host1");
            for &(ts, val) in &series[0].samples {
                assert!(
                    (val - 50.0).abs() < f64::EPSILON,
                    "expected increase 50.0 at ts={ts}, got {val}"
                );
            }

            // host2: increase should be 100 at both steps
            assert_eq!(series[1].labels.get("instance").unwrap(), "host2");
            for &(ts, val) in &series[1].samples {
                assert!(
                    (val - 100.0).abs() < f64::EPSILON,
                    "expected increase 100.0 at ts={ts}, got {val}"
                );
            }
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}

// ---- Range query with delta() ----

#[tokio::test]
async fn test_range_query_delta() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // delta(temperature[3s]) at t=3s
    // Window [0, 3s]: samples at 0,1,2,3s -> values 20,22,25,23
    // delta = 23 - 20 = 3.0
    let ts = chrono::Utc.timestamp_millis_opt(3000).unwrap();
    let result = engine
        .instant_query("delta(temperature[3s])", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series");
            assert!(
                (samples[0].value - 3.0).abs() < f64::EPSILON,
                "expected delta 3.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_range_query_delta_over_range() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // delta(temperature[3s]) over [3s, 6s] step 3s
    // At t=3s: window [0,3s] -> 20,22,25,23 -> delta = 3.0
    // At t=6s: window [3s,6s] -> 23,21,24,26 -> delta = 3.0
    let start = chrono::Utc.timestamp_millis_opt(3000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(6000).unwrap();
    let step = Duration::from_secs(3);

    let result = engine
        .range_query("delta(temperature[3s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "expected 1 series");
            assert_eq!(series[0].samples.len(), 2, "expected 2 steps");

            // t=3s: delta(20..23) = 3.0
            assert!(
                (series[0].samples[0].1 - 3.0).abs() < f64::EPSILON,
                "expected delta 3.0 at t=3s, got {}",
                series[0].samples[0].1
            );
            // t=6s: delta(23..26) = 3.0
            assert!(
                (series[0].samples[1].1 - 3.0).abs() < f64::EPSILON,
                "expected delta 3.0 at t=6s, got {}",
                series[0].samples[1].1
            );
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}

// ---- Instant query with idelta() ----

#[tokio::test]
async fn test_instant_query_idelta() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // idelta(temperature[3s]) at t=5s
    // Window [2s, 5s]: samples at 2s,3s,4s,5s -> values 25,23,21,24
    // idelta uses last two: 24 - 21 = 3.0
    let ts = chrono::Utc.timestamp_millis_opt(5000).unwrap();
    let result = engine
        .instant_query("idelta(temperature[3s])", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series");
            assert!(
                (samples[0].value - 3.0).abs() < f64::EPSILON,
                "expected idelta 3.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_range_query_idelta_over_range() {
    let source = make_gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // idelta(temperature[3s]) over [3s, 6s] step 3s
    // gauge_values: 20, 22, 25, 23, 21, 24, 26
    // At t=3s: window [0s,3s] -> 20,22,25,23 -> last two: 25,23 -> idelta = -2.0
    // At t=6s: window [3s,6s] -> 23,21,24,26 -> last two: 24,26 -> idelta = 2.0
    let start = chrono::Utc.timestamp_millis_opt(3000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(6000).unwrap();
    let step = Duration::from_secs(3);

    let result = engine
        .range_query("idelta(temperature[3s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "expected 1 series");
            assert_eq!(series[0].samples.len(), 2, "expected 2 steps");

            // t=3s: idelta(25, 23) = -2.0
            assert!(
                (series[0].samples[0].1 - (-2.0)).abs() < f64::EPSILON,
                "expected idelta -2.0 at t=3s, got {}",
                series[0].samples[0].1
            );
            // t=6s: idelta(24, 26) = 2.0
            assert!(
                (series[0].samples[1].1 - 2.0).abs() < f64::EPSILON,
                "expected idelta 2.0 at t=6s, got {}",
                series[0].samples[1].1
            );
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}

// ---- Range query returning matrix (plain selector) ----

#[tokio::test]
async fn test_range_query_plain_selector() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // Plain selector `http_requests_total` as range query over [0, 4s] step 2s
    // Should return aligned samples at each step for each series.
    let start = chrono::Utc.timestamp_millis_opt(0).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(4000).unwrap();
    let step = Duration::from_secs(2);

    let result = engine
        .range_query("http_requests_total", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 2, "expected 2 series");
            let mut series = series;
            series.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: steps at 0, 2s, 4s -> values 0, 20, 40
            assert_eq!(series[0].labels.get("instance").unwrap(), "host1");
            assert_eq!(series[0].samples.len(), 3, "expected 3 steps for host1");
            assert!((series[0].samples[0].1 - 0.0).abs() < f64::EPSILON);
            assert!((series[0].samples[1].1 - 20.0).abs() < f64::EPSILON);
            assert!((series[0].samples[2].1 - 40.0).abs() < f64::EPSILON);

            // host2: steps at 0, 2s, 4s -> values 0, 40, 80
            assert_eq!(series[1].labels.get("instance").unwrap(), "host2");
            assert_eq!(series[1].samples.len(), 3, "expected 3 steps for host2");
            assert!((series[1].samples[0].1 - 0.0).abs() < f64::EPSILON);
            assert!((series[1].samples[1].1 - 40.0).abs() < f64::EPSILON);
            assert!((series[1].samples[2].1 - 80.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}

// ---- Counter reset handling ----

#[tokio::test]
async fn test_rate_with_counter_reset() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    // Counter: 0, 10, 20, 5 (reset), 15
    // Total increase with reset handling: 10+10+5+10 = 35 over 4s = 8.75/s
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "resets_total",
                "resets_total",
                "resets_total",
                "resets_total",
                "resets_total",
            ])),
            Arc::new(Int64Array::from(vec![
                0,
                1_000_000_000,
                2_000_000_000,
                3_000_000_000,
                4_000_000_000,
            ])),
            Arc::new(Float64Array::from(vec![0.0, 10.0, 20.0, 5.0, 15.0])),
            Arc::new(StringArray::from(vec![
                "host1", "host1", "host1", "host1", "host1",
            ])),
        ],
    )
    .unwrap();

    let source = InMemoryMetricSource::new(schema, vec![batch]);
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(4000).unwrap();
    let result = engine
        .instant_query("rate(resets_total[5s])", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert!(
                (samples[0].value - 8.75).abs() < f64::EPSILON,
                "expected rate 8.75, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---- Error cases ----

#[tokio::test]
async fn test_bare_matrix_selector_error() {
    let source = make_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let result = engine.instant_query("http_requests_total[5s]", ts).await;

    assert!(result.is_err(), "bare matrix selector should be an error");
}
