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

/// A simple in-memory metric source for testing.
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
        Ok(vec![MetricMeta {
            name: "cpu_usage".into(),
            label_names: vec!["instance".into(), "job".into()],
            extra_columns: vec![],
        }])
    }
}

fn make_test_source() -> InMemoryMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
        Field::new("job", DataType::Utf8, false),
    ]));

    // Create sample data: cpu_usage metric with two series over several timestamps.
    // Timestamps in milliseconds.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
            ])),
            Arc::new(Int64Array::from(vec![
                // Series 1: instance=host1 at t=1000, 2000, 3000, 4000
                1000, 2000, 3000, 4000,
                // Series 2: instance=host2 at t=1000, 2000, 3000, 4000
                1000, 2000, 3000, 4000,
            ])),
            Arc::new(Float64Array::from(vec![
                // Series 1 values
                10.0, 20.0, 30.0, 40.0, // Series 2 values
                50.0, 60.0, 70.0, 80.0,
            ])),
            Arc::new(StringArray::from(vec![
                "host1", "host1", "host1", "host1", "host2", "host2", "host2", "host2",
            ])),
            Arc::new(StringArray::from(vec![
                "node_exporter",
                "node_exporter",
                "node_exporter",
                "node_exporter",
                "node_exporter",
                "node_exporter",
                "node_exporter",
                "node_exporter",
            ])),
        ],
    )
    .expect("failed to create test batch");

    InMemoryMetricSource::new(schema, vec![batch])
}

#[tokio::test]
async fn test_instant_query_basic() {
    let source = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // Query at t=3000: should get the most recent sample at or before 3000.
    let ts = chrono::Utc.timestamp_millis_opt(3000).unwrap();
    let result = engine.instant_query("cpu_usage", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");

            // Sort by instance label for deterministic assertion.
            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1 at eval_ts=3000 should have value 30.0 (exact match at t=3000).
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert_eq!(samples[0].timestamp_ms, 3000);
            assert!((samples[0].value - 30.0).abs() < f64::EPSILON);

            // host2 at eval_ts=3000 should have value 70.0.
            assert_eq!(samples[1].labels.get("instance").unwrap(), "host2");
            assert_eq!(samples[1].timestamp_ms, 3000);
            assert!((samples[1].value - 70.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_instant_query_lookback_window() {
    let source = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // Query at t=3500: no exact match, but within the 5-minute lookback window
    // the most recent sample is at t=3000.
    let ts = chrono::Utc.timestamp_millis_opt(3500).unwrap();
    let result = engine.instant_query("cpu_usage", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");

            let mut samples = samples;
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: most recent sample before 3500 is at t=3000, value 30.0.
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert_eq!(samples[0].timestamp_ms, 3500); // eval timestamp
            assert!((samples[0].value - 30.0).abs() < f64::EPSILON);

            // host2: most recent sample before 3500 is at t=3000, value 70.0.
            assert_eq!(samples[1].labels.get("instance").unwrap(), "host2");
            assert_eq!(samples[1].timestamp_ms, 3500);
            assert!((samples[1].value - 70.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_ln_instant_query() {
    let source = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // ln(cpu_usage) at t=3000: values are 30.0 and 70.0.
    let ts = chrono::Utc.timestamp_millis_opt(3000).unwrap();
    let result = engine.instant_query("ln(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // ln(30.0)
            let expected_host1 = 30.0_f64.ln();
            assert!(
                (samples[0].value - expected_host1).abs() < 1e-10,
                "expected ln(30) = {expected_host1}, got {}",
                samples[0].value
            );

            // ln(70.0)
            let expected_host2 = 70.0_f64.ln();
            assert!(
                (samples[1].value - expected_host2).abs() < 1e-10,
                "expected ln(70) = {expected_host2}, got {}",
                samples[1].value
            );

            // __name__ should be dropped
            assert!(
                !samples[0].labels.contains_key("__name__"),
                "ln() should drop __name__"
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_instant_query_with_label_filter() {
    let source = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // Query with label matcher: should only return host1 series.
    let ts = chrono::Utc.timestamp_millis_opt(4000).unwrap();
    let result = engine
        .instant_query(r#"cpu_usage{instance="host1"}"#, ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series after filtering");
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert!((samples[0].value - 40.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// A metric source with fractional sample values, used to exercise ceil/floor/round.
fn make_fractional_source() -> InMemoryMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
        Field::new("job", DataType::Utf8, false),
    ]));

    // Two series at t=3000 with fractional values:
    //   host1: 1.2   -> ceil = 2.0
    //   host2: -1.7  -> ceil = -1.0
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["cpu_usage", "cpu_usage"])),
            Arc::new(Int64Array::from(vec![3000_i64, 3000_i64])),
            Arc::new(Float64Array::from(vec![1.2_f64, -1.7_f64])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
            Arc::new(StringArray::from(vec!["node_exporter", "node_exporter"])),
        ],
    )
    .expect("failed to create fractional test batch");

    InMemoryMetricSource::new(schema, vec![batch])
}

#[tokio::test]
async fn test_ceil_instant_query() {
    let source = make_fractional_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(3000).unwrap();
    let result = engine.instant_query("ceil(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");

            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: ceil(1.2) = 2.0
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert!(
                (samples[0].value - 2.0).abs() < f64::EPSILON,
                "expected ceil(1.2) = 2.0, got {}",
                samples[0].value
            );

            // host2: ceil(-1.7) = -1.0
            assert_eq!(samples[1].labels.get("instance").unwrap(), "host2");
            assert!(
                (samples[1].value - (-1.0)).abs() < f64::EPSILON,
                "expected ceil(-1.7) = -1.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_ceil_exact_integers_unchanged() {
    let source = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // host1=10.0 and host2=50.0 at t=1000: ceil leaves exact integers unchanged.
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let result = engine.instant_query("ceil(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert!(
                (samples[0].value - 10.0).abs() < f64::EPSILON,
                "expected ceil(10.0) = 10.0, got {}",
                samples[0].value
            );
            assert!(
                (samples[1].value - 50.0).abs() < f64::EPSILON,
                "expected ceil(50.0) = 50.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_ceil_with_label_filter() {
    let source = make_fractional_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(3000).unwrap();
    let result = engine
        .instant_query(r#"ceil(cpu_usage{instance="host1"})"#, ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series after filtering");
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert!(
                (samples[0].value - 2.0).abs() < f64::EPSILON,
                "expected ceil(1.2) = 2.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}
