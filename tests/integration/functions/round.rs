use std::sync::Arc;

use arrow::array::{Float64Array, StringArray, UInt64Array};
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
            label_names: vec!["instance".into()],
            extra_columns: vec![],
        }])
    }
}

/// Build a test source with two series at a single timestamp.
/// host1 value: 2.3, host2 value: 2.7
fn make_source(values: Vec<f64>) -> InMemoryMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let n = values.len();
    let names: Vec<&str> = (0..n).map(|_| "cpu_usage").collect();
    let timestamps: Vec<u64> = (0..n).map(|_| 1_000_000_000).collect();
    let instances: Vec<String> = (0..n).map(|i| format!("host{}", i + 1)).collect();
    let instance_refs: Vec<&str> = instances.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(UInt64Array::from(timestamps)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(instance_refs)),
        ],
    )
    .expect("failed to create test batch");

    InMemoryMetricSource::new(schema, vec![batch])
}

#[tokio::test]
async fn test_round_default_to_nearest() {
    // round(v) with default to_nearest=1: rounds to nearest integer.
    let source = make_source(vec![2.3, 2.7]);
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let result = engine.instant_query("round(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: round(2.3) = 2
            assert!(
                (samples[0].value - 2.0).abs() < f64::EPSILON,
                "expected 2.0, got {}",
                samples[0].value
            );
            // host2: round(2.7) = 3
            assert!(
                (samples[1].value - 3.0).abs() < f64::EPSILON,
                "expected 3.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_round_with_to_nearest() {
    // round(v, 0.5): rounds to nearest 0.5.
    // 2.3: floor(2.3/0.5 + 0.5)*0.5 = floor(5.1)*0.5 = 2.5
    // 2.8: floor(2.8/0.5 + 0.5)*0.5 = floor(6.1)*0.5 = 3.0
    let source = make_source(vec![2.3, 2.8]);
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let result = engine
        .instant_query("round(cpu_usage, 0.5)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: round(2.3, 0.5) = 2.5
            assert!(
                (samples[0].value - 2.5).abs() < 1e-10,
                "expected 2.5, got {}",
                samples[0].value
            );
            // host2: round(2.8, 0.5) = 3.0
            assert!(
                (samples[1].value - 3.0).abs() < 1e-10,
                "expected 3.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_round_to_nearest_integer_multiple() {
    // round(v, 5): rounds to nearest multiple of 5.
    let source = make_source(vec![12.0, 13.0]);
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let result = engine
        .instant_query("round(cpu_usage, 5)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // host1: round(12.0, 5) = 10
            assert!(
                (samples[0].value - 10.0).abs() < f64::EPSILON,
                "expected 10.0, got {}",
                samples[0].value
            );
            // host2: round(13.0, 5) = 15
            assert!(
                (samples[1].value - 15.0).abs() < f64::EPSILON,
                "expected 15.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_round_preserves_labels() {
    // round() should keep the same labels as the input vector.
    let source = make_source(vec![1.6]);
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let result = engine.instant_query("round(cpu_usage)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert!(
                (samples[0].value - 2.0).abs() < f64::EPSILON,
                "expected 2.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}
