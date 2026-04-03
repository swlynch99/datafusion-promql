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

struct InMemorySource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

#[async_trait]
impl MetricSource for InMemorySource {
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
            name: "gauge".into(),
            label_names: vec!["host".into()],
            extra_columns: vec![],
        }])
    }
}

/// Build a source with both positive and negative sample values.
fn make_source_with_negatives() -> InMemorySource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("host", DataType::Utf8, false),
    ]));

    // host=a: value -5.0 at t=1000
    // host=b: value 3.0 at t=1000
    // host=c: value 0.0 at t=1000
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["gauge", "gauge", "gauge"])),
            Arc::new(Int64Array::from(vec![1000, 1000, 1000])),
            Arc::new(Float64Array::from(vec![-5.0, 3.0, 0.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .expect("failed to create batch");

    InMemorySource { schema, batches: vec![batch] }
}

#[tokio::test]
async fn test_abs_negates_negative_values() {
    let source = make_source_with_negatives();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine.instant_query("abs(gauge)", ts).await.unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 3, "expected 3 series");
            samples.sort_by(|a, b| a.labels.get("host").cmp(&b.labels.get("host")));

            // host=a: abs(-5.0) = 5.0
            assert_eq!(samples[0].labels.get("host").unwrap(), "a");
            assert!((samples[0].value - 5.0).abs() < f64::EPSILON, "expected 5.0, got {}", samples[0].value);

            // host=b: abs(3.0) = 3.0 (unchanged)
            assert_eq!(samples[1].labels.get("host").unwrap(), "b");
            assert!((samples[1].value - 3.0).abs() < f64::EPSILON, "expected 3.0, got {}", samples[1].value);

            // host=c: abs(0.0) = 0.0
            assert_eq!(samples[2].labels.get("host").unwrap(), "c");
            assert!((samples[2].value - 0.0).abs() < f64::EPSILON, "expected 0.0, got {}", samples[2].value);
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_abs_drops_name_label() {
    let source = make_source_with_negatives();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine.instant_query("abs(gauge)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            for sample in &samples {
                assert!(
                    !sample.labels.contains_key("__name__"),
                    "abs() should drop __name__ from result labels"
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_abs_preserves_other_labels() {
    let source = make_source_with_negatives();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine.instant_query("abs(gauge)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            for sample in &samples {
                assert!(
                    sample.labels.contains_key("host"),
                    "abs() should preserve non-name labels"
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_abs_range_query() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("host", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["gauge", "gauge", "gauge"])),
            Arc::new(Int64Array::from(vec![1000, 2000, 3000])),
            Arc::new(Float64Array::from(vec![-10.0, -20.0, -30.0])),
            Arc::new(StringArray::from(vec!["x", "x", "x"])),
        ],
    )
    .expect("failed to create batch");

    let source = InMemorySource { schema, batches: vec![batch] };
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(3000).unwrap();
    let step = std::time::Duration::from_millis(1000);

    let result = engine
        .range_query("abs(gauge)", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "expected 1 series");
            let s = &series[0];
            assert_eq!(s.samples.len(), 3, "expected 3 steps");

            // All values should be positive (absolute values of -10, -20, -30).
            let values: Vec<f64> = s.samples.iter().map(|(_, v)| *v).collect();
            assert!((values[0] - 10.0).abs() < f64::EPSILON, "expected 10.0 at step 0");
            assert!((values[1] - 20.0).abs() < f64::EPSILON, "expected 20.0 at step 1");
            assert!((values[2] - 30.0).abs() < f64::EPSILON, "expected 30.0 at step 2");
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}
