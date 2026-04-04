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
            name: "metric".into(),
            label_names: vec!["host".into(), "env".into()],
            extra_columns: vec![],
        }])
    }
}

fn make_source() -> InMemorySource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("host", DataType::Utf8, false),
        Field::new("env", DataType::Utf8, false),
    ]));

    // Three series with different values and labels:
    // host=c, env=prod, value=10.0
    // host=a, env=staging, value=30.0
    // host=b, env=dev, value=20.0
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["metric", "metric", "metric"])),
            Arc::new(UInt64Array::from(vec![
                1_000_000_000,
                1_000_000_000,
                1_000_000_000,
            ])),
            Arc::new(Float64Array::from(vec![10.0, 30.0, 20.0])),
            Arc::new(StringArray::from(vec!["c", "a", "b"])),
            Arc::new(StringArray::from(vec!["prod", "staging", "dev"])),
        ],
    )
    .expect("failed to create batch");

    InMemorySource {
        schema,
        batches: vec![batch],
    }
}

#[tokio::test]
async fn test_sort_ascending_by_value() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine.instant_query("sort(metric)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3, "expected 3 series");

            let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
            assert!(
                (values[0] - 10.0).abs() < f64::EPSILON,
                "first should be 10.0, got {}",
                values[0]
            );
            assert!(
                (values[1] - 20.0).abs() < f64::EPSILON,
                "second should be 20.0, got {}",
                values[1]
            );
            assert!(
                (values[2] - 30.0).abs() < f64::EPSILON,
                "third should be 30.0, got {}",
                values[2]
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_sort_desc_by_value() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine.instant_query("sort_desc(metric)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3, "expected 3 series");

            let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
            assert!(
                (values[0] - 30.0).abs() < f64::EPSILON,
                "first should be 30.0, got {}",
                values[0]
            );
            assert!(
                (values[1] - 20.0).abs() < f64::EPSILON,
                "second should be 20.0, got {}",
                values[1]
            );
            assert!(
                (values[2] - 10.0).abs() < f64::EPSILON,
                "third should be 10.0, got {}",
                values[2]
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_sort_preserves_name_label() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine.instant_query("sort(metric)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            for sample in &samples {
                assert!(
                    sample.labels.contains_key("__name__"),
                    "sort() should preserve __name__ label"
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_sort_by_label_ascending() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine
        .instant_query("sort_by_label(metric, \"host\")", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3, "expected 3 series");

            let hosts: Vec<&str> = samples
                .iter()
                .map(|s| s.labels.get("host").unwrap().as_str())
                .collect();
            assert_eq!(hosts, vec!["a", "b", "c"], "should be sorted by host asc");
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_sort_by_label_desc() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine
        .instant_query("sort_by_label_desc(metric, \"host\")", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 3, "expected 3 series");

            let hosts: Vec<&str> = samples
                .iter()
                .map(|s| s.labels.get("host").unwrap().as_str())
                .collect();
            assert_eq!(hosts, vec!["c", "b", "a"], "should be sorted by host desc");
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_sort_by_multiple_labels() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("env", DataType::Utf8, false),
        Field::new("host", DataType::Utf8, false),
    ]));

    // Create data where env has duplicates to test secondary sort by host.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "metric", "metric", "metric", "metric",
            ])),
            Arc::new(UInt64Array::from(vec![
                1_000_000_000,
                1_000_000_000,
                1_000_000_000,
                1_000_000_000,
            ])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(StringArray::from(vec!["prod", "dev", "prod", "dev"])),
            Arc::new(StringArray::from(vec!["b", "a", "a", "b"])),
        ],
    )
    .expect("failed to create batch");

    let source = InMemorySource {
        schema,
        batches: vec![batch],
    };
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine
        .instant_query("sort_by_label(metric, \"env\", \"host\")", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 4, "expected 4 series");

            let labels: Vec<(&str, &str)> = samples
                .iter()
                .map(|s| {
                    (
                        s.labels.get("env").unwrap().as_str(),
                        s.labels.get("host").unwrap().as_str(),
                    )
                })
                .collect();
            // Should be sorted by env first, then host within same env.
            assert_eq!(
                labels,
                vec![("dev", "a"), ("dev", "b"), ("prod", "a"), ("prod", "b")],
                "should be sorted by env then host"
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_sort_range_query() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("host", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "metric", "metric", "metric", "metric", "metric", "metric",
            ])),
            Arc::new(UInt64Array::from(vec![
                1_000_000_000,
                1_000_000_000,
                1_000_000_000,
                2_000_000_000,
                2_000_000_000,
                2_000_000_000,
            ])),
            Arc::new(Float64Array::from(vec![30.0, 10.0, 20.0, 5.0, 15.0, 25.0])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "a", "b", "c"])),
        ],
    )
    .expect("failed to create batch");

    let source = InMemorySource {
        schema,
        batches: vec![batch],
    };
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let step = std::time::Duration::from_millis(1000);

    // sort() on a range query should still return valid matrix results.
    let result = engine
        .range_query("sort(metric)", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 3, "expected 3 series");
            // Each series should have 2 samples (one per step).
            for s in &series {
                assert_eq!(s.samples.len(), 2, "expected 2 samples per series");
            }
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}
