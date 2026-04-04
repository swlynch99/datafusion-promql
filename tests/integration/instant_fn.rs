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

struct InMemoryMetricSource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
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
            name: "mem_pages".into(),
            label_names: vec!["instance".into()],
            extra_columns: vec![],
        }])
    }
}

/// Build a test source with values that are exact powers of 2 so log2 results
/// are easy to assert precisely.
///
/// host1: value = 8.0   → log2 = 3.0
/// host2: value = 1024.0 → log2 = 10.0
fn make_pow2_source() -> InMemoryMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["mem_pages", "mem_pages"])),
            Arc::new(Int64Array::from(vec![1_000_000_000_i64, 1_000_000_000_i64])),
            Arc::new(Float64Array::from(vec![8.0, 1024.0])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
        ],
    )
    .expect("failed to create test batch");

    InMemoryMetricSource {
        schema,
        batches: vec![batch],
    }
}

#[tokio::test]
async fn test_log2_instant_query() {
    let source = make_pow2_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let result = engine.instant_query("log2(mem_pages)", ts).await.unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            // log2(8.0) == 3.0
            assert_eq!(samples[0].labels.get("instance").unwrap(), "host1");
            assert!(
                (samples[0].value - 3.0).abs() < f64::EPSILON,
                "log2(8) = 3.0, got {}",
                samples[0].value
            );

            // log2(1024.0) == 10.0
            assert_eq!(samples[1].labels.get("instance").unwrap(), "host2");
            assert!(
                (samples[1].value - 10.0).abs() < f64::EPSILON,
                "log2(1024) = 10.0, got {}",
                samples[1].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_log2_range_query() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    // Single series, two timestamps, values 4.0 and 16.0.
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["mem_pages", "mem_pages"])),
            Arc::new(Int64Array::from(vec![1_000_000_000_i64, 2_000_000_000_i64])),
            Arc::new(Float64Array::from(vec![4.0, 16.0])),
            Arc::new(StringArray::from(vec!["host1", "host1"])),
        ],
    )
    .expect("failed to create test batch");

    let source = InMemoryMetricSource {
        schema,
        batches: vec![batch],
    };
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(2000).unwrap();
    let step = std::time::Duration::from_millis(1000);

    let result = engine
        .range_query("log2(mem_pages)", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "expected 1 series");
            let s = &series[0];
            assert_eq!(s.samples.len(), 2, "expected 2 samples");

            // Step t=1000: value 4.0, log2(4) = 2.0
            assert!(
                (s.samples[0].1 - 2.0).abs() < f64::EPSILON,
                "log2(4) = 2.0, got {}",
                s.samples[0].1
            );
            // Step t=2000: value 16.0, log2(16) = 4.0
            assert!(
                (s.samples[1].1 - 4.0).abs() < f64::EPSILON,
                "log2(16) = 4.0, got {}",
                s.samples[1].1
            );
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}
