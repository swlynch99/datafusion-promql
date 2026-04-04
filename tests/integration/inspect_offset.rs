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

/// In-memory metric source with data at known timestamps for offset testing.
///
/// Data layout (timestamps in seconds, stored as nanoseconds):
///   metric_name="cpu", instance="host1":
///     t=1000 -> 10.0
///     t=2000 -> 20.0
///     t=3000 -> 30.0
///     t=4000 -> 40.0
///     t=5000 -> 50.0
struct OffsetTestSource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

impl OffsetTestSource {
    fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("__name__", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("instance", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["cpu", "cpu", "cpu", "cpu", "cpu"])),
                Arc::new(Int64Array::from(vec![
                    1_000_000_000_000i64, // t=1000s
                    2_000_000_000_000,    // t=2000s
                    3_000_000_000_000,    // t=3000s
                    4_000_000_000_000,    // t=4000s
                    5_000_000_000_000,    // t=5000s
                ])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
                Arc::new(StringArray::from(vec![
                    "host1", "host1", "host1", "host1", "host1",
                ])),
            ],
        )
        .expect("failed to create offset test batch");

        Self {
            schema,
            batches: vec![batch],
        }
    }
}

#[async_trait]
impl MetricSource for OffsetTestSource {
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
            name: "cpu".into(),
            label_names: vec!["instance".into()],
            extra_columns: vec![],
        }])
    }
}

/// Instant query with offset: `cpu offset 2000s` at eval_ts=5000s should
/// return the value from t=3000s (5000 - 2000 = 3000), which is 30.0.
/// The result timestamp should still be 5000s.
#[tokio::test]
async fn test_instant_query_with_offset() {
    let source = OffsetTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap(); // 5000s
    let result = engine.instant_query("cpu offset 2000s", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series");
            // Result timestamp is the eval timestamp (5000s).
            assert_eq!(samples[0].timestamp_ns, 5_000_000_000_000);
            // Value should come from t=3000s (eval 5000s - offset 2000s).
            assert!(
                (samples[0].value - 30.0).abs() < f64::EPSILON,
                "expected 30.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Without offset, querying at t=5000s should return value 50.0.
/// This is a baseline comparison for the offset tests.
#[tokio::test]
async fn test_instant_query_without_offset_baseline() {
    let source = OffsetTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap();
    let result = engine.instant_query("cpu", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0].timestamp_ns, 5_000_000_000_000);
            assert!((samples[0].value - 50.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Offset with range function: `rate(cpu[2000s] offset 1000s)` at eval_ts=5000s.
///
/// With offset=1000s, the effective eval time is 4000s.
/// The range window [4000s - 2000s, 4000s] = [2000s, 4000s] captures:
///   t=2000: 20.0, t=3000: 30.0, t=4000: 40.0
///
/// rate = (last - first) / (last_ts - first_ts) = (40 - 20) / (4000 - 2000) = 0.01 per second
/// But timestamps are in nanoseconds, so rate = (40-20) / ((4000-2000) * 1e9) * 1e9
/// = 20 / 2000e9 * 1e9 = 20/2000 = 0.01
///
/// Actually rate in PromQL is per-second: (last - first) / (last_ts_seconds - first_ts_seconds)
/// The engine stores timestamps in nanoseconds, so:
/// rate = (40 - 20) / ((4_000_000_000_000 - 2_000_000_000_000) / 1e9) = 20 / 2000 = 0.01
#[tokio::test]
async fn test_rate_with_offset() {
    let source = OffsetTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap(); // 5000s
    let result = engine
        .instant_query("rate(cpu[2000s] offset 1000s)", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series");
            // Result timestamp is the eval timestamp.
            assert_eq!(samples[0].timestamp_ns, 5_000_000_000_000);
            // rate = (40 - 20) / 2000 = 0.01
            assert!(
                (samples[0].value - 0.01).abs() < 1e-9,
                "expected rate ~0.01, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Rate without offset for comparison: `rate(cpu[2000s])` at eval_ts=5000s.
///
/// Window [3000s, 5000s] captures: t=3000: 30.0, t=4000: 40.0, t=5000: 50.0
/// rate = (50 - 30) / 2000 = 0.01
#[tokio::test]
async fn test_rate_without_offset_baseline() {
    let source = OffsetTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap();
    let result = engine
        .instant_query("rate(cpu[2000s])", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0].timestamp_ns, 5_000_000_000_000);
            assert!(
                (samples[0].value - 0.01).abs() < 1e-9,
                "expected rate ~0.01, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Offset that goes beyond available data: `cpu offset 10000s` at eval_ts=5000s.
/// Effective lookup time = 5000 - 10000 = -5000s, which is before any data.
/// Should return an empty result.
#[tokio::test]
async fn test_offset_beyond_data_returns_empty() {
    let source = OffsetTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap();
    let result = engine
        .instant_query("cpu offset 10000s", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert!(
                samples.is_empty(),
                "expected empty result for offset beyond data, got {} samples",
                samples.len()
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Range query with offset: verifying that each step uses the offset.
///
/// Query: `cpu offset 1000s` over [3000s, 5000s] step 1000s
/// Step timestamps: 3000, 4000, 5000
/// Effective lookups: 2000, 3000, 4000
/// Expected values:   20.0, 30.0, 40.0
#[tokio::test]
async fn test_range_query_with_offset() {
    let source = OffsetTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(3_000_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap();
    let step = std::time::Duration::from_secs(1000);

    let result = engine
        .range_query("cpu offset 1000s", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series_list) => {
            assert_eq!(series_list.len(), 1, "expected 1 series");
            let series = &series_list[0];
            assert_eq!(series.samples.len(), 3, "expected 3 samples (3 steps)");

            // Step at 3000s -> lookup at 2000s -> value 20.0
            assert_eq!(series.samples[0].0, 3_000_000_000_000);
            assert!((series.samples[0].1 - 20.0).abs() < f64::EPSILON);

            // Step at 4000s -> lookup at 3000s -> value 30.0
            assert_eq!(series.samples[1].0, 4_000_000_000_000);
            assert!((series.samples[1].1 - 30.0).abs() < f64::EPSILON);

            // Step at 5000s -> lookup at 4000s -> value 40.0
            assert_eq!(series.samples[2].0, 5_000_000_000_000);
            assert!((series.samples[2].1 - 40.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}
