use std::sync::Arc;

use arrow::array::{Float64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::TimeZone;
use datafusion::catalog::TableProvider;
use datafusion::datasource::MemTable;

use datafusion_promql::PromqlEngine;
use datafusion_promql::datasource::{Matcher, MetricMeta, MetricSource, TableFormat, ValueKind};
use datafusion_promql::error::Result;
use datafusion_promql::types::{QueryResult, TimeRange};

/// In-memory metric source with data at known timestamps for @ modifier testing.
///
/// Data layout (timestamps in seconds, stored as nanoseconds):
///   metric_name="cpu", instance="host1":
///     t=1000 -> 10.0
///     t=2000 -> 20.0
///     t=3000 -> 30.0
///     t=4000 -> 40.0
///     t=5000 -> 50.0
struct AtModifierTestSource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

impl AtModifierTestSource {
    fn new() -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("__name__", DataType::Utf8, false),
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("instance", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["cpu", "cpu", "cpu", "cpu", "cpu"])),
                Arc::new(UInt64Array::from(vec![
                    1_000_000_000_000u64, // t=1000s
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
        .expect("failed to create test batch");

        Self {
            schema,
            batches: vec![batch],
        }
    }
}

#[async_trait]
impl MetricSource for AtModifierTestSource {
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
        Ok(vec![MetricMeta {
            name: "cpu".into(),
            label_names: vec!["instance".into()],
            extra_columns: vec![],
        }])
    }
}

/// Instant query with @ modifier: `cpu @ 3000` at eval_ts=5000s should
/// return the value from t=3000s (looked up at @3000), which is 30.0.
/// The result timestamp should still be 5000s (the eval timestamp).
#[tokio::test]
async fn test_instant_query_with_at_modifier() {
    let source = AtModifierTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap(); // 5000s
    let result = engine.instant_query("cpu @ 3000", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series");
            // Result timestamp is the eval timestamp (5000s).
            assert_eq!(samples[0].timestamp_ns, 5_000_000_000_000);
            // Value should come from t=3000s (@ modifier).
            assert!(
                (samples[0].value - 30.0).abs() < f64::EPSILON,
                "expected 30.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Instant query with @ modifier pointing to t=1000: `cpu @ 1000` at eval_ts=5000s.
/// Should return value 10.0 from t=1000s, with result timestamp 5000s.
#[tokio::test]
async fn test_instant_query_at_earliest_data() {
    let source = AtModifierTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap();
    let result = engine.instant_query("cpu @ 1000", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series");
            assert_eq!(samples[0].timestamp_ns, 5_000_000_000_000);
            assert!(
                (samples[0].value - 10.0).abs() < f64::EPSILON,
                "expected 10.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// @ modifier combined with offset: `cpu @ 4000 offset 1000s` at eval_ts=5000s.
/// The @ pins lookup to 4000s, then offset shifts it back 1000s to 3000s.
/// Expected value: 30.0 (from t=3000s), reported at t=5000s.
#[tokio::test]
async fn test_instant_query_at_with_offset() {
    let source = AtModifierTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap();
    let result = engine
        .instant_query("cpu @ 4000 offset 1000s", ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "expected 1 series");
            assert_eq!(samples[0].timestamp_ns, 5_000_000_000_000);
            // Lookup at @4000 - offset 1000 = 3000, value 30.0
            assert!(
                (samples[0].value - 30.0).abs() < f64::EPSILON,
                "expected 30.0, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

/// Range query with @ modifier: `cpu @ 2000` over [3000s, 5000s] step 1000s.
/// Every step should look up at t=2000s and return value 20.0.
/// Output timestamps follow the step pattern: 3000, 4000, 5000.
#[tokio::test]
async fn test_range_query_with_at_modifier() {
    let source = AtModifierTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(3_000_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap();
    let step = std::time::Duration::from_secs(1000);

    let result = engine
        .range_query("cpu @ 2000", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series_list) => {
            assert_eq!(series_list.len(), 1, "expected 1 series");
            let series = &series_list[0];
            assert_eq!(series.samples.len(), 3, "expected 3 samples (3 steps)");

            // All steps look up at @2000 -> value 20.0
            for (i, &(ts, val)) in series.samples.iter().enumerate() {
                let expected_ts = (3000 + i as u64 * 1000) * 1_000_000_000;
                assert_eq!(ts, expected_ts, "step {i} timestamp mismatch");
                assert!(
                    (val - 20.0).abs() < f64::EPSILON,
                    "step {i}: expected 20.0, got {val}"
                );
            }
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}

/// Rate with @ modifier: `rate(cpu[2000s] @ 3000)` at eval_ts=5000s.
///
/// The @ modifier pins the range evaluation to t=3000s.
/// Window [3000s - 2000s, 3000s] = [1000s, 3000s] captures:
///   t=1000: 10.0, t=2000: 20.0, t=3000: 30.0
///
/// rate = (30 - 10) / 2000 = 0.01
#[tokio::test]
async fn test_rate_with_at_modifier() {
    let source = AtModifierTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let ts = chrono::Utc.timestamp_millis_opt(5_000_000).unwrap();
    let result = engine.instant_query("rate(cpu[2000s] @ 3000)", ts).await;

    match result {
        Ok(QueryResult::Vector(samples)) => {
            assert_eq!(samples.len(), 1, "expected 1 series");
            assert_eq!(samples[0].timestamp_ns, 5_000_000_000_000);
            // rate = (30 - 10) / 2000 = 0.01
            assert!(
                (samples[0].value - 0.01).abs() < 1e-9,
                "expected rate ~0.01, got {}",
                samples[0].value
            );
        }
        Ok(other) => panic!("expected Vector result, got {other:?}"),
        Err(e) => panic!("query error: {e:?}"),
    }
}

/// @ start() in a range query: `cpu @ start()` over [2000s, 4000s] step 1000s.
/// start() resolves to 2000s.
/// Every step looks up at t=2000s -> value 20.0.
#[tokio::test]
async fn test_at_start_in_range_query() {
    let source = AtModifierTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(2_000_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(4_000_000).unwrap();
    let step = std::time::Duration::from_secs(1000);

    let result = engine
        .range_query("cpu @ start()", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series_list) => {
            assert_eq!(series_list.len(), 1, "expected 1 series");
            let series = &series_list[0];
            assert_eq!(series.samples.len(), 3, "expected 3 samples");

            for (i, &(_ts, val)) in series.samples.iter().enumerate() {
                assert!(
                    (val - 20.0).abs() < f64::EPSILON,
                    "step {i}: expected 20.0 (@ start=2000s), got {val}"
                );
            }
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}

/// @ end() in a range query: `cpu @ end()` over [2000s, 4000s] step 1000s.
/// end() resolves to 4000s.
/// Every step looks up at t=4000s -> value 40.0.
#[tokio::test]
async fn test_at_end_in_range_query() {
    let source = AtModifierTestSource::new();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(2_000_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(4_000_000).unwrap();
    let step = std::time::Duration::from_secs(1000);

    let result = engine
        .range_query("cpu @ end()", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series_list) => {
            assert_eq!(series_list.len(), 1, "expected 1 series");
            let series = &series_list[0];
            assert_eq!(series.samples.len(), 3, "expected 3 samples");

            for (i, &(_ts, val)) in series.samples.iter().enumerate() {
                assert!(
                    (val - 40.0).abs() < f64::EPSILON,
                    "step {i}: expected 40.0 (@ end=4000s), got {val}"
                );
            }
        }
        other => panic!("expected Matrix result, got {other:?}"),
    }
}
