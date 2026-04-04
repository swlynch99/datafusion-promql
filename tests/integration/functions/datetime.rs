use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::{TimeZone, Utc};
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
            name: "test_metric".into(),
            label_names: vec!["instance".into()],
            extra_columns: vec![],
        }])
    }
}

/// Create a test source with data at a known timestamp.
///
/// Uses 2021-01-15 10:30:45 UTC (a Friday) as the sample timestamp.
/// - host1: value = 100.0
/// - host2: value = 200.0
fn make_test_source() -> (InMemoryMetricSource, i64) {
    // 2021-01-15 10:30:45 UTC
    let dt = Utc.with_ymd_and_hms(2021, 1, 15, 10, 30, 45).unwrap();
    let ts_ns = dt.timestamp_nanos_opt().unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["test_metric", "test_metric"])),
            Arc::new(Int64Array::from(vec![ts_ns, ts_ns])),
            Arc::new(Float64Array::from(vec![100.0, 200.0])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
        ],
    )
    .expect("failed to create test batch");

    (
        InMemoryMetricSource {
            schema,
            batches: vec![batch],
        },
        ts_ns,
    )
}

// ---- timestamp(v) ----

#[tokio::test]
async fn test_timestamp_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("timestamp(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2, "expected 2 series");
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            let expected_seconds = ts_ns as f64 / 1_000_000_000.0;
            for sample in &samples {
                assert!(
                    (sample.value - expected_seconds).abs() < 0.001,
                    "expected ~{expected_seconds}, got {}",
                    sample.value
                );
                // __name__ should be dropped
                assert!(!sample.labels.contains_key("__name__"));
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- hour(v) ----

#[tokio::test]
async fn test_hour_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("hour(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            // 2021-01-15 10:30:45 UTC → hour = 10
            for sample in &samples {
                assert_eq!(sample.value, 10.0, "expected hour=10, got {}", sample.value);
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- minute(v) ----

#[tokio::test]
async fn test_minute_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("minute(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            // 10:30:45 → minute = 30
            for sample in &samples {
                assert_eq!(
                    sample.value, 30.0,
                    "expected minute=30, got {}",
                    sample.value
                );
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- day_of_month(v) ----

#[tokio::test]
async fn test_day_of_month_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("day_of_month(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            // 2021-01-15 → day 15
            for sample in &samples {
                assert_eq!(
                    sample.value, 15.0,
                    "expected day_of_month=15, got {}",
                    sample.value
                );
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- day_of_week(v) ----

#[tokio::test]
async fn test_day_of_week_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("day_of_week(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            // 2021-01-15 is Friday → 5 (0=Sunday)
            for sample in &samples {
                assert_eq!(
                    sample.value, 5.0,
                    "expected day_of_week=5 (Friday), got {}",
                    sample.value
                );
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- day_of_year(v) ----

#[tokio::test]
async fn test_day_of_year_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("day_of_year(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            // 2021-01-15 → ordinal day 15
            for sample in &samples {
                assert_eq!(
                    sample.value, 15.0,
                    "expected day_of_year=15, got {}",
                    sample.value
                );
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- days_in_month(v) ----

#[tokio::test]
async fn test_days_in_month_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("days_in_month(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            // January has 31 days
            for sample in &samples {
                assert_eq!(
                    sample.value, 31.0,
                    "expected days_in_month=31, got {}",
                    sample.value
                );
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- month(v) ----

#[tokio::test]
async fn test_month_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("month(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            // January → month 1
            for sample in &samples {
                assert_eq!(sample.value, 1.0, "expected month=1, got {}", sample.value);
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- year(v) ----

#[tokio::test]
async fn test_year_function() {
    let (source, ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let eval_time = Utc.timestamp_millis_opt(ts_ns / 1_000_000).unwrap();

    let result = engine
        .instant_query("year(test_metric)", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            for sample in &samples {
                assert_eq!(
                    sample.value, 2021.0,
                    "expected year=2021, got {}",
                    sample.value
                );
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- time() ----

#[tokio::test]
async fn test_time_function_instant() {
    let (source, _ts_ns) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // Evaluate time() at a specific timestamp
    let eval_time = Utc.with_ymd_and_hms(2021, 6, 15, 12, 0, 0).unwrap();
    let expected_seconds = eval_time.timestamp() as f64;

    let result = engine.instant_query("time()", eval_time).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1, "time() should produce 1 sample");
            assert!(
                (samples[0].value - expected_seconds).abs() < 0.001,
                "expected ~{expected_seconds}, got {}",
                samples[0].value
            );
            // time() has no labels
            assert!(samples[0].labels.is_empty());
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- no-argument datetime functions ----

#[tokio::test]
async fn test_hour_no_args_instant() {
    let (source, _) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // Evaluate hour() at 14:00 UTC
    let eval_time = Utc.with_ymd_and_hms(2021, 3, 20, 14, 0, 0).unwrap();

    let result = engine.instant_query("hour()", eval_time).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert_eq!(
                samples[0].value, 14.0,
                "expected hour=14, got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

#[tokio::test]
async fn test_day_of_week_no_args_instant() {
    let (source, _) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    // 2021-03-20 is a Saturday → 6
    let eval_time = Utc.with_ymd_and_hms(2021, 3, 20, 0, 0, 0).unwrap();

    let result = engine
        .instant_query("day_of_week()", eval_time)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 1);
            assert_eq!(
                samples[0].value, 6.0,
                "expected day_of_week=6 (Saturday), got {}",
                samples[0].value
            );
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

// ---- Range query test for time() ----

#[tokio::test]
async fn test_time_function_range_query() {
    let (source, _) = make_test_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = Utc.with_ymd_and_hms(2021, 1, 1, 0, 0, 0).unwrap();
    let end = Utc.with_ymd_and_hms(2021, 1, 1, 0, 2, 0).unwrap();
    let step = std::time::Duration::from_secs(60);

    let result = engine
        .range_query("time()", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "time() should produce 1 series");
            let s = &series[0];
            assert_eq!(s.samples.len(), 3, "expected 3 steps (0m, 1m, 2m)");

            let start_secs = start.timestamp() as f64;
            for (i, (_ts, val)) in s.samples.iter().enumerate() {
                let expected = start_secs + (i as f64 * 60.0);
                assert!(
                    (val - expected).abs() < 0.001,
                    "step {i}: expected {expected}, got {val}"
                );
            }
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

// ---- Range query test for hour(v) ----

#[tokio::test]
async fn test_hour_range_query() {
    // Create source with data spanning multiple hours.
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    // Two samples: one at 10:00 UTC, one at 11:00 UTC
    let ts1 = Utc
        .with_ymd_and_hms(2021, 1, 15, 10, 0, 0)
        .unwrap()
        .timestamp_nanos_opt()
        .unwrap();
    let ts2 = Utc
        .with_ymd_and_hms(2021, 1, 15, 11, 0, 0)
        .unwrap()
        .timestamp_nanos_opt()
        .unwrap();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["test_metric", "test_metric"])),
            Arc::new(Int64Array::from(vec![ts1, ts2])),
            Arc::new(Float64Array::from(vec![42.0, 42.0])),
            Arc::new(StringArray::from(vec!["host1", "host1"])),
        ],
    )
    .unwrap();

    let source = InMemoryMetricSource {
        schema,
        batches: vec![batch],
    };
    let engine = PromqlEngine::new(Arc::new(source));

    let start = Utc.with_ymd_and_hms(2021, 1, 15, 10, 0, 0).unwrap();
    let end = Utc.with_ymd_and_hms(2021, 1, 15, 11, 0, 0).unwrap();
    let step = std::time::Duration::from_secs(3600);

    let result = engine
        .range_query("hour(test_metric)", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "expected 1 series");
            let s = &series[0];
            assert_eq!(s.samples.len(), 2, "expected 2 steps");

            // Step at 10:00 → hour = 10
            assert_eq!(s.samples[0].1, 10.0, "expected hour=10 at first step");
            // Step at 11:00 → hour = 11
            assert_eq!(s.samples[1].1, 11.0, "expected hour=11 at second step");
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}
