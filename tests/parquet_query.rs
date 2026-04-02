#![cfg(feature = "parquet")]

use std::sync::Arc;

use chrono::TimeZone;

use datafusion_promql::PromqlEngine;
use datafusion_promql::datasource::MetricSource;
use datafusion_promql::parquet::ParquetMetricSource;
use datafusion_promql::types::QueryResult;

/// Timestamp range of the test parquet file (nanoseconds converted to millis).
/// Data spans roughly 1750106216002 .. 1750106506001 (about 290 seconds).
const DATA_START_MS: i64 = 1_750_106_216_002;
const _DATA_END_MS: i64 = 1_750_106_506_001;

/// Pick a timestamp in the middle of the data range.
const MID_TS_MS: i64 = 1_750_106_360_000;

fn mid_timestamp() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc.timestamp_millis_opt(MID_TS_MS).unwrap()
}

async fn make_engine() -> PromqlEngine {
    let source = ParquetMetricSource::try_new("data/metrics.parquet")
        .await
        .expect("failed to create parquet source");
    PromqlEngine::new(Arc::new(source))
}

#[tokio::test]
async fn test_list_metrics() {
    let source = ParquetMetricSource::try_new("data/metrics.parquet")
        .await
        .unwrap();

    let metrics = source.list_metrics(None).await.unwrap();
    assert!(
        metrics.len() > 30,
        "expected many metrics, got {}",
        metrics.len()
    );

    // Check a few known metric names.
    let names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
    assert!(names.contains(&"cpu_cores"), "missing cpu_cores");
    assert!(names.contains(&"blockio_bytes"), "missing blockio_bytes");
    assert!(
        names.contains(&"cgroup_cpu_cycles"),
        "missing cgroup_cpu_cycles"
    );
}

#[tokio::test]
async fn test_simple_metric_instant_query() {
    let engine = make_engine().await;

    let result = engine.instant_query("cpu_cores", mid_timestamp()).await;
    let result = result.expect("query failed");

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(
                samples.len(),
                1,
                "cpu_cores should have 1 series (no labels)"
            );
            assert!(samples[0].value > 0.0, "cpu_cores should be positive");
            // __name__ label should be present.
            assert_eq!(samples[0].labels.get("__name__").unwrap(), "cpu_cores");
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

#[tokio::test]
async fn test_labeled_metric_instant_query() {
    let engine = make_engine().await;

    let result = engine
        .instant_query("blockio_bytes", mid_timestamp())
        .await
        .expect("query failed");

    match result {
        QueryResult::Vector(samples) => {
            assert!(
                samples.len() >= 2,
                "blockio_bytes should have multiple series (read, write, etc.), got {}",
                samples.len()
            );

            // All samples should have an "op" label.
            for s in &samples {
                assert!(
                    s.labels.contains_key("op"),
                    "expected 'op' label, got labels: {:?}",
                    s.labels
                );
            }

            // Check that "read" and "write" ops are present.
            let ops: Vec<&str> = samples
                .iter()
                .map(|s| s.labels.get("op").unwrap().as_str())
                .collect();
            assert!(ops.contains(&"read"), "missing op=read, got: {ops:?}");
            assert!(ops.contains(&"write"), "missing op=write, got: {ops:?}");
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

#[tokio::test]
async fn test_cgroup_metric_instant_query() {
    let engine = make_engine().await;

    let result = engine
        .instant_query("cgroup_cpu_cycles", mid_timestamp())
        .await
        .expect("query failed");

    match result {
        QueryResult::Vector(samples) => {
            assert!(
                samples.len() >= 5,
                "cgroup_cpu_cycles should have many series, got {}",
                samples.len()
            );

            // All samples should have "cgroup" and "id" labels.
            for s in &samples {
                assert!(
                    s.labels.contains_key("cgroup"),
                    "expected 'cgroup' label, got: {:?}",
                    s.labels
                );
                assert!(
                    s.labels.contains_key("id"),
                    "expected 'id' label, got: {:?}",
                    s.labels
                );
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_filter() {
    let engine = make_engine().await;

    let result = engine
        .instant_query(r#"blockio_bytes{op="read"}"#, mid_timestamp())
        .await
        .expect("query failed");

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(
                samples.len(),
                1,
                "expected 1 series for op=read, got {}",
                samples.len()
            );
            assert_eq!(samples[0].labels.get("op").unwrap(), "read");
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

#[tokio::test]
async fn test_range_query() {
    let engine = make_engine().await;

    // Use a time range that covers some of the data with a 60-second step.
    let start = chrono::Utc
        .timestamp_millis_opt(DATA_START_MS + 120_000)
        .unwrap();
    let end = chrono::Utc
        .timestamp_millis_opt(DATA_START_MS + 240_000)
        .unwrap();
    let step = std::time::Duration::from_secs(60);

    // Query rate of cpu_usage over 60s windows.
    let result = engine
        .range_query("rate(cpu_usage[60s])", start, end, step)
        .await
        .expect("query failed");

    match result {
        QueryResult::Matrix(series) => {
            assert!(
                !series.is_empty(),
                "expected at least one series from rate(cpu_usage[60s])"
            );
            for s in &series {
                assert!(
                    !s.samples.is_empty(),
                    "expected samples in series {:?}",
                    s.labels
                );
                // Rate values should be non-negative for a counter.
                for &(_, val) in &s.samples {
                    assert!(
                        val >= 0.0 || val.is_nan(),
                        "rate should be non-negative, got {val}"
                    );
                }
            }
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}
