use std::sync::Arc;

use chrono::TimeZone;

use datafusion_promql::PromqlEngine;
use datafusion_promql::datasource::MetricSource;
use datafusion_promql::parquet::ParquetMetricSource;
use datafusion_promql::types::QueryResult;

/// Timestamp range of dump.parquet (nanosecond precision).
/// Data spans approximately 1775585840003432410 .. 1775585901001610090 ns (~61 seconds).
const DATA_START_NS: i64 = 1_775_585_840_003_432_410;
const DATA_END_NS: i64 = 1_775_585_901_001_610_090;

/// Pick a timestamp in the middle of the data range.
fn mid_timestamp() -> chrono::DateTime<chrono::Utc> {
    let mid_ns = (DATA_START_NS + DATA_END_NS) / 2;
    chrono::Utc.timestamp_nanos(mid_ns)
}

async fn make_engine() -> PromqlEngine {
    let source = ParquetMetricSource::try_new("data/dump.parquet")
        .await
        .expect("failed to create parquet source from dump.parquet");
    PromqlEngine::new(Arc::new(source))
}

/// Verify that the source initializes in a reasonable time (formerly took 100+ seconds).
#[tokio::test]
async fn test_dump_source_initializes() {
    let _engine = make_engine().await;
}

#[tokio::test]
async fn test_dump_list_metrics() {
    let source = ParquetMetricSource::try_new("data/dump.parquet")
        .await
        .unwrap();

    let metrics: Vec<_> = source.list_metrics(None).await.unwrap();
    assert!(
        !metrics.is_empty(),
        "expected at least one metric in dump.parquet"
    );

    let names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
    // These metrics should be present in any Rezolus-format dump.
    assert!(
        names.contains(&"blockio_bytes"),
        "missing blockio_bytes; found: {names:?}"
    );
}

#[tokio::test]
async fn test_dump_instant_query() {
    let engine = make_engine().await;
    let ts = mid_timestamp();

    let result = engine
        .instant_query("blockio_bytes", ts)
        .await
        .expect("instant query failed");

    match result {
        QueryResult::Vector(samples) => {
            assert!(
                !samples.is_empty(),
                "expected samples from blockio_bytes at {ts}"
            );
            // Samples should have an "op" label.
            for s in &samples {
                assert!(
                    s.labels.contains_key("op"),
                    "expected 'op' label, got: {:?}",
                    s.labels
                );
            }
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

#[tokio::test]
async fn test_dump_range_query() {
    let engine = make_engine().await;

    let start = chrono::Utc.timestamp_nanos(DATA_START_NS + 10_000_000_000); // +10s
    let end = chrono::Utc.timestamp_nanos(DATA_START_NS + 50_000_000_000); // +50s
    let step = std::time::Duration::from_secs(15);

    let result = engine
        .range_query("rate(blockio_bytes[15s])", start, end, step)
        .await
        .expect("range query failed");

    match result {
        QueryResult::Matrix(series) => {
            assert!(
                !series.is_empty(),
                "expected at least one series from rate(blockio_bytes[15s])"
            );
            for s in &series {
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
