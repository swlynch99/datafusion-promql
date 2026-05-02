/// Integration tests for the streaming sliding-window path that handles
/// `irate` and `idelta` queries.
///
/// These tests exercise the `StreamingRangeFuncExec` physical node produced by
/// the `RangeVectorToAggregation` optimizer rule for those two functions.  Each
/// test also implicitly verifies that the result matches the semantics of the
/// equivalent Prometheus expression.
use std::sync::Arc;
use std::time::Duration;

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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct InMemorySource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

impl InMemorySource {
    fn new(schema: Arc<Schema>, batches: Vec<RecordBatch>) -> Self {
        Self { schema, batches }
    }
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
        Ok((
            Arc::new(table),
            TableFormat::Long {
                value_kind: ValueKind::Scalar,
            },
        ))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![])
    }
}

// ---------------------------------------------------------------------------
// irate — range query over multiple eval timestamps
// ---------------------------------------------------------------------------

/// Monotonically increasing counter at 10 req/s, sampled every second.
fn counter_source_single_series() -> InMemorySource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));
    let n: u64 = 11; // t = 0 .. 10 s
    let names: Vec<&str> = (0..n).map(|_| "reqs").collect();
    let timestamps: Vec<u64> = (0..n).map(|i| i * 1_000_000_000).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
    let instances: Vec<&str> = (0..n).map(|_| "host1").collect();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(UInt64Array::from(timestamps)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(instances)),
        ],
    )
    .unwrap();
    InMemorySource::new(schema, vec![batch])
}

#[tokio::test]
async fn test_streaming_irate_range_query() {
    // irate(reqs[5s]) over [5s, 10s] step 5s.
    //
    // At t=5s  : window [0s,5s],  last two samples = (4s,40) and (5s,50)
    //            irate = (50-40)/(1s) = 10.0
    // At t=10s : window [5s,10s], last two samples = (9s,90) and (10s,100)
    //            irate = (100-90)/(1s) = 10.0
    let source = counter_source_single_series();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(5_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let step = Duration::from_secs(5);

    let result = engine
        .range_query("irate(reqs[5s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "expected 1 series");
            let s = &series[0];
            assert_eq!(s.samples.len(), 2, "expected 2 steps");
            assert!(
                (s.samples[0].1 - 10.0).abs() < f64::EPSILON,
                "t=5s: expected irate 10.0, got {}",
                s.samples[0].1
            );
            assert!(
                (s.samples[1].1 - 10.0).abs() < f64::EPSILON,
                "t=10s: expected irate 10.0, got {}",
                s.samples[1].1
            );
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// irate — counter reset in the window
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_streaming_irate_counter_reset() {
    // Counter resets between the last two samples in the window.
    // Samples: (0s,0), (1s,10), (2s,20), (3s,5), (4s,15)
    //                                      ^ reset
    // irate at t=4s with [0s,4s] window:
    //   last two = (3s,5) and (4s,15) — no reset between these two
    //   irate = (15-5)/1s = 10.0
    //
    // irate at t=3s with [0s,3s] window:
    //   last two = (2s,20) and (3s,5) — reset detected (5 < 20)
    //   increase = 5 (just the new value), dt = 1s
    //   irate = 5.0
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["c", "c", "c", "c", "c"])),
            Arc::new(UInt64Array::from(vec![
                0,
                1_000_000_000,
                2_000_000_000,
                3_000_000_000,
                4_000_000_000,
            ])),
            Arc::new(Float64Array::from(vec![0.0, 10.0, 20.0, 5.0, 15.0])),
            Arc::new(StringArray::from(vec!["h1", "h1", "h1", "h1", "h1"])),
        ],
    )
    .unwrap();
    let source = InMemorySource::new(schema, vec![batch]);
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(3_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(4_000).unwrap();
    let step = Duration::from_secs(1);

    let result = engine
        .range_query("irate(c[5s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1);
            let s = &series[0];
            assert_eq!(s.samples.len(), 2);
            // t=3s: reset case → irate = 5.0/s
            assert!(
                (s.samples[0].1 - 5.0).abs() < f64::EPSILON,
                "t=3s: expected irate 5.0 (reset), got {}",
                s.samples[0].1
            );
            // t=4s: normal → irate = 10.0/s
            assert!(
                (s.samples[1].1 - 10.0).abs() < f64::EPSILON,
                "t=4s: expected irate 10.0, got {}",
                s.samples[1].1
            );
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// irate — multiple series, same range query
// ---------------------------------------------------------------------------

/// Two counter series: series A at 10 req/s, series B at 30 req/s.
fn two_series_counter_source() -> InMemorySource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("svc", DataType::Utf8, false),
    ]));

    let n: u64 = 6; // t = 0 .. 5 s
    let mut names: Vec<&str> = Vec::new();
    let mut timestamps: Vec<u64> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut svcs: Vec<&str> = Vec::new();

    for i in 0..n {
        names.push("requests");
        timestamps.push(i * 1_000_000_000);
        values.push(i as f64 * 10.0);
        svcs.push("a");
    }
    for i in 0..n {
        names.push("requests");
        timestamps.push(i * 1_000_000_000);
        values.push(i as f64 * 30.0);
        svcs.push("b");
    }

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(UInt64Array::from(timestamps)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(svcs)),
        ],
    )
    .unwrap();
    InMemorySource::new(schema, vec![batch])
}

#[tokio::test]
async fn test_streaming_irate_multi_series() {
    // irate(requests[3s]) over [3s, 5s] step 2s with two series.
    //
    // Series a (10/s): last two before each eval_t:
    //   t=3s: (2s,20),(3s,30) → irate=(30-20)/1=10
    //   t=5s: (4s,40),(5s,50) → irate=(50-40)/1=10
    //
    // Series b (30/s): last two:
    //   t=3s: (2s,60),(3s,90) → irate=(90-60)/1=30
    //   t=5s: (4s,120),(5s,150) → irate=(150-120)/1=30
    let source = two_series_counter_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(3_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(5_000).unwrap();
    let step = Duration::from_secs(2);

    let result = engine
        .range_query("irate(requests[3s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(mut series) => {
            assert_eq!(series.len(), 2, "expected 2 series");
            series.sort_by(|a, b| a.labels.get("svc").cmp(&b.labels.get("svc")));

            // Series a
            assert_eq!(series[0].labels.get("svc").unwrap(), "a");
            assert_eq!(series[0].samples.len(), 2);
            for &(ts, val) in &series[0].samples {
                assert!(
                    (val - 10.0).abs() < f64::EPSILON,
                    "series a: expected irate 10.0, got {val} at ts={ts}"
                );
            }

            // Series b
            assert_eq!(series[1].labels.get("svc").unwrap(), "b");
            assert_eq!(series[1].samples.len(), 2);
            for &(ts, val) in &series[1].samples {
                assert!(
                    (val - 30.0).abs() < f64::EPSILON,
                    "series b: expected irate 30.0, got {val} at ts={ts}"
                );
            }
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// irate — eval timestamp with only one sample in window (no output row)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_streaming_irate_insufficient_samples() {
    // Only one data point exists, so irate always returns nothing.
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["m"])),
            Arc::new(UInt64Array::from(vec![5_000_000_000u64])),
            Arc::new(Float64Array::from(vec![100.0])),
            Arc::new(StringArray::from(vec!["h1"])),
        ],
    )
    .unwrap();
    let source = InMemorySource::new(schema, vec![batch]);
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(4_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(8_000).unwrap();
    let step = Duration::from_secs(2);

    let result = engine
        .range_query("irate(m[5s])", start, end, step)
        .await
        .unwrap();

    // With only one sample in any window, irate returns no data.
    match result {
        QueryResult::Matrix(series) => {
            assert!(
                series.is_empty(),
                "expected no series (insufficient samples), got {}",
                series.len()
            );
        }
        QueryResult::Vector(samples) => {
            assert!(samples.is_empty(), "expected empty vector, got {samples:?}");
        }
        other => panic!("unexpected result type: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// idelta — range query
// ---------------------------------------------------------------------------

/// Gauge with values that go up and down.
fn gauge_source() -> InMemorySource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("sensor", DataType::Utf8, false),
    ]));
    // values: 20, 22, 25, 23, 21, 24, 26
    let gauge_values = [20.0f64, 22.0, 25.0, 23.0, 21.0, 24.0, 26.0];
    let n = gauge_values.len() as u64;
    let names: Vec<&str> = (0..n).map(|_| "temp").collect();
    let timestamps: Vec<u64> = (0..n).map(|i| i * 1_000_000_000).collect();
    let values: Vec<f64> = gauge_values.to_vec();
    let sensors: Vec<&str> = (0..n).map(|_| "room1").collect();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(UInt64Array::from(timestamps)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(sensors)),
        ],
    )
    .unwrap();
    InMemorySource::new(schema, vec![batch])
}

#[tokio::test]
async fn test_streaming_idelta_range_query() {
    // idelta(temp[3s]) over [3s, 6s] step 3s.
    //
    // Values:  t=0→20, t=1→22, t=2→25, t=3→23, t=4→21, t=5→24, t=6→26
    //
    // At t=3s : window [0s,3s] → last two = (2s,25) and (3s,23)
    //           idelta = 23-25 = -2.0
    // At t=6s : window [3s,6s] → last two = (5s,24) and (6s,26)
    //           idelta = 26-24 = 2.0
    let source = gauge_source();
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(3_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(6_000).unwrap();
    let step = Duration::from_secs(3);

    let result = engine
        .range_query("idelta(temp[3s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "expected 1 series");
            let s = &series[0];
            assert_eq!(s.samples.len(), 2, "expected 2 steps");
            assert!(
                (s.samples[0].1 - (-2.0)).abs() < f64::EPSILON,
                "t=3s: expected idelta -2.0, got {}",
                s.samples[0].1
            );
            assert!(
                (s.samples[1].1 - 2.0).abs() < f64::EPSILON,
                "t=6s: expected idelta 2.0, got {}",
                s.samples[1].1
            );
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// irate — window slides correctly: samples before window_start are evicted
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_streaming_irate_window_eviction() {
    // Counter sampled every second at 10/s.  We use a narrow 2s window so
    // that only 2-3 samples fall inside, and check that results at several
    // consecutive eval timestamps are all correct.
    //
    // Samples: t=0→0, t=1→10, t=2→20, t=3→30, t=4→40, t=5→50
    // irate(m[2s]) step 1s from t=2s to t=5s:
    //   t=2s : window [0,2] → last two = (1s,10),(2s,20) → irate=10/s
    //   t=3s : window [1,3] → last two = (2s,20),(3s,30) → irate=10/s
    //   t=4s : window [2,4] → last two = (3s,30),(4s,40) → irate=10/s
    //   t=5s : window [3,5] → last two = (4s,40),(5s,50) → irate=10/s
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));
    let n: u64 = 6;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from((0..n).map(|_| "m").collect::<Vec<_>>())),
            Arc::new(UInt64Array::from(
                (0..n).map(|i| i * 1_000_000_000).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..n).map(|i| i as f64 * 10.0).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from((0..n).map(|_| "h").collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    let source = InMemorySource::new(schema, vec![batch]);
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(2_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(5_000).unwrap();
    let step = Duration::from_secs(1);

    let result = engine
        .range_query("irate(m[2s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1);
            let s = &series[0];
            assert_eq!(s.samples.len(), 4, "expected 4 steps (t=2..5)");
            for &(ts, val) in &s.samples {
                assert!(
                    (val - 10.0).abs() < f64::EPSILON,
                    "expected irate=10.0 at ts={ts}, got {val}"
                );
            }
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// irate — eval timestamps beyond the last sample (no data should be emitted)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_streaming_irate_eval_beyond_data() {
    // Data ends at t=3s.  Eval timestamps at t=5s and t=10s have no data
    // in their [eval-5s, eval] windows (5s window, but data only exists 0-3s).
    // t=5s: window [0s,5s] → has data (0..3s), last two = (2s,20),(3s,30) → irate=10
    // t=10s: window [5s,10s] → NO data → no output row
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["m", "m", "m", "m"])),
            Arc::new(UInt64Array::from(vec![
                0u64,
                1_000_000_000,
                2_000_000_000,
                3_000_000_000,
            ])),
            Arc::new(Float64Array::from(vec![0.0, 10.0, 20.0, 30.0])),
            Arc::new(StringArray::from(vec!["h", "h", "h", "h"])),
        ],
    )
    .unwrap();
    let source = InMemorySource::new(schema, vec![batch]);
    let engine = PromqlEngine::new(Arc::new(source));

    let start = chrono::Utc.timestamp_millis_opt(5_000).unwrap();
    let end = chrono::Utc.timestamp_millis_opt(10_000).unwrap();
    let step = Duration::from_secs(5);

    let result = engine
        .range_query("irate(m[5s])", start, end, step)
        .await
        .unwrap();

    match result {
        QueryResult::Matrix(series) => {
            assert_eq!(series.len(), 1, "expected 1 series (t=5s has data)");
            let s = &series[0];
            assert_eq!(s.samples.len(), 1, "expected only t=5s sample");
            assert!(
                (s.samples[0].1 - 10.0).abs() < f64::EPSILON,
                "t=5s: expected irate 10.0, got {}",
                s.samples[0].1
            );
        }
        other => panic!("expected Matrix, got {other:?}"),
    }
}
