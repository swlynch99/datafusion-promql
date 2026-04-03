//! Tests that construct in-memory wide-format DataFrames using the Rezolus
//! column naming convention, then execute PromQL queries against them.
//!
//! This allows precise control over input data and expected results without
//! needing a real parquet file.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::{Float64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::TimeZone;
use datafusion::catalog::TableProvider;
use datafusion::datasource::MemTable;

use datafusion_promql::PromqlEngine;
use datafusion_promql::datasource::{
    ColumnMapping, Matcher, MetricMeta, MetricSource, TableFormat,
};
use datafusion_promql::parquet::{rezolus_column_mapping, rezolus_parse_column};
use datafusion_promql::types::{QueryResult, TimeRange};

// ---------------------------------------------------------------------------
// Test helper: RezolusMockSource
// ---------------------------------------------------------------------------

/// A builder for constructing in-memory wide-format tables that mimic Rezolus
/// parquet files. Timestamps are specified in milliseconds and stored internally
/// as nanosecond UInt64 values (matching real Rezolus data).
struct RezolusMockBuilder {
    /// Timestamps in milliseconds.
    timestamps_ms: Vec<i64>,
    /// (column_name, values) pairs. Values must have same length as timestamps.
    columns: Vec<(String, Vec<f64>)>,
}

impl RezolusMockBuilder {
    fn new() -> Self {
        Self {
            timestamps_ms: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// Set timestamps in milliseconds. Internally stored as nanosecond UInt64.
    fn timestamps_ms(mut self, ts: Vec<i64>) -> Self {
        self.timestamps_ms = ts;
        self
    }

    /// Add a metric column using rezolus naming convention.
    /// E.g. "cpu_cores", "blockio_bytes/read", "cgroup_cpu_cycles//path/id".
    fn column(mut self, name: &str, values: Vec<f64>) -> Self {
        assert_eq!(
            values.len(),
            self.timestamps_ms.len(),
            "column '{}' length mismatch: got {} values but {} timestamps",
            name,
            values.len(),
            self.timestamps_ms.len()
        );
        self.columns.push((name.to_string(), values));
        self
    }

    /// Build the mock source.
    fn build(self) -> RezolusMockSource {
        let n = self.timestamps_ms.len();

        // Build schema: timestamp (UInt64) + one Float64 column per metric.
        let mut fields = vec![Field::new("timestamp", DataType::UInt64, false)];
        for (name, _) in &self.columns {
            fields.push(Field::new(name, DataType::Float64, false));
        }
        let schema = Arc::new(Schema::new(fields));

        // Build arrays.
        let ts_nanos: Vec<u64> = self
            .timestamps_ms
            .iter()
            .map(|ms| (*ms as u64) * 1_000_000)
            .collect();
        let mut arrays: Vec<Arc<dyn arrow::array::Array>> =
            vec![Arc::new(UInt64Array::from(ts_nanos))];
        for (_, values) in &self.columns {
            arrays.push(Arc::new(Float64Array::from(values.clone())));
        }

        let batch =
            RecordBatch::try_new(Arc::clone(&schema), arrays).expect("failed to build RecordBatch");

        // Build metric metadata from columns.
        let mapping = rezolus_column_mapping();
        let mut metric_labels: std::collections::BTreeMap<String, BTreeSet<String>> =
            std::collections::BTreeMap::new();
        for (col_name, _) in &self.columns {
            if let Some((metric_name, labels)) = rezolus_parse_column(col_name) {
                let entry = metric_labels.entry(metric_name).or_default();
                for key in labels.keys() {
                    entry.insert(key.clone());
                }
            }
        }
        let metrics: Vec<MetricMeta> = metric_labels
            .into_iter()
            .map(|(name, label_names)| MetricMeta {
                name,
                label_names: label_names.into_iter().collect(),
                extra_columns: vec![],
            })
            .collect();

        let table = MemTable::try_new(Arc::clone(&schema), vec![vec![batch]])
            .expect("failed to build MemTable");

        RezolusMockSource {
            table: Arc::new(table),
            mapping,
            metrics,
            _n: n,
        }
    }
}

struct RezolusMockSource {
    table: Arc<dyn TableProvider>,
    mapping: ColumnMapping,
    metrics: Vec<MetricMeta>,
    _n: usize,
}

#[async_trait]
impl MetricSource for RezolusMockSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> datafusion_promql::error::Result<(Arc<dyn TableProvider>, TableFormat)> {
        Ok((
            Arc::clone(&self.table),
            TableFormat::Wide(self.mapping.clone()),
        ))
    }

    async fn list_metrics(
        &self,
        _name_matcher: Option<&Matcher>,
    ) -> datafusion_promql::error::Result<Vec<MetricMeta>> {
        Ok(self.metrics.clone())
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn ts(ms: i64) -> chrono::DateTime<chrono::Utc> {
    chrono::Utc.timestamp_millis_opt(ms).unwrap()
}

fn engine(source: RezolusMockSource) -> PromqlEngine {
    PromqlEngine::new(Arc::new(source))
}

/// Extract vector samples sorted by a key for deterministic assertions.
fn sorted_vector(result: QueryResult) -> Vec<(std::collections::BTreeMap<String, String>, f64)> {
    match result {
        QueryResult::Vector(mut samples) => {
            samples.sort_by(|a, b| a.labels.cmp(&b.labels));
            samples.into_iter().map(|s| (s.labels, s.value)).collect()
        }
        other => panic!("expected Vector, got {other:?}"),
    }
}

fn assert_matrix(result: &QueryResult) -> &Vec<datafusion_promql::types::RangeSamples> {
    match result {
        QueryResult::Matrix(series) => series,
        other => panic!("expected Matrix, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Tests: Plain metric (no labels)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_plain_metric_instant() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000, 4000, 5000])
        .column("cpu_cores", vec![4.0, 4.0, 4.0, 4.0, 4.0])
        .build();

    let e = engine(source);
    let result = e.instant_query("cpu_cores", ts(3000)).await.unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].0.get("__name__").unwrap(), "cpu_cores");
    assert_eq!(samples[0].1, 4.0);
}

#[tokio::test]
async fn test_plain_metric_value_changes() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("temperature", vec![20.0, 25.0, 30.0])
        .build();

    let e = engine(source);

    // Query at each timestamp, verify correct value returned.
    let r1 = e.instant_query("temperature", ts(1000)).await.unwrap();
    assert_eq!(sorted_vector(r1)[0].1, 20.0);

    let r2 = e.instant_query("temperature", ts(2000)).await.unwrap();
    assert_eq!(sorted_vector(r2)[0].1, 25.0);

    let r3 = e.instant_query("temperature", ts(3000)).await.unwrap();
    assert_eq!(sorted_vector(r3)[0].1, 30.0);
}

// ---------------------------------------------------------------------------
// Tests: Labeled metrics (single slash → op label)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_labeled_metric_all_series() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("blockio_bytes/read", vec![100.0, 200.0, 300.0])
        .column("blockio_bytes/write", vec![50.0, 100.0, 150.0])
        .build();

    let e = engine(source);
    let result = e.instant_query("blockio_bytes", ts(3000)).await.unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    // Sorted by labels, "read" < "write".
    assert_eq!(samples[0].0.get("op").unwrap(), "read");
    assert_eq!(samples[0].1, 300.0);
    assert_eq!(samples[1].0.get("op").unwrap(), "write");
    assert_eq!(samples[1].1, 150.0);
}

#[tokio::test]
async fn test_labeled_metric_filter_by_label() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("blockio_bytes/read", vec![100.0, 200.0, 300.0])
        .column("blockio_bytes/write", vec![50.0, 100.0, 150.0])
        .build();

    let e = engine(source);
    let result = e
        .instant_query(r#"blockio_bytes{op="read"}"#, ts(3000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].0.get("op").unwrap(), "read");
    assert_eq!(samples[0].1, 300.0);
}

#[tokio::test]
async fn test_labeled_metric_not_equal_filter() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("blockio_bytes/read", vec![100.0, 200.0, 300.0])
        .column("blockio_bytes/write", vec![50.0, 100.0, 150.0])
        .column("blockio_bytes/discard", vec![10.0, 20.0, 30.0])
        .build();

    let e = engine(source);
    let result = e
        .instant_query(r#"blockio_bytes{op!="read"}"#, ts(3000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);
    let ops: Vec<&str> = samples
        .iter()
        .map(|(l, _)| l.get("op").unwrap().as_str())
        .collect();
    assert!(ops.contains(&"write"));
    assert!(ops.contains(&"discard"));
}

// ---------------------------------------------------------------------------
// Tests: Two-level slash metrics (metric/op/id)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_two_level_slash_metric() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("softirq/net_rx/0", vec![100.0, 200.0, 300.0])
        .column("softirq/net_rx/1", vec![110.0, 210.0, 310.0])
        .column("softirq/net_tx/0", vec![50.0, 60.0, 70.0])
        .build();

    let e = engine(source);
    let result = e.instant_query("softirq", ts(3000)).await.unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 3);

    // All should have op and id labels.
    for (labels, _) in &samples {
        assert!(labels.contains_key("op"));
        assert!(labels.contains_key("id"));
    }
}

#[tokio::test]
async fn test_two_level_slash_filter_by_op() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("softirq/net_rx/0", vec![100.0, 200.0, 300.0])
        .column("softirq/net_rx/1", vec![110.0, 210.0, 310.0])
        .column("softirq/net_tx/0", vec![50.0, 60.0, 70.0])
        .build();

    let e = engine(source);
    let result = e
        .instant_query(r#"softirq{op="net_rx"}"#, ts(3000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);
    for (labels, _) in &samples {
        assert_eq!(labels.get("op").unwrap(), "net_rx");
    }
}

#[tokio::test]
async fn test_two_level_slash_filter_by_id() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("softirq/net_rx/0", vec![100.0, 200.0, 300.0])
        .column("softirq/net_rx/1", vec![110.0, 210.0, 310.0])
        .column("softirq/net_tx/0", vec![50.0, 60.0, 70.0])
        .build();

    let e = engine(source);
    let result = e
        .instant_query(r#"softirq{id="0"}"#, ts(3000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);
    for (labels, _) in &samples {
        assert_eq!(labels.get("id").unwrap(), "0");
    }
}

// ---------------------------------------------------------------------------
// Tests: Cgroup metrics (double-slash)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_cgroup_metric() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column(
            "cgroup_cpu_cycles//system.slice/chrony.service/28",
            vec![1000.0, 2000.0, 3000.0],
        )
        .column(
            "cgroup_cpu_cycles//system.slice/sshd.service/42",
            vec![500.0, 600.0, 700.0],
        )
        .build();

    let e = engine(source);
    let result = e
        .instant_query("cgroup_cpu_cycles", ts(3000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    for (labels, _) in &samples {
        assert!(labels.contains_key("cgroup"));
        assert!(labels.contains_key("id"));
    }

    // Check specific label values.
    let chrony = samples
        .iter()
        .find(|(l, _)| l.get("id").unwrap() == "28")
        .unwrap();
    assert_eq!(
        chrony.0.get("cgroup").unwrap(),
        "/system.slice/chrony.service"
    );
    assert_eq!(chrony.1, 3000.0);
}

#[tokio::test]
async fn test_cgroup_filter_by_cgroup_label() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column(
            "cgroup_cpu_cycles//system.slice/chrony.service/28",
            vec![1000.0, 2000.0, 3000.0],
        )
        .column(
            "cgroup_cpu_cycles//system.slice/sshd.service/42",
            vec![500.0, 600.0, 700.0],
        )
        .build();

    let e = engine(source);
    let result = e
        .instant_query(
            r#"cgroup_cpu_cycles{cgroup="/system.slice/sshd.service"}"#,
            ts(3000),
        )
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].0.get("id").unwrap(), "42");
    assert_eq!(samples[0].1, 700.0);
}

#[tokio::test]
async fn test_cgroup_root_path() {
    // cgroup_cpu_cycles///1 → cgroup="/", id="1"
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("cgroup_cpu_cycles///1", vec![100.0, 200.0, 300.0])
        .build();

    let e = engine(source);
    let result = e
        .instant_query("cgroup_cpu_cycles", ts(3000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].0.get("cgroup").unwrap(), "/");
    assert_eq!(samples[0].0.get("id").unwrap(), "1");
}

// ---------------------------------------------------------------------------
// Tests: Rate function on rezolus counter data
// ---------------------------------------------------------------------------

/// Build a counter source: values increase linearly at known rates.
fn make_counter_source() -> RezolusMockSource {
    // 11 timestamps at 1-second intervals.
    let timestamps: Vec<i64> = (0..11).map(|i| 10_000 + i * 1000).collect();

    // blockio_bytes/read grows at 100 bytes/sec
    let read_vals: Vec<f64> = (0..11).map(|i| (i * 100) as f64).collect();
    // blockio_bytes/write grows at 50 bytes/sec
    let write_vals: Vec<f64> = (0..11).map(|i| (i * 50) as f64).collect();

    RezolusMockBuilder::new()
        .timestamps_ms(timestamps)
        .column("blockio_bytes/read", read_vals)
        .column("blockio_bytes/write", write_vals)
        .build()
}

#[tokio::test]
async fn test_rate_on_rezolus_counter() {
    let e = engine(make_counter_source());

    // rate(blockio_bytes[5s]) at t=15000 (covers samples from 10000..15000).
    let result = e
        .instant_query("rate(blockio_bytes[5s])", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    let read = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "read")
        .unwrap();
    let write = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "write")
        .unwrap();

    // rate should be ~100.0/sec for read, ~50.0/sec for write.
    assert!(
        (read.1 - 100.0).abs() < 1.0,
        "expected rate ~100, got {}",
        read.1
    );
    assert!(
        (write.1 - 50.0).abs() < 1.0,
        "expected rate ~50, got {}",
        write.1
    );
}

#[tokio::test]
async fn test_irate_on_rezolus_counter() {
    let e = engine(make_counter_source());

    let result = e
        .instant_query("irate(blockio_bytes[5s])", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    let read = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "read")
        .unwrap();
    // irate uses last two samples: (14000, 400) and (15000, 500) → 100/sec.
    assert!(
        (read.1 - 100.0).abs() < 1.0,
        "expected irate ~100, got {}",
        read.1
    );
}

#[tokio::test]
async fn test_increase_on_rezolus_counter() {
    let e = engine(make_counter_source());

    let result = e
        .instant_query("increase(blockio_bytes[5s])", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    let read = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "read")
        .unwrap();
    // increase over 5 seconds at 100/sec = 500.
    assert!(
        (read.1 - 500.0).abs() < 5.0,
        "expected increase ~500, got {}",
        read.1
    );
}

// ---------------------------------------------------------------------------
// Tests: Delta function on gauge data
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_delta_on_gauge() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000, 4000, 5000])
        .column("temperature", vec![20.0, 22.0, 21.0, 25.0, 23.0])
        .build();

    let e = engine(source);
    let result = e
        .instant_query("delta(temperature[4s])", ts(5000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);
    // delta window [4s] at t=5000 covers [1000, 5000].
    // First sample: t=1000 (20.0), last sample: t=5000 (23.0).
    // delta = 23.0 - 20.0 = 3.0
    assert!(
        (samples[0].1 - 3.0).abs() < 0.1,
        "expected delta ~3.0, got {}",
        samples[0].1
    );
}

// ---------------------------------------------------------------------------
// Tests: Range query (rate over time range with steps)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_range_query_rate_rezolus() {
    let e = engine(make_counter_source());

    let start = ts(14000);
    let end = ts(20000);
    let step = std::time::Duration::from_secs(2);

    let result = e
        .range_query("rate(blockio_bytes[5s])", start, end, step)
        .await
        .unwrap();

    let series = assert_matrix(&result);
    assert!(!series.is_empty(), "expected at least one series");

    // Find the read series.
    let read_series = series
        .iter()
        .find(|s| s.labels.get("op").unwrap() == "read")
        .expect("missing read series");

    // Should have multiple step points.
    assert!(
        read_series.samples.len() >= 2,
        "expected multiple samples, got {}",
        read_series.samples.len()
    );

    // All rate values should be ~100.0 (linear counter).
    for &(_, val) in &read_series.samples {
        if !val.is_nan() {
            assert!((val - 100.0).abs() < 5.0, "expected rate ~100, got {val}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: Aggregations on rezolus data
// ---------------------------------------------------------------------------

fn make_multi_series_source() -> RezolusMockSource {
    let timestamps: Vec<i64> = (0..6).map(|i| 10_000 + i * 1000).collect();

    RezolusMockBuilder::new()
        .timestamps_ms(timestamps)
        // softirq/net_rx/0 grows at 10/sec
        .column(
            "softirq/net_rx/0",
            (0..6).map(|i| (i * 10) as f64).collect(),
        )
        // softirq/net_rx/1 grows at 20/sec
        .column(
            "softirq/net_rx/1",
            (0..6).map(|i| (i * 20) as f64).collect(),
        )
        // softirq/net_tx/0 grows at 5/sec
        .column("softirq/net_tx/0", (0..6).map(|i| (i * 5) as f64).collect())
        .build()
}

#[tokio::test]
async fn test_sum_by_op() {
    let e = engine(make_multi_series_source());

    // Sum rate by op label.
    let result = e
        .instant_query("sum(rate(softirq[5s])) by (op)", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    let net_rx = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "net_rx")
        .unwrap();
    let net_tx = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "net_tx")
        .unwrap();

    // net_rx: rate(id=0) ≈ 10 + rate(id=1) ≈ 20 = 30
    assert!(
        (net_rx.1 - 30.0).abs() < 2.0,
        "expected sum(rate) ~30 for net_rx, got {}",
        net_rx.1
    );
    // net_tx: rate(id=0) ≈ 5
    assert!(
        (net_tx.1 - 5.0).abs() < 2.0,
        "expected sum(rate) ~5 for net_tx, got {}",
        net_tx.1
    );
}

#[tokio::test]
async fn test_sum_by_id() {
    let e = engine(make_multi_series_source());

    let result = e
        .instant_query("sum(rate(softirq[5s])) by (id)", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    let id0 = samples
        .iter()
        .find(|(l, _)| l.get("id").unwrap() == "0")
        .unwrap();
    let id1 = samples
        .iter()
        .find(|(l, _)| l.get("id").unwrap() == "1")
        .unwrap();

    // id=0: rate(net_rx/0) ≈ 10 + rate(net_tx/0) ≈ 5 = 15
    assert!(
        (id0.1 - 15.0).abs() < 2.0,
        "expected sum(rate) ~15 for id=0, got {}",
        id0.1
    );
    // id=1: rate(net_rx/1) ≈ 20
    assert!(
        (id1.1 - 20.0).abs() < 2.0,
        "expected sum(rate) ~20 for id=1, got {}",
        id1.1
    );
}

#[tokio::test]
async fn test_sum_total() {
    let e = engine(make_multi_series_source());

    let result = e
        .instant_query("sum(rate(softirq[5s]))", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);

    // Total: 10 + 20 + 5 = 35
    assert!(
        (samples[0].1 - 35.0).abs() < 3.0,
        "expected sum(rate) ~35, got {}",
        samples[0].1
    );
}

#[tokio::test]
async fn test_avg_by_op() {
    let e = engine(make_multi_series_source());

    let result = e
        .instant_query("avg(rate(softirq[5s])) by (op)", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    let net_rx = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "net_rx")
        .unwrap();

    // net_rx avg: (10 + 20) / 2 = 15
    assert!(
        (net_rx.1 - 15.0).abs() < 2.0,
        "expected avg(rate) ~15 for net_rx, got {}",
        net_rx.1
    );
}

#[tokio::test]
async fn test_count_series() {
    let e = engine(make_multi_series_source());

    let result = e
        .instant_query("count(rate(softirq[5s]))", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].1, 3.0, "expected 3 series");
}

#[tokio::test]
async fn test_min_max() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("gauge/a", vec![10.0, 20.0, 30.0])
        .column("gauge/b", vec![5.0, 50.0, 15.0])
        .column("gauge/c", vec![100.0, 1.0, 25.0])
        .build();

    let e = engine(source);

    let min_result = e.instant_query("min(gauge)", ts(3000)).await.unwrap();
    let min_samples = sorted_vector(min_result);
    assert_eq!(min_samples.len(), 1);
    assert_eq!(min_samples[0].1, 15.0); // min of 30, 15, 25

    let max_result = e.instant_query("max(gauge)", ts(3000)).await.unwrap();
    let max_samples = sorted_vector(max_result);
    assert_eq!(max_samples.len(), 1);
    assert_eq!(max_samples[0].1, 30.0); // max of 30, 15, 25
}

// ---------------------------------------------------------------------------
// Tests: Binary operations on rezolus data
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_vector_scalar_multiply() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("blockio_bytes/read", vec![100.0, 200.0, 300.0])
        .column("blockio_bytes/write", vec![50.0, 100.0, 150.0])
        .build();

    let e = engine(source);
    let result = e
        .instant_query("blockio_bytes * 2", ts(3000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    let read = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "read")
        .unwrap();
    assert_eq!(read.1, 600.0);

    let write = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "write")
        .unwrap();
    assert_eq!(write.1, 300.0);
}

#[tokio::test]
async fn test_vector_scalar_addition() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("cpu_cores", vec![4.0, 4.0, 4.0])
        .build();

    let e = engine(source);
    let result = e.instant_query("cpu_cores + 10", ts(3000)).await.unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].1, 14.0);
}

#[tokio::test]
async fn test_comparison_filter() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("gauge/a", vec![10.0, 20.0, 30.0])
        .column("gauge/b", vec![5.0, 50.0, 15.0])
        .column("gauge/c", vec![100.0, 1.0, 25.0])
        .build();

    let e = engine(source);
    // Filter: keep only series where value > 20 at t=3000.
    let result = e.instant_query("gauge > 20", ts(3000)).await.unwrap();

    let samples = sorted_vector(result);
    // At t=3000: a=30, b=15, c=25 → a and c pass.
    assert_eq!(samples.len(), 2);
    let ops: Vec<&str> = samples
        .iter()
        .map(|(l, _)| l.get("op").unwrap().as_str())
        .collect();
    assert!(ops.contains(&"a"));
    assert!(ops.contains(&"c"));
}

#[tokio::test]
async fn test_unary_negation() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("cpu_cores", vec![4.0, 4.0, 4.0])
        .build();

    let e = engine(source);
    let result = e.instant_query("-cpu_cores", ts(3000)).await.unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].1, -4.0);
}

// ---------------------------------------------------------------------------
// Tests: Multiple metrics in same source
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_multiple_independent_metrics() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("cpu_cores", vec![4.0, 4.0, 4.0])
        .column("memory_total", vec![8e9, 8e9, 8e9])
        .column("blockio_bytes/read", vec![100.0, 200.0, 300.0])
        .build();

    let e = engine(source);

    // Query each metric independently.
    let cpu = e.instant_query("cpu_cores", ts(3000)).await.unwrap();
    assert_eq!(sorted_vector(cpu)[0].1, 4.0);

    let mem = e.instant_query("memory_total", ts(3000)).await.unwrap();
    assert_eq!(sorted_vector(mem)[0].1, 8e9);

    let bio = e.instant_query("blockio_bytes", ts(3000)).await.unwrap();
    assert_eq!(sorted_vector(bio).len(), 1);
    assert_eq!(
        sorted_vector(e.instant_query("blockio_bytes", ts(3000)).await.unwrap())[0].1,
        300.0
    );
}

// ---------------------------------------------------------------------------
// Tests: Lookback window behavior with rezolus data
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_lookback_window() {
    let source = RezolusMockBuilder::new()
        .timestamps_ms(vec![1000, 2000, 3000])
        .column("cpu_cores", vec![2.0, 4.0, 8.0])
        .build();

    let e = engine(source);

    // Query slightly after last sample → should still find it within lookback.
    let result = e.instant_query("cpu_cores", ts(3500)).await.unwrap();
    assert_eq!(sorted_vector(result)[0].1, 8.0);

    // Query way beyond lookback (5 minute default = 300s).
    let result = e.instant_query("cpu_cores", ts(400_000)).await.unwrap();
    let samples = sorted_vector(result);
    assert!(samples.is_empty(), "expected no samples beyond lookback");
}

// ---------------------------------------------------------------------------
// Tests: Range query producing matrix output
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_range_query_plain_selector() {
    let timestamps: Vec<i64> = (0..11).map(|i| 10_000 + i * 1000).collect();
    let values: Vec<f64> = (0..11).map(|i| (i * 10) as f64).collect();

    let source = RezolusMockBuilder::new()
        .timestamps_ms(timestamps)
        .column("cpu_usage", values)
        .build();

    let e = engine(source);

    let result = e
        .range_query(
            "rate(cpu_usage[5s])",
            ts(14000),
            ts(18000),
            std::time::Duration::from_secs(2),
        )
        .await
        .unwrap();

    let series = assert_matrix(&result);
    assert_eq!(series.len(), 1, "expected 1 series");

    // Rate of a counter growing at 10/sec should be ~10.
    for &(_, val) in &series[0].samples {
        if !val.is_nan() {
            assert!((val - 10.0).abs() < 1.0, "expected rate ~10, got {val}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: sum without syntax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_sum_without_id() {
    let e = engine(make_multi_series_source());

    // `sum without (id)` should group by remaining labels (op).
    let result = e
        .instant_query("sum(rate(softirq[5s])) without (id)", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2, "expected 2 groups (net_rx, net_tx)");

    let net_rx = samples
        .iter()
        .find(|(l, _)| l.get("op").unwrap() == "net_rx")
        .unwrap();
    // net_rx: 10 + 20 = 30
    assert!(
        (net_rx.1 - 30.0).abs() < 2.0,
        "expected ~30, got {}",
        net_rx.1
    );
}

// ---------------------------------------------------------------------------
// Tests: Cgroup metrics with rate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rate_cgroup_metric() {
    let timestamps: Vec<i64> = (0..11).map(|i| 10_000 + i * 1000).collect();

    let source = RezolusMockBuilder::new()
        .timestamps_ms(timestamps.clone())
        // Grows at 1000 cycles/sec.
        .column(
            "cgroup_cpu_cycles//system.slice/foo.service/10",
            (0..11).map(|i| (i * 1000) as f64).collect(),
        )
        // Grows at 500 cycles/sec.
        .column(
            "cgroup_cpu_cycles//system.slice/bar.service/20",
            (0..11).map(|i| (i * 500) as f64).collect(),
        )
        .build();

    let e = engine(source);
    let result = e
        .instant_query("rate(cgroup_cpu_cycles[5s])", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    assert_eq!(samples.len(), 2);

    let foo = samples
        .iter()
        .find(|(l, _)| l.get("id").unwrap() == "10")
        .unwrap();
    assert!(
        (foo.1 - 1000.0).abs() < 10.0,
        "expected rate ~1000, got {}",
        foo.1
    );

    let bar = samples
        .iter()
        .find(|(l, _)| l.get("id").unwrap() == "20")
        .unwrap();
    assert!(
        (bar.1 - 500.0).abs() < 10.0,
        "expected rate ~500, got {}",
        bar.1
    );
}

#[tokio::test]
async fn test_sum_rate_cgroup_by_cgroup() {
    let timestamps: Vec<i64> = (0..11).map(|i| 10_000 + i * 1000).collect();

    let source = RezolusMockBuilder::new()
        .timestamps_ms(timestamps.clone())
        .column(
            "cgroup_cpu_cycles//system.slice/foo.service/10",
            (0..11).map(|i| (i * 1000) as f64).collect(),
        )
        .column(
            "cgroup_cpu_cycles//system.slice/foo.service/20",
            (0..11).map(|i| (i * 500) as f64).collect(),
        )
        .build();

    let e = engine(source);
    let result = e
        .instant_query("sum(rate(cgroup_cpu_cycles[5s])) by (cgroup)", ts(15000))
        .await
        .unwrap();

    let samples = sorted_vector(result);
    // Both have same cgroup, so they sum together.
    assert_eq!(samples.len(), 1);
    assert!(
        (samples[0].1 - 1500.0).abs() < 20.0,
        "expected sum ~1500, got {}",
        samples[0].1
    );
}
