//! Tests for [`MetrikenMetricSource`] — fixture parquet → in-memory schema.

use std::sync::Arc;

use arrow::array::{Array, AsArray, Int64Array, ListArray, StructArray, UInt64Array};
use arrow::datatypes::{DataType, UInt64Type};
use datafusion::catalog::TableProvider;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::collect;
use datafusion_promql::datasource::{MetricSource, TableFormat};
use datafusion_promql::histogram::{HistogramConfig, is_histogram_column};
use datafusion_promql::metriken_parquet::MetrikenMetricSource;
use datafusion_promql::types::TimeRange;
use metriken_exposition::ParquetHistogramType;

use super::fixture::{FakeSnapshot, write_fixture};

/// Convenience: collect all batches a `TableProvider` would scan with no
/// projection / filters / limit.
async fn scan_all(provider: Arc<dyn TableProvider>) -> Vec<arrow::record_batch::RecordBatch> {
    let ctx = SessionContext::new();
    let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
    collect(plan, ctx.task_ctx()).await.unwrap()
}

/// Drives a histogram metric "lat" through the source: bucket index 2 grows
/// 0 → 0 → 3 → 5 → 1 (reset).  Returns the (timestamp, indices, counts) seen
/// per row.
fn build_three_snapshot_fixture(histogram_type: ParquetHistogramType) -> tempfile::NamedTempFile {
    // Use grouping_power=1 max_value_power=3 → 12 buckets (matches
    // metriken-exposition's own test fixtures).
    let snapshots = vec![
        // ts=1: cumulative buckets: index 2 has count 0
        FakeSnapshot::new(1).histogram("lat", 1, 3, vec![0, 0, 0, 0, 0, 0], &[("op", "read")]),
        // ts=2: index 2 grew to 3
        FakeSnapshot::new(2).histogram("lat", 1, 3, vec![0, 0, 3, 0, 0, 0], &[("op", "read")]),
        // ts=3: index 2 grew to 5, index 5 grew from 0 to 2
        FakeSnapshot::new(3).histogram("lat", 1, 3, vec![0, 0, 5, 0, 0, 2], &[("op", "read")]),
        // ts=4: counter reset — index 2 dropped to 1
        FakeSnapshot::new(4).histogram("lat", 1, 3, vec![0, 0, 1, 0, 0, 0], &[("op", "read")]),
    ];
    write_fixture(snapshots, histogram_type)
}

#[tokio::test]
async fn standard_histogram_schema_and_deltas() {
    let fixture = build_three_snapshot_fixture(ParquetHistogramType::Standard);
    let source = MetrikenMetricSource::try_new(fixture.path()).unwrap();

    // Schema: timestamp + lat (struct) — counter/gauge/duration absent.
    let schema = source.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["timestamp", "lat"]);

    let lat_field = schema.field_with_name("lat").unwrap();
    assert!(
        is_histogram_column(lat_field),
        "lat must be recognized as a histogram column"
    );
    assert_eq!(
        HistogramConfig::from_field(lat_field),
        Some(HistogramConfig::new(1, 3))
    );
    // The user label "op" survived on the field; metric_type was scrubbed.
    assert_eq!(lat_field.metadata().get("op"), Some(&"read".to_string()));
    assert!(!lat_field.metadata().contains_key("metric_type"));

    // list_metrics returns one metric "lat" with label "op".
    let metrics = source.list_metrics(None).await.unwrap();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].name, "lat");
    assert_eq!(metrics[0].label_names, vec!["op".to_string()]);

    // table_for_metric returns Wide format.
    let (provider, format) = source
        .table_for_metric("lat", &[], TimeRange::unbounded())
        .await
        .unwrap();
    assert!(matches!(format, TableFormat::Wide(_)));

    let batches = scan_all(provider).await;

    // Stitch all batches together for inspection.
    let mut all_ts: Vec<u64> = Vec::new();
    let mut all_idx: Vec<Vec<u64>> = Vec::new();
    let mut all_cnt: Vec<Vec<u64>> = Vec::new();

    for b in &batches {
        let ts = b
            .column_by_name("timestamp")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let st = b
            .column_by_name("lat")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let idx_list = st
            .column_by_name("indices")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let cnt_list = st
            .column_by_name("counts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        for row in 0..b.num_rows() {
            all_ts.push(ts.value(row));
            let i = idx_list.value(row);
            let c = cnt_list.value(row);
            all_idx.push(
                i.as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect(),
            );
            all_cnt.push(
                c.as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect(),
            );
        }
    }

    assert_eq!(all_ts.len(), 4);
    // ts=1 first row: empty delta (no prev).
    assert_eq!(all_idx[0], Vec::<u64>::new());
    assert_eq!(all_cnt[0], Vec::<u64>::new());
    // ts=2: bucket 2 grew 0→3.
    assert_eq!(all_idx[1], vec![2]);
    assert_eq!(all_cnt[1], vec![3]);
    // ts=3: bucket 2 grew 3→5 (delta 2), bucket 5 grew 0→2.
    assert_eq!(all_idx[2], vec![2, 5]);
    assert_eq!(all_cnt[2], vec![2, 2]);
    // ts=4: counter reset (bucket 2 dropped) → empty delta keeping the
    // timestamp axis aligned.
    assert_eq!(all_idx[3], Vec::<u64>::new());
    assert_eq!(all_cnt[3], Vec::<u64>::new());
}

#[tokio::test]
async fn sparse_histogram_schema_and_deltas() {
    let fixture = build_three_snapshot_fixture(ParquetHistogramType::Sparse);
    let source = MetrikenMetricSource::try_new(fixture.path()).unwrap();

    // Schema must still expose a single fused histogram column "lat", not
    // the on-disk pair of indices / counts columns.
    let schema = source.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["timestamp", "lat"]);

    let lat_field = schema.field_with_name("lat").unwrap();
    assert!(is_histogram_column(lat_field));
    assert_eq!(
        HistogramConfig::from_field(lat_field),
        Some(HistogramConfig::new(1, 3))
    );

    let (provider, _) = source
        .table_for_metric("lat", &[], TimeRange::unbounded())
        .await
        .unwrap();
    let batches = scan_all(provider).await;

    // Sanity: same delta values regardless of on-disk shape.
    let mut all_idx: Vec<Vec<u64>> = Vec::new();
    let mut all_cnt: Vec<Vec<u64>> = Vec::new();
    for b in &batches {
        let st = b
            .column_by_name("lat")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let idx_list = st
            .column_by_name("indices")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let cnt_list = st
            .column_by_name("counts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        for row in 0..b.num_rows() {
            let i = idx_list.value(row);
            let c = cnt_list.value(row);
            all_idx.push(
                i.as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect(),
            );
            all_cnt.push(
                c.as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect(),
            );
        }
    }
    assert_eq!(all_idx.len(), 4);
    assert_eq!(all_idx[0], Vec::<u64>::new());
    assert_eq!(all_cnt[0], Vec::<u64>::new());
    assert_eq!(all_idx[1], vec![2]);
    assert_eq!(all_cnt[1], vec![3]);
    assert_eq!(all_idx[2], vec![2, 5]);
    assert_eq!(all_cnt[2], vec![2, 2]);
    assert_eq!(all_idx[3], Vec::<u64>::new());
    assert_eq!(all_cnt[3], Vec::<u64>::new());
}

#[tokio::test]
async fn scalar_columns_are_passed_through() {
    let snapshots = vec![
        FakeSnapshot::new(10)
            .counter("cpu_cycles", 100, &[])
            .gauge("temp", 50, &[]),
        FakeSnapshot::new(20)
            .counter("cpu_cycles", 250, &[])
            .gauge("temp", 55, &[]),
    ];
    let fixture = write_fixture(snapshots, ParquetHistogramType::Standard);
    let source = MetrikenMetricSource::try_new(fixture.path()).unwrap();

    // Schema includes timestamp + both scalar metrics, no others.
    let schema = source.schema();
    let mut names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["cpu_cycles", "temp", "timestamp"]);

    let cpu_field = schema.field_with_name("cpu_cycles").unwrap();
    assert_eq!(cpu_field.data_type(), &DataType::UInt64);
    assert!(!is_histogram_column(cpu_field));

    let temp_field = schema.field_with_name("temp").unwrap();
    assert_eq!(temp_field.data_type(), &DataType::Int64);

    // list_metrics returns exactly the two scalar metrics.
    let metrics = source.list_metrics(None).await.unwrap();
    let mut names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["cpu_cycles", "temp"]);

    // Sample values are passed through verbatim.
    let (provider, format) = source
        .table_for_metric("cpu_cycles", &[], TimeRange::unbounded())
        .await
        .unwrap();
    assert!(matches!(format, TableFormat::Wide(_)));
    let batches = scan_all(provider).await;
    let mut cpu_vals: Vec<u64> = Vec::new();
    let mut temp_vals: Vec<i64> = Vec::new();
    for b in &batches {
        let cpu = b
            .column_by_name("cpu_cycles")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let temp = b
            .column_by_name("temp")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..b.num_rows() {
            cpu_vals.push(cpu.value(row));
            temp_vals.push(temp.value(row));
        }
    }
    assert_eq!(cpu_vals, vec![100, 250]);
    assert_eq!(temp_vals, vec![50, 55]);
}

#[tokio::test]
async fn file_metadata_round_trips() {
    let snapshots = vec![
        FakeSnapshot::new(1).counter("c", 1, &[]),
        FakeSnapshot::new(2).counter("c", 2, &[]),
    ];
    let fixture = write_fixture(snapshots, ParquetHistogramType::Standard);
    let source = MetrikenMetricSource::try_new(fixture.path()).unwrap();
    // ParquetSchema::finalize doesn't write sampling_interval_ms / source /
    // version unless set on the snapshot or passed via the metadata arg, so
    // these are simply None here.  We're really just checking the accessors
    // don't panic and the source loaded.
    assert!(source.sampling_interval_ms().is_none());
    assert!(source.source().is_none());
    assert!(source.version().is_none());
}
