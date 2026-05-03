//! Test helper: write a small metriken-exposition-shaped parquet file to a
//! tempfile, without depending on the `metriken-exposition` crate.
//!
//! The on-disk layout matches what `metriken-exposition::ParquetWriter` would
//! produce: column-per-metric in a wide table, with column-level field
//! metadata identifying `metric_type`, `metric`, `grouping_power`,
//! `max_value_power`, and any user labels.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use ::histogram::Histogram;
use arrow::array::{ArrayRef, Int64Builder, ListArray, ListBuilder, RecordBatch, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use tempfile::NamedTempFile;

/// On-disk histogram representation: matches `ParquetHistogramType` in
/// `metriken-exposition`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistogramType {
    /// Single `<name>:buckets` column of dense `List<UInt64>` rows.
    Standard,
    /// Two parallel `<name>:bucket_indices` and `<name>:bucket_counts`
    /// columns of `List<UInt64>` rows.
    Sparse,
}

#[derive(Clone, Debug)]
pub struct CounterRow {
    pub name: String,
    pub value: u64,
    pub labels: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
pub struct GaugeRow {
    pub name: String,
    pub value: i64,
    pub labels: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
pub struct HistogramRow {
    pub name: String,
    pub grouping_power: u8,
    pub max_value_power: u8,
    pub cumulative_buckets: Vec<u64>,
    pub labels: Vec<(String, String)>,
}

/// One synthetic metriken snapshot with explicit timestamp + duration.
#[derive(Default)]
pub struct FakeSnapshot {
    pub timestamp_ns: u64,
    pub duration_ns: u64,
    pub counters: Vec<CounterRow>,
    pub gauges: Vec<GaugeRow>,
    pub histograms: Vec<HistogramRow>,
}

impl FakeSnapshot {
    pub fn new(ts_secs: u64) -> Self {
        Self {
            timestamp_ns: ts_secs * 1_000_000_000,
            duration_ns: 1_000_000_000,
            counters: Vec::new(),
            gauges: Vec::new(),
            histograms: Vec::new(),
        }
    }

    pub fn counter(mut self, name: &str, value: u64, labels: &[(&str, &str)]) -> Self {
        self.counters.push(CounterRow {
            name: name.to_string(),
            value,
            labels: labels
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        });
        self
    }

    pub fn gauge(mut self, name: &str, value: i64, labels: &[(&str, &str)]) -> Self {
        self.gauges.push(GaugeRow {
            name: name.to_string(),
            value,
            labels: labels
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        });
        self
    }

    pub fn histogram(
        mut self,
        name: &str,
        grouping_power: u8,
        max_value_power: u8,
        cumulative_buckets: Vec<u64>,
        labels: &[(&str, &str)],
    ) -> Self {
        // Reject invalid configs eagerly so test-authoring mistakes show up
        // here instead of during parquet writing.
        Histogram::from_buckets(grouping_power, max_value_power, cumulative_buckets.clone())
            .expect("valid histogram for fixture");
        self.histograms.push(HistogramRow {
            name: name.to_string(),
            grouping_power,
            max_value_power,
            cumulative_buckets,
            labels: labels
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        });
        self
    }
}

/// Build the `Field` for a counter column.
fn counter_field(row: &CounterRow) -> Field {
    let mut meta = HashMap::new();
    meta.insert("metric".to_string(), row.name.clone());
    meta.insert("metric_type".to_string(), "counter".to_string());
    for (k, v) in &row.labels {
        meta.insert(k.clone(), v.clone());
    }
    Field::new(&row.name, DataType::UInt64, true).with_metadata(meta)
}

fn gauge_field(row: &GaugeRow) -> Field {
    let mut meta = HashMap::new();
    meta.insert("metric".to_string(), row.name.clone());
    meta.insert("metric_type".to_string(), "gauge".to_string());
    for (k, v) in &row.labels {
        meta.insert(k.clone(), v.clone());
    }
    Field::new(&row.name, DataType::Int64, true).with_metadata(meta)
}

/// Build the `Field`(s) for a histogram column, depending on the on-disk
/// shape.  Standard: one `<name>:buckets` field.  Sparse: two parallel
/// `<name>:bucket_indices` + `<name>:bucket_counts` fields.
fn histogram_fields(row: &HistogramRow, kind: HistogramType) -> Vec<Field> {
    let mut meta = HashMap::new();
    meta.insert("metric".to_string(), row.name.clone());
    meta.insert("grouping_power".to_string(), row.grouping_power.to_string());
    meta.insert(
        "max_value_power".to_string(),
        row.max_value_power.to_string(),
    );
    for (k, v) in &row.labels {
        meta.insert(k.clone(), v.clone());
    }

    let list_dt = DataType::new_list(DataType::UInt64, true);

    match kind {
        HistogramType::Standard => {
            let mut m = meta.clone();
            m.insert("metric_type".to_string(), "histogram".to_string());
            vec![Field::new(format!("{}:buckets", row.name), list_dt, true).with_metadata(m)]
        }
        HistogramType::Sparse => {
            let mut m = meta.clone();
            m.insert("metric_type".to_string(), "sparse_histogram".to_string());
            vec![
                Field::new(
                    format!("{}:bucket_indices", row.name),
                    list_dt.clone(),
                    true,
                )
                .with_metadata(m.clone()),
                Field::new(format!("{}:bucket_counts", row.name), list_dt, true).with_metadata(m),
            ]
        }
    }
}

/// Write `snapshots` to a tempfile parquet matching the metriken-exposition
/// layout, using `kind` for the histogram representation.
pub fn write_fixture(snapshots: Vec<FakeSnapshot>, kind: HistogramType) -> NamedTempFile {
    // Build the union schema by walking every snapshot.  Within each metric
    // kind we keep insertion order and dedupe by name (first wins for
    // labels / config), matching `ParquetSchema::push`.
    let mut counter_order: Vec<String> = Vec::new();
    let mut counter_def: HashMap<String, CounterRow> = HashMap::new();
    let mut gauge_order: Vec<String> = Vec::new();
    let mut gauge_def: HashMap<String, GaugeRow> = HashMap::new();
    let mut histogram_order: Vec<String> = Vec::new();
    let mut histogram_def: HashMap<String, HistogramRow> = HashMap::new();

    for snap in &snapshots {
        for c in &snap.counters {
            counter_def.entry(c.name.clone()).or_insert_with(|| {
                counter_order.push(c.name.clone());
                c.clone()
            });
        }
        for g in &snap.gauges {
            gauge_def.entry(g.name.clone()).or_insert_with(|| {
                gauge_order.push(g.name.clone());
                g.clone()
            });
        }
        for h in &snap.histograms {
            histogram_def.entry(h.name.clone()).or_insert_with(|| {
                histogram_order.push(h.name.clone());
                h.clone()
            });
        }
    }

    let mut fields: Vec<Field> = Vec::new();
    fields.push(
        Field::new("timestamp", DataType::UInt64, false).with_metadata({
            let mut m = HashMap::new();
            m.insert("metric_type".to_string(), "timestamp".to_string());
            m.insert("unit".to_string(), "nanoseconds".to_string());
            m
        }),
    );
    fields.push(
        Field::new("duration", DataType::UInt64, true).with_metadata({
            let mut m = HashMap::new();
            m.insert("metric_type".to_string(), "duration".to_string());
            m.insert("unit".to_string(), "nanoseconds".to_string());
            m
        }),
    );
    for name in &counter_order {
        fields.push(counter_field(&counter_def[name]));
    }
    for name in &gauge_order {
        fields.push(gauge_field(&gauge_def[name]));
    }
    for name in &histogram_order {
        for f in histogram_fields(&histogram_def[name], kind) {
            fields.push(f);
        }
    }

    let schema = Arc::new(Schema::new(fields));

    // Build per-snapshot row data column by column.
    let n = snapshots.len();
    let mut ts_arr = UInt64Builder::with_capacity(n);
    let mut dur_arr = UInt64Builder::with_capacity(n);
    let mut counter_builders: HashMap<&str, UInt64Builder> = counter_order
        .iter()
        .map(|n| (n.as_str(), UInt64Builder::with_capacity(snapshots.len())))
        .collect();
    let mut gauge_builders: HashMap<&str, Int64Builder> = gauge_order
        .iter()
        .map(|n| (n.as_str(), Int64Builder::with_capacity(snapshots.len())))
        .collect();
    // Two builders per histogram so Sparse can use both; Standard ignores the
    // second.  Both are keyed by metric name.
    let mut hist_first: HashMap<&str, ListBuilder<UInt64Builder>> = histogram_order
        .iter()
        .map(|n| (n.as_str(), ListBuilder::new(UInt64Builder::new())))
        .collect();
    let mut hist_second: HashMap<&str, ListBuilder<UInt64Builder>> = histogram_order
        .iter()
        .map(|n| (n.as_str(), ListBuilder::new(UInt64Builder::new())))
        .collect();

    for snap in &snapshots {
        ts_arr.append_value(snap.timestamp_ns);
        dur_arr.append_value(snap.duration_ns);

        let snap_counters: HashMap<&str, &CounterRow> =
            snap.counters.iter().map(|c| (c.name.as_str(), c)).collect();
        for name in &counter_order {
            let b = counter_builders.get_mut(name.as_str()).unwrap();
            match snap_counters.get(name.as_str()) {
                Some(c) => b.append_value(c.value),
                None => b.append_null(),
            }
        }

        let snap_gauges: HashMap<&str, &GaugeRow> =
            snap.gauges.iter().map(|g| (g.name.as_str(), g)).collect();
        for name in &gauge_order {
            let b = gauge_builders.get_mut(name.as_str()).unwrap();
            match snap_gauges.get(name.as_str()) {
                Some(g) => b.append_value(g.value),
                None => b.append_null(),
            }
        }

        let snap_hists: HashMap<&str, &HistogramRow> = snap
            .histograms
            .iter()
            .map(|h| (h.name.as_str(), h))
            .collect();
        for name in &histogram_order {
            let first = hist_first.get_mut(name.as_str()).unwrap();
            let second = hist_second.get_mut(name.as_str()).unwrap();
            match (kind, snap_hists.get(name.as_str())) {
                (HistogramType::Standard, Some(h)) => {
                    for v in &h.cumulative_buckets {
                        first.values().append_value(*v);
                    }
                    first.append(true);
                }
                (HistogramType::Standard, None) => {
                    first.append(false);
                }
                (HistogramType::Sparse, Some(h)) => {
                    // Convert dense → sparse using non-zero buckets.
                    for (i, v) in h.cumulative_buckets.iter().enumerate() {
                        if *v != 0 {
                            first.values().append_value(i as u64);
                            second.values().append_value(*v);
                        }
                    }
                    first.append(true);
                    second.append(true);
                }
                (HistogramType::Sparse, None) => {
                    first.append(false);
                    second.append(false);
                }
            }
        }
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    columns.push(Arc::new(ts_arr.finish()) as ArrayRef);
    columns.push(Arc::new(dur_arr.finish()) as ArrayRef);
    for name in &counter_order {
        let arr = counter_builders.remove(name.as_str()).unwrap().finish();
        columns.push(Arc::new(arr) as ArrayRef);
    }
    for name in &gauge_order {
        let arr = gauge_builders.remove(name.as_str()).unwrap().finish();
        columns.push(Arc::new(arr) as ArrayRef);
    }
    for name in &histogram_order {
        let mut first = hist_first.remove(name.as_str()).unwrap();
        let arr: ListArray = first.finish();
        columns.push(Arc::new(arr) as ArrayRef);
        if matches!(kind, HistogramType::Sparse) {
            let mut second = hist_second.remove(name.as_str()).unwrap();
            let arr: ListArray = second.finish();
            columns.push(Arc::new(arr) as ArrayRef);
        }
    }

    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("build record batch");

    let tmpfile = NamedTempFile::new().expect("create tempfile");
    let file = File::create(tmpfile.path()).expect("open tempfile for write");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter::try_new");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");

    tmpfile
}
