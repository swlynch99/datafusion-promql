//! Test helper: write a small metriken-exposition parquet file to a tempfile.

use std::collections::HashMap;
use std::fs::File;
use std::io::Seek;
use std::time::{Duration, SystemTime};

use ::histogram::Histogram;
use metriken_exposition::{
    Counter, Gauge, Histogram as SnapHistogram, ParquetHistogramType, ParquetOptions,
    ParquetSchema, Snapshot, SnapshotV2,
};
use tempfile::NamedTempFile;

/// One synthetic metriken snapshot with explicit timestamp + duration.
pub struct FakeSnapshot {
    pub timestamp: SystemTime,
    pub duration: Duration,
    pub counters: Vec<Counter>,
    pub gauges: Vec<Gauge>,
    pub histograms: Vec<SnapHistogram>,
}

impl FakeSnapshot {
    pub fn new(ts_secs: u64) -> Self {
        Self {
            timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(ts_secs),
            duration: Duration::from_secs(1),
            counters: Vec::new(),
            gauges: Vec::new(),
            histograms: Vec::new(),
        }
    }

    pub fn counter(mut self, name: &str, value: u64, labels: &[(&str, &str)]) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("metric".to_string(), name.to_string());
        for (k, v) in labels {
            metadata.insert((*k).to_string(), (*v).to_string());
        }
        self.counters.push(Counter {
            name: name.to_string(),
            value,
            metadata,
        });
        self
    }

    pub fn gauge(mut self, name: &str, value: i64, labels: &[(&str, &str)]) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("metric".to_string(), name.to_string());
        for (k, v) in labels {
            metadata.insert((*k).to_string(), (*v).to_string());
        }
        self.gauges.push(Gauge {
            name: name.to_string(),
            value,
            metadata,
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
        let mut metadata = HashMap::new();
        metadata.insert("metric".to_string(), name.to_string());
        metadata.insert("grouping_power".to_string(), grouping_power.to_string());
        metadata.insert("max_value_power".to_string(), max_value_power.to_string());
        for (k, v) in labels {
            metadata.insert((*k).to_string(), (*v).to_string());
        }
        let value =
            Histogram::from_buckets(grouping_power, max_value_power, cumulative_buckets).unwrap();
        self.histograms.push(SnapHistogram {
            name: name.to_string(),
            value,
            metadata,
        });
        self
    }

    fn into_snapshot(self) -> Snapshot {
        Snapshot::V2(SnapshotV2 {
            systemtime: self.timestamp,
            duration: self.duration,
            metadata: HashMap::new(),
            counters: self.counters,
            gauges: self.gauges,
            histograms: self.histograms,
        })
    }
}

/// Write the given snapshots to a tempfile parquet using
/// [`ParquetHistogramType::Standard`].
pub fn write_fixture(
    snapshots: Vec<FakeSnapshot>,
    histogram_type: ParquetHistogramType,
) -> NamedTempFile {
    let mut schema = ParquetSchema::new();
    let snapshots: Vec<Snapshot> = snapshots
        .into_iter()
        .map(FakeSnapshot::into_snapshot)
        .collect();
    for s in &snapshots {
        schema.push(s.clone());
    }

    let tmpfile = NamedTempFile::new().expect("create tempfile");
    let mut writer = schema
        .finalize(
            File::create(tmpfile.path()).expect("open tempfile for write"),
            ParquetOptions::new()
                .histogram_type(histogram_type)
                .max_batch_size(64),
            None,
        )
        .expect("finalize parquet schema");
    for s in snapshots {
        writer.push(s).expect("push snapshot");
    }
    let _meta = writer.finalize().expect("finalize parquet writer");

    // Reopen / rewind for test consumers that want a `File` handle. Using the
    // path is simpler and matches how MetrikenMetricSource opens files.
    let _ = File::open(tmpfile.path())
        .expect("reopen tempfile")
        .rewind();
    tmpfile
}
