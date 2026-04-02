use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::prelude::*;

use crate::datasource::{
    ColumnMapping, MatchOp, Matcher, MetricMeta, MetricSource, TableFormat,
};
use crate::error::{PromqlError, Result};
use crate::types::{Labels, TimeRange};

/// A [`MetricSource`] that reads wide-format Rezolus parquet files.
///
/// Column names encode metric name and labels using the Rezolus naming
/// convention. The engine's normalization layer converts these to long format
/// at query time.
pub struct ParquetMetricSource {
    table_provider: Arc<dyn TableProvider>,
    column_mapping: ColumnMapping,
    /// Cached metric metadata parsed from the parquet schema.
    metrics: Vec<MetricMeta>,
}

impl ParquetMetricSource {
    /// Create a new source from a parquet file at `path`.
    pub async fn try_new(path: impl AsRef<Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let ctx = SessionContext::new();
        ctx.register_parquet("__parquet_src", &path_str, ParquetReadOptions::default())
            .await
            .map_err(|e| PromqlError::DataSource(format!("failed to register parquet: {e}")))?;

        let table_provider = ctx
            .table_provider("__parquet_src")
            .await
            .map_err(|e| PromqlError::DataSource(format!("failed to get table provider: {e}")))?;

        let column_mapping = rezolus_column_mapping();

        // Build metric metadata from the schema.
        let metrics = build_metric_metadata(&table_provider, &column_mapping);

        Ok(Self {
            table_provider,
            column_mapping,
            metrics,
        })
    }
}

#[async_trait]
impl MetricSource for ParquetMetricSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        // Return the full table; the normalize layer will project only the
        // relevant columns and DataFusion will push that down to the parquet
        // reader.
        Ok((
            Arc::clone(&self.table_provider),
            TableFormat::Wide(self.column_mapping.clone()),
        ))
    }

    async fn list_metrics(&self, name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        let metrics = match name_matcher {
            None => self.metrics.clone(),
            Some(m) => self
                .metrics
                .iter()
                .filter(|meta| matcher_matches(&meta.name, m))
                .cloned()
                .collect(),
        };
        Ok(metrics)
    }
}

/// Build a [`ColumnMapping`] for the Rezolus parquet column naming convention.
fn rezolus_column_mapping() -> ColumnMapping {
    ColumnMapping {
        timestamp_column: "timestamp".to_string(),
        ignore_columns: vec!["duration".to_string()],
        parse_column: Arc::new(rezolus_parse_column),
    }
}

/// Parse a Rezolus-style column name into `(metric_name, labels)`.
///
/// Patterns (checked in order):
/// 1. Ends with `:buckets` → `None` (histogram bucket, skip)
/// 2. Contains `//` → cgroup metric:
///    `metric_name//cgroup_path/pid` → labels `{cgroup="/path", id="pid"}`
///    Edge case: `metric///1` → `{cgroup="/", id="1"}`
/// 3. Contains `/` → split at first `/`:
///    - If remainder contains `/`: `metric/subtype/id` → `{op="subtype", id="id"}`
///    - Otherwise: `metric/subtype` → `{op="subtype"}`
/// 4. No `/` → plain metric, no labels
fn rezolus_parse_column(col_name: &str) -> Option<(String, Labels)> {
    // Skip histogram bucket columns.
    if col_name.ends_with(":buckets") {
        return None;
    }

    // Cgroup pattern: contains "//"
    if let Some(pos) = col_name.find("//") {
        let metric_name = &col_name[..pos];
        let remainder = &col_name[pos + 2..]; // e.g. "system.slice/chrony.service/28" or "/1"

        let mut labels = Labels::new();
        if let Some(last_slash) = remainder.rfind('/') {
            let cgroup_path = &remainder[..last_slash]; // may be empty for "///1"
            let id = &remainder[last_slash + 1..];
            labels.insert("cgroup".to_string(), format!("/{cgroup_path}"));
            labels.insert("id".to_string(), id.to_string());
        } else {
            // No slash in remainder — treat entire remainder as cgroup path.
            labels.insert("cgroup".to_string(), format!("/{remainder}"));
        }
        return Some((metric_name.to_string(), labels));
    }

    // Slash-separated pattern.
    if let Some(first_slash) = col_name.find('/') {
        let metric_name = &col_name[..first_slash];
        let remainder = &col_name[first_slash + 1..];

        let mut labels = Labels::new();
        if let Some(second_slash) = remainder.find('/') {
            // Two-level: metric/op/id  (e.g. softirq/net_rx/0)
            let op = &remainder[..second_slash];
            let id = &remainder[second_slash + 1..];
            labels.insert("op".to_string(), op.to_string());
            labels.insert("id".to_string(), id.to_string());
        } else {
            // Single-level: metric/op  (e.g. blockio_bytes/read)
            labels.insert("op".to_string(), remainder.to_string());
        }
        return Some((metric_name.to_string(), labels));
    }

    // Plain metric name, no labels.
    Some((col_name.to_string(), Labels::new()))
}

/// Build deduplicated [`MetricMeta`] from the parquet schema.
fn build_metric_metadata(
    provider: &Arc<dyn TableProvider>,
    mapping: &ColumnMapping,
) -> Vec<MetricMeta> {
    let schema = provider.schema();
    let ignore: BTreeSet<&str> = mapping
        .ignore_columns
        .iter()
        .map(|s| s.as_str())
        .collect();

    // Collect (metric_name -> set of label names).
    let mut metric_labels: std::collections::BTreeMap<String, BTreeSet<String>> =
        std::collections::BTreeMap::new();

    for field in schema.fields() {
        let col_name = field.name().as_str();
        if col_name == mapping.timestamp_column || ignore.contains(col_name) {
            continue;
        }

        // Only include numeric columns (skip List<u64> histograms etc.).
        match field.data_type() {
            arrow::datatypes::DataType::UInt64
            | arrow::datatypes::DataType::Int64
            | arrow::datatypes::DataType::Float64 => {}
            _ => continue,
        }

        if let Some((metric_name, labels)) = (mapping.parse_column)(col_name) {
            let entry = metric_labels.entry(metric_name).or_default();
            for key in labels.keys() {
                entry.insert(key.clone());
            }
        }
    }

    metric_labels
        .into_iter()
        .map(|(name, label_names)| MetricMeta {
            name,
            label_names: label_names.into_iter().collect(),
            extra_columns: vec![],
        })
        .collect()
}

/// Check whether a metric name matches a single [`Matcher`].
fn matcher_matches(name: &str, matcher: &Matcher) -> bool {
    match matcher.op {
        MatchOp::Equal => name == matcher.value,
        MatchOp::NotEqual => name != matcher.value,
        MatchOp::RegexMatch | MatchOp::RegexNotMatch => {
            // For simplicity, only support exact match in list_metrics filtering.
            // Regex filtering could be added if needed.
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plain_metric() {
        let (name, labels) = rezolus_parse_column("cpu_cores").unwrap();
        assert_eq!(name, "cpu_cores");
        assert!(labels.is_empty());
    }

    #[test]
    fn test_parse_single_slash() {
        let (name, labels) = rezolus_parse_column("blockio_bytes/read").unwrap();
        assert_eq!(name, "blockio_bytes");
        assert_eq!(labels.get("op").unwrap(), "read");
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn test_parse_double_slash_cgroup() {
        let (name, labels) =
            rezolus_parse_column("cgroup_cpu_cycles//system.slice/chrony.service/28").unwrap();
        assert_eq!(name, "cgroup_cpu_cycles");
        assert_eq!(
            labels.get("cgroup").unwrap(),
            "/system.slice/chrony.service"
        );
        assert_eq!(labels.get("id").unwrap(), "28");
    }

    #[test]
    fn test_parse_root_cgroup() {
        // cgroup_cpu_cycles///1 means cgroup="/", id="1"
        let (name, labels) = rezolus_parse_column("cgroup_cpu_cycles///1").unwrap();
        assert_eq!(name, "cgroup_cpu_cycles");
        assert_eq!(labels.get("cgroup").unwrap(), "/");
        assert_eq!(labels.get("id").unwrap(), "1");
    }

    #[test]
    fn test_parse_two_level_slash() {
        let (name, labels) = rezolus_parse_column("softirq/net_rx/0").unwrap();
        assert_eq!(name, "softirq");
        assert_eq!(labels.get("op").unwrap(), "net_rx");
        assert_eq!(labels.get("id").unwrap(), "0");
    }

    #[test]
    fn test_parse_bucket_skipped() {
        assert!(rezolus_parse_column("blockio_latency/read:buckets").is_none());
        assert!(rezolus_parse_column("tcp_srtt:buckets").is_none());
    }
}
