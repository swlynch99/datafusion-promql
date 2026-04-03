use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::prelude::*;

use crate::datasource::{ColumnMapping, MatchOp, Matcher, MetricMeta, MetricSource, TableFormat};
use crate::error::{PromqlError, Result};
use crate::types::{Labels, TimeRange};

/// A [`MetricSource`] that reads wide-format parquet files whose columns carry
/// Arrow field-level metadata encoding the metric name and labels.
///
/// Each column's `field.metadata()` is inspected:
/// - `"metric"` key → metric name (fallback: column name with `:buckets` stripped)
/// - Reserved keys (`metric`, `unit`, `grouping_power`, `max_value_power`) are excluded
/// - All remaining key/value pairs become label key/value pairs
///
/// The DataFusion execution layer converts the wide table to long format at
/// query time via a UNION ALL of per-column projections.
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

/// Build a [`ColumnMapping`] that reads metric names and labels from Arrow
/// field metadata, following the same convention as metriken-query.
pub fn rezolus_column_mapping() -> ColumnMapping {
    ColumnMapping {
        timestamp_column: "timestamp".to_string(),
        ignore_columns: vec!["duration".to_string()],
        parse_column: Arc::new(parse_column_from_metadata),
    }
}

/// Parse a column's metric name and labels from its Arrow field metadata.
///
/// Follows the same convention as metriken-query:
/// - The `"metric"` metadata key provides the metric name. If absent, the
///   column name is used (with a `:buckets` suffix stripped if present).
/// - Reserved metadata keys (`metric`, `unit`, `grouping_power`,
///   `max_value_power`) are excluded from labels.
/// - All remaining metadata key/value pairs become label key/value pairs.
pub fn parse_column_from_metadata(field: &Field) -> Option<(String, Labels)> {
    let meta = field.metadata();

    let name = if let Some(n) = meta.get("metric") {
        n.clone()
    } else {
        let col_name = field.name();
        col_name
            .strip_suffix(":buckets")
            .unwrap_or(col_name)
            .to_string()
    };

    const RESERVED: &[&str] = &["metric", "unit", "grouping_power", "max_value_power"];

    let mut labels = Labels::new();
    for (k, v) in meta {
        if !RESERVED.contains(&k.as_str()) {
            labels.insert(k.clone(), v.clone());
        }
    }

    Some((name, labels))
}

/// Build deduplicated [`MetricMeta`] from the parquet schema.
fn build_metric_metadata(
    provider: &Arc<dyn TableProvider>,
    mapping: &ColumnMapping,
) -> Vec<MetricMeta> {
    let schema = provider.schema();
    let ignore: BTreeSet<&str> = mapping.ignore_columns.iter().map(|s| s.as_str()).collect();

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
            DataType::UInt64 | DataType::Int64 | DataType::Float64 => {}
            _ => continue,
        }

        if let Some((metric_name, labels)) = (mapping.parse_column)(field.as_ref()) {
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
    use std::collections::HashMap;

    fn make_field(name: &str, meta: HashMap<String, String>) -> Field {
        Field::new(name, DataType::UInt64, false).with_metadata(meta)
    }

    #[test]
    fn test_metric_name_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("metric".to_string(), "cpu_usage".to_string());
        meta.insert("cpu".to_string(), "0".to_string());
        let field = make_field("cpu_usage/0", meta);
        let (name, labels) = parse_column_from_metadata(&field).unwrap();
        assert_eq!(name, "cpu_usage");
        assert_eq!(labels.get("cpu").unwrap(), "0");
        assert!(!labels.contains_key("metric"));
    }

    #[test]
    fn test_column_name_fallback_no_metadata() {
        let field = make_field("cpu_cores", HashMap::new());
        let (name, labels) = parse_column_from_metadata(&field).unwrap();
        assert_eq!(name, "cpu_cores");
        assert!(labels.is_empty());
    }

    #[test]
    fn test_buckets_suffix_stripped_in_fallback() {
        let field = make_field("tcp_srtt:buckets", HashMap::new());
        let (name, labels) = parse_column_from_metadata(&field).unwrap();
        assert_eq!(name, "tcp_srtt");
        assert!(labels.is_empty());
    }

    #[test]
    fn test_reserved_keys_excluded_from_labels() {
        let mut meta = HashMap::new();
        meta.insert("metric".to_string(), "latency".to_string());
        meta.insert("unit".to_string(), "nanoseconds".to_string());
        meta.insert("grouping_power".to_string(), "3".to_string());
        meta.insert("max_value_power".to_string(), "63".to_string());
        meta.insert("op".to_string(), "read".to_string());
        let field = make_field("latency:buckets", meta);
        let (name, labels) = parse_column_from_metadata(&field).unwrap();
        assert_eq!(name, "latency");
        assert_eq!(labels.get("op").unwrap(), "read");
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn test_multiple_labels_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("metric".to_string(), "softirq".to_string());
        meta.insert("op".to_string(), "net_rx".to_string());
        meta.insert("id".to_string(), "0".to_string());
        let field = make_field("softirq/net_rx/0", meta);
        let (name, labels) = parse_column_from_metadata(&field).unwrap();
        assert_eq!(name, "softirq");
        assert_eq!(labels.get("op").unwrap(), "net_rx");
        assert_eq!(labels.get("id").unwrap(), "0");
    }

    #[test]
    fn test_cgroup_labels_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("metric".to_string(), "cgroup_cpu_cycles".to_string());
        meta.insert("cgroup".to_string(), "/system.slice/chrony.service".to_string());
        meta.insert("id".to_string(), "28".to_string());
        let field = make_field("cgroup_cpu_cycles//system.slice/chrony.service/28", meta);
        let (name, labels) = parse_column_from_metadata(&field).unwrap();
        assert_eq!(name, "cgroup_cpu_cycles");
        assert_eq!(labels.get("cgroup").unwrap(), "/system.slice/chrony.service");
        assert_eq!(labels.get("id").unwrap(), "28");
    }
}
