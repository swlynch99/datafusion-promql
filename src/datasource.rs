use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use async_trait::async_trait;
use datafusion::catalog::TableProvider;

use crate::error::Result;
use crate::histogram::{HistogramConfig, histogram_config};
use crate::types::{Labels, TimeRange};

/// Kind of values stored in a metric's `value` column.
///
/// Lets a [`MetricSource`] declare whether the column carries scalar samples
/// (Prometheus-style `Float64`) or a native histogram (the canonical
/// `Struct<indices: List<UInt64>, counts: List<UInt64>>` shape with bucket
/// layout described by [`HistogramConfig`]). The downstream planner uses
/// this to route histogram columns into histogram-aware code paths instead
/// of Float64-only ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueKind {
    /// Standard PromQL scalar value column, typed as `Float64`.
    Scalar,
    /// Native histogram value column with the given bucket layout.
    Histogram(HistogramConfig),
}

impl ValueKind {
    /// Infer a `ValueKind` from a column's Arrow `Field`.
    ///
    /// Returns [`ValueKind::Histogram`] iff `field` carries the canonical
    /// histogram column shape and metadata (see
    /// [`crate::histogram::is_histogram_column`]); otherwise
    /// [`ValueKind::Scalar`].
    pub fn from_field(field: &Field) -> Self {
        match histogram_config(field) {
            Some(config) => Self::Histogram(config),
            None => Self::Scalar,
        }
    }
}

/// Parser function that converts a column field into
/// `(metric_name, labels, value_kind)`. Returns `None` if the column should
/// be skipped.
pub type ColumnParser = Arc<dyn Fn(&Field) -> Option<(String, Labels, ValueKind)> + Send + Sync>;

/// Describes the format of the table returned by a [`MetricSource`].
#[derive(Debug, Clone)]
pub enum TableFormat {
    /// Canonical long format: one row per (timestamp, series).
    ///
    /// Required columns: `__name__` (Utf8), `timestamp` (Int64 nanoseconds),
    /// `value`, plus one Utf8 column per label. The `value` column is
    /// `Float64` when `value_kind` is [`ValueKind::Scalar`]; for
    /// [`ValueKind::Histogram`] it carries the canonical histogram struct
    /// shape produced by [`crate::histogram::histogram_data_type`].
    Long {
        /// Kind of the `value` column for the metric being scanned.
        value_kind: ValueKind,
    },

    /// Wide format: one row per timestamp, one column per metric series.
    ///
    /// The engine will normalize this into long format using the provided
    /// [`ColumnMapping`]. Each value column declares its own [`ValueKind`]
    /// via the [`ColumnMapping::parse_column`] callback, so a single source
    /// may mix scalar and histogram metrics.
    Wide(ColumnMapping),
}

/// Describes how to parse wide-format column fields into metric name + labels.
#[derive(Clone)]
pub struct ColumnMapping {
    /// Column name for the timestamp. Defaults to "timestamp".
    pub timestamp_column: String,
    /// Columns to ignore (not metrics). E.g. `["duration"]`.
    pub ignore_columns: Vec<String>,
    /// A function that parses a column field into
    /// `(metric_name, labels, value_kind)`. Returns `None` if the column
    /// should be skipped.
    pub parse_column: ColumnParser,
}

impl std::fmt::Debug for ColumnMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnMapping")
            .field("timestamp_column", &self.timestamp_column)
            .field("ignore_columns", &self.ignore_columns)
            .field("parse_column", &"<fn>")
            .finish()
    }
}

/// Metadata about a single metric exposed by the data source.
#[derive(Debug, Clone)]
pub struct MetricMeta {
    /// The metric name (PromQL `__name__`).
    pub name: String,
    /// Known label names for this metric (excluding `__name__`).
    pub label_names: Vec<String>,
    /// Additional data-source-specific columns.
    pub extra_columns: Vec<ExtraColumn>,
}

/// An extra column exposed by a data source beyond the standard
/// `(timestamp, value, labels)`.
#[derive(Debug, Clone)]
pub struct ExtraColumn {
    pub name: String,
    pub arrow_type: DataType,
}

/// A label matcher from a PromQL selector.
#[derive(Debug, Clone)]
pub struct Matcher {
    pub name: String,
    pub op: MatchOp,
    pub value: String,
}

/// Match operation for a label matcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchOp {
    Equal,
    NotEqual,
    RegexMatch,
    RegexNotMatch,
}

/// A swappable backend that provides metric data to the PromQL engine.
///
/// Implementations may return data in either long or wide format. If wide
/// format is returned, the engine normalizes it to long format before
/// applying PromQL semantics.
#[async_trait]
pub trait MetricSource: Send + Sync {
    /// Return a DataFusion [`TableProvider`] for the given metric query.
    ///
    /// The source should push down the time range and label matchers
    /// to the extent possible.
    async fn table_for_metric(
        &self,
        metric_name: &str,
        matchers: &[Matcher],
        time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)>;

    /// List available metrics (used for `{__name__=~"pattern"}` selectors).
    async fn list_metrics(&self, name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>>;
}
