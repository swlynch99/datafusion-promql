use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use async_trait::async_trait;
use datafusion::catalog::TableProvider;

use crate::error::Result;
use crate::types::{Labels, TimeRange};

/// Parser function that converts a column field into `(metric_name, labels)`.
/// Returns `None` if the column should be skipped.
pub type ColumnParser = Arc<dyn Fn(&Field) -> Option<(String, Labels)> + Send + Sync>;

/// Describes the format of the table returned by a [`MetricSource`].
#[derive(Debug, Clone)]
pub enum TableFormat {
    /// Canonical long format: one row per (timestamp, series).
    ///
    /// Required columns: `__name__` (Utf8), `timestamp` (Int64 millis),
    /// `value` (Float64), plus one Utf8 column per label.
    Long,

    /// Wide format: one row per timestamp, one column per metric series.
    ///
    /// The engine will normalize this into long format using the provided
    /// [`ColumnMapping`].
    Wide(ColumnMapping),
}

/// Describes how to parse wide-format column fields into metric name + labels.
#[derive(Clone)]
pub struct ColumnMapping {
    /// Column name for the timestamp. Defaults to "timestamp".
    pub timestamp_column: String,
    /// Columns to ignore (not metrics). E.g. `["duration"]`.
    pub ignore_columns: Vec<String>,
    /// A function that parses a column field into `(metric_name, labels)`.
    /// Returns `None` if the column should be skipped.
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
