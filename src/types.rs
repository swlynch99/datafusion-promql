use std::collections::BTreeMap;

/// Label set for a time series. Sorted by key for deterministic comparison.
pub type Labels = BTreeMap<String, String>;

/// The special label name used for the metric name in Prometheus.
pub const METRIC_NAME_LABEL: &str = "__name__";

/// Default lookback window in milliseconds (5 minutes).
pub const DEFAULT_LOOKBACK_MS: i64 = 300_000;

/// A time range in milliseconds since epoch.
#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    pub start_ms: i64,
    pub end_ms: i64,
}

/// The result of a PromQL query.
#[derive(Debug)]
pub enum QueryResult {
    /// Instant query result: a vector of samples at a single timestamp.
    Vector(Vec<InstantSample>),
    /// Range query result: a matrix of sample ranges per series.
    Matrix(Vec<RangeSamples>),
    /// A scalar value with its timestamp.
    Scalar(f64, i64),
    /// A string value with its timestamp.
    String(String, i64),
}

/// A single sample from an instant vector.
#[derive(Debug, Clone)]
pub struct InstantSample {
    pub labels: Labels,
    pub timestamp_ms: i64,
    pub value: f64,
}

/// A series of samples from a range vector.
#[derive(Debug, Clone)]
pub struct RangeSamples {
    pub labels: Labels,
    pub samples: Vec<(i64, f64)>,
}
