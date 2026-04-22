use std::borrow::Borrow;
use std::hash::{Hash, Hasher};

use smallvec::SmallVec;

/// Label set for a time series. Sorted by key for deterministic comparison.
///
/// Stores up to 4 key-value pairs inline (no heap allocation); spills to heap
/// for larger label sets. Iteration order is always sorted by key, matching
/// the previous BTreeMap-based semantics.
#[derive(Clone, Default)]
pub struct Labels(SmallVec<[(String, String); 4]>);

impl Labels {
    pub fn new() -> Self {
        Labels(SmallVec::new())
    }

    /// Insert a key-value pair. Returns the previous value if the key existed.
    pub fn insert(&mut self, key: String, value: String) -> Option<String> {
        match self.0.binary_search_by(|(k, _)| k.as_str().cmp(&key)) {
            Ok(i) => Some(std::mem::replace(&mut self.0[i].1, value)),
            Err(i) => {
                self.0.insert(i, (key, value));
                None
            }
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&String>
    where
        String: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.0
            .binary_search_by(|(k, _)| k.borrow().cmp(key))
            .ok()
            .map(|i| &self.0[i].1)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        String: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.0
            .binary_search_by(|(k, _)| k.borrow().cmp(key))
            .is_ok()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.0.iter().map(|(k, v)| (k, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &String> {
        self.0.iter().map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl PartialEq for Labels {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Labels {}

impl Hash for Labels {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialOrd for Labels {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Labels {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

/// Owned iterator over a `Labels`.
pub struct LabelsIntoIter(smallvec::IntoIter<[(String, String); 4]>);

impl Iterator for LabelsIntoIter {
    type Item = (String, String);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl IntoIterator for Labels {
    type Item = (String, String);
    type IntoIter = LabelsIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        LabelsIntoIter(self.0.into_iter())
    }
}

/// Borrowed iterator over a `Labels`.
pub struct LabelsIter<'a>(std::slice::Iter<'a, (String, String)>);

impl<'a> Iterator for LabelsIter<'a> {
    type Item = (&'a String, &'a String);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (k, v))
    }
}

impl<'a> IntoIterator for &'a Labels {
    type Item = (&'a String, &'a String);
    type IntoIter = LabelsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        LabelsIter(self.0.iter())
    }
}

/// The special label name used for the metric name in Prometheus.
pub const METRIC_NAME_LABEL: &str = "__name__";

/// Default lookback window in nanoseconds (5 minutes).
pub const DEFAULT_LOOKBACK_NS: u64 = 300_000_000_000;

/// A time range in nanoseconds since epoch.
///
/// When a bound is `None`, no constraint is applied on that side.
#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    pub start_ns: Option<u64>,
    pub end_ns: Option<u64>,
}

impl TimeRange {
    /// A time range with no constraints (no timestamp filters will be added).
    pub fn unbounded() -> Self {
        Self {
            start_ns: None,
            end_ns: None,
        }
    }
}

/// The result of a PromQL query.
#[derive(Debug)]
pub enum QueryResult {
    /// Instant query result: a vector of samples at a single timestamp.
    Vector(Vec<InstantSample>),
    /// Range query result: a matrix of sample ranges per series.
    Matrix(Vec<RangeSamples>),
    /// A scalar value with its timestamp.
    Scalar(f64, u64),
    /// A string value with its timestamp.
    String(String, u64),
}

/// A single sample from an instant vector.
#[derive(Debug, Clone)]
pub struct InstantSample {
    pub labels: Labels,
    pub timestamp_ns: u64,
    pub value: f64,
}

/// A series of samples from a range vector.
#[derive(Debug, Clone)]
pub struct RangeSamples {
    pub labels: Labels,
    pub samples: Vec<(u64, f64)>,
}
