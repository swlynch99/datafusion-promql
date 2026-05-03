//! Integration tests for [`MetrikenMetricSource`].
//!
//! Each test writes a small synthetic metriken-exposition parquet file to a
//! tempfile, then loads it through the source and inspects the result.

mod fixture;
mod source;
