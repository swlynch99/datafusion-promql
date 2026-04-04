use std::sync::Arc;

use arrow::array::{Float64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::TimeZone;
use datafusion::catalog::TableProvider;
use datafusion::datasource::MemTable;

use datafusion_promql::PromqlEngine;
use datafusion_promql::datasource::{Matcher, MetricMeta, MetricSource, TableFormat};
use datafusion_promql::error::Result;
use datafusion_promql::types::{QueryResult, TimeRange};

struct InMemoryMetricSource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

#[async_trait]
impl MetricSource for InMemoryMetricSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        let table = MemTable::try_new(Arc::clone(&self.schema), vec![self.batches.clone()])
            .map_err(|e| datafusion_promql::error::PromqlError::DataSource(e.to_string()))?;
        Ok((Arc::new(table), TableFormat::Long))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![MetricMeta {
            name: "gauge".into(),
            label_names: vec!["instance".into()],
            extra_columns: vec![],
        }])
    }
}

/// Build a source with values useful for trig testing.
/// host1: 0.0, host2: 1.0, host3: -1.0
fn make_trig_source() -> InMemoryMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["gauge", "gauge", "gauge"])),
            Arc::new(UInt64Array::from(vec![
                1_000_000_000_u64,
                1_000_000_000_u64,
                1_000_000_000_u64,
            ])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, -1.0])),
            Arc::new(StringArray::from(vec!["host1", "host2", "host3"])),
        ],
    )
    .expect("failed to create test batch");

    InMemoryMetricSource {
        schema,
        batches: vec![batch],
    }
}

/// Helper: run an instant query, sort by instance, return (instance, value) pairs.
async fn run_instant(engine: &PromqlEngine, query: &str) -> Vec<(String, f64)> {
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let result = engine.instant_query(query, ts).await.unwrap();
    match result {
        QueryResult::Vector(mut samples) => {
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));
            samples
                .iter()
                .map(|s| (s.labels.get("instance").unwrap().clone(), s.value))
                .collect()
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

fn assert_close(actual: f64, expected: f64, func: &str, input: &str) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "{func}({input}): expected NaN, got {actual}"
        );
    } else if expected.is_infinite() {
        assert_eq!(
            actual, expected,
            "{func}({input}): expected {expected}, got {actual}"
        );
    } else {
        assert!(
            (actual - expected).abs() < 1e-10,
            "{func}({input}): expected {expected}, got {actual}"
        );
    }
}

#[tokio::test]
async fn test_sin() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "sin(gauge)").await;

    assert_close(results[0].1, 0.0_f64.sin(), "sin", "0");
    assert_close(results[1].1, 1.0_f64.sin(), "sin", "1");
    assert_close(results[2].1, (-1.0_f64).sin(), "sin", "-1");
}

#[tokio::test]
async fn test_cos() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "cos(gauge)").await;

    assert_close(results[0].1, 0.0_f64.cos(), "cos", "0");
    assert_close(results[1].1, 1.0_f64.cos(), "cos", "1");
    assert_close(results[2].1, (-1.0_f64).cos(), "cos", "-1");
}

#[tokio::test]
async fn test_tan() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "tan(gauge)").await;

    assert_close(results[0].1, 0.0_f64.tan(), "tan", "0");
    assert_close(results[1].1, 1.0_f64.tan(), "tan", "1");
    assert_close(results[2].1, (-1.0_f64).tan(), "tan", "-1");
}

#[tokio::test]
async fn test_asin() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "asin(gauge)").await;

    assert_close(results[0].1, 0.0_f64.asin(), "asin", "0");
    assert_close(results[1].1, 1.0_f64.asin(), "asin", "1");
    assert_close(results[2].1, (-1.0_f64).asin(), "asin", "-1");
}

#[tokio::test]
async fn test_acos() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "acos(gauge)").await;

    assert_close(results[0].1, 0.0_f64.acos(), "acos", "0");
    assert_close(results[1].1, 1.0_f64.acos(), "acos", "1");
    assert_close(results[2].1, (-1.0_f64).acos(), "acos", "-1");
}

#[tokio::test]
async fn test_atan() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "atan(gauge)").await;

    assert_close(results[0].1, 0.0_f64.atan(), "atan", "0");
    assert_close(results[1].1, 1.0_f64.atan(), "atan", "1");
    assert_close(results[2].1, (-1.0_f64).atan(), "atan", "-1");
}

#[tokio::test]
async fn test_sinh() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "sinh(gauge)").await;

    assert_close(results[0].1, 0.0_f64.sinh(), "sinh", "0");
    assert_close(results[1].1, 1.0_f64.sinh(), "sinh", "1");
    assert_close(results[2].1, (-1.0_f64).sinh(), "sinh", "-1");
}

#[tokio::test]
async fn test_cosh() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "cosh(gauge)").await;

    assert_close(results[0].1, 0.0_f64.cosh(), "cosh", "0");
    assert_close(results[1].1, 1.0_f64.cosh(), "cosh", "1");
    assert_close(results[2].1, (-1.0_f64).cosh(), "cosh", "-1");
}

#[tokio::test]
async fn test_tanh() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "tanh(gauge)").await;

    assert_close(results[0].1, 0.0_f64.tanh(), "tanh", "0");
    assert_close(results[1].1, 1.0_f64.tanh(), "tanh", "1");
    assert_close(results[2].1, (-1.0_f64).tanh(), "tanh", "-1");
}

#[tokio::test]
async fn test_asinh() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "asinh(gauge)").await;

    assert_close(results[0].1, 0.0_f64.asinh(), "asinh", "0");
    assert_close(results[1].1, 1.0_f64.asinh(), "asinh", "1");
    assert_close(results[2].1, (-1.0_f64).asinh(), "asinh", "-1");
}

#[tokio::test]
async fn test_acosh() {
    // acosh is only defined for values >= 1, so use a special source.
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["gauge", "gauge"])),
            Arc::new(UInt64Array::from(vec![
                1_000_000_000_u64,
                1_000_000_000_u64,
            ])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
        ],
    )
    .unwrap();

    let source = InMemoryMetricSource {
        schema,
        batches: vec![batch],
    };
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "acosh(gauge)").await;

    assert_close(results[0].1, 1.0_f64.acosh(), "acosh", "1");
    assert_close(results[1].1, 2.0_f64.acosh(), "acosh", "2");
}

#[tokio::test]
async fn test_atanh() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "atanh(gauge)").await;

    assert_close(results[0].1, 0.0_f64.atanh(), "atanh", "0");
    // atanh(1) = +inf, atanh(-1) = -inf
    assert_close(results[1].1, 1.0_f64.atanh(), "atanh", "1");
    assert_close(results[2].1, (-1.0_f64).atanh(), "atanh", "-1");
}

#[tokio::test]
async fn test_deg() {
    // deg converts radians to degrees
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let pi = std::f64::consts::PI;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["gauge", "gauge"])),
            Arc::new(UInt64Array::from(vec![
                1_000_000_000_u64,
                1_000_000_000_u64,
            ])),
            Arc::new(Float64Array::from(vec![pi, pi / 2.0])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
        ],
    )
    .unwrap();

    let source = InMemoryMetricSource {
        schema,
        batches: vec![batch],
    };
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "deg(gauge)").await;

    assert_close(results[0].1, 180.0, "deg", "pi");
    assert_close(results[1].1, 90.0, "deg", "pi/2");
}

#[tokio::test]
async fn test_rad() {
    // rad converts degrees to radians
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["gauge", "gauge"])),
            Arc::new(UInt64Array::from(vec![
                1_000_000_000_u64,
                1_000_000_000_u64,
            ])),
            Arc::new(Float64Array::from(vec![180.0, 90.0])),
            Arc::new(StringArray::from(vec!["host1", "host2"])),
        ],
    )
    .unwrap();

    let source = InMemoryMetricSource {
        schema,
        batches: vec![batch],
    };
    let engine = PromqlEngine::new(Arc::new(source));
    let results = run_instant(&engine, "rad(gauge)").await;

    assert_close(results[0].1, std::f64::consts::PI, "rad", "180");
    assert_close(results[1].1, std::f64::consts::FRAC_PI_2, "rad", "90");
}

/// Verify trig functions drop the __name__ label.
#[tokio::test]
async fn test_trig_drops_metric_name() {
    let source = make_trig_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();
    let result = engine.instant_query("sin(gauge)", ts).await.unwrap();

    match result {
        QueryResult::Vector(samples) => {
            for s in &samples {
                assert!(
                    !s.labels.contains_key("__name__"),
                    "sin() should drop __name__, but got labels: {:?}",
                    s.labels
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}
