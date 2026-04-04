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
            name: "metric".into(),
            label_names: vec!["instance".into(), "job".into()],
            extra_columns: vec![],
        }])
    }
}

fn make_source() -> InMemoryMetricSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
        Field::new("job", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["metric", "metric"])),
            Arc::new(UInt64Array::from(vec![1_000_000_000, 1_000_000_000])),
            Arc::new(Float64Array::from(vec![1.0, 2.0])),
            Arc::new(StringArray::from(vec![
                "host1.example.com:9090",
                "host2.example.com:8080",
            ])),
            Arc::new(StringArray::from(vec!["prometheus", "grafana"])),
        ],
    )
    .expect("failed to create test batch");

    InMemoryMetricSource {
        schema,
        batches: vec![batch],
    }
}

// ---- label_replace tests ----

#[tokio::test]
async fn test_label_replace_capture_group() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Extract hostname before the first dot.
    let result = engine
        .instant_query(
            r#"label_replace(metric, "hostname", "$1", "instance", "([^.]+)\\..*")"#,
            ts,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert_eq!(samples[0].labels.get("hostname").unwrap(), "host1");
            assert_eq!(samples[1].labels.get("hostname").unwrap(), "host2");
            // Original labels preserved
            assert_eq!(
                samples[0].labels.get("instance").unwrap(),
                "host1.example.com:9090"
            );
            // Values unchanged
            assert!((samples[0].value - 1.0).abs() < f64::EPSILON);
            assert!((samples[1].value - 2.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_replace_no_match_preserves_dst() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Regex that won't match: looking for digits only in instance.
    // Since it doesn't match, dst_label "short" won't be created (stays empty).
    let result = engine
        .instant_query(
            r#"label_replace(metric, "short", "$1", "instance", "^(\\d+)$")"#,
            ts,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            // "short" label should not be set (no match, dst didn't exist, so stays empty)
            for s in &samples {
                assert!(
                    !s.labels.contains_key("short"),
                    "expected no 'short' label, got {:?}",
                    s.labels
                );
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_replace_static_replacement() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Match anything and replace with static string.
    let result = engine
        .instant_query(
            r#"label_replace(metric, "env", "production", "instance", ".*")"#,
            ts,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            for s in &samples {
                assert_eq!(s.labels.get("env").unwrap(), "production");
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_replace_overwrite_existing() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Overwrite the existing "job" label with a value derived from "instance".
    let result = engine
        .instant_query(
            r#"label_replace(metric, "job", "$1", "instance", "([^.]+).*")"#,
            ts,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));
            // job was overwritten
            assert_eq!(samples[0].labels.get("job").unwrap(), "host1");
            assert_eq!(samples[1].labels.get("job").unwrap(), "host2");
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_replace_preserves_name() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // label_replace should preserve __name__ unlike math functions.
    let result = engine
        .instant_query(
            r#"label_replace(metric, "hostname", "$1", "instance", "([^.]+).*")"#,
            ts,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            for s in &samples {
                assert_eq!(s.labels.get("__name__").unwrap(), "metric");
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

// ---- label_join tests ----

#[tokio::test]
async fn test_label_join_two_labels() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Join instance and job with "-"
    let result = engine
        .instant_query(
            r#"label_join(metric, "combined", "-", "instance", "job")"#,
            ts,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert_eq!(
                samples[0].labels.get("combined").unwrap(),
                "host1.example.com:9090-prometheus"
            );
            assert_eq!(
                samples[1].labels.get("combined").unwrap(),
                "host2.example.com:8080-grafana"
            );
            // Values unchanged
            assert!((samples[0].value - 1.0).abs() < f64::EPSILON);
            assert!((samples[1].value - 2.0).abs() < f64::EPSILON);
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_join_empty_separator() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Join with empty separator (concatenation)
    let result = engine
        .instant_query(r#"label_join(metric, "combined", "", "job", "job")"#, ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("job").cmp(&b.labels.get("job")));

            assert_eq!(samples[0].labels.get("combined").unwrap(), "grafanagrafana");
            assert_eq!(
                samples[1].labels.get("combined").unwrap(),
                "prometheusprometheus"
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_join_single_label() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Joining a single label just copies it.
    let result = engine
        .instant_query(r#"label_join(metric, "job_copy", "/", "job")"#, ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("job").cmp(&b.labels.get("job")));

            assert_eq!(samples[0].labels.get("job_copy").unwrap(), "grafana");
            assert_eq!(samples[1].labels.get("job_copy").unwrap(), "prometheus");
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_join_overwrite_existing() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Overwrite existing "job" label with joined value.
    let result = engine
        .instant_query(r#"label_join(metric, "job", "-", "job", "instance")"#, ts)
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("instance").cmp(&b.labels.get("instance")));

            assert_eq!(
                samples[0].labels.get("job").unwrap(),
                "prometheus-host1.example.com:9090"
            );
            assert_eq!(
                samples[1].labels.get("job").unwrap(),
                "grafana-host2.example.com:8080"
            );
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_join_preserves_name() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let result = engine
        .instant_query(
            r#"label_join(metric, "combined", "-", "instance", "job")"#,
            ts,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Vector(samples) => {
            assert_eq!(samples.len(), 2);
            for s in &samples {
                assert_eq!(s.labels.get("__name__").unwrap(), "metric");
            }
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}

#[tokio::test]
async fn test_label_join_missing_source_label() {
    let source = make_source();
    let engine = PromqlEngine::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // "nonexistent" label doesn't exist — treated as empty string.
    let result = engine
        .instant_query(
            r#"label_join(metric, "result", "-", "job", "nonexistent")"#,
            ts,
        )
        .await
        .unwrap();

    match result {
        QueryResult::Vector(mut samples) => {
            assert_eq!(samples.len(), 2);
            samples.sort_by(|a, b| a.labels.get("job").cmp(&b.labels.get("job")));

            // job + "-" + "" = "grafana-"
            assert_eq!(samples[0].labels.get("result").unwrap(), "grafana-");
            assert_eq!(samples[1].labels.get("result").unwrap(), "prometheus-");
        }
        other => panic!("expected Vector result, got {other:?}"),
    }
}
