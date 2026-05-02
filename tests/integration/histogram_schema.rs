//! Schema-type detection for histogram-typed value columns at plan time.
//!
//! These tests build small DataFusion `LogicalPlan`s against a synthetic
//! `MetricSource` whose `value` column carries the canonical histogram
//! struct shape (see `src/histogram/mod.rs`). They exercise two
//! invariants of Task 2.1:
//!
//! 1. A `Field`'s histogram metadata survives wrapping in standard
//!    DataFusion logical nodes (Filter, Projection on a column reference).
//! 2. Plan-time boundaries that hard-code a Float64 `value` column
//!    (scalar binary ops, instant functions, range functions, sort)
//!    surface a clean `PromqlError::NotImplemented` rather than producing
//!    a malformed plan.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, ListArray, StringArray, StructArray, UInt64Array};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::TimeZone;
use datafusion::catalog::TableProvider;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{LogicalPlanBuilder, col, lit};

use datafusion_promql::PromqlPlanner;
use datafusion_promql::datasource::{Matcher, MetricMeta, MetricSource, TableFormat, ValueKind};
use datafusion_promql::error::{PromqlError, Result};
use datafusion_promql::histogram::{
    HistogramConfig, histogram_data_type, is_histogram_column, schema_value_is_histogram,
};
use datafusion_promql::types::TimeRange;

/// Build a `RecordBatch` schema where `value` is a histogram-typed Struct
/// column with `(grouping_power, max_value_power) = (7, 32)`.
fn histogram_schema() -> (Arc<Schema>, HistogramConfig) {
    let config = HistogramConfig::new(7, 32);
    let value_field = Field::new("value", histogram_data_type(&config), false)
        .with_metadata(config.to_metadata());
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        value_field,
        Field::new("instance", DataType::Utf8, false),
    ]));
    (schema, config)
}

/// Build an empty histogram `StructArray` of `n` rows. Each row has zero
/// (indices, counts) entries — that's enough to satisfy the type checker
/// at plan time, and the planning code under test never inspects values.
fn empty_histogram_struct(n: usize, config: &HistogramConfig) -> StructArray {
    let item_field = Arc::new(Field::new("item", DataType::UInt64, false));
    let offsets = OffsetBuffer::new(vec![0i32; n + 1].into());
    let indices_inner: ArrayRef = Arc::new(UInt64Array::from(Vec::<u64>::new()));
    let counts_inner: ArrayRef = Arc::new(UInt64Array::from(Vec::<u64>::new()));
    let indices = ListArray::try_new(item_field.clone(), offsets.clone(), indices_inner, None)
        .expect("failed to build indices ListArray");
    let counts = ListArray::try_new(item_field, offsets, counts_inner, None)
        .expect("failed to build counts ListArray");

    // Match the inner fields (including metadata) declared by
    // `histogram_data_type`, so the resulting StructArray's data type is
    // exactly equal to the schema's value-column type.
    let DataType::Struct(struct_fields) = histogram_data_type(config) else {
        unreachable!("histogram_data_type always returns a Struct")
    };
    let arrays: Vec<ArrayRef> = vec![Arc::new(indices), Arc::new(counts)];
    StructArray::try_new(struct_fields, arrays, None).expect("failed to build histogram struct")
}

fn make_histogram_source() -> HistogramMetricSource {
    let (schema, config) = histogram_schema();
    let n = 2;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["latency", "latency"])),
            Arc::new(UInt64Array::from(vec![1_000_000_000_u64, 2_000_000_000])),
            Arc::new(empty_histogram_struct(n, &config)),
            Arc::new(StringArray::from(vec!["host1", "host1"])),
        ],
    )
    .expect("failed to create histogram batch");
    HistogramMetricSource {
        schema,
        batches: vec![batch],
    }
}

/// In-memory metric source whose `value` column is a histogram-typed Struct.
struct HistogramMetricSource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

#[async_trait]
impl MetricSource for HistogramMetricSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        let table = MemTable::try_new(Arc::clone(&self.schema), vec![self.batches.clone()])
            .map_err(|e| PromqlError::DataSource(e.to_string()))?;
        Ok((
            Arc::new(table),
            TableFormat::Long {
                value_kind: ValueKind::Scalar,
            },
        ))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![MetricMeta {
            name: "latency".into(),
            label_names: vec!["instance".into()],
            extra_columns: vec![],
        }])
    }
}

#[tokio::test]
async fn histogram_metadata_survives_filter_and_projection() {
    let (schema, config) = histogram_schema();

    // Sanity: the `value` field on the bare schema is a histogram column.
    let value_field = schema.field_with_name("value").unwrap();
    assert!(is_histogram_column(value_field));
    assert_eq!(HistogramConfig::from_field(value_field), Some(config));

    // Build a tiny scan + Filter + Projection over the histogram-typed
    // schema and verify the histogram column metadata flows through both
    // intervening nodes intact.
    let table = MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap();
    let table_source =
        datafusion::datasource::provider_as_source(Arc::new(table) as Arc<dyn TableProvider>);

    let plan = LogicalPlanBuilder::scan("latency", table_source, None)
        .unwrap()
        .filter(col("instance").eq(lit("host1")))
        .unwrap()
        .project(vec![
            col("__name__"),
            col("timestamp"),
            col("value"),
            col("instance"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let projected_value = plan
        .schema()
        .fields()
        .iter()
        .find(|f| f.name() == "value")
        .expect("projected schema is missing the value column");
    assert!(
        is_histogram_column(projected_value),
        "histogram column type was dropped through Filter+Projection"
    );
    assert_eq!(
        HistogramConfig::from_field(projected_value),
        Some(config),
        "histogram config metadata was dropped through Filter+Projection"
    );
    assert!(schema_value_is_histogram(plan.schema()));
}

#[tokio::test]
async fn scalar_binary_op_on_histogram_errors_cleanly() {
    let planner = PromqlPlanner::new(Arc::new(make_histogram_source()));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let err = planner
        .instant_logical_plan("latency + 5", ts)
        .await
        .expect_err("scalar binary op on histogram should error");
    match err {
        PromqlError::NotImplemented(msg) => {
            assert!(
                msg.contains("histogram"),
                "expected histogram NotImplemented, got: {msg}"
            );
        }
        other => panic!("expected NotImplemented, got: {other}"),
    }
}

#[tokio::test]
async fn unary_negation_on_histogram_errors_cleanly() {
    let planner = PromqlPlanner::new(Arc::new(make_histogram_source()));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let err = planner
        .instant_logical_plan("-latency", ts)
        .await
        .expect_err("unary negation on histogram should error");
    assert!(matches!(err, PromqlError::NotImplemented(_)), "got: {err}");
}

#[tokio::test]
async fn instant_function_on_histogram_errors_cleanly() {
    let planner = PromqlPlanner::new(Arc::new(make_histogram_source()));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let err = planner
        .instant_logical_plan("abs(latency)", ts)
        .await
        .expect_err("abs() on histogram should error");
    match err {
        PromqlError::NotImplemented(msg) => {
            assert!(msg.contains("abs"), "expected abs() in error, got: {msg}");
        }
        other => panic!("expected NotImplemented, got: {other}"),
    }
}

#[tokio::test]
async fn range_function_on_histogram_errors_cleanly() {
    let planner = PromqlPlanner::new(Arc::new(make_histogram_source()));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let err = planner
        .instant_logical_plan("rate(latency[5m])", ts)
        .await
        .expect_err("rate() on histogram should error");
    match err {
        PromqlError::NotImplemented(msg) => {
            assert!(msg.contains("rate"), "expected rate() in error, got: {msg}");
        }
        other => panic!("expected NotImplemented, got: {other}"),
    }
}

#[tokio::test]
async fn sort_on_histogram_errors_but_sort_by_label_is_allowed() {
    let planner = PromqlPlanner::new(Arc::new(make_histogram_source()));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    let err = planner
        .instant_logical_plan("sort(latency)", ts)
        .await
        .expect_err("sort() on histogram should error");
    assert!(matches!(err, PromqlError::NotImplemented(_)), "got: {err}");

    // sort_by_label only orders by label columns, so it remains valid against
    // a histogram-typed value column.
    let plan = planner
        .instant_logical_plan("sort_by_label(latency, \"instance\")", ts)
        .await
        .expect("sort_by_label should plan against a histogram source");
    assert!(schema_value_is_histogram(plan.schema()));
}

// ---------------------------------------------------------------------------
// Mixed-kind source: a single MetricSource that exposes both a scalar metric
// and a histogram metric, and tags each with its own `ValueKind`.
//
// Task 1.3 requires that a `MetricSource` can declare per-metric value kinds
// and that the kind flows through normalization into the resulting
// LogicalPlan schema. These tests register one such source and assert the
// plan schema reflects the right value-column type for each metric.
// ---------------------------------------------------------------------------

/// In-memory source that returns a Float64 `value` column for the metric
/// `cpu_cores` and a histogram-typed `value` column for the metric
/// `latency`. Each `table_for_metric` call selects the appropriate provider
/// and tags the returned `TableFormat::Long` with the matching `ValueKind`.
struct MixedKindMetricSource {
    scalar_schema: Arc<Schema>,
    scalar_batches: Vec<RecordBatch>,
    histogram_schema: Arc<Schema>,
    histogram_batches: Vec<RecordBatch>,
    histogram_config: HistogramConfig,
}

impl MixedKindMetricSource {
    fn new() -> Self {
        let scalar_schema = Arc::new(Schema::new(vec![
            Field::new("__name__", DataType::Utf8, false),
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("instance", DataType::Utf8, false),
        ]));
        let scalar_batch = RecordBatch::try_new(
            Arc::clone(&scalar_schema),
            vec![
                Arc::new(StringArray::from(vec!["cpu_cores", "cpu_cores"])),
                Arc::new(UInt64Array::from(vec![1_000_000_000_u64, 2_000_000_000])),
                Arc::new(Float64Array::from(vec![4.0, 4.0])),
                Arc::new(StringArray::from(vec!["host1", "host1"])),
            ],
        )
        .expect("failed to build scalar batch");

        let (histogram_schema, histogram_config) = histogram_schema();
        let n = 2;
        let histogram_batch = RecordBatch::try_new(
            Arc::clone(&histogram_schema),
            vec![
                Arc::new(StringArray::from(vec!["latency", "latency"])),
                Arc::new(UInt64Array::from(vec![1_000_000_000_u64, 2_000_000_000])),
                Arc::new(empty_histogram_struct(n, &histogram_config)),
                Arc::new(StringArray::from(vec!["host1", "host1"])),
            ],
        )
        .expect("failed to build histogram batch");

        Self {
            scalar_schema,
            scalar_batches: vec![scalar_batch],
            histogram_schema,
            histogram_batches: vec![histogram_batch],
            histogram_config,
        }
    }
}

#[async_trait]
impl MetricSource for MixedKindMetricSource {
    async fn table_for_metric(
        &self,
        metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        match metric_name {
            "cpu_cores" => {
                let table = MemTable::try_new(
                    Arc::clone(&self.scalar_schema),
                    vec![self.scalar_batches.clone()],
                )
                .map_err(|e| PromqlError::DataSource(e.to_string()))?;
                Ok((
                    Arc::new(table),
                    TableFormat::Long {
                        value_kind: ValueKind::Scalar,
                    },
                ))
            }
            "latency" => {
                let table = MemTable::try_new(
                    Arc::clone(&self.histogram_schema),
                    vec![self.histogram_batches.clone()],
                )
                .map_err(|e| PromqlError::DataSource(e.to_string()))?;
                Ok((
                    Arc::new(table),
                    TableFormat::Long {
                        value_kind: ValueKind::Histogram(self.histogram_config),
                    },
                ))
            }
            other => Err(PromqlError::DataSource(format!("unknown metric: {other}"))),
        }
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![
            MetricMeta {
                name: "cpu_cores".into(),
                label_names: vec!["instance".into()],
                extra_columns: vec![],
            },
            MetricMeta {
                name: "latency".into(),
                label_names: vec!["instance".into()],
                extra_columns: vec![],
            },
        ])
    }
}

#[tokio::test]
async fn mixed_kind_source_plan_schemas_reflect_per_metric_value_kind() {
    let source = MixedKindMetricSource::new();
    let expected_config = source.histogram_config;
    let planner = PromqlPlanner::new(Arc::new(source));
    let ts = chrono::Utc.timestamp_millis_opt(1000).unwrap();

    // Scalar metric: the planner should preserve a Float64 `value` column.
    let scalar_plan = planner
        .instant_logical_plan("cpu_cores", ts)
        .await
        .expect("scalar metric should plan cleanly");
    let scalar_value = scalar_plan
        .schema()
        .fields()
        .iter()
        .find(|f| f.name() == "value")
        .expect("scalar plan is missing the value column");
    assert_eq!(
        scalar_value.data_type(),
        &DataType::Float64,
        "scalar metric value column should be Float64, got {:?}",
        scalar_value.data_type()
    );
    assert!(
        !is_histogram_column(scalar_value),
        "scalar metric value column was misclassified as histogram"
    );
    assert!(!schema_value_is_histogram(scalar_plan.schema()));

    // Histogram metric: the planner should preserve the canonical histogram
    // struct shape, including the bucket-layout metadata declared by the
    // source.
    let histogram_plan = planner
        .instant_logical_plan("latency", ts)
        .await
        .expect("histogram metric should plan cleanly");
    let histogram_value = histogram_plan
        .schema()
        .fields()
        .iter()
        .find(|f| f.name() == "value")
        .expect("histogram plan is missing the value column");
    assert!(
        is_histogram_column(histogram_value),
        "histogram metric value column was not recognized as histogram"
    );
    assert_eq!(
        HistogramConfig::from_field(histogram_value),
        Some(expected_config),
        "histogram metric lost its bucket-layout metadata through planning"
    );
    assert!(schema_value_is_histogram(histogram_plan.schema()));
}
