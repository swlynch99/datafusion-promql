//! [`MetricSource`] for `metriken-exposition` parquet files.
//!
//! `metriken-exposition` writes one row per snapshot timestamp; metric values
//! occupy one column per metric (counters as `UInt64`, gauges as `Int64`).
//! Histograms appear in one of two shapes:
//!
//! - **Standard:** a single `<name>:buckets` column of `List<UInt64>` whose
//!   row is the dense per-bucket count vector, length equal to the histogram
//!   crate's full bucket count for the given Config.
//! - **Sparse:** two parallel `<name>:bucket_indices` and
//!   `<name>:bucket_counts` columns of `List<UInt64>`, encoding only the
//!   non-zero buckets at each snapshot.
//!
//! Each column carries field metadata identifying its `metric_type`
//! (`histogram` / `sparse_histogram` / `counter` / `gauge` / `timestamp` /
//! `duration`), plus `grouping_power`, `max_value_power`, `unit`, and any
//! user-defined labels for histograms. File-level metadata records
//! `sampling_interval_ms`, `source`, and `version`.
//!
//! [`MetrikenMetricSource`] reads such a file, fuses the two histogram
//! shapes into the canonical `Struct<indices, counts>` representation
//! (see `crate::histogram`), and pre-differences cumulative-since-start
//! counts into per-period deltas — matching the behavior of
//! `metriken-query`'s `stream_histogram_column`. Resets (any per-bucket
//! decrease) and the leading row produce explicit empty deltas so that
//! every parquet row keeps its slot on the shared timestamp axis.
//!
//! The output is exposed as a wide-format [`TableProvider`]
//! (`TableFormat::Wide`) backed by an in-memory [`MemTable`]. One row per
//! snapshot timestamp, one column per metric.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, ListArray, StructArray, UInt64Array};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::MemTable;
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::TableType;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::datasource::{ColumnMapping, MatchOp, Matcher, MetricMeta, MetricSource, TableFormat};
use crate::error::{PromqlError, Result};
use crate::histogram::{
    HISTOGRAM_COUNTS_FIELD, HISTOGRAM_INDICES_FIELD, HistogramConfig, histogram_data_type,
};
use crate::types::{Labels, TimeRange};

const METRIC_TYPE_KEY: &str = "metric_type";
const METRIC_NAME_KEY: &str = "metric";
const GROUPING_POWER_KEY: &str = "grouping_power";
const MAX_VALUE_POWER_KEY: &str = "max_value_power";

const METRIC_TYPE_HISTOGRAM: &str = "histogram";
const METRIC_TYPE_SPARSE_HISTOGRAM: &str = "sparse_histogram";
const METRIC_TYPE_COUNTER: &str = "counter";
const METRIC_TYPE_GAUGE: &str = "gauge";
const METRIC_TYPE_TIMESTAMP: &str = "timestamp";
const METRIC_TYPE_DURATION: &str = "duration";

const TIMESTAMP_COLUMN: &str = "timestamp";
const DURATION_COLUMN: &str = "duration";

/// Metadata keys that should never appear as user-facing labels.
const RESERVED_KEYS: &[&str] = &[
    METRIC_NAME_KEY,
    METRIC_TYPE_KEY,
    "unit",
    GROUPING_POWER_KEY,
    MAX_VALUE_POWER_KEY,
];

/// One row of a sparse cumulative histogram: `(indices, counts)`.
type SparseRow = (Vec<u64>, Vec<u64>);

/// Per-metric rolling cumulative state for the differencing pass.
type PrevState = Option<SparseRow>;

/// Build a [`ColumnMapping`] that resolves metriken-exposition wide columns.
pub fn metriken_column_mapping() -> ColumnMapping {
    ColumnMapping {
        timestamp_column: TIMESTAMP_COLUMN.to_string(),
        ignore_columns: vec![DURATION_COLUMN.to_string()],
        parse_column: Arc::new(parse_metriken_column),
    }
}

/// Parse a metriken column field's metric name and labels from its Arrow
/// field metadata.
///
/// `metric` provides the metric name (or, for legacy files lacking it, the
/// column name with any `:buckets`/`:bucket_indices`/`:bucket_counts` suffix
/// stripped). Reserved keys ([`RESERVED_KEYS`]) are excluded from labels;
/// every other key/value pair becomes a label.
pub fn parse_metriken_column(field: &Field) -> Option<(String, Labels)> {
    let meta = field.metadata();
    let name = meta
        .get(METRIC_NAME_KEY)
        .cloned()
        .unwrap_or_else(|| strip_histogram_suffix(field.name()).to_string());

    let mut labels = Labels::new();
    for (k, v) in meta {
        if RESERVED_KEYS.contains(&k.as_str()) {
            continue;
        }
        labels.insert(k.clone(), v.clone());
    }
    Some((name, labels))
}

fn strip_histogram_suffix(name: &str) -> &str {
    for suffix in [":buckets", ":bucket_indices", ":bucket_counts"] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            return stripped;
        }
    }
    name
}

/// A [`MetricSource`] over a metriken-exposition parquet file.
///
/// The file's contents are read, transformed (histogram fusion +
/// cumulative→delta differencing), and held in memory. Each call to
/// [`MetricSource::table_for_metric`] returns a [`MemTable`] over the
/// processed batches.
pub struct MetrikenMetricSource {
    schema: SchemaRef,
    batches: Arc<Vec<RecordBatch>>,
    column_mapping: ColumnMapping,
    metrics: Vec<MetricMeta>,
    /// `sampling_interval_ms` from file-level Parquet KV metadata.
    sampling_interval_ms: Option<u64>,
    /// `source` from file-level Parquet KV metadata.
    source: Option<String>,
    /// `version` from file-level Parquet KV metadata.
    version: Option<String>,
}

impl MetrikenMetricSource {
    /// Open and process a metriken-exposition parquet file.
    pub fn try_new(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| PromqlError::DataSource(format!("failed to open parquet file: {e}")))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| PromqlError::DataSource(format!("failed to read parquet: {e}")))?;
        let parquet_meta = builder.metadata().clone();
        let arrow_schema = builder.schema().clone();

        // Pull file-level Parquet KV metadata before consuming the builder.
        let mut file_kv: HashMap<String, String> = HashMap::new();
        if let Some(kv) = parquet_meta.file_metadata().key_value_metadata() {
            for entry in kv {
                file_kv.insert(entry.key.clone(), entry.value.clone().unwrap_or_default());
            }
        }
        let sampling_interval_ms = file_kv
            .get("sampling_interval_ms")
            .and_then(|v| v.parse::<u64>().ok());
        let source = file_kv.get("source").cloned();
        let version = file_kv.get("version").cloned();

        let reader = builder
            .build()
            .map_err(|e| PromqlError::DataSource(format!("failed to build parquet reader: {e}")))?;
        let mut input_batches: Vec<RecordBatch> = Vec::new();
        for batch in reader {
            let batch = batch.map_err(|e| {
                PromqlError::DataSource(format!("failed to read record batch: {e}"))
            })?;
            input_batches.push(batch);
        }

        let layout = SchemaLayout::analyze(&arrow_schema)?;
        let (out_schema, out_batches) = layout.process(&input_batches)?;

        let column_mapping = metriken_column_mapping();
        let metrics = build_metric_metadata(&out_schema, &column_mapping);

        Ok(Self {
            schema: out_schema,
            batches: Arc::new(out_batches),
            column_mapping,
            metrics,
            sampling_interval_ms,
            source,
            version,
        })
    }

    /// File-level `sampling_interval_ms`, if present.
    pub fn sampling_interval_ms(&self) -> Option<u64> {
        self.sampling_interval_ms
    }

    /// File-level `source` string, if present.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// File-level `version` string, if present.
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    /// Schema of the processed (fused + delta) wide table.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Borrow the processed record batches.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }
}

#[async_trait]
impl MetricSource for MetrikenMetricSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        let provider: Arc<dyn TableProvider> = Arc::new(MetrikenWideProvider {
            schema: Arc::clone(&self.schema),
            batches: Arc::clone(&self.batches),
        });
        Ok((provider, TableFormat::Wide(self.column_mapping.clone())))
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

#[derive(Debug)]
struct MetrikenWideProvider {
    schema: SchemaRef,
    batches: Arc<Vec<RecordBatch>>,
}

#[async_trait]
impl TableProvider for MetrikenWideProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let mem = MemTable::try_new(Arc::clone(&self.schema), vec![(*self.batches).clone()])?;
        mem.scan(state, projection, filters, limit).await
    }
}

fn matcher_matches(name: &str, matcher: &Matcher) -> bool {
    match matcher.op {
        MatchOp::Equal => name == matcher.value,
        MatchOp::NotEqual => name != matcher.value,
        MatchOp::RegexMatch | MatchOp::RegexNotMatch => true,
    }
}

/// Per-metric kind dispatch precomputed from the parquet schema.
enum InputColumn {
    /// Skip this column entirely (timestamp / duration / unrecognized type).
    Skip,
    /// `UInt64` counter column → passed through.
    Counter {
        metric_name: String,
        metadata: HashMap<String, String>,
    },
    /// `Int64` gauge column → passed through.
    Gauge {
        metric_name: String,
        metadata: HashMap<String, String>,
    },
    /// `List<UInt64>` standard (dense) histogram column.
    StandardHistogram {
        metric_name: String,
        config: HistogramConfig,
        /// Field metadata to attach to the output struct field.
        metadata: HashMap<String, String>,
    },
    /// `List<UInt64>` sparse-histogram indices column.
    SparseIndices {
        metric_name: String,
        config: HistogramConfig,
        /// Field metadata to attach to the output struct field. Only the
        /// indices side carries it; the counts side stores `Skip` because
        /// it is consumed by the indices side.
        metadata: HashMap<String, String>,
    },
    /// `List<UInt64>` sparse-histogram counts column. Consumed in lockstep
    /// with the matching `SparseIndices` column.
    SparseCounts,
}

/// Plan-time analysis of a parquet schema: which input columns map to which
/// output (post-processing) columns, and where each output column is found.
struct SchemaLayout {
    input_columns: Vec<InputColumn>,
    /// Mapping `metric_name` → input-column-index of the matching counts
    /// column. Used by `StandardHistogram` columns? No — only for sparse
    /// pairs: `SparseIndices` looks up its `:bucket_counts` partner here.
    sparse_counts_by_metric: HashMap<String, usize>,
    /// Index of the timestamp column in the input schema.
    timestamp_idx: usize,
}

impl SchemaLayout {
    fn analyze(schema: &Schema) -> Result<Self> {
        // First pass: classify every column.
        let mut sparse_counts_by_metric: HashMap<String, usize> = HashMap::new();
        let mut sparse_indices_by_metric: HashMap<String, usize> = HashMap::new();
        let mut timestamp_idx: Option<usize> = None;

        let mut input_columns: Vec<InputColumn> = Vec::with_capacity(schema.fields().len());

        for (idx, field) in schema.fields().iter().enumerate() {
            let metadata = field.metadata().clone();
            let metric_type = metadata.get(METRIC_TYPE_KEY).map(String::as_str);
            let col_name = field.name();

            match metric_type {
                Some(METRIC_TYPE_TIMESTAMP) => {
                    if timestamp_idx.is_some() {
                        return Err(PromqlError::DataSource(
                            "multiple timestamp columns in schema".into(),
                        ));
                    }
                    timestamp_idx = Some(idx);
                    input_columns.push(InputColumn::Skip);
                }
                Some(METRIC_TYPE_DURATION) => input_columns.push(InputColumn::Skip),
                Some(METRIC_TYPE_COUNTER) => {
                    let metric_name = metadata
                        .get(METRIC_NAME_KEY)
                        .cloned()
                        .unwrap_or_else(|| col_name.clone());
                    input_columns.push(InputColumn::Counter {
                        metric_name,
                        metadata,
                    });
                }
                Some(METRIC_TYPE_GAUGE) => {
                    let metric_name = metadata
                        .get(METRIC_NAME_KEY)
                        .cloned()
                        .unwrap_or_else(|| col_name.clone());
                    input_columns.push(InputColumn::Gauge {
                        metric_name,
                        metadata,
                    });
                }
                Some(METRIC_TYPE_HISTOGRAM) => {
                    let metric_name = metadata.get(METRIC_NAME_KEY).cloned().unwrap_or_else(|| {
                        col_name
                            .strip_suffix(":buckets")
                            .unwrap_or(col_name)
                            .to_string()
                    });
                    let config = HistogramConfig::from_metadata(&metadata).ok_or_else(|| {
                        PromqlError::DataSource(format!(
                            "histogram column '{col_name}' missing grouping_power / max_value_power"
                        ))
                    })?;
                    input_columns.push(InputColumn::StandardHistogram {
                        metric_name,
                        config,
                        metadata,
                    });
                }
                Some(METRIC_TYPE_SPARSE_HISTOGRAM) => {
                    let metric_name = metadata
                        .get(METRIC_NAME_KEY)
                        .cloned()
                        .unwrap_or_else(|| strip_histogram_suffix(col_name).to_string());
                    let config = HistogramConfig::from_metadata(&metadata).ok_or_else(|| {
                        PromqlError::DataSource(format!(
                            "sparse histogram column '{col_name}' missing \
                             grouping_power / max_value_power"
                        ))
                    })?;
                    if col_name.ends_with(":bucket_indices") {
                        if sparse_indices_by_metric
                            .insert(metric_name.clone(), idx)
                            .is_some()
                        {
                            return Err(PromqlError::DataSource(format!(
                                "duplicate sparse histogram indices column for metric \
                                 '{metric_name}'"
                            )));
                        }
                        input_columns.push(InputColumn::SparseIndices {
                            metric_name,
                            config,
                            metadata,
                        });
                    } else if col_name.ends_with(":bucket_counts") {
                        if sparse_counts_by_metric
                            .insert(metric_name.clone(), idx)
                            .is_some()
                        {
                            return Err(PromqlError::DataSource(format!(
                                "duplicate sparse histogram counts column for metric \
                                 '{metric_name}'"
                            )));
                        }
                        input_columns.push(InputColumn::SparseCounts);
                    } else {
                        return Err(PromqlError::DataSource(format!(
                            "sparse histogram column '{col_name}' has unrecognized suffix"
                        )));
                    }
                }
                _ => input_columns.push(InputColumn::Skip),
            }
        }

        // Sanity: every sparse indices column must have a paired counts column.
        for (metric_name, _) in sparse_indices_by_metric.iter() {
            if !sparse_counts_by_metric.contains_key(metric_name) {
                return Err(PromqlError::DataSource(format!(
                    "sparse histogram '{metric_name}' is missing its counts column"
                )));
            }
        }
        for (metric_name, _) in sparse_counts_by_metric.iter() {
            if !sparse_indices_by_metric.contains_key(metric_name) {
                return Err(PromqlError::DataSource(format!(
                    "sparse histogram '{metric_name}' is missing its indices column"
                )));
            }
        }

        let timestamp_idx = timestamp_idx
            .or_else(|| {
                schema
                    .fields()
                    .iter()
                    .position(|f| f.name() == TIMESTAMP_COLUMN)
            })
            .ok_or_else(|| PromqlError::DataSource("no timestamp column in schema".into()))?;

        Ok(Self {
            input_columns,
            sparse_counts_by_metric,
            timestamp_idx,
        })
    }

    /// Build the output schema and apply per-batch processing.
    ///
    /// Histogram columns are fused (sparse pair → struct, standard dense →
    /// sparse struct) and then differenced into per-period deltas across the
    /// concatenated input.
    fn process(&self, input_batches: &[RecordBatch]) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        // Materialize per-row data column-by-column. Concatenated arrays make
        // the differencing pass straightforward; the source emits a single
        // output record batch per input record batch (preserving row counts
        // for downstream alignment).
        if input_batches.is_empty() {
            // Empty file: synthesize an empty output schema with just timestamp.
            let mut fields: Vec<FieldRef> = Vec::new();
            fields.push(Arc::new(Field::new(
                TIMESTAMP_COLUMN,
                DataType::UInt64,
                false,
            )));
            for col in self.iter_metric_columns() {
                fields.push(self.build_output_field(col)?);
            }
            let schema = Arc::new(Schema::new(fields));
            return Ok((schema, Vec::new()));
        }

        // Build output schema.
        let mut fields: Vec<FieldRef> = Vec::with_capacity(self.input_columns.len());
        fields.push(Arc::new(Field::new(
            TIMESTAMP_COLUMN,
            DataType::UInt64,
            false,
        )));
        for col in self.iter_metric_columns() {
            fields.push(self.build_output_field(col)?);
        }
        let schema = Arc::new(Schema::new(fields));

        // Process each batch, then run the differencing across batches by
        // threading `prev_*` state.
        let mut output_batches = Vec::with_capacity(input_batches.len());

        // Track previous cumulative sparse pair per histogram metric across batches.
        let mut prev_per_metric: BTreeMap<String, PrevState> = BTreeMap::new();

        for batch in input_batches {
            let mut output_columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

            // Timestamp (always present).
            let ts_arr = batch.column(self.timestamp_idx).clone();
            let ts_typed = ts_arr
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| PromqlError::DataSource("timestamp column must be UInt64".into()))?
                .clone();
            output_columns.push(Arc::new(ts_typed));

            // Metric columns, in stable schema order.
            for (idx, col) in self.input_columns.iter().enumerate() {
                match col {
                    InputColumn::Skip | InputColumn::SparseCounts => continue,
                    InputColumn::Counter { .. } => {
                        let arr = batch.column(idx).clone();
                        // Sanity: must be UInt64.
                        let _ = arr.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| {
                            PromqlError::DataSource("counter column not UInt64".into())
                        })?;
                        output_columns.push(arr);
                    }
                    InputColumn::Gauge { .. } => {
                        let arr = batch.column(idx).clone();
                        let _ = arr.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                            PromqlError::DataSource("gauge column not Int64".into())
                        })?;
                        output_columns.push(arr);
                    }
                    InputColumn::StandardHistogram {
                        metric_name,
                        config,
                        ..
                    } => {
                        let list = downcast_list(batch.column(idx))?;
                        let prev = prev_per_metric.entry(metric_name.clone()).or_insert(None);
                        let struct_arr = standard_to_delta_struct(&list, *config, prev)?;
                        output_columns.push(Arc::new(struct_arr));
                    }
                    InputColumn::SparseIndices {
                        metric_name,
                        config,
                        ..
                    } => {
                        let counts_idx = *self
                            .sparse_counts_by_metric
                            .get(metric_name)
                            .ok_or_else(|| {
                                PromqlError::DataSource(format!(
                                    "sparse histogram '{metric_name}' missing counts column"
                                ))
                            })?;
                        let indices_list = downcast_list(batch.column(idx))?;
                        let counts_list = downcast_list(batch.column(counts_idx))?;
                        let prev = prev_per_metric.entry(metric_name.clone()).or_insert(None);
                        let struct_arr =
                            sparse_to_delta_struct(&indices_list, &counts_list, *config, prev)?;
                        output_columns.push(Arc::new(struct_arr));
                    }
                }
            }

            output_batches.push(
                RecordBatch::try_new(Arc::clone(&schema), output_columns)
                    .map_err(|e| PromqlError::DataSource(format!("record batch build: {e}")))?,
            );
        }

        Ok((schema, output_batches))
    }

    fn iter_metric_columns(&self) -> impl Iterator<Item = &InputColumn> + '_ {
        self.input_columns
            .iter()
            .filter(|c| !matches!(c, InputColumn::Skip | InputColumn::SparseCounts))
    }

    fn build_output_field(&self, col: &InputColumn) -> Result<FieldRef> {
        match col {
            InputColumn::Skip | InputColumn::SparseCounts => {
                unreachable!("filtered out by iter_metric_columns")
            }
            InputColumn::Counter {
                metric_name,
                metadata,
            } => Ok(Arc::new(
                Field::new(metric_name, DataType::UInt64, true)
                    .with_metadata(scrub_metric_type(metadata.clone())),
            )),
            InputColumn::Gauge {
                metric_name,
                metadata,
            } => Ok(Arc::new(
                Field::new(metric_name, DataType::Int64, true)
                    .with_metadata(scrub_metric_type(metadata.clone())),
            )),
            InputColumn::StandardHistogram {
                metric_name,
                config,
                metadata,
            }
            | InputColumn::SparseIndices {
                metric_name,
                config,
                metadata,
            } => {
                let dt = histogram_data_type(config);
                let mut meta = scrub_metric_type(metadata.clone());
                // Make sure the histogram config is on the outer field.
                meta.insert(
                    GROUPING_POWER_KEY.to_string(),
                    config.grouping_power.to_string(),
                );
                meta.insert(
                    MAX_VALUE_POWER_KEY.to_string(),
                    config.max_value_power.to_string(),
                );
                Ok(Arc::new(
                    Field::new(metric_name, dt, true).with_metadata(meta),
                ))
            }
        }
    }
}

/// Drop `metric_type` from a metadata map; it identifies the on-disk
/// representation and shouldn't surface as a user label or persist through
/// the engine.
fn scrub_metric_type(mut meta: HashMap<String, String>) -> HashMap<String, String> {
    meta.remove(METRIC_TYPE_KEY);
    meta
}

fn downcast_list(arr: &ArrayRef) -> Result<ListArray> {
    arr.as_any()
        .downcast_ref::<ListArray>()
        .cloned()
        .ok_or_else(|| PromqlError::DataSource("expected a List array".into()))
}

/// Build a histogram struct array of per-period deltas from a dense
/// (`Standard`) cumulative parquet column.
///
/// Each row's dense bucket vector is filtered to its non-zero entries to
/// recover sparse `(indices, counts)`, then differenced against the rolling
/// `prev` cumulative state. Resets, decode failures, and the first observed
/// row produce an explicit empty delta.
fn standard_to_delta_struct(
    list: &ListArray,
    config: HistogramConfig,
    prev: &mut PrevState,
) -> Result<StructArray> {
    let inner = list
        .values()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| PromqlError::DataSource("histogram list inner must be UInt64".into()))?
        .clone();
    let offsets = list.value_offsets().to_vec();

    let mut delta_indices: Vec<Vec<u64>> = Vec::with_capacity(list.len());
    let mut delta_counts: Vec<Vec<u64>> = Vec::with_capacity(list.len());

    for row in 0..list.len() {
        let curr = if list.is_null(row) {
            None
        } else {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            // Compress dense → sparse (only non-zero buckets).
            let mut indices = Vec::new();
            let mut counts = Vec::new();
            for (i, k) in (start..end).enumerate() {
                let c = inner.value(k);
                if c == 0 {
                    continue;
                }
                indices.push(i as u64);
                counts.push(c);
            }
            Some((indices, counts))
        };

        let (out_idx, out_cnt) = compute_delta(prev, &curr);
        delta_indices.push(out_idx);
        delta_counts.push(out_cnt);

        if let Some(c) = curr {
            *prev = Some(c);
        }
    }

    build_histogram_struct(delta_indices, delta_counts, config)
}

/// Build a histogram struct array of per-period deltas from a sparse pair of
/// cumulative parquet columns (`<name>:bucket_indices`, `<name>:bucket_counts`).
///
/// Each row already carries `(indices, counts)` for the cumulative-since-start
/// histogram; we difference it row-to-row using the same reset semantics as
/// the standard form.
fn sparse_to_delta_struct(
    indices_list: &ListArray,
    counts_list: &ListArray,
    config: HistogramConfig,
    prev: &mut PrevState,
) -> Result<StructArray> {
    if indices_list.len() != counts_list.len() {
        return Err(PromqlError::DataSource(format!(
            "sparse histogram indices/counts row count mismatch: {} vs {}",
            indices_list.len(),
            counts_list.len()
        )));
    }
    let idx_inner = indices_list
        .values()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            PromqlError::DataSource("sparse histogram indices inner must be UInt64".into())
        })?
        .clone();
    let cnt_inner = counts_list
        .values()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            PromqlError::DataSource("sparse histogram counts inner must be UInt64".into())
        })?
        .clone();
    let idx_offsets = indices_list.value_offsets().to_vec();
    let cnt_offsets = counts_list.value_offsets().to_vec();

    let mut delta_indices: Vec<Vec<u64>> = Vec::with_capacity(indices_list.len());
    let mut delta_counts: Vec<Vec<u64>> = Vec::with_capacity(indices_list.len());

    for row in 0..indices_list.len() {
        let curr = if indices_list.is_null(row) || counts_list.is_null(row) {
            None
        } else {
            let i_start = idx_offsets[row] as usize;
            let i_end = idx_offsets[row + 1] as usize;
            let c_start = cnt_offsets[row] as usize;
            let c_end = cnt_offsets[row + 1] as usize;
            if i_end - i_start != c_end - c_start {
                return Err(PromqlError::DataSource(format!(
                    "sparse histogram inner length mismatch at row {row}: {} indices vs {} counts",
                    i_end - i_start,
                    c_end - c_start
                )));
            }
            let indices: Vec<u64> = (i_start..i_end).map(|k| idx_inner.value(k)).collect();
            let counts: Vec<u64> = (c_start..c_end).map(|k| cnt_inner.value(k)).collect();
            // Drop zeros to maintain the sparse invariant on rows from older
            // writers that may include them.
            let (indices, counts) = drop_zeros(indices, counts);
            // Indices must be sorted ascending; they are produced by
            // SparseHistogram::from_iter in metriken-exposition. Assert in
            // debug builds and tolerate (sort) otherwise.
            debug_assert!(indices.windows(2).all(|w| w[0] < w[1]));
            Some((indices, counts))
        };

        let (out_idx, out_cnt) = compute_delta(prev, &curr);
        delta_indices.push(out_idx);
        delta_counts.push(out_cnt);

        if let Some(c) = curr {
            *prev = Some(c);
        }
    }

    build_histogram_struct(delta_indices, delta_counts, config)
}

fn drop_zeros(indices: Vec<u64>, counts: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    let mut out_idx = Vec::with_capacity(indices.len());
    let mut out_cnt = Vec::with_capacity(counts.len());
    for (i, c) in indices.into_iter().zip(counts) {
        if c != 0 {
            out_idx.push(i);
            out_cnt.push(c);
        }
    }
    (out_idx, out_cnt)
}

/// Compute `(indices, counts)` of the per-period delta between
/// `prev` and `curr` cumulative-since-start sparse histograms.
///
/// On reset (any per-bucket decrease), null `curr`, or absent `prev`, returns
/// an empty `(vec![], vec![])`. Both inputs are expected to have indices
/// sorted ascending.
fn compute_delta(prev: &PrevState, curr: &PrevState) -> SparseRow {
    let (Some(prev), Some(curr)) = (prev.as_ref(), curr.as_ref()) else {
        return (Vec::new(), Vec::new());
    };
    let (pi, pc) = (&prev.0, &prev.1);
    let (ci, cc) = (&curr.0, &curr.1);

    let mut out_idx = Vec::new();
    let mut out_cnt = Vec::new();
    let mut p = 0usize;
    let mut c = 0usize;
    while p < pi.len() || c < ci.len() {
        let p_idx = pi.get(p).copied();
        let c_idx = ci.get(c).copied();
        match (p_idx, c_idx) {
            (Some(pv), Some(cv)) if pv == cv => {
                if cc[c] < pc[p] {
                    return (Vec::new(), Vec::new());
                }
                let d = cc[c] - pc[p];
                if d > 0 {
                    out_idx.push(cv);
                    out_cnt.push(d);
                }
                p += 1;
                c += 1;
            }
            (Some(pv), Some(cv)) if pv < cv => {
                // prev had bucket pv with non-zero count; curr lost it → reset.
                if pc[p] > 0 {
                    return (Vec::new(), Vec::new());
                }
                p += 1;
            }
            (Some(_), Some(cv)) => {
                if cc[c] > 0 {
                    out_idx.push(cv);
                    out_cnt.push(cc[c]);
                }
                c += 1;
            }
            (Some(_), None) => {
                if pc[p] > 0 {
                    return (Vec::new(), Vec::new());
                }
                p += 1;
            }
            (None, Some(cv)) => {
                if cc[c] > 0 {
                    out_idx.push(cv);
                    out_cnt.push(cc[c]);
                }
                c += 1;
            }
            (None, None) => break,
        }
    }
    (out_idx, out_cnt)
}

/// Assemble a `Struct<indices: List<UInt64>, counts: List<UInt64>>` from a
/// row-major collection of per-period delta `(indices, counts)` pairs.
///
/// The struct's data type is set to the canonical
/// [`histogram_data_type(&config)`](crate::histogram::histogram_data_type),
/// which embeds the `grouping_power` / `max_value_power` metadata on the
/// inner fields so the per-row arrays match the schema's expected type
/// exactly.
///
/// Both inner lists are non-null, even on empty rows: an empty delta is
/// represented as a non-null `[]` so downstream operators can distinguish it
/// from a true null row produced by a missing column.
fn build_histogram_struct(
    delta_indices: Vec<Vec<u64>>,
    delta_counts: Vec<Vec<u64>>,
    config: HistogramConfig,
) -> Result<StructArray> {
    if delta_indices.len() != delta_counts.len() {
        return Err(PromqlError::DataSource(
            "delta indices/counts row mismatch".into(),
        ));
    }
    let n = delta_indices.len();

    let mut idx_values: Vec<u64> = Vec::new();
    let mut cnt_values: Vec<u64> = Vec::new();
    let mut idx_offsets: Vec<i32> = Vec::with_capacity(n + 1);
    let mut cnt_offsets: Vec<i32> = Vec::with_capacity(n + 1);
    idx_offsets.push(0);
    cnt_offsets.push(0);

    for (idx_row, cnt_row) in delta_indices.into_iter().zip(delta_counts.into_iter()) {
        idx_values.extend(idx_row);
        cnt_values.extend(cnt_row);
        idx_offsets.push(i32::try_from(idx_values.len()).map_err(|_| {
            PromqlError::DataSource("histogram total values overflow i32 list offsets".into())
        })?);
        cnt_offsets.push(i32::try_from(cnt_values.len()).map_err(|_| {
            PromqlError::DataSource("histogram total values overflow i32 list offsets".into())
        })?);
    }

    // Pull the canonical inner field types out of `histogram_data_type` so the
    // produced arrays' `data_type()` matches the schema verbatim — including
    // the per-inner-field `grouping_power` / `max_value_power` metadata that
    // `RecordBatch::try_new` cross-checks during construction.
    let canonical_dt = histogram_data_type(&config);
    let DataType::Struct(canonical_fields) = canonical_dt else {
        unreachable!("histogram_data_type always returns Struct");
    };
    let indices_field = canonical_fields
        .iter()
        .find(|f| f.name() == HISTOGRAM_INDICES_FIELD)
        .cloned()
        .expect("indices field present");
    let counts_field = canonical_fields
        .iter()
        .find(|f| f.name() == HISTOGRAM_COUNTS_FIELD)
        .cloned()
        .expect("counts field present");

    let DataType::List(item_indices) = indices_field.data_type().clone() else {
        unreachable!("inner type is List<UInt64>");
    };
    let DataType::List(item_counts) = counts_field.data_type().clone() else {
        unreachable!("inner type is List<UInt64>");
    };

    let indices_list = ListArray::try_new(
        item_indices,
        OffsetBuffer::new(idx_offsets.into()),
        Arc::new(UInt64Array::from(idx_values)),
        None,
    )
    .map_err(arrow_to_data_source)?;
    let counts_list = ListArray::try_new(
        item_counts,
        OffsetBuffer::new(cnt_offsets.into()),
        Arc::new(UInt64Array::from(cnt_values)),
        None,
    )
    .map_err(arrow_to_data_source)?;

    let fields = Fields::from(vec![indices_field, counts_field]);
    let arrays: Vec<ArrayRef> = vec![Arc::new(indices_list), Arc::new(counts_list)];
    let struct_arr = StructArray::try_new(fields, arrays, None).map_err(arrow_to_data_source)?;

    Ok(struct_arr)
}

fn arrow_to_data_source(e: ArrowError) -> PromqlError {
    PromqlError::DataSource(format!("arrow error: {e}"))
}

/// Build deduplicated [`MetricMeta`] from the processed schema.
fn build_metric_metadata(schema: &Schema, mapping: &ColumnMapping) -> Vec<MetricMeta> {
    let ignore: BTreeSet<&str> = mapping.ignore_columns.iter().map(|s| s.as_str()).collect();
    let mut metric_labels: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for field in schema.fields() {
        let col_name = field.name().as_str();
        if col_name == mapping.timestamp_column || ignore.contains(col_name) {
            continue;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_metriken_column_metric_and_labels() {
        let mut meta = HashMap::new();
        meta.insert("metric".to_string(), "latency".to_string());
        meta.insert("metric_type".to_string(), "histogram".to_string());
        meta.insert("unit".to_string(), "nanoseconds".to_string());
        meta.insert("grouping_power".to_string(), "3".to_string());
        meta.insert("max_value_power".to_string(), "63".to_string());
        meta.insert("op".to_string(), "read".to_string());
        let f = Field::new("latency:buckets", DataType::Null, true).with_metadata(meta);
        let (name, labels) = parse_metriken_column(&f).unwrap();
        assert_eq!(name, "latency");
        assert_eq!(labels.len(), 1);
        assert_eq!(labels.get("op").unwrap(), "read");
    }

    #[test]
    fn delta_handles_empty_prev() {
        let out = compute_delta(&None, &Some((vec![1, 2], vec![5, 6])));
        assert_eq!(out, (Vec::<u64>::new(), Vec::<u64>::new()));
    }

    #[test]
    fn delta_basic_aligned() {
        let prev = Some((vec![0u64, 4, 17], vec![10, 20, 5]));
        let curr = Some((vec![0u64, 4, 17, 30], vec![10, 25, 5, 1]));
        let (idx, cnt) = compute_delta(&prev, &curr);
        assert_eq!(idx, vec![4, 30]);
        assert_eq!(cnt, vec![5, 1]);
    }

    #[test]
    fn delta_reset_yields_empty() {
        // bucket 4 dropped from 20 → 1 (reset).
        let prev = Some((vec![0u64, 4], vec![10, 20]));
        let curr = Some((vec![0u64, 4], vec![10, 1]));
        let (idx, cnt) = compute_delta(&prev, &curr);
        assert!(idx.is_empty());
        assert!(cnt.is_empty());
    }

    #[test]
    fn delta_lost_bucket_means_reset() {
        // prev had bucket 4 with count 5; curr has no bucket 4 → reset.
        let prev = Some((vec![0u64, 4], vec![10, 5]));
        let curr = Some((vec![0u64], vec![10]));
        let (idx, cnt) = compute_delta(&prev, &curr);
        assert!(idx.is_empty());
        assert!(cnt.is_empty());
    }

    #[test]
    fn drop_zeros_filters() {
        let (i, c) = drop_zeros(vec![0, 1, 2, 3], vec![5, 0, 7, 0]);
        assert_eq!(i, vec![0, 2]);
        assert_eq!(c, vec![5, 7]);
    }
}
