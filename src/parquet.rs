use std::any::Any;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::DFSchema;
use datafusion::datasource::listing::{ListingTableUrl, PartitionedFile};
use datafusion::datasource::physical_plan::{
    FileScanConfigBuilder, ParquetFileReaderFactory, ParquetSource,
};
use datafusion::datasource::source::DataSourceExec;
use datafusion::error::Result as DFResult;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::expr_rewriter::unnormalize_col;
use datafusion::logical_expr::utils::conjunction;
use datafusion::logical_expr::{BinaryExpr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::physical_expr::{create_ordering, create_physical_expr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::prelude::*;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use parquet::file::metadata::{FileMetaData, ParquetMetaData, ParquetMetaDataBuilder};
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;

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
    /// Cached `(min_ns, max_ns)` timestamp range read from the parquet file's
    /// row-group statistics during initialization.  `None` when the file has no
    /// timestamp column or carries no row-group statistics.
    timestamp_range_ns: Option<(u64, u64)>,
    /// Pre-computed mapping from metric name to the column indices in the full
    /// schema that belong to that metric.  Built once at init time by iterating
    /// the schema with `ColumnMapping::parse_column` so that `table_for_metric`
    /// can construct a narrow schema without re-walking all columns.
    metric_column_indices: HashMap<String, Vec<usize>>,
    /// Index of the timestamp column in the full schema.
    timestamp_col_idx: usize,
}

impl ParquetMetricSource {
    /// Create a new source from a parquet file at `path`.
    ///
    /// This reads the Arrow schema from the parquet file footer (skipping row
    /// group statistics to avoid unnecessary work) and then registers the table
    /// with DataFusion using the pre-built schema, bypassing DataFusion's own
    /// schema-inference pass.
    pub async fn try_new(path: impl AsRef<Path>) -> Result<Self> {
        let schema = read_schema(path.as_ref())?;
        Self::try_new_with_schema(path, schema).await
    }

    /// Create a new source from a parquet file at `path`, using a pre-built
    /// Arrow `schema` instead of inferring it from the file.
    ///
    /// [`try_new`](Self::try_new) calls this internally after extracting the
    /// schema with [`read_schema`] (which skips row-group statistics).  Use
    /// this directly when you already have the schema from a previous
    /// [`read_schema`] call and want to avoid reading the footer again.
    ///
    /// This reads the full [`ParquetMetaData`] footer once and stores it on
    /// the source so every subsequent scan can reuse the parsed metadata
    /// instead of re-parsing it from the file.  On very wide files
    /// (tens of thousands of columns), the Thrift footer decode dominates
    /// query latency, so hoisting it out of the scan path is a large win.
    pub async fn try_new_with_schema(path: impl AsRef<Path>, schema: Arc<Schema>) -> Result<Self> {
        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();

        // Parse the footer once up front.  The parquet reader factory below
        // serves this cached `Arc<ParquetMetaData>` to every scan, so the
        // ~1 s footer decode on 100k-column files is amortised.
        let metadata = read_parquet_metadata(path_ref)?;

        // Resolve the file into an object-store URL + Path so DataFusion can
        // locate it through the default local-filesystem object store.
        let table_url = ListingTableUrl::parse(&path_str)
            .map_err(|e| PromqlError::DataSource(format!("failed to parse table URL: {e}")))?;
        let object_store_url = table_url.object_store();
        let object_path = table_url.prefix().clone();

        let file_size = std::fs::metadata(path_ref)
            .map_err(|e| PromqlError::DataSource(format!("failed to stat parquet file: {e}")))?
            .len();

        let partitioned_file = PartitionedFile::new_from_meta(object_store::ObjectMeta {
            location: object_path,
            last_modified: chrono::Utc::now(),
            size: file_size,
            e_tag: None,
            version: None,
        });

        // Pre-build the base `ParquetSource` once up front.  `ParquetSource::new`
        // eagerly walks every column in the table schema to build an initial
        // `ProjectionExprs`, which on 100 k-column files costs ~300 ms.  We clone
        // this per scan instead (a cheap `Arc` clone) and let
        // `FileScanConfigBuilder::with_projection_indices` narrow the projection.
        let base_source = ParquetSource::new(Arc::clone(&schema));

        let column_mapping = rezolus_column_mapping();
        let table_provider: Arc<dyn TableProvider> = Arc::new(CachedParquetTableProvider {
            schema,
            object_store_url,
            file_size,
            metadata,
            base_source,
            partitioned_file,
            sort_order: vec![vec![
                col(&column_mapping.timestamp_column).sort(true, false),
            ]],
            timestamp_column: column_mapping.timestamp_column.clone(),
        });

        let metrics = build_metric_metadata(&table_provider, &column_mapping);
        let (metric_column_indices, timestamp_col_idx) =
            build_metric_column_indices(&table_provider, &column_mapping);

        // Cache the timestamp range from row-group statistics so callers can
        // use it as default query bounds without a separate read_timestamp_range
        // call.  Failures (no timestamp column, no statistics) are silently
        // treated as None rather than aborting initialization.
        let timestamp_range_ns = read_timestamp_range(path).ok();

        Ok(Self {
            table_provider,
            column_mapping,
            metrics,
            timestamp_range_ns,
            metric_column_indices,
            timestamp_col_idx,
        })
    }

    /// Return the `(min_ns, max_ns)` timestamp range cached from the parquet
    /// file's row-group statistics, or `None` if the file carries no timestamp
    /// statistics.
    ///
    /// This can be used as default query bounds so that DataFusion's parquet
    /// reader can prune row groups even when the caller does not provide an
    /// explicit time range.
    pub fn timestamp_range(&self) -> Option<(u64, u64)> {
        self.timestamp_range_ns
    }
}

#[async_trait]
impl MetricSource for ParquetMetricSource {
    async fn table_for_metric(
        &self,
        metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        // Build a narrow schema containing only the timestamp column plus the
        // columns that belong to this metric.  This avoids cloning the full
        // 100k-column schema at every planning stage (FilterExec, optimizer,
        // physical planner) for what is a handful of columns per metric.
        let full_schema = self.table_provider.schema();
        let metric_indices = self
            .metric_column_indices
            .get(metric_name)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        let mut narrow_fields: Vec<FieldRef> = Vec::with_capacity(1 + metric_indices.len());
        let mut index_map: Vec<usize> = Vec::with_capacity(1 + metric_indices.len());

        // Timestamp column is always first.
        narrow_fields.push(full_schema.fields()[self.timestamp_col_idx].clone());
        index_map.push(self.timestamp_col_idx);

        for &idx in metric_indices {
            narrow_fields.push(full_schema.fields()[idx].clone());
            index_map.push(idx);
        }

        let narrow_schema = Arc::new(Schema::new(narrow_fields));
        let narrow_provider: Arc<dyn TableProvider> = Arc::new(NarrowTableProvider {
            inner: Arc::clone(&self.table_provider),
            narrow_schema,
            index_map,
        });

        Ok((
            narrow_provider,
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
/// - If the field has no metadata at all, falls back to [`rezolus_parse_column`]
///   to parse metric name and labels from the column name using the
///   slash-based naming convention.
pub fn parse_column_from_metadata(field: &Field) -> Option<(String, Labels)> {
    let meta = field.metadata();

    // No metadata: fall back to name-based parsing.
    if meta.is_empty() {
        return rezolus_parse_column(field.name());
    }

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

/// Parse a metric name and labels from a Rezolus-style slash-encoded column
/// name.
///
/// Rezolus (and the metriken library) historically encoded metric names and
/// labels into the column name using `/` as a separator:
///
/// - `metric_name` → metric name only, no labels
/// - `metric_name/op` → `{op="op"}`
/// - `metric_name/op/id` → `{op="op", id="id"}`
/// - `metric_name//cgroup_path/id` → `{cgroup="/cgroup_path", id="id"}`
///   (double slash signals a cgroup path; the last component is the numeric
///   `id` and everything before it, with a leading `/`, is the `cgroup` label)
///
/// # Examples
///
/// ```text
/// "cpu_cores"                                      → ("cpu_cores", {})
/// "blockio_bytes/read"                             → ("blockio_bytes", {op="read"})
/// "softirq/net_rx/0"                               → ("softirq", {op="net_rx", id="0"})
/// "cgroup_cpu_cycles//system.slice/foo.service/1"  → ("cgroup_cpu_cycles", {cgroup="/system.slice/foo.service", id="1"})
/// "cgroup_cpu_cycles///1"                          → ("cgroup_cpu_cycles", {cgroup="/", id="1"})
/// ```
pub fn rezolus_parse_column(col_name: &str) -> Option<(String, Labels)> {
    // Strip histogram suffix before parsing.
    let col_name = col_name.strip_suffix(":buckets").unwrap_or(col_name);

    let mut labels = Labels::new();

    let Some(slash_pos) = col_name.find('/') else {
        // No slash: plain metric name, no labels.
        return Some((col_name.to_string(), labels));
    };

    let metric_name = col_name[..slash_pos].to_string();
    let rest = &col_name[slash_pos + 1..];

    if let Some(after_double_slash) = rest.strip_prefix('/') {
        // Double slash: cgroup format.
        // `after_double_slash` = "system.slice/chrony.service/28" or "/1"
        // The last component is the `id`; everything before it (prepended with
        // a `/`) is the `cgroup` label.
        if let Some(last_slash) = after_double_slash.rfind('/') {
            let cgroup_part = &after_double_slash[..last_slash];
            let id = &after_double_slash[last_slash + 1..];
            labels.insert("cgroup".to_string(), format!("/{cgroup_part}"));
            labels.insert("id".to_string(), id.to_string());
        }
    } else {
        // Regular format: `op` or `op/id`.
        let mut parts = rest.splitn(2, '/');
        let op = parts.next().unwrap_or("");
        labels.insert("op".to_string(), op.to_string());
        if let Some(id) = parts.next() {
            labels.insert("id".to_string(), id.to_string());
        }
    }

    Some((metric_name, labels))
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

/// Pre-compute a mapping from metric name to the column indices (in the full
/// schema) that parse to that metric under `mapping.parse_column`.
///
/// Returns `(metric_column_indices, timestamp_col_idx)`.  Called once during
/// [`ParquetMetricSource::try_new_with_schema`] so that `table_for_metric` can
/// build narrow schemas in O(matched columns) rather than O(all columns).
fn build_metric_column_indices(
    provider: &Arc<dyn TableProvider>,
    mapping: &ColumnMapping,
) -> (HashMap<String, Vec<usize>>, usize) {
    let schema = provider.schema();
    let ignore: BTreeSet<&str> = mapping.ignore_columns.iter().map(|s| s.as_str()).collect();

    let ts_idx = schema.index_of(&mapping.timestamp_column).unwrap_or(0);

    let mut metric_map: HashMap<String, Vec<usize>> = HashMap::new();

    for (idx, field) in schema.fields().iter().enumerate() {
        let col_name = field.name().as_str();
        if col_name == mapping.timestamp_column || ignore.contains(col_name) {
            continue;
        }

        match field.data_type() {
            DataType::UInt64 | DataType::Int64 | DataType::Float64 => {}
            _ => continue,
        }

        if let Some((metric_name, _)) = (mapping.parse_column)(field.as_ref()) {
            metric_map.entry(metric_name).or_default().push(idx);
        }
    }

    (metric_map, ts_idx)
}

/// Read the Arrow schema from a parquet file's footer metadata.
///
/// The returned schema can be passed to
/// [`ParquetMetricSource::try_new_with_schema`] to avoid re-reading the
/// footer during DataFusion registration.
pub fn read_schema(path: impl AsRef<Path>) -> Result<Arc<Schema>> {
    let file = File::open(path.as_ref())
        .map_err(|e| PromqlError::DataSource(format!("failed to open parquet file: {e}")))?;
    // Skip Arrow IPC field metadata to match DataFusion's default `skip_metadata = true`
    // behaviour.  The parquet column names carry all the information we need for
    // `rezolus_parse_column`; retaining the Arrow-level metadata can produce
    // unexpected label keys when the file uses different key names than our
    // parsing conventions expect.
    let options = ArrowReaderOptions::new().with_skip_arrow_metadata(true);
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options)
        .map_err(|e| PromqlError::DataSource(format!("failed to read parquet schema: {e}")))?;
    Ok(Arc::clone(builder.schema()))
}

/// Parse the full [`ParquetMetaData`] footer from a parquet file on disk.
///
/// Stored on [`ParquetMetricSource`] and handed to the custom
/// [`ParquetFileReaderFactory`] so every `TableProvider::scan` call reuses
/// the same parsed footer instead of re-reading and decoding it from the
/// file on each scan.
///
/// The `ARROW:schema` key in `FileMetaData::key_value_metadata` is stripped
/// before caching.  Without this, every scan's `ArrowReaderMetadata::try_new`
/// call would re-decode the flatbuffer-encoded Arrow schema for every column
/// in the file (via `arrow_ipc::convert::fb_to_schema`), which dominates
/// per-query latency on very wide files.  The parquet primitive schema by
/// itself is enough for DataFusion to reconstruct the Arrow schema.
fn read_parquet_metadata(path: &Path) -> Result<Arc<ParquetMetaData>> {
    let file = File::open(path).map_err(|e| {
        PromqlError::DataSource(format!("failed to open parquet file for metadata: {e}"))
    })?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| PromqlError::DataSource(format!("failed to read parquet metadata: {e}")))?;
    Ok(Arc::new(strip_arrow_ipc_schema(reader.metadata().clone())))
}

/// Rebuild a [`ParquetMetaData`] with the `ARROW:schema` key-value metadata
/// removed.  See [`read_parquet_metadata`] for why.
fn strip_arrow_ipc_schema(metadata: ParquetMetaData) -> ParquetMetaData {
    let file_meta = metadata.file_metadata();
    let stripped_kv = file_meta.key_value_metadata().and_then(|kvs| {
        let filtered: Vec<_> = kvs
            .iter()
            .filter(|kv| kv.key != "ARROW:schema")
            .cloned()
            .collect();
        if filtered.is_empty() {
            None
        } else {
            Some(filtered)
        }
    });
    let new_file_meta = FileMetaData::new(
        file_meta.version(),
        file_meta.num_rows(),
        file_meta.created_by().map(str::to_string),
        stripped_kv,
        file_meta.schema_descr_ptr(),
        file_meta.column_orders().cloned(),
    );
    ParquetMetaDataBuilder::new(new_file_meta)
        .set_row_groups(metadata.row_groups().to_vec())
        .set_column_index(metadata.column_index().cloned())
        .set_offset_index(metadata.offset_index().cloned())
        .build()
}

/// Read the min and max `timestamp` values from parquet row-group statistics.
///
/// Returns `(min_ns, max_ns)` as nanosecond timestamps. This reads only the
/// file footer metadata — no row data is decoded.
pub fn read_timestamp_range(path: impl AsRef<Path>) -> Result<(u64, u64)> {
    let file = File::open(path.as_ref())
        .map_err(|e| PromqlError::DataSource(format!("failed to open parquet file: {e}")))?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| PromqlError::DataSource(format!("failed to read parquet metadata: {e}")))?;
    let metadata = reader.metadata();
    let file_meta = metadata.file_metadata();
    let schema = file_meta.schema_descr();

    // Find the timestamp column index.
    let ts_idx = (0..schema.num_columns())
        .find(|&i| schema.column(i).name() == "timestamp")
        .ok_or_else(|| {
            PromqlError::DataSource("no 'timestamp' column found in parquet schema".into())
        })?;

    let mut global_min: Option<u64> = None;
    let mut global_max: Option<u64> = None;

    for rg_idx in 0..metadata.num_row_groups() {
        let rg = metadata.row_group(rg_idx);
        let col = rg.column(ts_idx);
        if let Some(stats) = col.statistics() {
            let (Some(min_bytes), Some(max_bytes)) = (stats.min_bytes_opt(), stats.max_bytes_opt())
            else {
                continue;
            };
            // The timestamp column is UInt64 in the arrow schema but stored as
            // INT64 in parquet (signed). Read as bytes and reinterpret.
            if min_bytes.len() == 8 && max_bytes.len() == 8 {
                let min_val = u64::from_le_bytes(min_bytes.try_into().unwrap());
                let max_val = u64::from_le_bytes(max_bytes.try_into().unwrap());
                global_min = Some(global_min.map_or(min_val, |v: u64| v.min(min_val)));
                global_max = Some(global_max.map_or(max_val, |v: u64| v.max(max_val)));
            }
        }
    }

    match (global_min, global_max) {
        (Some(min), Some(max)) => Ok((min, max)),
        _ => Err(PromqlError::DataSource(
            "no timestamp statistics found in parquet metadata".into(),
        )),
    }
}

/// A [`TableProvider`] wrapper that presents a narrow subset of the underlying
/// table's schema to DataFusion's planner.
///
/// On wide parquet files with tens-of-thousands of columns, DataFusion's plan
/// stages (logical optimizer, `FilterExec::compute_properties`,
/// `DefaultPhysicalPlanner`) clone the full `Arc<Schema>` many times.
/// `NarrowTableProvider` exposes only the timestamp column plus the columns for
/// one specific metric, so those clones are cheap.
///
/// `scan()` remaps projection indices from the narrow schema back to the
/// underlying full-schema indices, preserving parquet column-chunk pushdown.
#[derive(Debug)]
struct NarrowTableProvider {
    inner: Arc<dyn TableProvider>,
    narrow_schema: Arc<Schema>,
    /// `index_map[i]` is the column index in the inner provider's full schema
    /// that corresponds to column `i` in `narrow_schema`.
    index_map: Vec<usize>,
}

#[async_trait]
impl TableProvider for NarrowTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.narrow_schema)
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Remap narrow-schema projection indices to full-schema indices so the
        // underlying parquet reader knows which column chunks to read.
        let full_projection: Vec<usize> = match projection {
            Some(p) => p.iter().map(|&i| self.index_map[i]).collect(),
            None => self.index_map.clone(),
        };
        self.inner
            .scan(state, Some(&full_projection), filters, limit)
            .await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }
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

/// A [`TableProvider`] backed by a single parquet file whose `ParquetMetaData`
/// footer was parsed once at construction time.
///
/// Every call to [`Self::scan`] builds a fresh `FileScanConfig`, but:
///
/// 1. The footer Thrift bytes are never re-decoded — our custom
///    [`CachedMetadataReaderFactory`] serves the cached `Arc<ParquetMetaData>`
///    whenever the parquet reader asks for file metadata.
/// 2. The base [`ParquetSource`] is pre-built once and cloned per scan,
///    avoiding a per-scan `from_indices` walk over all ~100 k columns in
///    `ParquetSource::new`.
///
/// Combined, these cut ~900 ms off the per-query floor compared to a
/// plain `ListingTable` on very wide files.
#[derive(Debug)]
struct CachedParquetTableProvider {
    schema: SchemaRef,
    object_store_url: ObjectStoreUrl,
    file_size: u64,
    metadata: Arc<ParquetMetaData>,
    /// Pre-built [`ParquetSource`] with the full table schema.  Cloning it
    /// per scan is cheap (its `ProjectionExprs` is `Arc<[_]>`-backed), so we
    /// pay the O(columns) `from_indices` build cost exactly once.
    base_source: ParquetSource,
    /// Pre-built single-partition file descriptor for this parquet file.
    partitioned_file: PartitionedFile,
    /// Logical sort order declared for the file (one inner `Vec<Sort>` per
    /// equivalence class).  Preserved across scans so the DataFusion optimizer
    /// can still avoid emitting a `SortExec` for `timestamp`-ordered queries.
    sort_order: Vec<Vec<datafusion::logical_expr::SortExpr>>,
    /// Name of the timestamp column.  Used by
    /// [`Self::supports_filters_pushdown`] to recognise pure timestamp-range
    /// predicates so they can be claimed as `Exact` pushdown (the parquet
    /// reader evaluates them exactly via row-group pruning / page index / row
    /// filter), avoiding a redundant above-scan `FilterExec`.
    timestamp_column: String,
}

#[async_trait]
impl TableProvider for CachedParquetTableProvider {
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
        let store = state.runtime_env().object_store(&self.object_store_url)?;

        // Custom reader factory: every scan reuses the single parsed
        // `Arc<ParquetMetaData>` so the parquet opener never re-decodes the
        // footer Thrift bytes.
        let factory: Arc<dyn ParquetFileReaderFactory> = Arc::new(CachedMetadataReaderFactory {
            store,
            metadata: Arc::clone(&self.metadata),
            file_size: self.file_size,
        });

        let mut source = self
            .base_source
            .clone()
            .with_parquet_file_reader_factory(factory);

        // Install pushed-down filters as the parquet source's predicate.  With
        // `supports_filters_pushdown` claiming `Exact` for timestamp-range
        // predicates, DataFusion removes the above-scan `FilterExec` and hands
        // the filters here via `filters`; it's then our job to install them so
        // the parquet reader can perform row-group pruning + page index + row
        // filtering.  For `Inexact` filters we still install them for better
        // pruning — the above-scan `FilterExec` remains as the authoritative
        // evaluator.
        //
        // The optimizer sometimes hands us the same predicate in both qualified
        // (`cpu_cores.timestamp >= ...`) and unqualified (`timestamp >= ...`)
        // forms; strip qualifiers and dedup so the installed predicate has one
        // clause per distinct filter.
        let mut seen = std::collections::HashSet::new();
        let unique_filters: Vec<Expr> = filters
            .iter()
            .map(|f| unnormalize_col((*f).clone()))
            .filter(|f| seen.insert(f.clone()))
            .collect();
        if let Some(predicate_expr) = conjunction(unique_filters) {
            let df_schema = DFSchema::try_from(Arc::clone(&self.schema))?;
            let physical_expr =
                create_physical_expr(&predicate_expr, &df_schema, state.execution_props())?;
            source = source.with_predicate(physical_expr);
        }

        let output_ordering = create_ordering(&self.schema, &self.sort_order)?;

        let config = FileScanConfigBuilder::new(self.object_store_url.clone(), Arc::new(source))
            .with_file(self.partitioned_file.clone())
            .with_projection_indices(projection.cloned())?
            .with_limit(limit)
            .with_output_ordering(output_ordering)
            .build();

        Ok(DataSourceExec::from_data_source(config))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // Claim `Exact` pushdown for pure timestamp-range / equality predicates
        // against the monotonic, statistics-backed timestamp column.  The parquet
        // reader evaluates those predicates exactly (row-group pruning + page
        // index + row filter), so the optimizer can drop the redundant
        // above-scan `FilterExec`.  Everything else (label matchers, etc.)
        // stays `Inexact` because the parquet reader has no way to reason about
        // those semantics, so DataFusion must retain its `FilterExec`.
        Ok(filters
            .iter()
            .map(|f| {
                if is_timestamp_range_filter(f, &self.timestamp_column) {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Inexact
                }
            })
            .collect())
    }
}

/// Classify a filter expression as a pure `<timestamp column> CMP <literal>`
/// comparison (or its mirror image), where `CMP` is one of `<`, `<=`, `>`,
/// `>=`, `=`.  The parquet reader evaluates such filters exactly through its
/// pruning predicate + row filter, so they can be claimed as `Exact`
/// pushdown.  Anything more complex (arithmetic, casts, boolean combinations)
/// falls through to `Inexact`.
fn is_timestamp_range_filter(expr: &Expr, ts_col: &str) -> bool {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = expr else {
        return false;
    };
    if !matches!(
        op,
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq | Operator::Eq,
    ) {
        return false;
    }
    let is_ts_col = |e: &Expr| matches!(e, Expr::Column(c) if c.name == ts_col);
    let is_literal = |e: &Expr| matches!(e, Expr::Literal(_, _));
    (is_ts_col(left) && is_literal(right)) || (is_ts_col(right) && is_literal(left))
}

/// [`ParquetFileReaderFactory`] that hands out readers whose `get_metadata`
/// returns a pre-parsed `Arc<ParquetMetaData>` without touching the file.
///
/// All other `AsyncFileReader` calls (bytes / byte ranges, optional page
/// index load) are delegated to a standard [`ParquetObjectReader`].
#[derive(Debug)]
struct CachedMetadataReaderFactory {
    store: Arc<dyn ObjectStore>,
    metadata: Arc<ParquetMetaData>,
    file_size: u64,
}

impl ParquetFileReaderFactory for CachedMetadataReaderFactory {
    fn create_reader(
        &self,
        _partition_index: usize,
        partitioned_file: PartitionedFile,
        _metadata_size_hint: Option<usize>,
        _metrics: &ExecutionPlanMetricsSet,
    ) -> DFResult<Box<dyn AsyncFileReader + Send>> {
        let inner = ParquetObjectReader::new(
            Arc::clone(&self.store),
            partitioned_file.object_meta.location.clone(),
        )
        .with_file_size(self.file_size);
        Ok(Box::new(CachedMetadataReader {
            inner,
            metadata: Arc::clone(&self.metadata),
        }))
    }
}

struct CachedMetadataReader {
    inner: ParquetObjectReader,
    metadata: Arc<ParquetMetaData>,
}

impl AsyncFileReader for CachedMetadataReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let metadata = Arc::clone(&self.metadata);
        async move { Ok(metadata) }.boxed()
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
        meta.insert(
            "cgroup".to_string(),
            "/system.slice/chrony.service".to_string(),
        );
        meta.insert("id".to_string(), "28".to_string());
        let field = make_field("cgroup_cpu_cycles//system.slice/chrony.service/28", meta);
        let (name, labels) = parse_column_from_metadata(&field).unwrap();
        assert_eq!(name, "cgroup_cpu_cycles");
        assert_eq!(
            labels.get("cgroup").unwrap(),
            "/system.slice/chrony.service"
        );
        assert_eq!(labels.get("id").unwrap(), "28");
    }
}
