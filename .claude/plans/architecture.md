# datafusion-promql Architecture Plan

## Context

The goal is to build a PromQL query engine on top of Apache DataFusion. Users write PromQL queries (e.g., `rate(cpu_usage[5m])`), and the engine translates them into DataFusion logical/physical plans for execution against a pluggable data source.

The test data (`data/metrics.parquet`) is Rezolus-style: 30 rows, 950 columns in wide format. Columns encode metric name + labels in the column name (e.g., `cgroup_cpu_cycles//system.slice/chrony.service/28`), with `timestamp` and `duration` columns. Some columns are `List<u64>` for histogram buckets (e.g., `blockio_latency/read:buckets`). This wide format is very different from Prometheus's long format (`__name__`, labels, timestamp, value), so the data source abstraction must bridge this gap.

---

## 1. Dependencies

```toml
[dependencies]
promql-parser = "0.8"          # PromQL parsing -> AST
datafusion = "53"              # Query engine
arrow = { version = "55", features = ["prettyprint"] }
async-trait = "0.1"
thiserror = "2"
chrono = "0.4"

[features]
parquet = ["datafusion/parquet"]

[dev-dependencies]
tokio = { version = "1", features = ["full"] }
```

**Why these choices:**
- `promql-parser`: Mature, maintained by GreptimeDB team, tracks Prometheus v3.x grammar, provides a complete typed AST
- `datafusion`: Handles query optimization, execution, Arrow memory format, streaming. We get parallelism, predicate pushdown, etc. for free
- No intermediate IR crate needed — translate directly from promql-parser AST to DataFusion plans

---

## 2. Module Layout

```
src/
├── lib.rs                 # Public API: PromqlEngine, QueryResult types, re-exports
├── error.rs               # Error types (PromqlError enum)
├── datasource.rs          # MetricSource trait + TableFormat enum for swappable backends
├── normalize.rs           # Wide-to-long format conversion (used by engine when source returns wide data)
├── plan/
│   ├── mod.rs             # Plan translation entry point
│   ├── expr.rs            # PromQL Expr -> DataFusion LogicalPlan recursive translator
│   ├── selector.rs        # Vector/matrix selector -> scan + filter plans
│   └── series.rs          # Series alignment and step evaluation logic
├── node/
│   ├── mod.rs
│   ├── range_eval.rs      # UserDefinedLogicalNode: RangeVectorEval (windowed range fn application)
│   ├── instant_eval.rs    # UserDefinedLogicalNode: InstantVectorEval (step alignment)
│   └── series_merge.rs    # UserDefinedLogicalNode: binary op series matching (on/ignoring/group_left/group_right)
├── exec/
│   ├── mod.rs
│   ├── range_eval.rs      # ExecutionPlan for RangeVectorEval
│   ├── instant_eval.rs    # ExecutionPlan for InstantVectorEval
│   └── series_merge.rs    # ExecutionPlan for series merge
├── func/
│   ├── mod.rs             # Public FunctionRegistry, lookup by name, user-extensible
│   ├── range.rs           # Range vector functions: rate, irate, increase, delta, deriv, etc.
│   ├── instant.rs         # Instant vector functions: abs, ceil, floor, clamp, etc.
│   └── aggregate.rs       # Aggregation operators: sum, avg, count, topk, bottomk, quantile, etc.
├── parquet.rs             # (feature = "parquet") ParquetMetricSource for wide-format Rezolus parquet files
└── types.rs               # Shared types: TimeSeries, Sample, TimeRange, Step, etc.
```

All modules under `plan/`, `node/`, `exec/`, and `func/` are **public** so advanced users can:
- Register custom PromQL functions via `FunctionRegistry`
- Compose custom logical/physical plan nodes
- Build their own plan translation on top of the primitives

---

## 3. Data Source Abstraction (`datasource.rs`)

The core trait that backends implement:

```rust
/// Describes the format of the table returned by a MetricSource.
pub enum TableFormat {
    /// Canonical long format: one row per (timestamp, series).
    /// Required columns: `__name__` (Utf8), `timestamp` (TimestampNanosecond),
    /// `value` (Float64), plus one Utf8 column per label.
    Long,

    /// Wide format: one row per timestamp, one column per metric series.
    /// Required columns: `timestamp` (TimestampNanosecond or UInt64).
    /// Metric columns follow a naming convention (e.g. `metric_name/label_value`).
    /// The engine will normalize this into long format using the provided
    /// ColumnMapping.
    Wide(ColumnMapping),
}

/// Describes how to parse wide-format column names into metric name + labels.
pub struct ColumnMapping {
    /// Column name for the timestamp. Defaults to "timestamp".
    pub timestamp_column: String,
    /// Columns to ignore (not metrics). E.g. ["duration"].
    pub ignore_columns: Vec<String>,
    /// A function that parses a column name into (metric_name, labels).
    /// Returns None if the column should be skipped.
    pub parse_column: Arc<dyn Fn(&str) -> Option<(String, Labels)> + Send + Sync>,
}

/// Metadata about a single metric exposed by the data source.
pub struct MetricMeta {
    /// The metric name (PromQL `__name__`).
    pub name: String,
    /// Known label names for this metric (excluding `__name__`).
    pub label_names: Vec<String>,
    /// Additional data-source-specific columns beyond (timestamp, value, labels).
    /// These are exposed as extra label-like dimensions in PromQL.
    pub extra_columns: Vec<ExtraColumn>,
}

pub struct ExtraColumn {
    pub name: String,
    pub arrow_type: DataType,
}

#[async_trait]
pub trait MetricSource: Send + Sync {
    /// Return a DataFusion TableProvider for the given metric query.
    ///
    /// The table can be in either long or wide format, as indicated by
    /// the returned TableFormat. If wide, the engine will normalize it
    /// to long format before applying PromQL semantics.
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
    async fn list_metrics(
        &self,
        name_matcher: Option<&Matcher>,
    ) -> Result<Vec<MetricMeta>>;
}
```

**Key design decisions:**
- The source can return data in **either long or wide format**. If wide, the engine normalizes it to long format via `normalize.rs` before plan execution. This makes it trivial to implement sources for wide-format stores (like Rezolus parquet) without requiring them to do the pivot themselves.
- `ColumnMapping` is a flexible hook: the `parse_column` closure lets each source define its own column naming convention. For Rezolus data, `cgroup_cpu_cycles//system.slice/chrony.service/28` would parse to metric `cgroup_cpu_cycles` with labels `{cgroup="/system.slice/chrony.service", id="28"}`.
- Matchers and time range are passed to the source for pushdown. The source can ignore them and let DataFusion filter, but good sources will push them down.
- `extra_columns` lets a source expose additional dimensions that appear as labels in PromQL.
- For the test parquet data, we provide a built-in `ParquetMetricSource` (behind the `parquet` feature flag) that reads wide-format files and supplies the appropriate `ColumnMapping`.

---

## 4. Public API (`lib.rs`)

```rust
/// The main engine. Holds a DataFusion SessionContext and a MetricSource.
pub struct PromqlEngine {
    ctx: SessionContext,
    source: Arc<dyn MetricSource>,
}

impl PromqlEngine {
    pub fn new(source: Arc<dyn MetricSource>) -> Self;

    /// Execute an instant query at a single timestamp.
    pub async fn instant_query(
        &self,
        query: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<QueryResult>;

    /// Execute a range query over [start, end] with step.
    pub async fn range_query(
        &self,
        query: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        step: Duration,
    ) -> Result<QueryResult>;
}

/// Query result matching Prometheus result types.
pub enum QueryResult {
    Vector(Vec<InstantSample>),   // instant query result
    Matrix(Vec<RangeSamples>),    // range query result
    Scalar(f64),
    String(String),
}

pub struct InstantSample {
    pub labels: Labels,
    pub timestamp: i64,
    pub value: f64,
}

pub struct RangeSamples {
    pub labels: Labels,
    pub samples: Vec<(i64, f64)>,
}
```

Users construct the engine with their `MetricSource`, then call `instant_query` or `range_query`.

**Public re-exports:** `MetricSource`, `MetricMeta`, `ExtraColumn`, `TableFormat`, `ColumnMapping`, `QueryResult`, `Labels`, error types, plus the `plan`, `node`, `exec`, and `func` modules for extensibility.

**Extensibility points:**
- `func::FunctionRegistry` — users can register custom PromQL functions
- `node::*` — custom logical nodes are public, can be composed in user plans
- `exec::*` — physical operators are public for custom execution strategies
- `plan::plan_expr()` is public — users can translate PromQL AST to LogicalPlan and inspect/modify it before execution

---

## 5. Translation Pipeline

```
PromQL string
    │
    ▼
promql_parser::parse()  →  promql_parser::Expr (AST)
    │
    ▼
plan::expr::plan_expr()  →  DataFusion LogicalPlan
    │                        (using custom UserDefinedLogicalNodes
    │                         for PromQL-specific operations)
    ▼
DataFusion optimizer     →  Optimized LogicalPlan
    │                        (predicate pushdown, projection, etc.)
    ▼
Physical planner         →  ExecutionPlan DAG
    │                        (custom ExtensionPlanner maps our
    │                         logical nodes to physical nodes)
    ▼
execute().collect()      →  Vec<RecordBatch>
    │
    ▼
Collect into QueryResult
```

### What needs custom nodes vs. standard DataFusion:

**Standard DataFusion (no custom nodes):**
- `NumberLiteral`, `StringLiteral` → literal expressions
- `Paren` → just recurse
- `Unary` → negation expression
- Instant vector functions (abs, ceil, floor, etc.) → scalar UDFs
- Simple label-based filtering → DataFusion Filter node

**Custom UserDefinedLogicalNode:**
- **`InstantVectorEval`**: Aligns raw samples to evaluation timestamps (lookback window, staleness). This is the "step evaluation" that picks the most recent sample within the lookback window for each step timestamp.
- **`RangeVectorEval`**: For range vector functions (rate, irate, increase, delta, etc.). Collects samples within the range window `[t-range, t]` at each step, applies the range function.
- **`SeriesMerge`**: Binary operations between two instant vectors with label matching semantics (`on`, `ignoring`, `group_left`, `group_right`). Standard joins don't capture PromQL's matching rules.
- **`AggregateEval`**: PromQL aggregations with `by`/`without` semantics, plus special aggregators like `topk`, `bottomk`, `count_values`.

---

## 6. Range Vector and Step Evaluation

This is the trickiest part. PromQL evaluates at discrete time steps:

1. **Step generation**: For `range_query(start, end, step)`, generate timestamps: `[start, start+step, start+2*step, ..., end]`
2. **InstantVectorEval** node: For each step timestamp `t`, find the most recent sample where `t - lookback <= sample.timestamp <= t`. Default lookback = 5 minutes.
3. **RangeVectorEval** node: For each step timestamp `t` and range duration `d`, collect all samples where `t - d <= sample.timestamp <= t`, then apply the range function (e.g., rate computes `(last - first) / (last_t - first_t)`).

**Implementation approach**: These are custom `ExecutionPlan` nodes that:
- Receive a sorted stream of `(timestamp, value, labels...)` from their child
- Maintain a sliding window buffer
- Emit one row per (step_timestamp, series) with the computed value

---

## 7. Implementation Phases

### Phase 1: Scaffolding + basic instant vector selectors
- Set up module structure, dependencies, error types
- Implement `MetricSource` trait
- Build a simple in-memory test source
- Translate `VectorSelector` → table scan + filter
- `InstantVectorEval` node (step alignment)
- `instant_query` with a plain metric selector works end-to-end
- **Test**: `cpu_usage` returns values at a given timestamp

### Phase 2: Range vectors + core functions
- `MatrixSelector` translation
- `RangeVectorEval` node
- Implement `rate`, `irate`, `increase`, `delta`
- `range_query` works end-to-end
- **Test**: `rate(cpu_usage[5m])` returns computed rates

### Phase 3: Aggregations + binary ops
- `AggregateEval` node with `by`/`without`
- Implement `sum`, `avg`, `count`, `min`, `max`
- `SeriesMerge` node for binary operations
- Binary arithmetic and comparison operators
- **Test**: `sum(rate(cpu_usage[5m])) by (instance)`

### Phase 4: Full function coverage + Parquet source
- Remaining instant functions (abs, ceil, floor, clamp, etc.)
- Remaining aggregators (topk, bottomk, quantile, count_values, stddev, stdvar)
- Remaining range functions (avg_over_time, min_over_time, etc.)
- `ParquetMetricSource` for the test data (wide→long pivot)
- Subquery support
- **Test**: Full queries against `data/metrics.parquet`

### Phase 5: Optimization + edge cases
- Predicate pushdown through custom nodes
- Offset and `@` modifier support
- `bool` modifier on comparison operators
- Staleness handling (NaN propagation)
- `absent()`, `absent_over_time()`
- `histogram_quantile()`

---

## 8. Verification Plan

- **Unit tests**: Each module gets tests. Range functions tested with known input/output pairs.
- **Integration tests**: End-to-end queries against an in-memory source with known data, comparing results to expected PromQL output.
- **Parquet test**: Once ParquetMetricSource exists, run queries against `data/metrics.parquet` and verify reasonable results.
- **Conformance**: Compare results against Prometheus's own evaluation for a set of test queries (manual initially, automated later).

---

## Critical Files to Create/Modify

| File | Purpose |
|------|---------|
| `Cargo.toml` | Add dependencies |
| `src/lib.rs` | Public API, PromqlEngine |
| `src/error.rs` | Error types |
| `src/types.rs` | Shared types (TimeRange, Labels, Sample) |
| `src/datasource.rs` | MetricSource trait, TableFormat, ColumnMapping |
| `src/normalize.rs` | Wide-to-long format conversion |
| `src/plan/expr.rs` | AST → LogicalPlan translation (most complex file) |
| `src/plan/selector.rs` | Selector → scan planning |
| `src/node/range_eval.rs` | RangeVectorEval logical node |
| `src/node/instant_eval.rs` | InstantVectorEval logical node |
| `src/exec/range_eval.rs` | Physical execution for range vectors |
| `src/exec/instant_eval.rs` | Physical execution for instant vectors |
| `src/func/mod.rs` | Public FunctionRegistry |
| `src/func/range.rs` | rate, irate, increase, delta implementations |
| `src/func/aggregate.rs` | sum, avg, count, etc. |
| `src/parquet.rs` | (feature = "parquet") ParquetMetricSource |
