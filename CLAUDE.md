# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build
cargo build --features parquet

# Run all tests
cargo test
cargo test --features parquet

# Run a specific test file
cargo test --test instant_query
cargo test --test range_query -- --nocapture

# Run a single test by name
cargo test --test instant_query test_name -- --nocapture

# Lint
cargo clippy --all-targets --all-features

# Format (requires nightly)
cargo +nightly fmt --all

# Format check (as in CI)
cargo +nightly fmt --all --check

# Visualize a PromQL AST
cargo run --bin query-graph -- 'sum(rate(metric[5m])) by (instance)'

# Show DataFusion logical/optimized plans (requires parquet feature)
cargo run --bin query-plan --features parquet -- 'rate(cpu_usage[5m])'

# Execute a query and plot results in the terminal (requires plot feature)
# Range query:
cargo run --bin query-plot --features plot -- -f data/metrics.parquet --start 1750106216 --end 1750106506 --step 60 'rate(cpu_usage[60s])'
# Instant query:
cargo run --bin query-plot --features plot -- -f data/metrics.parquet --timestamp 1750106360 'cpu_cores'
```

## Architecture

This crate translates PromQL queries into Apache DataFusion execution plans. The flow is:

```
PromQL string
  → promql-parser AST
  → DataFusion LogicalPlan (with custom UserDefinedLogicalNodes)
  → DataFusion optimizer
  → Physical ExecutionPlan
  → RecordBatches → QueryResult
```

### Key abstraction: `MetricSource` (`src/datasource.rs`)

The `MetricSource` trait is the pluggable data backend. Implementations return a `TableProvider` plus a `TableFormat`:
- `TableFormat::Long`: standard Prometheus layout (`__name__`, `timestamp`, `value`, label columns)
- `TableFormat::Wide(ColumnMapping)`: one column per series (Rezolus-style parquet). The engine automatically normalizes wide→long via `src/normalize.rs` using UNION ALL projections.

### Custom logical/physical nodes

PromQL semantics that don't map to standard SQL require custom DataFusion extension nodes:

| Logical node (`src/node/`) | Physical node (`src/exec/`) | Purpose |
|---|---|---|
| `InstantVectorEval` | `InstantVectorExec` | Single-timestamp step alignment with lookback window |
| `StepVectorEval` | `StepVectorExec` | Multi-timestamp range query step alignment |
| `RangeVectorEval` | `RangeVectorExec` | Sliding window sample collection for range functions |
| `RangeFuncEval` | `RangeFuncExec` | Range function application (rate, delta, etc.) |
| `BinaryEval` | `BinaryExec` | Vector-vector binary ops with `on`/`ignoring`/`group_left`/`group_right` |
| `ScalarBinaryEval` | `ScalarBinaryExec` | Vector-scalar binary ops |
| `InstantFunction` | *(lowered by optimizer)* | Instant function wrapper, converted to Projection |
| `DateTimeFunction` | *(lowered by optimizer)* | DateTime function wrapper, converted to Projection |

### Custom optimizer rules (`src/opt/logical/`)

Seven custom DataFusion optimizer rules handle PromQL-specific plan transformations:

- `InstantFuncToProjection` — Converts `InstantFunction` nodes to standard Projection nodes
- `DateTimeFuncToProjection` — Converts `DateTimeFunction` nodes to Projection nodes
- `RangeVectorToAggregation` — Converts `RangeVectorEval` patterns to DataFusion aggregation
- `PushInstantEvalThroughUnion` — Pushes `InstantVectorEval` past Union nodes (important for wide→long)
- `LiftConstantProjections` — Lifts constant expressions out of projections, flattens nested projections
- `FoldRedundantAggregation` — Removes redundant aggregation layers
- `RemoveNoopProjections` — Cleans up identity projections

### Plan translation (`src/plan/`)

- `plan/expr.rs`: Main recursive translator from `promql_parser::Expr` to `LogicalPlan`. This is the most complex file.
- `plan/selector.rs`: Translates `VectorSelector`/`MatrixSelector` to table scans with time-range and label-matcher filters.

### Functions (`src/func/`)

- `func/range.rs`: `rate`, `irate`, `increase`, `delta`, `idelta`, `avg_over_time` — operate on a sliding window of `(timestamp, value)` pairs
- `func/instant.rs` + `func/udf/`: Math (abs, ceil, floor, round, sqrt, exp, ln, log2, log10, sgn), trig (sin, cos, tan, asin, acos, atan + hyperbolic variants), clamping (clamp, clamp_min, clamp_max), deg, rad — implemented as DataFusion scalar UDFs
- `func/aggregate.rs`: `sum`, `avg`, `count`, `min`, `max`, `stddev`, `stdvar`, `group`, `topk`, `bottomk`, `quantile`, `count_values`, `limitk`, `limit_ratio`
- `func/datetime.rs`: `time`, `timestamp`, `day_of_month`, `day_of_week`, `day_of_year`, `days_in_month`, `hour`, `minute`, `month`, `year`
- `func/label.rs`: `label_replace`, `label_join`
- `func/sort.rs`: `sort`, `sort_desc`, `sort_by_label`, `sort_by_label_desc`

### Data format detail

The Rezolus parquet test data (`data/metrics.parquet`) has ~950 columns in wide format. Column names like `cgroup_cpu_cycles//system.slice/chrony.service/28` encode metric name + labels. The `ColumnMapping.parse_column` closure in `ParquetMetricSource` (`src/parquet.rs`, behind `--features parquet`) handles this parsing.

### What's not yet implemented

See `.claude/plans/functions.md` for the full list. Notable gaps:
- Range functions: `deriv`, `predict_linear`, `sum_over_time`, `count_over_time`, `min_over_time`, `max_over_time`, `stddev_over_time`, `stdvar_over_time`, `quantile_over_time`, `last_over_time`, `present_over_time`, `changes`, `resets`, `absent_over_time`
- Instant functions: `scalar`, `vector`, `absent`, `pi`
- Histogram: `histogram_quantile`
- Modifiers: `@` (fixed timestamp)
- Subqueries

## Testing approach

Integration tests live in `tests/integration/` and use an `InMemoryMetricSource` (defined inline in each test file) that implements `MetricSource` with Arrow `RecordBatch` data. Tests instantiate `PromqlEngine`, execute queries, and assert on returned `QueryResult` values.

Parquet-dependent tests (`parquet_query.rs`, `rezolus_query.rs`) are feature-gated and require `--features parquet`.
