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
| `InstantVectorEval` | `InstantVectorExec` | Step-timestamp alignment with lookback window |
| `RangeVectorEval` | `RangeVectorExec` | Sliding window for range functions (rate, delta, etc.) |
| `BinaryEval` | `BinaryExec` | Series matching for binary ops (`on`/`ignoring`/`group_left`/`group_right`) |
| `AggregateEval` | `AggregateExec` | Aggregation with `by`/`without` grouping |

`InstantFuncToProjection` in `src/plan/expr.rs` is a custom DataFusion optimizer rule that converts simple instant functions (abs, ceil, etc.) into projections to avoid custom execution nodes.

### Plan translation (`src/plan/`)

- `plan/expr.rs`: Main recursive translator from `promql_parser::Expr` to `LogicalPlan`. This is the most complex file.
- `plan/selector.rs`: Translates `VectorSelector`/`MatrixSelector` to table scans with time-range and label-matcher filters.

### Functions (`src/func/`)

- `func/range.rs`: `rate`, `irate`, `increase`, `delta` — operate on a sliding window of `(timestamp, value)` pairs
- `func/instant.rs`: Math/trig/clamping functions — implemented as DataFusion scalar UDFs
- `func/aggregate.rs`: Aggregation function registry

### Data format detail

The Rezolus parquet test data (`data/metrics.parquet`) has ~950 columns in wide format. Column names like `cgroup_cpu_cycles//system.slice/chrony.service/28` encode metric name + labels. The `ColumnMapping.parse_column` closure in `ParquetMetricSource` (`src/parquet.rs`, behind `--features parquet`) handles this parsing.

### What's not yet implemented

See `.claude/plans/functions.md` for the full list. Notable gaps:
- Range functions: `avg_over_time`, `deriv`, `predict_linear`, `*_over_time` variants
- Aggregators: `topk`, `bottomk`, `quantile`, `stddev`, `stdvar`
- Modifiers: `@` (fixed timestamp), `bool` on comparisons
- Subqueries

## Testing approach

Integration tests live in `tests/integration/` and use an `InMemoryMetricSource` (defined inline in each test file) that implements `MetricSource` with Arrow `RecordBatch` data. Tests instantiate `PromqlEngine`, execute queries, and assert on returned `QueryResult` values.

Parquet-dependent tests (`parquet_query.rs`, `rezolus_query.rs`) are feature-gated and require `--features parquet`.
