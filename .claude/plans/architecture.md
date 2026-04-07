# datafusion-promql Architecture

Last updated: 2026-04-07.

## Overview

A PromQL query engine built on Apache DataFusion. Users write PromQL queries
(e.g., `rate(cpu_usage[5m])`), and the engine translates them into DataFusion
logical/physical plans for execution against a pluggable data source.

---

## Dependencies

```toml
[dependencies]
promql-parser = "0.8"          # PromQL parsing -> AST (maintained by GreptimeDB)
datafusion = "53"              # Query engine, optimization, Arrow memory format
arrow = { version = "58", features = ["prettyprint"] }
clap = { version = "4", features = ["derive"] }  # CLI argument parsing
async-trait = "0.1"
thiserror = "2"
chrono = "0.4"
futures = "0.3"
regex = "1"
tokio = { version = "1", features = ["full"] }

[features]
parquet = ["datafusion/parquet", "dep:parquet"]
plot = ["parquet", "textplots", "rgb", "terminal_size"]
```

---

## Module Layout

```
src/
├── lib.rs                 # Public API: PromqlPlanner, PromqlEngine, QueryResult, re-exports
├── error.rs               # Error types (PromqlError enum)
├── types.rs               # Shared types: Labels, TimeRange, InstantSample, RangeSamples, QueryResult
├── datasource.rs          # MetricSource trait + TableFormat enum
├── normalize.rs           # Wide-to-long format conversion (UNION ALL projections)
├── plan/
│   ├── mod.rs             # Plan translation entry point, EvalParams
│   ├── expr.rs            # PromQL Expr -> DataFusion LogicalPlan recursive translator (~800 lines, most complex file)
│   └── selector.rs        # Vector/matrix selector -> table scan + filter plans
├── node/                  # Custom UserDefinedLogicalNode definitions
│   ├── mod.rs
│   ├── instant_eval.rs    # InstantVectorEval (single-timestamp step alignment)
│   ├── step_eval.rs       # StepVectorEval (multi-timestamp range query step alignment)
│   ├── range_eval.rs      # RangeVectorEval (sliding window for range functions)
│   ├── range_func_eval.rs # RangeFuncEval (range function application)
│   ├── binary_eval.rs     # BinaryEval + ScalarBinaryEval (series matching for binary ops)
│   ├── instant_function.rs # InstantFunction (generic instant function wrapper)
│   └── datetime_function.rs # DateTimeFunction (lowered to projection by optimizer)
├── exec/                  # Physical ExecutionPlan implementations
│   ├── mod.rs             # PromqlExtensionPlanner (maps logical -> physical nodes)
│   ├── instant_eval.rs    # InstantVectorExec
│   ├── step_eval.rs       # StepVectorExec
│   ├── range_eval.rs      # RangeVectorExec
│   ├── range_func_eval.rs # RangeFuncExec
│   └── binary_eval.rs     # BinaryExec + ScalarBinaryExec
├── func/                  # Function implementations
│   ├── mod.rs             # Function registry and lookup by name
│   ├── range.rs           # Range vector functions: rate, irate, increase, delta, idelta, avg_over_time
│   ├── range_udaf.rs      # User-defined aggregate function wrappers for range operations
│   ├── instant.rs         # Instant vector function enum (abs, ceil, floor, trig, etc.)
│   ├── aggregate.rs       # Aggregation operators: sum, avg, count, min, max, stddev, stdvar, group, topk, bottomk, quantile, count_values
│   ├── datetime.rs        # DateTime functions: timestamp, day_of_month, hour, minute, month, year, etc.
│   ├── label.rs           # label_replace, label_join
│   ├── sort.rs            # sort, sort_desc, sort_by_label, sort_by_label_desc
│   └── udf/              # Individual scalar UDF implementations (28 files)
│       ├── abs.rs, ceil.rs, floor.rs, round.rs, sqrt.rs, exp.rs, ln.rs, log2.rs, log10.rs, sgn.rs
│       ├── clamp.rs, clamp_min.rs, clamp_max.rs
│       ├── sin.rs, cos.rs, tan.rs, asin.rs, acos.rs, atan.rs
│       ├── sinh.rs, cosh.rs, tanh.rs, asinh.rs, acosh.rs, atanh.rs
│       └── deg.rs, rad.rs
├── opt/                   # Custom optimizer rules
│   └── logical/
│       ├── mod.rs
│       ├── instant_func_to_projection.rs  # Convert InstantFunction nodes to Projection
│       ├── datetime_func_to_projection.rs # Convert DateTimeFunction nodes to Projection
│       ├── range_vector_to_aggregation.rs # Convert RangeVectorEval patterns to DataFusion aggregation
│       ├── push_instant_eval_through_union.rs # Push InstantVectorEval past Union nodes
│       ├── lift_constant_projections.rs   # Lift constant expressions out of projections
│       ├── fold_redundant_aggregation.rs  # Remove redundant aggregation layers
│       └── remove_noop_projections.rs     # Clean up identity projections
├── parquet.rs             # (feature = "parquet") ParquetMetricSource for wide-format Rezolus files
└── bin/
    ├── query-graph.rs     # Visualize PromQL AST
    ├── query-plan.rs      # Show DataFusion logical/optimized plans (requires parquet)
    └── query-plot.rs      # Execute queries and plot in terminal (requires plot)
```

---

## Data Source Abstraction (`datasource.rs`)

The `MetricSource` trait is the pluggable data backend:

```rust
#[async_trait]
pub trait MetricSource: Send + Sync {
    async fn table_for_metric(
        &self,
        metric_name: &str,
        matchers: &[Matcher],
        time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)>;

    async fn list_metrics(
        &self,
        name_matcher: Option<&Matcher>,
    ) -> Result<Vec<MetricMeta>>;
}
```

`TableFormat` has two variants:
- **`Long`**: Standard Prometheus layout — `__name__` (Utf8), `timestamp` (UInt64), `value` (Float64), plus Utf8 label columns.
- **`Wide(ColumnMapping)`**: One column per series (Rezolus-style parquet). The engine automatically normalizes wide→long via `normalize.rs` using UNION ALL projections.

`ColumnMapping` provides a `parse_column` closure that maps column names to `(metric_name, labels)`. For the Rezolus test data, a column like `cgroup_cpu_cycles//system.slice/chrony.service/28` maps to metric `cgroup_cpu_cycles` with labels `{cgroup="/system.slice/chrony.service", id="28"}`.

---

## Public API (`lib.rs`)

Two layers:

**`PromqlPlanner`** — step-by-step access to each pipeline stage:
- `instant_logical_plan()` / `range_logical_plan()` → unoptimized `LogicalPlan`
- `optimize_logical_plan()` → optimized `LogicalPlan`
- `create_physical_plan()` → `Arc<dyn ExecutionPlan>`
- `execute()` → `Vec<RecordBatch>`
- `batches_to_vector()` / `batches_to_matrix()` → `QueryResult`

**`PromqlEngine`** — high-level convenience wrapper:
- `instant_query(query, timestamp)` → `QueryResult::Vector`
- `range_query(query, start, end, step)` → `QueryResult::Matrix`

Timestamps use `DateTime<Utc>` at the public API boundary, converted to `u64` nanoseconds internally.

---

## Translation Pipeline

```
PromQL string
    │
    ▼
promql_parser::parse()  →  promql_parser::Expr (AST)
    │
    ▼
plan::expr::plan_expr()  →  DataFusion LogicalPlan
    │                        (with custom UserDefinedLogicalNodes)
    ▼
DataFusion optimizer     →  Optimized LogicalPlan
    │                        (standard rules + 7 custom rules)
    ▼
Physical planner         →  ExecutionPlan DAG
    │                        (PromqlExtensionPlanner maps
    │                         logical nodes to physical nodes)
    ▼
execute().collect()      →  Vec<RecordBatch>
    │
    ▼
Collect into QueryResult
```

### Custom Nodes

| Logical node | Physical node | Purpose |
|---|---|---|
| `InstantVectorEval` | `InstantVectorExec` | Single-timestamp step alignment with lookback window |
| `StepVectorEval` | `StepVectorExec` | Multi-timestamp range query step alignment |
| `RangeVectorEval` | `RangeVectorExec` | Sliding window sample collection for range functions |
| `RangeFuncEval` | `RangeFuncExec` | Range function application (rate, delta, etc.) |
| `BinaryEval` | `BinaryExec` | Vector-vector binary ops with `on`/`ignoring`/`group_left`/`group_right` |
| `ScalarBinaryEval` | `ScalarBinaryExec` | Vector-scalar binary ops |
| `InstantFunction` | *(lowered by optimizer)* | Instant function wrapper, converted to Projection |
| `DateTimeFunction` | *(lowered by optimizer)* | DateTime function wrapper, converted to Projection |

### Custom Optimizer Rules

1. **`InstantFuncToProjection`** — Converts `InstantFunction` nodes to standard Projection nodes
2. **`DateTimeFuncToProjection`** — Converts `DateTimeFunction` nodes to standard Projection nodes
3. **`RangeVectorToAggregation`** — Converts `RangeVectorEval` patterns to DataFusion aggregation where possible
4. **`PushInstantEvalThroughUnion`** — Pushes `InstantVectorEval` past Union nodes (important for wide→long normalization)
5. **`LiftConstantProjections`** — Lifts constant expressions out of projections, flattens nested projections
6. **`FoldRedundantAggregation`** — Removes redundant aggregation layers
7. **`RemoveNoopProjections`** — Cleans up identity projections

---

## Range Vector and Step Evaluation

PromQL evaluates at discrete time steps:

1. **Step generation**: For `range_query(start, end, step)`, timestamps are: `[start, start+step, start+2*step, ..., end]`
2. **InstantVectorEval/StepVectorEval**: For each step timestamp `t`, find the most recent sample where `t - lookback <= sample.timestamp <= t`. Default lookback = 5 minutes. `InstantVectorEval` handles single-timestamp queries; `StepVectorEval` handles multi-step range queries.
3. **RangeVectorEval + RangeFuncEval**: For each step timestamp `t` and range duration `d`, collect all samples where `t - d <= sample.timestamp <= t`, then apply the range function.

Physical execution nodes receive timestamp-sorted input from their children and maintain sliding window buffers, emitting one row per (step_timestamp, series).

---

## Testing

Integration tests in `tests/integration/` (21 files, ~8,400 lines) use `InMemoryMetricSource` with hand-crafted Arrow `RecordBatch` data. Tests instantiate `PromqlEngine`, execute queries, and assert on `QueryResult` values.

Categories:
- **Core queries**: `instant_query.rs`, `range_query.rs`, `inspect_offset.rs`
- **Functions**: `abs.rs`, `clamp.rs`, `round.rs`, `instant.rs`, `trig.rs`, `datetime.rs`, `label.rs`, `sort.rs`
- **Aggregation**: `aggregate_ops.rs`, `aggregate_binary.rs`
- **Optimizer**: `lift_constant_projections.rs`, `push_instant_eval_through_union.rs`, `fold_redundant_aggregation.rs`
- **Parquet** (feature-gated): `parquet_query.rs`, `rezolus_query.rs`

---

## What's Not Yet Implemented

See `functions.md` for the detailed list. Major gaps:
- Most `*_over_time` range functions (`sum_over_time`, `min_over_time`, etc.)
- `deriv`, `predict_linear`
- `scalar`, `vector`, `absent`, `absent_over_time`
- `histogram_quantile`
- `@` timestamp modifier
- Subqueries
