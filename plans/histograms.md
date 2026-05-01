# Native Histogram Support

Status: design draft, not implemented. Modeled on metriken-query's handling of
the iopsystems/histogram crate's base-2 log-linear sparse layout, adapted to
DataFusion. The deliverable is parity with `histogram_quantile`,
`histogram_sum`, `histogram_count`, `histogram_avg`, `histogram_fraction`, plus
the metriken extensions `histogram_percentiles` and `histogram_heatmap`.

This document records five design decisions and lists the files each one
touches. It does not prescribe an implementation order.

## Glossary

- *Config*: `(grouping_power, max_value_power)` — the histogram crate's
  parameters that fix the bucket boundaries. Fixed per metric, not per row.
- *Sparse layout*: only buckets with non-zero counts are stored, as parallel
  `(index, count)` arrays.
- *Cumulative*: the histogram in a row is the running total since the agent
  started. *Delta*: the histogram is the increment over the previous row.

## 1. Arrow representation at runtime

**Decision: (b) a single `Struct<indices: List<UInt64>, counts: List<UInt64>>`
column.** The dense `:buckets` form on disk is unpacked to sparse `(indices,
counts)` at load time and surfaces in the engine as one struct column.

Why not (a) two parallel columns: every UDF/UDAF in `src/func/udf/` and
`src/func/range_udaf.rs` declares its arity through a `Signature`. Splitting the
histogram into two arguments doubles the surface area of every histogram
function — each call site has to project both columns, every optimizer rule
that pushes expressions through a node has to keep them paired, and the
`InstantVectorEval` / `StepVectorEval` / `RangeVectorEval` nodes (which today
carry a single `value` column through the pipeline) need to learn that some
"values" travel as pairs. A struct keeps it as one column.

Why not (c) an Arrow extension type: extension types are stored as metadata on
the field, but DataFusion's expression rewriting and projection pushdown
machinery operates on `DataType`. We would have to teach every layer to
preserve the extension metadata, and the `Signature` machinery has no
first-class support for "Float64 OR HistogramExt"; we'd be matching by struct
shape anyway. Same place we end up with (b), with extra plumbing.

Why (b) works:
- `RangeAggregateUdf` in `src/func/range_udaf.rs` already takes
  `Exact(vec![DataType::UInt64, DataType::Float64])` for `(timestamp, value)`.
  A histogram-flavored variant takes
  `Exact(vec![DataType::UInt64, struct_type])`, where `struct_type` is the
  fixed `Struct<indices: List<UInt64>, counts: List<UInt64>>`. One argument,
  not two, and the existing `state_fields` machinery (already returning
  `List<List<...>>` for samples) extends naturally.
- A scalar UDF for `histogram_quantile(q, h)` declares
  `Exact(vec![DataType::Float64, struct_type])` and reaches for
  `array.as_struct()` in `invoke_with_args` — the same pattern
  `src/func/udf/clamp.rs` uses for primitives.
- The `value` column in long format becomes `value: Struct<...>` for histogram
  metrics. Existing nodes that grep for the literal name `"value"` (and there
  are many — see `src/lib.rs::extract_row`, `src/normalize.rs`, every node in
  `src/node/`) keep working as long as they do not assume `Float64`. The bulk
  of the change is making the `value` column data-type-polymorphic at the
  schema level.

**Files affected:**
- `src/datasource.rs` — `TableFormat` doc comment on the long format must say
  `value` is `Float64` or `Struct<indices, counts>`. No new variant.
- `src/normalize.rs::plan_wide_single_scan` — when a wide column is histogram-
  typed, the cast in the projection is `cast` to the struct type, not Float64;
  the `WideUnpack` exec needs a histogram code path.
- `src/exec/wide_unpack.rs` — emit struct-typed values instead of Float64 for
  histogram columns.
- `src/lib.rs::extract_row`, `aggregate_by_label_series` — must downcast
  `value` to the right Arrow type. For `Struct`, we don't synthesize a numeric
  `QueryResult::Vector` row; the histogram functions terminate the histogram
  pipeline back into a Float64 before this layer.
- A new `src/func/histogram.rs` module to host `histogram_quantile` and
  friends, and a new `src/func/histogram_udaf.rs` for the streaming/range
  histogram aggregator (`histogram_sum_over_time`, etc.).

## 2. Where Config (`grouping_power`, `max_value_power`) lives

**Decision: field metadata on the histogram column at the
`MetricSource::table_for_metric` boundary, propagated as struct field metadata
on the `value` column.** No engine-side registry.

Why field metadata, not a registry:
- DataFusion already preserves Arrow `Field::metadata()` through projections
  and most physical operators. A registry keyed by metric name forces every
  operator we already have to look up "what config does this column have"
  every time it inspects a row. Field metadata travels with the schema for
  free.
- A registry has a real propagation problem at sum-of-histograms: when the
  optimizer pushes `sum` through a `WideUnpack`, the resulting histogram is a
  union of multiple input columns whose configs must agree. Verifying that at
  registry-lookup time means threading the metric name down through nodes
  that previously only cared about Arrow types. Verifying it at field-metadata
  time means the optimizer rule reads it off the input fields and refuses to
  push if they disagree, the same way it already reads dtypes.
- The metriken-exposition wire format already writes `grouping_power` and
  `max_value_power` as Arrow field metadata (see
  `metriken-exposition/src/parquet.rs:260-296`). The
  `ParquetMetricSource::rezolus_parse_column` in `src/parquet.rs` already
  reads other metadata keys; reading these two is a one-line addition to its
  reserved-keys list.

How it propagates:
- `MetricSource::table_for_metric` returns a provider whose schema already has
  the histogram column's field metadata populated. The wide-format normalizer
  must copy that metadata onto the unpacked `value` column (it's per-row, not
  per-column-of-the-Struct, so it goes on the outer `value` field).
- Custom logical nodes (`InstantVectorEval`, `StepVectorEval`, `RangeVectorEval`,
  `BinaryEval`) that today preserve Arrow types from input to output must also
  preserve the `value` field's metadata.
- `BinaryEval` between two histogram series (e.g. addition) requires Configs
  to match. The check happens in `plan_expr.rs` when the binary op is
  constructed; if the configs disagree, return a planning error rather than
  attempting to coerce at runtime.

**Files affected:**
- `src/parquet.rs::rezolus_parse_column` and `RESERVED` list (already has
  `grouping_power` / `max_value_power`).
- `src/normalize.rs::plan_wide_single_scan` — propagate field metadata onto
  the projected `value` column.
- `src/node/instant_eval.rs`, `src/node/step_eval.rs`, `src/node/range_eval.rs`,
  `src/node/binary_eval.rs` — make sure derived schemas keep the metadata on
  the `value` field.
- `src/plan/expr.rs` — at binary-op construction, validate Config equality
  for histogram operands.

## 3. Cumulative vs delta storage

**Decision: eager pre-differencing in the `MetricSource`, mirroring metriken-
query's `stream_histogram_column` (`metriken-query/src/tsdb/mod.rs:84`).** The
table provider returned to the engine yields per-period delta histograms, not
cumulatives.

Reasoning:
- The lazy alternative is a custom logical/physical node that consumes a
  cumulative column and produces a delta column. It would be the histogram
  analogue of the existing `RangeFuncEval`, but unlike `rate`/`delta` it
  cannot start from "first row of the window" — to compute the first delta in
  the window you need the row immediately before it. Plumbing that "lookback
  by one row" into a streaming node duplicates work that the data source can
  do once at load time.
- All existing range-vector functions in `src/func/range.rs` assume the
  scalar values they receive are the values they should aggregate over.
  `rate` already implicitly takes a cumulative counter and produces a per-
  second rate; the equivalent of "rate" on histograms is "the delta between
  successive cumulative snapshots", and metriken-query's approach of doing
  this once at ingest is the simplest match.
- Empty-delta semantics matter for time-axis alignment.
  `stream_histogram_column` records an explicit empty delta on null rows /
  decode failures / counter resets; we should do the same so the timestamp
  axis stays consistent with the scalar columns next to the histogram.
- Eager has a real cost: a histogram column with N rows holds N-1 deltas in
  memory after load. `stream_histogram_column` already addresses this by
  using `delta_to_32_or_empty` (u32 counts) — we should do the same in the
  parquet source. For non-parquet sources (live ingest), the
  `prev_histograms` sidecar pattern from metriken-query
  (`tsdb/mod.rs:386-403`) is the model.

When we'd revisit: if we add a streaming/push-based `MetricSource` whose
backing data is already-deltas (e.g. a live OTLP feed where the wire format is
delta histograms), eager differencing is wrong for that source. The
`MetricSource` abstraction should let each implementation declare cumulative-
vs-delta and only the cumulative case does the differencing at scan time.
This is a property of the source, not the engine.

**Files affected:**
- `src/parquet.rs` — `ParquetMetricSource::table_for_metric` for histogram
  metrics returns a provider whose Arrow output is per-period deltas, not
  cumulatives. Likely a wrapper `TableProvider` that does the differencing
  pass; or a new exec that wraps the parquet scan and emits deltas.
- `src/datasource.rs` — add a `cumulative: bool` (or richer enum) field to
  whatever per-metric metadata we surface, so the engine can reason about
  whether the source has already pre-differenced.
- A new module — say `src/histogram_delta.rs` — for the differencing logic
  itself, sharing pseudocode with `metriken-query/src/tsdb/mod.rs`.

## 4. `histogram_percentiles` / `histogram_heatmap` parsing

**Decision: pre-parse intercept, mirroring metriken-query's
`handle_histogram_percentiles` (`metriken-query/src/promql/mod.rs:1534`).**
Before handing the query string to `promql_parser::parser::parse`, check for
the literal prefixes `histogram_percentiles(` and `histogram_heatmap(`. If
they match, dispatch to a hand-rolled parser that handles the array literal
and the metric selector; otherwise, parse normally.

Why not fork promql-parser:
- Forking owns the parser indefinitely. promql-parser is upstream-maintained
  and tracks Prometheus syntax extensions (subqueries, `@`, etc.) that we
  want to inherit for free. A fork would have to manually rebase every
  upstream change.
- The array-literal extension has a narrow, well-defined surface: only inside
  `histogram_percentiles(` and only as the first argument. A fork would carry
  a much larger maintenance burden than a 30-line pre-parser.

Why not skip them entirely:
- They're the primary motivation for adapting metriken-query's design. The
  Rezolus viewer relies on `histogram_heatmap` for its latency UI, and any
  caller migrating off metriken-query will already be using the array form
  of `histogram_percentiles`.

The pre-parser is the same logic as metriken-query's `parse_optional_stride`
plus `split_last_top_level_comma`: find the array literal, parse comma-
separated floats, hand the metric selector substring to
`promql_parser::parser::parse` so we still get correct matcher semantics.

**Files affected:**
- A new pre-parser in `src/plan/mod.rs` or `src/lib.rs::PromqlEngine` —
  intercepts before `promql_parser::parser::parse`. It calls `plan_expr`
  with a synthesized AST (or bypasses it, returning a precomputed plan).
- `src/lib.rs::PromqlEngine::range_query` and `instant_query` — call the
  pre-parser before parsing.
- `src/plan/expr.rs` — register the underlying multi-quantile evaluation as
  an internal-only logical node; the public PromQL surface is the pre-parser.

## 5. Heatmap result shape

**Decision: synthesize per-bucket series with a synthetic `le` label, the
Prometheus-classical heatmap convention.** No new `QueryResult` variant.

Why not a `QueryResult::Heatmap`:
- The user-visible API of `QueryResult` is "rows of (timestamp, labels,
  f64)"; binaries like `query-plot` and downstream consumers in
  `tests/integration/` all unpack one of three shapes. Adding a heatmap
  variant means every consumer has to handle a fourth case, and the response
  shape becomes asymmetric: "matrix with bucket labels" is a strict subset of
  what `QueryResult::Matrix` already represents.
- The synthetic-label approach is what Prometheus's classic histogram does
  — bucket boundaries surface as a `le` label on each per-bucket time series.
  Reusing that convention means `histogram_heatmap(metric)` returns the same
  shape as a query against a Prometheus classic histogram, which is what
  most heatmap renderers already speak.

What gets emitted:
- One time series per non-empty bucket, with the histogram metric's labels
  plus `le = <upper bound>` (the histogram crate's bucket upper bound at that
  index, formatted as a decimal string for compatibility with Prometheus
  conventions).
- The value at each timestamp is the **cumulative** count for that bucket —
  i.e. the sum of all bucket counts at indices ≤ this one, matching
  Prometheus's classical `le`-cumulative convention. This way
  `histogram_heatmap` is a drop-in replacement for the classic Prometheus
  histogram pattern in renderers that already speak `le`.

Implementation pipeline (logical order, applied as the final stages of the
heatmap plan):

1. **Sparse-merge UDAF** — sums the histograms across the matched series
   into a single per-timestamp histogram. Config-aware (Configs must agree).
   New file: `src/func/histogram_udaf.rs`.
2. **`HistogramToCumulative` node (new custom logical/physical node)** —
   row-wise transform on the `value: Struct<indices, counts>` column that
   replaces the sparse non-cumulative representation with a dense (or sparse-
   prefix-summed) cumulative representation. Conceptually a per-row prefix
   sum over `(indices, counts)`: for each timestamp, expand the sparse pairs
   into a cumulative count keyed by bucket upper bound. Runs as the *last*
   step before the heatmap fan-out so that all upstream histogram operations
   (sum, delta, etc.) work on the compact sparse non-cumulative form they
   were designed for.
3. **Fan-out projection** — emits one row per `(bucket_upper_bound,
   timestamp)` pair, creating the synthetic `le` label and a Float64 `value`.
   At this point the histogram pipeline has terminated and the rest of the
   query behaves like a normal `QueryResult::Matrix`.

The `HistogramToCumulative` node is its own custom node (rather than a UDF)
because:
- It changes the logical schema of the `value` column (from sparse non-
  cumulative struct to either a dense `List<UInt64>` or a sparse-cumulative
  struct, depending on the variant we pick). UDFs in DataFusion are
  expression-level and don't naturally describe a row-wise schema-changing
  transform that needs to read the column's Config metadata to compute its
  output schema.
- It's a pipeline pinch point: the optimizer can verify it appears exactly
  once at the heatmap output and not somewhere upstream where it would
  defeat the sparse-storage optimizations.

We expose the node only through `histogram_heatmap` lowering — it is not a
standalone PromQL function.

**Files affected:**
- `src/func/histogram.rs` (new) — `histogram_heatmap` lowering: aggregate
  with sparse-merge UDAF, then `HistogramToCumulative`, then fan-out
  projection emitting per-bucket series.
- `src/func/histogram_udaf.rs` (new) — sparse-merge UDAF.
- `src/node/histogram_to_cumulative.rs` (new) — logical node definition.
- `src/exec/histogram_to_cumulative.rs` (new) — physical exec.
- `src/exec/mod.rs::PromqlExtensionPlanner` — register the new node mapping.
- No change to `src/types.rs::QueryResult`.

## Non-goals

- **Prometheus 2.40+ exponential native histograms.** Different bucket
  schema — they use a base-2-with-schema-parameter exponential layout (see
  Prometheus's `histogram.FloatHistogram` and the OTLP exponential histogram
  data model). We are targeting the iopsystems/histogram base-2 log-linear
  layout used by metriken-exposition. A future Prometheus-native histogram
  source would need a separate Config representation and bucket-bound
  computation; the surface PromQL functions are by-and-large the same, but
  the kernel is not.
- **Subquery histograms** (`histogram_quantile(0.99, rate(latency[1m])[5m:])`).
  Subqueries are not yet implemented for scalar metrics either; their
  histogram extension can wait until that lands.
- **`@` and `offset` modifiers on histogram columns specifically.** These
  should fall out of the scalar implementation once the `value` column type
  is polymorphic; no extra design work.
- **Mixing classic-Prometheus-style `_bucket{le=...}` series with native
  histograms in the same query.** A future feature; both paths work
  independently.

## Notes carried into review

- The struct-typed `value` column is the single largest invasion of the
  existing codebase. Every node in `src/node/` and `src/exec/` that today
  assumes `Float64` needs an audit. Suggestion: stage the work as
  (1) introduce the type and a no-op pipeline that just reads and emits it,
  (2) add `histogram_quantile` end-to-end, (3) add the rest.
- The eager-differencing decision is a property of the parquet source. The
  `MetricSource` trait should grow a way to declare cumulative vs delta, so a
  future live-ingest source can opt out.
- The `HistogramToCumulative` node is the only piece that materializes a
  cumulative-bucket form. Everything upstream stays sparse and non-cumulative,
  which preserves the storage and merge-cost wins of the sparse layout.
- Pre-parsing for `histogram_percentiles` / `histogram_heatmap` is a temporary
  shape; if upstream promql-parser ever gains array literals, we drop the
  intercept.
