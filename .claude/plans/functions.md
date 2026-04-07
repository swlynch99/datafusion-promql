# PromQL Function Implementation Status

This document tracks PromQL function implementation status in datafusion-promql.
Last updated: 2026-04-07.

## Currently Implemented

- **Range vector functions:** `rate`, `irate`, `increase`, `delta`, `idelta`, `avg_over_time`
- **Instant vector functions (math):** `abs`, `ceil`, `floor`, `round`, `sqrt`, `exp`, `ln`, `log2`, `log10`, `sgn`
- **Instant vector functions (trig):** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `deg`, `rad`
- **Instant vector functions (clamping):** `clamp`, `clamp_min`, `clamp_max`
- **Date/time functions:** `time`, `timestamp`, `day_of_month`, `day_of_week`, `day_of_year`, `days_in_month`, `hour`, `minute`, `month`, `year`
- **Label functions:** `label_replace`, `label_join`
- **Sorting functions:** `sort`, `sort_desc`, `sort_by_label`, `sort_by_label_desc`
- **Aggregation operators:** `sum`, `avg`, `count`, `min`, `max`, `stddev`, `stdvar`, `group`, `topk`, `bottomk`, `quantile`, `count_values`, `limitk`, `limit_ratio`
- **Binary operators:** `+`, `-`, `*`, `/`, `%`, `^`, `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Set operators:** `and`, `or`, `unless`
- **Modifiers:** `offset`, `bool`, `@`

---

## Not Yet Implemented

### Range Vector Functions

These operate on a matrix (range vector) and return an instant vector.

| Function | Description |
|----------|-------------|
| `deriv` | Calculate the per-second derivative of a gauge time series using simple linear regression. |
| `predict_linear` | Predict the value of a gauge `t` seconds in the future using simple linear regression over the range vector. Takes two arguments: a range vector and a scalar `t`. |
| `sum_over_time` | Sum of all sample values in the range. |
| `count_over_time` | Count of all samples in the range. |
| `min_over_time` | Minimum sample value in the range. |
| `max_over_time` | Maximum sample value in the range. |
| `stddev_over_time` | Population standard deviation of values in the range. |
| `stdvar_over_time` | Population variance of values in the range. |
| `quantile_over_time` | The φ-quantile (0 ≤ φ ≤ 1) of values in the range. Takes a scalar φ and a range vector. |
| `last_over_time` | The most recent sample value in the range. |
| `present_over_time` | Returns value 1 for any series that has samples in the range. |
| `changes` | Number of times the value changed within the range. |
| `resets` | Number of counter resets (value decreases) within the range. |
| `absent_over_time` | Returns an empty vector if the range vector has any elements, or a 1-element vector with value 1 if the range vector has no elements. |

### Instant Vector Functions

| Function | Description |
|----------|-------------|
| `pi` | Returns the mathematical constant π (no arguments). |
| `scalar` | Convert a single-element instant vector to a scalar. Returns NaN if the vector has != 1 element. |
| `vector` | Convert a scalar to a single-element instant vector. |
| `absent` | Returns a 1-element vector with value 1 if the input vector is empty, otherwise returns an empty vector. Preserves label matchers as labels on the result. |

### Histogram Functions

| Function | Description |
|----------|-------------|
| `histogram_quantile` | Calculate the φ-quantile from a conventional histogram. Takes a scalar φ and an instant vector of histogram bucket counts (series must have a `le` label). Interpolates within buckets. |

### Modifiers

| Modifier | Description |
|----------|-------------|
| Subqueries | Evaluate an instant vector expression over a range: `rate(metric[5m])[30m:1m]`. |
