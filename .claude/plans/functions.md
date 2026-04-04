# Missing PromQL Functions

This document tracks PromQL functions that still need to be implemented in
datafusion-promql. Functions are grouped by category.

## Currently Implemented

- **Range vector functions:** `rate`, `irate`, `increase`, `delta`
- **Aggregation operators:** `sum`, `avg`, `count`, `min`, `max`
- **Binary operators:** `+`, `-`, `*`, `/`, `%`, `^`, `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Set operators:** `and`, `or`, `unless`

---

## Range Vector Functions

These operate on a matrix (range vector) and return an instant vector.

| Function | Description |
|----------|-------------|
| `deriv` | Calculate the per-second derivative of a gauge time series using simple linear regression. |
| `predict_linear` | Predict the value of a gauge `t` seconds in the future using simple linear regression over the range vector. Takes two arguments: a range vector and a scalar `t`. |
| `avg_over_time` | Average value of all samples in the range. |
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
| `idelta` | Difference between the last two samples in the range (gauge equivalent of `irate`). |
| `absent_over_time` | Returns an empty vector if the range vector has any elements, or a 1-element vector with value 1 if the range vector has no elements. |

---

## Instant Vector Functions

These operate on instant vectors (or scalars) and return instant vectors (or scalars).

### Math Functions

| Function | Description |
|----------|-------------|
| `abs` | Absolute value of each sample. |
| `ceil` | Round each sample value up to the nearest integer. |
| `floor` | Round each sample value down to the nearest integer. |
| `round` | Round each sample value to the nearest integer. Optional `to_nearest` parameter (default 1) controls rounding granularity. |
| `sqrt` | Square root of each sample value. |
| `exp` | Exponential function: e raised to the power of each sample value. |
| `ln` | Natural logarithm of each sample value. |
| `log2` | Base-2 logarithm of each sample value. |
| `log10` | Base-10 logarithm of each sample value. |
| `sgn` | Returns the sign of each sample: -1 if negative, 0 if zero, 1 if positive. |

### Clamping Functions

| Function | Description |
|----------|-------------|
| `clamp` | Clamp each sample value to the range `[min, max]`. Takes a vector, a scalar min, and a scalar max. |
| `clamp_min` | Clamp each sample value to have a minimum of the given scalar. |
| `clamp_max` | Clamp each sample value to have a maximum of the given scalar. |

### Trigonometric Functions

| Function | Description |
|----------|-------------|
| `acos` | Arccosine of each sample value (radians). |
| `asin` | Arcsine of each sample value (radians). |
| `atan2` | Two-argument arctangent. This is a binary operator, not a function — `atan2(y, x)` computes the angle. |
| `cos` | Cosine of each sample value (radians). |
| `sin` | Sine of each sample value (radians). |
| `tan` | Tangent of each sample value (radians). |
| `acosh` | Inverse hyperbolic cosine of each sample value. |
| `asinh` | Inverse hyperbolic sine of each sample value. |
| `atanh` | Inverse hyperbolic tangent of each sample value. |
| `cosh` | Hyperbolic cosine of each sample value. |
| `sinh` | Hyperbolic sine of each sample value. |
| `tanh` | Hyperbolic tangent of each sample value. |
| `deg` | Convert each sample value from radians to degrees. |
| `rad` | Convert each sample value from degrees to radians. |
| `pi` | Returns the mathematical constant π (no arguments). |

### Date/Time Functions

| Function | Description |
|----------|-------------|
| `time` | Returns the evaluation timestamp (seconds since epoch) as a scalar. Takes no arguments. |
| `timestamp` | Returns the timestamp of each sample as a float64 (seconds since epoch). |
| `day_of_month` | Returns the day of the month (1–31) for the timestamp of each sample. Defaults to current eval time if no argument. |
| `day_of_week` | Returns the day of the week (0=Sunday, 6=Saturday) for each sample timestamp. |
| `day_of_year` | Returns the day of the year (1–365/366) for each sample timestamp. |
| `days_in_month` | Returns the number of days in the month for each sample timestamp. |
| `hour` | Returns the hour of the day (0–23) for each sample timestamp. |
| `minute` | Returns the minute of the hour (0–59) for each sample timestamp. |
| `month` | Returns the month of the year (1–12) for each sample timestamp. |
| `year` | Returns the year for each sample timestamp. |

### Label Manipulation Functions

| Function | Description |
|----------|-------------|
| `label_replace` | Replace label values using regex. `label_replace(v, dst_label, replacement, src_label, regex)` — for each series, if `src_label` matches `regex`, set `dst_label` to `replacement` (with `$1`-style capture group references). |
| `label_join` | Concatenate label values into a new label. `label_join(v, dst_label, separator, src_label_1, src_label_2, ...)` — joins the values of the source labels with the separator and writes the result to `dst_label`. |

### Type Conversion / Utility Functions

| Function | Description |
|----------|-------------|
| `scalar` | Convert a single-element instant vector to a scalar. Returns NaN if the vector has != 1 element. |
| `vector` | Convert a scalar to a single-element instant vector. |
| `absent` | Returns a 1-element vector with value 1 if the input vector is empty, otherwise returns an empty vector. Preserves label matchers as labels on the result. |

### Sorting Functions

| Function | Description |
|----------|-------------|
| `sort` | Sort the elements of a vector by sample value (ascending). |
| `sort_desc` | Sort the elements of a vector by sample value (descending). |
| `sort_by_label` | Sort the elements of a vector by the given label values (ascending, lexicographic). |
| `sort_by_label_desc` | Sort the elements of a vector by the given label values (descending, lexicographic). |

---

## Aggregation Operators

These aggregate across series, grouping by `by(...)` or `without(...)` clauses.

| Function | Description |
|----------|-------------|
| `topk` | Returns the top K elements by value. Unlike other aggregators, the result preserves the original labels (no grouping collapse). |
| `bottomk` | Returns the bottom K elements by value. Same label-preservation semantics as `topk`. |
| `quantile` | Calculate the φ-quantile (0 ≤ φ ≤ 1) over the grouped values. |
| `count_values` | Count the number of elements with the same value. Takes a string parameter used as the label name for the value. |
| `stddev` | Population standard deviation over the grouped values. |
| `stdvar` | Population variance over the grouped values. |
| `group` | Groups series together — all resulting values are 1. Useful for enumerating label combinations. |
| `limitk` | Limit to K arbitrary series from each group. |
| `limit_ratio` | Sample a ratio (0.0–1.0) of series from each group. |

---

## Histogram Functions

| Function | Description |
|----------|-------------|
| `histogram_quantile` | Calculate the φ-quantile from a conventional histogram. Takes a scalar φ and an instant vector of histogram bucket counts (series must have a `le` label). Interpolates within buckets. |

---

## Modifiers (Not Functions, but Missing Features)

These are expression modifiers rather than functions, but are tracked here since they affect evaluation:

| Modifier | Description |
|----------|-------------|
| ~~`offset`~~ | ~~Shift the time range of a selector backwards by a fixed duration: `metric_name offset 5m`.~~ **(implemented)** |
| `@` | Evaluate a selector at a fixed timestamp: `metric_name @ 1609459200`. |
| `bool` | On comparison binary operators, return 0/1 instead of filtering: `metric > bool 10`. |
| Subqueries | Evaluate an instant vector expression over a range: `rate(metric[5m])[30m:1m]`. |
