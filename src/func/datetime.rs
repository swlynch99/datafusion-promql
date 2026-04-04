use std::any::Any;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{Array, AsArray, Float64Builder};
use arrow::datatypes::{DataType, UInt64Type};
use datafusion::common::utils::take_function_args;
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

/// Date/time functions that extract components from the timestamp column.
///
/// Unlike instant functions (which transform `value`), datetime functions
/// transform the `timestamp` column (nanoseconds since epoch) into a new `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DateTimeFunction {
    /// Returns the timestamp of each sample as seconds since epoch (float64).
    Timestamp,
    /// Day of the month (1–31) in UTC.
    DayOfMonth,
    /// Day of the week (0=Sunday, 6=Saturday) in UTC.
    DayOfWeek,
    /// Day of the year (1–365/366) in UTC.
    DayOfYear,
    /// Number of days in the month for each sample's timestamp, in UTC.
    DaysInMonth,
    /// Hour of the day (0–23) in UTC.
    Hour,
    /// Minute of the hour (0–59) in UTC.
    Minute,
    /// Month of the year (1–12) in UTC.
    Month,
    /// The year in UTC.
    Year,
}

impl fmt::Display for DateTimeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timestamp => write!(f, "timestamp"),
            Self::DayOfMonth => write!(f, "day_of_month"),
            Self::DayOfWeek => write!(f, "day_of_week"),
            Self::DayOfYear => write!(f, "day_of_year"),
            Self::DaysInMonth => write!(f, "days_in_month"),
            Self::Hour => write!(f, "hour"),
            Self::Minute => write!(f, "minute"),
            Self::Month => write!(f, "month"),
            Self::Year => write!(f, "year"),
        }
    }
}

impl DateTimeFunction {
    /// Evaluate the function on a single nanosecond timestamp.
    pub fn evaluate_ns(&self, ts_ns: u64) -> f64 {
        use chrono::{Datelike, TimeZone, Timelike};

        match self {
            Self::Timestamp => ts_ns as f64 / 1_000_000_000.0,
            _ => {
                let dt = chrono::Utc.timestamp_nanos(ts_ns as i64);
                match self {
                    Self::Timestamp => unreachable!(),
                    Self::DayOfMonth => dt.day() as f64,
                    Self::DayOfWeek => {
                        // Prometheus: 0=Sunday, 6=Saturday
                        // chrono: Mon=0 .. Sun=6 (num_days_from_monday)
                        // So: Sunday = (6+1)%7 = 0, Monday = (0+1)%7 = 1, etc.
                        ((dt.weekday().num_days_from_monday() + 1) % 7) as f64
                    }
                    Self::DayOfYear => dt.ordinal() as f64,
                    Self::DaysInMonth => {
                        let m = dt.month();
                        let y = dt.year();
                        days_in_month(y, m) as f64
                    }
                    Self::Hour => dt.hour() as f64,
                    Self::Minute => dt.minute() as f64,
                    Self::Month => dt.month() as f64,
                    Self::Year => dt.year() as f64,
                }
            }
        }
    }

    /// Whether this function drops the `__name__` label from its output.
    #[cfg(test)]
    pub fn drops_metric_name(&self) -> bool {
        true
    }
}

/// Returns the number of days in the given month (1-indexed) of the given year.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Look up a datetime function by name.
///
/// Returns `None` if the name is not a known datetime function.
pub(crate) fn lookup_datetime_function(name: &str) -> Option<DateTimeFunction> {
    match name {
        "timestamp" => Some(DateTimeFunction::Timestamp),
        "day_of_month" => Some(DateTimeFunction::DayOfMonth),
        "day_of_week" => Some(DateTimeFunction::DayOfWeek),
        "day_of_year" => Some(DateTimeFunction::DayOfYear),
        "days_in_month" => Some(DateTimeFunction::DaysInMonth),
        "hour" => Some(DateTimeFunction::Hour),
        "minute" => Some(DateTimeFunction::Minute),
        "month" => Some(DateTimeFunction::Month),
        "year" => Some(DateTimeFunction::Year),
        _ => None,
    }
}

/// Returns true if `name` is the `time()` function.
pub(crate) fn is_time_function(name: &str) -> bool {
    name == "time"
}

/// Create a ScalarUDF for a datetime function.
///
/// The UDF takes a UInt64 column (nanosecond timestamps) and returns Float64.
pub(crate) fn make_datetime_udf(func: DateTimeFunction) -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(DateTimeUdf {
        func,
        signature: Signature::any(1, Volatility::Immutable),
    }))
}

/// Convert a `DateTimeFunction` to a DataFusion logical `Expr` applied to
/// the timestamp column, aliased as `"value"`.
pub(crate) fn datetime_func_to_expr(func: DateTimeFunction, input: Expr) -> Expr {
    let udf = make_datetime_udf(func);
    udf.call(vec![input]).alias("value")
}

#[derive(Debug)]
struct DateTimeUdf {
    func: DateTimeFunction,
    signature: Signature,
}

impl PartialEq for DateTimeUdf {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}
impl Eq for DateTimeUdf {}

impl Hash for DateTimeUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state);
    }
}

impl ScalarUDFImpl for DateTimeUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        match self.func {
            DateTimeFunction::Timestamp => "promql_timestamp",
            DateTimeFunction::DayOfMonth => "promql_day_of_month",
            DateTimeFunction::DayOfWeek => "promql_day_of_week",
            DateTimeFunction::DayOfYear => "promql_day_of_year",
            DateTimeFunction::DaysInMonth => "promql_days_in_month",
            DateTimeFunction::Hour => "promql_hour",
            DateTimeFunction::Minute => "promql_minute",
            DateTimeFunction::Month => "promql_month",
            DateTimeFunction::Year => "promql_year",
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        let func = self.func;

        match arg {
            ColumnarValue::Scalar(ScalarValue::UInt64(Some(ts_ns))) => Ok(ColumnarValue::Scalar(
                ScalarValue::Float64(Some(func.evaluate_ns(ts_ns))),
            )),
            ColumnarValue::Array(array) => {
                let ts_array = array.as_primitive::<UInt64Type>();
                let mut builder = Float64Builder::with_capacity(ts_array.len());
                for i in 0..ts_array.len() {
                    if ts_array.is_null(i) {
                        builder.append_null();
                    } else {
                        builder.append_value(func.evaluate_ns(ts_array.value(i)));
                    }
                }
                Ok(ColumnarValue::Array(Arc::new(builder.finish())))
            }
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Float64(None))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2021-01-15 10:30:45 UTC in nanoseconds
    // Unix timestamp: 1610706645
    const TEST_TS_NS: u64 = 1_610_706_645_000_000_000;

    #[test]
    fn test_timestamp() {
        let result = DateTimeFunction::Timestamp.evaluate_ns(TEST_TS_NS);
        assert!(
            (result - 1_610_706_645.0).abs() < 0.001,
            "expected ~1610706645.0, got {result}"
        );
    }

    #[test]
    fn test_day_of_month() {
        // 2021-01-15 → day 15
        let result = DateTimeFunction::DayOfMonth.evaluate_ns(TEST_TS_NS);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_day_of_week() {
        // 2021-01-15 is a Friday → 5 in Prometheus (0=Sunday)
        let result = DateTimeFunction::DayOfWeek.evaluate_ns(TEST_TS_NS);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_day_of_year() {
        // 2021-01-15 → ordinal day 15
        let result = DateTimeFunction::DayOfYear.evaluate_ns(TEST_TS_NS);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_days_in_month() {
        // January has 31 days
        let result = DateTimeFunction::DaysInMonth.evaluate_ns(TEST_TS_NS);
        assert_eq!(result, 31.0);
    }

    #[test]
    fn test_days_in_month_february_leap() {
        // 2020-02-15 00:00:00 UTC (leap year)
        let ts = chrono::TimeZone::timestamp_opt(&chrono::Utc, 1581724800, 0)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap() as u64;
        let result = DateTimeFunction::DaysInMonth.evaluate_ns(ts);
        assert_eq!(result, 29.0);
    }

    #[test]
    fn test_days_in_month_february_non_leap() {
        // 2021-02-15 00:00:00 UTC (non-leap year)
        let ts = chrono::TimeZone::timestamp_opt(&chrono::Utc, 1613347200, 0)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap() as u64;
        let result = DateTimeFunction::DaysInMonth.evaluate_ns(ts);
        assert_eq!(result, 28.0);
    }

    #[test]
    fn test_hour() {
        // 10:30:45 → hour 10
        let result = DateTimeFunction::Hour.evaluate_ns(TEST_TS_NS);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_minute() {
        // 10:30:45 → minute 30
        let result = DateTimeFunction::Minute.evaluate_ns(TEST_TS_NS);
        assert_eq!(result, 30.0);
    }

    #[test]
    fn test_month() {
        // January → 1
        let result = DateTimeFunction::Month.evaluate_ns(TEST_TS_NS);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_year() {
        let result = DateTimeFunction::Year.evaluate_ns(TEST_TS_NS);
        assert_eq!(result, 2021.0);
    }

    #[test]
    fn test_lookup_datetime_function() {
        assert_eq!(
            lookup_datetime_function("timestamp"),
            Some(DateTimeFunction::Timestamp)
        );
        assert_eq!(
            lookup_datetime_function("day_of_month"),
            Some(DateTimeFunction::DayOfMonth)
        );
        assert_eq!(
            lookup_datetime_function("day_of_week"),
            Some(DateTimeFunction::DayOfWeek)
        );
        assert_eq!(
            lookup_datetime_function("day_of_year"),
            Some(DateTimeFunction::DayOfYear)
        );
        assert_eq!(
            lookup_datetime_function("days_in_month"),
            Some(DateTimeFunction::DaysInMonth)
        );
        assert_eq!(
            lookup_datetime_function("hour"),
            Some(DateTimeFunction::Hour)
        );
        assert_eq!(
            lookup_datetime_function("minute"),
            Some(DateTimeFunction::Minute)
        );
        assert_eq!(
            lookup_datetime_function("month"),
            Some(DateTimeFunction::Month)
        );
        assert_eq!(
            lookup_datetime_function("year"),
            Some(DateTimeFunction::Year)
        );
        assert_eq!(lookup_datetime_function("unknown"), None);
    }

    #[test]
    fn test_is_time_function() {
        assert!(is_time_function("time"));
        assert!(!is_time_function("timestamp"));
        assert!(!is_time_function("hour"));
    }

    #[test]
    fn test_display() {
        assert_eq!(DateTimeFunction::Timestamp.to_string(), "timestamp");
        assert_eq!(DateTimeFunction::DayOfMonth.to_string(), "day_of_month");
        assert_eq!(DateTimeFunction::Hour.to_string(), "hour");
        assert_eq!(DateTimeFunction::Year.to_string(), "year");
    }
}
