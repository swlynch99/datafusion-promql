use std::any::Any;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringArray, StringBuilder};
use arrow::datatypes::DataType;
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use regex::Regex;

/// Create a ScalarUDF implementing PromQL's `label_replace` semantics.
///
/// For each row, if `src_label` matches `regex`, set `dst_label` to `replacement`
/// with `$1`-style capture group references expanded. If there is no match, the
/// original `dst_label` value is preserved unchanged.
pub(crate) fn make_label_replace_udf(replacement: String, regex_pattern: String) -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(LabelReplaceUdf::new(
        replacement,
        regex_pattern,
    )))
}

#[derive(Debug)]
struct LabelReplaceUdf {
    replacement: String,
    regex: Regex,
    regex_pattern: String,
    signature: Signature,
}

impl LabelReplaceUdf {
    fn new(replacement: String, regex_pattern: String) -> Self {
        // PromQL anchors the regex to match the entire string.
        let anchored = format!("^(?:{regex_pattern})$");
        let regex = Regex::new(&anchored).expect("invalid regex in label_replace");
        Self {
            replacement,
            regex,
            regex_pattern,
            // Takes two args: (src_label_value, current_dst_label_value)
            signature: Signature::new(
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for LabelReplaceUdf {
    fn eq(&self, other: &Self) -> bool {
        self.replacement == other.replacement && self.regex_pattern == other.regex_pattern
    }
}
impl Eq for LabelReplaceUdf {}

impl Hash for LabelReplaceUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.replacement.hash(state);
        self.regex_pattern.hash(state);
    }
}

impl ScalarUDFImpl for LabelReplaceUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "promql_label_replace"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let src_value = &args.args[0];
        let dst_value = &args.args[1];

        match (src_value, dst_value) {
            (ColumnarValue::Array(src_arr), ColumnarValue::Array(dst_arr)) => {
                let src_strings = src_arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("src must be StringArray");
                let dst_strings = dst_arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("dst must be StringArray");
                let mut builder = StringBuilder::with_capacity(src_strings.len(), 256);
                for i in 0..src_strings.len() {
                    let src = src_strings.value(i);
                    let dst = dst_strings.value(i);
                    let result = self.apply_replace(src, dst);
                    builder.append_value(&result);
                }
                Ok(ColumnarValue::Array(Arc::new(builder.finish()) as ArrayRef))
            }
            (ColumnarValue::Scalar(ScalarValue::Utf8(src)), ColumnarValue::Scalar(ScalarValue::Utf8(dst))) => {
                let src = src.as_deref().unwrap_or("");
                let dst = dst.as_deref().unwrap_or("");
                let result = self.apply_replace(src, dst);
                Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(result))))
            }
            _ => {
                // Mixed scalar/array - expand scalar to array
                let len = match (src_value, dst_value) {
                    (ColumnarValue::Array(a), _) => a.len(),
                    (_, ColumnarValue::Array(a)) => a.len(),
                    _ => 1,
                };
                let src_arr = src_value.clone().into_array(len)?;
                let dst_arr = dst_value.clone().into_array(len)?;
                let src_strings = src_arr.as_any().downcast_ref::<StringArray>().unwrap();
                let dst_strings = dst_arr.as_any().downcast_ref::<StringArray>().unwrap();
                let mut builder = StringBuilder::with_capacity(len, 256);
                for i in 0..len {
                    let src = src_strings.value(i);
                    let dst = dst_strings.value(i);
                    let result = self.apply_replace(src, dst);
                    builder.append_value(&result);
                }
                Ok(ColumnarValue::Array(Arc::new(builder.finish()) as ArrayRef))
            }
        }
    }
}

impl LabelReplaceUdf {
    /// Apply the label_replace logic for a single row.
    ///
    /// If `src` matches the regex, expand capture groups in the replacement template
    /// and return the result. Otherwise, return `current_dst` unchanged.
    fn apply_replace(&self, src: &str, current_dst: &str) -> String {
        if let Some(caps) = self.regex.captures(src) {
            expand_captures(&self.replacement, &caps)
        } else {
            current_dst.to_string()
        }
    }
}

/// Expand `$1`, `$2`, etc. capture group references in `template` using `caps`.
fn expand_captures(template: &str, caps: &regex::Captures<'_>) -> String {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '$' {
            // Parse the group number
            let mut num_str = String::new();
            while let Some(&digit) = chars.peek() {
                if digit.is_ascii_digit() {
                    num_str.push(digit);
                    chars.next();
                } else {
                    break;
                }
            }
            if num_str.is_empty() {
                // Just a literal '$' with no number
                result.push('$');
            } else {
                let group_num: usize = num_str.parse().unwrap();
                if let Some(m) = caps.get(group_num) {
                    result.push_str(m.as_str());
                }
                // If group doesn't exist, nothing is added (Prometheus behavior)
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Create a ScalarUDF implementing PromQL's `label_join` semantics.
///
/// Concatenates source label values with the given separator.
pub(crate) fn make_label_join_udf(separator: String, num_src_labels: usize) -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(LabelJoinUdf::new(
        separator,
        num_src_labels,
    )))
}

#[derive(Debug)]
struct LabelJoinUdf {
    separator: String,
    num_src_labels: usize,
    signature: Signature,
}

impl LabelJoinUdf {
    fn new(separator: String, num_src_labels: usize) -> Self {
        let arg_types = vec![DataType::Utf8; num_src_labels];
        Self {
            separator,
            num_src_labels,
            signature: Signature::new(
                TypeSignature::Exact(arg_types),
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for LabelJoinUdf {
    fn eq(&self, other: &Self) -> bool {
        self.separator == other.separator && self.num_src_labels == other.num_src_labels
    }
}
impl Eq for LabelJoinUdf {}

impl Hash for LabelJoinUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.separator.hash(state);
        self.num_src_labels.hash(state);
    }
}

impl ScalarUDFImpl for LabelJoinUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "promql_label_join"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        // Determine if we're working with arrays or scalars
        let len = args
            .args
            .iter()
            .find_map(|a| {
                if let ColumnarValue::Array(arr) = a {
                    Some(arr.len())
                } else {
                    None
                }
            })
            .unwrap_or(1);

        let arrays: Vec<_> = args
            .args
            .iter()
            .map(|a| a.clone().into_array(len))
            .collect::<DFResult<_>>()?;

        let string_arrays: Vec<&StringArray> = arrays
            .iter()
            .map(|a| a.as_any().downcast_ref::<StringArray>().unwrap())
            .collect();

        let mut builder = StringBuilder::with_capacity(len, 256);
        for row in 0..len {
            let mut parts = Vec::with_capacity(self.num_src_labels);
            for arr in &string_arrays {
                parts.push(arr.value(row));
            }
            let joined = parts.join(&self.separator);
            builder.append_value(&joined);
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish()) as ArrayRef))
    }
}

impl fmt::Display for LabelReplaceUdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "label_replace(replacement={}, regex={})",
            self.replacement, self.regex_pattern
        )
    }
}

impl fmt::Display for LabelJoinUdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "label_join(separator={}, num_src={})",
            self.separator, self.num_src_labels
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_captures_simple() {
        let re = Regex::new("^(.+)$").unwrap();
        let caps = re.captures("hello").unwrap();
        assert_eq!(expand_captures("$1-world", &caps), "hello-world");
    }

    #[test]
    fn test_expand_captures_multiple_groups() {
        let re = Regex::new("^([^.]+)\\.(.+)$").unwrap();
        let caps = re.captures("host1.example.com").unwrap();
        assert_eq!(expand_captures("$1-$2", &caps), "host1-example.com");
    }

    #[test]
    fn test_expand_captures_no_dollar() {
        let re = Regex::new("^(.+)$").unwrap();
        let caps = re.captures("hello").unwrap();
        assert_eq!(expand_captures("static", &caps), "static");
    }

    #[test]
    fn test_expand_captures_missing_group() {
        let re = Regex::new("^(.+)$").unwrap();
        let caps = re.captures("hello").unwrap();
        // $2 doesn't exist, so it's omitted
        assert_eq!(expand_captures("$1-$2", &caps), "hello-");
    }

    #[test]
    fn test_label_replace_match() {
        let udf = LabelReplaceUdf::new("$1".to_string(), "([^:]+):.+".to_string());
        assert_eq!(udf.apply_replace("host:8080", "old"), "host");
    }

    #[test]
    fn test_label_replace_no_match() {
        let udf = LabelReplaceUdf::new("$1".to_string(), "([^:]+):.+".to_string());
        // No colon, so no match — keep original dst
        assert_eq!(udf.apply_replace("host", "old"), "old");
    }

    #[test]
    fn test_label_replace_full_anchor() {
        // The regex is anchored: "^(?:pattern)$"
        // So "partial" won't match "^(?:par)$"
        let udf = LabelReplaceUdf::new("replaced".to_string(), "par".to_string());
        assert_eq!(udf.apply_replace("partial", "old"), "old");
        assert_eq!(udf.apply_replace("par", "old"), "replaced");
    }
}
