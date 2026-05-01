//! Plan-time helpers for detecting histogram-typed value columns.
//!
//! The PromQL planner translates promql-parser AST nodes into DataFusion
//! `LogicalPlan`s. Most existing code paths assume the `value` column is
//! `Float64`; histogram metrics carry the canonical
//! `Struct<indices: List<UInt64>, counts: List<UInt64>>` shape from
//! [`super::histogram_data_type`] instead. The helpers here let those
//! branch points dispatch on the column kind without threading separate
//! metadata through the planner.
//!
//! The check is structural: a `Field` is a histogram column iff its
//! `DataType` has the canonical struct shape **and** carries the
//! `grouping_power` / `max_value_power` metadata keys required by
//! [`HistogramConfig`].

use arrow::datatypes::{DataType, Field};
use datafusion::common::DFSchemaRef;

use super::{HISTOGRAM_COUNTS_FIELD, HISTOGRAM_INDICES_FIELD, HistogramConfig};

/// Name of the value column produced by vector / matrix selectors.
pub const VALUE_COLUMN: &str = "value";

/// Returns `true` if `field` carries the canonical histogram column shape.
///
/// A histogram column is a non-null `Struct` with exactly two children named
/// `indices` and `counts`, each typed as `List<UInt64>`, and the field's
/// metadata must declare a valid [`HistogramConfig`]. Fields that match the
/// struct shape but are missing the config metadata are *not* recognized as
/// histograms — the config is part of the column's identity.
pub fn is_histogram_column(field: &Field) -> bool {
    let DataType::Struct(fields) = field.data_type() else {
        return false;
    };
    if fields.len() != 2 {
        return false;
    }

    let Some(indices) = fields.iter().find(|f| f.name() == HISTOGRAM_INDICES_FIELD) else {
        return false;
    };
    let Some(counts) = fields.iter().find(|f| f.name() == HISTOGRAM_COUNTS_FIELD) else {
        return false;
    };

    if !is_list_uint64(indices.data_type()) || !is_list_uint64(counts.data_type()) {
        return false;
    }

    HistogramConfig::from_field(field).is_some()
}

/// Returns the [`HistogramConfig`] attached to `field` iff it is a histogram
/// column. Returns `None` for scalar (`Float64`) value columns and for
/// struct-shaped fields that lack the histogram config metadata.
pub fn histogram_config(field: &Field) -> Option<HistogramConfig> {
    if !is_histogram_column(field) {
        return None;
    }
    HistogramConfig::from_field(field)
}

/// Returns `true` if the schema's `value` column is a histogram-typed column.
///
/// Convenience wrapper for plan-time branch points that hold a
/// [`DFSchemaRef`] rather than a bare `Field`. Returns `false` if there is
/// no `value` column at all (e.g. an upstream node has already projected it
/// away).
pub fn schema_value_is_histogram(schema: &DFSchemaRef) -> bool {
    value_field(schema)
        .map(is_histogram_column)
        .unwrap_or(false)
}

/// Borrow the `value` field out of a logical-plan schema, if present.
pub fn value_field(schema: &DFSchemaRef) -> Option<&Field> {
    schema
        .fields()
        .iter()
        .find(|f| f.name() == VALUE_COLUMN)
        .map(|f| f.as_ref())
}

fn is_list_uint64(dt: &DataType) -> bool {
    match dt {
        DataType::List(inner) => inner.data_type() == &DataType::UInt64,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::Schema;
    use datafusion::common::DFSchema;

    use super::super::histogram_data_type;
    use super::*;

    fn histogram_field(name: &str) -> Field {
        let config = HistogramConfig::new(7, 32);
        Field::new(name, histogram_data_type(&config), false).with_metadata(config.to_metadata())
    }

    #[test]
    fn float64_value_is_not_histogram() {
        let f = Field::new("value", DataType::Float64, false);
        assert!(!is_histogram_column(&f));
        assert_eq!(histogram_config(&f), None);
    }

    #[test]
    fn canonical_histogram_field_is_recognized() {
        let f = histogram_field("value");
        assert!(is_histogram_column(&f));
        assert_eq!(histogram_config(&f), Some(HistogramConfig::new(7, 32)));
    }

    #[test]
    fn histogram_struct_without_metadata_is_not_recognized() {
        let config = HistogramConfig::new(7, 32);
        // Build the struct shape but strip the field-level config metadata.
        let f = Field::new("value", histogram_data_type(&config), false);
        assert!(!is_histogram_column(&f));
        assert_eq!(histogram_config(&f), None);
    }

    #[test]
    fn schema_value_is_histogram_dispatches_on_value_column() {
        let hist = histogram_field("value");
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            hist,
            Field::new("instance", DataType::Utf8, false),
        ]);
        let df = Arc::new(DFSchema::try_from(schema).unwrap());
        assert!(schema_value_is_histogram(&df));

        let scalar = Schema::new(vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("value", DataType::Float64, false),
        ]);
        let df = Arc::new(DFSchema::try_from(scalar).unwrap());
        assert!(!schema_value_is_histogram(&df));
    }

    #[test]
    fn schema_without_value_column_is_not_histogram() {
        let schema = Schema::new(vec![Field::new("timestamp", DataType::UInt64, false)]);
        let df = Arc::new(DFSchema::try_from(schema).unwrap());
        assert!(!schema_value_is_histogram(&df));
    }
}
