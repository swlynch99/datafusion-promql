//! Arrow-level representation of native histograms.
//!
//! See `plans/histograms.md` for the design rationale. The runtime shape is a
//! single `Struct<indices: List<UInt64>, counts: List<UInt64>>` column. The
//! `(grouping_power, max_value_power)` pair (the `histogram` crate's
//! [`Config`]) is carried as Arrow `Field` metadata under the keys
//! `grouping_power` and `max_value_power`, matching what
//! `metriken-exposition` writes to parquet.
//!
//! Sparse storage: a `HistogramArray` only holds entries for buckets with
//! non-zero counts. Both constructors silently filter zero-count entries on
//! the way in, so callers can pass dense input without needing to compact it
//! first.
//!
//! No DataFusion logical-plan integration lives here; this module is pure
//! Arrow plumbing.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ListArray, StructArray, UInt64Array};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Fields};
use arrow::error::ArrowError;

/// Name of the inner `indices` field of a histogram struct column.
pub const HISTOGRAM_INDICES_FIELD: &str = "indices";
/// Name of the inner `counts` field of a histogram struct column.
pub const HISTOGRAM_COUNTS_FIELD: &str = "counts";
/// Field-metadata key carrying the `grouping_power` of a histogram column.
pub const HISTOGRAM_GROUPING_POWER_KEY: &str = "grouping_power";
/// Field-metadata key carrying the `max_value_power` of a histogram column.
pub const HISTOGRAM_MAX_VALUE_POWER_KEY: &str = "max_value_power";

/// Bucket-layout configuration for a histogram column.
///
/// Mirrors the `histogram` crate's `Config` — fixed per metric, not per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistogramConfig {
    pub grouping_power: u8,
    pub max_value_power: u8,
}

impl HistogramConfig {
    pub const fn new(grouping_power: u8, max_value_power: u8) -> Self {
        Self {
            grouping_power,
            max_value_power,
        }
    }

    /// Render this config as the metadata map that gets attached to a `Field`.
    pub fn to_metadata(&self) -> HashMap<String, String> {
        let mut m = HashMap::with_capacity(2);
        m.insert(
            HISTOGRAM_GROUPING_POWER_KEY.to_string(),
            self.grouping_power.to_string(),
        );
        m.insert(
            HISTOGRAM_MAX_VALUE_POWER_KEY.to_string(),
            self.max_value_power.to_string(),
        );
        m
    }

    /// Parse a `HistogramConfig` from a metadata map. Returns `None` if either
    /// key is missing or fails to parse as `u8`.
    pub fn from_metadata(meta: &HashMap<String, String>) -> Option<Self> {
        let grouping_power = meta.get(HISTOGRAM_GROUPING_POWER_KEY)?.parse().ok()?;
        let max_value_power = meta.get(HISTOGRAM_MAX_VALUE_POWER_KEY)?.parse().ok()?;
        Some(Self {
            grouping_power,
            max_value_power,
        })
    }

    /// Read a `HistogramConfig` off a `Field`'s metadata.
    pub fn from_field(field: &Field) -> Option<Self> {
        Self::from_metadata(field.metadata())
    }

    /// Return a copy of `field` with this config merged into its metadata
    /// (existing keys are preserved unless they overlap).
    pub fn write_to_field(&self, field: Field) -> Field {
        let mut metadata = field.metadata().clone();
        metadata.insert(
            HISTOGRAM_GROUPING_POWER_KEY.to_string(),
            self.grouping_power.to_string(),
        );
        metadata.insert(
            HISTOGRAM_MAX_VALUE_POWER_KEY.to_string(),
            self.max_value_power.to_string(),
        );
        field.with_metadata(metadata)
    }
}

fn list_uint64_field(name: &str, config: &HistogramConfig) -> Field {
    let item = Arc::new(Field::new("item", DataType::UInt64, false));
    Field::new(name, DataType::List(item), false).with_metadata(config.to_metadata())
}

/// Canonical Arrow `DataType` for a histogram column carrying `config`.
///
/// The returned type is `Struct<indices: List<UInt64>, counts: List<UInt64>>`.
/// The two inner fields each carry the `grouping_power` / `max_value_power`
/// metadata so that the config travels with the data type itself.
pub fn histogram_data_type(config: &HistogramConfig) -> DataType {
    DataType::Struct(Fields::from(vec![
        Arc::new(list_uint64_field(HISTOGRAM_INDICES_FIELD, config)),
        Arc::new(list_uint64_field(HISTOGRAM_COUNTS_FIELD, config)),
    ]))
}

/// Newtype wrapper around a `StructArray` with the histogram column shape.
///
/// Construct with [`HistogramArray::try_new`] from an `(indices, counts)` pair
/// of `ListArray<UInt64>`, or with [`HistogramArray::from_pairs`] from a slice
/// of `(Vec<u32>, Vec<u64>)` rows.
#[derive(Debug, Clone)]
pub struct HistogramArray {
    inner: StructArray,
    indices: ListArray,
    counts: ListArray,
    indices_values: UInt64Array,
    counts_values: UInt64Array,
}

impl HistogramArray {
    /// Build a `HistogramArray` from a parallel `(indices, counts)` pair.
    ///
    /// Both lists must have the same outer length and the same per-row inner
    /// length, and the outer null masks must agree. The inner element type
    /// must be `UInt64`. Entries with `count == 0` are silently filtered out
    /// to preserve the sparse invariant.
    pub fn try_new(indices: ListArray, counts: ListArray) -> Result<Self, ArrowError> {
        if indices.len() != counts.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "HistogramArray: indices length ({}) != counts length ({})",
                indices.len(),
                counts.len()
            )));
        }

        check_inner_uint64(HISTOGRAM_INDICES_FIELD, &indices)?;
        check_inner_uint64(HISTOGRAM_COUNTS_FIELD, &counts)?;

        for row in 0..indices.len() {
            if indices.is_null(row) != counts.is_null(row) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "HistogramArray: null mismatch at row {row}"
                )));
            }
            if !indices.is_null(row) && indices.value_length(row) != counts.value_length(row) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "HistogramArray: row {row} has indices.len={} but counts.len={}",
                    indices.value_length(row),
                    counts.value_length(row),
                )));
            }
        }

        let raw_indices = indices
            .values()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("indices inner type checked above");
        let raw_counts = counts
            .values()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("counts inner type checked above");

        let (indices, counts) = if raw_counts.values().contains(&0) {
            filter_zero_counts(&indices, raw_indices, &counts, raw_counts)?
        } else {
            (indices, counts)
        };

        let indices_values = indices
            .values()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("indices inner type preserved by filter")
            .clone();
        let counts_values = counts
            .values()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("counts inner type preserved by filter")
            .clone();

        let indices_field = Arc::new(Field::new(
            HISTOGRAM_INDICES_FIELD,
            indices.data_type().clone(),
            false,
        ));
        let counts_field = Arc::new(Field::new(
            HISTOGRAM_COUNTS_FIELD,
            counts.data_type().clone(),
            false,
        ));
        let fields = Fields::from(vec![indices_field, counts_field]);
        let arrays: Vec<ArrayRef> = vec![Arc::new(indices.clone()), Arc::new(counts.clone())];
        let inner = StructArray::try_new(fields, arrays, None)?;

        Ok(Self {
            inner,
            indices,
            counts,
            indices_values,
            counts_values,
        })
    }

    /// Build a `HistogramArray` from a slice of `(indices, counts)` rows.
    ///
    /// `u32` indices are widened to `u64` to match the canonical column shape.
    /// Entries with `count == 0` are silently filtered out, preserving the
    /// sparse invariant. All rows are non-null; for nullable construction use
    /// [`Self::try_new`].
    pub fn from_pairs(rows: &[(Vec<u32>, Vec<u64>)]) -> Result<Self, ArrowError> {
        let mut indices_values: Vec<u64> = Vec::new();
        let mut counts_values: Vec<u64> = Vec::new();
        let mut offsets: Vec<i32> = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);

        for (idx, cnt) in rows {
            if idx.len() != cnt.len() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "HistogramArray::from_pairs: indices.len={} != counts.len={}",
                    idx.len(),
                    cnt.len()
                )));
            }
            for (i, c) in idx.iter().zip(cnt.iter()) {
                if *c == 0 {
                    continue;
                }
                indices_values.push(u64::from(*i));
                counts_values.push(*c);
            }
            offsets.push(i32::try_from(indices_values.len()).map_err(|_| {
                ArrowError::InvalidArgumentError(
                    "HistogramArray::from_pairs: total values overflow i32 offsets".into(),
                )
            })?);
        }

        let item_field = Arc::new(Field::new("item", DataType::UInt64, false));
        let indices_offsets = OffsetBuffer::new(offsets.clone().into());
        let counts_offsets = OffsetBuffer::new(offsets.into());
        let indices = ListArray::try_new(
            item_field.clone(),
            indices_offsets,
            Arc::new(UInt64Array::from(indices_values)),
            None,
        )?;
        let counts = ListArray::try_new(
            item_field,
            counts_offsets,
            Arc::new(UInt64Array::from(counts_values)),
            None,
        )?;

        Self::try_new(indices, counts)
    }

    /// Number of rows in the array.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` if the array has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` if the row at `row` is null.
    pub fn is_null(&self, row: usize) -> bool {
        self.indices.is_null(row)
    }

    /// Sparse bucket indices for the histogram at `row`.
    ///
    /// Panics if `row` is out of bounds. Returns an empty slice for null rows
    /// (callers should consult [`Self::is_null`] when the distinction
    /// matters).
    pub fn indices_at(&self, row: usize) -> &[u64] {
        let offsets = self.indices.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        &self.indices_values.values()[start..end]
    }

    /// Sparse bucket counts for the histogram at `row`.
    ///
    /// Panics if `row` is out of bounds. Returns an empty slice for null rows.
    pub fn counts_at(&self, row: usize) -> &[u64] {
        let offsets = self.counts.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        &self.counts_values.values()[start..end]
    }

    /// Borrow the underlying `StructArray`.
    pub fn as_struct(&self) -> &StructArray {
        &self.inner
    }

    /// Consume this wrapper and return the underlying `StructArray`.
    pub fn into_struct(self) -> StructArray {
        self.inner
    }
}

fn check_inner_uint64(name: &str, list: &ListArray) -> Result<(), ArrowError> {
    match list.data_type() {
        DataType::List(field) if field.data_type() == &DataType::UInt64 => Ok(()),
        other => Err(ArrowError::InvalidArgumentError(format!(
            "HistogramArray: {name} must be List<UInt64>, got {other}"
        ))),
    }
}

fn filter_zero_counts(
    indices: &ListArray,
    indices_values: &UInt64Array,
    counts: &ListArray,
    counts_values: &UInt64Array,
) -> Result<(ListArray, ListArray), ArrowError> {
    let raw_indices = indices_values.values();
    let raw_counts = counts_values.values();
    let in_offsets = indices.value_offsets();

    let mut new_indices: Vec<u64> = Vec::with_capacity(raw_counts.len());
    let mut new_counts: Vec<u64> = Vec::with_capacity(raw_counts.len());
    let mut new_offsets: Vec<i32> = Vec::with_capacity(indices.len() + 1);
    new_offsets.push(0);

    for row in 0..indices.len() {
        if !indices.is_null(row) {
            let start = in_offsets[row] as usize;
            let end = in_offsets[row + 1] as usize;
            for k in start..end {
                if raw_counts[k] == 0 {
                    continue;
                }
                new_indices.push(raw_indices[k]);
                new_counts.push(raw_counts[k]);
            }
        }
        new_offsets.push(i32::try_from(new_indices.len()).map_err(|_| {
            ArrowError::InvalidArgumentError(
                "HistogramArray: filtered values overflow i32 offsets".into(),
            )
        })?);
    }

    let item_field = Arc::new(Field::new("item", DataType::UInt64, false));
    let indices_offsets = OffsetBuffer::new(new_offsets.clone().into());
    let counts_offsets = OffsetBuffer::new(new_offsets.into());
    let new_indices_list = ListArray::try_new(
        item_field.clone(),
        indices_offsets,
        Arc::new(UInt64Array::from(new_indices)),
        indices.nulls().cloned(),
    )?;
    let new_counts_list = ListArray::try_new(
        item_field,
        counts_offsets,
        Arc::new(UInt64Array::from(new_counts)),
        counts.nulls().cloned(),
    )?;
    Ok((new_indices_list, new_counts_list))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_metadata_round_trip() {
        let config = HistogramConfig::new(7, 32);
        let meta = config.to_metadata();
        assert_eq!(meta.get(HISTOGRAM_GROUPING_POWER_KEY).unwrap(), "7");
        assert_eq!(meta.get(HISTOGRAM_MAX_VALUE_POWER_KEY).unwrap(), "32");
        assert_eq!(HistogramConfig::from_metadata(&meta), Some(config));
    }

    #[test]
    fn config_field_round_trip_preserves_other_metadata() {
        let mut existing = HashMap::new();
        existing.insert("metric".to_string(), "latency".to_string());
        let field = Field::new("value", DataType::UInt64, false).with_metadata(existing);

        let config = HistogramConfig::new(3, 63);
        let with_config = config.write_to_field(field);

        assert_eq!(HistogramConfig::from_field(&with_config), Some(config));
        assert_eq!(with_config.metadata().get("metric").unwrap(), "latency");
    }

    #[test]
    fn config_from_metadata_missing_keys_is_none() {
        let mut meta = HashMap::new();
        meta.insert(HISTOGRAM_GROUPING_POWER_KEY.to_string(), "3".to_string());
        assert!(HistogramConfig::from_metadata(&meta).is_none());

        let mut meta = HashMap::new();
        meta.insert("not_a_key".to_string(), "3".to_string());
        assert!(HistogramConfig::from_metadata(&meta).is_none());
    }

    #[test]
    fn data_type_builder_attaches_metadata() {
        let config = HistogramConfig::new(3, 63);
        let dt = histogram_data_type(&config);

        let DataType::Struct(fields) = dt else {
            panic!("expected struct, got {dt:?}");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), HISTOGRAM_INDICES_FIELD);
        assert_eq!(fields[1].name(), HISTOGRAM_COUNTS_FIELD);
        for f in fields.iter() {
            assert_eq!(
                f.data_type(),
                &DataType::List(Arc::new(Field::new("item", DataType::UInt64, false)))
            );
            assert_eq!(HistogramConfig::from_field(f), Some(config));
        }
    }

    #[test]
    fn from_pairs_empty() {
        let h = HistogramArray::from_pairs(&[]).unwrap();
        assert_eq!(h.len(), 0);
        assert!(h.is_empty());
    }

    #[test]
    fn from_pairs_single_row() {
        let row: (Vec<u32>, Vec<u64>) = (vec![0, 4, 17], vec![1, 5, 2]);
        let h = HistogramArray::from_pairs(std::slice::from_ref(&row)).unwrap();
        assert_eq!(h.len(), 1);
        assert!(!h.is_null(0));
        assert_eq!(h.indices_at(0), &[0u64, 4, 17]);
        assert_eq!(h.counts_at(0), &[1u64, 5, 2]);
    }

    #[test]
    fn from_pairs_multi_row_sparse() {
        let rows: Vec<(Vec<u32>, Vec<u64>)> = vec![
            (vec![0, 1, 2], vec![10, 20, 30]),
            (vec![], vec![]),
            (vec![100, 101], vec![1, 1]),
            (vec![5], vec![42]),
        ];
        let h = HistogramArray::from_pairs(&rows).unwrap();
        assert_eq!(h.len(), 4);
        assert_eq!(h.indices_at(0), &[0u64, 1, 2]);
        assert_eq!(h.counts_at(0), &[10u64, 20, 30]);
        assert_eq!(h.indices_at(1), &[] as &[u64]);
        assert_eq!(h.counts_at(1), &[] as &[u64]);
        assert_eq!(h.indices_at(2), &[100u64, 101]);
        assert_eq!(h.counts_at(2), &[1u64, 1]);
        assert_eq!(h.indices_at(3), &[5u64]);
        assert_eq!(h.counts_at(3), &[42u64]);
    }

    #[test]
    fn try_new_round_trips_through_struct() {
        let rows: Vec<(Vec<u32>, Vec<u64>)> = vec![(vec![1, 2], vec![7, 8]), (vec![9], vec![1])];
        let h = HistogramArray::from_pairs(&rows).unwrap();
        let s = h.clone().into_struct();

        let indices = s
            .column_by_name(HISTOGRAM_INDICES_FIELD)
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .clone();
        let counts = s
            .column_by_name(HISTOGRAM_COUNTS_FIELD)
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .clone();
        let h2 = HistogramArray::try_new(indices, counts).unwrap();

        assert_eq!(h2.len(), h.len());
        for row in 0..h.len() {
            assert_eq!(h2.indices_at(row), h.indices_at(row));
            assert_eq!(h2.counts_at(row), h.counts_at(row));
        }
    }

    #[test]
    fn try_new_rejects_length_mismatch() {
        let a = HistogramArray::from_pairs(&[(vec![1], vec![1])])
            .unwrap()
            .into_struct();
        let b = HistogramArray::from_pairs(&[(vec![1], vec![1]), (vec![2], vec![2])])
            .unwrap()
            .into_struct();

        let a_indices = a
            .column_by_name(HISTOGRAM_INDICES_FIELD)
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .clone();
        let b_counts = b
            .column_by_name(HISTOGRAM_COUNTS_FIELD)
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .clone();
        assert!(HistogramArray::try_new(a_indices, b_counts).is_err());
    }

    #[test]
    fn from_pairs_rejects_per_row_length_mismatch() {
        let rows: Vec<(Vec<u32>, Vec<u64>)> = vec![(vec![1, 2], vec![3])];
        assert!(HistogramArray::from_pairs(&rows).is_err());
    }

    #[test]
    fn from_pairs_strips_zero_count_entries() {
        let rows: Vec<(Vec<u32>, Vec<u64>)> = vec![
            (vec![0, 1, 2, 3], vec![10, 0, 30, 0]),
            (vec![5, 6], vec![0, 0]),
            (vec![7], vec![1]),
        ];
        let h = HistogramArray::from_pairs(&rows).unwrap();
        assert_eq!(h.len(), 3);
        assert_eq!(h.indices_at(0), &[0u64, 2]);
        assert_eq!(h.counts_at(0), &[10u64, 30]);
        assert_eq!(h.indices_at(1), &[] as &[u64]);
        assert_eq!(h.counts_at(1), &[] as &[u64]);
        assert_eq!(h.indices_at(2), &[7u64]);
        assert_eq!(h.counts_at(2), &[1u64]);
    }

    #[test]
    fn try_new_filters_zero_counts() {
        let item = Arc::new(Field::new("item", DataType::UInt64, false));
        let offsets = OffsetBuffer::new(vec![0i32, 3, 3, 5].into());
        let indices = ListArray::try_new(
            item.clone(),
            offsets.clone(),
            Arc::new(UInt64Array::from(vec![1u64, 2, 3, 7, 8])),
            None,
        )
        .unwrap();
        let counts = ListArray::try_new(
            item,
            offsets,
            Arc::new(UInt64Array::from(vec![5u64, 0, 9, 0, 11])),
            None,
        )
        .unwrap();

        let h = HistogramArray::try_new(indices, counts).unwrap();
        assert_eq!(h.len(), 3);
        assert_eq!(h.indices_at(0), &[1u64, 3]);
        assert_eq!(h.counts_at(0), &[5u64, 9]);
        assert_eq!(h.indices_at(1), &[] as &[u64]);
        assert_eq!(h.counts_at(1), &[] as &[u64]);
        assert_eq!(h.indices_at(2), &[8u64]);
        assert_eq!(h.counts_at(2), &[11u64]);

        for v in h.counts_at(0) {
            assert_ne!(*v, 0);
        }
    }
}
