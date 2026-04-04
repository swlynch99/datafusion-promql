use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field, Float64Type, UInt64Type};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::function::AccumulatorArgs;
use datafusion::logical_expr::function::StateFieldsArgs;
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, TypeSignature, Volatility,
};

use super::range::RangeFunction;

/// Create a DataFusion UDAF for the given range function.
///
/// The UDAF takes two arguments: `timestamp` (UInt64) and `value` (Float64),
/// accumulates `(timestamp, value)` pairs, and applies the range function
/// (rate, irate, increase, delta) on evaluate.
pub(crate) fn make_range_udaf(func: RangeFunction) -> Arc<AggregateUDF> {
    Arc::new(AggregateUDF::new_from_impl(RangeAggregateUdf::new(func)))
}

#[derive(Debug)]
struct RangeAggregateUdf {
    func: RangeFunction,
    name: String,
    signature: Signature,
}

impl RangeAggregateUdf {
    fn new(func: RangeFunction) -> Self {
        let name = format!("promql_{func}");
        let signature = Signature::new(
            TypeSignature::Exact(vec![DataType::UInt64, DataType::Float64]),
            Volatility::Immutable,
        );
        Self {
            func,
            name,
            signature,
        }
    }
}

impl PartialEq for RangeAggregateUdf {
    fn eq(&self, other: &Self) -> bool {
        self.func == other.func
    }
}
impl Eq for RangeAggregateUdf {}

impl Hash for RangeAggregateUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.func.hash(state);
    }
}

impl AggregateUDFImpl for RangeAggregateUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn accumulator(&self, _args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(RangeAccumulator::new(self.func)))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        Ok(vec![
            Arc::new(Field::new(
                "timestamps",
                DataType::List(Arc::new(Field::new_list_field(DataType::UInt64, true))),
                false,
            )),
            Arc::new(Field::new(
                "values",
                DataType::List(Arc::new(Field::new_list_field(DataType::Float64, true))),
                false,
            )),
        ])
    }
}

/// Accumulator that collects `(timestamp, value)` pairs and applies a
/// [`RangeFunction`] on evaluate.
#[derive(Debug)]
struct RangeAccumulator {
    func: RangeFunction,
    samples: Vec<(u64, f64)>,
}

impl RangeAccumulator {
    fn new(func: RangeFunction) -> Self {
        Self {
            func,
            samples: Vec::new(),
        }
    }
}

impl Accumulator for RangeAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let timestamps = values[0].as_primitive::<UInt64Type>();
        let vals = values[1].as_primitive::<Float64Type>();
        for i in 0..timestamps.len() {
            if !timestamps.is_null(i) && !vals.is_null(i) {
                self.samples.push((timestamps.value(i), vals.value(i)));
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        self.samples.sort_by_key(|(ts, _)| *ts);
        match self.func.evaluate(&self.samples) {
            Some(v) => Ok(ScalarValue::Float64(Some(v))),
            None => Ok(ScalarValue::Float64(None)),
        }
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.samples.capacity() * std::mem::size_of::<(u64, f64)>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let timestamps: Vec<ScalarValue> = self
            .samples
            .iter()
            .map(|(ts, _)| ScalarValue::UInt64(Some(*ts)))
            .collect();
        let values: Vec<ScalarValue> = self
            .samples
            .iter()
            .map(|(_, v)| ScalarValue::Float64(Some(*v)))
            .collect();

        Ok(vec![
            ScalarValue::List(ScalarValue::new_list(&timestamps, &DataType::UInt64, true)),
            ScalarValue::List(ScalarValue::new_list(&values, &DataType::Float64, true)),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let ts_list = states[0].as_list::<i32>();
        let val_list = states[1].as_list::<i32>();

        for i in 0..ts_list.len() {
            if ts_list.is_null(i) || val_list.is_null(i) {
                continue;
            }
            let ts_arr = ts_list.value(i);
            let val_arr = val_list.value(i);
            let ts_prim = ts_arr.as_primitive::<UInt64Type>();
            let val_prim = val_arr.as_primitive::<Float64Type>();
            for j in 0..ts_prim.len() {
                if !ts_prim.is_null(j) && !val_prim.is_null(j) {
                    self.samples.push((ts_prim.value(j), val_prim.value(j)));
                }
            }
        }
        Ok(())
    }
}
