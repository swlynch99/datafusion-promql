use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{DataType, Float64Type};
use datafusion::common::utils::take_function_args;
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use super::promql_clamp;

pub(super) fn expr(input: Expr, min: f64, max: f64) -> Expr {
    let udf = Arc::new(ScalarUDF::new_from_impl(PromqlClampUdf::new(min, max)));
    udf.call(vec![input])
}

#[derive(Debug)]
struct PromqlClampUdf {
    min: f64,
    max: f64,
    signature: Signature,
}

impl PromqlClampUdf {
    fn new(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            signature: Signature::uniform(1, vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl PartialEq for PromqlClampUdf {
    fn eq(&self, other: &Self) -> bool {
        self.min.to_bits() == other.min.to_bits() && self.max.to_bits() == other.max.to_bits()
    }
}
impl Eq for PromqlClampUdf {}

impl Hash for PromqlClampUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.min.to_bits().hash(state);
        self.max.to_bits().hash(state);
    }
}

impl ScalarUDFImpl for PromqlClampUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "promql_clamp"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        let (min, max) = (self.min, self.max);

        match arg {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) => Ok(ColumnarValue::Scalar(
                ScalarValue::Float64(Some(promql_clamp(v, min, max))),
            )),
            ColumnarValue::Array(array) => {
                let result = array
                    .as_primitive::<Float64Type>()
                    .unary::<_, Float64Type>(|x| promql_clamp(x, min, max));
                Ok(ColumnarValue::Array(Arc::new(result)))
            }
            _ => Ok(arg),
        }
    }
}
