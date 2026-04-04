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

pub(super) fn expr(input: Expr, max: f64) -> Expr {
    let udf = Arc::new(ScalarUDF::new_from_impl(PromqlClampMaxUdf::new(max)));
    udf.call(vec![input])
}

#[derive(Debug)]
struct PromqlClampMaxUdf {
    max: f64,
    signature: Signature,
}

impl PromqlClampMaxUdf {
    fn new(max: f64) -> Self {
        Self {
            max,
            signature: Signature::uniform(1, vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl PartialEq for PromqlClampMaxUdf {
    fn eq(&self, other: &Self) -> bool {
        self.max.to_bits() == other.max.to_bits()
    }
}
impl Eq for PromqlClampMaxUdf {}

impl Hash for PromqlClampMaxUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.max.to_bits().hash(state);
    }
}

impl ScalarUDFImpl for PromqlClampMaxUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "promql_clamp_max"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        let max = self.max;

        match arg {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) => Ok(ColumnarValue::Scalar(
                ScalarValue::Float64(Some(v.min(max))),
            )),
            ColumnarValue::Array(array) => {
                let result = array
                    .as_primitive::<Float64Type>()
                    .unary::<_, Float64Type>(|x| x.min(max));
                Ok(ColumnarValue::Array(Arc::new(result)))
            }
            _ => Ok(arg),
        }
    }
}
