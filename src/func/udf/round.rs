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

use super::promql_round;

pub(super) fn expr(input: Expr, to_nearest: f64) -> Expr {
    let udf = Arc::new(ScalarUDF::new_from_impl(PromqlRoundUdf::new(to_nearest)));
    udf.call(vec![input])
}

#[derive(Debug)]
struct PromqlRoundUdf {
    to_nearest: f64,
    signature: Signature,
}

impl PromqlRoundUdf {
    fn new(to_nearest: f64) -> Self {
        Self {
            to_nearest,
            signature: Signature::uniform(1, vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl PartialEq for PromqlRoundUdf {
    fn eq(&self, other: &Self) -> bool {
        self.to_nearest.to_bits() == other.to_nearest.to_bits()
    }
}
impl Eq for PromqlRoundUdf {}

impl Hash for PromqlRoundUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_nearest.to_bits().hash(state);
    }
}

impl ScalarUDFImpl for PromqlRoundUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "promql_round"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        let to_nearest = self.to_nearest;

        match arg {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(v))) => Ok(ColumnarValue::Scalar(
                ScalarValue::Float64(Some(promql_round(v, to_nearest))),
            )),
            ColumnarValue::Array(array) => {
                let result = array
                    .as_primitive::<Float64Type>()
                    .unary::<_, Float64Type>(|x| promql_round(x, to_nearest));
                Ok(ColumnarValue::Array(Arc::new(result)))
            }
            _ => Ok(arg),
        }
    }
}
