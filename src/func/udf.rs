use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{DataType, Float64Type};
use datafusion::common::utils::take_function_args;
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::functions::math::expr_fn;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use super::instant::InstantFunction;

/// Convert an `InstantFunction` to a DataFusion logical `Expr` applied to the
/// given input expression, aliased as `"value"`.
pub(crate) fn instant_func_to_expr(func: &InstantFunction, input: Expr) -> Expr {
    let applied = match func {
        InstantFunction::Abs => expr_fn::abs(input),
        InstantFunction::Acos => expr_fn::acos(input),
        InstantFunction::Acosh => expr_fn::acosh(input),
        InstantFunction::Asin => expr_fn::asin(input),
        InstantFunction::Asinh => expr_fn::asinh(input),
        InstantFunction::Atan => expr_fn::atan(input),
        InstantFunction::Atanh => expr_fn::atanh(input),
        InstantFunction::Ceil => expr_fn::ceil(input),
        InstantFunction::Cos => expr_fn::cos(input),
        InstantFunction::Cosh => expr_fn::cosh(input),
        InstantFunction::Deg => expr_fn::degrees(input),
        InstantFunction::Exp => expr_fn::exp(input),
        InstantFunction::Floor => expr_fn::floor(input),
        InstantFunction::Ln => expr_fn::ln(input),
        InstantFunction::Log2 => expr_fn::log2(input),
        InstantFunction::Log10 => expr_fn::log10(input),
        InstantFunction::Rad => expr_fn::radians(input),
        InstantFunction::Sqrt => expr_fn::sqrt(input),
        InstantFunction::Sgn => expr_fn::signum(input),
        InstantFunction::Sin => expr_fn::sin(input),
        InstantFunction::Sinh => expr_fn::sinh(input),
        InstantFunction::Tan => expr_fn::tan(input),
        InstantFunction::Tanh => expr_fn::tanh(input),
        InstantFunction::Round { to_nearest } => {
            let udf = make_promql_round_udf(*to_nearest);
            udf.call(vec![input])
        }
    };
    applied.alias("value")
}

/// PromQL `round(v, to_nearest)` semantics.
///
/// Rounds `value` to the nearest multiple of `to_nearest`.
/// Uses the algorithm: `floor(value / to_nearest + 0.5) * to_nearest`.
/// If `to_nearest` is 0, the value is returned unchanged.
pub(super) fn promql_round(value: f64, to_nearest: f64) -> f64 {
    if to_nearest == 0.0 {
        return value;
    }
    (value / to_nearest + 0.5).floor() * to_nearest
}

/// Create a ScalarUDF implementing PromQL's `round(v, to_nearest)` semantics.
fn make_promql_round_udf(to_nearest: f64) -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(PromqlRoundUdf::new(to_nearest)))
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
