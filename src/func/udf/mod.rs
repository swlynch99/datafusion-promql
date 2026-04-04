mod abs;
mod acos;
mod acosh;
mod asin;
mod asinh;
mod atan;
mod atanh;
mod ceil;
mod cos;
mod cosh;
mod deg;
mod exp;
mod floor;
mod ln;
mod log2;
mod log10;
mod rad;
mod round;
mod sgn;
mod sin;
mod sinh;
mod sqrt;
mod tan;
mod tanh;

use datafusion::logical_expr::Expr;

use super::instant::InstantFunction;

/// Convert an `InstantFunction` to a DataFusion logical `Expr` applied to the
/// given input expression, aliased as `"value"`.
pub(crate) fn instant_func_to_expr(func: &InstantFunction, input: Expr) -> Expr {
    let applied = match func {
        InstantFunction::Abs => abs::expr(input),
        InstantFunction::Acos => acos::expr(input),
        InstantFunction::Acosh => acosh::expr(input),
        InstantFunction::Asin => asin::expr(input),
        InstantFunction::Asinh => asinh::expr(input),
        InstantFunction::Atan => atan::expr(input),
        InstantFunction::Atanh => atanh::expr(input),
        InstantFunction::Ceil => ceil::expr(input),
        InstantFunction::Cos => cos::expr(input),
        InstantFunction::Cosh => cosh::expr(input),
        InstantFunction::Deg => deg::expr(input),
        InstantFunction::Exp => exp::expr(input),
        InstantFunction::Floor => floor::expr(input),
        InstantFunction::Ln => ln::expr(input),
        InstantFunction::Log2 => log2::expr(input),
        InstantFunction::Log10 => log10::expr(input),
        InstantFunction::Rad => rad::expr(input),
        InstantFunction::Sqrt => sqrt::expr(input),
        InstantFunction::Sgn => sgn::expr(input),
        InstantFunction::Sin => sin::expr(input),
        InstantFunction::Sinh => sinh::expr(input),
        InstantFunction::Tan => tan::expr(input),
        InstantFunction::Tanh => tanh::expr(input),
        InstantFunction::Round { to_nearest } => round::expr(input, *to_nearest),
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
