use datafusion::functions::math::expr_fn;
use datafusion::logical_expr::Expr;

pub(super) fn expr(input: Expr) -> Expr {
    expr_fn::exp(input)
}
