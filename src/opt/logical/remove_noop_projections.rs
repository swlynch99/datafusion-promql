use datafusion::common::tree_node::Transformed;
use datafusion::error::Result;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

/// Optimizer rule that removes projections where every expression is a no-op
/// identity alias (i.e. `col("X") AS "X"`).
///
/// A projection is considered a no-op if every expression simply passes through
/// a column from the input with the same name, producing the same schema in the
/// same order. In that case the projection node is redundant and can be removed.
#[derive(Debug)]
pub struct RemoveNoopProjections;

impl OptimizerRule for RemoveNoopProjections {
    fn name(&self) -> &str {
        "remove_noop_projections"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Projection(ref proj) = plan else {
            return Ok(Transformed::no(plan));
        };

        let input_schema = proj.input.schema();

        // The projection must have the same number of columns as its input.
        if proj.expr.len() != input_schema.fields().len() {
            return Ok(Transformed::no(plan));
        }

        // Every expression must be a passthrough of the corresponding input
        // column (same position, same name).
        for (i, expr) in proj.expr.iter().enumerate() {
            let (input_qualifier, input_field) = input_schema.qualified_field(i);

            if !is_noop_expr(expr, input_qualifier, input_field.name()) {
                return Ok(Transformed::no(plan));
            }
        }

        // All expressions are identity — remove the projection.
        let LogicalPlan::Projection(proj) = plan else {
            unreachable!();
        };
        Ok(Transformed::yes(std::sync::Arc::unwrap_or_clone(
            proj.input,
        )))
    }
}

/// Returns `true` if `expr` is a no-op identity reference to the given column,
/// meaning it produces the same name without transforming the value.
///
/// Matches:
/// - `col("name")` where name matches
/// - `col("name") AS "name"` where the alias equals the column name
/// - `qualifier.col("name")` with matching qualifier
/// - `qualifier.col("name") AS "name"` with matching qualifier and alias
fn is_noop_expr(
    expr: &Expr,
    expected_qualifier: Option<&datafusion::common::TableReference>,
    expected_name: &str,
) -> bool {
    match expr {
        Expr::Column(col) => {
            col.name == expected_name && qualifier_matches(&col.relation, expected_qualifier)
        }
        Expr::Alias(alias) => {
            // The alias must produce the same name as the input column.
            if alias.name != expected_name {
                return false;
            }
            // The inner expression must be a column reference to the same input.
            match alias.expr.as_ref() {
                Expr::Column(col) => {
                    col.name == expected_name
                        && qualifier_matches(&col.relation, expected_qualifier)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// Check if a column's qualifier matches the expected qualifier from the schema.
fn qualifier_matches(
    col_qualifier: &Option<datafusion::common::TableReference>,
    schema_qualifier: Option<&datafusion::common::TableReference>,
) -> bool {
    match (col_qualifier, schema_qualifier) {
        (None, None) => true,
        (None, Some(_)) => true, // Unqualified column ref matches any qualifier
        (Some(a), Some(b)) => a == b,
        (Some(_), None) => false,
    }
}
