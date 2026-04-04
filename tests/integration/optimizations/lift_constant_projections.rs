use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::Transformed;
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Union};
use datafusion::optimizer::OptimizerRule;
use datafusion::prelude::*;

use datafusion_promql::opt::logical::LiftConstantProjections;

/// Build a trivial single-row MemTable scan to use as the base of test projections.
fn make_scan(alias: &str) -> LogicalPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, false),
        Field::new("val", DataType::Float64, false),
    ]));
    let table = MemTable::try_new(schema, vec![vec![]]).expect("failed to create MemTable");
    LogicalPlanBuilder::scan(
        alias,
        datafusion::datasource::provider_as_source(Arc::new(table)),
        None,
    )
    .unwrap()
    .build()
    .unwrap()
}

/// Build a union from projection specs.
///
/// Each spec is a list of `(expr, alias)` pairs. All specs must have the same
/// number of columns to form a valid union.
fn make_union(branches: Vec<Vec<Expr>>) -> LogicalPlan {
    let inputs: Vec<Arc<LogicalPlan>> = branches
        .into_iter()
        .enumerate()
        .map(|(idx, exprs)| {
            let scan = make_scan(&format!("t{idx}"));
            let plan = LogicalPlanBuilder::from(scan)
                .project(exprs)
                .unwrap()
                .build()
                .unwrap();
            Arc::new(plan)
        })
        .collect();

    LogicalPlan::Union(Union::try_new_with_loose_types(inputs).unwrap())
}

fn apply_rule(plan: LogicalPlan) -> (LogicalPlan, bool) {
    let rule = LiftConstantProjections;
    let Transformed {
        data, transformed, ..
    } = rule.rewrite(plan, &NoopConfig).unwrap();
    (data, transformed)
}

/// Minimal OptimizerConfig that does nothing.
struct NoopConfig;

impl datafusion::optimizer::OptimizerConfig for NoopConfig {
    fn query_execution_start_time(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        None
    }

    fn alias_generator(&self) -> &Arc<AliasGenerator> {
        static GEN: std::sync::LazyLock<Arc<AliasGenerator>> =
            std::sync::LazyLock::new(|| Arc::new(AliasGenerator::default()));
        &GEN
    }

    fn options(&self) -> Arc<ConfigOptions> {
        Arc::new(ConfigOptions::default())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// When all branches share the same literal for a column, it should be lifted
/// into an outer projection above the union.
#[test]
fn test_shared_constant_is_lifted() {
    let plan = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ],
    ]);

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should have rewritten the plan");

    // Top-level should be a Projection.
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };

    // The outer projection should have 3 expressions (timestamp, value, __name__).
    assert_eq!(outer.expr.len(), 3);

    // The __name__ column (index 2) should be a literal in the outer projection.
    assert!(
        is_literal_alias(&outer.expr[2], "cpu", "__name__"),
        "expected lit('cpu') AS __name__ in outer projection, got: {:?}",
        outer.expr[2]
    );

    // Under the projection should be a Union.
    let LogicalPlan::Union(inner_union) = outer.input.as_ref() else {
        panic!("expected Union under outer projection");
    };

    // Inner branches should only have 2 columns (timestamp, value).
    for input in &inner_union.inputs {
        let LogicalPlan::Projection(inner_proj) = input.as_ref() else {
            panic!("expected Projection in union branch");
        };
        assert_eq!(
            inner_proj.expr.len(),
            2,
            "inner projection should have 2 columns after lifting __name__"
        );
    }
}

/// When branches have different literal values for a column, it should NOT be
/// lifted.
#[test]
fn test_different_constants_not_lifted() {
    let plan = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host1").alias("host"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host2").alias("host"),
        ],
    ]);

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should fire because __name__ is shared");

    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top");
    };

    // __name__ (index 2) should be lifted as a literal.
    assert!(
        is_literal_alias(&outer.expr[2], "cpu", "__name__"),
        "shared constant __name__ should be lifted"
    );

    // host (index 3) should NOT be a literal in the outer — it differs between branches.
    assert!(
        !is_any_literal(&outer.expr[3]),
        "differing constant 'host' should not be lifted, got: {:?}",
        outer.expr[3]
    );

    // Inner branches should still have 3 columns (timestamp, value, host).
    let LogicalPlan::Union(inner_union) = outer.input.as_ref() else {
        panic!("expected Union under outer projection");
    };
    for input in &inner_union.inputs {
        let LogicalPlan::Projection(inner_proj) = input.as_ref() else {
            panic!("expected Projection in union branch");
        };
        assert_eq!(
            inner_proj.expr.len(),
            3,
            "inner projections should keep timestamp, value, and host"
        );
    }
}

/// When no columns are shared constants, the plan should be unchanged.
#[test]
fn test_no_shared_constants_unchanged() {
    let plan = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("host1").alias("host"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("host2").alias("host"),
        ],
    ]);

    let (_result, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "rule should not fire when no constants are shared"
    );
}

/// When all columns are constant (no non-constant columns), bail out to avoid
/// creating an empty union.
#[test]
fn test_all_columns_constant_unchanged() {
    let plan = make_union(vec![
        vec![lit("cpu").alias("__name__"), lit("val").alias("label")],
        vec![lit("cpu").alias("__name__"), lit("val").alias("label")],
    ]);

    let (_result, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "rule should not fire when all columns are constant"
    );
}

/// Non-projection union branches should be left untouched.
#[test]
fn test_non_projection_branches_unchanged() {
    // Build a union of raw scans (not projections).
    let scan1 = make_scan("t0");
    let scan2 = make_scan("t1");
    let plan = LogicalPlan::Union(
        Union::try_new_with_loose_types(vec![Arc::new(scan1), Arc::new(scan2)]).unwrap(),
    );

    let (_result, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "rule should not fire on non-projection branches"
    );
}

/// Non-Union plans should pass through unchanged.
#[test]
fn test_non_union_plan_unchanged() {
    let scan = make_scan("t0");
    let plan = LogicalPlanBuilder::from(scan)
        .project(vec![col("ts").alias("timestamp"), lit("cpu").alias("name")])
        .unwrap()
        .build()
        .unwrap();

    let (_result, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should not fire on non-union plans");
}

/// A union with three branches where all share the same constant should work.
#[test]
fn test_three_branches_shared_constant() {
    let plan = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            lit("metric").alias("__name__"),
            lit("a").alias("id"),
        ],
        vec![
            col("ts").alias("timestamp"),
            lit("metric").alias("__name__"),
            lit("b").alias("id"),
        ],
        vec![
            col("ts").alias("timestamp"),
            lit("metric").alias("__name__"),
            lit("c").alias("id"),
        ],
    ]);

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected outer Projection");
    };
    assert_eq!(outer.expr.len(), 3);

    // __name__ (index 1) should be lifted.
    assert!(is_literal_alias(&outer.expr[1], "metric", "__name__"));

    // id (index 2) differs across branches, should not be lifted.
    assert!(!is_any_literal(&outer.expr[2]));

    let LogicalPlan::Union(inner_union) = outer.input.as_ref() else {
        panic!("expected Union");
    };
    assert_eq!(inner_union.inputs.len(), 3);
    for input in &inner_union.inputs {
        let LogicalPlan::Projection(p) = input.as_ref() else {
            panic!("expected Projection in branch");
        };
        // Should have timestamp + id = 2 columns.
        assert_eq!(p.expr.len(), 2);
    }
}

/// Multiple shared constants should all be lifted.
#[test]
fn test_multiple_shared_constants_lifted() {
    let plan = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("prod").alias("env"),
            lit("host1").alias("host"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("prod").alias("env"),
            lit("host2").alias("host"),
        ],
    ]);

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected outer Projection");
    };
    assert_eq!(outer.expr.len(), 5);

    // __name__ (index 2) and env (index 3) should be lifted.
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));
    assert!(is_literal_alias(&outer.expr[3], "prod", "env"));

    // host (index 4) differs, should not be lifted.
    assert!(!is_any_literal(&outer.expr[4]));

    // Inner branches should have 3 columns: timestamp, value, host.
    let LogicalPlan::Union(inner_union) = outer.input.as_ref() else {
        panic!("expected Union");
    };
    for input in &inner_union.inputs {
        let LogicalPlan::Projection(p) = input.as_ref() else {
            panic!("expected Projection");
        };
        assert_eq!(p.expr.len(), 3);
    }
}

/// The output schema of the rewritten plan should match the original.
#[test]
fn test_output_schema_preserved() {
    let plan = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host1").alias("host"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host2").alias("host"),
        ],
    ]);

    let original_field_names: Vec<String> = plan
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let result_field_names: Vec<String> = result
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    assert_eq!(
        original_field_names, result_field_names,
        "output schema field names should match the original"
    );
}

// ─── Sort lifting tests ─────────────────────────────────────────────────────

/// Helper: wrap a plan in a Sort node sorting by the given column.
fn wrap_in_sort(plan: LogicalPlan, sort_col: &str) -> LogicalPlan {
    LogicalPlanBuilder::from(plan)
        .sort(vec![col(sort_col).sort(true, false)])
        .unwrap()
        .build()
        .unwrap()
}

/// When a Sort wraps a Union and the sort does not reference the constant
/// columns, the constant projection should be lifted above the sort.
#[test]
fn test_lift_through_sort() {
    let union = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ],
    ]);
    let plan = wrap_in_sort(union, "value");

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should lift constants past sort");

    // Top-level should be Projection.
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert_eq!(outer.expr.len(), 3);
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));

    // Under the projection should be a Sort.
    let LogicalPlan::Sort(sort) = outer.input.as_ref() else {
        panic!("expected Sort under projection, got:\n{}", outer.input);
    };

    // Under the sort should be a Union.
    let LogicalPlan::Union(inner_union) = sort.input.as_ref() else {
        panic!("expected Union under sort");
    };

    // Inner branches should have 2 columns (timestamp, value).
    for input in &inner_union.inputs {
        let LogicalPlan::Projection(p) = input.as_ref() else {
            panic!("expected Projection in branch");
        };
        assert_eq!(p.expr.len(), 2);
    }
}

/// When the sort references a constant column, the rule should NOT lift.
#[test]
fn test_sort_referencing_constant_blocks_lift() {
    let union = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ],
    ]);
    // Sort by the constant column __name__.
    let plan = wrap_in_sort(union, "__name__");

    let (_result, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "rule should not lift when sort references a constant column"
    );
}

/// Sort over a Union where no constants are shared should be unchanged.
#[test]
fn test_sort_no_shared_constants_unchanged() {
    let union = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("host1").alias("host"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("host2").alias("host"),
        ],
    ]);
    let plan = wrap_in_sort(union, "value");

    let (_result, transformed) = apply_rule(plan);
    assert!(!transformed, "no shared constants to lift");
}

/// Output schema should be preserved when lifting through sort.
#[test]
fn test_sort_output_schema_preserved() {
    let union = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host1").alias("host"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host2").alias("host"),
        ],
    ]);
    let plan = wrap_in_sort(union, "value");

    let original_field_names: Vec<String> = plan
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let result_field_names: Vec<String> = result
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    assert_eq!(original_field_names, result_field_names);
}

/// Build a union where each branch has a Sort wrapping the Projection.
/// This tests lifting through wrappers inside union branches.
fn make_sorted_union(branches: Vec<Vec<Expr>>, sort_col: &str) -> LogicalPlan {
    let inputs: Vec<Arc<LogicalPlan>> = branches
        .into_iter()
        .enumerate()
        .map(|(idx, exprs)| {
            let scan = make_scan(&format!("t{idx}"));
            let proj = LogicalPlanBuilder::from(scan)
                .project(exprs)
                .unwrap()
                .build()
                .unwrap();
            let sorted = LogicalPlanBuilder::from(proj)
                .sort(vec![col(sort_col).sort(true, false)])
                .unwrap()
                .build()
                .unwrap();
            Arc::new(sorted)
        })
        .collect();

    LogicalPlan::Union(Union::try_new_with_loose_types(inputs).unwrap())
}

/// Lifting through wrappers inside union branches:
/// Union -> [Sort -> Proj, Sort -> Proj] should lift constants.
#[test]
fn test_lift_through_branch_wrappers() {
    let plan = make_sorted_union(
        vec![
            vec![
                col("ts").alias("timestamp"),
                col("val").alias("value"),
                lit("cpu").alias("__name__"),
            ],
            vec![
                col("ts").alias("timestamp"),
                col("val").alias("value"),
                lit("cpu").alias("__name__"),
            ],
        ],
        "value",
    );

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "should lift constants through branch sorts");

    // Top-level: Projection
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert_eq!(outer.expr.len(), 3);
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));

    // Under projection: Union
    let LogicalPlan::Union(inner_union) = outer.input.as_ref() else {
        panic!("expected Union under projection");
    };

    // Each branch should be Sort -> Projection(2 cols)
    for input in &inner_union.inputs {
        let LogicalPlan::Sort(_) = input.as_ref() else {
            panic!("expected Sort in branch, got:\n{input}");
        };
        let sort_input = input.inputs()[0];
        let LogicalPlan::Projection(p) = sort_input else {
            panic!("expected Projection under Sort in branch");
        };
        assert_eq!(p.expr.len(), 2);
    }
}

/// Combined: Sort -> Union -> [Sort -> Proj, Sort -> Proj]
/// Constants should be lifted all the way out.
#[test]
fn test_lift_through_outer_and_branch_wrappers() {
    let union = make_sorted_union(
        vec![
            vec![
                col("ts").alias("timestamp"),
                col("val").alias("value"),
                lit("cpu").alias("__name__"),
            ],
            vec![
                col("ts").alias("timestamp"),
                col("val").alias("value"),
                lit("cpu").alias("__name__"),
            ],
        ],
        "timestamp",
    );
    let plan = wrap_in_sort(union, "value");

    let (result, transformed) = apply_rule(plan);
    assert!(
        transformed,
        "should lift constants through both outer and branch wrappers"
    );

    // Structure: Projection -> Sort -> Union -> [Sort -> Proj, Sort -> Proj]
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top");
    };
    assert_eq!(outer.expr.len(), 3);
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));

    let LogicalPlan::Sort(_) = outer.input.as_ref() else {
        panic!("expected Sort under projection");
    };

    let LogicalPlan::Union(inner_union) = outer.input.inputs()[0] else {
        panic!("expected Union under outer Sort");
    };

    for input in &inner_union.inputs {
        let LogicalPlan::Sort(_) = input.as_ref() else {
            panic!("expected Sort in branch");
        };
        let LogicalPlan::Projection(p) = input.inputs()[0] else {
            panic!("expected Projection under branch Sort");
        };
        assert_eq!(p.expr.len(), 2);
    }
}

/// Branch wrapper referencing a constant should block lifting.
#[test]
fn test_branch_wrapper_referencing_constant_blocks_lift() {
    // Sort by the constant column inside each branch.
    let plan = make_sorted_union(
        vec![
            vec![
                col("ts").alias("timestamp"),
                col("val").alias("value"),
                lit("cpu").alias("__name__"),
            ],
            vec![
                col("ts").alias("timestamp"),
                col("val").alias("value"),
                lit("cpu").alias("__name__"),
            ],
        ],
        "__name__",
    );

    let (_result, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "should not lift when branch wrapper references constant"
    );
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Check if an expression is `lit(expected_value).alias(expected_name)`.
fn is_literal_alias(expr: &Expr, expected_value: &str, expected_name: &str) -> bool {
    match expr {
        Expr::Alias(alias) if alias.name == expected_name => match alias.expr.as_ref() {
            Expr::Literal(scalar, _) => scalar.to_string() == expected_value,
            _ => false,
        },
        _ => false,
    }
}

/// Check if an expression is any literal (with or without alias).
fn is_any_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_, _) => true,
        Expr::Alias(alias) => matches!(alias.expr.as_ref(), Expr::Literal(_, _)),
        _ => false,
    }
}
