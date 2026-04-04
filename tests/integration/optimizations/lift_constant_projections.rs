use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, LogicalPlanBuilder, Union};
use datafusion::optimizer::OptimizerRule;
use datafusion::prelude::*;

use datafusion_promql::node::InstantVectorEval;
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

// ─── Union pattern tests ────────────────────────────────────────────────────

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

/// When branches have different literal values for a column, only the shared
/// ones should be lifted.
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

/// Non-Union, non-Sort plans should pass through unchanged.
#[test]
fn test_non_union_plan_unchanged() {
    let scan = make_scan("t0");
    let plan = LogicalPlanBuilder::from(scan)
        .project(vec![col("ts").alias("timestamp"), lit("cpu").alias("name")])
        .unwrap()
        .build()
        .unwrap();

    let (_result, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should not fire on bare projections");
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

// ─── Sort pattern tests ────────────────────────────────────────────────────

/// Helper: wrap a plan in a Sort node sorting by the given column.
fn wrap_in_sort(plan: LogicalPlan, sort_col: &str) -> LogicalPlan {
    LogicalPlanBuilder::from(plan)
        .sort(vec![col(sort_col).sort(true, false)])
        .unwrap()
        .build()
        .unwrap()
}

/// Helper: build a Sort -> Projection plan.
fn make_sort_proj(proj_exprs: Vec<Expr>, sort_col: &str) -> LogicalPlan {
    let scan = make_scan("t0");
    let proj = LogicalPlanBuilder::from(scan)
        .project(proj_exprs)
        .unwrap()
        .build()
        .unwrap();
    wrap_in_sort(proj, sort_col)
}

/// When a Sort wraps a Projection and the sort does not reference the constant
/// columns, the constants should be lifted above the sort.
#[test]
fn test_sort_lifts_constants_from_projection() {
    let plan = make_sort_proj(
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ],
        "value",
    );

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

    // Under the sort should be a Projection with 2 columns.
    let LogicalPlan::Projection(inner_proj) = sort.input.as_ref() else {
        panic!("expected Projection under sort");
    };
    assert_eq!(inner_proj.expr.len(), 2);
}

/// When the sort references a constant column, the rule should NOT lift.
#[test]
fn test_sort_referencing_constant_blocks_lift() {
    let plan = make_sort_proj(
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ],
        "__name__",
    );

    let (_result, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "rule should not lift when sort references a constant column"
    );
}

/// Sort over a Projection with no constants should be unchanged.
#[test]
fn test_sort_no_constants_unchanged() {
    let plan = make_sort_proj(
        vec![col("ts").alias("timestamp"), col("val").alias("value")],
        "value",
    );

    let (_result, transformed) = apply_rule(plan);
    assert!(!transformed, "no constants to lift");
}

/// Sort over a Projection where all columns are constant should not lift
/// (would leave empty inner projection).
#[test]
fn test_sort_all_constants_unchanged() {
    let scan = make_scan("t0");
    let proj = LogicalPlanBuilder::from(scan)
        .project(vec![
            lit("cpu").alias("__name__"),
            lit("host1").alias("host"),
        ])
        .unwrap()
        .build()
        .unwrap();
    let plan = wrap_in_sort(proj, "__name__");

    let (_result, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "should not lift when all columns are constant"
    );
}

/// Output schema should be preserved when lifting through sort.
#[test]
fn test_sort_output_schema_preserved() {
    let plan = make_sort_proj(
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host1").alias("host"),
        ],
        "value",
    );

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

// ─── Composed pattern tests (multi-pass) ────────────────────────────────────

/// Apply the rule repeatedly until it reaches a fixpoint.
fn apply_rule_to_fixpoint(mut plan: LogicalPlan) -> (LogicalPlan, bool) {
    let rule = LiftConstantProjections;
    let mut ever_transformed = false;

    loop {
        // Apply bottom-up: first to children, then to the node itself.
        let Transformed {
            data, transformed, ..
        } = plan
            .transform_up(|node| rule.rewrite(node, &NoopConfig))
            .unwrap();
        plan = data;
        if !transformed {
            break;
        }
        ever_transformed = true;
    }

    (plan, ever_transformed)
}

/// Sort -> Union -> [Proj, Proj]: bottom-up applies union pattern first, then
/// sort pattern lifts constants above the sort.
#[test]
fn test_composed_sort_over_union() {
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

    let (result, transformed) = apply_rule_to_fixpoint(plan);
    assert!(transformed);

    // Structure: Projection -> Sort -> Projection -> Union -> [Proj, Proj]
    // The outer projection has the constants.
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert_eq!(outer.expr.len(), 3);
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));

    // Under projection should be Sort.
    let LogicalPlan::Sort(_) = outer.input.as_ref() else {
        panic!("expected Sort under outer projection");
    };
}

/// Union -> [Sort -> Proj, Sort -> Proj]: bottom-up applies sort pattern to
/// each branch first, then union pattern lifts the shared constants.
#[test]
fn test_composed_union_of_sorted_projections() {
    // Build Union -> [Sort -> Proj, Sort -> Proj]
    let inputs: Vec<Arc<LogicalPlan>> = (0..2)
        .map(|idx| {
            let scan = make_scan(&format!("t{idx}"));
            let proj = LogicalPlanBuilder::from(scan)
                .project(vec![
                    col("ts").alias("timestamp"),
                    col("val").alias("value"),
                    lit("cpu").alias("__name__"),
                ])
                .unwrap()
                .build()
                .unwrap();
            let sorted = LogicalPlanBuilder::from(proj)
                .sort(vec![col("timestamp").sort(true, false)])
                .unwrap()
                .build()
                .unwrap();
            Arc::new(sorted)
        })
        .collect();
    let plan = LogicalPlan::Union(Union::try_new_with_loose_types(inputs).unwrap());

    let (result, transformed) = apply_rule_to_fixpoint(plan);
    assert!(transformed);

    // The constants should end up lifted.
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert_eq!(outer.expr.len(), 3);
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));
}

// ─── Nested projection flattening tests ────────────────────────────────────

/// Basic: Projection over Projection should be flattened into one.
#[test]
fn test_nested_projection_flattened() {
    let scan = make_scan("t0");
    let inner = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ])
        .unwrap()
        .build()
        .unwrap();
    let plan = LogicalPlanBuilder::from(inner)
        .project(vec![
            col("timestamp").alias("timestamp"),
            col("value").alias("value"),
            col("__name__").alias("metric"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "nested projection should be flattened");

    // Should be a single Projection over a Scan.
    let LogicalPlan::Projection(proj) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert_eq!(proj.expr.len(), 3);

    // The inner input should NOT be a projection anymore.
    assert!(
        !matches!(proj.input.as_ref(), LogicalPlan::Projection(_)),
        "inner projection should have been eliminated"
    );

    // The constant __name__ should be inlined as a literal renamed to "metric".
    assert!(
        is_literal_alias(&proj.expr[2], "cpu", "metric"),
        "expected lit('cpu') AS metric, got: {:?}",
        proj.expr[2]
    );
}

/// Chained rename: Projection [B -> C] over Projection [A -> B] should
/// resolve to a single Projection [A -> C].
#[test]
fn test_nested_projection_chained_rename() {
    let scan = make_scan("t0");
    let inner = LogicalPlanBuilder::from(scan)
        .project(vec![col("ts").alias("b"), col("val").alias("value")])
        .unwrap()
        .build()
        .unwrap();
    let plan = LogicalPlanBuilder::from(inner)
        .project(vec![col("b").alias("c"), col("value").alias("value")])
        .unwrap()
        .build()
        .unwrap();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "chained rename should be flattened");

    let LogicalPlan::Projection(proj) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert_eq!(proj.expr.len(), 2);

    // Should NOT be a nested projection.
    assert!(
        !matches!(proj.input.as_ref(), LogicalPlan::Projection(_)),
        "inner projection should have been eliminated"
    );

    // The output schema should have columns "c" and "value".
    let field_names: Vec<&str> = result
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(field_names, vec!["c", "value"]);
}

/// Nested projection where outer adds a new literal should flatten correctly.
#[test]
fn test_nested_projection_outer_adds_literal() {
    let scan = make_scan("t0");
    let inner = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
        ])
        .unwrap()
        .build()
        .unwrap();
    let plan = LogicalPlanBuilder::from(inner)
        .project(vec![
            col("timestamp").alias("timestamp"),
            col("value").alias("value"),
            lit("extra").alias("tag"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let LogicalPlan::Projection(proj) = &result else {
        panic!("expected Projection at top");
    };
    assert_eq!(proj.expr.len(), 3);
    assert!(is_literal_alias(&proj.expr[2], "extra", "tag"));
    assert!(!matches!(proj.input.as_ref(), LogicalPlan::Projection(_)));
}

/// A bare Projection (not nested) should NOT be transformed.
#[test]
fn test_single_projection_unchanged() {
    let scan = make_scan("t0");
    let plan = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let (_result, transformed) = apply_rule(plan);
    assert!(!transformed, "single projection should not be changed");
}

/// Output schema should be preserved after flattening.
#[test]
fn test_nested_projection_schema_preserved() {
    let scan = make_scan("t0");
    let inner = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ])
        .unwrap()
        .build()
        .unwrap();
    let plan = LogicalPlanBuilder::from(inner)
        .project(vec![
            col("__name__").alias("metric"),
            col("timestamp").alias("ts"),
            col("value").alias("v"),
        ])
        .unwrap()
        .build()
        .unwrap();

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

/// Flattening composes with lift: Union creates nested projections that get
/// flattened in a subsequent pass.
#[test]
fn test_flatten_after_lift_union() {
    // Build: Projection -> Union -> [Projection, Projection]
    // The lift_constant rule produces:
    //   Projection(constants) -> Union -> [Projection(stripped), Projection(stripped)]
    // If the inner union branches already had projections, we get nested projections
    // in the branches that should be flattened.

    let scan0 = make_scan("t0");
    let inner0 = LogicalPlanBuilder::from(scan0)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
        ])
        .unwrap()
        .build()
        .unwrap();
    let branch0 = LogicalPlanBuilder::from(inner0)
        .project(vec![
            col("timestamp").alias("timestamp"),
            col("value").alias("value"),
            lit("cpu").alias("__name__"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let scan1 = make_scan("t1");
    let inner1 = LogicalPlanBuilder::from(scan1)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
        ])
        .unwrap()
        .build()
        .unwrap();
    let branch1 = LogicalPlanBuilder::from(inner1)
        .project(vec![
            col("timestamp").alias("timestamp"),
            col("value").alias("value"),
            lit("cpu").alias("__name__"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let plan = LogicalPlan::Union(
        Union::try_new_with_loose_types(vec![Arc::new(branch0), Arc::new(branch1)]).unwrap(),
    );

    let (result, transformed) = apply_rule_to_fixpoint(plan);
    assert!(transformed);

    // After fixpoint: outer Projection (with __name__ constant) -> Union -> [single Proj, single Proj]
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));

    // Each union branch should be a single (flattened) projection, not nested.
    let LogicalPlan::Union(inner_union) = outer.input.as_ref() else {
        panic!("expected Union under outer projection");
    };
    for input in &inner_union.inputs {
        let LogicalPlan::Projection(branch_proj) = input.as_ref() else {
            panic!("expected Projection in union branch, got:\n{input}");
        };
        assert!(
            !matches!(branch_proj.input.as_ref(), LogicalPlan::Projection(_)),
            "union branch should have flattened nested projection"
        );
    }
}

// ─── InstantVectorEval pattern tests ────────────────────────────────────────

/// Helper: wrap a plan in an InstantVectorEval node.
fn wrap_in_instant_vector_eval(input: LogicalPlan, label_columns: Vec<String>) -> LogicalPlan {
    let eval = InstantVectorEval::new(
        input,
        1_000_000_000,   // timestamp_ns
        300_000_000_000, // lookback_ns (5 min)
        0,               // offset_ns
        label_columns,
    );
    LogicalPlan::Extension(Extension {
        node: Arc::new(eval),
    })
}

/// When an InstantVectorEval wraps a Projection with constant columns,
/// those constants should be lifted above the eval node.
#[test]
fn test_instant_vector_eval_lifts_constants() {
    let scan = make_scan("t0");
    let proj = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            col("ts").alias("host"), // non-constant label
        ])
        .unwrap()
        .build()
        .unwrap();

    let plan = wrap_in_instant_vector_eval(proj, vec!["__name__".to_string(), "host".to_string()]);

    let (result, transformed) = apply_rule(plan);
    assert!(
        transformed,
        "rule should lift constants past InstantVectorEval"
    );

    // Top-level should be a Projection with the constant.
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert_eq!(outer.expr.len(), 4);
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));

    // Under the projection should be an InstantVectorEval.
    let LogicalPlan::Extension(ref ext) = *outer.input else {
        panic!("expected Extension under projection");
    };
    let eval = ext
        .node
        .as_any()
        .downcast_ref::<InstantVectorEval>()
        .expect("expected InstantVectorEval");

    // __name__ should be removed from label_columns.
    assert_eq!(eval.label_columns, vec!["host".to_string()]);

    // Inner projection should have 3 columns (timestamp, value, host).
    let LogicalPlan::Projection(ref inner_proj) = eval.input else {
        panic!("expected Projection inside InstantVectorEval");
    };
    assert_eq!(inner_proj.expr.len(), 3);
}

/// Timestamp and value constants should NOT be lifted.
#[test]
fn test_instant_vector_eval_preserves_timestamp_and_value() {
    let scan = make_scan("t0");
    // Unlikely scenario but tests the guard: timestamp and value are literals.
    let proj = LogicalPlanBuilder::from(scan)
        .project(vec![
            lit(1000u64).alias("timestamp"),
            lit(42.0).alias("value"),
            lit("cpu").alias("__name__"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let plan = wrap_in_instant_vector_eval(proj, vec!["__name__".to_string()]);

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "__name__ should still be lifted");

    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };

    // __name__ (index 2) should be lifted.
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));

    // timestamp and value should NOT be lifted — they should be column refs.
    assert!(
        !is_any_literal(&outer.expr[0]),
        "timestamp should not be lifted"
    );
    assert!(
        !is_any_literal(&outer.expr[1]),
        "value should not be lifted"
    );
}

/// When the child of InstantVectorEval is not a Projection, the rule should
/// not fire.
#[test]
fn test_instant_vector_eval_non_projection_child_unchanged() {
    let scan = make_scan("t0");
    let plan = wrap_in_instant_vector_eval(scan, vec![]);

    let (_result, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should not fire on non-projection child");
}

/// When there are no constant columns in the child projection, the rule
/// should not fire.
#[test]
fn test_instant_vector_eval_no_constants_unchanged() {
    let scan = make_scan("t0");
    let proj = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let plan = wrap_in_instant_vector_eval(proj, vec![]);

    let (_result, transformed) = apply_rule(plan);
    assert!(!transformed, "no constants to lift");
}

/// Multiple constants should all be lifted.
#[test]
fn test_instant_vector_eval_multiple_constants_lifted() {
    let scan = make_scan("t0");
    let proj = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("prod").alias("env"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let plan = wrap_in_instant_vector_eval(proj, vec!["__name__".to_string(), "env".to_string()]);

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top");
    };
    assert_eq!(outer.expr.len(), 4);
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));
    assert!(is_literal_alias(&outer.expr[3], "prod", "env"));

    let LogicalPlan::Extension(ref ext) = *outer.input else {
        panic!("expected Extension under projection");
    };
    let eval = ext
        .node
        .as_any()
        .downcast_ref::<InstantVectorEval>()
        .expect("expected InstantVectorEval");

    // Both constants removed from label_columns.
    assert!(eval.label_columns.is_empty());
}

/// Output schema should be preserved when lifting through InstantVectorEval.
#[test]
fn test_instant_vector_eval_schema_preserved() {
    let scan = make_scan("t0");
    let proj = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            col("ts").alias("host"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let plan = wrap_in_instant_vector_eval(proj, vec!["__name__".to_string(), "host".to_string()]);

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

/// Composed: InstantVectorEval over Union with shared constants should lift
/// all the way up in multiple passes.
#[test]
fn test_composed_instant_vector_eval_over_union() {
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

    let plan = wrap_in_instant_vector_eval(union, vec!["__name__".to_string(), "host".to_string()]);

    let (result, transformed) = apply_rule_to_fixpoint(plan);
    assert!(transformed);

    // After fixpoint: Projection(__name__=cpu) -> InstantVectorEval -> ... -> Union
    let LogicalPlan::Projection(outer) = &result else {
        panic!("expected Projection at top, got:\n{result}");
    };
    assert!(is_literal_alias(&outer.expr[2], "cpu", "__name__"));

    // host should not be lifted (differs between branches).
    assert!(!is_any_literal(&outer.expr[3]));
}

// ─── Helpers ────────────────────────────────────────────────────────────────

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
