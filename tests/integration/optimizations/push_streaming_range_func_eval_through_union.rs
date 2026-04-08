use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::Transformed;
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, LogicalPlanBuilder, Union};
use datafusion::optimizer::OptimizerRule;
use datafusion::prelude::*;

use datafusion_promql::RangeFunction;
use datafusion_promql::node::StreamingRangeFunctionEval;
use datafusion_promql::opt::logical::PushStreamingRangeFuncEvalThroughUnion;

/// Build a trivial single-row MemTable scan.
fn make_scan(alias: &str) -> LogicalPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::UInt64, false),
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

fn wrap_streaming_range_func(input: LogicalPlan, label_columns: Vec<String>) -> LogicalPlan {
    let eval = StreamingRangeFunctionEval::new(
        input,
        RangeFunction::Irate,
        None,
        300_000_000_000,     // 5min range
        Some(1_000_000_000), // instant query at 1s
        0,
        0,
        0,
        0,
        label_columns,
        None,
    )
    .expect("failed to create StreamingRangeFunctionEval");
    LogicalPlan::Extension(Extension {
        node: Arc::new(eval),
    })
}

fn apply_rule(plan: LogicalPlan) -> (LogicalPlan, bool) {
    let rule = PushStreamingRangeFuncEvalThroughUnion;
    let Transformed {
        data, transformed, ..
    } = rule.rewrite(plan, &NoopConfig).unwrap();
    (data, transformed)
}

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

// ─── Tests ──────────────────────────────────────────────────────────────────

/// When union branches have disjoint label constants, StreamingRangeFunctionEval
/// should be pushed down into each branch.
#[test]
fn test_push_down_disjoint_branches() {
    let union = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host-a").alias("instance"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host-b").alias("instance"),
        ],
    ]);

    let plan =
        wrap_streaming_range_func(union, vec!["__name__".to_string(), "instance".to_string()]);

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should have rewritten the plan");

    // Result should be a Union at the top level.
    let LogicalPlan::Union(ref union) = result else {
        panic!("expected Union at top level, got:\n{result}");
    };

    // Each branch should be a StreamingRangeFunctionEval.
    assert_eq!(union.inputs.len(), 2);
    for (i, input) in union.inputs.iter().enumerate() {
        let LogicalPlan::Extension(ref ext) = **input else {
            panic!("branch {i} should be Extension, got:\n{input}");
        };
        assert!(
            ext.node
                .as_any()
                .downcast_ref::<StreamingRangeFunctionEval>()
                .is_some(),
            "branch {i} should be StreamingRangeFunctionEval"
        );
    }
}

/// Three branches with unique label constants should all get
/// StreamingRangeFunctionEval.
#[test]
fn test_push_down_three_branches() {
    let union = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("a").alias("job"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("b").alias("job"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("c").alias("job"),
        ],
    ]);

    let plan = wrap_streaming_range_func(union, vec!["job".to_string()]);
    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let LogicalPlan::Union(ref union) = result else {
        panic!("expected Union");
    };
    assert_eq!(union.inputs.len(), 3);
}

/// When branches have identical label constants (not disjoint), the rule
/// should NOT fire.
#[test]
fn test_no_push_down_identical_labels() {
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

    let plan = wrap_streaming_range_func(union, vec!["__name__".to_string()]);
    let (_, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "identical labels should not trigger push down"
    );
}

/// When branches have some shared constants but at least one column differs,
/// the rule should fire because the fingerprint includes all constant label
/// columns and the overall tuple is unique.
#[test]
fn test_push_down_shared_plus_different() {
    let union = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host-a").alias("instance"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
            lit("host-b").alias("instance"),
        ],
    ]);

    let plan =
        wrap_streaming_range_func(union, vec!["__name__".to_string(), "instance".to_string()]);
    let (_, transformed) = apply_rule(plan);
    assert!(
        transformed,
        "should push down when at least one label column differs"
    );
}

/// When the input to StreamingRangeFunctionEval is not a Union, the rule
/// should not fire.
#[test]
fn test_no_push_down_non_union_input() {
    let scan = make_scan("t0");
    let proj = LogicalPlanBuilder::from(scan)
        .project(vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("cpu").alias("__name__"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let plan = wrap_streaming_range_func(proj, vec!["__name__".to_string()]);
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed);
}

/// When union branches are not projections, the rule should not fire.
#[test]
fn test_no_push_down_non_projection_branches() {
    let inputs: Vec<Arc<LogicalPlan>> = (0..2)
        .map(|i| Arc::new(make_scan(&format!("t{i}"))))
        .collect();
    let union = LogicalPlan::Union(Union::try_new_with_loose_types(inputs).unwrap());

    let plan = wrap_streaming_range_func(union, vec![]);
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed);
}

/// When no label columns have constant values, the rule should not fire.
#[test]
fn test_no_push_down_no_constant_labels() {
    let union = make_union(vec![
        vec![col("ts").alias("timestamp"), col("val").alias("value")],
        vec![col("ts").alias("timestamp"), col("val").alias("value")],
    ]);

    let plan = wrap_streaming_range_func(union, vec!["__name__".to_string()]);
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed);
}

/// The pushed-down StreamingRangeFunctionEval should preserve all parameters.
#[test]
fn test_push_down_preserves_parameters() {
    let union = make_union(vec![
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("a").alias("job"),
        ],
        vec![
            col("ts").alias("timestamp"),
            col("val").alias("value"),
            lit("b").alias("job"),
        ],
    ]);

    let eval = StreamingRangeFunctionEval::new(
        union,
        RangeFunction::Irate,
        Some(42.0),
        60_000_000_000, // range_ns
        None,           // eval_ts_ns (range query)
        1_000_000_000,  // start_ns
        10_000_000_000, // end_ns
        1_000_000_000,  // step_ns
        5_000_000_000,  // offset_ns
        vec!["job".to_string()],
        Some(99_000_000_000), // at_timestamp_ns
    )
    .expect("failed to create StreamingRangeFunctionEval");
    let plan = LogicalPlan::Extension(Extension {
        node: Arc::new(eval),
    });

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let LogicalPlan::Union(ref union) = result else {
        panic!("expected Union");
    };

    for input in &union.inputs {
        let LogicalPlan::Extension(ref ext) = **input else {
            panic!("expected Extension");
        };
        let eval = ext
            .node
            .as_any()
            .downcast_ref::<StreamingRangeFunctionEval>()
            .unwrap();
        assert_eq!(eval.func, RangeFunction::Irate);
        assert_eq!(eval.scalar_arg, Some(42.0));
        assert_eq!(eval.range_ns, 60_000_000_000);
        assert_eq!(eval.eval_ts_ns, None);
        assert_eq!(eval.start_ns, 1_000_000_000);
        assert_eq!(eval.end_ns, 10_000_000_000);
        assert_eq!(eval.step_ns, 1_000_000_000);
        assert_eq!(eval.offset_ns, 5_000_000_000);
        assert_eq!(eval.label_columns, vec!["job".to_string()]);
        assert_eq!(eval.at_timestamp_ns, Some(99_000_000_000));
    }
}
