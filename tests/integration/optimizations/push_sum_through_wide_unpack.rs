use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::Transformed;
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{Aggregate, Expr, Extension, LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::OptimizerRule;
use datafusion::prelude::col;

use datafusion_promql::node::{WideColumnMeta, WideUnpack};
use datafusion_promql::opt::logical::PushSumThroughWideUnpack;
use datafusion_promql::types::Labels;

// ─── Helpers ───────────────────────────────────────────────────────────────

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

fn apply_rule(plan: LogicalPlan) -> (LogicalPlan, bool) {
    let rule = PushSumThroughWideUnpack;
    let Transformed {
        data, transformed, ..
    } = rule.rewrite(plan, &NoopConfig).unwrap();
    (data, transformed)
}

/// Build a wide scan with `(timestamp, c0, c1, c2, c3)`.
fn make_wide_scan() -> LogicalPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("cpu//host-a/op-rx", DataType::Float64, true),
        Field::new("cpu//host-a/op-tx", DataType::Float64, true),
        Field::new("cpu//host-b/op-rx", DataType::Float64, true),
        Field::new("cpu//host-b/op-tx", DataType::Float64, true),
    ]));
    let table = MemTable::try_new(schema, vec![vec![]]).expect("failed to create MemTable");
    LogicalPlanBuilder::scan(
        "cpu",
        datafusion::datasource::provider_as_source(Arc::new(table)),
        None,
    )
    .unwrap()
    .build()
    .unwrap()
}

/// Build column metadata for the four wide columns above.
fn build_columns() -> Vec<WideColumnMeta> {
    let mk = |col_name: &str, host: &str, op: &str| -> WideColumnMeta {
        let mut labels = Labels::new();
        labels.insert("instance".into(), host.into());
        labels.insert("op".into(), op.into());
        WideColumnMeta {
            col_name: col_name.into(),
            metric_name: "cpu".into(),
            labels,
        }
    };
    vec![
        mk("cpu//host-a/op-rx", "host-a", "rx"),
        mk("cpu//host-a/op-tx", "host-a", "tx"),
        mk("cpu//host-b/op-rx", "host-b", "rx"),
        mk("cpu//host-b/op-tx", "host-b", "tx"),
    ]
}

fn wrap_unpack(input: LogicalPlan) -> LogicalPlan {
    let columns = build_columns();
    let label_keys = vec!["instance".to_string(), "op".to_string()];
    let unpack =
        WideUnpack::new(input, Arc::new(columns), Arc::new(label_keys)).expect("WideUnpack");
    LogicalPlan::Extension(Extension {
        node: Arc::new(unpack),
    })
}

/// Build `Aggregate(group=[timestamp, ...labels], sum(value)) WideUnpack(...)`.
fn build_sum_over_unpack(grouping_labels: &[&str]) -> LogicalPlan {
    let unpack = wrap_unpack(make_wide_scan());
    let mut group_exprs: Vec<Expr> = vec![col("timestamp")];
    for l in grouping_labels {
        group_exprs.push(col(*l));
    }
    let agg_expr = datafusion::functions_aggregate::sum::sum(col("value")).alias("value");
    LogicalPlanBuilder::from(unpack)
        .aggregate(group_exprs, vec![agg_expr])
        .unwrap()
        .build()
        .unwrap()
}

fn count_aggregate_nodes(plan: &LogicalPlan) -> usize {
    let mut count = 0;
    if matches!(plan, LogicalPlan::Aggregate(_)) {
        count += 1;
    }
    for child in plan.inputs() {
        count += count_aggregate_nodes(child);
    }
    count
}

fn count_wide_unpack(plan: &LogicalPlan) -> usize {
    let mut count = 0;
    if let LogicalPlan::Extension(ext) = plan
        && ext.node.as_any().downcast_ref::<WideUnpack>().is_some()
    {
        count += 1;
    }
    for child in plan.inputs() {
        count += count_wide_unpack(child);
    }
    count
}

/// Find the first WideUnpack node in the plan.
fn first_wide_unpack(plan: &LogicalPlan) -> Option<&WideUnpack> {
    if let LogicalPlan::Extension(ext) = plan
        && let Some(u) = ext.node.as_any().downcast_ref::<WideUnpack>()
    {
        return Some(u);
    }
    for child in plan.inputs() {
        if let Some(u) = first_wide_unpack(child) {
            return Some(u);
        }
    }
    None
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn fires_for_sum_no_by() {
    // sum(metric) — group by timestamp only, no labels.
    let plan = build_sum_over_unpack(&[]);
    assert_eq!(count_aggregate_nodes(&plan), 1);
    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should fire for sum() over WideUnpack");
    assert_eq!(count_aggregate_nodes(&result), 0);
    assert_eq!(count_wide_unpack(&result), 1);

    // All four columns collapse into a single output column when there are
    // no grouping labels.
    let unpack = first_wide_unpack(&result).expect("WideUnpack");
    assert_eq!(unpack.columns.len(), 1);
    assert!(unpack.label_keys.is_empty());
}

#[test]
fn fires_for_sum_by_instance() {
    // sum(metric) by (instance) — collapses on `op`, keeps `instance`.
    let plan = build_sum_over_unpack(&["instance"]);
    let (result, transformed) = apply_rule(plan);
    assert!(transformed);
    let unpack = first_wide_unpack(&result).expect("WideUnpack");
    // Two unique instances → two output columns.
    assert_eq!(unpack.columns.len(), 2);
    assert_eq!(unpack.label_keys.as_ref(), &vec!["instance".to_string()]);
    let instances: Vec<&str> = unpack
        .columns
        .iter()
        .map(|c| c.labels.get("instance").map(|s| s.as_str()).unwrap_or(""))
        .collect();
    assert!(instances.contains(&"host-a"));
    assert!(instances.contains(&"host-b"));
}

#[test]
fn fires_for_sum_by_both_labels() {
    // sum(metric) by (instance, op) — every wide column is its own group.
    let plan = build_sum_over_unpack(&["instance", "op"]);
    let (result, transformed) = apply_rule(plan);
    assert!(transformed);
    let unpack = first_wide_unpack(&result).expect("WideUnpack");
    assert_eq!(unpack.columns.len(), 4);
    assert_eq!(unpack.label_keys.len(), 2);
}

#[test]
fn no_fire_when_input_is_not_wide_unpack() {
    let scan = make_wide_scan();
    let plan = LogicalPlanBuilder::from(scan)
        .aggregate(
            vec![col("timestamp")],
            vec![
                datafusion::functions_aggregate::sum::sum(col("cpu//host-a/op-rx")).alias("value"),
            ],
        )
        .unwrap()
        .build()
        .unwrap();
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed);
}

#[test]
fn no_fire_for_min() {
    // min() instead of sum() — should not fire.
    let unpack = wrap_unpack(make_wide_scan());
    let plan = LogicalPlanBuilder::from(unpack)
        .aggregate(
            vec![col("timestamp")],
            vec![datafusion::functions_aggregate::min_max::min(col("value")).alias("value")],
        )
        .unwrap()
        .build()
        .unwrap();
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should not fire for non-sum aggregates");
}

#[test]
fn no_fire_when_timestamp_missing_from_grouping() {
    // Grouping omits `timestamp` — the rewrite would change semantics.
    let unpack = wrap_unpack(make_wide_scan());
    let plan = LogicalPlanBuilder::from(unpack)
        .aggregate(
            vec![col("instance")],
            vec![datafusion::functions_aggregate::sum::sum(col("value")).alias("value")],
        )
        .unwrap()
        .build()
        .unwrap();
    let (_, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "rule should not fire when timestamp is absent from the grouping"
    );
}

#[test]
fn rewrite_preserves_output_schema() {
    let plan = build_sum_over_unpack(&["instance"]);
    let LogicalPlan::Aggregate(Aggregate {
        schema: original, ..
    }) = &plan
    else {
        panic!("expected aggregate at top of input plan");
    };
    let original = original.clone();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let new_fields: Vec<&str> = result
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    let original_fields: Vec<&str> = original
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(new_fields, original_fields);

    // Types match too.
    for (a, b) in result
        .schema()
        .fields()
        .iter()
        .zip(original.fields().iter())
    {
        assert_eq!(
            a.data_type(),
            b.data_type(),
            "type mismatch on {}",
            a.name()
        );
    }
}

#[test]
fn no_fire_for_distinct_or_filtered_sum() {
    use datafusion::functions_aggregate::sum::sum_udaf;
    use datafusion::logical_expr::expr::AggregateFunction;

    let unpack = wrap_unpack(make_wide_scan());
    // sum(DISTINCT value) — must not fire.
    let distinct_sum = Expr::AggregateFunction(AggregateFunction::new_udf(
        sum_udaf(),
        vec![col("value")],
        true, // distinct
        None,
        vec![],
        None,
    ))
    .alias("value");
    let plan = LogicalPlanBuilder::from(unpack)
        .aggregate(vec![col("timestamp")], vec![distinct_sum])
        .unwrap()
        .build()
        .unwrap();
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should reject DISTINCT");
}
