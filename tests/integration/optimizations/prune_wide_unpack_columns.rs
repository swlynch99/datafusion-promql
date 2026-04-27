use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::Transformed;
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{Extension, LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::OptimizerRule;
use datafusion::prelude::col;

use datafusion_promql::node::{WideColumnMeta, WideUnpack};
use datafusion_promql::opt::logical::PruneWideUnpackColumns;
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
    let rule = PruneWideUnpackColumns;
    let Transformed {
        data, transformed, ..
    } = rule.rewrite(plan, &NoopConfig).unwrap();
    (data, transformed)
}

fn make_wide_scan() -> LogicalPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("cpu//host-a/op-rx", DataType::Float64, true),
        Field::new("cpu//host-a/op-tx", DataType::Float64, true),
        Field::new("cpu//host-b/op-rx", DataType::Float64, true),
    ]));
    let table = MemTable::try_new(schema, vec![vec![]]).expect("MemTable");
    LogicalPlanBuilder::scan(
        "cpu",
        datafusion::datasource::provider_as_source(Arc::new(table)),
        None,
    )
    .unwrap()
    .build()
    .unwrap()
}

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
fn prunes_unused_label_and_name() {
    // Projection only references timestamp/value/instance — `op` and
    // `__name__` are unused and should be dropped from the WideUnpack output.
    let unpack = wrap_unpack(make_wide_scan());
    let plan = LogicalPlanBuilder::from(unpack)
        .project(vec![col("timestamp"), col("value"), col("instance")])
        .unwrap()
        .build()
        .unwrap();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should fire when columns are unused");

    let unpack = first_wide_unpack(&result).expect("WideUnpack survives");
    assert!(!unpack.include_name, "__name__ should be pruned");
    assert_eq!(unpack.label_keys.as_ref(), &vec!["instance".to_string()]);
}

#[test]
fn keeps_referenced_columns() {
    // Projection references everything — nothing to prune.
    let unpack = wrap_unpack(make_wide_scan());
    let plan = LogicalPlanBuilder::from(unpack)
        .project(vec![
            col("timestamp"),
            col("value"),
            col("__name__"),
            col("instance"),
            col("op"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let (_result, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should not fire when nothing is unused");
}

#[test]
fn prunes_only_name_when_labels_used() {
    let unpack = wrap_unpack(make_wide_scan());
    let plan = LogicalPlanBuilder::from(unpack)
        .project(vec![
            col("timestamp"),
            col("value"),
            col("instance"),
            col("op"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);
    let unpack = first_wide_unpack(&result).expect("WideUnpack");
    assert!(!unpack.include_name);
    assert_eq!(unpack.label_keys.len(), 2);
}

#[test]
fn prunes_only_label_when_name_used() {
    let unpack = wrap_unpack(make_wide_scan());
    let plan = LogicalPlanBuilder::from(unpack)
        .project(vec![
            col("timestamp"),
            col("value"),
            col("__name__"),
            col("instance"),
        ])
        .unwrap()
        .build()
        .unwrap();

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);
    let unpack = first_wide_unpack(&result).expect("WideUnpack");
    assert!(unpack.include_name);
    assert_eq!(unpack.label_keys.as_ref(), &vec!["instance".to_string()]);
}

#[test]
fn no_fire_when_input_is_not_wide_unpack() {
    let plan = LogicalPlanBuilder::from(make_wide_scan())
        .project(vec![col("timestamp")])
        .unwrap()
        .build()
        .unwrap();
    let (_, transformed) = apply_rule(plan);
    assert!(!transformed);
}

#[test]
fn rewritten_unpack_drops_columns_from_schema() {
    let unpack = wrap_unpack(make_wide_scan());
    let plan = LogicalPlanBuilder::from(unpack)
        .project(vec![col("timestamp"), col("value"), col("instance")])
        .unwrap()
        .build()
        .unwrap();
    let (result, _) = apply_rule(plan);

    let unpack = first_wide_unpack(&result).expect("WideUnpack");
    let names: Vec<&str> = unpack
        .output_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    // Output is exactly [timestamp, value, instance] — no __name__, no op.
    assert_eq!(names, vec!["timestamp", "value", "instance"]);
}
