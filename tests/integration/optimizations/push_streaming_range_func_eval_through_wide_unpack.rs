use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::Transformed;
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{Extension, LogicalPlan, LogicalPlanBuilder};
use datafusion::optimizer::OptimizerRule;

use datafusion_promql::RangeFunction;
use datafusion_promql::node::{
    StreamingRangeFunctionEval, WideColumnMeta, WideStreamingRangeFunctionEval, WideUnpack,
};
use datafusion_promql::opt::logical::PushStreamingRangeFuncEvalThroughWideUnpack;
use datafusion_promql::types::Labels;

/// Build a wide-format MemTable scan with `(timestamp, col_0, col_1, col_2)`.
fn make_wide_scan() -> LogicalPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("cpu//host-a", DataType::Float64, true),
        Field::new("cpu//host-b", DataType::Float64, true),
        Field::new("cpu//host-c", DataType::Float64, true),
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

/// Build a WideUnpack wrapping the given wide-format input.
fn wrap_unpack(input: LogicalPlan) -> LogicalPlan {
    let mut a_labels = Labels::new();
    a_labels.insert("instance".to_string(), "host-a".to_string());
    let mut b_labels = Labels::new();
    b_labels.insert("instance".to_string(), "host-b".to_string());
    let mut c_labels = Labels::new();
    c_labels.insert("instance".to_string(), "host-c".to_string());

    let columns = vec![
        WideColumnMeta {
            col_name: "cpu//host-a".to_string(),
            metric_name: "cpu".to_string(),
            labels: a_labels,
        },
        WideColumnMeta {
            col_name: "cpu//host-b".to_string(),
            metric_name: "cpu".to_string(),
            labels: b_labels,
        },
        WideColumnMeta {
            col_name: "cpu//host-c".to_string(),
            metric_name: "cpu".to_string(),
            labels: c_labels,
        },
    ];
    let label_keys = vec!["instance".to_string()];
    let unpack = WideUnpack::new(input, Arc::new(columns), Arc::new(label_keys))
        .expect("failed to create WideUnpack");
    LogicalPlan::Extension(Extension {
        node: Arc::new(unpack),
    })
}

/// Wrap an input plan in a `StreamingRangeFunctionEval` with the supplied
/// label columns.
#[allow(clippy::too_many_arguments)]
fn wrap_streaming_range_func(
    input: LogicalPlan,
    label_columns: Vec<String>,
    func: RangeFunction,
    scalar_arg: Option<f64>,
    range_ns: u64,
    eval_ts_ns: Option<u64>,
    start_ns: u64,
    end_ns: u64,
    step_ns: u64,
    offset_ns: i64,
    at_timestamp_ns: Option<u64>,
) -> LogicalPlan {
    let eval = StreamingRangeFunctionEval::new(
        input,
        func,
        scalar_arg,
        range_ns,
        eval_ts_ns,
        start_ns,
        end_ns,
        step_ns,
        offset_ns,
        Arc::new(label_columns),
        at_timestamp_ns,
    )
    .expect("failed to create StreamingRangeFunctionEval");
    LogicalPlan::Extension(Extension {
        node: Arc::new(eval),
    })
}

fn apply_rule(plan: LogicalPlan) -> (LogicalPlan, bool) {
    let rule = PushStreamingRangeFuncEvalThroughWideUnpack;
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

/// When a StreamingRangeFunctionEval sits directly on top of a WideUnpack
/// with matching label columns, the rule should rewrite to
/// `WideUnpack > WideStreamingRangeFunctionEval`.
#[test]
fn test_push_down_matching_labels() {
    let scan = make_wide_scan();
    let unpack = wrap_unpack(scan);
    let plan = wrap_streaming_range_func(
        unpack,
        vec!["__name__".to_string(), "instance".to_string()],
        RangeFunction::Rate,
        None,
        60_000_000_000,
        None,
        0,
        10_000_000_000,
        1_000_000_000,
        0,
        None,
    );

    let (result, transformed) = apply_rule(plan);
    assert!(transformed, "rule should rewrite the plan");

    let LogicalPlan::Extension(ref ext) = result else {
        panic!("expected Extension at top, got:\n{result}");
    };
    assert!(
        ext.node.as_any().downcast_ref::<WideUnpack>().is_some(),
        "top node should be WideUnpack, got {}",
        ext.node.name()
    );

    let inputs = ext.node.inputs();
    assert_eq!(inputs.len(), 1);
    let LogicalPlan::Extension(inner) = inputs[0] else {
        panic!("expected inner Extension, got:\n{}", inputs[0]);
    };
    let wide_eval = inner
        .node
        .as_any()
        .downcast_ref::<WideStreamingRangeFunctionEval>()
        .expect("inner node should be WideStreamingRangeFunctionEval");
    assert_eq!(wide_eval.value_columns.len(), 3);
    assert_eq!(wide_eval.func, RangeFunction::Rate);
}

/// When the label columns do not match the WideUnpack's `{__name__} ∪
/// label_keys`, the rule should not fire. Missing `__name__` is a common
/// case.
#[test]
fn test_no_push_down_missing_name_label() {
    let scan = make_wide_scan();
    let unpack = wrap_unpack(scan);
    let plan = wrap_streaming_range_func(
        unpack,
        vec!["instance".to_string()], // missing __name__
        RangeFunction::Rate,
        None,
        60_000_000_000,
        None,
        0,
        10_000_000_000,
        1_000_000_000,
        0,
        None,
    );

    let (_, transformed) = apply_rule(plan);
    assert!(!transformed, "rule should not fire with mismatched labels");
}

/// When the StreamingRangeFunctionEval's input is not a WideUnpack, the rule
/// should not fire.
#[test]
fn test_no_push_down_non_unpack_input() {
    let scan = make_wide_scan();
    let plan = wrap_streaming_range_func(
        scan,
        vec!["__name__".to_string(), "instance".to_string()],
        RangeFunction::Rate,
        None,
        60_000_000_000,
        None,
        0,
        10_000_000_000,
        1_000_000_000,
        0,
        None,
    );

    let (_, transformed) = apply_rule(plan);
    assert!(!transformed);
}

/// The pushed-down WideStreamingRangeFunctionEval should preserve all
/// parameters from the original streaming eval.
#[test]
fn test_push_down_preserves_parameters() {
    let scan = make_wide_scan();
    let unpack = wrap_unpack(scan);
    let plan = wrap_streaming_range_func(
        unpack,
        vec!["__name__".to_string(), "instance".to_string()],
        RangeFunction::Irate,
        Some(42.0),
        60_000_000_000,
        None,
        1_000_000_000,
        10_000_000_000,
        1_000_000_000,
        5_000_000_000,
        Some(99_000_000_000),
    );

    let (result, transformed) = apply_rule(plan);
    assert!(transformed);

    let LogicalPlan::Extension(ref ext) = result else {
        panic!("expected Extension");
    };
    let inputs = ext.node.inputs();
    let LogicalPlan::Extension(inner) = inputs[0] else {
        panic!("expected inner Extension");
    };
    let wide_eval = inner
        .node
        .as_any()
        .downcast_ref::<WideStreamingRangeFunctionEval>()
        .unwrap();

    assert_eq!(wide_eval.func, RangeFunction::Irate);
    assert_eq!(wide_eval.scalar_arg, Some(42.0));
    assert_eq!(wide_eval.range_ns, 60_000_000_000);
    assert_eq!(wide_eval.eval_ts_ns, None);
    assert_eq!(wide_eval.start_ns, 1_000_000_000);
    assert_eq!(wide_eval.end_ns, 10_000_000_000);
    assert_eq!(wide_eval.step_ns, 1_000_000_000);
    assert_eq!(wide_eval.offset_ns, 5_000_000_000);
    assert_eq!(wide_eval.at_timestamp_ns, Some(99_000_000_000));
    assert_eq!(wide_eval.value_columns.len(), 3);
    assert_eq!(wide_eval.value_columns[0], "cpu//host-a");
    assert_eq!(wide_eval.value_columns[1], "cpu//host-b");
    assert_eq!(wide_eval.value_columns[2], "cpu//host-c");
}
