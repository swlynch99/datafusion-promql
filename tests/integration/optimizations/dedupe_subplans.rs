use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Float64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use chrono::TimeZone;
use datafusion::catalog::TableProvider;
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::OptimizerRule;

use datafusion_promql::PromqlEngine;
use datafusion_promql::PromqlPlanner;
use datafusion_promql::datasource::{Matcher, MetricMeta, MetricSource, TableFormat, ValueKind};
use datafusion_promql::error::Result;
use datafusion_promql::opt::logical::{DeduplicateSubplans, subplan_fingerprint};
use datafusion_promql::types::{QueryResult, TimeRange};

// ─── NoopConfig ─────────────────────────────────────────────────────────────

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

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn apply_rule(plan: LogicalPlan) -> (LogicalPlan, bool) {
    let rule = DeduplicateSubplans;
    let Transformed {
        data, transformed, ..
    } = rule.rewrite(plan, &NoopConfig).unwrap();
    (data, transformed)
}

/// Count how many extension nodes with `target_name` appear in the plan tree.
fn count_extension_nodes(plan: &LogicalPlan, target_name: &str) -> usize {
    let mut total = 0;
    plan.apply(|node| {
        if let LogicalPlan::Extension(ext) = node
            && ext.node.name() == target_name
        {
            total += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    total
}

/// Count distinct subplan fingerprints for extension nodes with `target_name`.
///
/// When two subtrees are structurally identical — same operators, expressions,
/// and schema — they share the same fingerprint.  This function counts only
/// distinct fingerprints, so `rate(a) * rate(a)` reports 1 unique fingerprint
/// for the `RangeFunctionEval` node even though there are 2 occurrences.
fn count_unique_fingerprints(plan: &LogicalPlan, target_name: &str) -> usize {
    let mut seen: HashSet<String> = HashSet::new();
    plan.apply(|node| {
        if let LogicalPlan::Extension(ext) = node
            && ext.node.name() == target_name
        {
            seen.insert(subplan_fingerprint(node));
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    seen.len()
}

/// Count how many times each subplan fingerprint appears (exposed for tests).
fn fingerprint_counts(plan: &LogicalPlan) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    plan.apply(|node| {
        *counts.entry(subplan_fingerprint(node)).or_insert(0) += 1;
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    counts
}

// ─── Metric source ────────────────────────────────────────────────────────────

struct SimpleSource {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
}

#[async_trait]
impl MetricSource for SimpleSource {
    async fn table_for_metric(
        &self,
        _metric_name: &str,
        _matchers: &[Matcher],
        _time_range: TimeRange,
    ) -> Result<(Arc<dyn TableProvider>, TableFormat)> {
        let table = MemTable::try_new(Arc::clone(&self.schema), vec![self.batches.clone()])
            .map_err(|e| datafusion_promql::error::PromqlError::DataSource(e.to_string()))?;
        Ok((
            Arc::new(table),
            TableFormat::Long {
                value_kind: ValueKind::Scalar,
            },
        ))
    }

    async fn list_metrics(&self, _name_matcher: Option<&Matcher>) -> Result<Vec<MetricMeta>> {
        Ok(vec![MetricMeta {
            name: "cpu_usage".into(),
            label_names: vec!["instance".into()],
            extra_columns: vec![],
        }])
    }
}

/// One series with 5 samples at 10-second intervals starting at t=10s.
/// Values rise by 10 per step so rate ≈ 1.0 /s (Δvalue = 40 over 40 s).
fn make_source() -> SimpleSource {
    let schema = Arc::new(Schema::new(vec![
        Field::new("__name__", DataType::Utf8, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("instance", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
                "cpu_usage",
            ])),
            Arc::new(UInt64Array::from(vec![
                10_000_000_000_u64,
                20_000_000_000,
                30_000_000_000,
                40_000_000_000,
                50_000_000_000,
            ])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
            Arc::new(StringArray::from(vec![
                "host1", "host1", "host1", "host1", "host1",
            ])),
        ],
    )
    .unwrap();

    SimpleSource {
        schema,
        batches: vec![batch],
    }
}

/// Build the *unoptimized* logical plan for an instant query.
async fn raw_plan(query: &str) -> LogicalPlan {
    let source = Arc::new(make_source());
    let planner = PromqlPlanner::new(source);
    let ts = chrono::Utc.timestamp_opt(50, 0).unwrap();
    planner
        .instant_logical_plan(query, ts)
        .await
        .expect("failed to build logical plan")
}

// ─── Unit tests: duplicate detection ─────────────────────────────────────────

#[tokio::test]
async fn rule_fires_for_duplicate_rate_subplan() {
    // rate(cpu_usage[60s]) * rate(cpu_usage[60s]) — the translator emits the
    // rate subtree twice independently.
    let plan = raw_plan("rate(cpu_usage[60s]) * rate(cpu_usage[60s])").await;

    // Before the rule: both rate subtrees are present.
    assert_eq!(
        count_extension_nodes(&plan, "RangeFunctionEval"),
        2,
        "raw plan should have 2 RangeFunctionEval nodes"
    );
    // Both subtrees are structurally identical → 1 unique fingerprint.
    assert_eq!(
        count_unique_fingerprints(&plan, "RangeFunctionEval"),
        1,
        "both RangeFunctionEval nodes should be structurally identical"
    );

    // The rule should fire because duplicates are present.
    let (result, transformed) = apply_rule(plan);
    assert!(
        transformed,
        "rule should fire when identical subplans are present"
    );

    // After the rule: still 2 structural nodes (both arms of the binary op are
    // needed), and still only 1 unique fingerprint.
    assert_eq!(
        count_extension_nodes(&result, "RangeFunctionEval"),
        2,
        "result should still have 2 RangeFunctionEval nodes"
    );
    assert_eq!(
        count_unique_fingerprints(&result, "RangeFunctionEval"),
        1,
        "after deduplication there should be 1 unique RangeFunctionEval fingerprint"
    );
}

#[tokio::test]
async fn rule_fires_for_different_window_sharing_same_scan() {
    // Even with different window sizes, both arms scan the same metric table
    // with identical predicates, so the table-scan subtrees are duplicated
    // and the rule should still fire.
    let plan = raw_plan("rate(cpu_usage[60s]) * rate(cpu_usage[30s])").await;

    let (_, transformed) = apply_rule(plan);
    assert!(
        transformed,
        "rule should fire because the underlying table scans are identical"
    );
}

#[tokio::test]
async fn rule_detects_duplicates_correctly() {
    // Verify the fingerprint-counting phase: with two identical rate calls,
    // at least one fingerprint in the tree appears more than once.
    let plan = raw_plan("rate(cpu_usage[60s]) * rate(cpu_usage[60s])").await;

    let counts = fingerprint_counts(&plan);
    let has_duplicate = counts.values().any(|&n| n > 1);
    assert!(
        has_duplicate,
        "fingerprint_counts should detect duplicates for rate(a) * rate(a)"
    );
}

#[tokio::test]
async fn rule_does_not_fire_for_simple_query() {
    // A query with no repeated subplan — the rule should be a no-op.
    let plan = raw_plan("rate(cpu_usage[60s])").await;
    let (_, transformed) = apply_rule(plan);
    assert!(
        !transformed,
        "rule should NOT fire for a query with no duplicate subplans"
    );
}

// ─── End-to-end correctness tests ────────────────────────────────────────────

#[tokio::test]
async fn e2e_rate_times_rate_produces_correct_value() {
    // rate(cpu[60s]) * rate(cpu[60s]) should equal rate² ≈ 1.0 * 1.0 = 1.0.
    let source = Arc::new(make_source());
    let engine = PromqlEngine::new(source);
    let ts = chrono::Utc.timestamp_opt(50, 0).unwrap();

    let result = engine
        .instant_query("rate(cpu_usage[60s]) * rate(cpu_usage[60s])", ts)
        .await
        .unwrap();
    let QueryResult::Vector(samples) = result else {
        panic!("expected Vector result");
    };
    assert_eq!(samples.len(), 1, "should have exactly one sample");
    // rate = Δvalue / Δtime = 40 / 40s = 1.0, so rate * rate = 1.0
    let val = samples[0].value;
    assert!(
        val > 0.5 && val < 2.0,
        "rate(cpu)*rate(cpu) should be near 1.0, got {val}"
    );
}

#[tokio::test]
async fn e2e_squared_rate_equals_rate_times_rate() {
    // rate(cpu[60s]) and rate(cpu[60s]) * rate(cpu[60s]) should satisfy
    // result_product ≈ single² to confirm the deduplication doesn't corrupt
    // the result.
    let source1 = Arc::new(make_source());
    let source2 = Arc::new(make_source());
    let ts = chrono::Utc.timestamp_opt(50, 0).unwrap();

    let r1 = PromqlEngine::new(source1)
        .instant_query("rate(cpu_usage[60s])", ts)
        .await
        .unwrap();
    let r2 = PromqlEngine::new(source2)
        .instant_query("rate(cpu_usage[60s]) * rate(cpu_usage[60s])", ts)
        .await
        .unwrap();

    let QueryResult::Vector(s1) = r1 else {
        panic!()
    };
    let QueryResult::Vector(s2) = r2 else {
        panic!()
    };
    assert_eq!(s1.len(), 1);
    assert_eq!(s2.len(), 1);

    let single = s1[0].value;
    let product = s2[0].value;
    let expected = single * single;
    assert!(
        (product - expected).abs() < 1e-9,
        "rate*rate ({product}) should equal rate² ({expected})"
    );
}
