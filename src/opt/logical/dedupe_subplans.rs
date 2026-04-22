use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::error::Result;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

/// Optimizer rule that collapses structurally identical subplans.
///
/// When the same logical subplan — same operators, expressions, and schema,
/// all the way to the leaves — appears more than once in the plan tree, this
/// rule replaces every occurrence after the first with a clone of the first
/// instance.  The intent is to give DataFusion (or a subsequent physical-level
/// pass) a single Arc to reuse, so that the work is computed only once.
///
/// A typical trigger is a PromQL binary operation like
///
/// ```text
/// rate(cpu[5m]) / ignoring(op) group_left sum by (id) (rate(cpu[5m]))
/// ```
///
/// where the translator emits the `rate(cpu[5m])` subtree independently for
/// each occurrence in the PromQL AST.
///
/// Fingerprinting uses `LogicalPlan::display_indent_schema`, which recursively
/// formats every operator, expression, and schema field, so two plans are
/// considered identical iff they are structurally equivalent.
#[derive(Debug)]
pub struct DeduplicateSubplans;

impl OptimizerRule for DeduplicateSubplans {
    fn name(&self) -> &str {
        "deduplicate_subplans"
    }

    // We do not use ApplyOrder because we need two full passes over the tree:
    // one to count fingerprint occurrences, one to do the rewiring.
    fn apply_order(&self) -> Option<datafusion::optimizer::optimizer::ApplyOrder> {
        None
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // ── Phase 1: count how many times each subplan fingerprint appears ──
        let mut counts: HashMap<String, usize> = HashMap::new();
        plan.apply(|node| {
            *counts.entry(subplan_fingerprint(node)).or_insert(0) += 1;
            Ok(TreeNodeRecursion::Continue)
        })?;

        // Fingerprints that appear more than once are candidates for sharing.
        let duplicated: HashSet<String> = counts
            .into_iter()
            .filter(|(_, n)| *n > 1)
            .map(|(k, _)| k)
            .collect();

        if duplicated.is_empty() {
            return Ok(Transformed::no(plan));
        }

        // ── Phase 2: bottom-up rewrite ─────────────────────────────────────
        // The first occurrence of each duplicate fingerprint is kept in place
        // and stored in the cache.  Every later occurrence is replaced with a
        // clone sourced from the cache so that downstream passes see identical
        // objects.
        let mut cache: HashMap<String, Arc<LogicalPlan>> = HashMap::new();

        plan.transform_up(|node| {
            let fp = subplan_fingerprint(&node);
            if !duplicated.contains(&fp) {
                return Ok(Transformed::no(node));
            }

            if let Some(cached) = cache.get(&fp) {
                // Subsequent occurrence: replace with cached clone.
                return Ok(Transformed::yes((**cached).clone()));
            }

            // First occurrence: cache it and leave it in place.
            cache.insert(fp, Arc::new(node.clone()));
            Ok(Transformed::no(node))
        })
    }
}

/// Compute a structural fingerprint for a logical plan and all its
/// descendants.  Two plans are fingerprint-equal iff they are structurally
/// identical (same operators, expressions, and schema throughout).
pub fn subplan_fingerprint(plan: &LogicalPlan) -> String {
    format!("{}", plan.display_indent_schema())
}
