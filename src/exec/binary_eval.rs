use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, Float64Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::Distribution;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

use crate::node::{BinaryOp, VectorMatching};

/// Output flush threshold (rows). Each match-key group flushes when its
/// in-progress builder reaches this count, so downstream operators see
/// pipelined batches instead of one final blob.
const OUTPUT_BATCH_ROWS: usize = 8192;

/// Interns label values to `u32` indices so per-row keys can be
/// `Box<[u32]>` instead of `Vec<String>`. The same interner is used for
/// both sides of a binary op so identical label values compare equal
/// across LHS and RHS.
#[derive(Default)]
struct LabelInterner {
    map: HashMap<Box<str>, u32>,
    values: Vec<Box<str>>,
}

impl LabelInterner {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.values.len() as u32;
        let owned: Box<str> = Box::from(s);
        self.map.insert(owned.clone(), id);
        self.values.push(owned);
        id
    }

    fn get(&self, id: u32) -> &str {
        &self.values[id as usize]
    }
}

/// Per-side ingested data: every row of every batch turned into u32
/// label indices, grouped by full series key.
#[derive(Default)]
struct SideData {
    /// Unique full label-value keys (positions match `samples`).
    full_keys: Vec<Box<[u32]>>,
    /// Per-full-key sample buffers. Already timestamp-sorted because
    /// the StepVectorExec/InstantVectorExec children emit rows in
    /// timestamp order; debug builds assert this.
    samples: Vec<Vec<(u64, f64)>>,
    /// full_key → index into `full_keys`/`samples`.
    full_index: HashMap<Box<[u32]>, u32>,
    /// match_key → list of full_key indices that share that match key.
    match_to_full: HashMap<Box<[u32]>, Vec<u32>>,
}

/// Selector describing which positions of a side's full label-value vector
/// form the matching key. `None` entries are filled with the empty-string
/// interner id (used when an `on(...)` label is missing on one side).
type MatchSelector = Vec<Option<usize>>;

fn match_selector(label_columns: &[String], matching: &VectorMatching) -> MatchSelector {
    match (&matching.on_labels, &matching.ignoring_labels) {
        (Some(on), _) => on
            .iter()
            .map(|l| label_columns.iter().position(|c| c == l))
            .collect(),
        (_, Some(ignoring)) => label_columns
            .iter()
            .enumerate()
            .filter(|(_, name)| !ignoring.contains(name) && name.as_str() != "__name__")
            .map(|(i, _)| Some(i))
            .collect(),
        (None, None) => label_columns
            .iter()
            .enumerate()
            .filter(|(_, name)| name.as_str() != "__name__")
            .map(|(i, _)| Some(i))
            .collect(),
    }
}

/// Walk every row of `batches`, intern label values into `interner`, and
/// populate `SideData`. The empty-string id is used for null label cells
/// and for `on(...)` labels that don't exist on this side.
fn ingest_side(
    batches: &[RecordBatch],
    label_columns: &[String],
    selector: &MatchSelector,
    interner: &mut LabelInterner,
    empty_id: u32,
) -> SideData {
    let mut data = SideData::default();
    let mut full_buf: Vec<u32> = Vec::with_capacity(label_columns.len());
    let mut match_buf: Vec<u32> = Vec::with_capacity(selector.len());

    for batch in batches {
        let ts_arr = batch
            .column_by_name("timestamp")
            .expect("missing timestamp")
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .expect("timestamp must be UInt64");
        let val_arr = batch
            .column_by_name("value")
            .expect("missing value")
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("value must be Float64");

        let label_arrays: Vec<Option<&arrow::array::StringArray>> = label_columns
            .iter()
            .map(|name| {
                batch.column_by_name(name).map(|col| {
                    col.as_any()
                        .downcast_ref::<arrow::array::StringArray>()
                        .unwrap_or_else(|| panic!("label column {name} must be Utf8"))
                })
            })
            .collect();

        for row in 0..batch.num_rows() {
            full_buf.clear();
            for arr in &label_arrays {
                let id = match arr {
                    Some(a) if !a.is_null(row) => interner.intern(a.value(row)),
                    _ => empty_id,
                };
                full_buf.push(id);
            }

            let series_idx = match data.full_index.get(full_buf.as_slice()) {
                Some(&idx) => idx,
                None => {
                    let idx = data.full_keys.len() as u32;
                    let key: Box<[u32]> = full_buf.clone().into_boxed_slice();
                    data.full_index.insert(key.clone(), idx);
                    data.full_keys.push(key);
                    data.samples.push(Vec::new());

                    match_buf.clear();
                    for slot in selector {
                        match slot {
                            Some(i) => match_buf.push(full_buf[*i]),
                            None => match_buf.push(empty_id),
                        }
                    }
                    let mk: Box<[u32]> = match_buf.clone().into_boxed_slice();
                    data.match_to_full.entry(mk).or_default().push(idx);

                    idx
                }
            };

            let ts = ts_arr.value(row);
            let val = val_arr.value(row);
            let bucket = &mut data.samples[series_idx as usize];
            debug_assert!(
                bucket.last().is_none_or(|(prev, _)| *prev <= ts),
                "binary_eval: child stream must emit timestamps non-decreasing per series"
            );
            bucket.push((ts, val));
        }
    }

    data
}

/// Mutable output accumulator that flushes whole `RecordBatch`es to a
/// caller-provided sink whenever it crosses `OUTPUT_BATCH_ROWS`.
struct OutputAccumulator {
    schema: SchemaRef,
    out_ts: UInt64Builder,
    out_val: Float64Builder,
    out_labels: Vec<StringBuilder>,
    rows: usize,
    batches: Vec<RecordBatch>,
}

impl OutputAccumulator {
    fn new(schema: SchemaRef, label_count: usize) -> Self {
        Self {
            schema,
            out_ts: UInt64Builder::new(),
            out_val: Float64Builder::new(),
            out_labels: (0..label_count).map(|_| StringBuilder::new()).collect(),
            rows: 0,
            batches: Vec::new(),
        }
    }

    fn append(&mut self, ts: u64, val: f64, label_values: &[&str]) {
        self.out_ts.append_value(ts);
        self.out_val.append_value(val);
        for (i, v) in label_values.iter().enumerate() {
            self.out_labels[i].append_value(v);
        }
        self.rows += 1;
    }

    fn maybe_flush(&mut self, baseline: &BaselineMetrics) -> Result<()> {
        if self.rows >= OUTPUT_BATCH_ROWS {
            self.flush(baseline)?;
        }
        Ok(())
    }

    fn flush(&mut self, baseline: &BaselineMetrics) -> Result<()> {
        if self.rows == 0 {
            return Ok(());
        }
        let mut columns: Vec<arrow::array::ArrayRef> =
            Vec::with_capacity(2 + self.out_labels.len());
        columns.push(Arc::new(self.out_ts.finish()));
        columns.push(Arc::new(self.out_val.finish()));
        for builder in &mut self.out_labels {
            columns.push(Arc::new(builder.finish()));
        }
        let batch = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;
        let batch = batch.record_output(baseline);
        self.batches.push(batch);
        self.rows = 0;
        Ok(())
    }

    fn finish(mut self, baseline: &BaselineMetrics) -> Result<Vec<RecordBatch>> {
        self.flush(baseline)?;
        Ok(self.batches)
    }
}

// ---------------------------------------------------------------------------
// BinaryExec: vector op vector
// ---------------------------------------------------------------------------

/// Physical plan node for binary operations between two instant vectors.
#[derive(Debug)]
pub(crate) struct BinaryExec {
    lhs: Arc<dyn ExecutionPlan>,
    rhs: Arc<dyn ExecutionPlan>,
    op: BinaryOp,
    return_bool: bool,
    matching: VectorMatching,
    lhs_label_columns: Vec<String>,
    rhs_label_columns: Vec<String>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl BinaryExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lhs: Arc<dyn ExecutionPlan>,
        rhs: Arc<dyn ExecutionPlan>,
        op: BinaryOp,
        return_bool: bool,
        matching: VectorMatching,
        output_schema: SchemaRef,
    ) -> Self {
        let lhs_label_columns: Vec<String> = lhs
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        let rhs_label_columns: Vec<String> = rhs
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));

        Self {
            lhs,
            rhs,
            op,
            return_bool,
            matching,
            lhs_label_columns,
            rhs_label_columns,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

/// Extract the matching key from a row based on VectorMatching config.
/// Indices into a side's full label-value vector (`Box<[u32]>`) for each
/// output label, or `None` if the label isn't present on that side.
fn output_label_lookup(label_columns: &[String], output_labels: &[String]) -> Vec<Option<usize>> {
    output_labels
        .iter()
        .map(|label| label_columns.iter().position(|c| c == label))
        .collect()
}

/// Resolve a full_key (`Box<[u32]>`) into the output-label string slice
/// vector via `lookup` and `interner`. Reuses `dst` to avoid allocating
/// per row.
fn resolve_output_labels<'a>(
    full_key: &[u32],
    lookup: &[Option<usize>],
    interner: &'a LabelInterner,
    empty: &'a str,
    dst: &mut Vec<&'a str>,
) {
    dst.clear();
    for slot in lookup {
        let s = match slot {
            Some(i) => interner.get(full_key[*i]),
            None => empty,
        };
        dst.push(s);
    }
}

/// Top-level vector-vector evaluator. Ingests both sides into u32-keyed
/// `SideData`, then walks them in match-key order to apply `op`. Output
/// is flushed in batches whenever the in-progress builder hits
/// [`OUTPUT_BATCH_ROWS`].
#[allow(clippy::too_many_arguments)]
fn run_binary_op(
    lhs_batches: &[RecordBatch],
    rhs_batches: &[RecordBatch],
    lhs_label_columns: &[String],
    rhs_label_columns: &[String],
    output_labels: &[String],
    op: BinaryOp,
    return_bool: bool,
    matching: &VectorMatching,
    output_schema: SchemaRef,
    baseline: &BaselineMetrics,
) -> Result<Vec<RecordBatch>> {
    // Time only the synchronous processing; matches the previous shape
    // where async batch collection isn't billed to elapsed_compute.
    let _timer = baseline.elapsed_compute().timer();

    let mut interner = LabelInterner::default();
    let empty_id = interner.intern("");

    let lhs_selector = match_selector(lhs_label_columns, matching);
    let rhs_selector = match_selector(rhs_label_columns, matching);

    let lhs_data = ingest_side(
        lhs_batches,
        lhs_label_columns,
        &lhs_selector,
        &mut interner,
        empty_id,
    );
    let rhs_data = ingest_side(
        rhs_batches,
        rhs_label_columns,
        &rhs_selector,
        &mut interner,
        empty_id,
    );

    let lhs_output_lookup = output_label_lookup(lhs_label_columns, output_labels);
    let rhs_output_lookup = output_label_lookup(rhs_label_columns, output_labels);

    let mut acc = OutputAccumulator::new(output_schema, output_labels.len());
    let mut label_buf: Vec<&str> = Vec::with_capacity(output_labels.len());

    if op.is_set_operator() {
        run_set_op(
            op,
            &lhs_data,
            &rhs_data,
            &lhs_output_lookup,
            &rhs_output_lookup,
            &interner,
            &mut acc,
            &mut label_buf,
            baseline,
        )?;
    } else {
        run_arith_op(
            op,
            return_bool,
            &lhs_data,
            &rhs_data,
            &lhs_output_lookup,
            &interner,
            &mut acc,
            &mut label_buf,
            baseline,
        )?;
    }

    acc.finish(baseline)
}

#[allow(clippy::too_many_arguments)]
fn run_arith_op<'a>(
    op: BinaryOp,
    return_bool: bool,
    lhs_data: &SideData,
    rhs_data: &SideData,
    lhs_output_lookup: &[Option<usize>],
    interner: &'a LabelInterner,
    acc: &mut OutputAccumulator,
    label_buf: &mut Vec<&'a str>,
    baseline: &BaselineMetrics,
) -> Result<()> {
    // Sort match keys for deterministic output order — same convention as
    // the previous implementation (lexicographic on label-value indices).
    let mut lhs_match_keys: Vec<&Box<[u32]>> = lhs_data.match_to_full.keys().collect();
    lhs_match_keys.sort_unstable();

    let bool_mode = return_bool && op.is_comparison();
    let mut rhs_by_ts: HashMap<u64, f64> = HashMap::new();

    // SAFETY of unsafe-free design: we take a `&'a` borrow of `interner`
    // and emit `&'a str` references into `label_buf`. The accumulator
    // copies them into Arrow `StringBuilder`, so the borrow ends before
    // the next mutable use.
    for mk in lhs_match_keys {
        let lhs_full_indices = &lhs_data.match_to_full[mk];
        let Some(rhs_full_indices) = rhs_data.match_to_full.get(mk.as_ref()) else {
            continue;
        };

        rhs_by_ts.clear();
        for &rhs_idx in rhs_full_indices {
            for &(ts, val) in &rhs_data.samples[rhs_idx as usize] {
                rhs_by_ts.insert(ts, val);
            }
        }

        for &lhs_idx in lhs_full_indices {
            let full_key = &lhs_data.full_keys[lhs_idx as usize];
            resolve_output_labels(full_key, lhs_output_lookup, interner, "", label_buf);

            for &(ts, lhs_val) in &lhs_data.samples[lhs_idx as usize] {
                let Some(&rhs_val) = rhs_by_ts.get(&ts) else {
                    continue;
                };
                if bool_mode {
                    let result = op.evaluate_bool(lhs_val, rhs_val);
                    acc.append(ts, result, label_buf);
                } else if let Some(result) = op.evaluate(lhs_val, rhs_val) {
                    acc.append(ts, result, label_buf);
                }
            }
            acc.maybe_flush(baseline)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_set_op<'a>(
    op: BinaryOp,
    lhs_data: &SideData,
    rhs_data: &SideData,
    lhs_output_lookup: &[Option<usize>],
    rhs_output_lookup: &[Option<usize>],
    interner: &'a LabelInterner,
    acc: &mut OutputAccumulator,
    label_buf: &mut Vec<&'a str>,
    baseline: &BaselineMetrics,
) -> Result<()> {
    match op {
        BinaryOp::Land => {
            for (mk, lhs_keys) in &lhs_data.match_to_full {
                if rhs_data.match_to_full.contains_key(mk) {
                    for &lhs_idx in lhs_keys {
                        emit_series(
                            lhs_data,
                            lhs_idx,
                            lhs_output_lookup,
                            interner,
                            acc,
                            label_buf,
                        );
                        acc.maybe_flush(baseline)?;
                    }
                }
            }
        }
        BinaryOp::Lor => {
            for lhs_keys in lhs_data.match_to_full.values() {
                for &lhs_idx in lhs_keys {
                    emit_series(
                        lhs_data,
                        lhs_idx,
                        lhs_output_lookup,
                        interner,
                        acc,
                        label_buf,
                    );
                    acc.maybe_flush(baseline)?;
                }
            }
            for (mk, rhs_keys) in &rhs_data.match_to_full {
                if !lhs_data.match_to_full.contains_key(mk) {
                    for &rhs_idx in rhs_keys {
                        emit_series(
                            rhs_data,
                            rhs_idx,
                            rhs_output_lookup,
                            interner,
                            acc,
                            label_buf,
                        );
                        acc.maybe_flush(baseline)?;
                    }
                }
            }
        }
        BinaryOp::Lunless => {
            for (mk, lhs_keys) in &lhs_data.match_to_full {
                if !rhs_data.match_to_full.contains_key(mk) {
                    for &lhs_idx in lhs_keys {
                        emit_series(
                            lhs_data,
                            lhs_idx,
                            lhs_output_lookup,
                            interner,
                            acc,
                            label_buf,
                        );
                        acc.maybe_flush(baseline)?;
                    }
                }
            }
        }
        _ => unreachable!("non-set-op routed to run_set_op"),
    }
    Ok(())
}

fn emit_series<'a>(
    side: &SideData,
    idx: u32,
    output_lookup: &[Option<usize>],
    interner: &'a LabelInterner,
    acc: &mut OutputAccumulator,
    label_buf: &mut Vec<&'a str>,
) {
    let full_key = &side.full_keys[idx as usize];
    resolve_output_labels(full_key, output_lookup, interner, "", label_buf);
    for &(ts, val) in &side.samples[idx as usize] {
        acc.append(ts, val, label_buf);
    }
}

impl DisplayAs for BinaryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryExec: op={}", self.op)
    }
}

impl ExecutionPlan for BinaryExec {
    fn name(&self) -> &str {
        "BinaryExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition, Distribution::SinglePartition]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.lhs, &self.rhs]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            Arc::clone(&children[1]),
            self.op,
            self.return_bool,
            self.matching.clone(),
            Arc::clone(&self.output_schema),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let lhs_stream = self.lhs.execute(partition, Arc::clone(&context))?;
        let rhs_stream = self.rhs.execute(partition, Arc::clone(&context))?;
        let output_schema = Arc::clone(&self.output_schema);
        let op = self.op;
        let return_bool = self.return_bool;
        let matching = self.matching.clone();
        let lhs_label_columns = self.lhs_label_columns.clone();
        let rhs_label_columns = self.rhs_label_columns.clone();
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        let output_labels: Vec<String> = output_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            let mut lhs_batches = Vec::new();
            let mut s = lhs_stream;
            while let Some(b) = s.next().await {
                lhs_batches.push(b?);
            }
            let mut rhs_batches = Vec::new();
            let mut s = rhs_stream;
            while let Some(b) = s.next().await {
                rhs_batches.push(b?);
            }

            let batches = run_binary_op(
                &lhs_batches,
                &rhs_batches,
                &lhs_label_columns,
                &rhs_label_columns,
                &output_labels,
                op,
                return_bool,
                &matching,
                Arc::clone(&output_schema),
                &baseline_metrics,
            )?;

            Ok::<_, datafusion::common::DataFusionError>(futures::stream::iter(
                batches.into_iter().map(Ok),
            ))
        });
        let stream = futures::TryStreamExt::try_flatten(stream);

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

// ---------------------------------------------------------------------------
// ScalarBinaryExec: vector op scalar (or scalar op vector)
// ---------------------------------------------------------------------------

/// Physical plan node for binary operations between a vector and a scalar.
#[derive(Debug)]
pub(crate) struct ScalarBinaryExec {
    child: Arc<dyn ExecutionPlan>,
    scalar_value: f64,
    op: BinaryOp,
    scalar_is_lhs: bool,
    return_bool: bool,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl ScalarBinaryExec {
    pub fn new(
        child: Arc<dyn ExecutionPlan>,
        scalar_value: f64,
        op: BinaryOp,
        scalar_is_lhs: bool,
        return_bool: bool,
        output_schema: SchemaRef,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));

        Self {
            child,
            scalar_value,
            op,
            scalar_is_lhs,
            return_bool,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for ScalarBinaryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scalar_is_lhs {
            write!(
                f,
                "ScalarBinaryExec: {} {} vector",
                self.scalar_value, self.op
            )
        } else {
            write!(
                f,
                "ScalarBinaryExec: vector {} {}",
                self.op, self.scalar_value
            )
        }
    }
}

impl ExecutionPlan for ScalarBinaryExec {
    fn name(&self) -> &str {
        "ScalarBinaryExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            self.scalar_value,
            self.op,
            self.scalar_is_lhs,
            self.return_bool,
            Arc::clone(&self.output_schema),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let child_stream = self.child.execute(partition, Arc::clone(&context))?;
        let output_schema = Arc::clone(&self.output_schema);
        let op = self.op;
        let scalar_value = self.scalar_value;
        let scalar_is_lhs = self.scalar_is_lhs;
        let return_bool = self.return_bool;
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        // Output label columns (everything except timestamp/value in output schema).
        let output_labels: Vec<String> = output_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != "timestamp" && n != "value")
            .collect();

        let stream = futures::stream::once(async move {
            use futures::StreamExt;

            let mut batches = Vec::new();
            let mut s = child_stream;
            while let Some(b) = s.next().await {
                batches.push(b?);
            }

            // Time only the synchronous processing that follows.
            let _timer = baseline_metrics.elapsed_compute().timer();

            let mut out_ts = UInt64Builder::new();
            let mut out_val = Float64Builder::new();
            let mut out_labels_builders: Vec<StringBuilder> =
                output_labels.iter().map(|_| StringBuilder::new()).collect();

            for batch in &batches {
                let ts_arr = batch
                    .column_by_name("timestamp")
                    .expect("missing timestamp")
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .expect("timestamp must be UInt64");
                let val_arr = batch
                    .column_by_name("value")
                    .expect("missing value")
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .expect("value must be Float64");

                let label_arrays: Vec<Option<&arrow::array::StringArray>> = output_labels
                    .iter()
                    .map(|name| {
                        batch.column_by_name(name).map(|col| {
                            col.as_any()
                                .downcast_ref::<arrow::array::StringArray>()
                                .unwrap_or_else(|| panic!("label column {name} must be Utf8"))
                        })
                    })
                    .collect();

                for row in 0..batch.num_rows() {
                    let ts = ts_arr.value(row);
                    let vec_val = val_arr.value(row);

                    let (lhs, rhs) = if scalar_is_lhs {
                        (scalar_value, vec_val)
                    } else {
                        (vec_val, scalar_value)
                    };

                    let result = if return_bool && op.is_comparison() {
                        Some(op.evaluate_bool(lhs, rhs))
                    } else {
                        op.evaluate(lhs, rhs)
                    };

                    if let Some(val) = result {
                        out_ts.append_value(ts);
                        out_val.append_value(val);
                        for (i, arr) in label_arrays.iter().enumerate() {
                            let v = arr.map(|a| a.value(row)).unwrap_or("");
                            out_labels_builders[i].append_value(v);
                        }
                    }
                }
            }

            let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();
            columns.push(Arc::new(out_ts.finish()));
            columns.push(Arc::new(out_val.finish()));
            for builder in &mut out_labels_builders {
                columns.push(Arc::new(builder.finish()));
            }

            let batch = RecordBatch::try_new(output_schema, columns)?;
            Ok(batch.record_output(&baseline_metrics))
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
