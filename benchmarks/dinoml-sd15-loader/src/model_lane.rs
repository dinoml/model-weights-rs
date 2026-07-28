use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use dinoml_checkpoint::BindingOperation;
use model_weights::cache::Cache;
use model_weights::identity::{BackendId, ContentDigest, ContractId, ManifestId, SelectionId};
use model_weights::materialize::{Materializer, PreparedOrigin, WeightDelivery};
use model_weights::operation::{
    Axis, Concat, Operation, OperationGraph, OperationGraphBuilder, Permute, Reshape, TensorFacts,
    ValueRef,
};
use model_weights::pipeline::{Pipeline, PreparedItem};
use model_weights::plan::{
    BindingPlan, ExtraSourcePolicy, PlanInputs, PlannedTransform, Requirement, SourceTensor,
    TargetTensor, TensorName,
};
use model_weights::prepare::{
    PreparationEngine, Representation, TransformSpec, builtin_contiguous_implementation,
};
use model_weights::quantization::Storage;
use model_weights::telemetry::{
    ExecutionEvent, ExecutionObserver, ExecutionPhase, OperationKind, OperationLocation,
};
use model_weights::tensor::DType;
use model_weights::{AccessMode, CancellationToken, Checkpoint};

use crate::cli::ExecutionConfig;
use crate::contract::{AppResult, Component, ComponentContract, ContractTarget, TargetStorage};
use crate::kyxc;
use crate::report::{
    ExecutionLimitReport, LaneReport, OperationLocationMetrics, OperationMetrics,
    OperationNodeMetrics, OriginCounts, PeakBytes, PhaseMetrics, TargetDigest, set_digest,
};

pub struct ModelSetup {
    components: Vec<ModelComponent>,
    preparation: PreparationEngine,
    cache: Option<Cache>,
    cache_directory: Option<String>,
}

struct ModelComponent {
    component: Component,
    checkpoint: Checkpoint,
    plan: BindingPlan,
}

pub struct ModelOutcome {
    pub report: LaneReport,
    pub digests: Vec<TargetDigest>,
}

impl ModelSetup {
    pub fn new(components: &[ComponentContract], cache: Option<&Path>) -> AppResult<Self> {
        let preparation = PreparationEngine::with_builtins()?;
        let cancellation = CancellationToken::new();
        let components = components
            .iter()
            .map(|component| {
                let checkpoint = Checkpoint::builder(component.source.clone())
                    .access_mode(access_mode())
                    .open()?;
                let plan = build_plan(&checkpoint, component, &cancellation)?;
                Ok(ModelComponent {
                    component: component.component,
                    checkpoint,
                    plan,
                })
            })
            .collect::<AppResult<Vec<_>>>()?;
        let cache_directory = cache.map(|path| path.display().to_string());
        let cache = cache.map(Cache::open).transpose()?;
        Ok(Self {
            components,
            preparation,
            cache,
            cache_directory,
        })
    }

    pub fn reset_prepared(&self) -> model_weights::Result<usize> {
        let Some(cache) = self.cache.as_ref() else {
            return Ok(0);
        };
        let cancellation = CancellationToken::new();
        let mut removed = 0_usize;
        for component in &self.components {
            let materializer = Materializer::new(
                &component.checkpoint,
                &component.plan,
                &self.preparation,
                &cancellation,
            )?
            .with_cache(cache);
            for binding in component.plan.bindings() {
                let Some(address) = materializer.prepared_cache_address_with_cancellation(
                    binding.target().name().as_str(),
                    &cancellation,
                )?
                else {
                    continue;
                };
                removed = removed.saturating_add(
                    cache
                        .evict(
                            model_weights::cache::CacheNamespace::Prepared,
                            address.key(),
                            model_weights::cache::EvictionReason::Explicit,
                            &cancellation,
                        )?
                        .is_some()
                        .into(),
                );
            }
        }
        Ok(removed)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "keeping the timed materialization loop cohesive makes its boundary auditable"
    )]
    pub fn run(
        &self,
        execution_config: ExecutionConfig,
        validate: bool,
        prepared_entries_reset: usize,
    ) -> AppResult<ModelOutcome> {
        let mut origins = OriginCounts::default();
        let mut phase_totals = BTreeMap::new();
        let mut operation_totals = BTreeMap::new();
        let mut operation_nodes = Vec::new();
        let mut target_count = 0_u64;
        let mut delivered_bytes = 0_u64;
        let mut pipeline_time = Duration::ZERO;
        let mut peak_source = 0_u64;
        let mut peak_scratch = 0_u64;
        let mut peak_prepared = 0_u64;
        let mut digests = Vec::new();
        let started = Instant::now();
        for component in &self.components {
            let cancellation = CancellationToken::new();
            let observer = BenchmarkObserver::new(
                component.component,
                component
                    .plan
                    .bindings()
                    .iter()
                    .map(|binding| binding.target().name().as_str().to_owned()),
            );
            let materializer = Materializer::new_with_observer(
                &component.checkpoint,
                &component.plan,
                &self.preparation,
                &cancellation,
                &observer,
            )?;
            let materializer = match self.cache.as_ref() {
                Some(cache) => materializer.with_cache(cache),
                None => materializer,
            };
            let limits =
                execution_config.with_max_work_items(component.plan.bindings().len().max(1));
            let pipeline = Pipeline::with_cancellation(limits, cancellation)?;
            let mut sink = |ordinal: u64,
                            item: PreparedItem<WeightDelivery>,
                            cancellation: &CancellationToken|
             -> model_weights::Result<()> {
                cancellation.check()?;
                let index = usize::try_from(ordinal).map_err(|error| {
                    model_weights::Error::from_category_with_source(
                        model_weights::ErrorCategory::Delivery,
                        "delivery ordinal does not fit usize",
                        error,
                    )
                })?;
                let name = observer.target_name(index).ok_or_else(|| {
                    model_weights::Error::from_category(
                        model_weights::ErrorCategory::Delivery,
                        "delivery ordinal is outside the binding plan",
                    )
                })?;
                match item.into_value() {
                    WeightDelivery::Prepared(weight) => {
                        let bytes = weight.bytes().as_slice();
                        let byte_len = u64::try_from(bytes.len()).map_err(|error| {
                            model_weights::Error::from_category_with_source(
                                model_weights::ErrorCategory::ResourceLimit,
                                "delivered byte length does not fit u64",
                                error,
                            )
                        })?;
                        record_origin(&mut origins, weight.origin(), byte_len);
                        if validate {
                            digests.push(TargetDigest::from_bytes(
                                component.component,
                                name,
                                bytes,
                            ));
                        } else {
                            let _ = std::hint::black_box(bytes);
                        }
                    }
                    delivery => {
                        let resident_bytes = delivery.resident_bytes();
                        let _ = std::hint::black_box(resident_bytes);
                        origins.other = origins.other.saturating_add(1);
                        origins.other_bytes = origins.other_bytes.saturating_add(resident_bytes);
                        return Err(model_weights::Error::from_category(
                            model_weights::ErrorCategory::Delivery,
                            "SD1.5 benchmark expected a prepared host delivery",
                        ));
                    }
                }
                Ok(())
            };
            let execution = materializer.execute(&pipeline, &mut sink, &observer)?;
            let counters = execution.counters();
            target_count = target_count.saturating_add(counters.delivered());
            delivered_bytes = delivered_bytes.saturating_add(counters.delivered_bytes());
            pipeline_time = pipeline_time.saturating_add(execution.wall_time());
            let peaks = execution.peak_bytes();
            peak_source = peak_source.max(peaks.source());
            peak_scratch = peak_scratch.max(peaks.scratch());
            peak_prepared = peak_prepared.max(peaks.prepared());
            merge_phase_totals(&mut phase_totals, observer.phase_snapshot());
            let component_operations = observer.operation_snapshot();
            merge_operation_totals(&mut operation_totals, &component_operations);
            operation_nodes.extend(observer.operation_node_report(component_operations));
        }
        let elapsed = started.elapsed();
        let output_set_sha256 = validate.then(|| set_digest(&digests));
        operation_nodes.sort_by(|left, right| {
            left.component
                .cmp(&right.component)
                .then_with(|| left.work_ordinal.cmp(&right.work_ordinal))
                .then_with(|| left.location.cmp(&right.location))
                .then_with(|| left.kind.cmp(right.kind))
        });
        Ok(ModelOutcome {
            report: LaneReport {
                lane: "model-weights",
                setup_ms: 0.0,
                materialization_ms: milliseconds(elapsed),
                pipeline_ms: Some(milliseconds(pipeline_time)),
                target_count,
                delivered_bytes,
                throughput_mib_per_second: throughput(delivered_bytes, elapsed),
                workers: execution_config.workers,
                execution_limits: Some(ExecutionLimitReport::from(execution_config)),
                cache_directory: self.cache_directory.clone(),
                prepared_entries_reset,
                output_set_sha256,
                origins,
                peak_bytes: PeakBytes {
                    source: Some(peak_source),
                    scratch: Some(peak_scratch),
                    prepared: Some(peak_prepared),
                },
                phases: phase_report(phase_totals),
                operations: operation_report(operation_totals),
                operation_nodes,
            },
            digests,
        })
    }
}

fn build_plan(
    checkpoint: &Checkpoint,
    component: &ComponentContract,
    cancellation: &CancellationToken,
) -> AppResult<BindingPlan> {
    let sources = checkpoint
        .inventory()
        .iter()
        .map(SourceTensor::try_from)
        .collect::<model_weights::Result<Vec<_>>>()?;
    let targets = component
        .targets
        .iter()
        .map(|target| {
            build_target(checkpoint, target).map_err(|error| {
                format!(
                    "building model-weights target {:?}: {error}",
                    target.metadata.name
                )
                .into()
            })
        })
        .collect::<AppResult<Vec<_>>>()?;
    let snapshot = checkpoint.snapshot_id(cancellation)?;
    let contract_bytes = serde_json::to_vec(&targets)?;
    let component_label = component.component.label().as_bytes();
    let inputs = PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash(
            "dinoml-sd15-benchmark-manifest-v1",
            [snapshot.digest().as_bytes().as_slice(), component_label],
        )),
        SelectionId::from_digest(ContentDigest::hash(
            "dinoml-sd15-benchmark-selection-v1",
            [component_label],
        )),
        ContractId::from_digest(ContentDigest::hash(
            "dinoml-sd15-benchmark-contract-v1",
            [contract_bytes.as_slice()],
        )),
        BackendId::from_digest(ContentDigest::hash(
            "dinoml-sd15-benchmark-backend-v1",
            [b"host/dinoml-runtime-abi10".as_slice()],
        )),
        checkpoint.source_digests(cancellation)?,
    );
    Ok(BindingPlan::builder(inputs)
        .sources(sources)
        .targets(targets)
        .extra_source_policy(ExtraSourcePolicy::Allow)
        .build()?)
}

fn build_target(checkpoint: &Checkpoint, target: &ContractTarget) -> AppResult<TargetTensor> {
    if matches!(
        target.binding().operation(),
        BindingOperation::Tensor { .. }
    ) && target.storage == TargetStorage::Logical
    {
        return build_direct_target(checkpoint, target);
    }
    build_graph_target(checkpoint, target)
}

fn build_direct_target(
    checkpoint: &Checkpoint,
    target: &ContractTarget,
) -> AppResult<TargetTensor> {
    let source_name = target
        .single_source_name()
        .ok_or("direct target does not have exactly one source")?;
    let record = plain_record(checkpoint, source_name)?;
    let Storage::Plain {
        dtype: source_dtype,
        ..
    } = record.storage()
    else {
        return Err(format!("checkpoint source {source_name:?} is quantized").into());
    };
    let target_dtype = runtime_dtype(target.metadata.dtype)?;
    let target_shape = runtime_shape(&target.metadata.shape)?;
    let mut builder = TargetTensor::builder(
        TensorName::parse(&target.metadata.name)?,
        Requirement::Required,
        target_shape,
        Representation::contiguous(target_dtype),
        target.output_bytes,
    )
    .source_shape(record.shape());
    if target.metadata.name != source_name {
        builder = builder.aliases([TensorName::parse(source_name)?]);
    }
    if *source_dtype != target_dtype {
        let target_representation = Representation::contiguous(target_dtype);
        builder = builder.transforms([PlannedTransform::new(
            TransformSpec::new(
                builtin_contiguous_implementation()?,
                Representation::contiguous(*source_dtype),
                target_representation,
            ),
            target.output_bytes,
        )]);
    }
    Ok(builder.build()?)
}

fn build_graph_target(checkpoint: &Checkpoint, target: &ContractTarget) -> AppResult<TargetTensor> {
    let target_dtype = runtime_dtype(target.metadata.dtype)?;
    let target_shape = runtime_shape(&target.metadata.shape)?;
    let target_strides = target
        .logical_strides
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let target_representation = match target.storage {
        TargetStorage::Logical => Representation::contiguous(target_dtype),
        TargetStorage::CkKyxc => Representation::new(target_dtype, kyxc::layout()?),
    };
    let target_storage_shape = match target.storage {
        TargetStorage::Logical => target_shape.clone(),
        TargetStorage::CkKyxc => ck_storage_shape(&target_shape)?,
    };

    let mut graph = OperationGraph::builder();
    let mut inputs = Vec::new();
    inputs.try_reserve_exact(target.ordered_source_names().len())?;
    let mut source_dtype = None;
    for source_name in target.ordered_source_names() {
        let record = plain_record(checkpoint, source_name)?;
        let Storage::Plain { dtype, .. } = record.storage() else {
            return Err(format!("checkpoint source {source_name:?} is quantized").into());
        };
        if source_dtype.is_some_and(|current| current != *dtype) {
            return Err(format!(
                "target {:?} groups sources with different dtypes",
                target.metadata.name
            )
            .into());
        }
        source_dtype = Some(*dtype);
        let facts = TensorFacts::contiguous(record.shape(), Representation::contiguous(*dtype))?;
        inputs.push(graph.add_input(TensorName::parse(source_name)?, facts)?);
    }
    let source_dtype = source_dtype.ok_or("operation graph has no source dtype")?;
    let mut current = build_logical_binding_operation(
        &mut graph,
        target.binding().operation(),
        &inputs,
        &target_shape,
    )?;
    if source_dtype != target_dtype {
        let cast = PlannedTransform::new(
            TransformSpec::new(
                builtin_contiguous_implementation()?,
                Representation::contiguous(source_dtype),
                Representation::contiguous(target_dtype),
            ),
            target_dtype.byte_len(&target_shape)?,
        );
        current = add_operation(&mut graph, Operation::Prepare(cast), [current])?;
    }
    if target.storage == TargetStorage::CkKyxc {
        current = add_operation(
            &mut graph,
            Operation::Permute(Permute::storage([0_u32, 2, 3, 1], kyxc::layout()?)?),
            [current],
        )?;
    }
    let graph = graph.build(current)?;
    let target_storage_strides = graph.output_facts().storage_strides().to_vec();
    let expected_facts = TensorFacts::new(
        target_shape.clone(),
        target_storage_shape.clone(),
        target_strides.clone(),
        target_storage_strides.clone(),
        target_representation.clone(),
        target.output_bytes,
    )?;
    if graph.output_facts() != &expected_facts {
        return Err(format!(
            "graph facts {:?} differ from target facts {:?}",
            graph.output_facts(),
            expected_facts
        )
        .into());
    }
    Ok(TargetTensor::builder(
        TensorName::parse(&target.metadata.name)?,
        Requirement::Required,
        target_shape,
        target_representation,
        target.output_bytes,
    )
    .storage_shape(target_storage_shape)
    .logical_strides(target_strides)
    .storage_strides(target_storage_strides)
    .operation_graph(graph)
    .build()?)
}

fn build_logical_binding_operation(
    graph: &mut OperationGraphBuilder,
    binding: &BindingOperation,
    inputs: &[ValueRef],
    target_shape: &[u64],
) -> AppResult<ValueRef> {
    Ok(match binding {
        BindingOperation::Tensor { .. } => {
            *inputs.first().ok_or("tensor binding has no graph input")?
        }
        BindingOperation::Concat { axis, .. } => add_operation(
            graph,
            Operation::Concat(Concat::new(Axis::try_from(*axis)?)),
            inputs,
        )?,
        BindingOperation::Transpose { axes, .. } => {
            let rank = target_shape.len();
            let mut order = (0..rank)
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let [left, right] = *axes;
            if left >= order.len() || right >= order.len() {
                return Err("transpose binding axes exceed the source rank".into());
            }
            order.swap(left, right);
            add_operation(graph, Operation::Permute(Permute::logical(order)?), inputs)?
        }
        BindingOperation::Reshape { .. } => add_operation(
            graph,
            Operation::Reshape(Reshape::new(target_shape)),
            inputs,
        )?,
        _ => return Err("target uses a newer binding operation".into()),
    })
}

fn add_operation(
    graph: &mut OperationGraphBuilder,
    operation: Operation,
    inputs: impl Into<Box<[ValueRef]>>,
) -> AppResult<ValueRef> {
    let implementation = operation.builtin_implementation()?;
    let outputs = graph.add_operation(implementation, operation, inputs)?;
    let [output] = outputs.as_ref() else {
        return Err("benchmark operation did not produce exactly one output".into());
    };
    Ok(*output)
}

fn plain_record<'a>(
    checkpoint: &'a Checkpoint,
    source_name: &str,
) -> AppResult<&'a model_weights::inventory::TensorRecord> {
    checkpoint
        .inventory()
        .tensor(source_name)
        .ok_or_else(|| format!("checkpoint source {source_name:?} is missing").into())
}

fn runtime_shape(shape: &[usize]) -> AppResult<Vec<u64>> {
    Ok(shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<_, _>>()?)
}

fn ck_storage_shape(logical_shape: &[u64]) -> AppResult<Vec<u64>> {
    let [output_channels, input_channels, kernel_height, kernel_width] = logical_shape else {
        return Err("CK KYXC target requires rank-four logical OIHW shape".into());
    };
    Ok(vec![
        *output_channels,
        *kernel_height,
        *kernel_width,
        *input_channels,
    ])
}

fn runtime_dtype(dtype: dinoml_runtime::DType) -> AppResult<DType> {
    Ok(match dtype {
        dinoml_runtime::DType::Float16 => DType::F16,
        dinoml_runtime::DType::BFloat16 => DType::Bf16,
        dinoml_runtime::DType::Float32 => DType::F32,
        other => return Err(format!("unsupported DinoML target dtype {other}").into()),
    })
}

const fn access_mode() -> AccessMode {
    #[cfg(windows)]
    {
        AccessMode::Mmap
    }
    #[cfg(not(windows))]
    {
        AccessMode::Read
    }
}

fn record_origin(origins: &mut OriginCounts, origin: PreparedOrigin, byte_len: u64) {
    match origin {
        PreparedOrigin::Source => {
            origins.source = origins.source.saturating_add(1);
            origins.source_bytes = origins.source_bytes.saturating_add(byte_len);
        }
        PreparedOrigin::Transform => {
            origins.transform = origins.transform.saturating_add(1);
            origins.transform_bytes = origins.transform_bytes.saturating_add(byte_len);
        }
        PreparedOrigin::OperationGraph => {
            origins.operation_graph = origins.operation_graph.saturating_add(1);
            origins.operation_graph_bytes = origins.operation_graph_bytes.saturating_add(byte_len);
        }
        PreparedOrigin::Cache => {
            origins.cache = origins.cache.saturating_add(1);
            origins.cache_bytes = origins.cache_bytes.saturating_add(byte_len);
        }
        _ => {
            origins.other = origins.other.saturating_add(1);
            origins.other_bytes = origins.other_bytes.saturating_add(byte_len);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PhaseTotal {
    duration: Duration,
    bytes: u64,
    invocations: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct OperationTotal {
    duration: Duration,
    input_bytes: u64,
    output_bytes: u64,
    materialized_output_bytes: u64,
    invocations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OperationSite {
    work_ordinal: Option<u64>,
    location: OperationLocation,
    kind: OperationKind,
}

#[derive(Debug)]
struct BenchmarkObserver {
    component: Component,
    target_names: Box<[String]>,
    phases: Mutex<BTreeMap<ExecutionPhase, PhaseTotal>>,
    operation_sites: Mutex<BTreeMap<OperationSite, OperationTotal>>,
}

impl BenchmarkObserver {
    fn new(
        component: Component,
        target_names: impl IntoIterator<Item = String>,
    ) -> BenchmarkObserver {
        Self {
            component,
            target_names: target_names.into_iter().collect(),
            phases: Mutex::default(),
            operation_sites: Mutex::default(),
        }
    }

    fn target_name(&self, index: usize) -> Option<&str> {
        self.target_names.get(index).map(String::as_str)
    }

    fn phase_snapshot(&self) -> BTreeMap<ExecutionPhase, PhaseTotal> {
        self.phases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn operation_snapshot(&self) -> BTreeMap<OperationSite, OperationTotal> {
        self.operation_sites
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn operation_node_report(
        &self,
        operations: BTreeMap<OperationSite, OperationTotal>,
    ) -> Vec<OperationNodeMetrics> {
        operations
            .into_iter()
            .map(|(site, total)| OperationNodeMetrics {
                component: self.component,
                target: site
                    .work_ordinal
                    .and_then(|ordinal| usize::try_from(ordinal).ok())
                    .and_then(|index| self.target_name(index))
                    .map(str::to_owned),
                work_ordinal: site.work_ordinal,
                location: operation_location_report(site.location),
                kind: site.kind.as_str(),
                elapsed_sum_ms: milliseconds(total.duration),
                input_bytes: total.input_bytes,
                output_bytes: total.output_bytes,
                materialized_output_bytes: total.materialized_output_bytes,
                invocations: total.invocations,
            })
            .collect()
    }
}

impl ExecutionObserver for BenchmarkObserver {
    fn observe(&self, event: &ExecutionEvent) {
        match event {
            ExecutionEvent::PhaseFinished {
                phase,
                duration,
                bytes,
                ..
            } => {
                let mut phases = self
                    .phases
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let total = phases.entry(*phase).or_default();
                total.duration = total.duration.saturating_add(*duration);
                total.bytes = total.bytes.saturating_add(*bytes);
                total.invocations = total.invocations.saturating_add(1);
            }
            ExecutionEvent::OperationFinished {
                work_ordinal,
                location,
                kind,
                duration,
                input_bytes,
                output_bytes,
                materialized_output_bytes,
                ..
            } => {
                let mut operation_sites = self
                    .operation_sites
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let total = operation_sites
                    .entry(OperationSite {
                        work_ordinal: *work_ordinal,
                        location: *location,
                        kind: *kind,
                    })
                    .or_default();
                total.duration = total.duration.saturating_add(*duration);
                total.input_bytes = total.input_bytes.saturating_add(*input_bytes);
                total.output_bytes = total.output_bytes.saturating_add(*output_bytes);
                total.materialized_output_bytes = total
                    .materialized_output_bytes
                    .saturating_add(*materialized_output_bytes);
                total.invocations = total.invocations.saturating_add(1);
            }
            _ => {}
        }
    }

    fn operation_events_enabled(&self) -> bool {
        true
    }
}

fn merge_phase_totals(
    aggregate: &mut BTreeMap<ExecutionPhase, PhaseTotal>,
    component: BTreeMap<ExecutionPhase, PhaseTotal>,
) {
    for (phase, component_total) in component {
        let total = aggregate.entry(phase).or_default();
        total.duration = total.duration.saturating_add(component_total.duration);
        total.bytes = total.bytes.saturating_add(component_total.bytes);
        total.invocations = total
            .invocations
            .saturating_add(component_total.invocations);
    }
}

fn merge_operation_totals(
    aggregate: &mut BTreeMap<OperationKind, OperationTotal>,
    component: &BTreeMap<OperationSite, OperationTotal>,
) {
    for (site, component_total) in component {
        let total = aggregate.entry(site.kind).or_default();
        total.duration = total.duration.saturating_add(component_total.duration);
        total.input_bytes = total
            .input_bytes
            .saturating_add(component_total.input_bytes);
        total.output_bytes = total
            .output_bytes
            .saturating_add(component_total.output_bytes);
        total.materialized_output_bytes = total
            .materialized_output_bytes
            .saturating_add(component_total.materialized_output_bytes);
        total.invocations = total
            .invocations
            .saturating_add(component_total.invocations);
    }
}

fn phase_report(phases: BTreeMap<ExecutionPhase, PhaseTotal>) -> BTreeMap<String, PhaseMetrics> {
    phases
        .into_iter()
        .map(|(phase, total)| {
            (
                phase_name(phase).to_owned(),
                PhaseMetrics {
                    milliseconds: milliseconds(total.duration),
                    bytes: total.bytes,
                    invocations: total.invocations,
                },
            )
        })
        .collect()
}

const fn operation_location_report(location: OperationLocation) -> OperationLocationMetrics {
    match location {
        OperationLocation::Binding => OperationLocationMetrics::Binding,
        OperationLocation::PlannedTransform { index } => {
            OperationLocationMetrics::PlannedTransform { index }
        }
        OperationLocation::GraphNode(node_id) => OperationLocationMetrics::GraphNode {
            node_id: node_id.ordinal(),
        },
        _ => OperationLocationMetrics::Other,
    }
}

fn operation_report(
    operations: BTreeMap<OperationKind, OperationTotal>,
) -> BTreeMap<String, OperationMetrics> {
    operations
        .into_iter()
        .map(|(kind, total)| {
            (
                kind.as_str().to_owned(),
                OperationMetrics {
                    elapsed_sum_ms: milliseconds(total.duration),
                    input_bytes: total.input_bytes,
                    output_bytes: total.output_bytes,
                    materialized_output_bytes: total.materialized_output_bytes,
                    invocations: total.invocations,
                },
            )
        })
        .collect()
}

const fn phase_name(phase: ExecutionPhase) -> &'static str {
    match phase {
        ExecutionPhase::Hashing => "hashing",
        ExecutionPhase::Mapping => "mapping",
        ExecutionPhase::SourceRead => "source_read",
        ExecutionPhase::CacheLookup => "cache_lookup",
        ExecutionPhase::Transform => "transform",
        ExecutionPhase::Preparation => "preparation",
        ExecutionPhase::QueueWait => "queue_wait",
        ExecutionPhase::DeliveryCallback => "delivery",
        _ => "other",
    }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark byte counts are far below f64's exact integer range"
)]
fn throughput(bytes: u64, duration: Duration) -> f64 {
    if duration.is_zero() {
        return 0.0;
    }
    bytes as f64 / duration.as_secs_f64() / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use model_weights::materialize::PreparedOrigin;
    use model_weights::operation::NodeId;
    use model_weights::telemetry::OperationLocation;

    use crate::contract::Component;
    use crate::report::{OperationLocationMetrics, OriginCounts};

    use super::{
        BenchmarkObserver, ExecutionEvent, ExecutionObserver, OperationKind,
        merge_operation_totals, operation_report, record_origin,
    };

    #[test]
    fn benchmark_observer_reports_component_target_and_operation_site_totals() {
        let observer = BenchmarkObserver::new(
            Component::Unet,
            (0..8).map(|index| format!("target-{index}")),
        );
        assert!(observer.operation_events_enabled());

        for _ in 0..2 {
            observer.observe(&ExecutionEvent::OperationFinished {
                work_ordinal: Some(7),
                location: OperationLocation::GraphNode(NodeId::from_ordinal(3)),
                kind: OperationKind::Permute,
                duration: Duration::from_micros(1_500),
                input_bytes: 16,
                output_bytes: 16,
                materialized_output_bytes: 16,
            });
        }

        let totals = observer.operation_snapshot();
        assert_eq!(totals.len(), 1);
        let mut aggregate = BTreeMap::new();
        merge_operation_totals(&mut aggregate, &totals);
        let aggregate = operation_report(aggregate);
        let permute = aggregate.get("permute").expect("permute kind total");
        assert!((permute.elapsed_sum_ms - 3.0).abs() < f64::EPSILON);
        assert_eq!(permute.input_bytes, 32);
        assert_eq!(permute.output_bytes, 32);
        assert_eq!(permute.materialized_output_bytes, 32);
        assert_eq!(permute.invocations, 2);

        let nodes = observer.operation_node_report(totals);
        let [node] = nodes.as_slice() else {
            panic!("expected one operation-node total");
        };
        assert_eq!(node.component, Component::Unet);
        assert_eq!(node.target.as_deref(), Some("target-7"));
        assert_eq!(node.work_ordinal, Some(7));
        assert_eq!(
            node.location,
            OperationLocationMetrics::GraphNode { node_id: 3 }
        );
        assert_eq!(node.kind, "permute");
        assert!((node.elapsed_sum_ms - 3.0).abs() < f64::EPSILON);
        assert_eq!(node.invocations, 2);
    }

    #[test]
    fn route_origins_keep_operation_graph_separate_from_transform() {
        let mut origins = OriginCounts::default();
        record_origin(&mut origins, PreparedOrigin::Source, 10);
        record_origin(&mut origins, PreparedOrigin::Transform, 20);
        record_origin(&mut origins, PreparedOrigin::OperationGraph, 30);
        record_origin(&mut origins, PreparedOrigin::Cache, 40);

        assert_eq!(origins.source, 1);
        assert_eq!(origins.source_bytes, 10);
        assert_eq!(origins.transform, 1);
        assert_eq!(origins.transform_bytes, 20);
        assert_eq!(origins.operation_graph, 1);
        assert_eq!(origins.operation_graph_bytes, 30);
        assert_eq!(origins.cache, 1);
        assert_eq!(origins.cache_bytes, 40);
        assert_eq!(origins.other, 0);
        assert_eq!(origins.other_bytes, 0);
    }
}
