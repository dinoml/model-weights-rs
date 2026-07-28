//! Public-API integration coverage for grouped tensor assembly.

use std::error::Error as StdError;
use std::fs::File;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use half::f16;
use model_weights::cache::Cache;
use model_weights::identity::{BackendId, ContentDigest, ContractId, ManifestId, SelectionId};
use model_weights::limits::ExecutionLimits;
use model_weights::materialize::{
    Materializer, OperationExecution, PreparedOrigin, WeightDelivery,
};
use model_weights::operation::{
    Axis, Concat, NodeId, Operation, OperationGraph, Reshape, TensorFacts,
};
use model_weights::pipeline::{Pipeline, PreparedItem};
use model_weights::plan::{
    BindingPlan, PlanInputs, PlannedTransform, Requirement, SourceTensor, TargetTensor, TensorName,
};
use model_weights::prepare::{
    PreparationEngine, Representation, TransformSpec, builtin_contiguous_implementation,
};
use model_weights::quantization::Storage;
use model_weights::telemetry::{
    ExecutionEvent, ExecutionObserver, ExecutionPhase, NoopObserver, OperationKind,
    OperationLocation, with_operation_events,
};
use model_weights::tensor::DType;
use model_weights::{CancellationToken, Checkpoint, ErrorCategory, Result};
use tempfile::TempDir;

type TestResult = std::result::Result<(), Box<dyn StdError>>;

const TARGET_NAME: &str = "assembled.qkv";
const RESHAPE_TARGET_NAME: &str = "reshaped.q";
const SEMANTIC_ORDER: [&str; 3] = ["q", "k", "v"];
const REORDERED_INPUTS: [&str; 3] = ["k", "q", "v"];

struct OperationsDisabledObserver;

impl ExecutionObserver for OperationsDisabledObserver {
    fn observe(&self, _event: &ExecutionEvent) {
        panic!("operation-disabled observer received an event");
    }
}

#[test]
fn grouped_concat_prepare_round_trips_materializes_and_reuses_cache() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("qkv.safetensors");
    write_qkv_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = grouped_plan(&checkpoint, &SEMANTIC_ORDER, &cancellation)?;

    assert_eq!(
        checkpoint
            .inventory()
            .iter()
            .map(model_weights::inventory::TensorRecord::name)
            .collect::<Vec<_>>(),
        ["k", "q", "v"],
        "the fixture inventory must differ from semantic Q/K/V order"
    );
    let binding = &plan.bindings()[0];
    assert_eq!(
        binding
            .sources()
            .iter()
            .map(|source| source.name().as_str())
            .collect::<Vec<_>>(),
        SEMANTIC_ORDER
    );

    let canonical = plan.to_canonical_json()?;
    let decoded = BindingPlan::from_canonical_json(&canonical)?;
    assert_eq!(decoded.to_canonical_json()?.as_ref(), canonical.as_ref());
    assert_eq!(decoded.id(), plan.id());
    assert_eq!(decoded, plan);

    let engine = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &engine, &cancellation)?.with_cache(&cache);

    let cold = materializer.materialize(TARGET_NAME, &cancellation)?;
    let WeightDelivery::Prepared(cold) = cold else {
        return Err(io::Error::other("host graph did not produce prepared bytes").into());
    };
    assert_eq!(cold.origin(), PreparedOrigin::OperationGraph);
    assert_eq!(
        cold.source_names()
            .iter()
            .map(TensorName::as_str)
            .collect::<Vec<_>>(),
        SEMANTIC_ORDER
    );
    assert_eq!(cold.shape(), [6]);
    assert_eq!(
        cold.representation(),
        &Representation::contiguous(DType::F16)
    );
    assert_eq!(cold.resident_bytes(), 12);
    assert_eq!(cold.bytes().as_slice(), expected_f16_bytes());

    let warm = materializer.materialize(TARGET_NAME, &cancellation)?;
    let WeightDelivery::Prepared(warm) = warm else {
        return Err(io::Error::other("warm graph lookup did not produce prepared bytes").into());
    };
    assert_eq!(warm.origin(), PreparedOrigin::Cache);
    assert_eq!(warm.shape(), [6]);
    assert_eq!(warm.bytes().as_slice(), expected_f16_bytes());
    Ok(())
}

#[test]
fn pipeline_reports_each_graph_node_with_kind_bytes_and_work_identity() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("qkv.safetensors");
    write_qkv_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = grouped_plan(&checkpoint, &SEMANTIC_ORDER, &cancellation)?;
    let engine = PreparationEngine::with_builtins()?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &cancellation)?;
    let pipeline = Pipeline::new(execution_limits(24, 24, 12))?;
    let events = Arc::new(Mutex::new(Vec::new()));
    let observer_events = Arc::clone(&events);
    let observer = with_operation_events(move |event: &ExecutionEvent| {
        observer_events
            .lock()
            .expect("operation telemetry lock must not be poisoned")
            .push(event.clone());
    });

    let report = materializer.execute(
        &pipeline,
        &mut |_ordinal, _item: PreparedItem<WeightDelivery>, _cancellation: &CancellationToken| {
            Ok(())
        },
        &observer,
    )?;
    drop(observer);
    let events = Arc::try_unwrap(events)
        .map_err(|_events| io::Error::other("pipeline retained the telemetry collector"))?
        .into_inner()
        .map_err(|_poisoned| io::Error::other("operation telemetry lock was poisoned"))?;
    let operations = events
        .into_iter()
        .filter(|event| matches!(event, ExecutionEvent::OperationFinished { .. }))
        .collect::<Vec<_>>();
    let [concat, cast] = operations.as_slice() else {
        return Err(io::Error::other(format!(
            "expected concat and cast operation events, got {}",
            operations.len()
        ))
        .into());
    };
    let concat_duration = assert_operation_event(
        concat,
        (
            Some(0),
            OperationLocation::GraphNode(NodeId::from_ordinal(0)),
            OperationKind::Concat,
            24,
            24,
            24,
        ),
    );
    let cast_duration = assert_operation_event(
        cast,
        (
            Some(0),
            OperationLocation::GraphNode(NodeId::from_ordinal(1)),
            OperationKind::Cast,
            24,
            12,
            12,
        ),
    );
    assert!(
        concat_duration.saturating_add(cast_duration)
            <= report.phase_duration(ExecutionPhase::Transform)
    );
    Ok(())
}

#[test]
fn input_reordering_changes_identity_and_delegation_preserves_order() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("qkv.safetensors");
    write_qkv_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let semantic = grouped_plan(&checkpoint, &SEMANTIC_ORDER, &cancellation)?;
    let reordered = grouped_plan(&checkpoint, &REORDERED_INPUTS, &cancellation)?;
    let engine = PreparationEngine::with_builtins()?;
    let semantic_materializer = Materializer::new(&checkpoint, &semantic, &engine, &cancellation)?;
    let reordered_materializer =
        Materializer::new(&checkpoint, &reordered, &engine, &cancellation)?;

    assert_ne!(semantic.id(), reordered.id());
    assert_ne!(
        semantic_materializer
            .prepared_cache_address(TARGET_NAME)?
            .ok_or_else(|| io::Error::other("semantic graph has no prepared-cache address"))?,
        reordered_materializer
            .prepared_cache_address(TARGET_NAME)?
            .ok_or_else(|| io::Error::other("reordered graph has no prepared-cache address"))?
    );

    let delegated = semantic_materializer
        .with_operation_execution(OperationExecution::Delegate)
        .materialize(TARGET_NAME, &cancellation)?;
    let WeightDelivery::Operation(handoff) = delegated else {
        return Err(
            io::Error::other("delegated graph did not produce an operation handoff").into(),
        );
    };
    assert_eq!(
        handoff
            .inputs()
            .iter()
            .map(|input| input.source().name().as_str())
            .collect::<Vec<_>>(),
        SEMANTIC_ORDER
    );
    assert_eq!(
        handoff
            .inputs()
            .iter()
            .map(|input| input.bytes().as_slice())
            .collect::<Vec<_>>(),
        [
            f32_bytes(&[1.0, 2.0]).as_slice(),
            f32_bytes(&[3.0, 4.0]).as_slice(),
            f32_bytes(&[5.0, 6.0]).as_slice(),
        ]
    );
    assert_eq!(handoff.resident_bytes(), 24);
    assert_eq!(
        Some(handoff.graph()),
        handoff.target().operation_graph(),
        "the handoff must retain the exact validated graph"
    );
    Ok(())
}

#[test]
fn each_one_byte_short_pipeline_budget_is_rejected_before_delivery() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("qkv.safetensors");
    write_qkv_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = grouped_plan(&checkpoint, &SEMANTIC_ORDER, &cancellation)?;
    let graph = plan.bindings()[0]
        .target()
        .operation_graph()
        .ok_or_else(|| io::Error::other("grouped target has no operation graph"))?;
    assert_eq!(graph.estimate_host_scratch_bytes()?, 24);
    let engine = PreparationEngine::with_builtins()?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &cancellation)?;
    let cases = [
        ("source", execution_limits(23, 24, 12)),
        ("scratch", execution_limits(24, 23, 12)),
        ("prepared", execution_limits(24, 24, 11)),
    ];

    for (budget, limits) in cases {
        let pipeline = Pipeline::new(limits)?;
        let mut delivery_calls = 0_u64;
        let error = materializer
            .execute(
                &pipeline,
                &mut |_ordinal,
                      _item: PreparedItem<WeightDelivery>,
                      _cancellation: &CancellationToken| {
                    delivery_calls += 1;
                    Ok(())
                },
                &NoopObserver,
            )
            .expect_err("a one-byte-short graph budget was accepted");

        assert_eq!(
            error.category(),
            ErrorCategory::ResourceLimit,
            "{budget} budget failed with the wrong category"
        );
        assert_eq!(
            error.message(),
            "a pipeline work item exceeds a configured byte budget"
        );
        assert_eq!(
            delivery_calls, 0,
            "{budget} budget failure reached the delivery sink"
        );
    }
    Ok(())
}

#[test]
fn reshape_only_graph_retains_its_source_view_and_has_no_prepared_cache_entry() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("q.safetensors");
    write_single_q_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = reshape_plan(&checkpoint, &cancellation)?;
    let graph = plan.bindings()[0]
        .target()
        .operation_graph()
        .ok_or_else(|| io::Error::other("reshape target has no operation graph"))?;
    assert_eq!(graph.output_input_alias(), Some(0));

    let Storage::Plain { span, .. } = plan.bindings()[0].source().storage() else {
        return Err(io::Error::other("reshape source is not plain storage").into());
    };
    let source = checkpoint.read_span(*span)?;
    let source_pointer = source.as_slice().as_ptr();
    let engine = PreparationEngine::with_builtins()?;
    let events = Arc::new(Mutex::new(Vec::new()));
    let observer_events = Arc::clone(&events);
    let observer = with_operation_events(move |event: &ExecutionEvent| {
        observer_events
            .lock()
            .expect("reshape telemetry lock must not be poisoned")
            .push(event.clone());
    });
    let execution = graph.execute_host_observed(
        std::slice::from_ref(&source),
        &engine,
        &cancellation,
        None,
        &observer,
    )?;
    drop(observer);
    assert_eq!(execution.output().as_slice().as_ptr(), source_pointer);
    assert_eq!(execution.output().as_slice(), source.as_slice());
    let events = Arc::try_unwrap(events)
        .map_err(|_events| io::Error::other("graph retained the reshape telemetry collector"))?
        .into_inner()
        .map_err(|_poisoned| io::Error::other("reshape telemetry lock was poisoned"))?;
    let operations = events
        .iter()
        .filter(|event| matches!(event, ExecutionEvent::OperationFinished { .. }))
        .collect::<Vec<_>>();
    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0],
        ExecutionEvent::OperationFinished {
            work_ordinal: None,
            location: OperationLocation::GraphNode(node),
            kind: OperationKind::Reshape,
            input_bytes: 8,
            output_bytes: 8,
            materialized_output_bytes: 0,
            ..
        } if *node == NodeId::from_ordinal(0)
    ));
    let disabled_execution = graph.execute_host_observed(
        std::slice::from_ref(&source),
        &engine,
        &cancellation,
        None,
        &OperationsDisabledObserver,
    )?;
    assert_eq!(
        disabled_execution.output().as_slice().as_ptr(),
        source_pointer
    );

    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &engine, &cancellation)?.with_cache(&cache);
    assert_eq!(
        materializer.prepared_cache_address(RESHAPE_TARGET_NAME)?,
        None
    );
    for _ in 0..2 {
        let WeightDelivery::Prepared(delivery) =
            materializer.materialize(RESHAPE_TARGET_NAME, &cancellation)?
        else {
            return Err(io::Error::other("reshape graph did not return prepared bytes").into());
        };
        assert_eq!(delivery.origin(), PreparedOrigin::OperationGraph);
        assert_eq!(delivery.bytes().as_slice(), source.as_slice());
    }
    Ok(())
}

fn grouped_plan(
    checkpoint: &Checkpoint,
    input_order: &[&str; 3],
    cancellation: &CancellationToken,
) -> Result<BindingPlan> {
    let sources = checkpoint
        .inventory()
        .iter()
        .map(SourceTensor::try_from)
        .collect::<Result<Vec<_>>>()?;
    BindingPlan::builder(plan_inputs(checkpoint.source_digests(cancellation)?))
        .sources(sources)
        .targets([grouped_target(input_order)?])
        .build()
}

fn grouped_target(input_order: &[&str; 3]) -> Result<TargetTensor> {
    let graph = concat_cast_graph(input_order)?;
    let output = graph.output_facts();
    let shape = output.logical_shape().to_vec();
    let storage_shape = output.storage_shape().to_vec();
    let logical_strides = output.logical_strides().to_vec();
    let representation = output.representation().clone();
    let output_size = output.byte_len();

    TargetTensor::builder(
        TensorName::parse(TARGET_NAME)?,
        Requirement::Required,
        shape,
        representation,
        output_size,
    )
    .storage_shape(storage_shape)
    .logical_strides(logical_strides)
    .operation_graph(graph)
    .build()
}

fn reshape_plan(checkpoint: &Checkpoint, cancellation: &CancellationToken) -> Result<BindingPlan> {
    let sources = checkpoint
        .inventory()
        .iter()
        .map(SourceTensor::try_from)
        .collect::<Result<Vec<_>>>()?;
    BindingPlan::builder(plan_inputs(checkpoint.source_digests(cancellation)?))
        .sources(sources)
        .targets([reshape_target()?])
        .build()
}

fn reshape_target() -> Result<TargetTensor> {
    let mut builder = OperationGraph::builder();
    let source = builder.add_input(
        TensorName::parse("q")?,
        TensorFacts::contiguous([2_u64], Representation::contiguous(DType::F32))?,
    )?;
    let reshape = Operation::Reshape(Reshape::new([1_u64, 2]));
    let implementation = reshape.builtin_implementation()?;
    let reshaped = builder.add_operation(implementation, reshape, [source])?;
    let graph = builder.build(reshaped[0])?;
    let output = graph.output_facts();

    TargetTensor::builder(
        TensorName::parse(RESHAPE_TARGET_NAME)?,
        Requirement::Required,
        output.logical_shape(),
        output.representation().clone(),
        output.byte_len(),
    )
    .storage_shape(output.storage_shape())
    .logical_strides(output.logical_strides())
    .storage_strides(output.storage_strides())
    .operation_graph(graph)
    .build()
}

fn concat_cast_graph(input_order: &[&str; 3]) -> Result<OperationGraph> {
    let mut builder = OperationGraph::builder();
    let f32 = Representation::contiguous(DType::F32);
    let mut inputs = Vec::with_capacity(input_order.len());
    for name in input_order {
        inputs.push(builder.add_input(
            TensorName::parse(name)?,
            TensorFacts::contiguous([2_u64], f32.clone())?,
        )?);
    }

    let concat = Operation::Concat(Concat::new(Axis::from_index(0)));
    let concat_implementation = concat.builtin_implementation()?;
    let concatenated = builder.add_operation(concat_implementation, concat, inputs)?;

    let f16 = Representation::contiguous(DType::F16);
    let prepare = Operation::Prepare(PlannedTransform::new(
        TransformSpec::new(builtin_contiguous_implementation()?, f32, f16),
        12,
    ));
    let prepare_implementation = prepare.builtin_implementation()?;
    let prepared = builder.add_operation(prepare_implementation, prepare, concatenated)?;
    builder.build(prepared[0])
}

fn plan_inputs(source_digests: Box<[ContentDigest]>) -> PlanInputs {
    PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash(
            "issue-11-manifest-v1",
            [b"qkv-fixture"],
        )),
        SelectionId::from_digest(ContentDigest::hash(
            "issue-11-selection-v1",
            [b"assembled-qkv"],
        )),
        ContractId::from_digest(ContentDigest::hash(
            "issue-11-contract-v1",
            [b"concat-f32-cast-f16"],
        )),
        BackendId::from_digest(ContentDigest::hash(
            "issue-11-backend-v1",
            [b"host-operations"],
        )),
        source_digests,
    )
}

const fn execution_limits(
    source_bytes: u64,
    scratch_bytes: u64,
    prepared_bytes: u64,
) -> ExecutionLimits {
    ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes,
        scratch_bytes,
        prepared_bytes,
    }
}

fn expected_f16_bytes() -> Vec<u8> {
    [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn assert_operation_event(
    event: &ExecutionEvent,
    expected: (Option<u64>, OperationLocation, OperationKind, u64, u64, u64),
) -> std::time::Duration {
    let ExecutionEvent::OperationFinished {
        work_ordinal,
        location,
        kind,
        duration,
        input_bytes,
        output_bytes,
        materialized_output_bytes,
    } = event
    else {
        panic!("expected an operation event");
    };
    assert_eq!(
        (
            *work_ordinal,
            *location,
            *kind,
            *input_bytes,
            *output_bytes,
            *materialized_output_bytes,
        ),
        expected
    );
    *duration
}

fn write_qkv_fixture(path: &std::path::Path) -> io::Result<()> {
    let header = br#"{"v":{"dtype":"F32","shape":[2],"data_offsets":[16,24]},"q":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"k":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let payload = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let header_len = u64::try_from(header.len()).map_err(io::Error::other)?;
    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header)?;
    file.write_all(&payload)?;
    file.sync_all()
}

fn write_single_q_fixture(path: &std::path::Path) -> io::Result<()> {
    let header = br#"{"q":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let payload = f32_bytes(&[1.0, 2.0]);
    let header_len = u64::try_from(header.len()).map_err(io::Error::other)?;
    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header)?;
    file.write_all(&payload)?;
    file.sync_all()
}
