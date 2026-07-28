//! Public-API conformance for the complete inventory-to-delivery path.

use std::error::Error as StdError;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use model_weights::cache::{Cache, CacheCompatibility, CacheKey, CacheLookup, CacheNamespace};
use model_weights::identity::{BackendId, ContentDigest, ContractId, ManifestId, SelectionId};
use model_weights::limits::ExecutionLimits;
use model_weights::pipeline::{Pipeline, PrepareContext, PreparedItem, ResourceWeights, WorkItem};
use model_weights::plan::{
    Binding, BindingPlan, ExtraSourcePolicy, PlanInputs, PlannedTransform, Requirement,
    SourceTensor, TargetTensor, TensorName,
};
use model_weights::prepare::{
    PreparationEngine, Representation, TransformSpec, builtin_contiguous_implementation,
};
use model_weights::telemetry::{ExecutionEvent, ExecutionPhase};
use model_weights::tensor::DType;
use model_weights::{CancellationToken, Checkpoint, Error, ErrorCategory, Result};
use tempfile::TempDir;

#[derive(Debug)]
struct DeliveredTensor {
    name: TensorName,
    bytes: Vec<u8>,
    cache_hit: bool,
}

#[derive(Debug)]
struct PassResult {
    delivered: Vec<(u64, DeliveredTensor)>,
    source_reads: Vec<TensorName>,
    events: Vec<ExecutionEvent>,
    report: model_weights::telemetry::ExecutionReport,
}

#[test]
fn selected_plan_runs_cold_and_warm_through_cache_prepare_and_pipeline()
-> std::result::Result<(), Box<dyn StdError>> {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("model.safetensors");
    write_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source_digests = checkpoint.source_digests(&cancellation)?;
    let inputs = plan_inputs(source_digests.clone());
    let sources = checkpoint
        .inventory()
        .iter()
        .map(SourceTensor::try_from)
        .collect::<Result<Vec<_>>>()?;
    let target = selected_target()?;

    let forward = BindingPlan::builder(inputs.clone())
        .sources(sources.clone())
        .targets([target.clone()])
        .extra_source_policy(ExtraSourcePolicy::Allow)
        .build()?;
    let reverse = BindingPlan::builder(inputs)
        .sources(sources.into_iter().rev())
        .targets([target])
        .extra_source_policy(ExtraSourcePolicy::Allow)
        .build()?;

    assert_eq!(forward.id(), reverse.id());
    assert_eq!(forward.to_canonical_json()?, reverse.to_canonical_json()?);
    assert_eq!(forward.bindings().len(), 1);
    assert_eq!(forward.bindings()[0].source().name().as_str(), "selected");
    assert_eq!(
        forward
            .unused_sources()
            .iter()
            .map(TensorName::as_str)
            .collect::<Vec<_>>(),
        ["unused.variant"]
    );

    let cache = Cache::open(temporary.path().join("cache"))?;
    let compatibility = CacheCompatibility::prepared(
        1,
        ContentDigest::hash(
            "conformance-transform-v1",
            [forward.id().digest().as_bytes()],
        ),
        forward.inputs().backend().digest(),
    );
    let engine = PreparationEngine::with_builtins()?;

    let cold = execute_pass(&checkpoint, &forward, &cache, &compatibility, &engine)?;
    assert_eq!(
        cold.source_reads
            .iter()
            .map(TensorName::as_str)
            .collect::<Vec<_>>(),
        ["selected"]
    );
    assert_delivery(&cold, false);
    assert_phase(&cold.events, ExecutionPhase::CacheLookup);
    assert_phase(&cold.events, ExecutionPhase::SourceRead);
    assert_phase(&cold.events, ExecutionPhase::Transform);
    assert_report_is_bounded(&cold.report);

    let warm = execute_pass(&checkpoint, &forward, &cache, &compatibility, &engine)?;
    assert!(warm.source_reads.is_empty());
    assert_delivery(&warm, true);
    assert_phase(&warm.events, ExecutionPhase::CacheLookup);
    assert!(
        !warm.events.iter().any(|event| matches!(
            event,
            ExecutionEvent::PhaseFinished {
                phase: ExecutionPhase::SourceRead | ExecutionPhase::Transform,
                ..
            }
        )),
        "a warm cache hit must not reread or reprepare source bytes"
    );
    assert_report_is_bounded(&warm.report);
    Ok(())
}

#[test]
fn pre_cancelled_end_to_end_pipeline_never_starts_preparation() -> Result<()> {
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let pipeline = Pipeline::with_cancellation(execution_limits(), cancellation)?;
    let events = Arc::new(Mutex::new(Vec::new()));
    let observer_events = Arc::clone(&events);
    let observer = move |event: &ExecutionEvent| {
        observer_events
            .lock()
            .expect("telemetry collector lock must not be poisoned")
            .push(event.clone());
    };
    let consumed_work_items = Arc::new(AtomicUsize::new(0));
    let iterator_counter = Arc::clone(&consumed_work_items);
    let work = [WorkItem::new(0, (), ResourceWeights::new(8, 4, 4))]
        .into_iter()
        .inspect(move |_item| {
            iterator_counter.fetch_add(1, Ordering::SeqCst);
        });
    let mut sink_calls = 0_u64;
    let error = pipeline
        .execute(
            work,
            |_value, _context: &mut PrepareContext<'_>| {
                panic!("pre-cancelled preparation callback must not run")
            },
            &mut |_ordinal, _item: PreparedItem<()>, _token: &CancellationToken| {
                sink_calls += 1;
                Ok(())
            },
            &observer,
        )
        .expect_err("pre-cancelled execution must fail");

    assert!(error.is_cancelled());
    assert_eq!(consumed_work_items.load(Ordering::SeqCst), 0);
    assert_eq!(sink_calls, 0);
    let events = events
        .lock()
        .expect("telemetry collector lock must not be poisoned");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ExecutionEvent::Cancelled))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ExecutionEvent::Finished { success: false, .. }))
    );
    Ok(())
}

fn execute_pass(
    checkpoint: &Checkpoint,
    plan: &BindingPlan,
    cache: &Cache,
    compatibility: &CacheCompatibility,
    engine: &PreparationEngine,
) -> Result<PassResult> {
    let work = plan
        .bindings()
        .iter()
        .enumerate()
        .map(|(ordinal, binding)| {
            let ordinal = u64::try_from(ordinal).expect("test binding ordinal must fit in a u64");
            WorkItem::new(
                ordinal,
                binding.clone(),
                ResourceWeights::new(
                    binding.source().storage().span().len(),
                    binding.target().output_size(),
                    binding.target().output_size(),
                ),
            )
        })
        .collect::<Vec<_>>();
    let pipeline = Pipeline::new(execution_limits())?;
    let source_reads = Arc::new(Mutex::new(Vec::new()));
    let callback_source_reads = Arc::clone(&source_reads);
    let events = Arc::new(Mutex::new(Vec::new()));
    let observer_events = Arc::clone(&events);
    let observer = move |event: &ExecutionEvent| {
        observer_events
            .lock()
            .expect("telemetry collector lock must not be poisoned")
            .push(event.clone());
    };
    let mut delivered = Vec::new();
    let seams = ConformanceSeams {
        checkpoint,
        plan,
        cache,
        compatibility,
        engine,
        source_reads: &callback_source_reads,
    };
    let report = pipeline.execute(
        work,
        |binding: Binding, context: &mut PrepareContext<'_>| {
            prepare_binding(&binding, context, &seams)
        },
        &mut |ordinal, item: PreparedItem<DeliveredTensor>, token: &CancellationToken| {
            token.check()?;
            delivered.push((ordinal, item.into_value()));
            Ok(())
        },
        &observer,
    )?;
    drop(callback_source_reads);
    drop(observer);
    let source_reads = Arc::try_unwrap(source_reads)
        .expect("pipeline must release the source-read collector")
        .into_inner()
        .expect("source-read collector lock must not be poisoned");
    let events = Arc::try_unwrap(events)
        .expect("pipeline must release the telemetry collector")
        .into_inner()
        .expect("telemetry collector lock must not be poisoned");
    Ok(PassResult {
        delivered,
        source_reads,
        events,
        report,
    })
}

struct ConformanceSeams<'a> {
    checkpoint: &'a Checkpoint,
    plan: &'a BindingPlan,
    cache: &'a Cache,
    compatibility: &'a CacheCompatibility,
    engine: &'a PreparationEngine,
    source_reads: &'a Mutex<Vec<TensorName>>,
}

fn prepare_binding(
    binding: &Binding,
    context: &mut PrepareContext<'_>,
    seams: &ConformanceSeams<'_>,
) -> Result<PreparedItem<DeliveredTensor>> {
    let key = prepared_key(seams.plan, binding, seams.compatibility);
    let lookup = context.measure(
        ExecutionPhase::CacheLookup,
        binding.target().output_size(),
        |token| {
            seams.cache.lookup_with_cancellation(
                CacheNamespace::Prepared,
                key,
                seams.compatibility,
                token,
            )
        },
    )?;
    let (bytes, cache_hit) = match lookup {
        CacheLookup::Hit(entry) => (read_cache_entry(entry)?, true),
        CacheLookup::Miss(_) => {
            seams
                .source_reads
                .lock()
                .expect("source-read collector lock must not be poisoned")
                .push(binding.source().name().clone());
            let source = context.measure(
                ExecutionPhase::SourceRead,
                binding.source().storage().span().len(),
                |token| {
                    token.check()?;
                    seams
                        .checkpoint
                        .read_span(binding.source().storage().span())
                },
            )?;
            let prepared = context.measure(
                ExecutionPhase::Transform,
                binding.target().output_size(),
                |token| {
                    seams.engine.prepare_chain_with_cancellation(
                        binding.target().transforms(),
                        binding.target().shape(),
                        &source,
                        token,
                    )
                },
            )?;
            let publication = seams.cache.publish_bytes(
                CacheNamespace::Prepared,
                key,
                seams.compatibility,
                prepared.as_slice(),
            )?;
            (read_cache_entry(publication.into_entry())?, false)
        }
        _ => {
            return Err(Error::from_category(
                ErrorCategory::Cache,
                "conformance test does not recognize this cache lookup outcome",
            ));
        }
    };
    let byte_len = u64::try_from(bytes.len()).map_err(|error| {
        Error::from_category_with_source(
            ErrorCategory::ResourceLimit,
            "prepared byte length does not fit u64",
            error,
        )
    })?;
    Ok(PreparedItem::new(
        DeliveredTensor {
            name: binding.target().name().clone(),
            bytes,
            cache_hit,
        },
        byte_len,
    ))
}

fn read_cache_entry(entry: model_weights::cache::CacheEntry) -> Result<Vec<u8>> {
    let capacity = usize::try_from(entry.info().payload_len()).map_err(|error| {
        Error::from_category_with_source(
            ErrorCategory::ResourceLimit,
            "cache payload length does not fit usize",
            error,
        )
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    entry
        .into_payload()
        .read_to_end(&mut bytes)
        .map_err(|error| {
            Error::from_category_with_source(
                ErrorCategory::Io,
                "failed to read validated cache payload",
                error,
            )
        })?;
    Ok(bytes)
}

fn prepared_key(
    plan: &BindingPlan,
    binding: &Binding,
    compatibility: &CacheCompatibility,
) -> CacheKey {
    let plan_digest = plan.id().digest();
    CacheKey::derive(
        CacheNamespace::Prepared,
        compatibility,
        [
            plan_digest.as_bytes().as_slice(),
            binding.target().name().as_str().as_bytes(),
        ],
    )
}

fn selected_target() -> Result<TargetTensor> {
    let implementation = builtin_contiguous_implementation()?;
    let transform = TransformSpec::new(
        implementation,
        Representation::contiguous(DType::F32),
        Representation::contiguous(DType::F16),
    );
    TargetTensor::builder(
        TensorName::parse("selected")?,
        Requirement::Required,
        [2],
        Representation::contiguous(DType::F16),
        4,
    )
    .transforms([PlannedTransform::new(transform, 4)])
    .build()
}

fn plan_inputs(source_digests: Box<[ContentDigest]>) -> PlanInputs {
    PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash(
            "conformance-manifest-v1",
            [b"two-tensor-fixture"],
        )),
        SelectionId::from_digest(ContentDigest::hash(
            "conformance-selection-v1",
            [b"selected"],
        )),
        ContractId::from_digest(ContentDigest::hash(
            "conformance-contract-v1",
            [b"selected-f16"],
        )),
        BackendId::from_digest(ContentDigest::hash(
            "conformance-backend-v1",
            [b"host-contiguous"],
        )),
        source_digests,
    )
}

const fn execution_limits() -> ExecutionLimits {
    ExecutionLimits {
        workers: 2,
        max_work_items: 2,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 8,
        scratch_bytes: 4,
        prepared_bytes: 4,
    }
}

fn assert_delivery(pass: &PassResult, cache_hit: bool) {
    assert_eq!(pass.delivered.len(), 1);
    let (ordinal, tensor) = &pass.delivered[0];
    assert_eq!(*ordinal, 0);
    assert_eq!(tensor.name.as_str(), "selected");
    assert_eq!(tensor.bytes, [0x00, 0x3c, 0x00, 0xc0]);
    assert_eq!(tensor.cache_hit, cache_hit);
    assert_eq!(pass.report.counters().submitted(), 1);
    assert_eq!(pass.report.counters().prepared(), 1);
    assert_eq!(pass.report.counters().delivered(), 1);
    assert_eq!(pass.report.counters().delivered_bytes(), 4);
}

fn assert_phase(events: &[ExecutionEvent], phase: ExecutionPhase) {
    assert!(events.iter().any(|event| matches!(
        event,
        ExecutionEvent::PhaseFinished {
            phase: observed,
            ..
        } if *observed == phase
    )));
}

fn assert_report_is_bounded(report: &model_weights::telemetry::ExecutionReport) {
    assert!(report.peak_bytes().source() <= 8);
    assert!(report.peak_bytes().scratch() <= 4);
    assert!(report.peak_bytes().prepared() <= 4);
    assert!(report.peak_delivery_queue_depth() <= 1);
}

fn write_fixture(path: &std::path::Path) -> std::io::Result<()> {
    let header = br#"{"selected":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"unused.variant":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let mut file = File::create(path)?;
    file.write_all(
        &u64::try_from(header.len())
            .expect("fixture header length must fit u64")
            .to_le_bytes(),
    )?;
    file.write_all(header)?;
    for value in [1.0_f32, -2.0, 100.0, 200.0] {
        file.write_all(&value.to_le_bytes())?;
    }
    file.sync_all()
}
