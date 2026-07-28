//! Public materialization, handoff, and plan-cache API coverage.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use model_weights::cache::{Cache, CacheLookup, CacheNamespace, CacheValidation};
use model_weights::identity::{
    BackendId, ContentDigest, ContractId, ImplementationId, ManifestId, SelectionId, StableName,
};
use model_weights::limits::ExecutionLimits;
use model_weights::materialize::{
    BindingPlanCacheLookup, Materializer, PreparedOrigin, WeightDelivery, binding_source_content,
    lookup_binding_plan, publish_binding_plan,
};
use model_weights::overlay::{
    AliasTable, BaseTensor, CompositionMode, ConflictPolicy, FiniteScale, OverlayLayer,
    OverlayOperation, OverlayPlan,
};
use model_weights::pipeline::{Pipeline, PreparedItem};
use model_weights::plan::{
    BindingPlan, ConversionRecipe, ExtraSourcePolicy, PlanInputs, PlannedTransform, RecipeInput,
    RecipeStep, Requirement, SourceTensor, TargetTensor, TensorName,
};
use model_weights::prepare::{
    OutputStrategy, PreparationEngine, PreparationProvider, PrepareRequest, ProviderRegistry,
    Representation, TransformSpec, builtin_contiguous_implementation,
};
use model_weights::quantization::{QuantizedRoute, RouteCapability, Storage};
use model_weights::telemetry::{
    ExecutionEvent, ExecutionPhase, ExecutionReport, NoopObserver, OperationKind,
    OperationLocation, with_operation_events,
};
use model_weights::tensor::{DType, FileId, SourceSpan};
use model_weights::{CancellationToken, Checkpoint, Error, ErrorCategory, Result};
use tempfile::TempDir;

type TestResult = std::result::Result<(), Box<dyn StdError>>;

#[test]
fn materializer_rejects_a_plan_with_different_source_digests() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let source = inventory_source(&checkpoint, "weight")?;
    let wrong_digest = ContentDigest::hash("materializer-test-wrong-source", [b"not-the-file"]);
    let plan = BindingPlan::builder(plan_inputs([wrong_digest].into()))
        .sources([source])
        .targets([plain_identity_target()?])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;

    let result = Materializer::new(&checkpoint, &plan, &engine, &CancellationToken::new());

    let Err(error) = result else {
        return Err(
            io::Error::other("materializer accepted a plan with the wrong source digest").into(),
        );
    };
    assert_eq!(error.category(), ErrorCategory::Integrity);
    assert_eq!(
        error.message(),
        "binding plan source digests differ from the opened checkpoint"
    );
    Ok(())
}

#[test]
fn materializer_rejects_an_unused_source_absent_from_the_inventory() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source_digests = checkpoint.source_digests(&cancellation)?;
    let selected = inventory_source(&checkpoint, "weight")?;
    let absent = SourceTensor::new(
        TensorName::parse("absent.unused")?,
        selected.shape(),
        selected.storage().clone(),
    )?;
    let plan = BindingPlan::builder(plan_inputs(source_digests))
        .sources([selected, absent])
        .targets([plain_identity_target()?])
        .extra_source_policy(ExtraSourcePolicy::Allow)
        .build()?;
    let engine = PreparationEngine::with_builtins()?;

    let result = Materializer::new(&checkpoint, &plan, &engine, &cancellation);

    let Err(error) = result else {
        return Err(io::Error::other(
            "materializer accepted plan inventory evidence absent from the checkpoint",
        )
        .into());
    };
    assert_eq!(error.category(), ErrorCategory::Binding);
    assert_eq!(
        error.message(),
        "binding plan source is absent from the opened checkpoint"
    );
    Ok(())
}

#[test]
fn materializer_rejects_a_plan_that_omits_checkpoint_inventory_evidence() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("two-weights.safetensors");
    write_two_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source = inventory_source(&checkpoint, "a")?;
    let target = TargetTensor::builder(
        TensorName::parse("a")?,
        Requirement::Required,
        [2_u64],
        Representation::contiguous(DType::F32),
        8,
    )
    .build()?;
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([source])
        .targets([target])
        .extra_source_policy(ExtraSourcePolicy::Allow)
        .build()?;
    let engine = PreparationEngine::with_builtins()?;

    let result = Materializer::new(&checkpoint, &plan, &engine, &cancellation);

    let Err(error) = result else {
        return Err(io::Error::other(
            "materializer accepted a plan that omitted checkpoint tensor b",
        )
        .into());
    };
    assert_eq!(error.category(), ErrorCategory::Integrity);
    assert_eq!(
        error.message(),
        "binding plan source inventory omits an opened checkpoint tensor"
    );
    Ok(())
}

#[test]
fn materializer_work_limit_precedes_address_and_preparation_work() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("two-weights.safetensors");
    write_two_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let provider_id = implementation("never-run-over-limit")?;
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([
            inventory_source(&checkpoint, "a")?,
            inventory_source(&checkpoint, "b")?,
        ])
        .targets([
            copying_transform_target("a", &provider_id)?,
            copying_transform_target("b", &provider_id)?,
        ])
        .build()?;
    let validated = Arc::new(AtomicBool::new(false));
    let executed = Arc::new(AtomicBool::new(false));
    let mut registry = ProviderRegistry::new();
    registry.register(NeverRunProvider {
        implementation: provider_id,
        validated: Arc::clone(&validated),
        executed: Arc::clone(&executed),
    })?;
    let engine = PreparationEngine::new(registry);
    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &engine, &cancellation)?.with_cache(&cache);
    let pipeline_cancellation = CancellationToken::new();
    pipeline_cancellation.cancel();
    let pipeline = Pipeline::with_cancellation(
        ExecutionLimits {
            max_work_items: 1,
            prepared_bytes: 8,
            ..execution_limits()
        },
        pipeline_cancellation,
    )?;
    let mut sink_calls = 0_u64;

    let error = materializer
        .execute(
            &pipeline,
            &mut |_ordinal,
                  _item: PreparedItem<WeightDelivery>,
                  _cancellation: &CancellationToken| {
                sink_calls += 1;
                Ok(())
            },
            &NoopObserver,
        )
        .expect_err("materializer accepted more bindings than the work-item limit");

    assert_eq!(error.category(), ErrorCategory::ResourceLimit);
    assert_eq!(
        error.message(),
        "pipeline work-item count exceeds the configured limit"
    );
    assert!(!validated.load(Ordering::SeqCst));
    assert!(!executed.load(Ordering::SeqCst));
    assert_eq!(sink_calls, 0);
    Ok(())
}

#[test]
fn pre_cancelled_materializer_skips_no_cache_resource_planning() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let setup_cancellation = CancellationToken::new();
    let representation = Representation::contiguous(DType::F32);
    let transform = TransformSpec::new(
        builtin_contiguous_implementation()?,
        representation.clone(),
        representation.clone(),
    );
    let target = TargetTensor::builder(
        TensorName::parse("weight")?,
        Requirement::Required,
        [2_u64],
        representation,
        8,
    )
    .transforms([
        PlannedTransform::new(transform.clone(), 8).with_scratch_bytes(u64::MAX),
        PlannedTransform::new(transform, 8),
    ])
    .build()?;
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&setup_cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([target])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &setup_cancellation)?;
    let execution_cancellation = CancellationToken::new();
    execution_cancellation.cancel();
    let pipeline = Pipeline::with_cancellation(execution_limits(), execution_cancellation)?;

    let error = materializer
        .execute(
            &pipeline,
            &mut |_ordinal,
                  _item: PreparedItem<WeightDelivery>,
                  _cancellation: &CancellationToken| {
                panic!("pre-cancelled materializer must not deliver")
            },
            &NoopObserver,
        )
        .expect_err("pre-cancelled materializer execution must fail");

    assert_eq!(error.category(), ErrorCategory::Cancelled);
    Ok(())
}

#[test]
fn materializer_hashes_an_ordinary_source_once_per_checkpoint_handle() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let file_bytes = checkpoint_path.metadata()?.len();
    let checkpoint_a = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint_a.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint_a, "weight")?])
        .targets([plain_identity_target()?])
        .build()?;
    let checkpoint_b = Checkpoint::open(&checkpoint_path)?;
    let engine = PreparationEngine::with_builtins()?;
    let first_events = Arc::new(Mutex::new(Vec::new()));
    let first_observer_events = Arc::clone(&first_events);
    let first_observer = move |event: &ExecutionEvent| {
        first_observer_events
            .lock()
            .expect("test telemetry lock must not be poisoned")
            .push(event.clone());
    };

    let _first = Materializer::new_with_observer(
        &checkpoint_b,
        &plan,
        &engine,
        &cancellation,
        &first_observer,
    )?;

    let first_hash_bytes = first_events
        .lock()
        .map_err(|_poisoned| io::Error::other("test telemetry lock was poisoned"))?
        .iter()
        .filter_map(|event| match event {
            ExecutionEvent::PhaseFinished {
                phase: ExecutionPhase::Hashing,
                ordinal: None,
                bytes,
                ..
            } => Some(*bytes),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(first_hash_bytes, [file_bytes]);

    let second_events = Arc::new(Mutex::new(Vec::new()));
    let second_observer_events = Arc::clone(&second_events);
    let second_observer = move |event: &ExecutionEvent| {
        second_observer_events
            .lock()
            .expect("test telemetry lock must not be poisoned")
            .push(event.clone());
    };
    let _second = Materializer::new_with_observer(
        &checkpoint_b,
        &plan,
        &engine,
        &cancellation,
        &second_observer,
    )?;
    assert!(
        !second_events
            .lock()
            .map_err(|_poisoned| io::Error::other("test telemetry lock was poisoned"))?
            .iter()
            .any(|event| matches!(
                event,
                ExecutionEvent::PhaseStarted {
                    phase: ExecutionPhase::Hashing,
                    ..
                } | ExecutionEvent::PhaseFinished {
                    phase: ExecutionPhase::Hashing,
                    ..
                }
            )),
        "a cached source digest emitted redundant hashing telemetry"
    );
    Ok(())
}

#[test]
fn canonical_binding_plan_publishes_and_round_trips_through_plan_cache() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([plain_identity_target()?])
        .build()?;
    let cache = Cache::open(temporary.path().join("cache"))?;

    let _publication = publish_binding_plan(&cache, &plan, &cancellation)?;
    let lookup = lookup_binding_plan(&cache, plan.id(), CacheValidation::Full, &cancellation)?;

    let BindingPlanCacheLookup::Hit(cached) = lookup else {
        return Err(io::Error::other("published binding plan was not a cache hit").into());
    };
    assert_eq!(cached.to_canonical_json()?, plan.to_canonical_json()?);
    Ok(())
}

#[test]
fn transformed_plain_weight_executes_cold_then_warm_through_materializer() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([plain_transform_target()?])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &engine, &cancellation)?.with_cache(&cache);
    let pipeline = Pipeline::new(execution_limits())?;

    let (cold_origin, cold_bytes, cold_report) = execute_one(&materializer, &pipeline)?;
    let (warm_origin, warm_bytes, warm_report) = execute_one(&materializer, &pipeline)?;

    assert_eq!(
        (
            cold_origin,
            cold_bytes,
            cold_report.counters().delivered_bytes()
        ),
        (PreparedOrigin::Transform, vec![0x00, 0x3c, 0x00, 0xc0], 4)
    );
    assert_eq!(
        (
            warm_origin,
            warm_bytes,
            warm_report.counters().delivered_bytes()
        ),
        (PreparedOrigin::Cache, vec![0x00, 0x3c, 0x00, 0xc0], 4)
    );
    Ok(())
}

#[test]
fn direct_materialization_reports_source_identity_and_each_planned_cast() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source = inventory_source(&checkpoint, "weight")?;
    let source_digests = checkpoint.source_digests(&cancellation)?;
    let identity_plan = BindingPlan::builder(plan_inputs(source_digests.clone()))
        .sources([source.clone()])
        .targets([plain_identity_target()?])
        .build()?;
    let cast_plan = BindingPlan::builder(plan_inputs(source_digests))
        .sources([source])
        .targets([plain_transform_target()?])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;
    let identity_materializer =
        Materializer::new(&checkpoint, &identity_plan, &engine, &cancellation)?;
    let cast_materializer = Materializer::new(&checkpoint, &cast_plan, &engine, &cancellation)?;
    let events = Arc::new(Mutex::new(Vec::new()));
    let observer_events = Arc::clone(&events);
    let observer = with_operation_events(move |event: &ExecutionEvent| {
        observer_events
            .lock()
            .expect("direct operation telemetry lock must not be poisoned")
            .push(event.clone());
    });

    let _identity =
        identity_materializer.materialize_with_observer("weight", &cancellation, &observer)?;
    let _cast = cast_materializer.materialize_with_observer("weight", &cancellation, &observer)?;
    drop(observer);
    let events = Arc::try_unwrap(events)
        .map_err(|_events| io::Error::other("materializer retained the telemetry collector"))?
        .into_inner()
        .map_err(|_poisoned| io::Error::other("direct operation telemetry lock was poisoned"))?;
    let operations = events
        .iter()
        .filter(|event| matches!(event, ExecutionEvent::OperationFinished { .. }))
        .collect::<Vec<_>>();
    assert_eq!(operations.len(), 2);
    assert!(matches!(
        operations[0],
        ExecutionEvent::OperationFinished {
            work_ordinal: None,
            location: OperationLocation::Binding,
            kind: OperationKind::Identity,
            input_bytes: 8,
            output_bytes: 8,
            materialized_output_bytes: 0,
            ..
        }
    ));
    assert!(matches!(
        operations[1],
        ExecutionEvent::OperationFinished {
            work_ordinal: None,
            location: OperationLocation::PlannedTransform { index: 0 },
            kind: OperationKind::Cast,
            input_bytes: 8,
            output_bytes: 4,
            materialized_output_bytes: 4,
            ..
        }
    ));
    Ok(())
}

#[test]
fn prepared_address_ignores_unrelated_global_plan_identity() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source_digests = checkpoint.source_digests(&cancellation)?;
    let source = inventory_source(&checkpoint, "weight")?;
    let first = BindingPlan::builder(plan_inputs_with_selection(
        source_digests.clone(),
        b"component-a",
    ))
    .sources([source.clone()])
    .targets([plain_transform_target()?])
    .build()?;
    let second = BindingPlan::builder(plan_inputs_with_selection(source_digests, b"component-b"))
        .sources([source])
        .targets([plain_transform_target()?])
        .build()?;
    assert_ne!(first.id(), second.id());

    let engine = PreparationEngine::with_builtins()?;
    let first_materializer = Materializer::new(&checkpoint, &first, &engine, &cancellation)?;
    let second_materializer = Materializer::new(&checkpoint, &second, &engine, &cancellation)?;

    assert_eq!(
        first_materializer.prepared_cache_address("weight")?,
        second_materializer.prepared_cache_address("weight")?,
        "global selection changes must not invalidate identical target bytes"
    );
    Ok(())
}

#[test]
fn provider_scratch_changes_prepared_cache_identity() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source_digests = checkpoint.source_digests(&cancellation)?;
    let source = inventory_source(&checkpoint, "weight")?;
    let provider_id = implementation("scratch-cache-identity")?;
    let zero_scratch = BindingPlan::builder(plan_inputs(source_digests.clone()))
        .sources([source.clone()])
        .targets([three_stage_scratch_transform_target(&provider_id, 0)?])
        .build()?;
    let five_scratch = BindingPlan::builder(plan_inputs(source_digests))
        .sources([source])
        .targets([three_stage_scratch_transform_target(&provider_id, 5)?])
        .build()?;
    let mut registry = ProviderRegistry::new();
    registry.register(ScratchCopyProvider {
        implementation: provider_id,
        scratch_bytes: 5,
    })?;
    let engine = PreparationEngine::new(registry);
    let zero_materializer = Materializer::new(&checkpoint, &zero_scratch, &engine, &cancellation)?;
    let five_materializer = Materializer::new(&checkpoint, &five_scratch, &engine, &cancellation)?;

    assert_ne!(zero_scratch.id(), five_scratch.id());
    assert_ne!(
        zero_materializer.prepared_cache_address("weight")?,
        five_materializer.prepared_cache_address("weight")?
    );
    Ok(())
}

#[test]
fn full_validation_replaces_a_same_length_corrupt_prepared_entry() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([plain_transform_target()?])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &engine, &cancellation)?.with_cache(&cache);
    let address = materializer
        .prepared_cache_address("weight")?
        .ok_or_else(|| io::Error::other("transformed binding had no cache address"))?;

    let cold = materializer.materialize("weight", &cancellation)?;
    let WeightDelivery::Prepared(cold) = cold else {
        return Err(io::Error::other("cold transform did not return prepared bytes").into());
    };
    assert_eq!(cold.origin(), PreparedOrigin::Transform);
    let lookup = cache.lookup_with_validation(
        CacheNamespace::Prepared,
        address.key(),
        address.compatibility(),
        CacheValidation::TrustedMetadata,
    )?;
    let CacheLookup::Hit(entry) = lookup else {
        return Err(io::Error::other("cold transform did not populate the cache").into());
    };
    let payload_path = entry.info().payload_path().to_owned();
    drop(entry);
    fs::write(payload_path, [0xff_u8; 4])?;

    let repaired = materializer.materialize("weight", &cancellation)?;
    let WeightDelivery::Prepared(repaired) = repaired else {
        return Err(io::Error::other("repaired transform did not return prepared bytes").into());
    };
    assert_eq!(repaired.origin(), PreparedOrigin::Transform);
    assert_eq!(repaired.bytes().as_slice(), [0x00, 0x3c, 0x00, 0xc0]);
    assert!(matches!(
        cache.lookup_with_validation(
            CacheNamespace::Prepared,
            address.key(),
            address.compatibility(),
            CacheValidation::Full,
        )?,
        CacheLookup::Hit(_)
    ));
    Ok(())
}

#[test]
fn three_stage_transform_requires_the_sum_of_adjacent_intermediates() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([three_stage_transform_target()?])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &cancellation)?;
    let below_peak = Pipeline::new(ExecutionLimits {
        scratch_bytes: 11,
        ..execution_limits()
    })?;

    let result = execute_one(&materializer, &below_peak);

    let Err(error) = result else {
        return Err(io::Error::other(
            "pipeline accepted less scratch than two adjacent intermediates require",
        )
        .into());
    };
    assert_eq!(error.category(), ErrorCategory::ResourceLimit);

    let exact_peak = Pipeline::new(ExecutionLimits {
        scratch_bytes: 12,
        ..execution_limits()
    })?;
    let (origin, bytes, _report) = execute_one(&materializer, &exact_peak)?;
    assert_eq!(origin, PreparedOrigin::Transform);
    assert_eq!(bytes, [0x00, 0x3c, 0x00, 0xc0]);
    Ok(())
}

#[test]
fn provider_workspace_and_adjacent_intermediates_share_one_scratch_budget() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let provider_id = implementation("scratch-copy")?;
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([three_stage_scratch_transform_target(&provider_id, 5)?])
        .build()?;
    let mut registry = ProviderRegistry::new();
    registry.register(ScratchCopyProvider {
        implementation: provider_id,
        scratch_bytes: 5,
    })?;
    let engine = PreparationEngine::new(registry);
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &cancellation)?;
    let below_peak = Pipeline::new(ExecutionLimits {
        scratch_bytes: 20,
        prepared_bytes: 8,
        ..execution_limits()
    })?;

    let result = execute_one(&materializer, &below_peak);

    let Err(error) = result else {
        return Err(io::Error::other(
            "pipeline accepted less than provider workspace plus adjacent intermediates",
        )
        .into());
    };
    assert_eq!(error.category(), ErrorCategory::ResourceLimit);

    let exact_peak = Pipeline::new(ExecutionLimits {
        scratch_bytes: 21,
        prepared_bytes: 8,
        ..execution_limits()
    })?;
    let (origin, bytes, _report) = execute_one(&materializer, &exact_peak)?;
    assert_eq!(origin, PreparedOrigin::Transform);
    assert_eq!(bytes, [0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0]);
    Ok(())
}

#[test]
fn lazy_add_overlay_handoffs_then_reuses_provider_published_bytes() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([plain_identity_target()?])
        .build()?;
    let binding = plan
        .bindings()
        .first()
        .ok_or_else(|| io::Error::other("plain fixture produced no binding"))?;
    let base = BaseTensor::new(
        TensorName::parse("weight")?,
        [2_u64],
        binding_source_content(&plan, binding)?,
    )?;
    let first = add_overlay_layer("weight", "adapter.first")?;
    let second = add_overlay_layer("weight", "adapter.second")?;
    let forward = Arc::new(OverlayPlan::build(
        vec![base.clone()],
        AliasTable::empty(),
        vec![first.clone(), second.clone()],
        ConflictPolicy::Ordered,
        CompositionMode::Lazy,
    )?);
    let reverse = Arc::new(OverlayPlan::build(
        vec![base],
        AliasTable::empty(),
        vec![second, first],
        ConflictPolicy::Ordered,
        CompositionMode::Lazy,
    )?);
    let engine = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("cache"))?;
    let forward_materializer = Materializer::new(&checkpoint, &plan, &engine, &cancellation)?
        .with_cache(&cache)
        .with_overlay_plan(Arc::clone(&forward))?;
    let reverse_materializer = Materializer::new(&checkpoint, &plan, &engine, &cancellation)?
        .with_cache(&cache)
        .with_overlay_plan(reverse)?;
    let forward_address = forward_materializer
        .prepared_cache_address("weight")?
        .ok_or_else(|| io::Error::other("lazy overlay had no final prepared-cache address"))?;
    let reverse_address = reverse_materializer
        .prepared_cache_address("weight")?
        .ok_or_else(|| io::Error::other("reordered overlay had no prepared-cache address"))?;

    assert_ne!(forward_address, reverse_address);
    let cold = forward_materializer.materialize("weight", &cancellation)?;
    let WeightDelivery::Overlay(handoff) = cold else {
        return Err(io::Error::other("cold lazy overlay did not return a handoff").into());
    };
    assert_eq!(
        handoff.target_digest(),
        forward.target_digest(&TensorName::parse("weight")?)?
    );
    assert_eq!(handoff.binding().operations().len(), 2);
    let WeightDelivery::Prepared(base_weight) = handoff.base() else {
        return Err(io::Error::other("overlay base was not materialized as host bytes").into());
    };
    assert_eq!(base_weight.origin(), PreparedOrigin::Source);
    assert_eq!(
        base_weight.bytes().as_slice(),
        [0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0]
    );

    let final_bytes = [0x5a_u8; 8];
    let _publication =
        forward_materializer.publish_prepared_bytes("weight", &final_bytes, &cancellation)?;
    let warm = forward_materializer.materialize("weight", &cancellation)?;
    let WeightDelivery::Prepared(weight) = warm else {
        return Err(io::Error::other("published overlay result was not reused").into());
    };
    assert_eq!(weight.origin(), PreparedOrigin::Cache);
    assert_eq!(weight.bytes().as_slice(), final_bytes);
    Ok(())
}

#[test]
fn overlay_plan_may_cover_components_absent_from_a_partial_binding_plan() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([plain_identity_target()?])
        .build()?;
    let binding = plan
        .bindings()
        .first()
        .ok_or_else(|| io::Error::other("plain fixture produced no binding"))?;
    let selected = BaseTensor::new(
        TensorName::parse("weight")?,
        [2_u64],
        binding_source_content(&plan, binding)?,
    )?;
    let unselected = BaseTensor::new(
        TensorName::parse("other_component.weight")?,
        [2_u64],
        ContentDigest::hash("materializer-test-unselected-base", [b"other"]),
    )?;
    let overlay = Arc::new(OverlayPlan::build(
        vec![selected, unselected],
        AliasTable::empty(),
        vec![
            add_overlay_layer("weight", "adapter.selected")?,
            add_overlay_layer("other_component.weight", "adapter.unselected")?,
        ],
        ConflictPolicy::Ordered,
        CompositionMode::Lazy,
    )?);
    let engine = PreparationEngine::with_builtins()?;

    let materializer = Materializer::new(&checkpoint, &plan, &engine, &cancellation)?
        .with_overlay_plan_with_cancellation(Arc::clone(&overlay), &cancellation)?;

    assert_eq!(
        materializer
            .overlay_plan()
            .ok_or_else(|| io::Error::other("partial overlay was not attached"))?
            .bindings()
            .len(),
        2
    );
    let delivery = materializer.materialize("weight", &cancellation)?;
    let WeightDelivery::Overlay(handoff) = delivery else {
        return Err(io::Error::other("selected overlay binding was not delivered").into());
    };
    assert_eq!(handoff.binding().base().name().as_str(), "weight");
    assert_eq!(handoff.binding().operations().len(), 1);
    Ok(())
}

#[test]
fn provider_conversion_output_publishes_and_reuses_prepared_bytes() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source = inventory_source(&checkpoint, "weight")?;
    let source_name = TensorName::parse("weight")?;
    let target_name = TensorName::parse("converted.weight")?;
    let recipe_implementation = implementation("convert-f32-f16")?;
    let step = RecipeStep::new(
        StableName::parse("converted")?,
        recipe_implementation.clone(),
        [RecipeInput::External(source_name.clone())],
        BTreeMap::new(),
    );
    let recipe = ConversionRecipe::new(
        1,
        recipe_implementation,
        vec![source_name.clone()],
        vec![step],
        BTreeMap::from([(
            target_name.clone(),
            RecipeInput::Step(StableName::parse("converted")?),
        )]),
    )?;
    let target = TargetTensor::builder(
        target_name,
        Requirement::Required,
        [2_u64],
        Representation::contiguous(DType::F16),
        4,
    )
    .aliases([source_name])
    .conversion_recipe(recipe)
    .build()?;
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([source])
        .targets([target])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &engine, &cancellation)?.with_cache(&cache);

    assert!(
        materializer
            .prepared_cache_address("converted.weight")?
            .is_some()
    );
    let cold = materializer.materialize("converted.weight", &cancellation)?;
    let WeightDelivery::Conversion(handoff) = cold else {
        return Err(io::Error::other("cold conversion did not return a provider handoff").into());
    };
    assert_eq!(handoff.source().name().as_str(), "weight");
    assert_eq!(handoff.target().name().as_str(), "converted.weight");
    assert_eq!(handoff.source_bytes().as_slice().len(), 8);

    let converted = [0x00_u8, 0x3c, 0x00, 0xc0];
    let _publication =
        materializer.publish_prepared_bytes("converted.weight", &converted, &cancellation)?;
    let warm = materializer.materialize("converted.weight", &cancellation)?;
    let WeightDelivery::Prepared(warm) = warm else {
        return Err(io::Error::other("published conversion output was not reused").into());
    };
    assert_eq!(warm.origin(), PreparedOrigin::Cache);
    assert_eq!(warm.bytes().as_slice(), converted);
    Ok(())
}

#[test]
fn overlay_attachment_still_rejects_a_mismatched_selected_base() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([plain_identity_target()?])
        .build()?;
    let mismatched = BaseTensor::new(
        TensorName::parse("weight")?,
        [2_u64],
        ContentDigest::hash("materializer-test-wrong-overlay-base", [b"wrong"]),
    )?;
    let overlay = Arc::new(OverlayPlan::build(
        vec![mismatched],
        AliasTable::empty(),
        vec![add_overlay_layer("weight", "adapter.mismatched")?],
        ConflictPolicy::Ordered,
        CompositionMode::Lazy,
    )?);
    let engine = PreparationEngine::with_builtins()?;

    let result = Materializer::new(&checkpoint, &plan, &engine, &cancellation)?
        .with_overlay_plan_with_cancellation(overlay, &cancellation);

    let Err(error) = result else {
        return Err(io::Error::other("mismatched selected overlay base was accepted").into());
    };
    assert_eq!(error.category(), ErrorCategory::Integrity);
    assert_eq!(
        error.message(),
        "overlay base content identity differs from its selected plan source"
    );
    Ok(())
}

#[test]
fn cancellable_overlay_attachment_observes_a_pre_cancelled_token() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let setup_cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&setup_cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([plain_identity_target()?])
        .build()?;
    let binding = plan
        .bindings()
        .first()
        .ok_or_else(|| io::Error::other("plain fixture produced no binding"))?;
    let base = BaseTensor::new(
        TensorName::parse("weight")?,
        [2_u64],
        binding_source_content(&plan, binding)?,
    )?;
    let overlay = Arc::new(OverlayPlan::build(
        vec![base],
        AliasTable::empty(),
        vec![add_overlay_layer("weight", "adapter.cancelled")?],
        ConflictPolicy::Ordered,
        CompositionMode::Lazy,
    )?);
    let engine = PreparationEngine::with_builtins()?;
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let result = Materializer::new(&checkpoint, &plan, &engine, &setup_cancellation)?
        .with_overlay_plan_with_cancellation(overlay, &cancellation);

    let Err(error) = result else {
        return Err(io::Error::other("pre-cancelled overlay attachment succeeded").into());
    };
    assert_eq!(error.category(), ErrorCategory::Cancelled);
    Ok(())
}

#[test]
fn cancellable_overlay_digest_and_address_observe_a_pre_cancelled_token() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("plain.safetensors");
    write_plain_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let setup_cancellation = CancellationToken::new();
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&setup_cancellation)?))
        .sources([inventory_source(&checkpoint, "weight")?])
        .targets([plain_identity_target()?])
        .build()?;
    let binding = plan
        .bindings()
        .first()
        .ok_or_else(|| io::Error::other("plain fixture produced no binding"))?;
    let base = BaseTensor::new(
        TensorName::parse("weight")?,
        [2_u64],
        binding_source_content(&plan, binding)?,
    )?;
    let overlay = Arc::new(OverlayPlan::build(
        vec![base],
        AliasTable::empty(),
        vec![add_overlay_layer("weight", "adapter.digest-cancelled")?],
        ConflictPolicy::Ordered,
        CompositionMode::Lazy,
    )?);
    let engine = PreparationEngine::with_builtins()?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &setup_cancellation)?
        .with_overlay_plan(Arc::clone(&overlay))?;
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let digest_result =
        overlay.target_digest_with_cancellation(&TensorName::parse("weight")?, &cancellation);
    let Err(digest_error) = digest_result else {
        return Err(io::Error::other("pre-cancelled overlay digest succeeded").into());
    };
    assert_eq!(digest_error.category(), ErrorCategory::Cancelled);

    let address_result =
        materializer.prepared_cache_address_with_cancellation("weight", &cancellation);
    let Err(address_error) = address_result else {
        return Err(io::Error::other("pre-cancelled overlay address succeeded").into());
    };
    assert_eq!(address_error.category(), ErrorCategory::Cancelled);
    Ok(())
}

#[test]
fn quantized_weight_is_an_explicit_handoff_without_implicit_decode() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("quantized.safetensors");
    write_quantized_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source = inventory_source(&checkpoint, "packed")?;
    let Storage::Quantized(storage) = source.storage() else {
        return Err(io::Error::other("F4 fixture was not represented as quantized storage").into());
    };
    let capability = RouteCapability::new(
        storage.encoding().clone(),
        QuantizedRoute::HostDequant {
            target_dtype: DType::F16,
        },
        implementation("host-dequant")?,
        None,
        None,
    )?;
    let target = TargetTensor::builder(
        TensorName::parse("packed")?,
        Requirement::Required,
        [4_u64],
        Representation::contiguous(DType::F16),
        8,
    )
    .quantized_route(capability)
    .build()?;
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([source])
        .targets([target])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &engine, &cancellation)?.with_cache(&cache);

    assert!(materializer.prepared_cache_address("packed")?.is_some());
    let delivery = materializer.materialize("packed", &cancellation)?;

    let WeightDelivery::Quantized(handoff) = delivery else {
        return Err(io::Error::other("quantized binding was implicitly materialized").into());
    };
    assert_eq!(handoff.payload().as_slice(), [0x12, 0x34]);
    assert_eq!(handoff.resident_bytes(), 2);
    assert!(matches!(
        handoff.capability().route(),
        QuantizedRoute::HostDequant {
            target_dtype: DType::F16
        }
    ));

    let dequantized = [0_u8; 8];
    let _publication =
        materializer.publish_prepared_bytes("packed", &dequantized, &cancellation)?;
    let warm = materializer.materialize("packed", &cancellation)?;
    let WeightDelivery::Prepared(warm) = warm else {
        return Err(io::Error::other("published host-dequant output was not reused").into());
    };
    assert_eq!(warm.origin(), PreparedOrigin::Cache);
    assert_eq!(warm.bytes().as_slice(), dequantized);
    Ok(())
}

#[test]
fn packed_direct_route_has_no_misleading_host_prepared_address() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("quantized.safetensors");
    write_quantized_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let source = inventory_source(&checkpoint, "packed")?;
    let Storage::Quantized(storage) = source.storage() else {
        return Err(io::Error::other("F4 fixture was not represented as quantized storage").into());
    };
    let capability = RouteCapability::new(
        storage.encoding().clone(),
        QuantizedRoute::PackedDirect,
        implementation("packed-direct")?,
        Some(backend_id()),
        None,
    )?;
    let target = TargetTensor::builder(
        TensorName::parse("packed")?,
        Requirement::Required,
        [4_u64],
        Representation::contiguous(DType::U8),
        2,
    )
    .quantized_route(capability)
    .build()?;
    let plan = BindingPlan::builder(plan_inputs(checkpoint.source_digests(&cancellation)?))
        .sources([source])
        .targets([target])
        .build()?;
    let engine = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &engine, &cancellation)?.with_cache(&cache);

    assert_eq!(materializer.prepared_cache_address("packed")?, None);
    assert_eq!(
        materializer
            .publish_prepared_bytes("packed", &[0x12, 0x34], &cancellation)
            .expect_err("direct packed bytes must not enter the host prepared cache")
            .category(),
        ErrorCategory::Unsupported
    );
    let WeightDelivery::Quantized(handoff) = materializer.materialize("packed", &cancellation)?
    else {
        return Err(io::Error::other("packed-direct route did not remain a handoff").into());
    };
    assert!(matches!(
        handoff.capability().route(),
        QuantizedRoute::PackedDirect
    ));
    Ok(())
}

fn execute_one(
    materializer: &Materializer<'_>,
    pipeline: &Pipeline,
) -> Result<(PreparedOrigin, Vec<u8>, ExecutionReport)> {
    let mut delivered = None;
    let report = materializer.execute(
        pipeline,
        &mut |ordinal, item: PreparedItem<WeightDelivery>, cancellation: &CancellationToken| {
            cancellation.check()?;
            if delivered.replace((ordinal, item.into_value())).is_some() {
                return Err(Error::from_category(
                    ErrorCategory::Delivery,
                    "single-binding test received more than one delivery",
                ));
            }
            Ok(())
        },
        &NoopObserver,
    )?;
    let Some((ordinal, delivery)) = delivered else {
        return Err(Error::from_category(
            ErrorCategory::Delivery,
            "single-binding test received no delivery",
        ));
    };
    if ordinal != 0 {
        return Err(Error::from_category(
            ErrorCategory::Delivery,
            "single-binding test received a nonzero ordinal",
        ));
    }
    let WeightDelivery::Prepared(weight) = delivery else {
        return Err(Error::from_category(
            ErrorCategory::Delivery,
            "plain transform did not deliver prepared host bytes",
        ));
    };
    Ok((weight.origin(), weight.bytes().as_slice().to_vec(), report))
}

fn inventory_source(
    checkpoint: &Checkpoint,
    name: &str,
) -> std::result::Result<SourceTensor, Box<dyn StdError>> {
    let record = checkpoint
        .inventory()
        .tensor(name)
        .ok_or_else(|| io::Error::other(format!("fixture tensor {name:?} is absent")))?;
    Ok(SourceTensor::try_from(record)?)
}

#[derive(Debug)]
struct ScratchCopyProvider {
    implementation: ImplementationId,
    scratch_bytes: u64,
}

impl PreparationProvider for ScratchCopyProvider {
    fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    fn validate(
        &self,
        request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<OutputStrategy> {
        cancellation.check()?;
        let output_bytes = u64::try_from(request.source().len()).map_err(|_error| {
            Error::from_category(ErrorCategory::ResourceLimit, "test output length")
        })?;
        Ok(OutputStrategy::Allocate {
            output_bytes,
            scratch_bytes: self.scratch_bytes,
        })
    }

    fn prepare_into(
        &self,
        request: &PrepareRequest<'_>,
        output: &mut [u8],
        scratch: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        cancellation.check()?;
        let expected_scratch = usize::try_from(self.scratch_bytes).map_err(|_error| {
            Error::from_category(ErrorCategory::ResourceLimit, "test scratch length")
        })?;
        if output.len() != request.source().len() || scratch.len() != expected_scratch {
            return Err(Error::from_category(
                ErrorCategory::Integrity,
                "scratch-copy provider received incorrect storage",
            ));
        }
        scratch.fill(0xa5);
        output.copy_from_slice(request.source().as_slice());
        cancellation.check()
    }
}

#[derive(Debug)]
struct NeverRunProvider {
    implementation: ImplementationId,
    validated: Arc<AtomicBool>,
    executed: Arc<AtomicBool>,
}

impl PreparationProvider for NeverRunProvider {
    fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    fn validate(
        &self,
        request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<OutputStrategy> {
        self.validated.store(true, Ordering::SeqCst);
        cancellation.check()?;
        let output_bytes = u64::try_from(request.source().len()).map_err(|_error| {
            Error::from_category(ErrorCategory::ResourceLimit, "test output length")
        })?;
        Ok(OutputStrategy::Allocate {
            output_bytes,
            scratch_bytes: 0,
        })
    }

    fn prepare_into(
        &self,
        request: &PrepareRequest<'_>,
        output: &mut [u8],
        scratch: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        self.executed.store(true, Ordering::SeqCst);
        cancellation.check()?;
        if output.len() != request.source().len() || !scratch.is_empty() {
            return Err(Error::from_category(
                ErrorCategory::Integrity,
                "never-run provider received incorrect storage",
            ));
        }
        output.copy_from_slice(request.source().as_slice());
        Ok(())
    }
}

fn copying_transform_target(name: &str, implementation: &ImplementationId) -> Result<TargetTensor> {
    let representation = Representation::contiguous(DType::F32);
    let transform = TransformSpec::new(
        implementation.clone(),
        representation.clone(),
        representation.clone(),
    );
    TargetTensor::builder(
        TensorName::parse(name)?,
        Requirement::Required,
        [2_u64],
        representation,
        8,
    )
    .transforms([PlannedTransform::new(transform, 8)])
    .build()
}

fn plain_identity_target() -> Result<TargetTensor> {
    TargetTensor::builder(
        TensorName::parse("weight")?,
        Requirement::Required,
        [2_u64],
        Representation::contiguous(DType::F32),
        8,
    )
    .build()
}

fn plain_transform_target() -> Result<TargetTensor> {
    let transform = TransformSpec::new(
        builtin_contiguous_implementation()?,
        Representation::contiguous(DType::F32),
        Representation::contiguous(DType::F16),
    );
    TargetTensor::builder(
        TensorName::parse("weight")?,
        Requirement::Required,
        [2_u64],
        Representation::contiguous(DType::F16),
        4,
    )
    .transforms([PlannedTransform::new(transform, 4)])
    .build()
}

fn three_stage_transform_target() -> Result<TargetTensor> {
    let implementation = builtin_contiguous_implementation()?;
    let f32 = Representation::contiguous(DType::F32);
    let f16 = Representation::contiguous(DType::F16);
    TargetTensor::builder(
        TensorName::parse("weight")?,
        Requirement::Required,
        [2_u64],
        f16.clone(),
        4,
    )
    .transforms([
        PlannedTransform::new(
            TransformSpec::new(implementation.clone(), f32.clone(), f16.clone()),
            4,
        ),
        PlannedTransform::new(
            TransformSpec::new(implementation.clone(), f16.clone(), f32.clone()),
            8,
        ),
        PlannedTransform::new(TransformSpec::new(implementation, f32, f16), 4),
    ])
    .build()
}

fn three_stage_scratch_transform_target(
    implementation: &ImplementationId,
    scratch_bytes: u64,
) -> Result<TargetTensor> {
    let representation = Representation::contiguous(DType::F32);
    let planned = || {
        PlannedTransform::new(
            TransformSpec::new(
                implementation.clone(),
                representation.clone(),
                representation.clone(),
            ),
            8,
        )
        .with_scratch_bytes(scratch_bytes)
    };
    let transforms = [planned(), planned(), planned()];
    TargetTensor::builder(
        TensorName::parse("weight")?,
        Requirement::Required,
        [2_u64],
        representation,
        8,
    )
    .transforms(transforms)
    .build()
}

fn implementation(operation: &str) -> Result<ImplementationId> {
    Ok(ImplementationId::new(
        StableName::parse("materializer-test")?,
        StableName::parse(operation)?,
        1,
    ))
}

fn add_overlay_layer(target_name: &str, source_name: &str) -> Result<OverlayLayer> {
    let source = SourceTensor::new(
        TensorName::parse(source_name)?,
        [2_u64],
        Storage::Plain {
            dtype: DType::F32,
            span: SourceSpan::new(FileId::from_ordinal(0), 0, 8)?,
        },
    )?;
    OverlayLayer::new(
        [ContentDigest::hash(
            "materializer-test-overlay-source",
            [source_name.as_bytes()],
        )],
        [OverlayOperation::add(
            TensorName::parse(target_name)?,
            source,
            FiniteScale::new(1.0)?,
            implementation("overlay-add")?,
        )],
    )
}

fn plan_inputs(source_digests: Box<[ContentDigest]>) -> PlanInputs {
    plan_inputs_with_selection(source_digests, b"weight")
}

fn plan_inputs_with_selection(
    source_digests: Box<[ContentDigest]>,
    selection: &[u8],
) -> PlanInputs {
    PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash(
            "materializer-test-manifest",
            [b"fixture"],
        )),
        SelectionId::from_digest(ContentDigest::hash(
            "materializer-test-selection",
            [selection],
        )),
        ContractId::from_digest(ContentDigest::hash(
            "materializer-test-contract",
            [b"weight"],
        )),
        backend_id(),
        source_digests,
    )
}

fn backend_id() -> BackendId {
    BackendId::from_digest(ContentDigest::hash("materializer-test-backend", [b"host"]))
}

const fn execution_limits() -> ExecutionLimits {
    ExecutionLimits {
        workers: 1,
        max_work_items: 8,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 8,
        scratch_bytes: 1,
        prepared_bytes: 4,
    }
}

fn write_plain_fixture(path: &Path) -> io::Result<()> {
    let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut payload = Vec::with_capacity(8);
    payload.extend_from_slice(&1.0_f32.to_le_bytes());
    payload.extend_from_slice(&(-2.0_f32).to_le_bytes());
    write_safetensors(path, header, &payload)
}

fn write_two_plain_fixture(path: &Path) -> io::Result<()> {
    let header = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let mut payload = Vec::with_capacity(16);
    for value in [1.0_f32, -2.0, 3.0, -4.0] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    write_safetensors(path, header, &payload)
}

fn write_quantized_fixture(path: &Path) -> io::Result<()> {
    let header = br#"{"packed":{"dtype":"F4","shape":[4],"data_offsets":[0,2]}}"#;
    write_safetensors(path, header, &[0x12, 0x34])
}

fn write_safetensors(path: &Path, header: &[u8], payload: &[u8]) -> io::Result<()> {
    let header_len = u64::try_from(header.len()).map_err(io::Error::other)?;
    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header)?;
    file.write_all(payload)?;
    file.sync_all()
}
