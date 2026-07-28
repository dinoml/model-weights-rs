//! Deferred resource accounting for warm prepared-cache routes.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fs::File;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use model_weights::cache::Cache;
use model_weights::identity::{
    BackendId, ContentDigest, ContractId, ImplementationId, ManifestId, SelectionId, StableName,
};
use model_weights::limits::ExecutionLimits;
use model_weights::materialize::{Materializer, PreparedOrigin, WeightDelivery};
use model_weights::pipeline::{Pipeline, PreparedItem};
use model_weights::plan::{
    BindingPlan, ConversionRecipe, PlanInputs, PlannedTransform, RecipeInput, RecipeStep,
    Requirement, SourceTensor, TargetTensor, TensorName,
};
use model_weights::prepare::{
    PreparationEngine, Representation, TransformSpec, builtin_contiguous_implementation,
};
use model_weights::quantization::{QuantizedRoute, RouteCapability, Storage};
use model_weights::source::SourceDescriptor;
use model_weights::telemetry::{ExecutionEvent, ExecutionPhase, ExecutionReport, NoopObserver};
use model_weights::tensor::DType;
use model_weights::{AccessMode, CancellationToken, Checkpoint, Error, ErrorCategory, Result};
use tempfile::TempDir;

type TestResult<T = ()> = std::result::Result<T, Box<dyn StdError>>;

#[test]
fn warm_cache_hit_bypasses_cold_source_and_scratch_limits() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("weight.safetensors");
    write_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = build_plan(&checkpoint, &cancellation)?;
    let preparation = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &preparation, &cancellation)?.with_cache(&cache);

    let restricted_limits = ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 1,
        scratch_bytes: 1,
        prepared_bytes: 4,
    };
    let cold_pipeline = Pipeline::new(restricted_limits.clone())?;
    let cold_error = execute_one(&materializer, &cold_pipeline)
        .expect_err("an empty cache must still enforce cold source and scratch limits");

    let generous = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 8,
        scratch_bytes: 4,
        prepared_bytes: 4,
    })?;
    let (cold_origin, cold_bytes, _cold_report) = execute_one(&materializer, &generous)?;
    let warm_pipeline = Pipeline::new(restricted_limits)?;
    let (warm_origin, warm_bytes, warm_report) = execute_one(&materializer, &warm_pipeline)?;
    let peaks = warm_report.peak_bytes();

    assert_eq!(
        (
            cold_error.category(),
            cold_origin,
            cold_bytes,
            warm_origin,
            warm_bytes,
            peaks.source(),
            peaks.scratch(),
            peaks.prepared(),
        ),
        (
            ErrorCategory::ResourceLimit,
            PreparedOrigin::Transform,
            vec![0x80, 0x3f, 0x00, 0xc0],
            PreparedOrigin::Cache,
            vec![0x80, 0x3f, 0x00, 0xc0],
            0,
            0,
            4,
        )
    );
    Ok(())
}

#[test]
fn warm_conversion_output_needs_only_its_final_prepared_bytes() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("conversion.safetensors");
    write_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = build_conversion_plan(&checkpoint, &cancellation)?;
    let preparation = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("conversion-cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &preparation, &cancellation)?.with_cache(&cache);

    let too_small_for_cold = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 8,
        scratch_bytes: 1,
        prepared_bytes: 4,
    })?;
    let cold_error = execute_conversion(&materializer, &too_small_for_cold)
        .expect_err("a cold conversion handoff must reserve all eight resident source bytes");
    assert_eq!(cold_error.category(), ErrorCategory::ResourceLimit);

    let cold_pipeline = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 8,
        scratch_bytes: 1,
        prepared_bytes: 8,
    })?;
    let cold_report = execute_conversion(&materializer, &cold_pipeline)?;
    assert_eq!(cold_report.peak_bytes().source(), 8);
    assert_eq!(cold_report.peak_bytes().prepared(), 8);

    let converted = [0x00_u8, 0x3c, 0x00, 0xc0];
    let _publication =
        materializer.publish_prepared_bytes("converted.weight", &converted, &cancellation)?;

    let warm_pipeline = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 1,
        scratch_bytes: 1,
        prepared_bytes: 4,
    })?;
    let (origin, bytes, warm_report) = execute_one(&materializer, &warm_pipeline)?;
    assert_eq!(origin, PreparedOrigin::Cache);
    assert_eq!(bytes, converted);
    assert_eq!(warm_report.peak_bytes().source(), 0);
    assert_eq!(warm_report.peak_bytes().scratch(), 0);
    assert_eq!(warm_report.peak_bytes().prepared(), 4);
    Ok(())
}

#[test]
fn warm_host_dequant_output_reserves_more_than_its_packed_cold_handoff() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("packed.safetensors");
    write_quantized_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = build_quantized_plan(&checkpoint, &cancellation)?;
    let preparation = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("packed-cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &preparation, &cancellation)?.with_cache(&cache);

    let cold_pipeline = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 2,
        scratch_bytes: 1,
        prepared_bytes: 2,
    })?;
    let cold_report = execute_quantized(&materializer, &cold_pipeline)?;
    assert_eq!(cold_report.peak_bytes().source(), 2);
    assert_eq!(cold_report.peak_bytes().prepared(), 2);

    let dequantized = [0_u8; 8];
    let _publication =
        materializer.publish_prepared_bytes("packed", &dequantized, &cancellation)?;
    let too_small_for_warm = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 1,
        scratch_bytes: 1,
        prepared_bytes: 2,
    })?;
    let warm_error = execute_one(&materializer, &too_small_for_warm)
        .expect_err("an eight-byte cached output must not use a two-byte prepared reservation");
    assert_eq!(warm_error.category(), ErrorCategory::ResourceLimit);

    let warm_pipeline = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 1,
        scratch_bytes: 1,
        prepared_bytes: 8,
    })?;
    let (origin, bytes, warm_report) = execute_one(&materializer, &warm_pipeline)?;
    assert_eq!(origin, PreparedOrigin::Cache);
    assert_eq!(bytes, dequantized);
    assert_eq!(warm_report.peak_bytes().source(), 0);
    assert_eq!(warm_report.peak_bytes().prepared(), 8);
    Ok(())
}

#[test]
fn cache_candidate_admission_keeps_later_warm_output_out_of_cold_headroom() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("two-conversions.safetensors");
    write_two_weight_fixture(&checkpoint_path)?;
    let checkpoint = Checkpoint::open(&checkpoint_path)?;
    let cancellation = CancellationToken::new();
    let plan = build_two_conversion_plan(&checkpoint, &cancellation)?;
    let preparation = PreparationEngine::with_builtins()?;
    let cache = Cache::open(temporary.path().join("two-conversion-cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &preparation, &cancellation)?.with_cache(&cache);
    let _publication = materializer.publish_prepared_bytes(
        "converted.b",
        &[0x00, 0x3c, 0x00, 0xc0],
        &cancellation,
    )?;

    let pipeline_cancellation = CancellationToken::new();
    let pipeline = Pipeline::with_cancellation(
        ExecutionLimits {
            workers: 2,
            max_work_items: 2,
            delivery_queue_depth: 2,
            dispatch_lookahead: 3,
            source_bytes: 8,
            scratch_bytes: 1,
            prepared_bytes: 8,
        },
        pipeline_cancellation.clone(),
    )?;
    let first_lookup_waiting = Arc::new(AtomicBool::new(false));
    let later_prepared_early = Arc::new(AtomicBool::new(false));
    let observer_waiting = Arc::clone(&first_lookup_waiting);
    let observer_early = Arc::clone(&later_prepared_early);
    let observer = move |event: &ExecutionEvent| match event {
        ExecutionEvent::PhaseStarted {
            phase: ExecutionPhase::CacheLookup,
            ordinal: Some(0),
        } => {
            observer_waiting.store(true, Ordering::SeqCst);
            for _ in 0..100 {
                if observer_early.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(20));
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
            observer_waiting.store(false, Ordering::SeqCst);
        }
        ExecutionEvent::WorkPrepared { ordinal: 1, .. }
            if observer_waiting.load(Ordering::SeqCst) =>
        {
            observer_early.store(true, Ordering::SeqCst);
        }
        _ => {}
    };
    let (finished_sender, finished_receiver) = mpsc::channel();
    let watchdog = thread::spawn(move || {
        if finished_receiver
            .recv_timeout(Duration::from_secs(2))
            .is_err()
        {
            pipeline_cancellation.cancel();
        }
    });
    let mut deliveries = Vec::new();
    let execution = {
        let mut sink = |ordinal: u64,
                        item: PreparedItem<WeightDelivery>,
                        cancellation: &CancellationToken|
         -> Result<()> {
            cancellation.check()?;
            let route = match item.into_value() {
                WeightDelivery::Conversion(_) => "conversion",
                WeightDelivery::Prepared(weight) if weight.origin() == PreparedOrigin::Cache => {
                    "cache"
                }
                _ => {
                    return Err(Error::from_category(
                        ErrorCategory::Delivery,
                        "admission test received an unexpected delivery route",
                    ));
                }
            };
            deliveries.push((ordinal, route));
            Ok(())
        };
        materializer.execute(&pipeline, &mut sink, &observer)
    };
    let _ = finished_sender.send(());
    watchdog
        .join()
        .map_err(|_panic| io::Error::other("admission-test watchdog panicked"))?;
    let report = execution?;

    assert!(!later_prepared_early.load(Ordering::SeqCst));
    assert_eq!(deliveries, [(0, "conversion"), (1, "cache")]);
    assert_eq!(report.peak_bytes().prepared(), 8);
    Ok(())
}

#[cfg(feature = "mmap")]
#[test]
fn mapped_conversion_charges_resident_prepared_but_no_source_read_bytes() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("mapped-conversion.safetensors");
    write_fixture(&checkpoint_path)?;
    let checkpoint = open_retained_checkpoint(&checkpoint_path, AccessMode::Mmap)?;
    let cancellation = CancellationToken::new();
    let plan = build_conversion_plan(&checkpoint, &cancellation)?;
    let preparation = PreparationEngine::with_builtins()?;
    let materializer = Materializer::new(&checkpoint, &plan, &preparation, &cancellation)?;
    let pipeline = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 1,
        scratch_bytes: 1,
        prepared_bytes: 8,
    })?;

    let report = execute_conversion(&materializer, &pipeline)?;
    assert_eq!(report.peak_bytes().source(), 0);
    assert_eq!(report.peak_bytes().prepared(), 8);
    Ok(())
}

#[test]
fn retained_auto_conversion_reserves_its_owned_fallback_read() -> TestResult {
    let temporary = TempDir::new()?;
    let checkpoint_path = temporary.path().join("owned-conversion.safetensors");
    write_fixture(&checkpoint_path)?;
    let checkpoint = open_retained_checkpoint(&checkpoint_path, AccessMode::Auto)?;
    let cancellation = CancellationToken::new();
    let plan = build_conversion_plan(&checkpoint, &cancellation)?;
    let preparation = PreparationEngine::with_builtins()?;
    let materializer = Materializer::new(&checkpoint, &plan, &preparation, &cancellation)?;
    let restricted = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 1,
        scratch_bytes: 1,
        prepared_bytes: 8,
    })?;

    let error = execute_conversion(&materializer, &restricted)
        .expect_err("retained Auto access must budget its possible owned-read fallback");
    assert_eq!(error.category(), ErrorCategory::ResourceLimit);

    let pipeline = Pipeline::new(ExecutionLimits {
        workers: 1,
        max_work_items: 1,
        delivery_queue_depth: 1,
        dispatch_lookahead: 2,
        source_bytes: 8,
        scratch_bytes: 1,
        prepared_bytes: 8,
    })?;
    let report = execute_conversion(&materializer, &pipeline)?;
    assert_eq!(report.peak_bytes().source(), 8);
    assert_eq!(report.peak_bytes().prepared(), 8);
    Ok(())
}

fn execute_one(
    materializer: &Materializer<'_>,
    pipeline: &Pipeline,
) -> Result<(PreparedOrigin, Vec<u8>, ExecutionReport)> {
    let mut delivered = None;
    let report = {
        let mut sink = |_ordinal: u64,
                        item: PreparedItem<WeightDelivery>,
                        cancellation: &CancellationToken|
         -> Result<()> {
            cancellation.check()?;
            let WeightDelivery::Prepared(weight) = item.into_value() else {
                return Err(Error::from_category(
                    ErrorCategory::Delivery,
                    "warm-cache budget test expected prepared host bytes",
                ));
            };
            delivered = Some((weight.origin(), weight.bytes().as_slice().to_vec()));
            Ok(())
        };
        materializer.execute(pipeline, &mut sink, &NoopObserver)?
    };
    let (origin, bytes) = delivered.ok_or_else(|| {
        Error::from_category(
            ErrorCategory::Delivery,
            "warm-cache budget test received no delivery",
        )
    })?;
    Ok((origin, bytes, report))
}

fn execute_conversion(
    materializer: &Materializer<'_>,
    pipeline: &Pipeline,
) -> Result<ExecutionReport> {
    let mut delivered = false;
    let report = {
        let mut sink = |_ordinal: u64,
                        item: PreparedItem<WeightDelivery>,
                        cancellation: &CancellationToken|
         -> Result<()> {
            cancellation.check()?;
            let WeightDelivery::Conversion(handoff) = item.into_value() else {
                return Err(Error::from_category(
                    ErrorCategory::Delivery,
                    "cold conversion budget test expected a conversion handoff",
                ));
            };
            if handoff.resident_bytes() != 8 {
                return Err(Error::from_category(
                    ErrorCategory::Delivery,
                    "cold conversion handoff has an unexpected resident size",
                ));
            }
            delivered = true;
            Ok(())
        };
        materializer.execute(pipeline, &mut sink, &NoopObserver)?
    };
    if !delivered {
        return Err(Error::from_category(
            ErrorCategory::Delivery,
            "cold conversion budget test received no delivery",
        ));
    }
    Ok(report)
}

fn execute_quantized(
    materializer: &Materializer<'_>,
    pipeline: &Pipeline,
) -> Result<ExecutionReport> {
    let mut delivered = false;
    let report = {
        let mut sink = |_ordinal: u64,
                        item: PreparedItem<WeightDelivery>,
                        cancellation: &CancellationToken|
         -> Result<()> {
            cancellation.check()?;
            let WeightDelivery::Quantized(handoff) = item.into_value() else {
                return Err(Error::from_category(
                    ErrorCategory::Delivery,
                    "cold host-dequant budget test expected a quantized handoff",
                ));
            };
            if handoff.resident_bytes() != 2 {
                return Err(Error::from_category(
                    ErrorCategory::Delivery,
                    "cold quantized handoff has an unexpected resident size",
                ));
            }
            delivered = true;
            Ok(())
        };
        materializer.execute(pipeline, &mut sink, &NoopObserver)?
    };
    if !delivered {
        return Err(Error::from_category(
            ErrorCategory::Delivery,
            "cold host-dequant budget test received no delivery",
        ));
    }
    Ok(report)
}

fn build_plan(checkpoint: &Checkpoint, cancellation: &CancellationToken) -> Result<BindingPlan> {
    let source = checkpoint
        .inventory()
        .tensor("weight")
        .ok_or_else(|| Error::from_category(ErrorCategory::Binding, "fixture weight is missing"))?;
    let source = SourceTensor::try_from(source)?;
    let implementation = builtin_contiguous_implementation()?;
    let first = TransformSpec::new(
        implementation.clone(),
        Representation::contiguous(DType::F32),
        Representation::contiguous(DType::F16),
    );
    let second = TransformSpec::new(
        implementation,
        Representation::contiguous(DType::F16),
        Representation::contiguous(DType::Bf16),
    );
    let target = TargetTensor::builder(
        TensorName::parse("weight")?,
        Requirement::Required,
        [2],
        Representation::contiguous(DType::Bf16),
        4,
    )
    .transforms([
        PlannedTransform::new(first, 4),
        PlannedTransform::new(second, 4),
    ])
    .build()?;
    let inputs = PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash("warm-budget-manifest-v1", [b"fixture"])),
        SelectionId::from_digest(ContentDigest::hash("warm-budget-selection-v1", [b"weight"])),
        ContractId::from_digest(ContentDigest::hash(
            "warm-budget-contract-v1",
            [b"f32-f16-bf16"],
        )),
        BackendId::from_digest(ContentDigest::hash(
            "warm-budget-backend-v1",
            [b"host-contiguous"],
        )),
        checkpoint.source_digests(cancellation)?,
    );
    BindingPlan::builder(inputs)
        .sources([source])
        .targets([target])
        .build()
}

fn build_conversion_plan(
    checkpoint: &Checkpoint,
    cancellation: &CancellationToken,
) -> Result<BindingPlan> {
    let source = checkpoint
        .inventory()
        .tensor("weight")
        .ok_or_else(|| Error::from_category(ErrorCategory::Binding, "fixture weight is missing"))?;
    let source = SourceTensor::try_from(source)?;
    let source_name = TensorName::parse("weight")?;
    let target_name = TensorName::parse("converted.weight")?;
    let implementation = ImplementationId::new(
        StableName::parse("warm-budget-test")?,
        StableName::parse("convert-f32-f16")?,
        1,
    );
    let recipe = ConversionRecipe::new(
        1,
        implementation.clone(),
        vec![source_name.clone()],
        vec![RecipeStep::new(
            StableName::parse("converted")?,
            implementation,
            [RecipeInput::External(source_name.clone())],
            BTreeMap::new(),
        )],
        BTreeMap::from([(
            target_name.clone(),
            RecipeInput::Step(StableName::parse("converted")?),
        )]),
    )?;
    let target = TargetTensor::builder(
        target_name,
        Requirement::Required,
        [2],
        Representation::contiguous(DType::F16),
        4,
    )
    .aliases([source_name])
    .conversion_recipe(recipe)
    .build()?;
    let inputs = PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash(
            "warm-conversion-manifest-v1",
            [b"fixture"],
        )),
        SelectionId::from_digest(ContentDigest::hash(
            "warm-conversion-selection-v1",
            [b"weight"],
        )),
        ContractId::from_digest(ContentDigest::hash(
            "warm-conversion-contract-v1",
            [b"f32-f16"],
        )),
        BackendId::from_digest(ContentDigest::hash(
            "warm-conversion-backend-v1",
            [b"provider"],
        )),
        checkpoint.source_digests(cancellation)?,
    );
    BindingPlan::builder(inputs)
        .sources([source])
        .targets([target])
        .build()
}

fn build_two_conversion_plan(
    checkpoint: &Checkpoint,
    cancellation: &CancellationToken,
) -> Result<BindingPlan> {
    let source_a = checkpoint
        .inventory()
        .tensor("a")
        .ok_or_else(|| Error::from_category(ErrorCategory::Binding, "fixture a is missing"))?;
    let source_b = checkpoint
        .inventory()
        .tensor("b")
        .ok_or_else(|| Error::from_category(ErrorCategory::Binding, "fixture b is missing"))?;
    let source_a = SourceTensor::try_from(source_a)?;
    let source_b = SourceTensor::try_from(source_b)?;
    let target_a = conversion_target("a", "converted.a", "convert-a")?;
    let target_b = conversion_target("b", "converted.b", "convert-b")?;
    let inputs = PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash(
            "warm-two-conversion-manifest-v1",
            [b"fixture"],
        )),
        SelectionId::from_digest(ContentDigest::hash(
            "warm-two-conversion-selection-v1",
            [b"a,b"],
        )),
        ContractId::from_digest(ContentDigest::hash(
            "warm-two-conversion-contract-v1",
            [b"f32-f16"],
        )),
        BackendId::from_digest(ContentDigest::hash(
            "warm-two-conversion-backend-v1",
            [b"provider"],
        )),
        checkpoint.source_digests(cancellation)?,
    );
    BindingPlan::builder(inputs)
        .sources([source_a, source_b])
        .targets([target_a, target_b])
        .build()
}

fn conversion_target(source: &str, target: &str, operation: &str) -> Result<TargetTensor> {
    let source_name = TensorName::parse(source)?;
    let target_name = TensorName::parse(target)?;
    let implementation = ImplementationId::new(
        StableName::parse("warm-budget-test")?,
        StableName::parse(operation)?,
        1,
    );
    let step_name = StableName::parse("converted")?;
    let recipe = ConversionRecipe::new(
        1,
        implementation.clone(),
        vec![source_name.clone()],
        vec![RecipeStep::new(
            step_name.clone(),
            implementation,
            [RecipeInput::External(source_name.clone())],
            BTreeMap::new(),
        )],
        BTreeMap::from([(target_name.clone(), RecipeInput::Step(step_name))]),
    )?;
    TargetTensor::builder(
        target_name,
        Requirement::Required,
        [2_u64],
        Representation::contiguous(DType::F16),
        4,
    )
    .aliases([source_name])
    .conversion_recipe(recipe)
    .build()
}

fn build_quantized_plan(
    checkpoint: &Checkpoint,
    cancellation: &CancellationToken,
) -> Result<BindingPlan> {
    let source = checkpoint
        .inventory()
        .tensor("packed")
        .ok_or_else(|| Error::from_category(ErrorCategory::Binding, "packed fixture is missing"))?;
    let source = SourceTensor::try_from(source)?;
    let Storage::Quantized(storage) = source.storage() else {
        return Err(Error::from_category(
            ErrorCategory::Binding,
            "packed fixture did not produce quantized storage",
        ));
    };
    let implementation = ImplementationId::new(
        StableName::parse("warm-budget-test")?,
        StableName::parse("host-dequant")?,
        1,
    );
    let capability = RouteCapability::new(
        storage.encoding().clone(),
        QuantizedRoute::HostDequant {
            target_dtype: DType::F16,
        },
        implementation,
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
    let inputs = PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash(
            "warm-quantized-manifest-v1",
            [b"fixture"],
        )),
        SelectionId::from_digest(ContentDigest::hash(
            "warm-quantized-selection-v1",
            [b"packed"],
        )),
        ContractId::from_digest(ContentDigest::hash(
            "warm-quantized-contract-v1",
            [b"f4-f16"],
        )),
        BackendId::from_digest(ContentDigest::hash(
            "warm-quantized-backend-v1",
            [b"provider"],
        )),
        checkpoint.source_digests(cancellation)?,
    );
    BindingPlan::builder(inputs)
        .sources([source])
        .targets([target])
        .build()
}

#[expect(
    unsafe_code,
    reason = "the temporary fixture remains alive and immutable for the returned checkpoint lifetime"
)]
fn open_retained_checkpoint(
    path: &std::path::Path,
    access_mode: AccessMode,
) -> TestResult<Checkpoint> {
    let cancellation = CancellationToken::new();
    let local = Checkpoint::open(path)?;
    let digest = local
        .source_digests(&cancellation)?
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("fixture checkpoint has no source digest"))?;
    let size = std::fs::metadata(path)?.len();
    drop(local);
    // SAFETY: the enclosing test owns the temporary directory and does not
    // mutate or remove this file until the returned checkpoint is dropped.
    let source =
        unsafe { SourceDescriptor::retained("weight.safetensors", path, size, digest, ())? };
    Ok(Checkpoint::builder(source)
        .access_mode(access_mode)
        .open()?)
}

fn write_fixture(path: &std::path::Path) -> io::Result<()> {
    let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut file = File::create(path)?;
    file.write_all(
        &u64::try_from(header.len())
            .expect("fixture header length must fit u64")
            .to_le_bytes(),
    )?;
    file.write_all(header)?;
    for value in [1.0_f32, -2.0] {
        file.write_all(&value.to_le_bytes())?;
    }
    file.sync_all()
}

fn write_quantized_fixture(path: &std::path::Path) -> io::Result<()> {
    let header = br#"{"packed":{"dtype":"F4","shape":[4],"data_offsets":[0,2]}}"#;
    let mut file = File::create(path)?;
    file.write_all(
        &u64::try_from(header.len())
            .expect("fixture header length must fit u64")
            .to_le_bytes(),
    )?;
    file.write_all(header)?;
    file.write_all(&[0x12, 0x34])?;
    file.sync_all()
}

fn write_two_weight_fixture(path: &std::path::Path) -> io::Result<()> {
    let header = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let mut file = File::create(path)?;
    file.write_all(
        &u64::try_from(header.len())
            .expect("fixture header length must fit u64")
            .to_le_bytes(),
    )?;
    file.write_all(header)?;
    for value in [1.0_f32, -2.0, 3.0, -4.0] {
        file.write_all(&value.to_le_bytes())?;
    }
    file.sync_all()
}
