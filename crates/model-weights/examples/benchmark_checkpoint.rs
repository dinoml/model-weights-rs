//! Reproducible bounded checkpoint materialization and prepared-cache probe.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::ffi::{OsStr, OsString};
use std::io;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use model_weights::cache::{Cache, CacheNamespace, EvictionReason};
use model_weights::identity::{
    BackendId, ContentDigest, ContractId, ManifestId, SelectionId, SnapshotId,
};
use model_weights::limits::ExecutionLimits;
use model_weights::materialize::{Materializer, PreparedOrigin, WeightDelivery};
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
    ExecutionEvent, ExecutionObserver, ExecutionPhase, ExecutionReport,
};
use model_weights::tensor::DType;
use model_weights::{CancellationToken, Checkpoint};

const MEBIBYTE: f64 = 1024.0 * 1024.0;

fn main() -> std::result::Result<(), Box<dyn StdError>> {
    let arguments = parse_arguments()?;
    let cancellation = CancellationToken::new();

    let setup_inventory_started = Instant::now();
    let checkpoint = Checkpoint::open(&arguments.path)?;
    let setup_inventory_elapsed = setup_inventory_started.elapsed();

    let setup_identity_bytes = checkpoint.pending_digest_bytes(&cancellation)?;
    let setup_identity_started = Instant::now();
    let snapshot = checkpoint.snapshot_id(&cancellation)?;
    let source_digests = checkpoint.source_digests(&cancellation)?;
    let setup_identity_elapsed = setup_identity_started.elapsed();
    let execution_pending_digest_bytes = checkpoint.pending_digest_bytes(&cancellation)?;

    let planning_started = Instant::now();
    let plan = build_plan(
        &checkpoint,
        snapshot,
        source_digests,
        &arguments.selected_names,
        arguments.target_dtype,
    )?;
    let planning_elapsed = planning_started.elapsed();

    let preparation = PreparationEngine::with_builtins()?;
    let cache = arguments
        .cache_directory
        .as_ref()
        .map(Cache::open)
        .transpose()?;
    let reset_entries = reset_selected_prepared_cache(
        &checkpoint,
        &plan,
        &preparation,
        cache.as_ref(),
        &cancellation,
    )?;

    let limits = execution_limits(&plan);

    let cold = run_pass(&checkpoint, &plan, &preparation, cache.as_ref(), &limits)?;
    let warm = run_pass(&checkpoint, &plan, &preparation, cache.as_ref(), &limits)?;

    println!("snapshot={snapshot}");
    println!("plan={}", plan.id());
    println!("target_dtype={}", arguments.target_dtype.as_str());
    match &arguments.cache_directory {
        Some(directory) => println!("cache_directory={}", directory.display()),
        None => println!("cache_directory=disabled"),
    }
    println!("cache_entries_reset={reset_entries}");
    println!("files={}", checkpoint.inventory().files().len());
    println!("inventory_tensors={}", checkpoint.inventory().len());
    println!("selected_tensors={}", plan.bindings().len());
    println!("unused_tensors={}", plan.unused_sources().len());
    println!("pipeline_workers={}", limits.workers);
    println!("pipeline_work_item_limit={}", limits.max_work_items);
    println!(
        "pipeline_delivery_queue_limit={}",
        limits.delivery_queue_depth
    );
    println!("pipeline_dispatch_lookahead={}", limits.dispatch_lookahead);
    println!("pipeline_source_limit_bytes={}", limits.source_bytes);
    println!("pipeline_scratch_limit_bytes={}", limits.scratch_bytes);
    println!("pipeline_prepared_limit_bytes={}", limits.prepared_bytes);
    println!(
        "setup_inventory_ms={:.3}",
        milliseconds(setup_inventory_elapsed)
    );
    println!(
        "setup_identity_ms={:.3}",
        milliseconds(setup_identity_elapsed)
    );
    println!("setup_identity_bytes={setup_identity_bytes}");
    println!("planning_ms={:.3}", milliseconds(planning_elapsed));
    println!("execution_pending_digest_bytes={execution_pending_digest_bytes}");
    print_pass("cold", &cold);
    print_pass("warm", &warm);
    Ok(())
}

#[derive(Debug)]
struct BenchmarkArguments {
    path: PathBuf,
    selected_names: Vec<String>,
    cache_directory: Option<PathBuf>,
    target_dtype: TargetDtype,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum TargetDtype {
    #[default]
    Source,
    F16,
    Bf16,
    F32,
}

impl TargetDtype {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }

    const fn resolve(self, source: DType) -> DType {
        match self {
            Self::Source => source,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::Bf16,
            Self::F32 => DType::F32,
        }
    }
}

fn parse_arguments() -> io::Result<BenchmarkArguments> {
    parse_arguments_from(std::env::args_os().skip(1))
}

fn parse_arguments_from(
    arguments: impl IntoIterator<Item = OsString>,
) -> io::Result<BenchmarkArguments> {
    let mut arguments = arguments.into_iter();
    let mut cache_directory = None;
    let mut target_dtype = TargetDtype::Source;
    let mut positional_only = false;
    let mut positional = Vec::new();

    while let Some(argument) = arguments.next() {
        if !positional_only && argument == OsStr::new("--") {
            positional_only = true;
        } else if !positional_only && argument == OsStr::new("--cache") {
            let directory = arguments.next().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "--cache requires a directory")
            })?;
            cache_directory = Some(PathBuf::from(directory));
        } else if !positional_only && argument == OsStr::new("--dtype") {
            let value = arguments.next().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "--dtype requires a value")
            })?;
            target_dtype = parse_target_dtype(&value)?;
        } else if !positional_only && argument.to_string_lossy().starts_with('-') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown option {}; {}", argument.to_string_lossy(), usage()),
            ));
        } else {
            positional.push(argument);
        }
    }

    let mut positional = positional.into_iter();
    let path = positional
        .next()
        .map(PathBuf::from)
        .ok_or_else(usage_error)?;
    let mut selected_names = positional
        .map(|name| {
            name.into_string().map_err(|_name| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "selected tensor names must be valid UTF-8",
                )
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    if selected_names.is_empty() {
        return Err(usage_error());
    }
    selected_names.sort_unstable();
    selected_names.dedup();
    Ok(BenchmarkArguments {
        path,
        selected_names,
        cache_directory,
        target_dtype,
    })
}

fn parse_target_dtype(value: &OsStr) -> io::Result<TargetDtype> {
    match value.to_str() {
        Some("source") => Ok(TargetDtype::Source),
        Some("f16") => Ok(TargetDtype::F16),
        Some("bf16") => Ok(TargetDtype::Bf16),
        Some("f32") => Ok(TargetDtype::F32),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--dtype must be one of source, f16, bf16, or f32",
        )),
    }
}

fn build_plan(
    checkpoint: &Checkpoint,
    snapshot: SnapshotId,
    source_digests: Box<[ContentDigest]>,
    selected_names: &[String],
    target_dtype: TargetDtype,
) -> std::result::Result<BindingPlan, Box<dyn StdError>> {
    let sources = checkpoint
        .inventory()
        .iter()
        .map(SourceTensor::try_from)
        .collect::<model_weights::Result<Vec<_>>>()?;
    let implementation = builtin_contiguous_implementation()?;
    let targets = selected_names
        .iter()
        .map(|name| selected_target(checkpoint, name, target_dtype, &implementation))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let contract_bytes = serde_json::to_vec(&targets)?;
    let snapshot_digest = snapshot.digest();
    let inputs = PlanInputs::new(
        ManifestId::from_digest(ContentDigest::hash(
            "benchmark-manifest-v2",
            [snapshot_digest.as_bytes()],
        )),
        SelectionId::from_digest(ContentDigest::hash(
            "benchmark-selection-v2",
            selected_names.iter().map(String::as_bytes),
        )),
        ContractId::from_digest(ContentDigest::hash(
            "benchmark-contract-v2",
            [contract_bytes],
        )),
        BackendId::from_digest(ContentDigest::hash(
            "benchmark-backend-v2",
            [b"host-contiguous"],
        )),
        source_digests,
    );
    Ok(BindingPlan::builder(inputs)
        .sources(sources)
        .targets(targets)
        .extra_source_policy(ExtraSourcePolicy::Allow)
        .build()?)
}

fn selected_target(
    checkpoint: &Checkpoint,
    name: &str,
    requested_dtype: TargetDtype,
    implementation: &model_weights::identity::ImplementationId,
) -> std::result::Result<TargetTensor, Box<dyn StdError>> {
    let record = checkpoint.inventory().tensor(name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("selected tensor {name:?} is not present in the checkpoint"),
        )
    })?;
    let Storage::Plain {
        dtype: source_dtype,
        ..
    } = record.storage()
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "selected tensor {name:?} is packed; this probe requires an explicit consumer quantized route"
            ),
        )
        .into());
    };
    let target_dtype = requested_dtype.resolve(*source_dtype);
    if target_dtype != *source_dtype
        && (!is_builtin_float(*source_dtype) || !is_builtin_float(target_dtype))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "selected tensor {name:?} cannot use the built-in {source_dtype:?} to {target_dtype:?} cast"
            ),
        )
        .into());
    }
    let output_size = target_dtype.byte_len(record.shape())?;
    let mut builder = TargetTensor::builder(
        TensorName::parse(name)?,
        Requirement::Required,
        record.shape(),
        Representation::contiguous(target_dtype),
        output_size,
    );
    if target_dtype != *source_dtype {
        let transform = TransformSpec::new(
            implementation.clone(),
            Representation::contiguous(*source_dtype),
            Representation::contiguous(target_dtype),
        );
        builder = builder.transforms([PlannedTransform::new(transform, output_size)]);
    }
    Ok(builder.build()?)
}

const fn is_builtin_float(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::Bf16 | DType::F32)
}

fn reset_selected_prepared_cache(
    checkpoint: &Checkpoint,
    plan: &BindingPlan,
    preparation: &PreparationEngine,
    cache: Option<&Cache>,
    cancellation: &CancellationToken,
) -> model_weights::Result<usize> {
    let Some(cache) = cache else {
        return Ok(0);
    };
    let materializer =
        Materializer::new(checkpoint, plan, preparation, cancellation)?.with_cache(cache);
    let mut removed = 0_usize;
    for binding in plan.bindings() {
        if let Some(address) = materializer.prepared_cache_address_with_cancellation(
            binding.target().name().as_str(),
            cancellation,
        )? {
            removed = removed.saturating_add(
                cache
                    .evict(
                        CacheNamespace::Prepared,
                        address.key(),
                        EvictionReason::Explicit,
                        cancellation,
                    )?
                    .is_some()
                    .into(),
            );
        }
    }
    Ok(removed)
}

fn execution_limits(plan: &BindingPlan) -> ExecutionLimits {
    let mut limits = ExecutionLimits::default();
    limits.workers = limits.workers.min(plan.bindings().len().max(1));
    limits.dispatch_lookahead = limits.dispatch_lookahead.min(plan.bindings().len().max(1));
    for binding in plan.bindings() {
        limits.source_bytes = limits
            .source_bytes
            .max(binding.source().storage().span().len());
        limits.prepared_bytes = limits.prepared_bytes.max(binding.target().output_size());
    }
    limits
}

#[derive(Debug, Clone, Copy, Default)]
struct PhaseTotal {
    duration: Duration,
    bytes: u64,
    invocations: u64,
}

#[derive(Debug, Default)]
struct BenchmarkObserver {
    phases: Mutex<BTreeMap<ExecutionPhase, PhaseTotal>>,
}

impl BenchmarkObserver {
    fn snapshot(&self) -> BTreeMap<ExecutionPhase, PhaseTotal> {
        self.phases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl ExecutionObserver for BenchmarkObserver {
    fn observe(&self, event: &ExecutionEvent) {
        let ExecutionEvent::PhaseFinished {
            phase,
            duration,
            bytes,
            ..
        } = event
        else {
            return;
        };
        let mut phases = self
            .phases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let total = phases.entry(*phase).or_default();
        total.duration = total.duration.saturating_add(*duration);
        total.bytes = total.bytes.saturating_add(*bytes);
        total.invocations = total.invocations.saturating_add(1);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct OriginCounts {
    source: u64,
    transform: u64,
    cache: u64,
    other: u64,
}

#[derive(Debug)]
struct PassReport {
    total_time: Duration,
    pipeline: ExecutionReport,
    phases: BTreeMap<ExecutionPhase, PhaseTotal>,
    origins: OriginCounts,
}

fn run_pass(
    checkpoint: &Checkpoint,
    plan: &BindingPlan,
    preparation: &PreparationEngine,
    cache: Option<&Cache>,
    limits: &ExecutionLimits,
) -> model_weights::Result<PassReport> {
    let total_started = Instant::now();
    let cancellation = CancellationToken::new();
    let observer = BenchmarkObserver::default();
    let materializer =
        Materializer::new_with_observer(checkpoint, plan, preparation, &cancellation, &observer)?;
    let materializer = match cache {
        Some(cache) => materializer.with_cache(cache),
        None => materializer,
    };
    let pipeline = Pipeline::with_cancellation(limits.clone(), cancellation)?;
    let mut origins = OriginCounts::default();
    let report = {
        let mut sink = |_ordinal: u64,
                        item: PreparedItem<WeightDelivery>,
                        cancellation: &CancellationToken|
         -> model_weights::Result<()> {
            cancellation.check()?;
            let delivery = item.into_value();
            match delivery {
                WeightDelivery::Prepared(weight) => {
                    std::hint::black_box(weight.bytes().as_slice());
                    match weight.origin() {
                        PreparedOrigin::Source => {
                            origins.source = origins.source.saturating_add(1);
                        }
                        PreparedOrigin::Transform => {
                            origins.transform = origins.transform.saturating_add(1);
                        }
                        PreparedOrigin::Cache => {
                            origins.cache = origins.cache.saturating_add(1);
                        }
                        _ => {
                            origins.other = origins.other.saturating_add(1);
                        }
                    }
                }
                other => {
                    std::hint::black_box(other.resident_bytes());
                    origins.other = origins.other.saturating_add(1);
                }
            }
            Ok(())
        };
        materializer.execute(&pipeline, &mut sink, &observer)?
    };
    Ok(PassReport {
        total_time: total_started.elapsed(),
        pipeline: report,
        phases: observer.snapshot(),
        origins,
    })
}

fn print_pass(name: &str, pass: &PassReport) {
    println!("{name}.total_ms={:.3}", milliseconds(pass.total_time));
    println!(
        "{name}.pipeline_ms={:.3}",
        milliseconds(pass.pipeline.wall_time())
    );
    for (phase, label) in [
        (ExecutionPhase::Hashing, "hashing"),
        (ExecutionPhase::Mapping, "mapping"),
        (ExecutionPhase::SourceRead, "source_read"),
        (ExecutionPhase::CacheLookup, "cache_lookup"),
        (ExecutionPhase::Transform, "transform"),
        (ExecutionPhase::Preparation, "preparation"),
        (ExecutionPhase::QueueWait, "queue_wait"),
        (ExecutionPhase::DeliveryCallback, "delivery"),
    ] {
        let total = pass.phases.get(&phase).copied().unwrap_or_default();
        println!(
            "{name}.phase.{label}.ms={:.3}",
            milliseconds(total.duration)
        );
        println!("{name}.phase.{label}.bytes={}", total.bytes);
        println!("{name}.phase.{label}.invocations={}", total.invocations);
    }
    let counters = pass.pipeline.counters();
    let peaks = pass.pipeline.peak_bytes();
    println!("{name}.submitted={}", counters.submitted());
    println!("{name}.prepared={}", counters.prepared());
    println!("{name}.delivered={}", counters.delivered());
    println!("{name}.failed={}", counters.failed());
    println!("{name}.delivered_bytes={}", counters.delivered_bytes());
    println!(
        "{name}.peak_delivery_queue_depth={}",
        pass.pipeline.peak_delivery_queue_depth()
    );
    println!("{name}.peak_source_bytes={}", peaks.source());
    println!("{name}.peak_scratch_bytes={}", peaks.scratch());
    println!("{name}.peak_prepared_bytes={}", peaks.prepared());
    println!(
        "{name}.throughput_mib_per_second={:.3}",
        pass.pipeline.throughput_bytes_per_second() / MEBIBYTE
    );
    println!("{name}.origin.source={}", pass.origins.source);
    println!("{name}.origin.transform={}", pass.origins.transform);
    println!("{name}.origin.cache={}", pass.origins.cache);
    println!("{name}.origin.other={}", pass.origins.other);
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

const fn usage() -> &'static str {
    "usage: benchmark_checkpoint [--dtype source|f16|bf16|f32] [--cache DIR] \
     <checkpoint-or-index> <TENSOR> [TENSOR ...]"
}

fn usage_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, usage())
}

#[cfg(test)]
mod tests {
    use super::{
        TargetDtype, build_plan, execution_limits, parse_arguments_from,
        reset_selected_prepared_cache, run_pass,
    };
    use std::ffi::OsString;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;

    use model_weights::cache::Cache;
    use model_weights::prepare::PreparationEngine;
    use model_weights::telemetry::ExecutionPhase;
    use model_weights::{CancellationToken, Checkpoint};
    use tempfile::TempDir;

    #[test]
    fn arguments_parse_dtype_cache_and_deduplicate_tensor_names() {
        let arguments = parse_arguments_from(
            [
                "--dtype",
                "bf16",
                "--cache",
                "prepared-cache",
                "model.safetensors",
                "z.weight",
                "a.weight",
                "z.weight",
            ]
            .map(OsString::from),
        )
        .expect("valid benchmark arguments should parse");

        assert_eq!(arguments.target_dtype, TargetDtype::Bf16);
        assert_eq!(
            arguments.cache_directory,
            Some(PathBuf::from("prepared-cache"))
        );
        assert_eq!(arguments.path, PathBuf::from("model.safetensors"));
        assert_eq!(arguments.selected_names, ["a.weight", "z.weight"]);
    }

    #[test]
    fn arguments_require_at_least_one_selected_tensor() {
        let result = parse_arguments_from([OsString::from("model.safetensors")]);

        assert!(result.is_err());
    }

    #[test]
    fn benchmark_passes_observe_cold_transform_then_warm_cache()
    -> Result<(), Box<dyn std::error::Error>> {
        let temporary = TempDir::new()?;
        let checkpoint_path = temporary.path().join("weight.safetensors");
        let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut file = File::create(&checkpoint_path)?;
        file.write_all(
            &u64::try_from(header.len())
                .expect("fixture header length must fit u64")
                .to_le_bytes(),
        )?;
        file.write_all(header)?;
        for value in [1.0_f32, -2.0] {
            file.write_all(&value.to_le_bytes())?;
        }
        file.sync_all()?;

        let cancellation = CancellationToken::new();
        let setup = Checkpoint::open(&checkpoint_path)?;
        let snapshot = setup.snapshot_id(&cancellation)?;
        let plan = build_plan(
            &setup,
            snapshot,
            setup.source_digests(&cancellation)?,
            &["weight".to_owned()],
            TargetDtype::F16,
        )?;
        let preparation = PreparationEngine::with_builtins()?;
        let cache = Cache::open(temporary.path().join("cache"))?;
        reset_selected_prepared_cache(&setup, &plan, &preparation, Some(&cache), &cancellation)?;

        let limits = execution_limits(&plan);
        let cold = run_pass(&setup, &plan, &preparation, Some(&cache), &limits)?;
        let warm = run_pass(&setup, &plan, &preparation, Some(&cache), &limits)?;

        assert_eq!(
            (
                cold.origins.transform,
                cold.origins.cache,
                warm.origins.transform,
                warm.origins.cache,
                cold.phases.contains_key(&ExecutionPhase::Transform),
                warm.phases.contains_key(&ExecutionPhase::CacheLookup),
                cold.phases.contains_key(&ExecutionPhase::Hashing),
                warm.phases.contains_key(&ExecutionPhase::Hashing),
            ),
            (1, 0, 0, 1, true, true, false, false)
        );
        Ok(())
    }
}
