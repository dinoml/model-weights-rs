//! Native carrier normalization and typed prepared-cache delivery contracts.

use model_weights::cache::{Cache, CacheNamespace};
use model_weights::identity::{
    BackendId, ContentDigest, ContractId, ImplementationId, ManifestId, SelectionId, StableName,
};
use model_weights::materialize::{Materializer, PreparedOrigin, WeightDelivery};
use model_weights::native_quantization::{METADATA_KEY, NativeFormat};
use model_weights::packed_prepare::{
    PackedPreparationProvider, PackedPreparationSizes, PackedPrepareRequest,
};
use model_weights::plan::{
    BindingPlan, PlanInputs, Requirement, SourceTensor, TargetTensor, TensorName,
};
use model_weights::prepare::{Layout, PreparationEngine, Representation};
use model_weights::quantization::{LayoutRequirement, QuantizedRoute, RouteCapability, Storage};
use model_weights::tensor::{ByteView, DType};
use model_weights::{CancellationToken, Checkpoint, Result};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn id(name: &str) -> Result<ImplementationId> {
    Ok(ImplementationId::new(
        StableName::parse("test")?,
        StableName::parse(name)?,
        1,
    ))
}

fn file(path: &Path, format: &str, revision: u32, duplicated: bool) -> std::io::Result<()> {
    let descriptor = format!(
        r#"{{"format":"{format}","format_revision":{revision},"encoding":"canonical_rows","encoding_revision":1,"logical_shape":[2,64],"payload":"layer.weight"}}"#
    );
    let entries = if duplicated {
        format!(r#""layer.weight":{descriptor},"layer.weight":{descriptor}"#)
    } else {
        format!(r#""layer.weight":{descriptor}"#)
    };
    let metadata = format!(r#"{{"schema_version":1,"tensors":{{{entries}}}}}"#);
    let header = serde_json::json!({"__metadata__":{METADATA_KEY:metadata},
        "layer.weight":{"dtype":"U8","shape":[2,34],"data_offsets":[0,68]}});
    let mut header = serde_json::to_vec(&header)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&[0; 68])
}

#[test]
fn native_inventory_is_logical_and_rows_can_be_sliced() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("native.safetensors");
    file(&path, "libgguf.s4g64", 1, false).unwrap();
    let checkpoint = Checkpoint::open(path)?;
    let tensor = checkpoint.inventory().tensor("layer.weight").unwrap();
    assert_eq!(tensor.shape(), [2, 64]);
    let Storage::Quantized(storage) = tensor.storage() else {
        panic!("marked carrier became plain U8")
    };
    assert_eq!(storage.encoding(), &NativeFormat::S4G64.encoding()?);
    assert_eq!(storage.span().len(), 68);
    let slice = storage.slice_rows(1, 2)?;
    assert_eq!(slice.span().len(), 34);
    assert_eq!(slice.logical_shape(), [1, 64]);
    Ok(())
}

#[test]
fn unknown_duplicate_and_wrong_revision_descriptors_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid.safetensors");
    for (format, revision, duplicated) in [
        ("libgguf.unknown", 1, false),
        ("libgguf.s4g64", 2, false),
        ("libgguf.s4g64", 1, true),
    ] {
        file(&path, format, revision, duplicated).unwrap();
        let _error = Checkpoint::open(&path).unwrap_err();
    }
}

#[test]
fn libgguf_converter_fixtures_normalize_all_native_formats() -> Result<()> {
    let directory = tempfile::tempdir().unwrap();
    for (name, kind, fixture) in [
        (
            "s4g32",
            NativeFormat::S4G32,
            include_bytes!("fixtures/native_quantization/s4g32.safetensors").as_slice(),
        ),
        (
            "s4g64",
            NativeFormat::S4G64,
            include_bytes!("fixtures/native_quantization/s4g64.safetensors").as_slice(),
        ),
        (
            "s8g32",
            NativeFormat::S8G32,
            include_bytes!("fixtures/native_quantization/s8g32.safetensors").as_slice(),
        ),
        (
            "s4h256",
            NativeFormat::S4H256,
            include_bytes!("fixtures/native_quantization/s4h256.safetensors").as_slice(),
        ),
    ] {
        let path = directory.path().join(format!("{name}.safetensors"));
        std::fs::write(&path, fixture).unwrap();
        let checkpoint = Checkpoint::open(path)?;
        let record = checkpoint.inventory().tensor("layer.weight").unwrap();
        assert_eq!(record.shape(), [2, 256]);
        let Storage::Quantized(storage) = record.storage() else {
            panic!("native carrier was not normalized")
        };
        assert_eq!(storage.encoding(), &kind.encoding()?);
        let Some(model_weights::quantization::MetadataValue::Text(encoder)) = storage
            .metadata()
            .get(&StableName::parse("libgguf.encoder")?)
        else {
            panic!("native encoder provenance was discarded")
        };
        let provenance: serde_json::Value = serde_json::from_str(encoder).unwrap();
        assert_eq!(provenance["name"], "clip_grid");
        assert_eq!(provenance["revision"], 1);
        assert!(matches!(
            checkpoint
                .inventory()
                .tensor("layer.bias")
                .unwrap()
                .storage(),
            Storage::Plain { .. }
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct CopyPacked {
    identity: ImplementationId,
    calls: Arc<AtomicUsize>,
    cancel_during: bool,
}
impl PackedPreparationProvider for CopyPacked {
    fn implementation(&self) -> &ImplementationId {
        &self.identity
    }
    fn validate(
        &self,
        request: PackedPrepareRequest<'_>,
        token: &CancellationToken,
    ) -> Result<PackedPreparationSizes> {
        token.check()?;
        Ok(PackedPreparationSizes {
            output_bytes: request.source.span().len(),
            scratch_bytes: 16,
        })
    }
    fn prepare_into(
        &self,
        _: PackedPrepareRequest<'_>,
        payload: &ByteView,
        _: &BTreeMap<StableName, ByteView>,
        output: &mut [u8],
        scratch: &mut [u8],
        token: &CancellationToken,
    ) -> Result<()> {
        token.check()?;
        assert_eq!(scratch.len(), 16);
        assert!(scratch.iter().all(|b| *b == 0));
        self.calls.fetch_add(1, Ordering::SeqCst);
        output.copy_from_slice(payload.as_slice());
        if self.cancel_during {
            token.cancel();
        }
        Ok(())
    }
}

#[test]
fn host_repack_cold_and_warm_deliver_typed_bytes_without_repacking() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("native.safetensors");
    file(&path, "libgguf.s4g64", 1, false).unwrap();
    let checkpoint = Checkpoint::open(path)?;
    let token = CancellationToken::new();
    let digest = ContentDigest::hash("native-test", [b"contract"]);
    let backend = BackendId::from_digest(digest);
    let inputs = PlanInputs::new(
        ManifestId::from_digest(digest),
        SelectionId::from_digest(digest),
        ContractId::from_digest(digest),
        backend,
        checkpoint.source_digests(&token)?,
    );
    let layout = Layout::custom(
        StableName::parse("test.packed")?,
        1,
        b"exact-layout".as_slice(),
    );
    let capability = RouteCapability::new(
        NativeFormat::S4G64.encoding()?,
        QuantizedRoute::HostRepack {
            target_encoding: id("prepared")?,
        },
        id("pack")?,
        Some(backend),
        LayoutRequirement::from_layout(&layout),
    )?;
    let target = TargetTensor::builder(
        TensorName::parse("layer.weight")?,
        Requirement::Required,
        [2_u64, 64],
        Representation::new(DType::F32, layout),
        68,
    )
    .quantized_route(capability)
    .build()?;
    let sources = checkpoint
        .inventory()
        .iter()
        .map(SourceTensor::try_from)
        .collect::<Result<Vec<_>>>()?;
    let plan = BindingPlan::builder(inputs)
        .sources(sources)
        .targets([target])
        .build()?;
    let mut preparation = PreparationEngine::with_builtins()?;
    let calls = Arc::new(AtomicUsize::new(0));
    preparation.packed_providers_mut().register(CopyPacked {
        identity: id("pack")?,
        calls: Arc::clone(&calls),
        cancel_during: false,
    })?;
    let cache = Cache::open(dir.path().join("cache"))?;
    let materializer =
        Materializer::new(&checkpoint, &plan, &preparation, &token)?.with_cache(&cache);
    let WeightDelivery::PreparedQuantized(cold) =
        materializer.materialize("layer.weight", &token)?
    else {
        panic!("untyped cold delivery")
    };
    assert_eq!(cold.origin(), PreparedOrigin::Transform);
    assert_eq!(cold.bytes().len(), 68);
    assert_eq!(cold.descriptor().logical_shape(), [2, 64]);
    let WeightDelivery::PreparedQuantized(warm) =
        materializer.materialize("layer.weight", &token)?
    else {
        panic!("untyped warm delivery")
    };
    assert_eq!(warm.origin(), PreparedOrigin::Cache);
    assert_eq!(cold.descriptor(), warm.descriptor());
    assert_eq!(cold.bytes().as_slice(), warm.bytes().as_slice());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(warm.resident_bytes() > 68);
    token.cancel();
    let _error = materializer
        .materialize("layer.weight", &token)
        .unwrap_err();
    Ok(())
}

fn packed_plan(
    checkpoint: &Checkpoint,
    backend_label: &[u8],
    implementation: &str,
    layout_revision: u32,
    layout_bytes: &[u8],
) -> Result<BindingPlan> {
    let digest = ContentDigest::hash("native-packed-plan", [backend_label]);
    let backend = BackendId::from_digest(digest);
    let inputs = PlanInputs::new(
        ManifestId::from_digest(digest),
        SelectionId::from_digest(digest),
        ContractId::from_digest(digest),
        backend,
        checkpoint.source_digests(&CancellationToken::new())?,
    );
    let layout = Layout::custom(
        StableName::parse("test.packed")?,
        layout_revision,
        layout_bytes,
    );
    let route = RouteCapability::new(
        NativeFormat::S4G64.encoding()?,
        QuantizedRoute::HostRepack {
            target_encoding: id("prepared")?,
        },
        id(implementation)?,
        Some(backend),
        LayoutRequirement::from_layout(&layout),
    )?;
    let target = TargetTensor::builder(
        TensorName::parse("layer.weight")?,
        Requirement::Required,
        [2_u64, 64],
        Representation::new(DType::I8, layout),
        68,
    )
    .quantized_route(route)
    .build()?;
    BindingPlan::builder(inputs)
        .sources(
            checkpoint
                .inventory()
                .iter()
                .map(SourceTensor::try_from)
                .collect::<Result<Vec<_>>>()?,
        )
        .targets([target])
        .build()
}

fn packed_engine(calls: &Arc<AtomicUsize>, cancel_during: bool) -> Result<PreparationEngine> {
    let mut engine = PreparationEngine::with_builtins()?;
    engine.packed_providers_mut().register(CopyPacked {
        identity: id("pack")?,
        calls: Arc::clone(calls),
        cancel_during,
    })?;
    Ok(engine)
}

#[test]
fn host_repack_addresses_pin_layout_revision_backend_provider_and_source() -> Result<()> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.safetensors");
    file(&path, "libgguf.s4g64", 1, false).unwrap();
    let checkpoint = Checkpoint::open(&path)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = packed_engine(&calls, false)?;
    let token = CancellationToken::new();
    let mut addresses = std::collections::BTreeSet::new();
    for (backend, provider, revision, layout) in [
        (b"cpu".as_slice(), "pack", 1, b"layout".as_slice()),
        (b"gpu", "pack", 1, b"layout"),
        (b"cpu", "pack2", 1, b"layout"),
        (b"cpu", "pack", 2, b"layout"),
        (b"cpu", "pack", 1, b"different"),
    ] {
        let plan = packed_plan(&checkpoint, backend, provider, revision, layout)?;
        // Addressing pins exact provider identities before provider execution.
        let materializer = Materializer::new(&checkpoint, &plan, &engine, &token)?;
        assert!(
            addresses.insert(
                materializer
                    .prepared_cache_address("layer.weight")?
                    .unwrap()
                    .key()
            )
        );
    }
    let changed = directory.path().join("changed.safetensors");
    let mut bytes = std::fs::read(&path).unwrap();
    let last_block = bytes.len() - 34;
    bytes[last_block + 1] = 0x3c;
    *bytes.last_mut().unwrap() = 1;
    std::fs::write(&changed, bytes).unwrap();
    let changed = Checkpoint::open(changed)?;
    let plan = packed_plan(&changed, b"cpu", "pack", 1, b"layout")?;
    let materializer = Materializer::new(&changed, &plan, &engine, &token)?;
    assert!(
        addresses.insert(
            materializer
                .prepared_cache_address("layer.weight")?
                .unwrap()
                .key()
        )
    );
    assert_eq!(addresses.len(), 6);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn host_repack_corrupt_and_truncated_cache_entries_are_rebuilt() -> Result<()> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.safetensors");
    file(&path, "libgguf.s4g64", 1, false).unwrap();
    let checkpoint = Checkpoint::open(path)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = packed_engine(&calls, false)?;
    let token = CancellationToken::new();
    let plan = packed_plan(&checkpoint, b"cpu", "pack", 1, b"layout")?;
    let cache = Cache::open(directory.path().join("cache"))?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &token)?.with_cache(&cache);
    let address = materializer
        .prepared_cache_address("layer.weight")?
        .unwrap();
    let payload = cache
        .entry_path(CacheNamespace::Prepared, address.key())
        .join("payload.bin");
    drop(materializer.materialize("layer.weight", &token)?);
    let mut damaged = std::fs::read(&payload).unwrap();
    damaged[0] ^= 1;
    std::fs::write(&payload, damaged).unwrap();
    let WeightDelivery::PreparedQuantized(rebuilt) =
        materializer.materialize("layer.weight", &token)?
    else {
        panic!("typed rebuild")
    };
    assert_eq!(rebuilt.origin(), PreparedOrigin::Transform);
    drop(rebuilt);
    std::fs::write(&payload, b"truncated").unwrap();
    let WeightDelivery::PreparedQuantized(rebuilt) =
        materializer.materialize("layer.weight", &token)?
    else {
        panic!("typed rebuild")
    };
    assert_eq!(rebuilt.origin(), PreparedOrigin::Transform);
    drop(rebuilt);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let WeightDelivery::PreparedQuantized(warm) =
        materializer.materialize("layer.weight", &token)?
    else {
        panic!("typed hit")
    };
    assert_eq!(warm.origin(), PreparedOrigin::Cache);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    Ok(())
}

#[test]
fn host_repack_cancellation_during_packing_never_publishes() -> Result<()> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.safetensors");
    file(&path, "libgguf.s4g64", 1, false).unwrap();
    let checkpoint = Checkpoint::open(path)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = packed_engine(&calls, true)?;
    let token = CancellationToken::new();
    let plan = packed_plan(&checkpoint, b"cpu", "pack", 1, b"layout")?;
    let cache = Cache::open(directory.path().join("cache"))?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &token)?.with_cache(&cache);
    let address = materializer
        .prepared_cache_address("layer.weight")?
        .unwrap();
    let error = materializer
        .materialize("layer.weight", &token)
        .unwrap_err();
    assert_eq!(error.category(), model_weights::ErrorCategory::Cancelled);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        !cache
            .entry_path(CacheNamespace::Prepared, address.key())
            .exists()
    );
    Ok(())
}

#[test]
fn host_repack_concurrent_publishers_deliver_one_consistent_typed_entry() -> Result<()> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.safetensors");
    file(&path, "libgguf.s4g64", 1, false).unwrap();
    let checkpoint = Checkpoint::open(path)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = packed_engine(&calls, false)?;
    let token = CancellationToken::new();
    let plan = packed_plan(&checkpoint, b"cpu", "pack", 1, b"layout")?;
    let cache = Cache::open(directory.path().join("cache"))?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &token)?.with_cache(&cache);
    std::thread::scope(|scope| {
        let tasks = (0..2)
            .map(|_| scope.spawn(|| materializer.materialize("layer.weight", &token)))
            .collect::<Vec<_>>();
        for task in tasks {
            let WeightDelivery::PreparedQuantized(weight) = task.join().unwrap().unwrap() else {
                panic!("typed concurrent result")
            };
            assert_eq!(weight.bytes().as_slice(), [0_u8; 68]);
        }
    });
    assert!((1..=2).contains(&calls.load(Ordering::SeqCst)));
    let before = calls.load(Ordering::SeqCst);
    let WeightDelivery::PreparedQuantized(warm) =
        materializer.materialize("layer.weight", &token)?
    else {
        panic!("typed warm result")
    };
    assert_eq!(warm.origin(), PreparedOrigin::Cache);
    assert_eq!(calls.load(Ordering::SeqCst), before);
    Ok(())
}

#[test]
fn host_repack_pipeline_checks_output_and_scratch_bounds_before_execution() -> Result<()> {
    use model_weights::limits::ExecutionLimits;
    use model_weights::pipeline::{Pipeline, PreparedItem};
    use model_weights::telemetry::NoopObserver;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.safetensors");
    file(&path, "libgguf.s4g64", 1, false).unwrap();
    let checkpoint = Checkpoint::open(path)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = packed_engine(&calls, false)?;
    let token = CancellationToken::new();
    let plan = packed_plan(&checkpoint, b"cpu", "pack", 1, b"layout")?;
    let materializer = Materializer::new(&checkpoint, &plan, &engine, &token)?;
    for (scratch, prepared) in [(15, 1024), (1024, 67)] {
        let pipeline = Pipeline::new(ExecutionLimits {
            workers: 1,
            scratch_bytes: scratch,
            prepared_bytes: prepared,
            ..ExecutionLimits::default()
        })?;
        let mut delivered = 0;
        let mut sink =
            |_: u64, _: PreparedItem<WeightDelivery>, _: &CancellationToken| -> Result<()> {
                delivered += 1;
                Ok(())
            };
        let error = materializer
            .execute(&pipeline, &mut sink, &NoopObserver)
            .unwrap_err();
        assert_eq!(
            error.category(),
            model_weights::ErrorCategory::ResourceLimit
        );
        assert_eq!(delivered, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    Ok(())
}
