use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use half::{bf16, f16};

use super::{
    Layout, OutputStrategy, PreparationEngine, PreparationProvider, PrepareRequest,
    ProviderRegistry, Representation, TransformSpec, builtin_contiguous_implementation,
};
use crate::identity::{ImplementationId, StableName};
use crate::plan::PlannedTransform;
use crate::tensor::{ByteView, DType};
use crate::{CancellationToken, Error, ErrorCategory, Result};

fn bytes(values: impl Into<Box<[u8]>>) -> ByteView {
    ByteView::from_boxed(values.into())
}

fn implementation(provider: &str, operation: &str, version: u32) -> Result<ImplementationId> {
    Ok(ImplementationId::new(
        StableName::parse(provider)?,
        StableName::parse(operation)?,
        version,
    ))
}

fn contiguous_transform(source: DType, target: DType) -> Result<TransformSpec> {
    Ok(TransformSpec::new(
        builtin_contiguous_implementation()?,
        Representation::contiguous(source),
        Representation::contiguous(target),
    ))
}

fn prepare(
    source: &ByteView,
    source_dtype: DType,
    target_dtype: DType,
    shape: &[u64],
) -> Result<ByteView> {
    let transform = contiguous_transform(source_dtype, target_dtype)?;
    let expected_output_bytes = target_dtype.byte_len(shape)?;
    let request = PrepareRequest::new(&transform, shape, source, expected_output_bytes);
    PreparationEngine::with_builtins()?.prepare(&request)
}

fn f32_bytes(values: &[f32]) -> Box<[u8]> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn u16_bytes(values: &[u16]) -> Box<[u8]> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn read_u16(bytes: &[u8]) -> Result<Box<[u16]>> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let [low, high] = chunk else {
                return Err(Error::integrity("test encountered an invalid u16 width"));
            };
            Ok(u16::from_le_bytes([*low, *high]))
        })
        .collect::<Result<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

fn read_f32(bytes: &[u8]) -> Result<Box<[f32]>> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let [a, b, c, d] = chunk else {
                return Err(Error::integrity("test encountered an invalid f32 width"));
            };
            Ok(f32::from_le_bytes([*a, *b, *c, *d]))
        })
        .collect::<Result<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

#[test]
fn contiguous_identity_preserves_the_exact_byte_view() -> Result<()> {
    let owner = bytes(Box::<[u8]>::from([
        0xaa, 0xbb, 0x00, 0x00, 0x80, 0x3f, 0xcc, 0xdd,
    ]));
    let source = owner.slice(2..6)?;
    let prepared = prepare(&source, DType::F32, DType::F32, &[])?;

    assert_eq!(prepared.as_slice().as_ptr(), source.as_slice().as_ptr());
    assert_eq!(prepared.len(), source.len());
    assert_eq!(prepared.as_slice(), source.as_slice());
    Ok(())
}

#[test]
fn f32_casts_have_reference_values_and_special_value_parity() -> Result<()> {
    let values = [
        0.0_f32,
        -0.0,
        1.0,
        -2.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let source = bytes(f32_bytes(&values));

    let half_output = prepare(
        &source,
        DType::F32,
        DType::F16,
        &[u64::try_from(values.len()).map_err(|_error| Error::limit("test shape overflow"))?],
    )?;
    let f16_bits = read_u16(half_output.as_slice())?;
    assert_eq!(
        &f16_bits[..6],
        &[0x0000, 0x8000, 0x3c00, 0xc100, 0x7c00, 0xfc00]
    );
    assert!(f16::from_bits(f16_bits[6]).is_nan());

    let brain_output = prepare(
        &source,
        DType::F32,
        DType::Bf16,
        &[u64::try_from(values.len()).map_err(|_error| Error::limit("test shape overflow"))?],
    )?;
    let bf16_bits = read_u16(brain_output.as_slice())?;
    assert_eq!(
        &bf16_bits[..6],
        &[0x0000, 0x8000, 0x3f80, 0xc020, 0x7f80, 0xff80]
    );
    assert!(bf16::from_bits(bf16_bits[6]).is_nan());
    Ok(())
}

#[test]
fn half_precision_casts_cover_both_directions() -> Result<()> {
    let f16_source = bytes(u16_bytes(&[0x3e00, 0xc100, 0x8000, 0x7c00]));
    let f16_as_f32 = prepare(&f16_source, DType::F16, DType::F32, &[4])?;
    assert_eq!(
        read_f32(f16_as_f32.as_slice())?.as_ref(),
        &[1.5, -2.5, -0.0, f32::INFINITY]
    );
    let f16_as_bf16 = prepare(&f16_source, DType::F16, DType::Bf16, &[4])?;
    assert_eq!(
        read_u16(f16_as_bf16.as_slice())?.as_ref(),
        &[0x3fc0, 0xc020, 0x8000, 0x7f80]
    );

    let bf16_source = bytes(u16_bytes(&[0x3fc0, 0xc020, 0x8000, 0x7f80]));
    let bf16_as_f32 = prepare(&bf16_source, DType::Bf16, DType::F32, &[4])?;
    assert_eq!(
        read_f32(bf16_as_f32.as_slice())?.as_ref(),
        &[1.5, -2.5, -0.0, f32::INFINITY]
    );
    let bf16_as_f16 = prepare(&bf16_source, DType::Bf16, DType::F16, &[4])?;
    assert_eq!(
        read_u16(bf16_as_f16.as_slice())?.as_ref(),
        &[0x3e00, 0xc100, 0x8000, 0x7c00]
    );
    Ok(())
}

#[test]
fn casts_process_values_across_internal_block_boundaries() -> Result<()> {
    let values = (0..20_000_u16)
        .map(|value| f32::from(value) / 32.0)
        .collect::<Vec<_>>();
    let source = bytes(f32_bytes(&values));
    let prepared = prepare(&source, DType::F32, DType::F16, &[20_000])?;
    let actual = read_u16(prepared.as_slice())?;

    assert_eq!(actual.len(), values.len());
    for (actual, source) in actual.iter().zip(&values) {
        assert_eq!(*actual, f16::from_f32(*source).to_bits());
    }
    Ok(())
}

#[test]
fn preparation_rejects_overflow_and_source_length_mismatch() -> Result<()> {
    let source = bytes(Box::<[u8]>::from([0_u8; 4]));
    let transform = contiguous_transform(DType::F32, DType::F16)?;
    let overflow_request = PrepareRequest::new(&transform, &[u64::MAX, 2], &source, u64::MAX);
    let overflow = PreparationEngine::with_builtins()?.prepare(&overflow_request);
    assert_eq!(
        overflow.err().map(|error| error.category()),
        Some(ErrorCategory::ResourceLimit)
    );

    let mismatch_transform = contiguous_transform(DType::F32, DType::F16)?;
    let mismatch_request = PrepareRequest::new(&mismatch_transform, &[2], &source, 4);
    let mismatch = PreparationEngine::with_builtins()?.prepare(&mismatch_request);
    assert_eq!(
        mismatch.err().map(|error| error.category()),
        Some(ErrorCategory::Integrity)
    );
    Ok(())
}

#[test]
fn preparation_rejects_unsupported_dtype_and_layout_pairs() -> Result<()> {
    let source = bytes(Box::<[u8]>::from([0_u8; 8]));
    let dtype_transform = contiguous_transform(DType::F64, DType::F32)?;
    let dtype_request = PrepareRequest::new(&dtype_transform, &[], &source, 4);
    let dtype_result = PreparationEngine::with_builtins()?.prepare(&dtype_request);
    assert_eq!(
        dtype_result.err().map(|error| error.category()),
        Some(ErrorCategory::Unsupported)
    );

    let layout = Layout::custom(
        StableName::parse("example.runtime.layout")?,
        1,
        Box::<[u8]>::from([]),
    );
    let layout_transform = TransformSpec::new(
        builtin_contiguous_implementation()?,
        Representation::contiguous(DType::F64),
        Representation::new(DType::F64, layout),
    );
    let layout_request = PrepareRequest::new(&layout_transform, &[], &source, 8);
    let layout_result = PreparationEngine::with_builtins()?.prepare(&layout_request);
    assert_eq!(
        layout_result.err().map(|error| error.category()),
        Some(ErrorCategory::Unsupported)
    );
    Ok(())
}

#[derive(Debug)]
struct ReuseProvider {
    implementation: ImplementationId,
}

impl PreparationProvider for ReuseProvider {
    fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    fn validate(
        &self,
        request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<OutputStrategy> {
        cancellation.check()?;
        if u64::try_from(request.source().len())
            .map_err(|_error| Error::limit("source length does not fit u64"))?
            != request.expected_output_bytes()
        {
            return Err(Error::integrity("custom provider input length mismatch"));
        }
        Ok(OutputStrategy::ReuseSource)
    }

    fn prepare_into(
        &self,
        _request: &PrepareRequest<'_>,
        _output: &mut [u8],
        _scratch: &mut [u8],
        _cancellation: &CancellationToken,
    ) -> Result<()> {
        Err(Error::invalid(
            "reuse provider must not receive an allocated output",
        ))
    }
}

#[derive(Debug)]
struct MismatchedSizeProvider {
    implementation: ImplementationId,
    executed: Arc<AtomicBool>,
}

impl PreparationProvider for MismatchedSizeProvider {
    fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    fn validate(
        &self,
        _request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<OutputStrategy> {
        cancellation.check()?;
        Ok(OutputStrategy::Allocate {
            output_bytes: 8,
            scratch_bytes: 0,
        })
    }

    fn prepare_into(
        &self,
        _request: &PrepareRequest<'_>,
        _output: &mut [u8],
        _scratch: &mut [u8],
        _cancellation: &CancellationToken,
    ) -> Result<()> {
        self.executed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug)]
struct ScratchProvider {
    implementation: ImplementationId,
    output_bytes: u64,
    scratch_bytes: u64,
    executed: Arc<AtomicBool>,
    observed_storage: Arc<Mutex<Option<(usize, usize, bool)>>>,
}

impl PreparationProvider for ScratchProvider {
    fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    fn validate(
        &self,
        _request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<OutputStrategy> {
        cancellation.check()?;
        Ok(OutputStrategy::Allocate {
            output_bytes: self.output_bytes,
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
        let scratch_was_zeroed = scratch.iter().all(|byte| *byte == 0);
        *self
            .observed_storage
            .lock()
            .map_err(|_poisoned| Error::limit("scratch observation lock was poisoned"))? =
            Some((output.len(), scratch.len(), scratch_was_zeroed));
        self.executed.store(true, Ordering::SeqCst);
        if output.len() != request.source().len() {
            return Err(Error::integrity(
                "scratch provider source and output lengths differ",
            ));
        }
        scratch.fill(0xa5);
        output.copy_from_slice(request.source().as_slice());
        cancellation.check()
    }
}

#[derive(Debug)]
struct CancellingValidationProvider {
    implementation: ImplementationId,
    validated: Arc<AtomicBool>,
    executed: Arc<AtomicBool>,
}

impl PreparationProvider for CancellingValidationProvider {
    fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    fn validate(
        &self,
        _request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<OutputStrategy> {
        self.validated.store(true, Ordering::SeqCst);
        cancellation.cancel();
        cancellation.check()?;
        Ok(OutputStrategy::ReuseSource)
    }

    fn prepare_into(
        &self,
        _request: &PrepareRequest<'_>,
        _output: &mut [u8],
        _scratch: &mut [u8],
        _cancellation: &CancellationToken,
    ) -> Result<()> {
        self.executed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn registry_resolves_the_exact_provider_version() -> Result<()> {
    let version_one = implementation("example", "validate-layout", 1)?;
    let version_two = implementation("example", "validate-layout", 2)?;
    let mut registry = ProviderRegistry::new();
    registry.register(ReuseProvider {
        implementation: version_one.clone(),
    })?;
    registry.register(ReuseProvider {
        implementation: version_two.clone(),
    })?;

    assert_eq!(
        registry.resolve(&version_one)?.implementation().version(),
        1
    );
    assert_eq!(
        registry.resolve(&version_two)?.implementation().version(),
        2
    );
    let missing = implementation("example", "validate-layout", 3)?;
    assert_eq!(
        registry
            .resolve(&missing)
            .err()
            .map(|error| error.category()),
        Some(ErrorCategory::Unsupported)
    );
    Ok(())
}

#[test]
fn custom_provider_can_reuse_bytes_without_backend_policy_in_core() -> Result<()> {
    let provider_id = implementation("python.reference", "diffusers-layout", 7)?;
    let mut registry = ProviderRegistry::new();
    registry.register(ReuseProvider {
        implementation: provider_id.clone(),
    })?;
    let engine = PreparationEngine::new(registry);

    let source = bytes(Box::<[u8]>::from([1_u8, 2, 3, 4]));
    let representation = Representation::contiguous(DType::U8);
    let transform = TransformSpec::new(provider_id, representation.clone(), representation);
    let request = PrepareRequest::new(&transform, &[4], &source, 4);
    let prepared = engine.prepare(&request)?;

    assert_eq!(prepared.as_slice().as_ptr(), source.as_slice().as_ptr());
    assert_eq!(prepared.as_slice(), source.as_slice());
    Ok(())
}

#[test]
fn provider_size_is_checked_before_allocation_and_execution() -> Result<()> {
    let provider_id = implementation("example", "sized-output", 1)?;
    let executed = Arc::new(AtomicBool::new(false));
    let mut registry = ProviderRegistry::new();
    registry.register(MismatchedSizeProvider {
        implementation: provider_id.clone(),
        executed: Arc::clone(&executed),
    })?;
    let engine = PreparationEngine::new(registry);

    let source = bytes(Box::<[u8]>::from([1_u8, 2, 3, 4]));
    let representation = Representation::contiguous(DType::U8);
    let transform = TransformSpec::new(provider_id, representation.clone(), representation);
    let request = PrepareRequest::new(&transform, &[4], &source, 4);
    let result = engine.prepare(&request);

    assert_eq!(
        result.err().map(|error| error.category()),
        Some(ErrorCategory::Integrity)
    );
    assert!(!executed.load(Ordering::SeqCst));
    Ok(())
}

#[test]
fn provider_scratch_is_checked_before_allocation_and_execution() -> Result<()> {
    let provider_id = implementation("example", "sized-scratch", 1)?;
    let executed = Arc::new(AtomicBool::new(false));
    let observed_storage = Arc::new(Mutex::new(None));
    let mut registry = ProviderRegistry::new();
    registry.register(ScratchProvider {
        implementation: provider_id.clone(),
        output_bytes: 4,
        scratch_bytes: 7,
        executed: Arc::clone(&executed),
        observed_storage: Arc::clone(&observed_storage),
    })?;
    let engine = PreparationEngine::new(registry);

    let source = bytes(Box::<[u8]>::from([1_u8, 2, 3, 4]));
    let representation = Representation::contiguous(DType::U8);
    let transform = TransformSpec::new(provider_id, representation.clone(), representation);
    let request = PrepareRequest::new(&transform, &[4], &source, 4).with_expected_scratch_bytes(6);
    let result = engine.prepare(&request);

    assert_eq!(
        result.err().map(|error| error.category()),
        Some(ErrorCategory::Integrity)
    );
    assert!(!executed.load(Ordering::SeqCst));
    assert_eq!(
        *observed_storage
            .lock()
            .map_err(|_poisoned| Error::limit("scratch observation lock was poisoned"))?,
        None
    );
    Ok(())
}

#[test]
fn provider_receives_exact_zeroed_caller_owned_scratch() -> Result<()> {
    let provider_id = implementation("example", "exact-scratch", 1)?;
    let executed = Arc::new(AtomicBool::new(false));
    let observed_storage = Arc::new(Mutex::new(None));
    let mut registry = ProviderRegistry::new();
    registry.register(ScratchProvider {
        implementation: provider_id.clone(),
        output_bytes: 4,
        scratch_bytes: 7,
        executed: Arc::clone(&executed),
        observed_storage: Arc::clone(&observed_storage),
    })?;
    let engine = PreparationEngine::new(registry);

    let source = bytes(Box::<[u8]>::from([1_u8, 2, 3, 4]));
    let representation = Representation::contiguous(DType::U8);
    let transform = TransformSpec::new(provider_id, representation.clone(), representation);
    let request = PrepareRequest::new(&transform, &[4], &source, 4).with_expected_scratch_bytes(7);
    let prepared = engine.prepare(&request)?;

    assert_eq!(prepared.as_slice(), source.as_slice());
    assert!(executed.load(Ordering::SeqCst));
    assert_eq!(
        *observed_storage
            .lock()
            .map_err(|_poisoned| Error::limit("scratch observation lock was poisoned"))?,
        Some((4, 7, true))
    );
    Ok(())
}

#[test]
fn custom_provider_validation_observes_cancellation() -> Result<()> {
    let provider_id = implementation("example", "cancel-validation", 1)?;
    let validated = Arc::new(AtomicBool::new(false));
    let executed = Arc::new(AtomicBool::new(false));
    let mut registry = ProviderRegistry::new();
    registry.register(CancellingValidationProvider {
        implementation: provider_id.clone(),
        validated: Arc::clone(&validated),
        executed: Arc::clone(&executed),
    })?;
    let engine = PreparationEngine::new(registry);

    let source = bytes(Box::<[u8]>::from([1_u8, 2, 3, 4]));
    let representation = Representation::contiguous(DType::U8);
    let transform = TransformSpec::new(provider_id, representation.clone(), representation);
    let request = PrepareRequest::new(&transform, &[4], &source, 4);
    let cancellation = CancellationToken::new();
    let result = engine.prepare_with_cancellation(&request, &cancellation);

    assert_eq!(
        result.err().map(|error| error.category()),
        Some(ErrorCategory::Cancelled)
    );
    assert!(validated.load(Ordering::SeqCst));
    assert!(!executed.load(Ordering::SeqCst));
    assert!(cancellation.is_cancelled());
    Ok(())
}

#[test]
fn transform_descriptors_round_trip_through_json() -> Result<()> {
    let layout = Layout::custom(
        StableName::parse("example.runtime.kernels")?,
        3,
        Box::<[u8]>::from([1_u8, 4, 9]),
    );
    let transform = TransformSpec::new(
        implementation("example", "pack", 11)?,
        Representation::contiguous(DType::F32),
        Representation::new(DType::F16, layout),
    );

    let encoded = serde_json::to_vec(&transform).map_err(|source| {
        Error::with_source(ErrorCategory::InvalidFormat, "test encode", source)
    })?;
    let decoded: TransformSpec = serde_json::from_slice(&encoded).map_err(|source| {
        Error::with_source(ErrorCategory::InvalidFormat, "test decode", source)
    })?;

    assert_eq!(decoded, transform);
    assert_eq!(decoded.target(), transform.target());
    Ok(())
}

#[test]
fn providers_can_be_shared_by_cloned_registries() -> Result<()> {
    let provider_id = implementation("example", "shared", 1)?;
    let mut registry = ProviderRegistry::new();
    registry.register_shared(Arc::new(ReuseProvider {
        implementation: provider_id.clone(),
    }))?;
    let cloned = registry.clone();

    assert_eq!(cloned.resolve(&provider_id)?.implementation(), &provider_id);
    Ok(())
}

#[test]
fn planned_transform_chain_uses_each_pinned_output_size() -> Result<()> {
    let source = bytes(f32_bytes(&[1.5, -2.5, 0.0, f32::INFINITY]));
    let first = PlannedTransform::new(
        contiguous_transform(DType::F32, DType::F16)?,
        DType::F16.byte_len(&[4])?,
    );
    let second = PlannedTransform::new(
        contiguous_transform(DType::F16, DType::Bf16)?,
        DType::Bf16.byte_len(&[4])?,
    );
    let prepared =
        PreparationEngine::with_builtins()?.prepare_chain(&[first, second], &[4], &source)?;

    assert_eq!(
        read_u16(prepared.as_slice())?.as_ref(),
        &[0x3fc0, 0xc020, 0x0000, 0x7f80]
    );
    Ok(())
}

#[test]
fn planned_transform_chain_observes_pre_cancelled_token() -> Result<()> {
    let source = bytes(f32_bytes(&[1.0]));
    let transform = PlannedTransform::new(
        contiguous_transform(DType::F32, DType::F16)?,
        DType::F16.byte_len(&[])?,
    );
    let cancellation = crate::CancellationToken::new();
    cancellation.cancel();
    let result = PreparationEngine::with_builtins()?.prepare_chain_with_cancellation(
        &[transform],
        &[],
        &source,
        &cancellation,
    );

    assert_eq!(
        result.err().map(|error| error.category()),
        Some(ErrorCategory::Cancelled)
    );
    Ok(())
}
