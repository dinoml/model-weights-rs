//! Deterministic host preparation with versioned, consumer-extensible providers.
//!
//! Plans identify byte-affecting implementations with [`ImplementationId`].
//! This module resolves those exact identities, validates a transform before
//! allocating, and either retains an existing [`ByteView`] or writes one final
//! host buffer. Backend policy and device execution remain consumer concerns.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use crate::identity::{ImplementationId, StableName};
use crate::plan::PlannedTransform;
use crate::tensor::{ByteView, DType};
use crate::{CancellationToken, Error, ErrorCategory, Result};

const BUILTIN_PROVIDER: &str = "model-weights";
const BUILTIN_CONTIGUOUS_CAST: &str = "contiguous-cast";

/// The byte-affecting version of the built-in contiguous cast implementation.
pub const BUILTIN_CONTIGUOUS_CAST_VERSION: u32 = 1;

// Keep each source block cache-sized while writing directly into final storage.
const CAST_BLOCK_ELEMENTS: usize = 16 * 1024;

/// Describes the physical byte layout required by a tensor consumer.
///
/// Custom descriptors are opaque canonical bytes interpreted only by their
/// provider. This keeps runtime-specific ABI policy outside this crate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Layout {
    /// Contiguous row-major elements with no padding.
    Contiguous,
    /// A consumer-defined, versioned physical layout descriptor.
    Custom {
        /// Stable consumer or format-defined layout name.
        name: StableName,
        /// Version of the layout descriptor ABI.
        version: u32,
        /// Canonical provider-defined descriptor bytes.
        parameters: Box<[u8]>,
    },
}

impl Layout {
    /// Creates a consumer-defined layout from canonical descriptor bytes.
    #[must_use]
    pub fn custom(name: StableName, version: u32, parameters: impl Into<Box<[u8]>>) -> Self {
        Self::Custom {
            name,
            version,
            parameters: parameters.into(),
        }
    }

    /// Returns whether the layout is contiguous row-major storage.
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        matches!(self, Self::Contiguous)
    }

    /// Returns the custom layout fields, if this is a custom descriptor.
    #[must_use]
    pub fn custom_parts(&self) -> Option<(&StableName, u32, &[u8])> {
        match self {
            Self::Custom {
                name,
                version,
                parameters,
            } => Some((name, *version, parameters)),
            Self::Contiguous => None,
        }
    }
}

/// Describes one tensor's scalar dtype and physical layout.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Representation {
    dtype: DType,
    layout: Layout,
}

impl Representation {
    /// Creates a representation from a scalar dtype and physical layout.
    #[must_use]
    pub const fn new(dtype: DType, layout: Layout) -> Self {
        Self { dtype, layout }
    }

    /// Creates a contiguous row-major representation.
    #[must_use]
    pub const fn contiguous(dtype: DType) -> Self {
        Self::new(dtype, Layout::Contiguous)
    }

    /// Returns the scalar storage dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the physical layout descriptor.
    #[must_use]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// Pins one source-to-target transform to an exact provider implementation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransformSpec {
    implementation: ImplementationId,
    source: Representation,
    target: Representation,
}

impl TransformSpec {
    /// Creates a fully versioned transform specification.
    #[must_use]
    pub const fn new(
        implementation: ImplementationId,
        source: Representation,
        target: Representation,
    ) -> Self {
        Self {
            implementation,
            source,
            target,
        }
    }

    /// Returns the exact byte-affecting implementation identity.
    #[must_use]
    pub const fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    /// Returns the required source representation.
    #[must_use]
    pub const fn source(&self) -> &Representation {
        &self.source
    }

    /// Returns the resulting target representation.
    #[must_use]
    pub const fn target(&self) -> &Representation {
        &self.target
    }
}

/// Borrows all inputs needed to validate and execute one transform.
#[derive(Debug, Clone, Copy)]
pub struct PrepareRequest<'a> {
    transform: &'a TransformSpec,
    shape: &'a [u64],
    source: &'a ByteView,
    expected_output_bytes: u64,
    expected_scratch_bytes: u64,
}

impl<'a> PrepareRequest<'a> {
    /// Creates a zero-scratch request without interpreting provider policy.
    ///
    /// `expected_output_bytes` comes from the approved binding plan and is
    /// checked against the selected provider before output allocation.
    /// Use [`Self::with_expected_scratch_bytes`] for a transform whose approved
    /// plan declares caller-owned workspace.
    #[must_use]
    pub const fn new(
        transform: &'a TransformSpec,
        shape: &'a [u64],
        source: &'a ByteView,
        expected_output_bytes: u64,
    ) -> Self {
        Self {
            transform,
            shape,
            source,
            expected_output_bytes,
            expected_scratch_bytes: 0,
        }
    }

    /// Sets the exact caller-owned workspace length recorded by the plan.
    #[must_use]
    pub const fn with_expected_scratch_bytes(mut self, expected_scratch_bytes: u64) -> Self {
        self.expected_scratch_bytes = expected_scratch_bytes;
        self
    }

    /// Returns the versioned transform specification.
    #[must_use]
    pub const fn transform(&self) -> &TransformSpec {
        self.transform
    }

    /// Returns the logical tensor shape.
    #[must_use]
    pub const fn shape(&self) -> &[u64] {
        self.shape
    }

    /// Returns the immutable source byte view.
    #[must_use]
    pub const fn source(&self) -> &ByteView {
        self.source
    }

    /// Returns the output byte length recorded by the approved plan.
    #[must_use]
    pub const fn expected_output_bytes(&self) -> u64 {
        self.expected_output_bytes
    }

    /// Returns the workspace byte length recorded by the approved plan.
    #[must_use]
    pub const fn expected_scratch_bytes(&self) -> u64 {
        self.expected_scratch_bytes
    }
}

/// Describes validated host storage needed by a preparation provider.
///
/// The engine obtains this value before allocating. The enum is extensible so
/// future executors can add consumer-owned device routes without changing
/// serialized [`TransformSpec`] semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputStrategy {
    /// Preserve the source owner and exact byte range without copying.
    ReuseSource,
    /// Allocate final output and transient host workspace.
    ///
    /// The default provider execution path supplies zero-initialized output
    /// and workspace slices. A provider may instead build the final output
    /// safely in order through [`PreparationProvider::prepare_allocated`].
    Allocate {
        /// Exact number of output bytes.
        output_bytes: u64,
        /// Exact number of caller-owned workspace bytes.
        scratch_bytes: u64,
    },
}

/// Implements one exact, versioned preparation recipe.
///
/// Providers may represent custom layouts, reference conversions, language
/// bindings, or runtime-specific recipes. They validate capabilities and exact
/// output/workspace sizes before the engine allocates. A future device executor
/// can resolve the same implementation identity while preserving plan and
/// cache semantics.
pub trait PreparationProvider: Debug + Send + Sync {
    /// Returns the exact identity registered for this implementation.
    fn implementation(&self) -> &ImplementationId;

    /// Validates support, input metadata, and required output storage.
    ///
    /// This method is metadata-only: it must not read or transform payload
    /// bytes, allocate output or size-dependent workspace, or perform other
    /// unbounded work. It must observe `cancellation` around any bounded
    /// validation loop.
    ///
    /// # Errors
    ///
    /// Returns an error when the representation pair, shape, source byte
    /// length, or planned output/workspace length is invalid or unsupported.
    fn validate(
        &self,
        request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<OutputStrategy>;

    /// Writes a previously validated transform using caller-owned storage.
    ///
    /// The default [`Self::prepare_allocated`] implementation calls this method
    /// only for [`OutputStrategy::Allocate`] and supplies zero-initialized
    /// output and scratch slices at exactly the validated lengths.
    /// Implementations must initialize the complete output deterministically.
    /// All size-dependent workspace must come from `scratch`; providers must
    /// not make undeclared host or device workspace allocations.
    ///
    /// Providers performing block or tile work must check `cancellation` at
    /// bounded intervals. This keeps cancellation latency independent of the
    /// total tensor size.
    ///
    /// # Errors
    ///
    /// Returns an error when execution cannot satisfy the validated recipe or
    /// cooperative cancellation is observed.
    fn prepare_into(
        &self,
        request: &PrepareRequest<'_>,
        output: &mut [u8],
        scratch: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<()>;

    /// Allocates and initializes the complete owned output for one transform.
    ///
    /// The engine calls this only after [`Self::validate`] and exact storage
    /// validation. The default implementation preserves the provider contract
    /// by allocating zero-initialized output and scratch slices before calling
    /// [`Self::prepare_into`]. Providers whose output can be built safely in
    /// order may override this method to avoid initializing bytes twice.
    ///
    /// An override must return exactly `output_len` initialized bytes, use no
    /// size-dependent workspace beyond `scratch_len`, and retain the bounded
    /// cancellation behavior documented by [`Self::prepare_into`].
    ///
    /// # Errors
    ///
    /// Returns an allocation, cancellation, or provider execution error.
    fn prepare_allocated(
        &self,
        request: &PrepareRequest<'_>,
        output_len: usize,
        scratch_len: usize,
        cancellation: &CancellationToken,
    ) -> Result<Box<[u8]>> {
        cancellation.check()?;
        let mut output = allocate_zeroed(output_len, "prepared output allocation failed")?;
        cancellation.check()?;
        let mut scratch = allocate_zeroed(scratch_len, "preparation scratch allocation failed")?;
        self.prepare_into(request, &mut output, &mut scratch, cancellation)?;
        Ok(output)
    }
}

/// Stores preparation providers under exact implementation identities.
#[derive(Debug, Clone)]
pub struct ProviderRegistry {
    providers: BTreeMap<ImplementationId, Arc<dyn PreparationProvider>>,
}

impl ProviderRegistry {
    /// Creates an empty provider registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            providers: BTreeMap::new(),
        }
    }

    /// Creates a registry containing the built-in contiguous provider.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if a built-in stable name cannot be
    /// constructed.
    pub fn with_builtins() -> Result<Self> {
        let mut registry = Self::new();
        registry.register(ContiguousProvider::new()?)?;
        Ok(registry)
    }

    /// Registers an owned provider under its exact implementation version.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if that exact identity is already
    /// registered.
    pub fn register<P>(&mut self, provider: P) -> Result<()>
    where
        P: PreparationProvider + 'static,
    {
        self.register_shared(Arc::new(provider))
    }

    /// Registers a shared provider under its exact implementation version.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if that exact identity is already
    /// registered.
    pub fn register_shared(&mut self, provider: Arc<dyn PreparationProvider>) -> Result<()> {
        let implementation = provider.implementation().clone();
        if self.providers.contains_key(&implementation) {
            return Err(Error::invalid(format!(
                "preparation provider {} / {} version {} is already registered",
                implementation.provider(),
                implementation.operation(),
                implementation.version()
            )));
        }
        self.providers.insert(implementation, provider);
        Ok(())
    }

    /// Resolves an exact provider implementation and version.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-capability error when the exact provider,
    /// operation, and version are not registered.
    pub fn resolve(&self, implementation: &ImplementationId) -> Result<&dyn PreparationProvider> {
        self.providers
            .get(implementation)
            .map(Arc::as_ref)
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "no preparation provider registered for {} / {} version {}",
                    implementation.provider(),
                    implementation.operation(),
                    implementation.version()
                ))
            })
    }

    /// Returns the number of exact implementation versions registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Returns whether no preparation providers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Executes validated host transforms through a versioned provider registry.
#[derive(Debug, Clone)]
pub struct PreparationEngine {
    providers: ProviderRegistry,
}

impl PreparationEngine {
    /// Creates an engine from a caller-configured registry.
    #[must_use]
    pub const fn new(providers: ProviderRegistry) -> Self {
        Self { providers }
    }

    /// Creates an engine containing the built-in contiguous provider.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if a built-in stable name cannot be
    /// constructed.
    pub fn with_builtins() -> Result<Self> {
        Ok(Self::new(ProviderRegistry::with_builtins()?))
    }

    /// Returns the provider registry.
    #[must_use]
    pub const fn providers(&self) -> &ProviderRegistry {
        &self.providers
    }

    /// Returns the provider registry for explicit consumer registration.
    #[must_use]
    pub const fn providers_mut(&mut self) -> &mut ProviderRegistry {
        &mut self.providers
    }

    /// Validates and prepares one tensor into immutable host bytes.
    ///
    /// Identity transforms clone only the [`ByteView`] handle, preserving its
    /// owner and exact range. Allocating providers receive exact output and
    /// caller-owned scratch buffers after capability and size validation.
    ///
    /// # Errors
    ///
    /// Returns an error for an unregistered implementation, unsupported
    /// representation pair, invalid shape or length, allocation-size overflow,
    /// or provider execution failure.
    pub fn prepare(&self, request: &PrepareRequest<'_>) -> Result<ByteView> {
        self.prepare_with_cancellation(request, &CancellationToken::new())
    }

    /// Validates and prepares one tensor with cooperative cancellation.
    ///
    /// The token is checked before validation, before allocation, throughout
    /// built-in block transforms, and after provider execution. Custom
    /// providers receive the same token and must establish bounded yield
    /// points for long-running operations.
    ///
    /// # Errors
    ///
    /// Returns a cancellation error or any error described by
    /// [`Self::prepare`].
    pub fn prepare_with_cancellation(
        &self,
        request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<ByteView> {
        cancellation.check()?;
        let provider = self
            .providers
            .resolve(request.transform().implementation())?;
        let strategy = provider.validate(request, cancellation)?;
        cancellation.check()?;
        match strategy {
            OutputStrategy::ReuseSource => {
                validate_reused_output(request)?;
                cancellation.check()?;
                Ok(request.source().clone())
            }
            OutputStrategy::Allocate {
                output_bytes,
                scratch_bytes,
            } => {
                let (output_len, scratch_len) =
                    validate_allocated_storage(request, output_bytes, scratch_bytes)?;
                cancellation.check()?;
                let output =
                    provider.prepare_allocated(request, output_len, scratch_len, cancellation)?;
                cancellation.check()?;
                if output.len() != output_len {
                    return Err(Error::integrity(format!(
                        "preparation provider returned {} output bytes, expected {output_len}",
                        output.len()
                    )));
                }
                Ok(ByteView::from_boxed(output))
            }
        }
    }

    /// Executes an ordered, size-pinned transform chain.
    ///
    /// An empty chain returns a clone of the source view without copying.
    /// Each intermediate result is dropped as soon as the following step
    /// completes.
    ///
    /// # Errors
    ///
    /// Returns an error from exact provider resolution, step validation,
    /// allocation, or provider execution.
    pub fn prepare_chain(
        &self,
        transforms: &[PlannedTransform],
        shape: &[u64],
        source: &ByteView,
    ) -> Result<ByteView> {
        self.prepare_chain_with_cancellation(transforms, shape, source, &CancellationToken::new())
    }

    /// Executes an ordered transform chain with cooperative cancellation.
    ///
    /// Cancellation is checked before each step. Providers performing
    /// long-running work should expose their own prompt yield points or be
    /// invoked through the bounded pipeline.
    ///
    /// # Errors
    ///
    /// Returns a cancellation error or an error from exact provider
    /// resolution, step validation, allocation, or provider execution.
    pub fn prepare_chain_with_cancellation(
        &self,
        transforms: &[PlannedTransform],
        shape: &[u64],
        source: &ByteView,
        cancellation: &CancellationToken,
    ) -> Result<ByteView> {
        let mut current = source.clone();
        for planned in transforms {
            cancellation.check()?;
            let request =
                PrepareRequest::new(planned.transform(), shape, &current, planned.output_size())
                    .with_expected_scratch_bytes(planned.scratch_bytes());
            current = self.prepare_with_cancellation(&request, cancellation)?;
        }
        cancellation.check()?;
        Ok(current)
    }
}

/// Returns the exact identity of the built-in contiguous cast implementation.
///
/// # Errors
///
/// Returns an invalid-format error if a built-in stable name cannot be
/// constructed.
pub fn builtin_contiguous_implementation() -> Result<ImplementationId> {
    Ok(ImplementationId::new(
        StableName::parse(BUILTIN_PROVIDER)?,
        StableName::parse(BUILTIN_CONTIGUOUS_CAST)?,
        BUILTIN_CONTIGUOUS_CAST_VERSION,
    ))
}

fn validate_reused_output(request: &PrepareRequest<'_>) -> Result<()> {
    if request.expected_scratch_bytes() != 0 {
        return Err(Error::integrity(format!(
            "provider selected source reuse without workspace, but the plan records {} scratch bytes",
            request.expected_scratch_bytes()
        )));
    }
    let source_len = u64::try_from(request.source().len()).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "source byte length does not fit u64",
            source,
        )
    })?;
    if source_len != request.expected_output_bytes() {
        return Err(Error::integrity(format!(
            "provider selected source reuse for {source_len} bytes, but the plan requires {} bytes",
            request.expected_output_bytes()
        )));
    }
    Ok(())
}

fn validate_allocated_storage(
    request: &PrepareRequest<'_>,
    provider_output_bytes: u64,
    provider_scratch_bytes: u64,
) -> Result<(usize, usize)> {
    if provider_output_bytes != request.expected_output_bytes() {
        return Err(Error::integrity(format!(
            "provider requires {provider_output_bytes} output bytes, but the plan records {} bytes",
            request.expected_output_bytes()
        )));
    }
    if provider_scratch_bytes != request.expected_scratch_bytes() {
        return Err(Error::integrity(format!(
            "provider requires {provider_scratch_bytes} scratch bytes, but the plan records {} bytes",
            request.expected_scratch_bytes()
        )));
    }
    let output_len = usize::try_from(provider_output_bytes).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "prepared byte length does not fit usize",
            source,
        )
    })?;
    let scratch_len = usize::try_from(provider_scratch_bytes).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "preparation scratch length does not fit usize",
            source,
        )
    })?;
    Ok((output_len, scratch_len))
}

fn allocate_zeroed(byte_len: usize, error_message: &'static str) -> Result<Box<[u8]>> {
    let mut output = Vec::new();
    output.try_reserve_exact(byte_len).map_err(|source| {
        Error::with_source(ErrorCategory::ResourceLimit, error_message, source)
    })?;
    output.resize(byte_len, 0);
    Ok(output.into_boxed_slice())
}

#[derive(Debug)]
struct ContiguousProvider {
    implementation: ImplementationId,
}

impl ContiguousProvider {
    fn new() -> Result<Self> {
        Ok(Self {
            implementation: builtin_contiguous_implementation()?,
        })
    }
}

impl PreparationProvider for ContiguousProvider {
    fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    fn validate(
        &self,
        request: &PrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<OutputStrategy> {
        cancellation.check()?;
        if request.transform().implementation() != self.implementation() {
            return Err(Error::unsupported(
                "contiguous provider received a different implementation identity",
            ));
        }
        let source = request.transform().source();
        let target = request.transform().target();
        if !source.layout().is_contiguous() || !target.layout().is_contiguous() {
            return Err(Error::unsupported(
                "built-in preparation supports only contiguous source and target layouts",
            ));
        }

        cancellation.check()?;
        let source_bytes = source.dtype().byte_len(request.shape())?;
        let actual_source_bytes = u64::try_from(request.source().len()).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "source byte length does not fit u64",
                source,
            )
        })?;
        if actual_source_bytes != source_bytes {
            return Err(Error::integrity(format!(
                "contiguous {:?} source has {actual_source_bytes} bytes, expected {source_bytes}",
                source.dtype()
            )));
        }

        cancellation.check()?;
        let target_bytes = target.dtype().byte_len(request.shape())?;
        if request.expected_output_bytes() != target_bytes {
            return Err(Error::integrity(format!(
                "contiguous {:?} target requires {target_bytes} bytes, but the plan records {} bytes",
                target.dtype(),
                request.expected_output_bytes()
            )));
        }

        if source.dtype() == target.dtype() {
            return Ok(OutputStrategy::ReuseSource);
        }
        if is_builtin_float(source.dtype()) && is_builtin_float(target.dtype()) {
            return Ok(OutputStrategy::Allocate {
                output_bytes: target_bytes,
                scratch_bytes: 0,
            });
        }
        Err(Error::unsupported(format!(
            "built-in contiguous conversion from {:?} to {:?}",
            source.dtype(),
            target.dtype()
        )))
    }

    fn prepare_into(
        &self,
        request: &PrepareRequest<'_>,
        output: &mut [u8],
        scratch: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        cancellation.check()?;
        let expected_len = usize::try_from(request.expected_output_bytes()).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "prepared byte length does not fit usize",
                source,
            )
        })?;
        if output.len() != expected_len {
            return Err(Error::integrity(format!(
                "contiguous provider received {} output bytes, expected {expected_len}",
                output.len()
            )));
        }
        if !scratch.is_empty() || request.expected_scratch_bytes() != 0 {
            return Err(Error::integrity(
                "contiguous provider requires exactly zero scratch bytes",
            ));
        }
        cast_float_bytes(
            request.transform().source().dtype(),
            request.source().as_slice(),
            request.transform().target().dtype(),
            output,
            cancellation,
        )
    }

    fn prepare_allocated(
        &self,
        request: &PrepareRequest<'_>,
        output_len: usize,
        scratch_len: usize,
        cancellation: &CancellationToken,
    ) -> Result<Box<[u8]>> {
        cancellation.check()?;
        let expected_len = usize::try_from(request.expected_output_bytes()).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "prepared byte length does not fit usize",
                source,
            )
        })?;
        if output_len != expected_len {
            return Err(Error::integrity(format!(
                "contiguous provider received {output_len} output bytes, expected {expected_len}"
            )));
        }
        if scratch_len != 0 || request.expected_scratch_bytes() != 0 {
            return Err(Error::integrity(
                "contiguous provider requires exactly zero scratch bytes",
            ));
        }
        cast_float_bytes_owned(
            request.transform().source().dtype(),
            request.source().as_slice(),
            request.transform().target().dtype(),
            output_len,
            cancellation,
        )
    }
}

const fn is_builtin_float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::Bf16)
}

fn cast_float_bytes(
    source_dtype: DType,
    source: &[u8],
    target_dtype: DType,
    output: &mut [u8],
    cancellation: &CancellationToken,
) -> Result<()> {
    let (source_width, target_width, elements) =
        validate_float_cast(source_dtype, source, target_dtype, output.len())?;

    for element_start in (0..elements).step_by(CAST_BLOCK_ELEMENTS) {
        cancellation.check()?;
        let block_elements = CAST_BLOCK_ELEMENTS.min(elements - element_start);
        let source_start = element_start
            .checked_mul(source_width)
            .ok_or_else(|| Error::limit("float cast source offset overflows usize"))?;
        let source_end = block_elements
            .checked_mul(source_width)
            .and_then(|length| source_start.checked_add(length))
            .ok_or_else(|| Error::limit("float cast source range overflows usize"))?;
        let target_start = element_start
            .checked_mul(target_width)
            .ok_or_else(|| Error::limit("float cast target offset overflows usize"))?;
        let target_end = block_elements
            .checked_mul(target_width)
            .and_then(|length| target_start.checked_add(length))
            .ok_or_else(|| Error::limit("float cast target range overflows usize"))?;
        let source_block = source
            .get(source_start..source_end)
            .ok_or_else(|| Error::integrity("float cast source block is out of bounds"))?;
        let output_block = output
            .get_mut(target_start..target_end)
            .ok_or_else(|| Error::integrity("float cast output block is out of bounds"))?;
        cast_float_block(source_dtype, source_block, target_dtype, output_block)?;
    }
    cancellation.check()?;
    Ok(())
}

fn cast_float_bytes_owned(
    source_dtype: DType,
    source: &[u8],
    target_dtype: DType,
    output_len: usize,
    cancellation: &CancellationToken,
) -> Result<Box<[u8]>> {
    let (source_width, _target_width, elements) =
        validate_float_cast(source_dtype, source, target_dtype, output_len)?;
    match (source_dtype, target_dtype) {
        (DType::F32, DType::F16) => collect_cast_blocks(
            source,
            source_width,
            elements,
            output_len,
            cancellation,
            |source| {
                f16::from_f32(read_f32_exact(source))
                    .to_bits()
                    .to_le_bytes()
            },
        ),
        (DType::F32, DType::Bf16) => collect_cast_blocks(
            source,
            source_width,
            elements,
            output_len,
            cancellation,
            |source| {
                bf16::from_f32(read_f32_exact(source))
                    .to_bits()
                    .to_le_bytes()
            },
        ),
        (DType::F16, DType::F32) => collect_cast_blocks(
            source,
            source_width,
            elements,
            output_len,
            cancellation,
            |source| {
                f16::from_bits(read_u16_exact(source))
                    .to_f32()
                    .to_le_bytes()
            },
        ),
        (DType::Bf16, DType::F32) => collect_cast_blocks(
            source,
            source_width,
            elements,
            output_len,
            cancellation,
            |source| {
                bf16::from_bits(read_u16_exact(source))
                    .to_f32()
                    .to_le_bytes()
            },
        ),
        (DType::F16, DType::Bf16) => collect_cast_blocks(
            source,
            source_width,
            elements,
            output_len,
            cancellation,
            |source| {
                let value = f16::from_bits(read_u16_exact(source)).to_f32();
                bf16::from_f32(value).to_bits().to_le_bytes()
            },
        ),
        (DType::Bf16, DType::F16) => collect_cast_blocks(
            source,
            source_width,
            elements,
            output_len,
            cancellation,
            |source| {
                let value = bf16::from_bits(read_u16_exact(source)).to_f32();
                f16::from_f32(value).to_bits().to_le_bytes()
            },
        ),
        _ => Err(Error::unsupported(format!(
            "built-in contiguous conversion from {source_dtype:?} to {target_dtype:?}"
        ))),
    }
}

fn collect_cast_blocks<const TARGET_WIDTH: usize>(
    source: &[u8],
    source_width: usize,
    elements: usize,
    output_len: usize,
    cancellation: &CancellationToken,
    mut convert: impl FnMut(&[u8]) -> [u8; TARGET_WIDTH],
) -> Result<Box<[u8]>> {
    let mut output = Vec::<[u8; TARGET_WIDTH]>::new();
    output.try_reserve_exact(elements).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "prepared output allocation failed",
            source,
        )
    })?;
    cancellation.check()?;

    for element_start in (0..elements).step_by(CAST_BLOCK_ELEMENTS) {
        cancellation.check()?;
        let block_elements = CAST_BLOCK_ELEMENTS.min(elements - element_start);
        let source_start = element_start
            .checked_mul(source_width)
            .ok_or_else(|| Error::limit("float cast source offset overflows usize"))?;
        let source_end = block_elements
            .checked_mul(source_width)
            .and_then(|length| source_start.checked_add(length))
            .ok_or_else(|| Error::limit("float cast source range overflows usize"))?;
        let source_block = source
            .get(source_start..source_end)
            .ok_or_else(|| Error::integrity("float cast source block is out of bounds"))?;
        output.extend(source_block.chunks_exact(source_width).map(&mut convert));
    }
    cancellation.check()?;

    let output = output.into_flattened();
    if output.len() != output_len {
        return Err(Error::integrity(format!(
            "float cast initialized {} output bytes, expected {output_len}",
            output.len()
        )));
    }
    Ok(output.into_boxed_slice())
}

fn validate_float_cast(
    source_dtype: DType,
    source: &[u8],
    target_dtype: DType,
    output_len: usize,
) -> Result<(usize, usize, usize)> {
    let source_width = float_width(source_dtype)?;
    let target_width = float_width(target_dtype)?;
    if source.len() % source_width != 0 {
        return Err(Error::integrity(
            "source byte length is not a whole number of float elements",
        ));
    }
    let elements = source.len() / source_width;
    let expected_output = elements
        .checked_mul(target_width)
        .ok_or_else(|| Error::limit("float cast output length overflows usize"))?;
    if output_len != expected_output {
        return Err(Error::integrity(format!(
            "float cast received {output_len} output bytes, expected {expected_output}"
        )));
    }
    Ok((source_width, target_width, elements))
}

fn float_width(dtype: DType) -> Result<usize> {
    match dtype {
        DType::F32 => Ok(4),
        DType::F16 | DType::Bf16 => Ok(2),
        _ => Err(Error::unsupported(
            "built-in casts support only f32, f16, and bf16",
        )),
    }
}

fn cast_float_block(
    source_dtype: DType,
    source: &[u8],
    target_dtype: DType,
    output: &mut [u8],
) -> Result<()> {
    match (source_dtype, target_dtype) {
        (DType::F32, DType::F16) => {
            for (source, target) in source.chunks_exact(4).zip(output.chunks_exact_mut(2)) {
                write_u16(target, f16::from_f32(read_f32(source)?).to_bits())?;
            }
        }
        (DType::F32, DType::Bf16) => {
            for (source, target) in source.chunks_exact(4).zip(output.chunks_exact_mut(2)) {
                write_u16(target, bf16::from_f32(read_f32(source)?).to_bits())?;
            }
        }
        (DType::F16, DType::F32) => {
            for (source, target) in source.chunks_exact(2).zip(output.chunks_exact_mut(4)) {
                write_f32(target, f16::from_bits(read_u16(source)?).to_f32())?;
            }
        }
        (DType::Bf16, DType::F32) => {
            for (source, target) in source.chunks_exact(2).zip(output.chunks_exact_mut(4)) {
                write_f32(target, bf16::from_bits(read_u16(source)?).to_f32())?;
            }
        }
        (DType::F16, DType::Bf16) => {
            for (source, target) in source.chunks_exact(2).zip(output.chunks_exact_mut(2)) {
                let value = f16::from_bits(read_u16(source)?).to_f32();
                write_u16(target, bf16::from_f32(value).to_bits())?;
            }
        }
        (DType::Bf16, DType::F16) => {
            for (source, target) in source.chunks_exact(2).zip(output.chunks_exact_mut(2)) {
                let value = bf16::from_bits(read_u16(source)?).to_f32();
                write_u16(target, f16::from_f32(value).to_bits())?;
            }
        }
        _ => {
            return Err(Error::unsupported(format!(
                "built-in contiguous conversion from {source_dtype:?} to {target_dtype:?}"
            )));
        }
    }
    Ok(())
}

fn read_u16(bytes: &[u8]) -> Result<u16> {
    let [low, high] = bytes else {
        return Err(Error::integrity("float source element is not two bytes"));
    };
    Ok(u16::from_le_bytes([*low, *high]))
}

fn read_f32(bytes: &[u8]) -> Result<f32> {
    let [a, b, c, d] = bytes else {
        return Err(Error::integrity("f32 source element is not four bytes"));
    };
    Ok(f32::from_le_bytes([*a, *b, *c, *d]))
}

#[inline]
fn read_u16_exact(bytes: &[u8]) -> u16 {
    debug_assert_eq!(bytes.len(), 2);
    u16::from_le_bytes([bytes[0], bytes[1]])
}

#[inline]
fn read_f32_exact(bytes: &[u8]) -> f32 {
    debug_assert_eq!(bytes.len(), 4);
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn write_u16(output: &mut [u8], value: u16) -> Result<()> {
    let [low, high] = output else {
        return Err(Error::integrity("float target element is not two bytes"));
    };
    let [value_low, value_high] = value.to_le_bytes();
    *low = value_low;
    *high = value_high;
    Ok(())
}

fn write_f32(output: &mut [u8], value: f32) -> Result<()> {
    let [a, b, c, d] = output else {
        return Err(Error::integrity("f32 target element is not four bytes"));
    };
    let [value_a, value_b, value_c, value_d] = value.to_le_bytes();
    *a = value_a;
    *b = value_b;
    *c = value_c;
    *d = value_d;
    Ok(())
}

#[cfg(test)]
#[path = "prepare_engine_contract_tests.rs"]
mod tests;

#[cfg(test)]
mod initialized_cast_tests {
    use super::*;

    fn assert_owned_matches_slice_writer(
        source_dtype: DType,
        source: &[u8],
        target_dtype: DType,
    ) -> Result<()> {
        let source_width = float_width(source_dtype)?;
        let target_width = float_width(target_dtype)?;
        let output_len = (source.len() / source_width)
            .checked_mul(target_width)
            .ok_or_else(|| Error::limit("test output length overflows usize"))?;
        let cancellation = CancellationToken::new();
        let mut expected = vec![0xa5; output_len];
        cast_float_bytes(
            source_dtype,
            source,
            target_dtype,
            &mut expected,
            &cancellation,
        )?;
        let actual = cast_float_bytes_owned(
            source_dtype,
            source,
            target_dtype,
            output_len,
            &cancellation,
        )?;
        assert_eq!(actual.as_ref(), expected);
        Ok(())
    }

    #[test]
    fn initialized_builder_matches_all_half_precision_bit_patterns() -> Result<()> {
        let all_half_bits = (u16::MIN..=u16::MAX)
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();

        assert_owned_matches_slice_writer(DType::F16, &all_half_bits, DType::F32)?;
        assert_owned_matches_slice_writer(DType::F16, &all_half_bits, DType::Bf16)?;
        assert_owned_matches_slice_writer(DType::Bf16, &all_half_bits, DType::F32)?;
        assert_owned_matches_slice_writer(DType::Bf16, &all_half_bits, DType::F16)
    }

    #[test]
    fn initialized_builder_preserves_f32_edges_across_block_boundaries() -> Result<()> {
        let edge_bits: [u32; 18] = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3300_0000,
            0x3380_0000,
            0x387f_ffff,
            0x3880_0000,
            0x3f80_0000,
            0x477f_e000,
            0x477f_ffff,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0x7fc0_0000,
            0xffc1_2345,
        ];
        let element_count = CAST_BLOCK_ELEMENTS + edge_bits.len();
        let source = (0..element_count)
            .flat_map(|index| edge_bits[index % edge_bits.len()].to_le_bytes())
            .collect::<Vec<_>>();

        assert_owned_matches_slice_writer(DType::F32, &source, DType::F16)?;
        assert_owned_matches_slice_writer(DType::F32, &source, DType::Bf16)
    }
}
