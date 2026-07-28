//! Validated execution of binding plans over checkpoints.
//!
//! This module is the runtime-neutral seam between inventory/planning and the
//! generic bounded pipeline. Plain bindings are prepared into host bytes.
//! Declarative conversions and quantized routes are explicit provider/runtime
//! handoffs on a cache miss; validated provider-finalized host outputs can be
//! returned from the prepared cache.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sha2::{Digest as _, Sha256};

use crate::cache::{
    Cache, CacheCompatibility, CacheKey, CacheLookup, CacheMissReason, CacheNamespace,
    CachePublication, CacheValidation, EvictionReason,
};
use crate::identity::{ContentDigest, PlanId, StableName};
use crate::operation::OperationGraph;
use crate::overlay::{OverlayBinding, OverlayPlan};
use crate::pipeline::{
    Pipeline, PrepareContext, PreparedItem, PreparedSink, ResourceWeights, WorkItem,
};
use crate::plan::{
    Binding, BindingPlan, ConversionRecipe, PLAN_SCHEMA_VERSION, PlannedTransform, SourceTensor,
    TargetTensor, TensorName,
};
use crate::prepare::{PreparationEngine, PrepareRequest, Representation};
use crate::quantization::{QuantizedRoute, QuantizedStorage, RouteCapability, Storage};
use crate::source::SourceKind;
use crate::telemetry::{
    ExecutionEvent, ExecutionObserver, ExecutionPhase, ExecutionReport, NoopObserver,
    OperationKind, OperationLocation,
};
use crate::tensor::{ByteView, SourceSpan};
use crate::{AccessMode, CancellationToken, Checkpoint, Error, ErrorCategory, Result};

const PREPARED_CACHE_FORMAT_VERSION: u32 = 2;
const CACHE_READ_BLOCK_BYTES: usize = 1024 * 1024;
const MAX_CACHED_PLAN_BYTES: u64 = 64 * 1024 * 1024;

/// Identifies how finalized host bytes were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PreparedOrigin {
    /// The source already had the requested representation.
    Source,
    /// A pinned host transform produced the bytes.
    Transform,
    /// A validated typed operation graph produced the bytes.
    OperationGraph,
    /// A validated prepared-cache entry supplied the bytes.
    Cache,
}

/// Finalized host bytes for one target binding.
#[derive(Debug, Clone)]
pub struct PreparedWeight {
    source_names: Box<[TensorName]>,
    target: TargetTensor,
    bytes: ByteView,
    origin: PreparedOrigin,
    resident_bytes: u64,
}

impl PreparedWeight {
    /// Returns the first selected checkpoint tensor name.
    ///
    /// Existing one-source consumers can continue to use this accessor.
    /// Group-aware consumers should use [`Self::source_names`].
    ///
    /// # Panics
    ///
    /// Panics only if the validated prepared-weight invariant is violated and
    /// the ordered source list is empty.
    #[must_use]
    pub fn source_name(&self) -> &TensorName {
        self.source_names
            .first()
            .expect("validated prepared weights always contain a source")
    }

    /// Returns every selected checkpoint tensor name in semantic input order.
    #[must_use]
    pub const fn source_names(&self) -> &[TensorName] {
        &self.source_names
    }

    /// Returns the consumer target name.
    #[must_use]
    pub const fn target_name(&self) -> &TensorName {
        self.target.name()
    }

    /// Returns the complete consumer target contract.
    #[must_use]
    pub const fn target(&self) -> &TargetTensor {
        &self.target
    }

    /// Returns the consumer-visible logical shape.
    #[must_use]
    pub const fn shape(&self) -> &[u64] {
        self.target.shape()
    }

    /// Returns the finalized dtype and physical layout.
    #[must_use]
    pub const fn representation(&self) -> &Representation {
        self.target.representation()
    }

    /// Returns immutable finalized host bytes.
    #[must_use]
    pub const fn bytes(&self) -> &ByteView {
        &self.bytes
    }

    /// Returns how these bytes were obtained.
    #[must_use]
    pub const fn origin(&self) -> PreparedOrigin {
        self.origin
    }

    /// Returns host bytes retained through consumer delivery.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// A declarative conversion plus its exact source bytes.
///
/// A Rust or Python-backed provider executes the complete recipe into the
/// target representation. Recipes remain mutually exclusive with ordinary
/// transform chains and typed operation graphs, so no execution step is
/// silently skipped.
#[derive(Debug, Clone)]
pub struct ConversionHandoff {
    source: SourceTensor,
    target: TargetTensor,
    recipe: ConversionRecipe,
    source_bytes: ByteView,
    resident_bytes: u64,
}

impl ConversionHandoff {
    /// Returns the selected source descriptor.
    #[must_use]
    pub const fn source(&self) -> &SourceTensor {
        &self.source
    }

    /// Returns the complete provider-output target contract.
    #[must_use]
    pub const fn target(&self) -> &TargetTensor {
        &self.target
    }

    /// Returns the exact language-neutral conversion recipe.
    #[must_use]
    pub const fn recipe(&self) -> &ConversionRecipe {
        &self.recipe
    }

    /// Returns the immutable external-input bytes.
    #[must_use]
    pub const fn source_bytes(&self) -> &ByteView {
        &self.source_bytes
    }

    /// Returns host bytes retained through provider handoff.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// Selects core host execution or an explicit consumer/provider handoff for
/// typed operation graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperationExecution {
    /// Execute recognized versioned operations through the bounded core host
    /// interpreter.
    Host,
    /// Deliver ordered source bytes and the validated graph for consumer-owned
    /// fused CPU or GPU execution.
    Delegate,
}

/// One ordered source descriptor and immutable byte view supplied to a
/// delegated operation graph.
#[derive(Debug, Clone)]
pub struct OperationInputBytes {
    source: SourceTensor,
    bytes: ByteView,
}

impl OperationInputBytes {
    /// Returns the exact source descriptor and span.
    #[must_use]
    pub const fn source(&self) -> &SourceTensor {
        &self.source
    }

    /// Returns immutable source bytes.
    #[must_use]
    pub const fn bytes(&self) -> &ByteView {
        &self.bytes
    }
}

/// A validated typed operation graph plus ordered source bytes.
///
/// Consumers can execute or fuse the same deterministic semantics on CPU or
/// GPU, then publish finalized host bytes through
/// [`Materializer::publish_prepared_bytes`].
#[derive(Debug, Clone)]
pub struct OperationHandoff {
    inputs: Box<[OperationInputBytes]>,
    target: TargetTensor,
    graph: OperationGraph,
    resident_bytes: u64,
}

impl OperationHandoff {
    /// Returns ordered graph inputs in exact semantic edge order.
    #[must_use]
    pub const fn inputs(&self) -> &[OperationInputBytes] {
        &self.inputs
    }

    /// Returns the complete target contract.
    #[must_use]
    pub const fn target(&self) -> &TargetTensor {
        &self.target
    }

    /// Returns the validated, versioned operation graph.
    #[must_use]
    pub const fn graph(&self) -> &OperationGraph {
        &self.graph
    }

    /// Returns source bytes retained until provider handoff completes.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// Packed payloads and an exact runtime-selected route capability.
#[derive(Debug, Clone)]
pub struct QuantizedHandoff {
    source: SourceTensor,
    target: TargetTensor,
    storage: QuantizedStorage,
    capability: RouteCapability,
    payload: ByteView,
    companions: BTreeMap<StableName, ByteView>,
    resident_bytes: u64,
}

impl QuantizedHandoff {
    /// Returns the selected packed source descriptor.
    #[must_use]
    pub const fn source(&self) -> &SourceTensor {
        &self.source
    }

    /// Returns the complete target contract.
    #[must_use]
    pub const fn target(&self) -> &TargetTensor {
        &self.target
    }

    /// Returns normalized encoding, packing, grouping, and companion metadata.
    #[must_use]
    pub const fn storage(&self) -> &QuantizedStorage {
        &self.storage
    }

    /// Returns the exact provider/backend route selected by the plan.
    #[must_use]
    pub const fn capability(&self) -> &RouteCapability {
        &self.capability
    }

    /// Returns the immutable primary packed payload.
    #[must_use]
    pub const fn payload(&self) -> &ByteView {
        &self.payload
    }

    /// Returns immutable companion payloads keyed by normalized role.
    #[must_use]
    pub const fn companions(&self) -> &BTreeMap<StableName, ByteView> {
        &self.companions
    }

    /// Returns packed host bytes retained through runtime handoff.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// A prepared base plus an ordered, independently identified overlay recipe.
///
/// The handoff is intentionally lazy. A consumer can fuse the operations into
/// a kernel, execute them through a provider, or explicitly eager-merge them.
/// Overlay source acquisition and allocation remain outside this crate.
#[derive(Debug, Clone)]
pub struct OverlayHandoff {
    base: Box<WeightDelivery>,
    plan: Arc<OverlayPlan>,
    binding_index: usize,
    target_digest: ContentDigest,
    resident_bytes: u64,
}

impl OverlayHandoff {
    /// Returns the materialized or delegated base delivery.
    #[must_use]
    pub fn base(&self) -> &WeightDelivery {
        self.base.as_ref()
    }

    /// Returns the complete ordered overlay plan.
    #[must_use]
    pub fn plan(&self) -> &OverlayPlan {
        &self.plan
    }

    /// Returns the lazy composition binding for this canonical base.
    #[must_use]
    pub fn binding(&self) -> &OverlayBinding {
        &self.plan.bindings()[self.binding_index]
    }

    /// Returns the identity affected only by this base and its operations.
    #[must_use]
    pub const fn target_digest(&self) -> ContentDigest {
        self.target_digest
    }

    /// Returns host bytes retained through provider/runtime handoff.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// One ordinal-deliverable result of executing a binding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WeightDelivery {
    /// Final host bytes ready for a consumer allocation or upload.
    Prepared(Box<PreparedWeight>),
    /// A provider-owned declarative conversion still to execute.
    Conversion(Box<ConversionHandoff>),
    /// A consumer/provider-owned typed operation graph still to execute.
    Operation(Box<OperationHandoff>),
    /// A runtime-owned packed route still to execute or consume directly.
    Quantized(Box<QuantizedHandoff>),
    /// A lazy ordered overlay still to execute or fuse.
    Overlay(Box<OverlayHandoff>),
}

impl WeightDelivery {
    /// Returns the consumer target descriptor.
    #[must_use]
    pub fn target(&self) -> &TargetTensor {
        match self {
            Self::Prepared(weight) => weight.target(),
            Self::Conversion(handoff) => handoff.target(),
            Self::Operation(handoff) => handoff.target(),
            Self::Quantized(handoff) => handoff.target(),
            Self::Overlay(handoff) => handoff.base().target(),
        }
    }

    /// Returns the consumer target name.
    #[must_use]
    pub fn target_name(&self) -> &TensorName {
        match self {
            Self::Prepared(weight) => weight.target_name(),
            Self::Conversion(handoff) => handoff.target().name(),
            Self::Operation(handoff) => handoff.target().name(),
            Self::Quantized(handoff) => handoff.target().name(),
            Self::Overlay(handoff) => handoff.base().target_name(),
        }
    }

    /// Returns host bytes retained until the sink accepts this delivery.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        match self {
            Self::Prepared(weight) => weight.resident_bytes(),
            Self::Conversion(handoff) => handoff.resident_bytes(),
            Self::Operation(handoff) => handoff.resident_bytes(),
            Self::Quantized(handoff) => handoff.resident_bytes(),
            Self::Overlay(handoff) => handoff.resident_bytes(),
        }
    }
}

/// The deterministic prepared-cache address for one binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedCacheAddress {
    key: CacheKey,
    compatibility: CacheCompatibility,
}

impl PreparedCacheAddress {
    /// Returns the content-addressed cache key.
    #[must_use]
    pub const fn key(&self) -> CacheKey {
        self.key
    }

    /// Returns the transform and backend ABI compatibility contract.
    #[must_use]
    pub const fn compatibility(&self) -> &CacheCompatibility {
        &self.compatibility
    }
}

/// Derives the base content identity expected by [`OverlayPlan`].
///
/// The identity includes the exact normalized source descriptor and only the
/// ordered file digests referenced by its payload and companions. Unrelated
/// overlay layers and unrelated checkpoint shards therefore do not invalidate
/// this base identity.
///
/// # Errors
///
/// Returns a binding error when `binding` is not part of `plan` or references
/// a file ordinal without a corresponding source digest, and an
/// invalid-format error if source serialization fails.
pub fn binding_source_content(plan: &BindingPlan, binding: &Binding) -> Result<ContentDigest> {
    binding_source_content_with_cancellation(plan, binding, &CancellationToken::new())
}

/// Derives the base content identity while observing cooperative cancellation.
///
/// # Errors
///
/// Returns a binding error when `binding` is not part of `plan` or references
/// a file ordinal without a corresponding source digest, an invalid-format
/// error if source serialization fails, or a cancellation error.
pub fn binding_source_content_with_cancellation(
    plan: &BindingPlan,
    binding: &Binding,
    cancellation: &CancellationToken,
) -> Result<ContentDigest> {
    cancellation.check()?;
    let candidate = plan
        .bindings()
        .binary_search_by(|candidate| candidate.target().name().cmp(binding.target().name()))
        .ok()
        .map(|index| &plan.bindings()[index]);
    if candidate != Some(binding) {
        return Err(Error::binding(
            "overlay base identity requires a binding from the supplied plan",
        ));
    }
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(binding.sources().len())
        .map_err(|_error| Error::limit("could not allocate ordered source identity metadata"))?;
    for source in binding.sources() {
        cancellation.check()?;
        let mut ordinals = BTreeSet::new();
        collect_file_ordinals(source.storage(), &mut ordinals, cancellation)?;
        let mut digests = Vec::new();
        digests
            .try_reserve_exact(ordinals.len())
            .map_err(|_error| Error::limit("could not allocate source digest identity metadata"))?;
        for ordinal in ordinals {
            cancellation.check()?;
            let digest_index = usize::try_from(ordinal)
                .map_err(|_error| Error::limit("source file ordinal does not fit usize"))?;
            let digest = plan
                .inputs()
                .source_digests()
                .get(digest_index)
                .copied()
                .ok_or_else(|| {
                    Error::binding("source file ordinal has no ordered source digest")
                })?;
            digests.push((ordinal, digest));
        }
        sources.push(OrderedSourceIdentity { source, digests });
    }
    let identity = serde_json::to_vec(&sources).map_err(|error| {
        Error::with_source(
            ErrorCategory::InvalidFormat,
            "serialize ordered binding source identity",
            error,
        )
    })?;
    cancellation.check()?;
    let digest = ContentDigest::hash("ordered-binding-sources-v2", [&identity]);
    cancellation.check()?;
    Ok(digest)
}

#[derive(serde::Serialize)]
struct OrderedSourceIdentity<'a> {
    source: &'a SourceTensor,
    digests: Vec<(u32, ContentDigest)>,
}

/// The deterministic cache address of one canonical binding plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanCacheAddress {
    key: CacheKey,
    compatibility: CacheCompatibility,
}

impl PlanCacheAddress {
    /// Derives an address from a canonical plan identity and schema version.
    #[must_use]
    pub fn new(plan_id: PlanId) -> Self {
        let schema = PLAN_SCHEMA_VERSION.to_le_bytes();
        let transform = ContentDigest::hash("binding-plan-cache-schema-v2", [&schema]);
        let compatibility = CacheCompatibility::plan(PLAN_SCHEMA_VERSION, transform);
        let key = CacheKey::derive(
            CacheNamespace::Plan,
            &compatibility,
            [plan_id.digest().as_bytes()],
        );
        Self { key, compatibility }
    }

    /// Returns the content-addressed cache key.
    #[must_use]
    pub const fn key(&self) -> CacheKey {
        self.key
    }

    /// Returns the plan schema compatibility contract.
    #[must_use]
    pub const fn compatibility(&self) -> &CacheCompatibility {
        &self.compatibility
    }
}

/// The parsed result of looking up one canonical binding plan.
#[derive(Debug)]
#[non_exhaustive]
pub enum BindingPlanCacheLookup {
    /// A canonical plan whose internal identity matches the requested address.
    Hit(BindingPlan),
    /// The cache returned a deterministic miss reason.
    Miss(CacheMissReason),
}

/// Publishes a canonical binding plan in the physically separate plan cache.
///
/// # Errors
///
/// Returns a plan serialization, cache lease/publication, I/O, or cancellation
/// error.
pub fn publish_binding_plan(
    cache: &Cache,
    plan: &BindingPlan,
    cancellation: &CancellationToken,
) -> Result<CachePublication> {
    let address = PlanCacheAddress::new(plan.id());
    let bytes = plan.to_canonical_json()?;
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_error| Error::limit("binding plan byte length does not fit u64"))?;
    if byte_len > MAX_CACHED_PLAN_BYTES {
        return Err(Error::limit(
            "binding plan exceeds the defensive cache byte limit",
        ));
    }
    cache.publish_reader(
        CacheNamespace::Plan,
        address.key(),
        address.compatibility(),
        bytes.as_ref(),
        cancellation,
    )
}

/// Looks up, parses, and revalidates a canonical binding plan.
///
/// # Errors
///
/// Returns an I/O, integrity, resource-limit, cancellation, or canonical-plan
/// validation error. Cache incompatibility and absence remain inspectable
/// [`BindingPlanCacheLookup::Miss`] values.
pub fn lookup_binding_plan(
    cache: &Cache,
    plan_id: PlanId,
    validation: CacheValidation,
    cancellation: &CancellationToken,
) -> Result<BindingPlanCacheLookup> {
    let address = PlanCacheAddress::new(plan_id);
    let lookup = cache.lookup_with_validation_and_cancellation(
        CacheNamespace::Plan,
        address.key(),
        address.compatibility(),
        CacheValidation::TrustedMetadata,
        cancellation,
    )?;
    let entry = match lookup {
        CacheLookup::Hit(entry) => entry,
        CacheLookup::Miss(reason) => return Ok(BindingPlanCacheLookup::Miss(reason)),
    };
    if entry.info().payload_len() > MAX_CACHED_PLAN_BYTES {
        return Err(Error::limit(
            "cached binding plan exceeds the defensive byte limit",
        ));
    }
    let bytes = read_cache_entry(entry, validation, cancellation)?;
    let plan = BindingPlan::from_canonical_json(bytes.as_slice())?;
    if plan.id() != plan_id {
        return Err(Error::integrity(
            "cached binding plan identity differs from the requested cache address",
        ));
    }
    Ok(BindingPlanCacheLookup::Hit(plan))
}

/// Validates and executes one binding plan against one opened checkpoint.
#[derive(Debug)]
pub struct Materializer<'a> {
    checkpoint: &'a Checkpoint,
    plan: &'a BindingPlan,
    preparation: &'a PreparationEngine,
    cache: Option<&'a Cache>,
    cache_validation: CacheValidation,
    overlay: Option<Arc<OverlayPlan>>,
    operation_execution: OperationExecution,
}

#[derive(Clone, Copy)]
struct OperationObservation<'a> {
    work_ordinal: Option<u64>,
    observer: &'a dyn ExecutionObserver,
}

impl<'a> Materializer<'a> {
    /// Validates a plan against the exact opened inventory and source digests.
    ///
    /// Trusted retained digests are reused. Ordinary local sources are hashed
    /// on demand, then cached by the checkpoint handle.
    ///
    /// # Errors
    ///
    /// Returns an integrity or binding error when the plan belongs to different
    /// bytes or contains a source descriptor not present in this checkpoint,
    /// plus any source hashing or cancellation error.
    pub fn new(
        checkpoint: &'a Checkpoint,
        plan: &'a BindingPlan,
        preparation: &'a PreparationEngine,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        Self::new_with_observer(checkpoint, plan, preparation, cancellation, &NoopObserver)
    }

    /// Validates a materializer while emitting source-hashing telemetry.
    ///
    /// Trusted or already-cached digests emit no hashing phase. Ordinary bytes
    /// that still require hashing produce one bounded phase measurement before
    /// any work items are created.
    ///
    /// # Errors
    ///
    /// Returns any error described by [`Self::new`].
    pub fn new_with_observer<O>(
        checkpoint: &'a Checkpoint,
        plan: &'a BindingPlan,
        preparation: &'a PreparationEngine,
        cancellation: &CancellationToken,
        observer: &O,
    ) -> Result<Self>
    where
        O: ExecutionObserver,
    {
        let pending_hash_bytes = checkpoint.pending_digest_bytes(cancellation)?;
        if pending_hash_bytes > 0 {
            observer.observe(&ExecutionEvent::PhaseStarted {
                phase: ExecutionPhase::Hashing,
                ordinal: None,
            });
        }
        let started = Instant::now();
        let digests = checkpoint.source_digests(cancellation);
        if pending_hash_bytes > 0 {
            observer.observe(&ExecutionEvent::PhaseFinished {
                phase: ExecutionPhase::Hashing,
                ordinal: None,
                duration: started.elapsed(),
                bytes: pending_hash_bytes,
            });
        }
        let actual_digests = digests?;
        if actual_digests.as_ref() != plan.inputs().source_digests() {
            return Err(Error::integrity(
                "binding plan source digests differ from the opened checkpoint",
            ));
        }
        validate_complete_plan_inventory(checkpoint, plan, cancellation)?;
        for binding in plan.bindings() {
            cancellation.check()?;
            validate_inventory_binding(checkpoint, binding)?;
        }
        cancellation.check()?;
        Ok(Self {
            checkpoint,
            plan,
            preparation,
            cache: None,
            cache_validation: CacheValidation::Full,
            overlay: None,
            operation_execution: OperationExecution::Host,
        })
    }

    /// Enables prepared-byte reuse and publication for transformed bindings.
    #[must_use]
    pub const fn with_cache(mut self, cache: &'a Cache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Selects prepared-cache integrity validation.
    ///
    /// The default is [`CacheValidation::Full`]. Trusted metadata avoids an
    /// otherwise redundant warm-path hash only when the cache root is protected
    /// from out-of-band mutation.
    #[must_use]
    pub const fn with_cache_validation(mut self, validation: CacheValidation) -> Self {
        self.cache_validation = validation;
        self
    }

    /// Selects host interpretation or consumer/provider delegation for typed
    /// operation graphs.
    ///
    /// The default is [`OperationExecution::Host`]. Both routes share the
    /// exact serialized graph and prepared-cache identity.
    #[must_use]
    pub const fn with_operation_execution(mut self, execution: OperationExecution) -> Self {
        self.operation_execution = execution;
        self
    }

    /// Returns the selected operation-graph execution route.
    #[must_use]
    pub const fn operation_execution(&self) -> OperationExecution {
        self.operation_execution
    }

    /// Attaches a validated ordered overlay plan.
    ///
    /// Selected overlay bases must match their target shape and source-derived
    /// content identity. Bases belonging to unselected components are retained
    /// without preventing partial-model materialization. The plan is retained
    /// once behind an [`Arc`] so per-target handoffs remain cheap to clone.
    ///
    /// # Errors
    ///
    /// Returns a binding or integrity error for a mismatched selected base.
    pub fn with_overlay_plan(self, overlay: Arc<OverlayPlan>) -> Result<Self> {
        self.with_overlay_plan_with_cancellation(overlay, &CancellationToken::new())
    }

    /// Attaches an ordered overlay plan with cooperative validation cancellation.
    ///
    /// Selected targets are indexed once by their alias-resolved base name, so
    /// validation scales with the selected plan and overlay rather than scanning
    /// every selected binding for every overlay base.
    ///
    /// # Errors
    ///
    /// Returns a binding or integrity error for a mismatched selected base, or
    /// a cancellation error while indexing or validating the plans.
    pub fn with_overlay_plan_with_cancellation(
        mut self,
        overlay: Arc<OverlayPlan>,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        validate_overlay_plan(self.plan, &overlay, cancellation)?;
        self.overlay = Some(overlay);
        Ok(self)
    }

    /// Returns the attached overlay plan, if any.
    #[must_use]
    pub fn overlay_plan(&self) -> Option<&OverlayPlan> {
        self.overlay.as_deref()
    }

    /// Returns the validated plan.
    #[must_use]
    pub const fn plan(&self) -> &BindingPlan {
        self.plan
    }

    /// Returns the opened checkpoint.
    #[must_use]
    pub const fn checkpoint(&self) -> &Checkpoint {
        self.checkpoint
    }

    /// Finds a selected binding by exact target name.
    ///
    /// # Errors
    ///
    /// Returns a binding error when the target is absent from the plan.
    pub fn binding(&self, target_name: &str) -> Result<&Binding> {
        self.plan
            .bindings()
            .binary_search_by(|binding| binding.target().name().as_str().cmp(target_name))
            .ok()
            .map(|index| &self.plan.bindings()[index])
            .ok_or_else(|| Error::binding("target name is not present in the binding plan"))
    }

    /// Derives the prepared-cache address for one finalized host output.
    ///
    /// Source-compatible zero-copy bindings and device/direct packed routes
    /// return `None` because they have no finalized host representation to
    /// reuse. Provider-completed conversion recipes and host dequantization
    /// routes are addressable and can be published with
    /// [`Self::publish_prepared_bytes`].
    ///
    /// # Errors
    ///
    /// Returns a binding, integrity, invalid-format, or resource-limit error
    /// while resolving and serializing the selected source/target identity.
    pub fn prepared_cache_address(
        &self,
        target_name: &str,
    ) -> Result<Option<PreparedCacheAddress>> {
        self.prepared_cache_address_with_cancellation(target_name, &CancellationToken::new())
    }

    /// Derives a prepared-cache address with cooperative cancellation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format or cancellation error while deriving the
    /// selected source, target, and overlay identity.
    pub fn prepared_cache_address_with_cancellation(
        &self,
        target_name: &str,
        cancellation: &CancellationToken,
    ) -> Result<Option<PreparedCacheAddress>> {
        cancellation.check()?;
        let binding = self.binding(target_name)?;
        let overlay_digest = self
            .overlay_for_binding(binding, cancellation)?
            .map(|(_index, digest)| digest);
        let address = self.cache_address_for_binding_with_cancellation(
            binding,
            overlay_digest,
            cancellation,
        )?;
        cancellation.check()?;
        Ok(address)
    }

    /// Publishes provider-produced final bytes under this materializer's exact
    /// source, backend, transform, and per-target overlay identity.
    ///
    /// This is the cache integration point for lazy overlays, conversion
    /// recipes, and host-dequantization providers. Device-scratch, direct
    /// packed, fused-in-tile, and repack routes intentionally have no host
    /// prepared-cache address.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-capability error when no cache or final prepared
    /// address is configured, an integrity error for a byte-length mismatch, or
    /// a cache publication/cancellation error.
    pub fn publish_prepared_bytes(
        &self,
        target_name: &str,
        bytes: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<CachePublication> {
        cancellation.check()?;
        let binding = self.binding(target_name)?;
        let actual = u64::try_from(bytes.len())
            .map_err(|_error| Error::limit("prepared byte length does not fit u64"))?;
        if actual != binding.target().output_size() {
            return Err(Error::integrity(
                "provider-produced bytes differ from the planned output size",
            ));
        }
        let cache = self
            .cache
            .ok_or_else(|| Error::unsupported("materializer has no prepared cache"))?;
        let address = self
            .prepared_cache_address_with_cancellation(target_name, cancellation)?
            .ok_or_else(|| Error::unsupported("binding has no final prepared-cache address"))?;
        cache.publish_reader(
            CacheNamespace::Prepared,
            address.key(),
            address.compatibility(),
            bytes,
            cancellation,
        )
    }

    /// Executes one selected binding.
    ///
    /// # Errors
    ///
    /// Returns a source, cache, preparation, capability, or cancellation error.
    pub fn materialize(
        &self,
        target_name: &str,
        cancellation: &CancellationToken,
    ) -> Result<WeightDelivery> {
        let binding = self.binding(target_name)?;
        self.materialize_binding(binding, cancellation, None, None)
    }

    /// Executes one selected binding while emitting operation-level telemetry.
    ///
    /// Direct materialization has no pipeline ordinal, so emitted operation
    /// events set `work_ordinal` to `None`. An observer that disables operation
    /// events follows the same uninstrumented path as [`Self::materialize`].
    ///
    /// # Errors
    ///
    /// Returns any error described by [`Self::materialize`].
    pub fn materialize_with_observer(
        &self,
        target_name: &str,
        cancellation: &CancellationToken,
        observer: &dyn ExecutionObserver,
    ) -> Result<WeightDelivery> {
        let binding = self.binding(target_name)?;
        let observation = observer
            .operation_events_enabled()
            .then_some(OperationObservation {
                work_ordinal: None,
                observer,
            });
        self.materialize_binding(binding, cancellation, None, observation)
    }

    /// Executes selected bindings through a bounded, ordinal delivery pipeline.
    ///
    /// Work is created only for `plan.bindings()`, in canonical target order.
    /// The binding count is rejected against `max_work_items` before any
    /// per-binding resource or cache-address work begins.
    /// Prepared-cache candidates reserve their delivered bytes first; a miss
    /// atomically acquires the additional source and scratch requirements.
    /// Conversion and quantized cache misses remain explicit delivery variants
    /// rather than being silently executed. A validated provider-finalized
    /// recipe or host-dequant output may instead arrive as `Prepared(Cache)`.
    ///
    /// # Errors
    ///
    /// Returns a configured pipeline error or any binding materialization and
    /// sink delivery error.
    pub fn execute<S, O>(
        &self,
        pipeline: &Pipeline,
        sink: &mut S,
        observer: &O,
    ) -> Result<ExecutionReport>
    where
        S: PreparedSink<WeightDelivery>,
        O: ExecutionObserver,
    {
        let bindings = self.plan.bindings();
        if bindings.len() > pipeline.limits().max_work_items {
            return Err(Error::limit(
                "pipeline work-item count exceeds the configured limit",
            ));
        }
        let cancellation = pipeline.cancellation();
        cancellation.check()?;
        let mut work = Vec::new();
        work.try_reserve_exact(bindings.len())
            .map_err(|_error| Error::limit("could not allocate materializer work metadata"))?;
        for (index, binding) in bindings.iter().enumerate() {
            cancellation.check()?;
            let ordinal = u64::try_from(index)
                .map_err(|_error| Error::limit("binding ordinal does not fit u64"))?;
            let resources = self.resource_weights(binding)?;
            let scheduled = if self.cache.is_some()
                && self
                    .prepared_cache_address_with_cancellation(
                        binding.target().name().as_str(),
                        pipeline.cancellation(),
                    )?
                    .is_some()
            {
                let route_maximum = resources
                    .prepared_bytes()
                    .max(binding.target().output_size());
                ResourceWeights::new(0, 0, route_maximum.min(pipeline.limits().prepared_bytes))
            } else {
                resources
            };
            work.push(WorkItem::new(ordinal, binding, scheduled));
        }
        cancellation.check()?;
        let operation_observer = observer
            .operation_events_enabled()
            .then_some(observer as &dyn ExecutionObserver);

        pipeline.execute(
            work,
            |binding, context| {
                let cancellation = context.cancellation().clone();
                let operation_observation =
                    operation_observer.map(|observer| OperationObservation {
                        work_ordinal: Some(context.ordinal()),
                        observer,
                    });
                let delivery = self.materialize_binding(
                    binding,
                    &cancellation,
                    Some(context),
                    operation_observation,
                )?;
                let resident_bytes = delivery.resident_bytes();
                Ok(PreparedItem::new(delivery, resident_bytes))
            },
            sink,
            observer,
        )
    }

    fn cache_address_for_binding_with_cancellation(
        &self,
        binding: &Binding,
        overlay_digest: Option<ContentDigest>,
        cancellation: &CancellationToken,
    ) -> Result<Option<PreparedCacheAddress>> {
        cancellation.check()?;
        let has_final_host_output = if let Some(graph) = binding.target().operation_graph() {
            overlay_digest.is_some() || graph.output_input_alias().is_none()
        } else {
            match binding.source().storage() {
                Storage::Plain { .. } => {
                    binding.target().conversion_recipe().is_some()
                        || !binding.target().transforms().is_empty()
                        || overlay_digest.is_some()
                }
                Storage::Quantized(_) => {
                    binding
                        .target()
                        .quantized_route()
                        .is_some_and(|capability| {
                            matches!(capability.route(), QuantizedRoute::HostDequant { .. })
                        })
                }
            }
        };
        if !has_final_host_output {
            cancellation.check()?;
            return Ok(None);
        }
        let mut contract = serde_json::to_vec(binding.target()).map_err(|source| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "serialize prepared binding cache contract",
                source,
            )
        })?;
        cancellation.check()?;
        match overlay_digest {
            Some(digest) => {
                contract.push(1);
                contract.extend_from_slice(digest.as_bytes());
            }
            None => contract.push(0),
        }
        let transform = ContentDigest::hash("prepared-binding-contract-v2", [&contract]);
        cancellation.check()?;
        let backend = self.plan.inputs().backend().digest();
        let compatibility =
            CacheCompatibility::prepared(PREPARED_CACHE_FORMAT_VERSION, transform, backend);
        // Prepared entries are scoped to the one source/target contract that
        // produced them. A global plan identifier would make unrelated target,
        // shard, or overlay changes invalidate otherwise reusable bytes.
        let source_content =
            binding_source_content_with_cancellation(self.plan, binding, cancellation)?;
        let key = CacheKey::derive(
            CacheNamespace::Prepared,
            &compatibility,
            [
                source_content.as_bytes().as_slice(),
                binding.target().name().as_str().as_bytes(),
            ],
        );
        cancellation.check()?;
        Ok(Some(PreparedCacheAddress { key, compatibility }))
    }

    fn materialize_binding(
        &self,
        binding: &Binding,
        cancellation: &CancellationToken,
        mut context: Option<&mut PrepareContext<'_>>,
        operation_observation: Option<OperationObservation<'_>>,
    ) -> Result<WeightDelivery> {
        cancellation.check()?;
        if let Some((binding_index, target_digest)) =
            self.overlay_for_binding(binding, cancellation)?
        {
            let address = self.cache_address_for_binding_with_cancellation(
                binding,
                Some(target_digest),
                cancellation,
            )?;
            if let Some(delivery) =
                self.try_cached_prepared(binding, address.as_ref(), cancellation, &mut context)?
            {
                return Ok(delivery);
            }
            let base = self.materialize_base_binding(
                binding,
                cancellation,
                &mut context,
                operation_observation,
            )?;
            let resident_bytes = base.resident_bytes();
            let plan = Arc::clone(
                self.overlay
                    .as_ref()
                    .ok_or_else(|| Error::integrity("overlay binding lost its plan"))?,
            );
            return Ok(WeightDelivery::Overlay(Box::new(OverlayHandoff {
                base: Box::new(base),
                plan,
                binding_index,
                target_digest,
                resident_bytes,
            })));
        }
        self.materialize_base_binding(binding, cancellation, &mut context, operation_observation)
    }

    fn materialize_base_binding(
        &self,
        binding: &Binding,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
        operation_observation: Option<OperationObservation<'_>>,
    ) -> Result<WeightDelivery> {
        if let Some(graph) = binding.target().operation_graph() {
            return self.materialize_operation_graph(
                binding,
                graph,
                cancellation,
                context,
                operation_observation,
            );
        }
        match binding.source().storage() {
            Storage::Plain { span, .. } => {
                if let Some(recipe) = binding.target().conversion_recipe() {
                    return self.materialize_conversion(
                        binding,
                        *span,
                        recipe,
                        cancellation,
                        context,
                    );
                }
                self.materialize_plain(binding, *span, cancellation, context, operation_observation)
            }
            Storage::Quantized(storage) => {
                self.materialize_quantized(binding, storage, cancellation, context)
            }
        }
    }

    fn materialize_operation_graph(
        &self,
        binding: &Binding,
        graph: &OperationGraph,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
        operation_observation: Option<OperationObservation<'_>>,
    ) -> Result<WeightDelivery> {
        let cache_address =
            self.cache_address_for_binding_with_cancellation(binding, None, cancellation)?;
        if let Some(delivery) =
            self.try_cached_prepared(binding, cache_address.as_ref(), cancellation, context)?
        {
            return Ok(delivery);
        }
        let required = self.resource_weights(binding)?;
        if let Some(pipeline_context) = context.as_deref_mut() {
            return pipeline_context.with_resources(required, |pipeline_context| {
                let mut nested_context = Some(pipeline_context);
                self.materialize_operation_graph_after_cache_miss(
                    binding,
                    graph,
                    cache_address.as_ref(),
                    cancellation,
                    &mut nested_context,
                    operation_observation,
                )
            });
        }
        self.materialize_operation_graph_after_cache_miss(
            binding,
            graph,
            cache_address.as_ref(),
            cancellation,
            context,
            operation_observation,
        )
    }

    fn materialize_operation_graph_after_cache_miss(
        &self,
        binding: &Binding,
        graph: &OperationGraph,
        cache_address: Option<&PreparedCacheAddress>,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
        operation_observation: Option<OperationObservation<'_>>,
    ) -> Result<WeightDelivery> {
        let inputs = self.read_operation_inputs(binding, cancellation, context)?;
        cancellation.check()?;
        if self.operation_execution == OperationExecution::Delegate {
            let resident_bytes = unique_plain_span_bytes(binding.sources())?;
            let inputs = binding
                .sources()
                .iter()
                .cloned()
                .zip(inputs)
                .map(|(source, bytes)| OperationInputBytes { source, bytes })
                .collect::<Box<[_]>>();
            return Ok(WeightDelivery::Operation(Box::new(OperationHandoff {
                inputs,
                target: binding.target().clone(),
                graph: graph.clone(),
                resident_bytes,
            })));
        }

        let execution = measure(
            context,
            ExecutionPhase::Transform,
            binding.target().output_size(),
            cancellation,
            |cancellation| match operation_observation {
                Some(observation) => graph.execute_host_observed(
                    &inputs,
                    self.preparation,
                    cancellation,
                    observation.work_ordinal,
                    observation.observer,
                ),
                None => graph.execute_host(&inputs, self.preparation, cancellation),
            },
        )?;
        if graph.nodes().is_empty() {
            observe_identity(
                operation_observation,
                OperationLocation::Binding,
                binding.target().output_size(),
            );
        }
        let estimated_scratch = graph.estimate_host_scratch_bytes()?;
        if execution.peak_scratch_bytes() > estimated_scratch {
            return Err(Error::integrity(
                "operation graph exceeded its prevalidated scratch estimate",
            ));
        }
        let bytes = execution.into_output();
        if view_len_u64(&bytes)? != binding.target().output_size() {
            return Err(Error::integrity(
                "operation graph output differs from the planned byte length",
            ));
        }
        if let (Some(cache), Some(address)) = (self.cache, cache_address) {
            let _publication = cache.publish_reader(
                CacheNamespace::Prepared,
                address.key(),
                address.compatibility(),
                bytes.as_slice(),
                cancellation,
            )?;
        }
        prepared_delivery(binding, bytes, PreparedOrigin::OperationGraph)
    }

    fn read_operation_inputs(
        &self,
        binding: &Binding,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
    ) -> Result<Box<[ByteView]>> {
        let mut span_views = HashMap::<SourceSpan, ByteView>::new();
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(binding.sources().len())
            .map_err(|_error| Error::limit("could not allocate operation input views"))?;
        for source in binding.sources() {
            cancellation.check()?;
            let Storage::Plain { span, .. } = source.storage() else {
                return Err(Error::unsupported(
                    "operation graph input is not plain tensor storage",
                ));
            };
            let bytes = if let Some(bytes) = span_views.get(span) {
                bytes.clone()
            } else {
                let bytes = self.read_source(*span, cancellation, context)?;
                span_views.insert(*span, bytes.clone());
                bytes
            };
            inputs.push(bytes);
        }
        cancellation.check()?;
        Ok(inputs.into_boxed_slice())
    }

    fn materialize_plain(
        &self,
        binding: &Binding,
        span: SourceSpan,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
        operation_observation: Option<OperationObservation<'_>>,
    ) -> Result<WeightDelivery> {
        let cache_address =
            self.cache_address_for_binding_with_cancellation(binding, None, cancellation)?;
        if let Some(delivery) =
            self.try_cached_prepared(binding, cache_address.as_ref(), cancellation, context)?
        {
            return Ok(delivery);
        }

        let required = self.resource_weights(binding)?;
        if let Some(pipeline_context) = context.as_deref_mut() {
            return pipeline_context.with_resources(required, |pipeline_context| {
                let mut nested_context = Some(pipeline_context);
                self.materialize_plain_after_cache_miss(
                    binding,
                    span,
                    cache_address.as_ref(),
                    cancellation,
                    &mut nested_context,
                    operation_observation,
                )
            });
        }
        self.materialize_plain_after_cache_miss(
            binding,
            span,
            cache_address.as_ref(),
            cancellation,
            context,
            operation_observation,
        )
    }

    fn materialize_plain_after_cache_miss(
        &self,
        binding: &Binding,
        span: SourceSpan,
        cache_address: Option<&PreparedCacheAddress>,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
        operation_observation: Option<OperationObservation<'_>>,
    ) -> Result<WeightDelivery> {
        let source = self.read_source(span, cancellation, context)?;
        cancellation.check()?;
        let (bytes, origin) = if binding.target().transforms().is_empty() {
            observe_identity(
                operation_observation,
                OperationLocation::Binding,
                binding.target().output_size(),
            );
            (source, PreparedOrigin::Source)
        } else {
            let prepared = measure(
                context,
                ExecutionPhase::Transform,
                binding.target().output_size(),
                cancellation,
                |cancellation| {
                    prepare_chain_with_operation_telemetry(
                        self.preparation,
                        binding.target().transforms(),
                        binding.target().shape(),
                        &source,
                        cancellation,
                        operation_observation,
                    )
                },
            )?;
            (prepared, PreparedOrigin::Transform)
        };
        if view_len_u64(&bytes)? != binding.target().output_size() {
            return Err(Error::integrity(
                "materialized host bytes differ from the planned output size",
            ));
        }
        if let (Some(cache), Some(address)) = (self.cache, cache_address) {
            let _publication = cache.publish_reader(
                CacheNamespace::Prepared,
                address.key(),
                address.compatibility(),
                bytes.as_slice(),
                cancellation,
            )?;
        }
        prepared_delivery(binding, bytes, origin)
    }

    fn materialize_conversion(
        &self,
        binding: &Binding,
        span: SourceSpan,
        recipe: &ConversionRecipe,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
    ) -> Result<WeightDelivery> {
        let cache_address =
            self.cache_address_for_binding_with_cancellation(binding, None, cancellation)?;
        if let Some(delivery) =
            self.try_cached_prepared(binding, cache_address.as_ref(), cancellation, context)?
        {
            return Ok(delivery);
        }

        let required = self.resource_weights(binding)?;
        if let Some(pipeline_context) = context.as_deref_mut() {
            return pipeline_context.with_resources(required, |pipeline_context| {
                let mut nested_context = Some(pipeline_context);
                self.materialize_conversion_after_cache_miss(
                    binding,
                    span,
                    recipe,
                    cancellation,
                    &mut nested_context,
                )
            });
        }
        self.materialize_conversion_after_cache_miss(binding, span, recipe, cancellation, context)
    }

    fn materialize_conversion_after_cache_miss(
        &self,
        binding: &Binding,
        span: SourceSpan,
        recipe: &ConversionRecipe,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
    ) -> Result<WeightDelivery> {
        let source = self.read_source(span, cancellation, context)?;
        cancellation.check()?;
        let resident_bytes = view_len_u64(&source)?;
        Ok(WeightDelivery::Conversion(Box::new(ConversionHandoff {
            source: binding.source().clone(),
            target: binding.target().clone(),
            recipe: recipe.clone(),
            source_bytes: source,
            resident_bytes,
        })))
    }

    fn resource_weights(&self, binding: &Binding) -> Result<ResourceWeights> {
        if let Some(graph) = binding.target().operation_graph() {
            let (source_bytes, source_resident_bytes) =
                self.operation_source_bytes(binding.sources())?;
            let (scratch_bytes, prepared_bytes) = match self.operation_execution {
                OperationExecution::Host => (
                    graph.estimate_host_scratch_bytes()?,
                    binding.target().output_size(),
                ),
                OperationExecution::Delegate => (0, source_resident_bytes),
            };
            return Ok(ResourceWeights::new(
                source_bytes,
                scratch_bytes,
                prepared_bytes,
            ));
        }
        let (source_bytes, source_resident_bytes) = match binding.source().storage() {
            Storage::Plain { span, .. } => {
                let source_bytes = if self.source_span_is_guaranteed_mapped(*span)? {
                    0
                } else {
                    span.len()
                };
                (source_bytes, span.len())
            }
            Storage::Quantized(storage) => self.quantized_source_bytes(storage)?,
        };
        let scratch_bytes = peak_transform_scratch_bytes(binding.target().transforms())?;
        let prepared_bytes = if binding.target().conversion_recipe().is_some()
            || matches!(binding.source().storage(), Storage::Quantized(_))
        {
            source_resident_bytes
        } else {
            binding.target().output_size()
        };
        Ok(ResourceWeights::new(
            source_bytes,
            scratch_bytes,
            prepared_bytes,
        ))
    }

    fn operation_source_bytes(&self, sources: &[SourceTensor]) -> Result<(u64, u64)> {
        let mut unique_spans = HashSet::new();
        let mut source_bytes = 0_u64;
        let mut resident_bytes = 0_u64;
        for source in sources {
            let Storage::Plain { span, .. } = source.storage() else {
                return Err(Error::unsupported(
                    "operation graph input is not plain tensor storage",
                ));
            };
            if !unique_spans.insert(*span) {
                continue;
            }
            resident_bytes = resident_bytes
                .checked_add(span.len())
                .ok_or_else(|| Error::limit("operation input resident bytes overflow u64"))?;
            if !self.source_span_is_guaranteed_mapped(*span)? {
                source_bytes = source_bytes
                    .checked_add(span.len())
                    .ok_or_else(|| Error::limit("operation source-read bytes overflow u64"))?;
            }
        }
        Ok((source_bytes, resident_bytes))
    }

    fn quantized_source_bytes(&self, storage: &QuantizedStorage) -> Result<(u64, u64)> {
        quantized_source_byte_totals(storage, |span| self.source_span_is_guaranteed_mapped(span))
    }

    fn source_span_is_guaranteed_mapped(&self, span: SourceSpan) -> Result<bool> {
        Ok(cfg!(feature = "mmap")
            && self.source_kind(span)? == SourceKind::RetainedSnapshot
            && self.checkpoint.access_mode() == AccessMode::Mmap)
    }

    fn source_span_uses_mapping(&self, span: SourceSpan) -> Result<bool> {
        Ok(cfg!(feature = "mmap")
            && self.source_kind(span)? == SourceKind::RetainedSnapshot
            && matches!(
                self.checkpoint.access_mode(),
                AccessMode::Auto | AccessMode::Mmap
            ))
    }

    fn source_kind(&self, span: SourceSpan) -> Result<SourceKind> {
        let file_index = usize::try_from(span.file().ordinal())
            .map_err(|_error| Error::limit("source file ordinal does not fit usize"))?;
        let source = self
            .checkpoint
            .inventory()
            .files()
            .get(file_index)
            .ok_or_else(|| Error::integrity("source span refers to a missing inventory file"))?;
        Ok(source.kind())
    }

    fn read_source(
        &self,
        span: SourceSpan,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
    ) -> Result<ByteView> {
        let mapped = self.source_span_uses_mapping(span)?;
        let phase = if mapped {
            ExecutionPhase::Mapping
        } else {
            ExecutionPhase::SourceRead
        };
        measure(context, phase, span.len(), cancellation, |cancellation| {
            self.checkpoint
                .read_span_with_cancellation(span, cancellation)
        })
    }

    fn overlay_for_binding(
        &self,
        binding: &Binding,
        cancellation: &CancellationToken,
    ) -> Result<Option<(usize, ContentDigest)>> {
        cancellation.check()?;
        let Some(overlay) = &self.overlay else {
            return Ok(None);
        };
        cancellation.check()?;
        let Some(binding_index) = overlay_binding_index(overlay, binding.target().name()) else {
            return Ok(None);
        };
        cancellation.check()?;
        if overlay.bindings()[binding_index].operations().is_empty() {
            return Ok(None);
        }
        let target_digest =
            overlay.target_digest_with_cancellation(binding.target().name(), cancellation)?;
        cancellation.check()?;
        Ok(Some((binding_index, target_digest)))
    }

    fn try_cached_prepared(
        &self,
        binding: &Binding,
        address: Option<&PreparedCacheAddress>,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
    ) -> Result<Option<WeightDelivery>> {
        let (Some(cache), Some(address)) = (self.cache, address) else {
            return Ok(None);
        };
        let lookup = measure(
            context,
            ExecutionPhase::CacheLookup,
            0,
            cancellation,
            |cancellation| {
                cache.lookup_with_validation_and_cancellation(
                    CacheNamespace::Prepared,
                    address.key(),
                    address.compatibility(),
                    CacheValidation::TrustedMetadata,
                    cancellation,
                )
            },
        )?;
        let CacheLookup::Hit(entry) = lookup else {
            return Ok(None);
        };
        if entry.info().payload_len() != binding.target().output_size() {
            let _record = cache.evict(
                CacheNamespace::Prepared,
                address.key(),
                EvictionReason::Corrupt,
                cancellation,
            )?;
            return Ok(None);
        }

        let required = ResourceWeights::new(0, 0, binding.target().output_size());
        let read = if let Some(pipeline_context) = context.as_deref_mut() {
            pipeline_context.with_resources(required, |pipeline_context| {
                let mut nested_context = Some(pipeline_context);
                measure(
                    &mut nested_context,
                    ExecutionPhase::CacheLookup,
                    binding.target().output_size(),
                    cancellation,
                    |cancellation| read_cache_entry(entry, self.cache_validation, cancellation),
                )
            })
        } else {
            measure(
                context,
                ExecutionPhase::CacheLookup,
                binding.target().output_size(),
                cancellation,
                |cancellation| read_cache_entry(entry, self.cache_validation, cancellation),
            )
        };
        match read {
            Ok(bytes) => prepared_delivery(binding, bytes, PreparedOrigin::Cache).map(Some),
            Err(error) if error.category() == ErrorCategory::Integrity => {
                let _record = cache.evict(
                    CacheNamespace::Prepared,
                    address.key(),
                    EvictionReason::Corrupt,
                    cancellation,
                )?;
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn materialize_quantized(
        &self,
        binding: &Binding,
        storage: &QuantizedStorage,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
    ) -> Result<WeightDelivery> {
        let capability = binding.target().quantized_route().ok_or_else(|| {
            Error::unsupported("quantized binding has no selected route capability")
        })?;
        let cache_address =
            self.cache_address_for_binding_with_cancellation(binding, None, cancellation)?;
        if let Some(delivery) =
            self.try_cached_prepared(binding, cache_address.as_ref(), cancellation, context)?
        {
            return Ok(delivery);
        }
        let required = self.resource_weights(binding)?;
        if let Some(pipeline_context) = context.as_deref_mut() {
            return pipeline_context.with_resources(required, |pipeline_context| {
                let mut nested_context = Some(pipeline_context);
                self.materialize_quantized_after_cache_miss(
                    binding,
                    storage,
                    capability,
                    cancellation,
                    &mut nested_context,
                )
            });
        }
        self.materialize_quantized_after_cache_miss(
            binding,
            storage,
            capability,
            cancellation,
            context,
        )
    }

    fn materialize_quantized_after_cache_miss(
        &self,
        binding: &Binding,
        storage: &QuantizedStorage,
        capability: &RouteCapability,
        cancellation: &CancellationToken,
        context: &mut Option<&mut PrepareContext<'_>>,
    ) -> Result<WeightDelivery> {
        let resident_bytes = storage_byte_len(storage)?;
        let payload = self.read_source(storage.span(), cancellation, context)?;
        let mut span_views = HashMap::new();
        span_views.insert(storage.span(), payload.clone());
        let mut companions = BTreeMap::new();
        for (role, companion) in storage.companions() {
            cancellation.check()?;
            let bytes = if let Some(bytes) = span_views.get(&companion.span()) {
                bytes.clone()
            } else {
                let bytes = self.read_source(companion.span(), cancellation, context)?;
                span_views.insert(companion.span(), bytes.clone());
                bytes
            };
            companions.insert(role.clone(), bytes);
        }
        cancellation.check()?;
        Ok(WeightDelivery::Quantized(Box::new(QuantizedHandoff {
            source: binding.source().clone(),
            target: binding.target().clone(),
            storage: storage.clone(),
            capability: capability.clone(),
            payload,
            companions,
            resident_bytes,
        })))
    }
}

fn prepared_delivery(
    binding: &Binding,
    bytes: ByteView,
    origin: PreparedOrigin,
) -> Result<WeightDelivery> {
    let resident_bytes = view_len_u64(&bytes)?;
    Ok(WeightDelivery::Prepared(Box::new(PreparedWeight {
        source_names: binding
            .sources()
            .iter()
            .map(|source| source.name().clone())
            .collect(),
        target: binding.target().clone(),
        bytes,
        origin,
        resident_bytes,
    })))
}

fn validate_inventory_binding(checkpoint: &Checkpoint, binding: &Binding) -> Result<()> {
    for source in binding.sources() {
        validate_inventory_source(checkpoint, source)?;
    }
    Ok(())
}

fn validate_complete_plan_inventory(
    checkpoint: &Checkpoint,
    plan: &BindingPlan,
    cancellation: &CancellationToken,
) -> Result<()> {
    for source in plan.sources() {
        cancellation.check()?;
        validate_inventory_source(checkpoint, source)?;
    }
    if plan.sources().len() != checkpoint.inventory().len() {
        return Err(Error::integrity(
            "binding plan source inventory omits an opened checkpoint tensor",
        ));
    }
    for record in checkpoint.inventory().iter() {
        cancellation.check()?;
        let source = plan
            .sources()
            .binary_search_by(|source| source.name().as_str().cmp(record.name()))
            .ok()
            .and_then(|index| plan.sources().get(index))
            .ok_or_else(|| {
                Error::integrity(
                    "binding plan source inventory differs from the opened checkpoint inventory",
                )
            })?;
        if source.shape() != record.shape() || source.storage() != record.storage() {
            return Err(Error::integrity(
                "binding plan source inventory differs from the opened checkpoint inventory",
            ));
        }
    }
    cancellation.check()
}

fn validate_inventory_source(checkpoint: &Checkpoint, source: &SourceTensor) -> Result<()> {
    let record = checkpoint
        .inventory()
        .tensor(source.name().as_str())
        .ok_or_else(|| {
            Error::binding("binding plan source is absent from the opened checkpoint")
        })?;
    if record.shape() != source.shape() || record.storage() != source.storage() {
        return Err(Error::integrity(
            "binding plan source descriptor differs from the opened checkpoint inventory",
        ));
    }
    Ok(())
}

fn validate_overlay_plan(
    plan: &BindingPlan,
    overlay: &OverlayPlan,
    cancellation: &CancellationToken,
) -> Result<()> {
    let mut selected_by_base = BTreeMap::<TensorName, Vec<&Binding>>::new();
    for binding in plan.bindings() {
        cancellation.check()?;
        let canonical = overlay.aliases().resolve(binding.target().name()).clone();
        cancellation.check()?;
        selected_by_base.entry(canonical).or_default().push(binding);
    }
    for overlay_binding in overlay.bindings() {
        cancellation.check()?;
        let Some(matches) = selected_by_base.get(overlay_binding.base().name()) else {
            continue;
        };
        for binding in matches {
            cancellation.check()?;
            validate_overlay_base(plan, binding, overlay_binding, cancellation)?;
        }
    }
    cancellation.check()
}

fn validate_overlay_base(
    plan: &BindingPlan,
    binding: &Binding,
    overlay_binding: &OverlayBinding,
    cancellation: &CancellationToken,
) -> Result<()> {
    cancellation.check()?;
    if binding.target().shape() != overlay_binding.base().shape() {
        return Err(Error::binding(
            "overlay base shape differs from its selected plan target",
        ));
    }
    if binding_source_content_with_cancellation(plan, binding, cancellation)?
        != overlay_binding.base().content()
    {
        return Err(Error::integrity(
            "overlay base content identity differs from its selected plan source",
        ));
    }
    cancellation.check()
}

fn overlay_binding_index(overlay: &OverlayPlan, target: &TensorName) -> Option<usize> {
    let canonical = overlay.aliases().resolve(target);
    overlay
        .bindings()
        .binary_search_by(|binding| binding.base().name().cmp(canonical))
        .ok()
}

fn peak_transform_scratch_bytes(transforms: &[crate::plan::PlannedTransform]) -> Result<u64> {
    // The original source and final output use their own pipeline budgets.
    // Scratch covers only provider workspace and transform intermediates that
    // remain simultaneously live while one step prepares its successor.
    transforms
        .iter()
        .enumerate()
        .try_fold(0_u64, |peak, (index, transform)| {
            let input_intermediate = index
                .checked_sub(1)
                .and_then(|previous| transforms.get(previous))
                .map_or(0, crate::plan::PlannedTransform::output_size);
            let output_intermediate = transforms
                .get(index.saturating_add(1))
                .map_or(0, |_next| transform.output_size());
            let live_intermediates = input_intermediate
                .checked_add(output_intermediate)
                .ok_or_else(|| Error::limit("adjacent transform scratch bytes overflow u64"))?;
            let live_scratch = live_intermediates
                .checked_add(transform.scratch_bytes())
                .ok_or_else(|| Error::limit("provider transform scratch bytes overflow u64"))?;
            Ok(peak.max(live_scratch))
        })
}

fn collect_file_ordinals(
    storage: &Storage,
    ordinals: &mut BTreeSet<u32>,
    cancellation: &CancellationToken,
) -> Result<()> {
    cancellation.check()?;
    ordinals.insert(storage.span().file().ordinal());
    if let Storage::Quantized(storage) = storage {
        for companion in storage.companions().values() {
            cancellation.check()?;
            ordinals.insert(companion.span().file().ordinal());
        }
    }
    cancellation.check()
}

fn storage_byte_len(storage: &QuantizedStorage) -> Result<u64> {
    quantized_source_byte_totals(storage, |_span| Ok(false))
        .map(|(_read_bytes, resident_bytes)| resident_bytes)
}

fn unique_plain_span_bytes(sources: &[SourceTensor]) -> Result<u64> {
    let mut spans = HashSet::new();
    let mut bytes = 0_u64;
    for source in sources {
        let Storage::Plain { span, .. } = source.storage() else {
            return Err(Error::unsupported(
                "operation graph input is not plain tensor storage",
            ));
        };
        if spans.insert(*span) {
            bytes = bytes
                .checked_add(span.len())
                .ok_or_else(|| Error::limit("operation input resident bytes overflow u64"))?;
        }
    }
    Ok(bytes)
}

fn quantized_source_byte_totals(
    storage: &QuantizedStorage,
    mut is_mapped: impl FnMut(SourceSpan) -> Result<bool>,
) -> Result<(u64, u64)> {
    let mut unique_spans = HashSet::new();
    let mut read_bytes = 0_u64;
    let mut resident_bytes = 0_u64;
    for span in std::iter::once(storage.span()).chain(
        storage
            .companions()
            .values()
            .map(crate::quantization::CompanionTensor::span),
    ) {
        if !unique_spans.insert(span) {
            continue;
        }
        resident_bytes = resident_bytes
            .checked_add(span.len())
            .ok_or_else(|| Error::limit("quantized resident byte length overflows u64"))?;
        if !is_mapped(span)? {
            read_bytes = read_bytes
                .checked_add(span.len())
                .ok_or_else(|| Error::limit("quantized source-read bytes overflow u64"))?;
        }
    }
    Ok((read_bytes, resident_bytes))
}

fn prepare_chain_with_operation_telemetry(
    preparation: &PreparationEngine,
    transforms: &[PlannedTransform],
    shape: &[u64],
    source: &ByteView,
    cancellation: &CancellationToken,
    observation: Option<OperationObservation<'_>>,
) -> Result<ByteView> {
    let Some(observation) = observation else {
        return preparation.prepare_chain_with_cancellation(
            transforms,
            shape,
            source,
            cancellation,
        );
    };

    let mut current = source.clone();
    for (index, planned) in transforms.iter().enumerate() {
        cancellation.check()?;
        let input_bytes = view_len_u64(&current)?;
        let request =
            PrepareRequest::new(planned.transform(), shape, &current, planned.output_size())
                .with_expected_scratch_bytes(planned.scratch_bytes());
        let started = Instant::now();
        let output = preparation.prepare_with_cancellation(&request, cancellation)?;
        let duration = started.elapsed();
        let reused = same_view(&output, &current);
        observation
            .observer
            .observe(&ExecutionEvent::OperationFinished {
                work_ordinal: observation.work_ordinal,
                location: OperationLocation::PlannedTransform { index },
                kind: if reused {
                    OperationKind::Identity
                } else {
                    OperationKind::for_transform(planned.transform())
                },
                duration,
                input_bytes,
                output_bytes: planned.output_size(),
                materialized_output_bytes: if reused { 0 } else { planned.output_size() },
            });
        current = output;
    }
    cancellation.check()?;
    Ok(current)
}

fn observe_identity(
    observation: Option<OperationObservation<'_>>,
    location: OperationLocation,
    bytes: u64,
) {
    if let Some(observation) = observation {
        observation
            .observer
            .observe(&ExecutionEvent::OperationFinished {
                work_ordinal: observation.work_ordinal,
                location,
                kind: OperationKind::Identity,
                duration: Duration::ZERO,
                input_bytes: bytes,
                output_bytes: bytes,
                materialized_output_bytes: 0,
            });
    }
}

fn same_view(left: &ByteView, right: &ByteView) -> bool {
    left.len() == right.len() && std::ptr::eq(left.as_slice().as_ptr(), right.as_slice().as_ptr())
}

fn view_len_u64(view: &ByteView) -> Result<u64> {
    u64::try_from(view.len())
        .map_err(|_error| Error::limit("materialized byte length does not fit u64"))
}

fn measure<R>(
    context: &mut Option<&mut PrepareContext<'_>>,
    phase: ExecutionPhase,
    bytes: u64,
    cancellation: &CancellationToken,
    operation: impl FnOnce(&CancellationToken) -> R,
) -> R {
    match context {
        Some(context) => context.measure(phase, bytes, operation),
        None => operation(cancellation),
    }
}

fn read_cache_entry(
    entry: crate::cache::CacheEntry,
    validation: CacheValidation,
    cancellation: &CancellationToken,
) -> Result<ByteView> {
    cancellation.check()?;
    let expected_digest = entry.info().payload_digest();
    let expected_length = entry.info().payload_len();
    let length = usize::try_from(expected_length)
        .map_err(|_error| Error::limit("cached payload length does not fit usize"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_error| Error::limit("could not allocate cached prepared bytes"))?;
    let mut file = entry.into_payload();
    let block_len = length.min(CACHE_READ_BLOCK_BYTES);
    let mut block = Vec::new();
    block
        .try_reserve_exact(block_len)
        .map_err(|_error| Error::limit("could not allocate cache read buffer"))?;
    block.resize(block_len, 0);
    let mut hasher = (validation == CacheValidation::Full).then(Sha256::new);
    while bytes.len() < length {
        cancellation.check()?;
        let remaining = length - bytes.len();
        let chunk_len = remaining.min(block.len());
        let read = file.read(&mut block[..chunk_len]).map_err(|source| {
            Error::with_source(
                ErrorCategory::Io,
                "read validated prepared-cache payload",
                source,
            )
        })?;
        if read == 0 {
            return Err(Error::integrity(
                "validated prepared-cache payload ended unexpectedly",
            ));
        }
        let chunk = &block[..read];
        if let Some(hasher) = &mut hasher {
            hasher.update(chunk);
        }
        bytes.extend_from_slice(chunk);
    }
    cancellation.check()?;
    let actual_length = file
        .metadata()
        .map_err(|source| Error::io("inspect validated prepared-cache payload", source))?
        .len();
    if actual_length != expected_length {
        return Err(Error::integrity(
            "cached payload length changed while it was read",
        ));
    }
    if let Some(hasher) = hasher {
        let actual_digest = ContentDigest::from_bytes(hasher.finalize().into());
        if actual_digest != expected_digest {
            return Err(Error::integrity(
                "cached payload digest differs from its metadata envelope",
            ));
        }
    }
    Ok(ByteView::from_boxed(bytes.into_boxed_slice()))
}

#[cfg(test)]
mod tests {
    use super::{peak_transform_scratch_bytes, quantized_source_byte_totals};
    use crate::identity::{ImplementationId, StableName};
    use crate::plan::PlannedTransform;
    use crate::prepare::{Representation, TransformSpec};
    use crate::quantization::{Companion, CompanionTensor, Packing, QuantizedStorage};
    use crate::tensor::{DType, FileId, SourceSpan};
    use crate::{ErrorCategory, Result};

    fn planned_transform(output_size: u64, scratch_bytes: u64) -> Result<PlannedTransform> {
        let representation = Representation::contiguous(DType::U8);
        Ok(PlannedTransform::new(
            TransformSpec::new(
                ImplementationId::new(
                    StableName::parse("materialize-test")?,
                    StableName::parse("scratch")?,
                    1,
                ),
                representation.clone(),
                representation,
            ),
            output_size,
        )
        .with_scratch_bytes(scratch_bytes))
    }

    #[test]
    fn transform_scratch_peak_includes_provider_and_adjacent_intermediates() -> Result<()> {
        let transforms = [
            planned_transform(4, 1)?,
            planned_transform(8, 3)?,
            planned_transform(4, 5)?,
        ];

        assert_eq!(peak_transform_scratch_bytes(&transforms)?, 15);
        Ok(())
    }

    #[test]
    fn transform_scratch_peak_rejects_checked_addition_overflow() -> Result<()> {
        let transforms = [planned_transform(u64::MAX, 0)?, planned_transform(1, 1)?];

        assert_eq!(
            peak_transform_scratch_bytes(&transforms)
                .err()
                .map(|error| error.category()),
            Some(ErrorCategory::ResourceLimit)
        );
        Ok(())
    }

    #[test]
    fn quantized_resource_totals_deduplicate_shared_companion_spans() -> Result<()> {
        let span = SourceSpan::new(FileId::from_ordinal(0), 0, 2)?;
        let implementation = ImplementationId::new(
            StableName::parse("materialize-test")?,
            StableName::parse("packed")?,
            1,
        );
        let companion = CompanionTensor::new("shared", DType::U8, [2_u64], span)?;
        let storage =
            QuantizedStorage::builder(implementation, [4_u64], span, Packing::flat_blocks(2, 1)?)
                .companions([
                    Companion::new(StableName::parse("scale")?, companion.clone()),
                    Companion::new(StableName::parse("zero")?, companion),
                ])
                .build()?;
        let mut classified_spans = 0_u64;

        let totals = quantized_source_byte_totals(&storage, |_span| {
            classified_spans += 1;
            Ok(false)
        })?;

        assert_eq!(totals, (2, 2));
        assert_eq!(classified_spans, 1);
        Ok(())
    }
}
