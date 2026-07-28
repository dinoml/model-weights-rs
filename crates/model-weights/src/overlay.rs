//! Exact aliases and validated, ordered model-weight overlays.
//!
//! Format-specific adapter parsing stays outside this module. Callers supply
//! normalized base shapes, source tensors, scales, and provider recipe
//! identities. Validation records a lazy composition recipe; it never mutates
//! a checkpoint or eagerly merges weights.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::identity::{ContentDigest, ImplementationId, OverlayId};
use crate::plan::{ConversionRecipe, SourceTensor, TensorName};
use crate::quantization::Storage;
use crate::{CancellationToken, Error, ErrorCategory, Result};

/// The canonical ordered-overlay schema version.
pub const OVERLAY_SCHEMA_VERSION: u32 = 1;

/// One exact alias or tied-weight reference.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Alias {
    alias: TensorName,
    target: TensorName,
}

impl Alias {
    /// Creates an alias that resolves to `target`.
    #[must_use]
    pub const fn new(alias: TensorName, target: TensorName) -> Self {
        Self { alias, target }
    }

    /// Returns the alternate exact name.
    #[must_use]
    pub const fn alias(&self) -> &TensorName {
        &self.alias
    }

    /// Returns the next exact name in the alias chain.
    #[must_use]
    pub const fn target(&self) -> &TensorName {
        &self.target
    }
}

/// A validated, deterministic exact-name alias graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "AliasTableWire")]
pub struct AliasTable {
    entries: BTreeMap<TensorName, TensorName>,
}

impl AliasTable {
    /// Creates an empty alias graph.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Validates exact aliases and tied-weight references.
    ///
    /// # Errors
    ///
    /// Returns a binding error for a duplicate alias, self-reference, or
    /// direct or indirect cycle.
    pub fn new(aliases: impl IntoIterator<Item = Alias>) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for alias in aliases {
            if alias.alias == alias.target {
                return Err(Error::binding("alias cannot reference itself"));
            }
            if entries.insert(alias.alias, alias.target).is_some() {
                return Err(Error::binding(
                    "alias graph contains more than one entry for an exact name",
                ));
            }
        }
        validate_alias_cycles(&entries)?;
        Ok(Self { entries })
    }

    /// Resolves all aliases to a canonical exact name.
    #[must_use]
    pub fn resolve<'a>(&'a self, name: &'a TensorName) -> &'a TensorName {
        let mut current = name;
        while let Some(next) = self.entries.get(current) {
            current = next;
        }
        current
    }

    /// Returns aliases in deterministic exact-name order.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&TensorName, &TensorName)> {
        self.entries.iter()
    }

    /// Returns whether no aliases are declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a digest of the canonical alias graph.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if canonical serialization fails.
    pub fn digest(&self) -> Result<ContentDigest> {
        let bytes = canonical_json(self, "serialize alias graph")?;
        Ok(ContentDigest::hash("alias-table-v1", [bytes]))
    }
}

impl Default for AliasTable {
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a> IntoIterator for &'a AliasTable {
    type Item = (&'a TensorName, &'a TensorName);
    type IntoIter = std::collections::btree_map::Iter<'a, TensorName, TensorName>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

#[derive(Debug, Clone, Deserialize)]
struct AliasTableWire {
    entries: BTreeMap<TensorName, TensorName>,
}

impl TryFrom<AliasTableWire> for AliasTable {
    type Error = Error;

    fn try_from(wire: AliasTableWire) -> Result<Self> {
        Self::new(
            wire.entries
                .into_iter()
                .map(|(alias, target)| Alias::new(alias, target)),
        )
    }
}

/// A finite, canonicalized overlay scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteScale(f64);

impl FiniteScale {
    /// Validates an overlay scale.
    ///
    /// Positive and negative zero are normalized to one representation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error for NaN or either infinity.
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::invalid("overlay scale must be finite"));
        }
        Ok(Self(if value == 0.0 { 0.0 } else { value }))
    }

    /// Returns the finite scale.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.0
    }
}

impl Serialize for FiniteScale {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for FiniteScale {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(f64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// One immutable base tensor available to overlay composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "BaseTensorWire")]
pub struct BaseTensor {
    name: TensorName,
    shape: Box<[u64]>,
    content: ContentDigest,
}

impl BaseTensor {
    /// Creates a base descriptor with independent content provenance.
    ///
    /// # Errors
    ///
    /// Returns a resource-limit error when shape element arithmetic overflows.
    pub fn new(
        name: TensorName,
        shape: impl Into<Box<[u64]>>,
        content: ContentDigest,
    ) -> Result<Self> {
        let shape = shape.into();
        checked_elements(&shape, "base tensor")?;
        Ok(Self {
            name,
            shape,
            content,
        })
    }

    /// Returns the canonical base name.
    #[must_use]
    pub const fn name(&self) -> &TensorName {
        &self.name
    }

    /// Returns the logical base shape.
    #[must_use]
    pub const fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the independent base content identity.
    #[must_use]
    pub const fn content(&self) -> ContentDigest {
        self.content
    }
}

#[derive(Debug, Clone, Deserialize)]
struct BaseTensorWire {
    name: TensorName,
    shape: Box<[u64]>,
    content: ContentDigest,
}

impl TryFrom<BaseTensorWire> for BaseTensor {
    type Error = Error;

    fn try_from(wire: BaseTensorWire) -> Result<Self> {
        Self::new(wire.name, wire.shape, wire.content)
    }
}

/// One normalized lazy overlay operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OverlayOperation {
    /// Replace the current target value.
    Replace {
        /// Exact base or alias target.
        target: TensorName,
        /// Complete replacement source.
        value: Box<SourceTensor>,
        /// Versioned provider semantics.
        implementation: ImplementationId,
    },
    /// Add a shape-compatible tensor after scaling.
    Add {
        /// Exact base or alias target.
        target: TensorName,
        /// Additive source.
        value: Box<SourceTensor>,
        /// Finite multiplier applied to the source.
        scale: FiniteScale,
        /// Versioned provider semantics.
        implementation: ImplementationId,
    },
    /// Apply a two-factor rank update to a matrix.
    LowRank {
        /// Exact base or alias target.
        target: TensorName,
        /// Factor shaped `[rank, input]`.
        down: Box<SourceTensor>,
        /// Factor shaped `[output, rank]`.
        up: Box<SourceTensor>,
        /// Declared factor rank.
        rank: u64,
        /// Finite update multiplier.
        scale: FiniteScale,
        /// Versioned provider semantics.
        implementation: ImplementationId,
    },
    /// Execute a normalized provider DAG lazily.
    Recipe {
        /// Exact base or alias target.
        target: TensorName,
        /// Recipe inputs keyed by exact external name.
        inputs: BTreeMap<TensorName, SourceTensor>,
        /// Declared logical output shape.
        output_shape: Box<[u64]>,
        /// Language-neutral provider recipe.
        recipe: ConversionRecipe,
    },
}

impl OverlayOperation {
    /// Creates a lazy replacement operation.
    #[must_use]
    pub fn replace(
        target: TensorName,
        value: SourceTensor,
        implementation: ImplementationId,
    ) -> Self {
        Self::Replace {
            target,
            value: Box::new(value),
            implementation,
        }
    }

    /// Creates a lazy scaled-addition operation.
    #[must_use]
    pub fn add(
        target: TensorName,
        value: SourceTensor,
        scale: FiniteScale,
        implementation: ImplementationId,
    ) -> Self {
        Self::Add {
            target,
            value: Box::new(value),
            scale,
            implementation,
        }
    }

    /// Creates a lazy two-factor matrix rank update.
    #[must_use]
    pub fn low_rank(
        target: TensorName,
        down: SourceTensor,
        up: SourceTensor,
        rank: u64,
        scale: FiniteScale,
        implementation: ImplementationId,
    ) -> Self {
        Self::LowRank {
            target,
            down: Box::new(down),
            up: Box::new(up),
            rank,
            scale,
            implementation,
        }
    }

    /// Creates a lazy provider-defined recipe operation.
    #[must_use]
    pub fn recipe(
        target: TensorName,
        inputs: BTreeMap<TensorName, SourceTensor>,
        output_shape: impl Into<Box<[u64]>>,
        recipe: ConversionRecipe,
    ) -> Self {
        Self::Recipe {
            target,
            inputs,
            output_shape: output_shape.into(),
            recipe,
        }
    }

    /// Returns the exact target name before alias resolution.
    #[must_use]
    pub const fn target(&self) -> &TensorName {
        match self {
            Self::Replace { target, .. }
            | Self::Add { target, .. }
            | Self::LowRank { target, .. }
            | Self::Recipe { target, .. } => target,
        }
    }

    fn sources(&self) -> Box<dyn Iterator<Item = &SourceTensor> + '_> {
        match self {
            Self::Replace { value, .. } | Self::Add { value, .. } => {
                Box::new(std::iter::once(value.as_ref()))
            }
            Self::LowRank { down, up, .. } => Box::new([down.as_ref(), up.as_ref()].into_iter()),
            Self::Recipe { inputs, .. } => Box::new(inputs.values()),
        }
    }
}

/// One independently invalidatable ordered overlay layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "OverlayLayerWire")]
pub struct OverlayLayer {
    schema_version: u32,
    id: OverlayId,
    source_digests: Box<[ContentDigest]>,
    operations: Box<[OverlayOperation]>,
}

impl OverlayLayer {
    /// Creates a content-addressed normalized layer.
    ///
    /// Operation order is semantic and is preserved exactly. File ordinals in
    /// every operation source are checked against `source_digests`.
    ///
    /// # Errors
    ///
    /// Returns a binding, invalid-format, or resource-limit error for an empty
    /// layer, missing source digest, invalid recipe input, or serialization
    /// failure.
    pub fn new(
        source_digests: impl Into<Box<[ContentDigest]>>,
        operations: impl Into<Box<[OverlayOperation]>>,
    ) -> Result<Self> {
        let source_digests = source_digests.into();
        let operations = operations.into();
        if operations.is_empty() {
            return Err(Error::binding(
                "overlay layer must contain at least one operation",
            ));
        }
        for operation in &operations {
            validate_operation_sources(operation, source_digests.len())?;
            validate_recipe_inputs(operation)?;
        }
        let payload = OverlayLayerIdentity {
            schema_version: OVERLAY_SCHEMA_VERSION,
            source_digests: &source_digests,
            operations: &operations,
        };
        let bytes = canonical_json(&payload, "serialize overlay layer identity")?;
        let id = OverlayId::from_digest(ContentDigest::hash("overlay-layer-v1", [bytes]));
        Ok(Self {
            schema_version: OVERLAY_SCHEMA_VERSION,
            id,
            source_digests,
            operations,
        })
    }

    /// Returns the normalized layer schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the independent layer identity.
    #[must_use]
    pub const fn id(&self) -> OverlayId {
        self.id
    }

    /// Returns source digests in file-ordinal order.
    #[must_use]
    pub const fn source_digests(&self) -> &[ContentDigest] {
        &self.source_digests
    }

    /// Returns lazy operations in semantic order.
    #[must_use]
    pub const fn operations(&self) -> &[OverlayOperation] {
        &self.operations
    }

    /// Serializes the independently identified layer to canonical JSON.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if serialization fails.
    pub fn to_canonical_json(&self) -> Result<Box<[u8]>> {
        canonical_json(self, "serialize overlay layer")
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OverlayLayerWire {
    schema_version: u32,
    id: OverlayId,
    source_digests: Box<[ContentDigest]>,
    operations: Box<[OverlayOperation]>,
}

impl TryFrom<OverlayLayerWire> for OverlayLayer {
    type Error = Error;

    fn try_from(wire: OverlayLayerWire) -> Result<Self> {
        if wire.schema_version != OVERLAY_SCHEMA_VERSION {
            return Err(Error::invalid("unsupported overlay layer schema version"));
        }
        let layer = Self::new(wire.source_digests, wire.operations)?;
        if layer.id != wire.id {
            return Err(Error::integrity(
                "overlay layer identity does not match normalized content",
            ));
        }
        Ok(layer)
    }
}

#[derive(Serialize)]
struct OverlayLayerIdentity<'a> {
    schema_version: u32,
    source_digests: &'a [ContentDigest],
    operations: &'a [OverlayOperation],
}

/// How multiple operations targeting one base tensor are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConflictPolicy {
    /// Reject a second operation for the same canonical target.
    Reject,
    /// Apply every operation in explicit layer and operation order.
    Ordered,
}

/// Whether a consumer requests lazy composition or an explicit eager merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CompositionMode {
    /// Retain recipes and compose at preparation or kernel time.
    Lazy,
    /// Explicitly request eager materialization by a separate executor.
    Eager,
}

/// Locates one operation without collapsing its layer identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverlayOperationRef {
    layer: OverlayId,
    operation_index: u32,
}

impl OverlayOperationRef {
    /// Returns the independently invalidatable layer.
    #[must_use]
    pub const fn layer(self) -> OverlayId {
        self.layer
    }

    /// Returns the operation's semantic ordinal within the layer.
    #[must_use]
    pub const fn operation_index(self) -> u32 {
        self.operation_index
    }
}

/// Lazy composition for one canonical base tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverlayBinding {
    base: BaseTensor,
    operations: Box<[OverlayOperationRef]>,
}

impl OverlayBinding {
    /// Returns the independently identified base tensor.
    #[must_use]
    pub const fn base(&self) -> &BaseTensor {
        &self.base
    }

    /// Returns ordered layer/operation references.
    #[must_use]
    pub const fn operations(&self) -> &[OverlayOperationRef] {
        &self.operations
    }
}

/// A validated ordered overlay composition plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "OverlayPlanWire")]
pub struct OverlayPlan {
    schema_version: u32,
    mode: CompositionMode,
    conflict_policy: ConflictPolicy,
    aliases: AliasTable,
    bases: Box<[BaseTensor]>,
    layers: Box<[OverlayLayer]>,
    bindings: Box<[OverlayBinding]>,
    digest: ContentDigest,
}

impl OverlayPlan {
    /// Validates aliases, base tensors, and ordered overlay layers.
    ///
    /// # Errors
    ///
    /// Returns a binding or resource-limit error for duplicate bases/layers,
    /// dangling aliases, missing targets, conflicts, incompatible shapes or
    /// ranks, invalid recipes, or operation ordinals that exceed `u32`.
    pub fn build(
        mut bases: Vec<BaseTensor>,
        aliases: AliasTable,
        layers: Vec<OverlayLayer>,
        conflict_policy: ConflictPolicy,
        mode: CompositionMode,
    ) -> Result<Self> {
        bases.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        if bases.windows(2).any(|pair| pair[0].name == pair[1].name) {
            return Err(Error::binding(
                "overlay plan contains a duplicate base tensor",
            ));
        }
        let base_indices = bases
            .iter()
            .enumerate()
            .map(|(index, base)| (base.name.clone(), index))
            .collect::<BTreeMap<_, _>>();
        for (alias, _) in &aliases {
            if base_indices.contains_key(alias) {
                return Err(Error::binding(
                    "alias graph shadows an exact base tensor name",
                ));
            }
            if !base_indices.contains_key(aliases.resolve(alias)) {
                return Err(Error::binding(
                    "alias graph resolves to a missing base tensor",
                ));
            }
        }

        let mut layer_ids = BTreeSet::new();
        let mut operation_refs = vec![Vec::<OverlayOperationRef>::new(); bases.len()];
        for layer in &layers {
            if !layer_ids.insert(layer.id()) {
                return Err(Error::binding(
                    "overlay plan contains the same layer identity more than once",
                ));
            }
            for (operation_index, operation) in layer.operations().iter().enumerate() {
                let canonical = aliases.resolve(operation.target());
                let base_index = base_indices.get(canonical).copied().ok_or_else(|| {
                    Error::binding(format!(
                        "overlay target {} does not resolve to a base tensor",
                        operation.target()
                    ))
                })?;
                validate_operation_shape(operation, &bases[base_index])?;
                if conflict_policy == ConflictPolicy::Reject
                    && !operation_refs[base_index].is_empty()
                {
                    return Err(Error::binding(format!(
                        "multiple overlay operations target canonical tensor {}",
                        bases[base_index].name()
                    )));
                }
                let operation_index = u32::try_from(operation_index).map_err(|error| {
                    Error::with_source(
                        ErrorCategory::ResourceLimit,
                        "overlay operation index does not fit u32",
                        error,
                    )
                })?;
                operation_refs[base_index].push(OverlayOperationRef {
                    layer: layer.id(),
                    operation_index,
                });
            }
        }

        let bindings = bases
            .iter()
            .cloned()
            .zip(operation_refs)
            .map(|(base, operations)| OverlayBinding {
                base,
                operations: operations.into_boxed_slice(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let bases = bases.into_boxed_slice();
        let layers = layers.into_boxed_slice();
        let payload = OverlayPlanIdentity {
            schema_version: OVERLAY_SCHEMA_VERSION,
            mode,
            conflict_policy,
            aliases: &aliases,
            bases: &bases,
            layers: &layers,
            bindings: &bindings,
        };
        let bytes = canonical_json(&payload, "serialize overlay plan identity")?;
        let digest = ContentDigest::hash("overlay-plan-v1", [bytes]);
        Ok(Self {
            schema_version: OVERLAY_SCHEMA_VERSION,
            mode,
            conflict_policy,
            aliases,
            bases,
            layers,
            bindings,
            digest,
        })
    }

    /// Parses and verifies canonical ordered-overlay JSON.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format or integrity error for malformed,
    /// non-canonical, unsupported, or internally inconsistent content.
    pub fn from_canonical_json(bytes: &[u8]) -> Result<Self> {
        let plan: Self = serde_json::from_slice(bytes).map_err(|error| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "parse ordered-overlay JSON",
                error,
            )
        })?;
        if plan.to_canonical_json()?.as_ref() != bytes {
            return Err(Error::invalid("ordered-overlay JSON is not canonical"));
        }
        Ok(plan)
    }

    /// Returns the overlay plan schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the explicit composition mode.
    #[must_use]
    pub const fn mode(&self) -> CompositionMode {
        self.mode
    }

    /// Returns the conflict policy.
    #[must_use]
    pub const fn conflict_policy(&self) -> ConflictPolicy {
        self.conflict_policy
    }

    /// Returns the validated exact aliases.
    #[must_use]
    pub const fn aliases(&self) -> &AliasTable {
        &self.aliases
    }

    /// Returns canonical bases in exact-name order.
    #[must_use]
    pub const fn bases(&self) -> &[BaseTensor] {
        &self.bases
    }

    /// Returns layers in semantic composition order.
    #[must_use]
    pub const fn layers(&self) -> &[OverlayLayer] {
        &self.layers
    }

    /// Returns lazy bindings in canonical base-name order.
    #[must_use]
    pub const fn bindings(&self) -> &[OverlayBinding] {
        &self.bindings
    }

    /// Returns the complete ordered composition identity.
    #[must_use]
    pub const fn digest(&self) -> ContentDigest {
        self.digest
    }

    /// Returns an identity affected only by one base and its ordered overlay operations.
    ///
    /// # Errors
    ///
    /// Returns a binding error when `target` does not resolve to a base, or an
    /// invalid-format error if canonical serialization fails.
    pub fn target_digest(&self, target: &TensorName) -> Result<ContentDigest> {
        self.target_digest_with_cancellation(target, &CancellationToken::new())
    }

    /// Returns one target identity with cooperative cancellation.
    ///
    /// # Errors
    ///
    /// Returns a binding error when `target` does not resolve to a base, an
    /// integrity error for an invalid internal operation reference, an
    /// invalid-format error if canonical serialization fails, or a cancellation
    /// error while indexing layers or collecting operations.
    pub fn target_digest_with_cancellation(
        &self,
        target: &TensorName,
        cancellation: &CancellationToken,
    ) -> Result<ContentDigest> {
        cancellation.check()?;
        let canonical = self.aliases.resolve(target);
        cancellation.check()?;
        let binding = self
            .bindings
            .binary_search_by(|binding| binding.base.name().cmp(canonical))
            .ok()
            .map(|index| &self.bindings[index])
            .ok_or_else(|| Error::binding("overlay digest target does not resolve to a base"))?;
        cancellation.check()?;
        let mut layers = BTreeMap::new();
        for layer in &self.layers {
            cancellation.check()?;
            layers.insert(layer.id(), layer);
        }
        cancellation.check()?;
        let mut operations = Vec::new();
        operations
            .try_reserve_exact(binding.operations.len())
            .map_err(|_error| {
                Error::limit("could not allocate target overlay operation identity")
            })?;
        for operation_ref in &binding.operations {
            cancellation.check()?;
            let layer = layers
                .get(&operation_ref.layer)
                .copied()
                .ok_or_else(|| Error::integrity("overlay binding references a missing layer"))?;
            let index = usize::try_from(operation_ref.operation_index).map_err(|error| {
                Error::with_source(
                    ErrorCategory::ResourceLimit,
                    "overlay operation index does not fit usize",
                    error,
                )
            })?;
            let operation = layer.operations().get(index).ok_or_else(|| {
                Error::integrity("overlay binding operation index lies outside its layer")
            })?;
            operations.push(target_operation_digest(layer, operation, cancellation)?);
        }
        cancellation.check()?;
        let payload = TargetOverlayIdentity {
            mode: self.mode,
            conflict_policy: self.conflict_policy,
            base: &binding.base,
            operations: &operations,
        };
        let bytes = canonical_json(&payload, "serialize target overlay identity")?;
        cancellation.check()?;
        let digest = ContentDigest::hash("overlay-target-v2", [bytes]);
        cancellation.check()?;
        Ok(digest)
    }

    /// Serializes the complete plan to deterministic JSON bytes.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if serialization fails.
    pub fn to_canonical_json(&self) -> Result<Box<[u8]>> {
        canonical_json(self, "serialize overlay plan")
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OverlayPlanWire {
    schema_version: u32,
    mode: CompositionMode,
    conflict_policy: ConflictPolicy,
    aliases: AliasTable,
    bases: Box<[BaseTensor]>,
    layers: Box<[OverlayLayer]>,
    bindings: Box<[OverlayBinding]>,
    digest: ContentDigest,
}

impl TryFrom<OverlayPlanWire> for OverlayPlan {
    type Error = Error;

    fn try_from(wire: OverlayPlanWire) -> Result<Self> {
        if wire.schema_version != OVERLAY_SCHEMA_VERSION {
            return Err(Error::invalid("unsupported ordered-overlay schema version"));
        }
        let plan = Self::build(
            wire.bases.into_vec(),
            wire.aliases,
            wire.layers.into_vec(),
            wire.conflict_policy,
            wire.mode,
        )?;
        if plan.bindings != wire.bindings || plan.digest != wire.digest {
            return Err(Error::integrity(
                "overlay plan derived bindings or identity do not match normalized content",
            ));
        }
        Ok(plan)
    }
}

#[derive(Serialize)]
struct OverlayPlanIdentity<'a> {
    schema_version: u32,
    mode: CompositionMode,
    conflict_policy: ConflictPolicy,
    aliases: &'a AliasTable,
    bases: &'a [BaseTensor],
    layers: &'a [OverlayLayer],
    bindings: &'a [OverlayBinding],
}

#[derive(Serialize)]
struct TargetOverlayIdentity<'a> {
    mode: CompositionMode,
    conflict_policy: ConflictPolicy,
    base: &'a BaseTensor,
    operations: &'a [ContentDigest],
}

#[derive(Serialize)]
struct TargetOperationIdentity<'a> {
    operation: &'a OverlayOperation,
    source_digests: &'a [ReferencedSourceDigest],
}

#[derive(Serialize)]
struct ReferencedSourceDigest {
    file_ordinal: u32,
    content: ContentDigest,
}

fn validate_alias_cycles(entries: &BTreeMap<TensorName, TensorName>) -> Result<()> {
    for start in entries.keys() {
        let mut seen = BTreeSet::new();
        let mut current = start;
        while let Some(next) = entries.get(current) {
            if !seen.insert(current) {
                return Err(Error::binding("alias graph contains a cycle"));
            }
            current = next;
        }
    }
    Ok(())
}

fn target_operation_digest(
    layer: &OverlayLayer,
    operation: &OverlayOperation,
    cancellation: &CancellationToken,
) -> Result<ContentDigest> {
    cancellation.check()?;
    let source_digests = referenced_source_digests(layer, operation, cancellation)?;
    let payload = TargetOperationIdentity {
        operation,
        source_digests: &source_digests,
    };
    let bytes = canonical_json(&payload, "serialize target overlay operation identity")?;
    cancellation.check()?;
    Ok(ContentDigest::hash("overlay-target-operation-v1", [bytes]))
}

fn referenced_source_digests(
    layer: &OverlayLayer,
    operation: &OverlayOperation,
    cancellation: &CancellationToken,
) -> Result<Vec<ReferencedSourceDigest>> {
    let mut file_ordinals = Vec::new();
    for source in operation.sources() {
        cancellation.check()?;
        extend_storage_file_ordinals(source.storage(), &mut file_ordinals, cancellation)?;
    }
    file_ordinals.sort_unstable();
    file_ordinals.dedup();

    let mut source_digests = Vec::new();
    source_digests
        .try_reserve_exact(file_ordinals.len())
        .map_err(|_error| {
            Error::limit("could not allocate target overlay source digest identity")
        })?;
    for file_ordinal in file_ordinals {
        cancellation.check()?;
        let index = usize::try_from(file_ordinal).map_err(|error| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "overlay source file ordinal does not fit usize",
                error,
            )
        })?;
        let content = layer.source_digests().get(index).copied().ok_or_else(|| {
            Error::integrity("overlay operation references a missing source digest")
        })?;
        source_digests.push(ReferencedSourceDigest {
            file_ordinal,
            content,
        });
    }
    cancellation.check()?;
    Ok(source_digests)
}

fn extend_storage_file_ordinals(
    storage: &Storage,
    file_ordinals: &mut Vec<u32>,
    cancellation: &CancellationToken,
) -> Result<()> {
    let companion_count = match storage {
        Storage::Plain { .. } => 0,
        Storage::Quantized(quantized) => quantized.companions().len(),
    };
    let additional = companion_count
        .checked_add(1)
        .ok_or_else(|| Error::limit("overlay source file ordinal count overflows usize"))?;
    file_ordinals
        .try_reserve(additional)
        .map_err(|_error| Error::limit("could not allocate target overlay source file ordinals"))?;
    file_ordinals.push(storage.span().file().ordinal());
    if let Storage::Quantized(quantized) = storage {
        for companion in quantized.companions().values() {
            cancellation.check()?;
            file_ordinals.push(companion.span().file().ordinal());
        }
    }
    Ok(())
}

fn validate_operation_sources(operation: &OverlayOperation, digest_count: usize) -> Result<()> {
    for source in operation.sources() {
        if storage_has_missing_digest(source.storage(), digest_count)? {
            return Err(Error::binding(
                "overlay source or quantized companion file ordinal has no ordered content digest",
            ));
        }
    }
    Ok(())
}

fn storage_has_missing_digest(storage: &Storage, digest_count: usize) -> Result<bool> {
    let file_index = usize::try_from(storage.span().file().ordinal()).map_err(|error| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "overlay source file ordinal does not fit usize",
            error,
        )
    })?;
    if file_index >= digest_count {
        return Ok(true);
    }
    let Storage::Quantized(quantized) = storage else {
        return Ok(false);
    };
    for companion in quantized.companions().values() {
        let file_index = usize::try_from(companion.span().file().ordinal()).map_err(|error| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "overlay companion file ordinal does not fit usize",
                error,
            )
        })?;
        if file_index >= digest_count {
            return Ok(true);
        }
    }
    Ok(false)
}

fn validate_recipe_inputs(operation: &OverlayOperation) -> Result<()> {
    let OverlayOperation::Recipe {
        target,
        inputs,
        output_shape,
        recipe,
    } = operation
    else {
        return Ok(());
    };
    checked_elements(output_shape, "overlay recipe output")?;
    if recipe.inputs().len() != inputs.len()
        || recipe
            .inputs()
            .iter()
            .any(|name| inputs.get(name).is_none_or(|source| source.name() != name))
    {
        return Err(Error::binding(
            "overlay recipe inputs do not match supplied source tensors",
        ));
    }
    if !recipe.outputs().contains_key(target) {
        return Err(Error::binding(
            "overlay recipe does not declare its target tensor as an output",
        ));
    }
    Ok(())
}

fn validate_operation_shape(operation: &OverlayOperation, base: &BaseTensor) -> Result<()> {
    match operation {
        OverlayOperation::Replace { value, .. } | OverlayOperation::Add { value, .. } => {
            if value.shape() != base.shape() {
                return Err(Error::binding(
                    "replacement or additive overlay shape differs from its base tensor",
                ));
            }
        }
        OverlayOperation::LowRank { down, up, rank, .. } => {
            let [output, input] = base.shape() else {
                return Err(Error::binding(
                    "low-rank overlay currently requires a rank-two base tensor",
                ));
            };
            if *rank == 0 || down.shape() != [*rank, *input] || up.shape() != [*output, *rank] {
                return Err(Error::binding(
                    "low-rank overlay factor shapes do not match base shape and declared rank",
                ));
            }
        }
        OverlayOperation::Recipe { output_shape, .. } => {
            if output_shape.as_ref() != base.shape() {
                return Err(Error::binding(
                    "overlay recipe output shape differs from its base tensor",
                ));
            }
        }
    }
    Ok(())
}

fn checked_elements(shape: &[u64], description: &str) -> Result<u64> {
    shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| Error::limit(format!("{description} element count overflows u64")))
    })
}

fn canonical_json(value: &impl Serialize, message: &'static str) -> Result<Box<[u8]>> {
    serde_json::to_vec(value)
        .map(Vec::into_boxed_slice)
        .map_err(|error| Error::with_source(ErrorCategory::InvalidFormat, message, error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::StableName;
    use crate::quantization::{Companion, CompanionTensor, Packing, QuantizedStorage};
    use crate::tensor::{DType, FileId, SourceSpan};

    fn tensor(value: &str) -> Result<TensorName> {
        TensorName::parse(value)
    }

    fn stable(value: &str) -> Result<StableName> {
        StableName::parse(value)
    }

    fn implementation(operation: &str) -> Result<ImplementationId> {
        Ok(ImplementationId::new(
            stable("test-provider")?,
            stable(operation)?,
            1,
        ))
    }

    fn source(name: &str, shape: &[u64], file: u32) -> Result<SourceTensor> {
        let span = SourceSpan::new(FileId::from_ordinal(file), 0, DType::F32.byte_len(shape)?)?;
        SourceTensor::new(
            tensor(name)?,
            shape,
            Storage::Plain {
                dtype: DType::F32,
                span,
            },
        )
    }

    fn base(name: &str, shape: &[u64]) -> Result<BaseTensor> {
        BaseTensor::new(
            tensor(name)?,
            shape,
            ContentDigest::hash("base", [name.as_bytes()]),
        )
    }

    fn same_layer_two_target_plan(
        digest_a: ContentDigest,
        digest_b: ContentDigest,
        scale_b: f64,
    ) -> Result<OverlayPlan> {
        let layer = OverlayLayer::new(
            [digest_a, digest_b],
            [
                OverlayOperation::add(
                    tensor("a")?,
                    source("add-a", &[2], 0)?,
                    FiniteScale::new(1.0)?,
                    implementation("add")?,
                ),
                OverlayOperation::add(
                    tensor("b")?,
                    source("add-b", &[2], 1)?,
                    FiniteScale::new(scale_b)?,
                    implementation("add")?,
                ),
            ],
        )?;
        OverlayPlan::build(
            vec![base("a", &[2])?, base("b", &[2])?],
            AliasTable::empty(),
            vec![layer],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        )
    }

    fn quantized_source_with_companion(
        name: &str,
        payload_file: u32,
        companion_file: u32,
    ) -> Result<SourceTensor> {
        let shape = [4_u64];
        let payload = SourceSpan::new(FileId::from_ordinal(payload_file), 0, 2)?;
        let role = stable("scale")?;
        let companion = CompanionTensor::new(
            "scale",
            DType::F32,
            [1],
            SourceSpan::new(FileId::from_ordinal(companion_file), 0, 4)?,
        )?;
        let storage = QuantizedStorage::builder(
            implementation("q4")?,
            shape,
            payload,
            Packing::flat_blocks(2, 1)?,
        )
        .required_companions([role.clone()])
        .companions([Companion::new(role, companion)])
        .build()?;
        SourceTensor::new(tensor(name)?, shape, Storage::Quantized(storage))
    }

    #[test]
    fn finite_scale_rejects_non_finite_values_and_normalizes_zero() -> Result<()> {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let Err(_) = FiniteScale::new(invalid) else {
                return Err(Error::invalid(
                    "non-finite overlay scale was unexpectedly accepted",
                ));
            };
        }
        assert_eq!(FiniteScale::new(-0.0)?.value().to_bits(), 0.0_f64.to_bits());
        Ok(())
    }

    #[test]
    fn base_tensor_deserialization_replays_shape_validation() -> Result<()> {
        let digest = ContentDigest::hash("base", [b"overflow"]);
        let encoded =
            format!(r#"{{"name":"weight","shape":[18446744073709551615,2],"content":"{digest}"}}"#);
        let Err(_) = serde_json::from_str::<BaseTensor>(&encoded) else {
            return Err(Error::invalid(
                "overflowing base tensor shape unexpectedly deserialized",
            ));
        };
        Ok(())
    }

    #[test]
    fn alias_table_rejects_indirect_cycle() -> Result<()> {
        let result = AliasTable::new([
            Alias::new(tensor("a")?, tensor("b")?),
            Alias::new(tensor("b")?, tensor("a")?),
        ]);

        let Err(error) = result else {
            return Err(Error::binding("alias cycle was unexpectedly accepted"));
        };
        assert_eq!(error.message(), "alias graph contains a cycle");
        Ok(())
    }

    #[test]
    fn tied_aliases_share_one_base_binding_and_target_identity() -> Result<()> {
        let canonical = tensor("weight")?;
        let tied = tensor("tied.weight")?;
        let aliases = AliasTable::new([Alias::new(tied.clone(), canonical.clone())])?;
        let layer = OverlayLayer::new(
            [ContentDigest::hash("adapter", [b"tied"])],
            [OverlayOperation::add(
                tied.clone(),
                source("add", &[2], 0)?,
                FiniteScale::new(1.0)?,
                implementation("add")?,
            )],
        )?;
        let plan = OverlayPlan::build(
            vec![base("weight", &[2])?],
            aliases,
            vec![layer],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        )?;

        assert_eq!(plan.bindings().len(), 1);
        assert_eq!(plan.target_digest(&canonical)?, plan.target_digest(&tied)?);
        Ok(())
    }

    #[test]
    fn overlay_plan_rejects_a_missing_target() -> Result<()> {
        let layer = OverlayLayer::new(
            [ContentDigest::hash("adapter", [b"missing"])],
            [OverlayOperation::add(
                tensor("missing")?,
                source("add", &[2], 0)?,
                FiniteScale::new(1.0)?,
                implementation("add")?,
            )],
        )?;
        let result = OverlayPlan::build(
            vec![base("weight", &[2])?],
            AliasTable::empty(),
            vec![layer],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        );

        let Err(error) = result else {
            return Err(Error::binding(
                "overlay operation with a missing target was unexpectedly accepted",
            ));
        };
        assert_eq!(
            error.message(),
            "overlay target missing does not resolve to a base tensor"
        );
        Ok(())
    }

    #[test]
    fn canonical_overlay_plan_round_trips() -> Result<()> {
        let layer = OverlayLayer::new(
            [ContentDigest::hash("adapter", [b"round-trip"])],
            [OverlayOperation::add(
                tensor("weight")?,
                source("add", &[2], 0)?,
                FiniteScale::new(0.5)?,
                implementation("add")?,
            )],
        )?;
        let plan = OverlayPlan::build(
            vec![base("weight", &[2])?],
            AliasTable::empty(),
            vec![layer],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        )?;
        let decoded = OverlayPlan::from_canonical_json(&plan.to_canonical_json()?)?;

        assert_eq!(decoded, plan);
        Ok(())
    }

    #[test]
    fn low_rank_overlay_rejects_incompatible_rank() -> Result<()> {
        let operation = OverlayOperation::LowRank {
            target: tensor("weight")?,
            down: Box::new(source("down", &[3, 4], 0)?),
            up: Box::new(source("up", &[2, 3], 0)?),
            rank: 2,
            scale: FiniteScale::new(1.0)?,
            implementation: implementation("lora")?,
        };
        let layer = OverlayLayer::new([ContentDigest::hash("adapter", [b"one"])], [operation])?;
        let result = OverlayPlan::build(
            vec![base("weight", &[2, 4])?],
            AliasTable::empty(),
            vec![layer],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        );

        let Err(error) = result else {
            return Err(Error::binding(
                "incompatible low-rank factors were unexpectedly accepted",
            ));
        };
        assert_eq!(
            error.message(),
            "low-rank overlay factor shapes do not match base shape and declared rank"
        );
        Ok(())
    }

    #[test]
    fn layer_order_changes_plan_and_target_digest() -> Result<()> {
        let make_layer = |source_name: &str| -> Result<OverlayLayer> {
            OverlayLayer::new(
                [ContentDigest::hash("adapter", [source_name.as_bytes()])],
                [OverlayOperation::Add {
                    target: tensor("weight")?,
                    value: Box::new(source(source_name, &[2], 0)?),
                    scale: FiniteScale::new(1.0)?,
                    implementation: implementation("add")?,
                }],
            )
        };
        let first = make_layer("first")?;
        let second = make_layer("second")?;
        let forward = OverlayPlan::build(
            vec![base("weight", &[2])?],
            AliasTable::empty(),
            vec![first.clone(), second.clone()],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        )?;
        let reverse = OverlayPlan::build(
            vec![base("weight", &[2])?],
            AliasTable::empty(),
            vec![second, first],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        )?;

        assert_ne!(forward.digest(), reverse.digest());
        assert_ne!(
            forward.target_digest(&tensor("weight")?)?,
            reverse.target_digest(&tensor("weight")?)?
        );
        Ok(())
    }

    #[test]
    fn unrelated_layer_does_not_change_target_digest() -> Result<()> {
        let layer_a = OverlayLayer::new(
            [ContentDigest::hash("adapter", [b"a"])],
            [OverlayOperation::Add {
                target: tensor("a")?,
                value: Box::new(source("add-a", &[2], 0)?),
                scale: FiniteScale::new(1.0)?,
                implementation: implementation("add")?,
            }],
        )?;
        let without_b = OverlayPlan::build(
            vec![base("a", &[2])?, base("b", &[2])?],
            AliasTable::empty(),
            vec![layer_a.clone()],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        )?;
        let layer_b = OverlayLayer::new(
            [ContentDigest::hash("adapter", [b"b"])],
            [OverlayOperation::Add {
                target: tensor("b")?,
                value: Box::new(source("add-b", &[2], 0)?),
                scale: FiniteScale::new(1.0)?,
                implementation: implementation("add")?,
            }],
        )?;
        let with_b = OverlayPlan::build(
            vec![base("a", &[2])?, base("b", &[2])?],
            AliasTable::empty(),
            vec![layer_a, layer_b],
            ConflictPolicy::Ordered,
            CompositionMode::Lazy,
        )?;

        assert_eq!(
            without_b.target_digest(&tensor("a")?)?,
            with_b.target_digest(&tensor("a")?)?
        );
        Ok(())
    }

    #[test]
    fn same_layer_unrelated_operation_does_not_change_target_digest() -> Result<()> {
        let digest_a = ContentDigest::hash("adapter", [b"a"]);
        let digest_b = ContentDigest::hash("adapter", [b"b"]);
        let before = same_layer_two_target_plan(digest_a, digest_b, 1.0)?;
        let changed_b = same_layer_two_target_plan(digest_a, digest_b, 2.0)?;

        assert_ne!(before.digest(), changed_b.digest());
        assert_eq!(
            before.target_digest(&tensor("a")?)?,
            changed_b.target_digest(&tensor("a")?)?
        );
        Ok(())
    }

    #[test]
    fn same_layer_unreferenced_source_digest_does_not_change_target_digest() -> Result<()> {
        let digest_a = ContentDigest::hash("adapter", [b"a"]);
        let before = same_layer_two_target_plan(
            digest_a,
            ContentDigest::hash("adapter", [b"b-before"]),
            1.0,
        )?;
        let changed_b = same_layer_two_target_plan(
            digest_a,
            ContentDigest::hash("adapter", [b"b-after"]),
            1.0,
        )?;

        assert_ne!(before.digest(), changed_b.digest());
        assert_eq!(
            before.target_digest(&tensor("a")?)?,
            changed_b.target_digest(&tensor("a")?)?
        );
        Ok(())
    }

    #[test]
    fn same_layer_referenced_source_digest_changes_target_digest() -> Result<()> {
        let digest_b = ContentDigest::hash("adapter", [b"b"]);
        let before = same_layer_two_target_plan(
            ContentDigest::hash("adapter", [b"a-before"]),
            digest_b,
            1.0,
        )?;
        let changed_a = same_layer_two_target_plan(
            ContentDigest::hash("adapter", [b"a-after"]),
            digest_b,
            1.0,
        )?;

        assert_ne!(
            before.target_digest(&tensor("a")?)?,
            changed_a.target_digest(&tensor("a")?)?
        );
        Ok(())
    }

    #[test]
    fn referenced_quantized_companion_digest_changes_target_digest() -> Result<()> {
        let plan_with_companion_digest = |companion_digest: ContentDigest| -> Result<OverlayPlan> {
            let layer = OverlayLayer::new(
                [
                    ContentDigest::hash("adapter", [b"payload"]),
                    companion_digest,
                ],
                [OverlayOperation::add(
                    tensor("a")?,
                    quantized_source_with_companion("quantized-a", 0, 1)?,
                    FiniteScale::new(1.0)?,
                    implementation("add")?,
                )],
            )?;
            OverlayPlan::build(
                vec![base("a", &[4])?],
                AliasTable::empty(),
                vec![layer],
                ConflictPolicy::Ordered,
                CompositionMode::Lazy,
            )
        };
        let before = plan_with_companion_digest(ContentDigest::hash("adapter", [b"scale-before"]))?;
        let changed = plan_with_companion_digest(ContentDigest::hash("adapter", [b"scale-after"]))?;

        assert_ne!(
            before.target_digest(&tensor("a")?)?,
            changed.target_digest(&tensor("a")?)?
        );
        Ok(())
    }
}
