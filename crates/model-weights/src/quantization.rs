//! Format-neutral quantized storage descriptors and capability declarations.
//!
//! Format adapters such as GGUF readers normalize their metadata into the
//! types in this module. This crate does not decode packed values or choose a
//! runtime policy. Consumers retain those responsibilities and can associate
//! executable hooks with the versioned [`RouteCapability::implementation`]
//! identity.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::identity::{BackendId, ImplementationId, StableName};
use crate::prepare::Layout;
use crate::tensor::{DType, SourceSpan};
use crate::{Error, ErrorCategory, Result};

/// The schema version emitted for quantized storage descriptors.
pub const QUANTIZED_STORAGE_SCHEMA_VERSION: u32 = 1;

/// Scalar metadata normalized by a format adapter.
///
/// Floating-point values are represented as canonical text by adapters. This
/// keeps metadata equality and serialized cache inputs deterministic across
/// languages and JSON implementations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MetadataValue {
    /// A Boolean value.
    Bool(bool),
    /// A signed integer.
    I64(i64),
    /// An unsigned integer.
    U64(u64),
    /// UTF-8 text.
    Text(Box<str>),
    /// Opaque bytes.
    Bytes(Box<[u8]>),
    /// An ordered sequence.
    Sequence(Box<[MetadataValue]>),
    /// A deterministically ordered object.
    Object(BTreeMap<StableName, MetadataValue>),
}

/// Describes where fixed-size blocks restart in a logical tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PackingOrder {
    /// Blocks run across the flattened logical tensor.
    Flat,
    /// Blocks restart for every line along the named logical axis.
    Axis(u32),
}

/// Describes a fixed-size packed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "BlockPackingWire")]
pub struct BlockPacking {
    values_per_block: u32,
    bytes_per_block: u32,
    order: PackingOrder,
}

impl BlockPacking {
    /// Creates a fixed-size block descriptor.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error when either block dimension is zero.
    pub fn new(values_per_block: u32, bytes_per_block: u32, order: PackingOrder) -> Result<Self> {
        if values_per_block == 0 || bytes_per_block == 0 {
            return Err(Error::invalid(
                "quantized block value and byte counts must be greater than zero",
            ));
        }
        Ok(Self {
            values_per_block,
            bytes_per_block,
            order,
        })
    }

    /// Returns the logical values represented by one block.
    #[must_use]
    pub const fn values_per_block(self) -> u32 {
        self.values_per_block
    }

    /// Returns the stored bytes occupied by one block.
    #[must_use]
    pub const fn bytes_per_block(self) -> u32 {
        self.bytes_per_block
    }

    /// Returns where block boundaries restart.
    #[must_use]
    pub const fn order(self) -> PackingOrder {
        self.order
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct BlockPackingWire {
    values_per_block: u32,
    bytes_per_block: u32,
    order: PackingOrder,
}

impl TryFrom<BlockPackingWire> for BlockPacking {
    type Error = Error;

    fn try_from(wire: BlockPackingWire) -> Result<Self> {
        Self::new(wire.values_per_block, wire.bytes_per_block, wire.order)
    }
}

/// Describes whether the payload has known fixed-size packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Packing {
    /// Values occupy fixed-size blocks.
    Blocks(BlockPacking),
    /// The adapter preserved an unsupported but well-formed opaque payload.
    Opaque,
}

impl Packing {
    /// Creates flat fixed-size packing.
    ///
    /// This is convenient for encodings such as packed four- and six-bit
    /// values. For row- or axis-blocked formats, construct [`BlockPacking`]
    /// with [`PackingOrder::Axis`].
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error when either block dimension is zero.
    pub fn flat_blocks(values_per_block: u32, bytes_per_block: u32) -> Result<Self> {
        Ok(Self::Blocks(BlockPacking::new(
            values_per_block,
            bytes_per_block,
            PackingOrder::Flat,
        )?))
    }
}

/// Describes quantization groups along one logical axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "GroupingWire")]
pub struct Grouping {
    axis: u32,
    values_per_group: u64,
}

impl Grouping {
    /// Creates a grouping rule.
    ///
    /// The enclosing [`QuantizedStorage`] validates that `axis` exists.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error when `values_per_group` is zero.
    pub fn new(axis: u32, values_per_group: u64) -> Result<Self> {
        if values_per_group == 0 {
            return Err(Error::invalid(
                "quantized values per group must be greater than zero",
            ));
        }
        Ok(Self {
            axis,
            values_per_group,
        })
    }

    /// Returns the logical grouping axis.
    #[must_use]
    pub const fn axis(self) -> u32 {
        self.axis
    }

    /// Returns the logical values sharing group metadata.
    #[must_use]
    pub const fn values_per_group(self) -> u64 {
        self.values_per_group
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct GroupingWire {
    axis: u32,
    values_per_group: u64,
}

impl TryFrom<GroupingWire> for Grouping {
    type Error = Error;

    fn try_from(wire: GroupingWire) -> Result<Self> {
        Self::new(wire.axis, wire.values_per_group)
    }
}

/// One normalized scale, zero-point, codebook, or other companion tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "CompanionTensorWire")]
pub struct CompanionTensor {
    tensor: Box<str>,
    dtype: DType,
    shape: Box<[u64]>,
    span: SourceSpan,
}

impl CompanionTensor {
    /// Creates a contiguous, non-quantized companion descriptor.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format or resource-limit error when the shape byte
    /// length differs from the source span.
    pub fn new(
        tensor: impl Into<Box<str>>,
        dtype: DType,
        shape: impl Into<Box<[u64]>>,
        span: SourceSpan,
    ) -> Result<Self> {
        let tensor = tensor.into();
        if tensor.is_empty() {
            return Err(Error::invalid(
                "quantized companion tensor name must not be empty",
            ));
        }
        let shape = shape.into();
        if dtype.byte_len(&shape)? != span.len() {
            return Err(Error::invalid(
                "quantized companion span length does not match its dtype and shape",
            ));
        }
        Ok(Self {
            tensor,
            dtype,
            shape,
            span,
        })
    }

    /// Returns the source tensor name.
    #[must_use]
    pub const fn tensor(&self) -> &str {
        &self.tensor
    }

    /// Returns the companion scalar dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the logical companion shape.
    #[must_use]
    pub const fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the source byte span.
    #[must_use]
    pub const fn span(&self) -> SourceSpan {
        self.span
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CompanionTensorWire {
    tensor: Box<str>,
    dtype: DType,
    shape: Box<[u64]>,
    span: SourceSpan,
}

impl TryFrom<CompanionTensorWire> for CompanionTensor {
    type Error = Error;

    fn try_from(wire: CompanionTensorWire) -> Result<Self> {
        Self::new(wire.tensor, wire.dtype, wire.shape, wire.span)
    }
}

/// A role-name and tensor pair supplied to a quantized storage builder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Companion {
    role: StableName,
    tensor: CompanionTensor,
}

impl Companion {
    /// Associates a provider-defined role with a companion tensor.
    #[must_use]
    pub const fn new(role: StableName, tensor: CompanionTensor) -> Self {
        Self { role, tensor }
    }

    /// Returns the provider-defined role.
    #[must_use]
    pub const fn role(&self) -> &StableName {
        &self.role
    }

    /// Returns the normalized tensor descriptor.
    #[must_use]
    pub const fn tensor(&self) -> &CompanionTensor {
        &self.tensor
    }
}

/// Honest storage metadata for one packed quantized tensor.
///
/// The `encoding` identity names a format-neutral normalized encoding and its
/// schema version. It does not imply that this crate can decode the payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "QuantizedStorageWire")]
pub struct QuantizedStorage {
    schema_version: u32,
    encoding: ImplementationId,
    logical_shape: Box<[u64]>,
    payload: SourceSpan,
    packing: Packing,
    grouping: Option<Grouping>,
    required_companions: Box<[StableName]>,
    companions: BTreeMap<StableName, CompanionTensor>,
    metadata: BTreeMap<StableName, MetadataValue>,
}

impl QuantizedStorage {
    /// Creates a descriptor without grouping, companions, or extra metadata.
    ///
    /// Use [`Self::builder`] when the encoding requires those fields.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format or resource-limit error for an overflowing
    /// shape, an out-of-range packing axis, or a fixed-block payload length
    /// that does not match the logical shape.
    pub fn new(
        encoding: ImplementationId,
        logical_shape: impl Into<Box<[u64]>>,
        payload: SourceSpan,
        packing: Packing,
    ) -> Result<Self> {
        Self::builder(encoding, logical_shape, payload, packing).build()
    }

    /// Returns a builder for a complete normalized descriptor.
    pub fn builder(
        encoding: ImplementationId,
        logical_shape: impl Into<Box<[u64]>>,
        payload: SourceSpan,
        packing: Packing,
    ) -> QuantizedStorageBuilder {
        QuantizedStorageBuilder {
            encoding,
            logical_shape: logical_shape.into(),
            payload,
            packing,
            grouping: None,
            required_companions: Vec::new(),
            companions: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Returns the descriptor schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the normalized encoding identity.
    #[must_use]
    pub const fn encoding(&self) -> &ImplementationId {
        &self.encoding
    }

    /// Returns the logical tensor shape represented by the payload.
    #[must_use]
    pub const fn logical_shape(&self) -> &[u64] {
        &self.logical_shape
    }

    /// Returns the packed payload span.
    #[must_use]
    pub const fn span(&self) -> SourceSpan {
        self.payload
    }

    /// Returns the packing rule.
    #[must_use]
    pub const fn packing(&self) -> Packing {
        self.packing
    }

    /// Returns the optional quantization grouping rule.
    #[must_use]
    pub const fn grouping(&self) -> Option<Grouping> {
        self.grouping
    }

    /// Returns required companion roles in canonical order.
    #[must_use]
    pub const fn required_companions(&self) -> &[StableName] {
        &self.required_companions
    }

    /// Returns companion tensors keyed by provider-defined role.
    #[must_use]
    pub const fn companions(&self) -> &BTreeMap<StableName, CompanionTensor> {
        &self.companions
    }

    /// Returns normalized provider metadata.
    #[must_use]
    pub const fn metadata(&self) -> &BTreeMap<StableName, MetadataValue> {
        &self.metadata
    }
}

/// Builds a validated quantized storage descriptor.
#[derive(Debug)]
#[must_use]
pub struct QuantizedStorageBuilder {
    encoding: ImplementationId,
    logical_shape: Box<[u64]>,
    payload: SourceSpan,
    packing: Packing,
    grouping: Option<Grouping>,
    required_companions: Vec<StableName>,
    companions: Vec<Companion>,
    metadata: BTreeMap<StableName, MetadataValue>,
}

impl QuantizedStorageBuilder {
    /// Sets the grouping rule.
    pub fn grouping(mut self, grouping: Grouping) -> Self {
        self.grouping = Some(grouping);
        self
    }

    /// Declares companion roles that must be present.
    pub fn required_companions(mut self, roles: impl IntoIterator<Item = StableName>) -> Self {
        self.required_companions.extend(roles);
        self
    }

    /// Adds normalized companion tensors.
    pub fn companions(mut self, companions: impl IntoIterator<Item = Companion>) -> Self {
        self.companions.extend(companions);
        self
    }

    /// Replaces the normalized provider metadata.
    pub fn metadata(mut self, metadata: BTreeMap<StableName, MetadataValue>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Validates and builds the descriptor.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format, binding, or resource-limit error when block
    /// arithmetic overflows, an axis lies outside the logical rank, a role is
    /// duplicated or missing, or the payload length is inconsistent.
    pub fn build(self) -> Result<QuantizedStorage> {
        validate_shape_and_packing(&self.logical_shape, self.payload, self.packing)?;
        if let Some(grouping) = self.grouping {
            validate_axis(
                grouping.axis(),
                self.logical_shape.len(),
                "quantized grouping",
            )?;
        }

        let mut required_companions = self.required_companions;
        required_companions.sort_unstable();
        if required_companions
            .windows(2)
            .any(|window| window[0] == window[1])
        {
            return Err(Error::binding(
                "quantized descriptor contains a duplicate required companion role",
            ));
        }

        let mut companions = BTreeMap::new();
        for companion in self.companions {
            if companions
                .insert(companion.role, companion.tensor)
                .is_some()
            {
                return Err(Error::binding(
                    "quantized descriptor contains a duplicate companion role",
                ));
            }
        }
        if required_companions
            .iter()
            .any(|role| !companions.contains_key(role))
        {
            return Err(Error::binding(
                "quantized descriptor is missing a required companion role",
            ));
        }

        Ok(QuantizedStorage {
            schema_version: QUANTIZED_STORAGE_SCHEMA_VERSION,
            encoding: self.encoding,
            logical_shape: self.logical_shape,
            payload: self.payload,
            packing: self.packing,
            grouping: self.grouping,
            required_companions: required_companions.into_boxed_slice(),
            companions,
            metadata: self.metadata,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct QuantizedStorageWire {
    schema_version: u32,
    encoding: ImplementationId,
    logical_shape: Box<[u64]>,
    payload: SourceSpan,
    packing: Packing,
    grouping: Option<Grouping>,
    required_companions: Box<[StableName]>,
    companions: BTreeMap<StableName, CompanionTensor>,
    metadata: BTreeMap<StableName, MetadataValue>,
}

impl TryFrom<QuantizedStorageWire> for QuantizedStorage {
    type Error = Error;

    fn try_from(wire: QuantizedStorageWire) -> Result<Self> {
        if wire.schema_version != QUANTIZED_STORAGE_SCHEMA_VERSION {
            return Err(Error::invalid(
                "unsupported quantized storage descriptor schema version",
            ));
        }
        let companions = wire
            .companions
            .into_iter()
            .map(|(role, tensor)| Companion::new(role, tensor));
        let builder = Self::builder(
            wire.encoding,
            wire.logical_shape,
            wire.payload,
            wire.packing,
        )
        .required_companions(wire.required_companions)
        .companions(companions)
        .metadata(wire.metadata);
        match wire.grouping {
            Some(grouping) => builder.grouping(grouping).build(),
            None => builder.build(),
        }
    }
}

/// Distinguishes ordinary scalar storage from packed quantized storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Storage {
    /// An ordinary contiguous scalar tensor.
    Plain {
        /// The scalar storage dtype.
        dtype: DType,
        /// The exact source bytes.
        span: SourceSpan,
    },
    /// A packed tensor that must follow an explicit quantized route.
    Quantized(QuantizedStorage),
}

impl Storage {
    /// Returns the exact primary payload span.
    #[must_use]
    pub const fn span(&self) -> SourceSpan {
        match self {
            Self::Plain { span, .. } => *span,
            Self::Quantized(storage) => storage.span(),
        }
    }

    /// Returns an ordinary scalar dtype only for plain storage.
    ///
    /// `None` prevents packed bytes from being exposed as a misleading
    /// contiguous scalar buffer.
    #[must_use]
    pub const fn logical_dtype(&self) -> Option<DType> {
        match self {
            Self::Plain { dtype, .. } => Some(*dtype),
            Self::Quantized(_) => None,
        }
    }
}

/// One policy-selectable operation over packed weights.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum QuantizedRoute {
    /// Decode on the host into a provider-supported scalar dtype.
    HostDequant {
        /// The decoded scalar dtype.
        target_dtype: DType,
    },
    /// Keep packed bytes for a backend kernel that reads them directly.
    PackedDirect,
    /// Keep packed bytes for dequantization fused into a kernel tile.
    FusedInTile,
    /// Ask a backend hook to decode into runtime-owned scratch storage.
    DeviceDequantToScratch {
        /// The decoded scratch dtype.
        target_dtype: DType,
    },
    /// Convert to another packed encoding without materializing model policy.
    Repack {
        /// The target normalized encoding identity.
        target_encoding: ImplementationId,
    },
}

/// A complete custom layout ABI requirement for a quantized route.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayoutRequirement {
    name: StableName,
    version: u32,
    parameters: Box<[u8]>,
}

impl LayoutRequirement {
    /// Creates a versioned layout requirement from canonical provider bytes.
    #[must_use]
    pub fn new(name: StableName, version: u32, parameters: impl Into<Box<[u8]>>) -> Self {
        Self {
            name,
            version,
            parameters: parameters.into(),
        }
    }

    /// Copies a custom preparation layout into a quantized requirement.
    ///
    /// Contiguous layouts return `None`, which is the canonical capability
    /// representation for contiguous output.
    #[must_use]
    pub fn from_layout(layout: &Layout) -> Option<Self> {
        layout
            .custom_parts()
            .map(|(name, version, parameters)| Self::new(name.clone(), version, parameters))
    }

    /// Returns the layout name.
    #[must_use]
    pub const fn name(&self) -> &StableName {
        &self.name
    }

    /// Returns the layout ABI version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns canonical provider-defined layout parameters.
    #[must_use]
    pub const fn parameters(&self) -> &[u8] {
        &self.parameters
    }

    /// Returns whether this requirement exactly matches a preparation layout.
    #[must_use]
    pub fn matches_layout(&self, layout: &Layout) -> bool {
        layout
            .custom_parts()
            .is_some_and(|(name, version, parameters)| {
                name == self.name() && version == self.version() && parameters == self.parameters()
            })
    }
}

/// A granular, versioned claim for one quantized operation.
///
/// The declaration is inert. An application such as `DinoML` selects policy and
/// associates `implementation` with its own host or device execution hook.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "RouteCapabilityWire")]
pub struct RouteCapability {
    source_encoding: ImplementationId,
    route: QuantizedRoute,
    implementation: ImplementationId,
    backend: Option<BackendId>,
    target_layout: Option<LayoutRequirement>,
}

impl RouteCapability {
    /// Creates and validates a capability declaration.
    ///
    /// `backend` is required for packed-direct, fused-in-tile, and
    /// device-to-scratch routes. Host decode and repack declarations may be
    /// backend-independent.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-capability error for a backend-dependent route
    /// without a backend identity.
    pub fn new(
        source_encoding: ImplementationId,
        route: QuantizedRoute,
        implementation: ImplementationId,
        backend: Option<BackendId>,
        target_layout: Option<LayoutRequirement>,
    ) -> Result<Self> {
        if matches!(
            route,
            QuantizedRoute::PackedDirect
                | QuantizedRoute::FusedInTile
                | QuantizedRoute::DeviceDequantToScratch { .. }
        ) && backend.is_none()
        {
            return Err(Error::unsupported(
                "backend-dependent quantized route is missing a backend identity",
            ));
        }
        Ok(Self {
            source_encoding,
            route,
            implementation,
            backend,
            target_layout,
        })
    }

    /// Returns the accepted source encoding.
    #[must_use]
    pub const fn source_encoding(&self) -> &ImplementationId {
        &self.source_encoding
    }

    /// Returns the declared operation.
    #[must_use]
    pub const fn route(&self) -> &QuantizedRoute {
        &self.route
    }

    /// Returns the executable hook identity and byte-affecting version.
    #[must_use]
    pub const fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    /// Returns the required backend ABI, if any.
    #[must_use]
    pub const fn backend(&self) -> Option<BackendId> {
        self.backend
    }

    /// Returns the required target layout name, if any.
    #[must_use]
    pub const fn target_layout(&self) -> Option<&LayoutRequirement> {
        self.target_layout.as_ref()
    }

    /// Returns whether every request dimension matches this capability.
    #[must_use]
    pub fn matches(
        &self,
        source_encoding: &ImplementationId,
        route: &QuantizedRoute,
        backend: Option<BackendId>,
        target_layout: Option<&LayoutRequirement>,
    ) -> bool {
        self.source_encoding == *source_encoding
            && self.route == *route
            && self.backend == backend
            && self.target_layout.as_ref() == target_layout
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RouteCapabilityWire {
    source_encoding: ImplementationId,
    route: QuantizedRoute,
    implementation: ImplementationId,
    backend: Option<BackendId>,
    target_layout: Option<LayoutRequirement>,
}

impl TryFrom<RouteCapabilityWire> for RouteCapability {
    type Error = Error;

    fn try_from(wire: RouteCapabilityWire) -> Result<Self> {
        Self::new(
            wire.source_encoding,
            wire.route,
            wire.implementation,
            wire.backend,
            wire.target_layout,
        )
    }
}

/// A deterministic collection of quantized capability declarations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "CapabilitySetWire")]
pub struct CapabilitySet {
    capabilities: Box<[RouteCapability]>,
}

impl CapabilitySet {
    /// Sorts and validates capability declarations.
    ///
    /// # Errors
    ///
    /// Returns a binding error when the same declaration appears more than
    /// once.
    pub fn new(mut capabilities: Vec<RouteCapability>) -> Result<Self> {
        capabilities.sort_unstable();
        if capabilities.windows(2).any(|window| window[0] == window[1]) {
            return Err(Error::binding(
                "quantized capability set contains a duplicate declaration",
            ));
        }
        Ok(Self {
            capabilities: capabilities.into_boxed_slice(),
        })
    }

    /// Returns declarations in canonical order.
    pub fn iter(&self) -> std::slice::Iter<'_, RouteCapability> {
        self.capabilities.iter()
    }

    /// Returns matching declarations without applying runtime policy.
    pub fn matching<'a>(
        &'a self,
        source_encoding: &'a ImplementationId,
        route: &'a QuantizedRoute,
        backend: Option<BackendId>,
        target_layout: Option<&'a LayoutRequirement>,
    ) -> impl Iterator<Item = &'a RouteCapability> + 'a {
        self.capabilities.iter().filter(move |capability| {
            capability.matches(source_encoding, route, backend, target_layout)
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CapabilitySetWire {
    capabilities: Vec<RouteCapability>,
}

impl TryFrom<CapabilitySetWire> for CapabilitySet {
    type Error = Error;

    fn try_from(wire: CapabilitySetWire) -> Result<Self> {
        Self::new(wire.capabilities)
    }
}

impl<'a> IntoIterator for &'a CapabilitySet {
    type Item = &'a RouteCapability;
    type IntoIter = std::slice::Iter<'a, RouteCapability>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Publishes inert quantized capabilities for application-owned hooks.
///
/// Implementations should return a stable set whose entries identify the
/// corresponding executable hooks. The trait deliberately has no decode
/// method because host buffers, device allocations, streams, and kernels are
/// consumer-owned concerns.
pub trait QuantizedRouteProvider {
    /// Returns the provider's deterministic capability declarations.
    fn capabilities(&self) -> &CapabilitySet;
}

fn validate_shape_and_packing(shape: &[u64], payload: SourceSpan, packing: Packing) -> Result<()> {
    let elements = shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| Error::limit("quantized tensor element count overflows u64"))
    })?;
    let Packing::Blocks(block) = packing else {
        return Ok(());
    };

    let expected = match block.order() {
        PackingOrder::Flat => {
            block_bytes(elements, block.values_per_block(), block.bytes_per_block())?
        }
        PackingOrder::Axis(axis) => {
            validate_axis(axis, shape.len(), "quantized block")?;
            let axis_index = usize::try_from(axis).map_err(|error| {
                Error::with_source(
                    ErrorCategory::ResourceLimit,
                    "quantized block axis does not fit usize",
                    error,
                )
            })?;
            let axis_values = shape[axis_index];
            if let Some(lines) = elements.checked_div(axis_values) {
                let bytes_per_line = block_bytes(
                    axis_values,
                    block.values_per_block(),
                    block.bytes_per_block(),
                )?;
                lines.checked_mul(bytes_per_line).ok_or_else(|| {
                    Error::limit("quantized axis-blocked byte length overflows u64")
                })?
            } else {
                0
            }
        }
    };
    if expected != payload.len() {
        return Err(Error::invalid(
            "quantized payload span length does not match its shape and block packing",
        ));
    }
    Ok(())
}

fn block_bytes(values: u64, values_per_block: u32, bytes_per_block: u32) -> Result<u64> {
    if values == 0 {
        return Ok(0);
    }
    let block_values = u64::from(values_per_block);
    let rounded = values
        .checked_add(block_values - 1)
        .ok_or_else(|| Error::limit("quantized block count rounding overflows u64"))?;
    let blocks = rounded / block_values;
    blocks
        .checked_mul(u64::from(bytes_per_block))
        .ok_or_else(|| Error::limit("quantized packed byte length overflows u64"))
}

fn validate_axis(axis: u32, rank: usize, description: &str) -> Result<()> {
    let axis = usize::try_from(axis).map_err(|error| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "quantized axis does not fit usize",
            error,
        )
    })?;
    if axis >= rank {
        return Err(Error::invalid(format!(
            "{description} axis lies outside the logical tensor rank"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::ContentDigest;
    use crate::tensor::FileId;

    fn name(value: &str) -> Result<StableName> {
        StableName::parse(value)
    }

    fn implementation(operation: &str) -> Result<ImplementationId> {
        Ok(ImplementationId::new(
            name("test-provider")?,
            name(operation)?,
            1,
        ))
    }

    #[test]
    fn packed_f4_storage_has_no_scalar_dtype() -> Result<()> {
        let span = SourceSpan::new(FileId::from_ordinal(0), 16, 4)?;
        let storage = Storage::Quantized(QuantizedStorage::new(
            implementation("packed-f4")?,
            [8_u64],
            span,
            Packing::flat_blocks(2, 1)?,
        )?);

        assert_eq!(storage.logical_dtype(), None);
        Ok(())
    }

    #[test]
    fn axis_block_packing_accounts_for_per_row_padding() -> Result<()> {
        let span = SourceSpan::new(FileId::from_ordinal(0), 0, 8)?;
        let packing = Packing::Blocks(BlockPacking::new(4, 2, PackingOrder::Axis(1))?);

        let storage =
            QuantizedStorage::new(implementation("row-blocked")?, [2_u64, 5], span, packing)?;

        assert_eq!(storage.span().len(), 8);
        Ok(())
    }

    #[test]
    fn required_companion_must_be_present() -> Result<()> {
        let span = SourceSpan::new(FileId::from_ordinal(0), 0, 1)?;
        let result = QuantizedStorage::builder(
            implementation("missing-scale")?,
            [2_u64],
            span,
            Packing::flat_blocks(2, 1)?,
        )
        .required_companions([name("scale")?])
        .build();

        let error = result.err().ok_or_else(|| {
            Error::binding("quantized descriptor unexpectedly accepted a missing companion")
        })?;
        assert_eq!(
            error.message(),
            "quantized descriptor is missing a required companion role"
        );
        Ok(())
    }

    #[test]
    fn route_allows_provider_defined_f32_dequant_target() -> Result<()> {
        let capability = RouteCapability::new(
            implementation("q4")?,
            QuantizedRoute::HostDequant {
                target_dtype: DType::F32,
            },
            implementation("host-dequant")?,
            None,
            None,
        )?;

        assert!(matches!(
            capability.route(),
            QuantizedRoute::HostDequant {
                target_dtype: DType::F32
            }
        ));
        Ok(())
    }

    #[test]
    fn capability_set_serialization_is_order_independent() -> Result<()> {
        let backend = BackendId::from_digest(ContentDigest::hash("backend", [b"test"]));
        let first = RouteCapability::new(
            implementation("q4")?,
            QuantizedRoute::PackedDirect,
            implementation("z-kernel")?,
            Some(backend),
            None,
        )?;
        let second = RouteCapability::new(
            implementation("q4")?,
            QuantizedRoute::FusedInTile,
            implementation("a-kernel")?,
            Some(backend),
            None,
        )?;
        let forward = serde_json::to_vec(&CapabilitySet::new(vec![first.clone(), second.clone()])?)
            .map_err(|error| {
                Error::with_source(crate::ErrorCategory::InvalidFormat, "serialize", error)
            })?;
        let reverse =
            serde_json::to_vec(&CapabilitySet::new(vec![second, first])?).map_err(|error| {
                Error::with_source(crate::ErrorCategory::InvalidFormat, "serialize", error)
            })?;

        assert_eq!(forward, reverse);
        Ok(())
    }
}
