//! Typed, validated tensor-assembly operation graphs.
//!
//! An [`OperationGraph`] names its external tensors in caller-defined order
//! and records nodes in topological execution order. Every node pins an exact
//! [`ImplementationId`] and uses a typed [`Operation`]. The builder infers all
//! output [`TensorFacts`] before a graph can be constructed, so malformed
//! shapes, strides, representations, byte lengths, and edges fail before
//! tensor storage is allocated.
//!
//! Graph serialization preserves input, node, edge, and split-output order.
//! Deserialization replays all inference and validation instead of trusting
//! serialized derived state.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::identity::{ImplementationId, StableName};
use crate::plan::{PlannedTransform, TensorName};
use crate::prepare::{Layout, PreparationEngine, PrepareRequest, Representation};
use crate::telemetry::{ExecutionEvent, ExecutionObserver, OperationKind, OperationLocation};
use crate::tensor::{ByteView, DType};
use crate::{CancellationToken, Error, ErrorCategory, Result};

/// The canonical operation-graph schema version.
pub const OPERATION_GRAPH_SCHEMA_VERSION: u32 = 1;

const COPY_TILE_ELEMENTS: u64 = 16 * 1024;
const COPY_TILE_ELEMENTS_USIZE: usize = 16 * 1024;
const ALLOCATION_TILE_BYTES: usize = 1024 * 1024;

/// Validated logical and physical facts for one dense tensor value.
///
/// `logical_strides` describes the consumer-visible logical view.
/// `storage_strides` maps those same logical axes into physical storage. The
/// logical and storage shapes must contain the same number of elements, and
/// both stride sets must describe dense, non-overlapping mappings. This
/// permits logical OIHW strides to remain contiguous while physical storage is
/// OHWI.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "TensorFactsWire")]
pub struct TensorFacts {
    logical_shape: Box<[u64]>,
    storage_shape: Box<[u64]>,
    logical_strides: Box<[u64]>,
    storage_strides: Box<[u64]>,
    representation: Representation,
    byte_len: u64,
}

impl TensorFacts {
    /// Creates checked facts for a dense tensor.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error when ranks, element counts, strides, or
    /// the byte length disagree. Returns a resource-limit error when element,
    /// stride, or byte arithmetic overflows.
    pub fn new(
        logical_shape: impl Into<Box<[u64]>>,
        storage_shape: impl Into<Box<[u64]>>,
        logical_strides: impl Into<Box<[u64]>>,
        storage_strides: impl Into<Box<[u64]>>,
        representation: Representation,
        byte_len: u64,
    ) -> Result<Self> {
        let logical_shape = logical_shape.into();
        let storage_shape = storage_shape.into();
        let logical_strides = logical_strides.into();
        let storage_strides = storage_strides.into();
        validate_rank(logical_shape.len())?;
        validate_rank(storage_shape.len())?;
        if logical_shape.len() != storage_shape.len() {
            return Err(Error::invalid(
                "logical and storage tensor ranks must be equal",
            ));
        }
        if logical_shape.len() != logical_strides.len() {
            return Err(Error::invalid(
                "logical shape and element-stride ranks must be equal",
            ));
        }
        if logical_shape.len() != storage_strides.len() {
            return Err(Error::invalid(
                "logical shape and storage-stride ranks must be equal",
            ));
        }

        let logical_elements = checked_element_count(&logical_shape)?;
        let storage_elements = checked_element_count(&storage_shape)?;
        if logical_elements != storage_elements {
            return Err(Error::invalid(
                "logical and storage shapes must contain the same element count",
            ));
        }
        validate_dense_strides(&logical_shape, &logical_strides, logical_elements)?;
        validate_dense_strides(&logical_shape, &storage_strides, logical_elements)?;
        validate_storage_shape(
            &logical_shape,
            &storage_shape,
            &storage_strides,
            logical_elements,
        )?;
        if representation.layout().is_contiguous() {
            let expected_storage_strides = contiguous_strides(&logical_shape)?;
            if storage_shape != logical_shape || storage_strides != expected_storage_strides {
                return Err(Error::invalid(
                    "contiguous representation requires logical-order physical storage",
                ));
            }
        }

        let expected_bytes = representation.dtype().byte_len(&storage_shape)?;
        if byte_len != expected_bytes {
            return Err(Error::invalid(format!(
                "tensor facts record {byte_len} bytes, but {:?} storage requires {expected_bytes}",
                representation.dtype()
            )));
        }

        Ok(Self {
            logical_shape,
            storage_shape,
            logical_strides,
            storage_strides,
            representation,
            byte_len,
        })
    }

    /// Creates a contiguous row-major tensor with one logical and storage shape.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error when `representation` is not contiguous.
    /// Returns a resource-limit error when rank, stride, element, or byte
    /// arithmetic overflows.
    pub fn contiguous(
        shape: impl Into<Box<[u64]>>,
        representation: Representation,
    ) -> Result<Self> {
        if !representation.layout().is_contiguous() {
            return Err(Error::invalid(
                "contiguous tensor facts require a contiguous representation",
            ));
        }
        let shape = shape.into();
        let strides = contiguous_strides(&shape)?;
        let byte_len = representation.dtype().byte_len(&shape)?;
        Self::new(
            shape.clone(),
            shape,
            strides.clone(),
            strides,
            representation,
            byte_len,
        )
    }

    /// Returns the logical axis dimensions.
    #[must_use]
    pub const fn logical_shape(&self) -> &[u64] {
        &self.logical_shape
    }

    /// Returns the physical storage axis dimensions.
    #[must_use]
    pub const fn storage_shape(&self) -> &[u64] {
        &self.storage_shape
    }

    /// Returns consumer-visible element strides indexed by logical axis.
    #[must_use]
    pub const fn logical_strides(&self) -> &[u64] {
        &self.logical_strides
    }

    /// Returns physical element strides indexed by logical axis.
    #[must_use]
    pub const fn storage_strides(&self) -> &[u64] {
        &self.storage_strides
    }

    /// Returns the scalar dtype and physical layout descriptor.
    #[must_use]
    pub const fn representation(&self) -> &Representation {
        &self.representation
    }

    /// Returns the exact physical byte length.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Returns the validated logical element count.
    #[must_use]
    pub fn element_count(&self) -> u64 {
        self.logical_shape
            .iter()
            .copied()
            .fold(1_u64, u64::saturating_mul)
    }

    fn is_canonical_contiguous(&self) -> bool {
        self.representation.layout().is_contiguous()
            && self.logical_shape == self.storage_shape
            && contiguous_strides(&self.logical_shape).is_ok_and(|strides| {
                strides == self.logical_strides && strides == self.storage_strides
            })
    }
}

#[derive(Debug, Deserialize)]
struct TensorFactsWire {
    logical_shape: Box<[u64]>,
    storage_shape: Box<[u64]>,
    logical_strides: Box<[u64]>,
    storage_strides: Box<[u64]>,
    representation: Representation,
    byte_len: u64,
}

impl TryFrom<TensorFactsWire> for TensorFacts {
    type Error = Error;

    fn try_from(wire: TensorFactsWire) -> Result<Self> {
        Self::new(
            wire.logical_shape,
            wire.storage_shape,
            wire.logical_strides,
            wire.storage_strides,
            wire.representation,
            wire.byte_len,
        )
    }
}

/// Identifies one tensor axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Axis(u32);

impl Axis {
    /// Creates an axis from its zero-based index.
    #[must_use]
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    /// Returns the zero-based axis index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }

    fn as_usize(self) -> Result<usize> {
        usize::try_from(self.0).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "tensor axis does not fit usize",
                source,
            )
        })
    }
}

impl TryFrom<usize> for Axis {
    type Error = Error;

    fn try_from(index: usize) -> Result<Self> {
        let index = u32::try_from(index).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "tensor axis does not fit u32",
                source,
            )
        })?;
        Ok(Self(index))
    }
}

/// A half-open range along one logical tensor axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AxisRange {
    start: u64,
    end: u64,
}

impl AxisRange {
    /// Creates a half-open range.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error when `start` is greater than `end`.
    pub fn new(start: u64, end: u64) -> Result<Self> {
        if start > end {
            return Err(Error::invalid(
                "tensor axis range start must not exceed its end",
            ));
        }
        Ok(Self { start, end })
    }

    /// Returns the inclusive start index.
    #[must_use]
    pub const fn start(self) -> u64 {
        self.start
    }

    /// Returns the exclusive end index.
    #[must_use]
    pub const fn end(self) -> u64 {
        self.end
    }

    /// Returns the number of selected indices.
    #[must_use]
    pub const fn len(self) -> u64 {
        self.end - self.start
    }

    /// Returns whether this range selects no indices.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Concatenates two or more tensors along one logical axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Concat {
    axis: Axis,
}

impl Concat {
    /// Creates a concatenation operation.
    #[must_use]
    pub const fn new(axis: Axis) -> Self {
        Self { axis }
    }

    /// Returns the concatenation axis.
    #[must_use]
    pub const fn axis(self) -> Axis {
        self.axis
    }
}

/// Selects whether a permutation changes logical or physical axis order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PermuteMode {
    /// Permute logical axes and produce contiguous row-major storage.
    Logical,
    /// Keep logical axes and permute physical storage into `target_layout`.
    Storage {
        /// Descriptor for the resulting physical layout.
        target_layout: Layout,
    },
}

/// Applies a rank-complete axis permutation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Permute {
    order: Box<[u32]>,
    mode: PermuteMode,
}

impl Permute {
    /// Creates a logical-axis permutation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error unless `order` contains every axis from
    /// zero through `order.len() - 1` exactly once.
    pub fn logical(order: impl Into<Box<[u32]>>) -> Result<Self> {
        Self::new(order.into(), PermuteMode::Logical)
    }

    /// Creates a physical-storage permutation.
    ///
    /// Logical shape remains unchanged. For example, order `[0, 2, 3, 1]`
    /// changes OIHW physical storage to OHWI while retaining logical OIHW.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error unless `order` is a complete
    /// permutation.
    pub fn storage(order: impl Into<Box<[u32]>>, target_layout: Layout) -> Result<Self> {
        Self::new(order.into(), PermuteMode::Storage { target_layout })
    }

    fn new(order: Box<[u32]>, mode: PermuteMode) -> Result<Self> {
        validate_permutation(&order)?;
        Ok(Self { order, mode })
    }

    /// Returns physical output axes in source logical-axis order.
    #[must_use]
    pub const fn order(&self) -> &[u32] {
        &self.order
    }

    /// Returns whether logical or physical axes are permuted.
    #[must_use]
    pub const fn mode(&self) -> &PermuteMode {
        &self.mode
    }
}

/// Slices every logical axis into one materialized output.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Slice {
    ranges: Box<[AxisRange]>,
}

impl Slice {
    /// Creates a slice with one half-open range per logical axis.
    #[must_use]
    pub fn new(ranges: impl Into<Box<[AxisRange]>>) -> Self {
        Self {
            ranges: ranges.into(),
        }
    }

    /// Returns ranges in logical-axis order.
    #[must_use]
    pub const fn ranges(&self) -> &[AxisRange] {
        &self.ranges
    }
}

/// Splits one logical axis into ordered, non-overlapping outputs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Split {
    axis: Axis,
    ranges: Box<[AxisRange]>,
}

impl Split {
    /// Creates an ordered split.
    ///
    /// Gaps are permitted, but ranges must be non-empty, ordered by start, and
    /// non-overlapping. Bounds are checked when the operation is added to a
    /// graph with input facts.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error for no ranges, an empty range, or
    /// unordered or overlapping ranges.
    pub fn new(axis: Axis, ranges: impl Into<Box<[AxisRange]>>) -> Result<Self> {
        let ranges = ranges.into();
        if ranges.is_empty() {
            return Err(Error::invalid(
                "split must contain at least one output range",
            ));
        }
        let mut previous_end = None;
        for range in &ranges {
            if range.is_empty() {
                return Err(Error::invalid("split ranges must not be empty"));
            }
            if previous_end.is_some_and(|end| range.start() < end) {
                return Err(Error::invalid(
                    "split ranges must be ordered and non-overlapping",
                ));
            }
            previous_end = Some(range.end());
        }
        Ok(Self { axis, ranges })
    }

    /// Returns the split axis.
    #[must_use]
    pub const fn axis(&self) -> Axis {
        self.axis
    }

    /// Returns the ordered output ranges.
    #[must_use]
    pub const fn ranges(&self) -> &[AxisRange] {
        &self.ranges
    }
}

/// Reinterprets contiguous storage under a new logical shape.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Reshape {
    shape: Box<[u64]>,
}

impl Reshape {
    /// Creates a metadata-only reshape operation.
    #[must_use]
    pub fn new(shape: impl Into<Box<[u64]>>) -> Self {
        Self {
            shape: shape.into(),
        }
    }

    /// Returns the requested output shape.
    #[must_use]
    pub const fn shape(&self) -> &[u64] {
        &self.shape
    }
}

/// A typed built-in graph operation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "parameters", rename_all = "snake_case")]
pub enum Operation {
    /// N-input logical concatenation.
    Concat(Concat),
    /// Logical or physical axis permutation.
    Permute(Permute),
    /// Per-axis materialized slicing.
    Slice(Slice),
    /// Ordered multi-output axis splitting.
    Split(Split),
    /// Metadata-only contiguous reshape.
    Reshape(Reshape),
    /// One versioned provider preparation, such as a dtype cast.
    Prepare(PlannedTransform),
}

impl Operation {
    /// Returns the exact implementation identity used by the built-in host executor.
    ///
    /// [`Operation::Prepare`] returns the provider identity pinned by its
    /// [`PlannedTransform`]. Structural operations use the `model-weights`
    /// provider and schema-specific operation version 1.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if a built-in stable identifier cannot
    /// be constructed.
    pub fn builtin_implementation(&self) -> Result<ImplementationId> {
        if let Self::Prepare(transform) = self {
            return Ok(transform.transform().implementation().clone());
        }
        let operation = match self {
            Self::Concat(_) => "concat",
            Self::Permute(_) => "permute",
            Self::Slice(_) => "slice",
            Self::Split(_) => "split",
            Self::Reshape(_) => "reshape",
            Self::Prepare(_) => unreachable!("prepare returned above"),
        };
        Ok(ImplementationId::new(
            StableName::parse("model-weights")?,
            StableName::parse(operation)?,
            1,
        ))
    }
}

/// Identifies an external input by its graph-local ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InputId(u32);

impl InputId {
    /// Creates an input identifier from a graph-local ordinal.
    #[must_use]
    pub const fn from_ordinal(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Returns the graph-local ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.0
    }
}

/// Identifies a node by its topological ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(u32);

impl NodeId {
    /// Creates a node identifier from a graph-local ordinal.
    #[must_use]
    pub const fn from_ordinal(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Returns the graph-local ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.0
    }
}

/// Identifies one output of a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OutputIndex(u32);

impl OutputIndex {
    /// Creates an output index from its node-local ordinal.
    #[must_use]
    pub const fn from_ordinal(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Returns the node-local ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.0
    }
}

/// References an external tensor or one output from an earlier node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ValueRef {
    /// A caller-supplied graph input.
    Input {
        /// Graph-local ordered input identifier.
        input: InputId,
    },
    /// One output of an earlier graph node.
    NodeOutput {
        /// Topological node identifier.
        node: NodeId,
        /// Node-local output identifier.
        output: OutputIndex,
    },
}

impl ValueRef {
    /// Creates a reference to an external graph input.
    #[must_use]
    pub const fn input(input: InputId) -> Self {
        Self::Input { input }
    }

    /// Creates a reference to one graph-node output.
    #[must_use]
    pub const fn node_output(node: NodeId, output: OutputIndex) -> Self {
        Self::NodeOutput { node, output }
    }
}

/// One ordered, named external graph input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationInput {
    name: TensorName,
    facts: TensorFacts,
}

impl OperationInput {
    /// Returns the exact external tensor name.
    #[must_use]
    pub const fn name(&self) -> &TensorName {
        &self.name
    }

    /// Returns the validated external tensor facts.
    #[must_use]
    pub const fn facts(&self) -> &TensorFacts {
        &self.facts
    }
}

/// One validated graph node in topological execution order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationNode {
    implementation: ImplementationId,
    inputs: Box<[ValueRef]>,
    operation: Operation,
    #[serde(skip)]
    outputs: Box<[TensorFacts]>,
}

impl OperationNode {
    /// Returns the exact byte-affecting implementation identity.
    #[must_use]
    pub const fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    /// Returns input edges in operation-defined order.
    #[must_use]
    pub const fn inputs(&self) -> &[ValueRef] {
        &self.inputs
    }

    /// Returns the typed operation parameters.
    #[must_use]
    pub const fn operation(&self) -> &Operation {
        &self.operation
    }

    /// Returns all inferred node outputs in stable order.
    #[must_use]
    pub const fn outputs(&self) -> &[TensorFacts] {
        &self.outputs
    }
}

#[derive(Debug, Deserialize)]
struct OperationNodeWire {
    implementation: ImplementationId,
    inputs: Box<[ValueRef]>,
    operation: Operation,
}

/// Incrementally constructs one validated operation graph.
#[derive(Debug, Clone, Default)]
pub struct OperationGraphBuilder {
    inputs: Vec<OperationInput>,
    input_names: BTreeSet<TensorName>,
    nodes: Vec<OperationNode>,
}

impl OperationGraphBuilder {
    /// Creates an empty operation-graph builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inputs: Vec::new(),
            input_names: BTreeSet::new(),
            nodes: Vec::new(),
        }
    }

    /// Appends one named external input and returns its graph value.
    ///
    /// Inputs retain insertion order; names are not sorted or normalized.
    ///
    /// # Errors
    ///
    /// Returns a binding error for a duplicate name or a resource-limit error
    /// when the input count does not fit the serialized identifier.
    pub fn add_input(&mut self, name: TensorName, facts: TensorFacts) -> Result<ValueRef> {
        if self.input_names.contains(&name) {
            return Err(Error::binding(
                "operation graph contains a duplicate external input name",
            ));
        }
        let ordinal = u32::try_from(self.inputs.len()).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "operation graph input count exceeds u32",
                source,
            )
        })?;
        self.input_names.insert(name.clone());
        self.inputs.push(OperationInput { name, facts });
        Ok(ValueRef::input(InputId::from_ordinal(ordinal)))
    }

    /// Appends one topological operation and returns all inferred outputs.
    ///
    /// # Errors
    ///
    /// Returns a binding or invalid-format error for missing or forward edges,
    /// invalid operation parameters, incompatible facts, or an unsupported
    /// shape/layout combination. Returns a resource-limit error when checked
    /// shape, stride, byte, or identifier arithmetic overflows.
    pub fn add_operation(
        &mut self,
        implementation: ImplementationId,
        operation: Operation,
        inputs: impl Into<Box<[ValueRef]>>,
    ) -> Result<Box<[ValueRef]>> {
        let inputs = inputs.into();
        let input_facts = inputs
            .iter()
            .map(|value| {
                self.facts(*value)
                    .ok_or_else(|| Error::binding("operation references an unavailable value"))
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = infer_outputs(&operation, &input_facts)?;
        let node_ordinal = u32::try_from(self.nodes.len()).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "operation graph node count exceeds u32",
                source,
            )
        })?;
        let output_count = u32::try_from(outputs.len()).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "operation node output count exceeds u32",
                source,
            )
        })?;
        let node = NodeId::from_ordinal(node_ordinal);
        self.nodes.push(OperationNode {
            implementation,
            inputs,
            operation,
            outputs,
        });
        Ok((0..output_count)
            .map(|output| ValueRef::node_output(node, OutputIndex::from_ordinal(output)))
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Finalizes the graph with one selected output.
    ///
    /// # Errors
    ///
    /// Returns a binding error when `output` is unavailable.
    pub fn build(self, output: ValueRef) -> Result<OperationGraph> {
        if self.facts(output).is_none() {
            return Err(Error::binding(
                "operation graph final output is unavailable",
            ));
        }
        Ok(OperationGraph {
            schema_version: OPERATION_GRAPH_SCHEMA_VERSION,
            inputs: self.inputs.into_boxed_slice(),
            nodes: self.nodes.into_boxed_slice(),
            output,
        })
    }

    fn facts(&self, value: ValueRef) -> Option<&TensorFacts> {
        match value {
            ValueRef::Input { input } => usize::try_from(input.ordinal())
                .ok()
                .and_then(|index| self.inputs.get(index))
                .map(OperationInput::facts),
            ValueRef::NodeOutput { node, output } => usize::try_from(node.ordinal())
                .ok()
                .and_then(|index| self.nodes.get(index))
                .and_then(|node| {
                    usize::try_from(output.ordinal())
                        .ok()
                        .and_then(|index| node.outputs.get(index))
                }),
        }
    }
}

/// A versioned, fully inferred tensor operation graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "OperationGraphWire")]
pub struct OperationGraph {
    schema_version: u32,
    inputs: Box<[OperationInput]>,
    nodes: Box<[OperationNode]>,
    output: ValueRef,
}

impl OperationGraph {
    /// Returns a new empty graph builder.
    #[must_use]
    pub const fn builder() -> OperationGraphBuilder {
        OperationGraphBuilder::new()
    }

    /// Returns the operation-graph schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns named external inputs in caller-defined order.
    #[must_use]
    pub const fn inputs(&self) -> &[OperationInput] {
        &self.inputs
    }

    /// Returns nodes in topological execution order.
    #[must_use]
    pub const fn nodes(&self) -> &[OperationNode] {
        &self.nodes
    }

    /// Returns the selected final graph value.
    #[must_use]
    pub const fn output(&self) -> ValueRef {
        self.output
    }

    /// Looks up inferred facts for one graph value.
    #[must_use]
    pub fn facts(&self, value: ValueRef) -> Option<&TensorFacts> {
        match value {
            ValueRef::Input { input } => usize::try_from(input.ordinal())
                .ok()
                .and_then(|index| self.inputs.get(index))
                .map(OperationInput::facts),
            ValueRef::NodeOutput { node, output } => usize::try_from(node.ordinal())
                .ok()
                .and_then(|index| self.nodes.get(index))
                .and_then(|node| {
                    usize::try_from(output.ordinal())
                        .ok()
                        .and_then(|index| node.outputs.get(index))
                }),
        }
    }

    /// Returns inferred facts for the selected final output.
    #[must_use]
    pub fn output_facts(&self) -> &TensorFacts {
        // Construction and deserialization both validate this reference.
        self.facts(self.output)
            .unwrap_or_else(|| unreachable!("validated graph output must exist"))
    }

    /// Returns the external input ordinal aliased by the selected output.
    ///
    /// A direct input and any chain containing only metadata-only reshapes or
    /// fact-preserving identity permutations retain the original immutable
    /// byte view. Materializing operations and provider preparation return
    /// `None`.
    #[must_use]
    pub fn output_input_alias(&self) -> Option<usize> {
        let mut value = self.output;
        loop {
            match value {
                ValueRef::Input { input } => {
                    let index = usize::try_from(input.ordinal()).ok()?;
                    return (index < self.inputs.len()).then_some(index);
                }
                ValueRef::NodeOutput { node, output } => {
                    let index = usize::try_from(node.ordinal()).ok()?;
                    let node = self.nodes.get(index)?;
                    let output_index = usize::try_from(output.ordinal()).ok()?;
                    let input_index = self.node_output_alias_input(node, output_index)?;
                    value = *node.inputs().get(input_index)?;
                }
            }
        }
    }

    fn node_output_alias_input(&self, node: &OperationNode, output_index: usize) -> Option<usize> {
        if output_index != 0 {
            return None;
        }
        let [input] = node.inputs() else {
            return None;
        };
        let [output_facts] = node.outputs() else {
            return None;
        };
        match node.operation() {
            Operation::Reshape(_) => Some(0),
            Operation::Permute(permute)
                if is_identity_order(permute.order())
                    && self
                        .facts(*input)
                        .is_some_and(|facts| facts == output_facts) =>
            {
                Some(0)
            }
            Operation::Concat(_)
            | Operation::Permute(_)
            | Operation::Slice(_)
            | Operation::Split(_)
            | Operation::Prepare(_) => None,
        }
    }

    /// Computes a prevalidated upper bound for graph-owned host scratch bytes.
    ///
    /// The estimate follows node liveness and excludes the final returned
    /// allocation. It assumes every [`Operation::Prepare`] materializes its
    /// declared output; an identity provider may use less at execution time.
    /// External input mappings and the returned output are accounted for
    /// separately by the materialization pipeline.
    ///
    /// # Errors
    ///
    /// Returns a resource-limit error if liveness arithmetic overflows.
    pub fn estimate_host_scratch_bytes(&self) -> Result<u64> {
        let mut schedule = LivenessSchedule::new(self)?;
        for (node_index, node) in self.nodes.iter().enumerate() {
            let node_id = NodeId::from_ordinal(index_as_u32(
                node_index,
                "operation graph node index exceeds u32",
            )?);
            let mut produced = Vec::with_capacity(node.outputs().len());
            for (output_index, facts) in node.outputs().iter().enumerate() {
                let value = ValueRef::node_output(
                    node_id,
                    OutputIndex::from_ordinal(index_as_u32(
                        output_index,
                        "operation node output index exceeds u32",
                    )?),
                );
                let allocation =
                    if let Some(input_index) = self.node_output_alias_input(node, output_index) {
                        let input = node.inputs().get(input_index).ok_or_else(|| {
                            Error::integrity("aliased operation input is unavailable")
                        })?;
                        schedule
                            .values
                            .get(input)
                            .ok_or_else(|| Error::integrity("aliased operation input is not live"))?
                            .allocation
                    } else {
                        Some(schedule.create_allocation(facts.byte_len())?)
                    };
                schedule.insert_value(value, allocation)?;
                produced.push(value);
            }
            let transient = match node.operation() {
                Operation::Prepare(transform) => transform.scratch_bytes(),
                Operation::Concat(_)
                | Operation::Permute(_)
                | Operation::Slice(_)
                | Operation::Split(_)
                | Operation::Reshape(_) => 0,
            };
            schedule.observe_transient(transient)?;
            schedule.consume_inputs(node.inputs())?;
            schedule.remove_unused(&produced)?;
        }
        schedule.scratch_excluding_output(self.output)
    }

    /// Executes the graph using checked host allocations and provider prepares.
    ///
    /// `inputs` must follow [`Self::inputs`] order exactly. Structural kernels
    /// check cancellation at bounded element or byte intervals. Reshape is
    /// zero-copy; slices and split outputs are always materialized.
    ///
    /// # Errors
    ///
    /// Returns an error for input count or byte mismatches, a node whose
    /// implementation identity differs from
    /// [`Operation::builtin_implementation`], allocation or offset overflow,
    /// provider failure, or cooperative cancellation.
    pub fn execute_host(
        &self,
        inputs: &[ByteView],
        preparation: &PreparationEngine,
        cancellation: &CancellationToken,
    ) -> Result<HostExecution> {
        self.execute_host_inner(inputs, preparation, cancellation, None)
    }

    /// Executes the graph while emitting one event per completed host node.
    ///
    /// `work_ordinal` associates nodes with a bounded-pipeline item when one
    /// exists. An observer that disables operation events avoids node timers
    /// and receives no [`ExecutionEvent::OperationFinished`] values.
    ///
    /// # Errors
    ///
    /// Returns any error described by [`Self::execute_host`].
    pub fn execute_host_observed(
        &self,
        inputs: &[ByteView],
        preparation: &PreparationEngine,
        cancellation: &CancellationToken,
        work_ordinal: Option<u64>,
        observer: &dyn ExecutionObserver,
    ) -> Result<HostExecution> {
        let observation = observer
            .operation_events_enabled()
            .then_some(OperationObservation {
                work_ordinal,
                observer,
            });
        self.execute_host_inner(inputs, preparation, cancellation, observation)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "execution keeps byte production and liveness accounting in one transaction"
    )]
    fn execute_host_inner(
        &self,
        inputs: &[ByteView],
        preparation: &PreparationEngine,
        cancellation: &CancellationToken,
        observation: Option<OperationObservation<'_>>,
    ) -> Result<HostExecution> {
        cancellation.check()?;
        if inputs.len() != self.inputs.len() {
            return Err(Error::binding(format!(
                "operation graph requires {} ordered inputs, but received {}",
                self.inputs.len(),
                inputs.len()
            )));
        }
        let mut schedule = LivenessSchedule::new(self)?;
        let mut runtime = BTreeMap::<ValueRef, RuntimeValue>::new();
        for (input_index, (descriptor, bytes)) in self.inputs.iter().zip(inputs).enumerate() {
            validate_view_len(bytes, descriptor.facts().byte_len(), "operation input")?;
            let value = ValueRef::input(InputId::from_ordinal(index_as_u32(
                input_index,
                "operation input index exceeds u32",
            )?));
            runtime.insert(
                value,
                RuntimeValue {
                    bytes: bytes.clone(),
                    allocation: None,
                },
            );
        }

        for (node_index, node) in self.nodes.iter().enumerate() {
            cancellation.check()?;
            let node_id = NodeId::from_ordinal(index_as_u32(
                node_index,
                "operation graph node index exceeds u32",
            )?);
            let expected_implementation = node.operation().builtin_implementation()?;
            if node.implementation() != &expected_implementation {
                return Err(Error::unsupported(format!(
                    "host executor requires {} / {} version {}, but the node pins {} / {} version {}",
                    expected_implementation.provider(),
                    expected_implementation.operation(),
                    expected_implementation.version(),
                    node.implementation().provider(),
                    node.implementation().operation(),
                    node.implementation().version()
                )));
            }
            let input_values = node
                .inputs()
                .iter()
                .map(|value| {
                    runtime
                        .get(value)
                        .cloned()
                        .ok_or_else(|| Error::integrity("live operation input is unavailable"))
                })
                .collect::<Result<Vec<_>>>()?;
            let input_facts = node
                .inputs()
                .iter()
                .map(|value| {
                    self.facts(*value)
                        .ok_or_else(|| Error::integrity("operation input facts are unavailable"))
                })
                .collect::<Result<Vec<_>>>()?;
            let operation_started = observation.map(|_| Instant::now());
            let execution =
                execute_node(node, &input_values, &input_facts, preparation, cancellation)?;
            let operation_duration = operation_started.map(|started| started.elapsed());
            if execution.outputs.len() != node.outputs().len() {
                return Err(Error::integrity(
                    "host operation returned an unexpected output count",
                ));
            }
            for (produced, facts) in execution.outputs.iter().zip(node.outputs()) {
                validate_view_len(&produced.bytes, facts.byte_len(), "operation output")?;
            }
            if let (Some(observation), Some(duration)) = (observation, operation_duration) {
                let materialized_output_bytes = execution
                    .outputs
                    .iter()
                    .zip(node.outputs())
                    .filter(|(output, _facts)| matches!(output.origin, OutputOrigin::Fresh))
                    .fold(0_u64, |bytes, (_output, facts)| {
                        bytes.saturating_add(facts.byte_len())
                    });
                let kind = operation_kind(node.operation(), materialized_output_bytes);
                observation
                    .observer
                    .observe(&ExecutionEvent::OperationFinished {
                        work_ordinal: observation.work_ordinal,
                        location: OperationLocation::GraphNode(node_id),
                        kind,
                        duration,
                        input_bytes: input_facts
                            .iter()
                            .fold(0_u64, |bytes, facts| bytes.saturating_add(facts.byte_len())),
                        output_bytes: node
                            .outputs()
                            .iter()
                            .fold(0_u64, |bytes, facts| bytes.saturating_add(facts.byte_len())),
                        materialized_output_bytes,
                    });
            }

            let mut produced_values = Vec::with_capacity(execution.outputs.len());
            for (output_index, (produced, facts)) in execution
                .outputs
                .into_iter()
                .zip(node.outputs())
                .enumerate()
            {
                let allocation = match produced.origin {
                    OutputOrigin::Alias(input_index) => {
                        input_values
                            .get(input_index)
                            .ok_or_else(|| {
                                Error::integrity("operation output aliases a missing input")
                            })?
                            .allocation
                    }
                    OutputOrigin::Fresh => Some(schedule.create_allocation(facts.byte_len())?),
                };
                let value = ValueRef::node_output(
                    node_id,
                    OutputIndex::from_ordinal(index_as_u32(
                        output_index,
                        "operation node output index exceeds u32",
                    )?),
                );
                schedule.insert_value(value, allocation)?;
                runtime.insert(
                    value,
                    RuntimeValue {
                        bytes: produced.bytes,
                        allocation,
                    },
                );
                produced_values.push(value);
            }
            schedule.observe_transient(execution.transient_scratch)?;
            consume_runtime_inputs(node.inputs(), &mut schedule, &mut runtime)?;
            remove_unused_runtime(&produced_values, &mut schedule, &mut runtime)?;
        }
        cancellation.check()?;
        let output = runtime
            .get(&self.output)
            .ok_or_else(|| Error::integrity("operation graph final output is not live"))?;
        let peak_scratch_bytes = schedule.scratch_excluding_output(self.output)?;
        Ok(HostExecution {
            output: output.bytes.clone(),
            peak_scratch_bytes,
        })
    }
}

/// Result of one checked host operation-graph execution.
#[derive(Debug, Clone)]
pub struct HostExecution {
    output: ByteView,
    peak_scratch_bytes: u64,
}

impl HostExecution {
    /// Returns the immutable final output bytes.
    #[must_use]
    pub const fn output(&self) -> &ByteView {
        &self.output
    }

    /// Consumes the result and returns its immutable final output.
    #[must_use]
    pub fn into_output(self) -> ByteView {
        self.output
    }

    /// Returns exact peak graph-owned bytes beyond the returned allocation.
    ///
    /// This includes live materialized intermediates and provider-declared
    /// transient scratch, but excludes ordered external inputs.
    #[must_use]
    pub const fn peak_scratch_bytes(&self) -> u64 {
        self.peak_scratch_bytes
    }
}

#[derive(Debug, Deserialize)]
struct OperationGraphWire {
    schema_version: u32,
    inputs: Box<[OperationInput]>,
    nodes: Box<[OperationNodeWire]>,
    output: ValueRef,
}

impl TryFrom<OperationGraphWire> for OperationGraph {
    type Error = Error;

    fn try_from(wire: OperationGraphWire) -> Result<Self> {
        if wire.schema_version != OPERATION_GRAPH_SCHEMA_VERSION {
            return Err(Error::binding(format!(
                "unsupported operation graph schema version {}",
                wire.schema_version
            )));
        }
        let mut builder = OperationGraphBuilder::new();
        for input in wire.inputs {
            let _ = builder.add_input(input.name, input.facts)?;
        }
        for node in wire.nodes {
            let _ = builder.add_operation(node.implementation, node.operation, node.inputs)?;
        }
        builder.build(wire.output)
    }
}

#[derive(Debug, Clone)]
struct RuntimeValue {
    bytes: ByteView,
    allocation: Option<u64>,
}

#[derive(Clone, Copy)]
struct OperationObservation<'a> {
    work_ordinal: Option<u64>,
    observer: &'a dyn ExecutionObserver,
}

#[derive(Debug, Clone, Copy)]
struct LiveValue {
    allocation: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct LiveAllocation {
    bytes: u64,
    references: u64,
}

#[derive(Debug)]
struct LivenessSchedule {
    remaining_uses: BTreeMap<ValueRef, u64>,
    values: BTreeMap<ValueRef, LiveValue>,
    allocations: BTreeMap<u64, LiveAllocation>,
    next_allocation: u64,
    live_bytes: u64,
    peak_bytes: u64,
}

impl LivenessSchedule {
    fn new(graph: &OperationGraph) -> Result<Self> {
        let mut remaining_uses = BTreeMap::<ValueRef, u64>::new();
        for node in graph.nodes() {
            for &input in node.inputs() {
                increment_count(&mut remaining_uses, input, "operation value use count")?;
            }
        }
        increment_count(
            &mut remaining_uses,
            graph.output(),
            "operation graph output use count",
        )?;
        let mut values = BTreeMap::new();
        for input_index in 0..graph.inputs().len() {
            let input = InputId::from_ordinal(index_as_u32(
                input_index,
                "operation input index exceeds u32",
            )?);
            values.insert(ValueRef::input(input), LiveValue { allocation: None });
        }
        Ok(Self {
            remaining_uses,
            values,
            allocations: BTreeMap::new(),
            next_allocation: 0,
            live_bytes: 0,
            peak_bytes: 0,
        })
    }

    fn create_allocation(&mut self, bytes: u64) -> Result<u64> {
        let allocation = self.next_allocation;
        self.next_allocation = self
            .next_allocation
            .checked_add(1)
            .ok_or_else(|| Error::limit("operation allocation identifier overflows u64"))?;
        self.live_bytes = self
            .live_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::limit("live operation allocation bytes overflow u64"))?;
        self.peak_bytes = self.peak_bytes.max(self.live_bytes);
        self.allocations.insert(
            allocation,
            LiveAllocation {
                bytes,
                references: 0,
            },
        );
        Ok(allocation)
    }

    fn insert_value(&mut self, value: ValueRef, allocation: Option<u64>) -> Result<()> {
        if self.values.contains_key(&value) {
            return Err(Error::integrity(
                "operation schedule produced a duplicate value",
            ));
        }
        if let Some(allocation) = allocation {
            let state = self.allocations.get_mut(&allocation).ok_or_else(|| {
                Error::integrity("operation value references an unknown allocation")
            })?;
            state.references = state
                .references
                .checked_add(1)
                .ok_or_else(|| Error::limit("operation allocation references overflow u64"))?;
        }
        self.values.insert(value, LiveValue { allocation });
        Ok(())
    }

    fn observe_transient(&mut self, scratch_bytes: u64) -> Result<()> {
        let with_scratch = self
            .live_bytes
            .checked_add(scratch_bytes)
            .ok_or_else(|| Error::limit("operation live scratch bytes overflow u64"))?;
        self.peak_bytes = self.peak_bytes.max(with_scratch);
        Ok(())
    }

    fn consume_inputs(&mut self, inputs: &[ValueRef]) -> Result<()> {
        for &input in inputs {
            let _ = self.consume_value(input)?;
        }
        Ok(())
    }

    fn consume_value(&mut self, value: ValueRef) -> Result<bool> {
        let uses = self.remaining_uses.get_mut(&value).ok_or_else(|| {
            Error::integrity("operation schedule consumed a value without a recorded use")
        })?;
        *uses = uses
            .checked_sub(1)
            .ok_or_else(|| Error::integrity("operation value use count underflowed"))?;
        if *uses == 0 {
            self.remove_value(value)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn remove_unused(&mut self, values: &[ValueRef]) -> Result<()> {
        for &value in values {
            if self.remaining_uses.get(&value).copied().unwrap_or(0) == 0 {
                self.remove_value(value)?;
            }
        }
        Ok(())
    }

    fn remove_value(&mut self, value: ValueRef) -> Result<()> {
        let live = self
            .values
            .remove(&value)
            .ok_or_else(|| Error::integrity("operation schedule value is not live"))?;
        let Some(allocation) = live.allocation else {
            return Ok(());
        };
        let state = self
            .allocations
            .get_mut(&allocation)
            .ok_or_else(|| Error::integrity("operation schedule allocation is not live"))?;
        state.references = state
            .references
            .checked_sub(1)
            .ok_or_else(|| Error::integrity("operation allocation reference count underflowed"))?;
        if state.references == 0 {
            let bytes = state.bytes;
            self.allocations.remove(&allocation);
            self.live_bytes = self
                .live_bytes
                .checked_sub(bytes)
                .ok_or_else(|| Error::integrity("operation live byte count underflowed"))?;
        }
        Ok(())
    }

    fn scratch_excluding_output(&self, output: ValueRef) -> Result<u64> {
        let output_bytes = self
            .values
            .get(&output)
            .and_then(|value| value.allocation)
            .and_then(|allocation| self.allocations.get(&allocation))
            .map_or(0, |allocation| allocation.bytes);
        self.peak_bytes
            .checked_sub(output_bytes)
            .ok_or_else(|| Error::integrity("operation output allocation exceeds peak live bytes"))
    }
}

fn increment_count(
    counts: &mut BTreeMap<ValueRef, u64>,
    value: ValueRef,
    description: &'static str,
) -> Result<()> {
    let count = counts.entry(value).or_default();
    *count = count
        .checked_add(1)
        .ok_or_else(|| Error::limit(format!("{description} overflows u64")))?;
    Ok(())
}

fn consume_runtime_inputs(
    inputs: &[ValueRef],
    schedule: &mut LivenessSchedule,
    runtime: &mut BTreeMap<ValueRef, RuntimeValue>,
) -> Result<()> {
    for &input in inputs {
        if schedule.consume_value(input)? {
            runtime.remove(&input).ok_or_else(|| {
                Error::integrity("runtime operation input is unexpectedly unavailable")
            })?;
        }
    }
    Ok(())
}

fn remove_unused_runtime(
    values: &[ValueRef],
    schedule: &mut LivenessSchedule,
    runtime: &mut BTreeMap<ValueRef, RuntimeValue>,
) -> Result<()> {
    for &value in values {
        if schedule.remaining_uses.get(&value).copied().unwrap_or(0) == 0 {
            schedule.remove_value(value)?;
            runtime.remove(&value).ok_or_else(|| {
                Error::integrity("unused runtime operation output is unavailable")
            })?;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct NodeExecution {
    outputs: Vec<ProducedValue>,
    transient_scratch: u64,
}

#[derive(Debug)]
struct ProducedValue {
    bytes: ByteView,
    origin: OutputOrigin,
}

#[derive(Debug, Clone, Copy)]
enum OutputOrigin {
    Fresh,
    Alias(usize),
}

fn operation_kind(operation: &Operation, materialized_output_bytes: u64) -> OperationKind {
    match operation {
        Operation::Concat(_) => OperationKind::Concat,
        Operation::Permute(_) => OperationKind::Permute,
        Operation::Slice(_) => OperationKind::Slice,
        Operation::Split(_) => OperationKind::Split,
        Operation::Reshape(_) => OperationKind::Reshape,
        Operation::Prepare(_) if materialized_output_bytes == 0 => OperationKind::Identity,
        Operation::Prepare(transform) => OperationKind::for_transform(transform.transform()),
    }
}

fn execute_node(
    node: &OperationNode,
    inputs: &[RuntimeValue],
    input_facts: &[&TensorFacts],
    preparation: &PreparationEngine,
    cancellation: &CancellationToken,
) -> Result<NodeExecution> {
    if input_facts.len() != inputs.len() {
        return Err(Error::integrity(
            "runtime operation input count differs from graph edges",
        ));
    }
    match node.operation() {
        Operation::Concat(concat) => {
            execute_concat_node(*concat, node, inputs, input_facts, cancellation)
        }
        Operation::Permute(permute) => {
            execute_permute_node(permute, node, inputs, input_facts, cancellation)
        }
        Operation::Slice(slice) => {
            execute_slice_node(slice, node, inputs, input_facts, cancellation)
        }
        Operation::Split(split) => {
            execute_split_node(split, node, inputs, input_facts, cancellation)
        }
        Operation::Reshape(_) => execute_reshape_node(inputs),
        Operation::Prepare(transform) => {
            execute_prepare_node(transform, inputs, input_facts, preparation, cancellation)
        }
    }
}

fn execute_concat_node(
    concat: Concat,
    node: &OperationNode,
    inputs: &[RuntimeValue],
    input_facts: &[&TensorFacts],
    cancellation: &CancellationToken,
) -> Result<NodeExecution> {
    let output_facts = only_output(node)?;
    let input_views = inputs.iter().map(|input| &input.bytes).collect::<Vec<_>>();
    if input_facts
        .iter()
        .all(|facts| facts.is_canonical_contiguous())
        && output_facts.is_canonical_contiguous()
    {
        let output = build_contiguous_concat(
            concat.axis(),
            input_facts,
            output_facts,
            &input_views,
            cancellation,
        )?;
        return Ok(fresh_node_execution(output));
    }
    let mut output = allocate_tensor(output_facts, cancellation)?;
    execute_concat_bytes(
        concat,
        input_facts,
        output_facts,
        &input_views,
        &mut output,
        cancellation,
    )?;
    Ok(fresh_node_execution(output))
}

fn execute_permute_node(
    permute: &Permute,
    node: &OperationNode,
    inputs: &[RuntimeValue],
    input_facts: &[&TensorFacts],
    cancellation: &CancellationToken,
) -> Result<NodeExecution> {
    let (input, input_facts, output_facts) = unary_execution_parts(node, inputs, input_facts)?;
    validate_view_len(&input.bytes, input_facts.byte_len(), "permutation input")?;
    if is_identity_permutation(permute, input_facts, output_facts) {
        cancellation.check()?;
        return Ok(NodeExecution {
            outputs: vec![ProducedValue {
                bytes: input.bytes.clone(),
                origin: OutputOrigin::Alias(0),
            }],
            transient_scratch: 0,
        });
    }
    if let Some(output) = build_oihw_to_ohwi_if_supported(
        permute,
        input_facts,
        output_facts,
        input.bytes.as_slice(),
        cancellation,
    )? {
        return Ok(fresh_node_execution(output));
    }
    let mut output = allocate_tensor(output_facts, cancellation)?;
    execute_permute_bytes(
        permute,
        input_facts,
        output_facts,
        &input.bytes,
        &mut output,
        cancellation,
    )?;
    Ok(fresh_node_execution(output))
}

fn execute_slice_node(
    slice: &Slice,
    node: &OperationNode,
    inputs: &[RuntimeValue],
    input_facts: &[&TensorFacts],
    cancellation: &CancellationToken,
) -> Result<NodeExecution> {
    let (input, input_facts, output_facts) = unary_execution_parts(node, inputs, input_facts)?;
    let mut output = allocate_tensor(output_facts, cancellation)?;
    execute_slice_bytes(
        slice.ranges(),
        input_facts,
        &input.bytes,
        output_facts,
        &mut output,
        cancellation,
    )?;
    Ok(fresh_node_execution(output))
}

fn execute_split_node(
    split: &Split,
    node: &OperationNode,
    inputs: &[RuntimeValue],
    input_facts: &[&TensorFacts],
    cancellation: &CancellationToken,
) -> Result<NodeExecution> {
    let input = inputs
        .first()
        .ok_or_else(|| Error::integrity("split runtime input is unavailable"))?;
    let facts = input_facts
        .first()
        .copied()
        .ok_or_else(|| Error::integrity("split input facts are unavailable"))?;
    let mut outputs = Vec::with_capacity(split.ranges().len());
    for (&range, output_facts) in split.ranges().iter().zip(node.outputs()) {
        cancellation.check()?;
        let mut ranges = facts
            .logical_shape()
            .iter()
            .copied()
            .map(|dimension| AxisRange {
                start: 0,
                end: dimension,
            })
            .collect::<Vec<_>>();
        ranges[split.axis().as_usize()?] = range;
        let mut output = allocate_tensor(output_facts, cancellation)?;
        execute_slice_bytes(
            &ranges,
            facts,
            &input.bytes,
            output_facts,
            &mut output,
            cancellation,
        )?;
        outputs.push(ProducedValue {
            bytes: ByteView::from_boxed(output),
            origin: OutputOrigin::Fresh,
        });
    }
    Ok(NodeExecution {
        outputs,
        transient_scratch: 0,
    })
}

fn execute_reshape_node(inputs: &[RuntimeValue]) -> Result<NodeExecution> {
    let input = inputs
        .first()
        .ok_or_else(|| Error::integrity("reshape runtime input is unavailable"))?;
    Ok(NodeExecution {
        outputs: vec![ProducedValue {
            bytes: input.bytes.clone(),
            origin: OutputOrigin::Alias(0),
        }],
        transient_scratch: 0,
    })
}

fn execute_prepare_node(
    transform: &PlannedTransform,
    inputs: &[RuntimeValue],
    input_facts: &[&TensorFacts],
    preparation: &PreparationEngine,
    cancellation: &CancellationToken,
) -> Result<NodeExecution> {
    let input = inputs
        .first()
        .ok_or_else(|| Error::integrity("prepare runtime input is unavailable"))?;
    let facts = input_facts
        .first()
        .copied()
        .ok_or_else(|| Error::integrity("prepare input facts are unavailable"))?;
    let request = PrepareRequest::new(
        transform.transform(),
        facts.logical_shape(),
        &input.bytes,
        transform.output_size(),
    )
    .with_expected_scratch_bytes(transform.scratch_bytes());
    let output = preparation.prepare_with_cancellation(&request, cancellation)?;
    let reused = same_view(&output, &input.bytes);
    Ok(NodeExecution {
        outputs: vec![ProducedValue {
            bytes: output,
            origin: if reused {
                OutputOrigin::Alias(0)
            } else {
                OutputOrigin::Fresh
            },
        }],
        transient_scratch: if reused { 0 } else { transform.scratch_bytes() },
    })
}

fn unary_execution_parts<'a>(
    node: &'a OperationNode,
    inputs: &'a [RuntimeValue],
    input_facts: &'a [&TensorFacts],
) -> Result<(&'a RuntimeValue, &'a TensorFacts, &'a TensorFacts)> {
    let [input] = inputs else {
        return Err(Error::integrity(
            "unary runtime operation requires exactly one input",
        ));
    };
    let [facts] = input_facts else {
        return Err(Error::integrity(
            "unary runtime operation requires exactly one input-facts value",
        ));
    };
    Ok((input, *facts, only_output(node)?))
}

fn only_output(node: &OperationNode) -> Result<&TensorFacts> {
    let [output] = node.outputs() else {
        return Err(Error::integrity(
            "unary runtime operation requires exactly one output",
        ));
    };
    Ok(output)
}

fn fresh_node_execution(output: Box<[u8]>) -> NodeExecution {
    NodeExecution {
        outputs: vec![ProducedValue {
            bytes: ByteView::from_boxed(output),
            origin: OutputOrigin::Fresh,
        }],
        transient_scratch: 0,
    }
}

fn allocate_tensor(facts: &TensorFacts, cancellation: &CancellationToken) -> Result<Box<[u8]>> {
    cancellation.check()?;
    let length = validate_host_len(facts.byte_len(), "operation output byte length")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "operation output allocation failed",
            source,
        )
    })?;
    while bytes.len() < length {
        cancellation.check()?;
        let next = bytes
            .len()
            .saturating_add(ALLOCATION_TILE_BYTES)
            .min(length);
        bytes.resize(next, 0);
    }
    cancellation.check()?;
    Ok(bytes.into_boxed_slice())
}

fn execute_concat_bytes(
    concat: Concat,
    input_facts: &[&TensorFacts],
    output_facts: &TensorFacts,
    inputs: &[&ByteView],
    output: &mut [u8],
    cancellation: &CancellationToken,
) -> Result<()> {
    if input_facts.len() != inputs.len() {
        return Err(Error::integrity(
            "concatenation facts and byte-view counts differ",
        ));
    }
    for (&facts, bytes) in input_facts.iter().zip(inputs) {
        validate_view_len(bytes, facts.byte_len(), "concatenation input")?;
    }
    let axis = validate_axis(concat.axis(), output_facts.logical_shape().len())?;
    let dtype = output_facts.representation().dtype();
    let width = dtype_width(dtype)?;
    let logical_row_strides = contiguous_strides(output_facts.logical_shape())?;
    let mut axis_starts = Vec::with_capacity(input_facts.len());
    let mut next_start = 0_u64;
    for facts in input_facts {
        axis_starts.push(next_start);
        next_start = next_start
            .checked_add(facts.logical_shape()[axis])
            .ok_or_else(|| Error::limit("concatenation input axis offsets overflow u64"))?;
    }

    for linear in 0..output_facts.element_count() {
        check_copy_cancellation(linear, cancellation)?;
        let axis_coordinate = logical_coordinate(
            linear,
            axis,
            output_facts.logical_shape(),
            &logical_row_strides,
        )?;
        let input_index = axis_starts
            .iter()
            .zip(input_facts)
            .position(|(&start, facts)| {
                axis_coordinate < start.saturating_add(facts.logical_shape()[axis])
            })
            .ok_or_else(|| Error::integrity("concatenation coordinate has no source input"))?;
        let input = input_facts[input_index];
        let mut source_element = 0_u64;
        let mut target_element = 0_u64;
        for logical_axis in 0..output_facts.logical_shape().len() {
            let mut coordinate = logical_coordinate(
                linear,
                logical_axis,
                output_facts.logical_shape(),
                &logical_row_strides,
            )?;
            if logical_axis == axis {
                coordinate = coordinate
                    .checked_sub(axis_starts[input_index])
                    .ok_or_else(|| {
                        Error::integrity("concatenation local coordinate underflowed")
                    })?;
            }
            source_element = checked_offset_term(
                source_element,
                coordinate,
                input.storage_strides()[logical_axis],
            )?;
            let output_coordinate = logical_coordinate(
                linear,
                logical_axis,
                output_facts.logical_shape(),
                &logical_row_strides,
            )?;
            target_element = checked_offset_term(
                target_element,
                output_coordinate,
                output_facts.storage_strides()[logical_axis],
            )?;
        }
        copy_element(
            inputs[input_index].as_slice(),
            source_element,
            output,
            target_element,
            width,
        )?;
    }
    cancellation.check()
}

fn build_contiguous_concat(
    axis: Axis,
    input_facts: &[&TensorFacts],
    output_facts: &TensorFacts,
    inputs: &[&ByteView],
    cancellation: &CancellationToken,
) -> Result<Box<[u8]>> {
    cancellation.check()?;
    if input_facts.len() != inputs.len() {
        return Err(Error::integrity(
            "concatenation facts and byte-view counts differ",
        ));
    }
    for (&facts, bytes) in input_facts.iter().zip(inputs) {
        validate_view_len(bytes, facts.byte_len(), "concatenation input")?;
    }
    let axis = validate_axis(axis, output_facts.logical_shape().len())?;
    let output_length =
        validate_host_len(output_facts.byte_len(), "concatenation output byte length")?;
    let mut output = Vec::new();
    output.try_reserve_exact(output_length).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "concatenation output allocation failed",
            source,
        )
    })?;
    let shape = input_facts[0].logical_shape();
    let outer = checked_element_count(&shape[..axis])?;
    let inner_elements = checked_element_count(&shape[axis + 1..])?;
    let width =
        u64::try_from(dtype_width(input_facts[0].representation().dtype())?).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "scalar byte width does not fit u64",
                source,
            )
        })?;
    let inner_bytes = inner_elements
        .checked_mul(width)
        .ok_or_else(|| Error::limit("concatenation inner byte length overflows u64"))?;
    let mut target_offset = 0_u64;
    for outer_index in 0..outer {
        cancellation.check()?;
        for (&facts, input) in input_facts.iter().zip(inputs) {
            let chunk_bytes = facts.logical_shape()[axis]
                .checked_mul(inner_bytes)
                .ok_or_else(|| Error::limit("concatenation chunk length overflows u64"))?;
            let source_offset = outer_index
                .checked_mul(chunk_bytes)
                .ok_or_else(|| Error::limit("concatenation source offset overflows u64"))?;
            append_byte_range_tiled(
                input.as_slice(),
                source_offset,
                chunk_bytes,
                &mut output,
                cancellation,
            )?;
            target_offset = target_offset
                .checked_add(chunk_bytes)
                .ok_or_else(|| Error::limit("concatenation output offset overflows u64"))?;
        }
    }
    let target_len = u64::try_from(output.len()).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "concatenation output length does not fit u64",
            source,
        )
    })?;
    if target_offset != target_len || output.len() != output_length {
        return Err(Error::integrity(
            "concatenation did not initialize its complete output",
        ));
    }
    cancellation.check()?;
    Ok(output.into_boxed_slice())
}

fn execute_permute_bytes(
    permute: &Permute,
    input_facts: &TensorFacts,
    output_facts: &TensorFacts,
    input: &ByteView,
    output: &mut [u8],
    cancellation: &CancellationToken,
) -> Result<()> {
    validate_view_len(input, input_facts.byte_len(), "permutation input")?;
    let width = dtype_width(input_facts.representation().dtype())?;
    let output_row_strides = contiguous_strides(output_facts.logical_shape())?;
    let order = permutation_indices(permute.order())?;

    for linear in 0..output_facts.element_count() {
        check_copy_cancellation(linear, cancellation)?;
        let mut source_element = 0_u64;
        let mut target_element = 0_u64;
        match permute.mode() {
            PermuteMode::Logical => {
                for (output_axis, &input_axis) in order.iter().enumerate() {
                    let coordinate = logical_coordinate(
                        linear,
                        output_axis,
                        output_facts.logical_shape(),
                        &output_row_strides,
                    )?;
                    source_element = checked_offset_term(
                        source_element,
                        coordinate,
                        input_facts.storage_strides()[input_axis],
                    )?;
                    target_element = checked_offset_term(
                        target_element,
                        coordinate,
                        output_facts.storage_strides()[output_axis],
                    )?;
                }
            }
            PermuteMode::Storage { .. } => {
                for logical_axis in 0..output_facts.logical_shape().len() {
                    let coordinate = logical_coordinate(
                        linear,
                        logical_axis,
                        output_facts.logical_shape(),
                        &output_row_strides,
                    )?;
                    source_element = checked_offset_term(
                        source_element,
                        coordinate,
                        input_facts.storage_strides()[logical_axis],
                    )?;
                    target_element = checked_offset_term(
                        target_element,
                        coordinate,
                        output_facts.storage_strides()[logical_axis],
                    )?;
                }
            }
        }
        copy_element(
            input.as_slice(),
            source_element,
            output,
            target_element,
            width,
        )?;
    }
    cancellation.check()
}

fn build_oihw_to_ohwi_if_supported(
    permute: &Permute,
    input_facts: &TensorFacts,
    output_facts: &TensorFacts,
    input: &[u8],
    cancellation: &CancellationToken,
) -> Result<Option<Box<[u8]>>> {
    if !matches!(permute.mode(), PermuteMode::Storage { .. })
        || permute.order() != [0, 2, 3, 1]
        || !input_facts.is_canonical_contiguous()
    {
        return Ok(None);
    }
    cancellation.check()?;
    let [output_channels, input_channels, kernel_height, kernel_width] =
        input_facts.logical_shape()
    else {
        return Ok(None);
    };
    let output_channels = dimension_as_usize(*output_channels)?;
    let input_channels = dimension_as_usize(*input_channels)?;
    let kernel_height = dimension_as_usize(*kernel_height)?;
    let kernel_width = dimension_as_usize(*kernel_width)?;
    let spatial = kernel_height
        .checked_mul(kernel_width)
        .ok_or_else(|| Error::limit("OIHW spatial element count overflows usize"))?;
    let output_channel_elements = input_channels
        .checked_mul(spatial)
        .ok_or_else(|| Error::limit("OIHW output-channel element count overflows usize"))?;
    let elements = output_channels
        .checked_mul(output_channel_elements)
        .ok_or_else(|| Error::limit("OIHW element count overflows usize"))?;
    let width = dtype_width(input_facts.representation().dtype())?;
    let expected_bytes = elements
        .checked_mul(width)
        .ok_or_else(|| Error::limit("OIHW byte length overflows usize"))?;
    let output_bytes =
        validate_host_len(output_facts.byte_len(), "permutation output byte length")?;
    if input.len() != expected_bytes || output_bytes != expected_bytes {
        return Err(Error::integrity(
            "OIHW-to-OHWI buffers differ from inferred byte length",
        ));
    }
    let output = match width {
        1 => build_oihw_to_ohwi_elements::<1>(
            input,
            output_channels,
            input_channels,
            spatial,
            elements,
            cancellation,
        )?,
        2 if spatial == 9
            && matches!(
                input_facts.representation().dtype(),
                DType::F16 | DType::Bf16
            )
            && crate::operation_simd::oihw_to_ohwi_3x3_u16_avx2_available() =>
        {
            let mut output = allocate_tensor(output_facts, cancellation)?;
            crate::operation_simd::permute_oihw_to_ohwi_3x3_u16(
                input,
                &mut output,
                output_channels,
                input_channels,
                cancellation,
            )?;
            output
        }
        2 => build_oihw_to_ohwi_elements::<2>(
            input,
            output_channels,
            input_channels,
            spatial,
            elements,
            cancellation,
        )?,
        4 => build_oihw_to_ohwi_elements::<4>(
            input,
            output_channels,
            input_channels,
            spatial,
            elements,
            cancellation,
        )?,
        8 => build_oihw_to_ohwi_elements::<8>(
            input,
            output_channels,
            input_channels,
            spatial,
            elements,
            cancellation,
        )?,
        _ => return Ok(None),
    };
    Ok(Some(output))
}

fn build_oihw_to_ohwi_elements<const WIDTH: usize>(
    input: &[u8],
    output_channels: usize,
    input_channels: usize,
    spatial: usize,
    elements: usize,
    cancellation: &CancellationToken,
) -> Result<Box<[u8]>> {
    if elements == 0 {
        cancellation.check()?;
        return Ok(Vec::new().into_boxed_slice());
    }
    let mut output = Vec::<[u8; WIDTH]>::new();
    output.try_reserve_exact(elements).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "OIHW-to-OHWI output allocation failed",
            source,
        )
    })?;
    let source_channel_stride = spatial
        .checked_mul(WIDTH)
        .ok_or_else(|| Error::limit("OIHW input-channel byte stride overflows usize"))?;
    let output_channel_elements = input_channels
        .checked_mul(spatial)
        .ok_or_else(|| Error::limit("OIHW output-channel element count overflows usize"))?;

    for output_channel in 0..output_channels {
        let channel_base = output_channel
            .checked_mul(output_channel_elements)
            .ok_or_else(|| Error::limit("OIHW channel offset overflows usize"))?;
        for spatial_index in 0..spatial {
            let mut block_start = 0_usize;
            while block_start < input_channels {
                cancellation.check()?;
                let block_end = block_start
                    .saturating_add(COPY_TILE_ELEMENTS_USIZE)
                    .min(input_channels);
                let source_element = channel_base
                    .checked_add(
                        block_start
                            .checked_mul(spatial)
                            .ok_or_else(|| Error::limit("OIHW source offset overflows usize"))?,
                    )
                    .and_then(|offset| offset.checked_add(spatial_index))
                    .ok_or_else(|| Error::limit("OIHW source offset overflows usize"))?;
                let mut source_start = source_element
                    .checked_mul(WIDTH)
                    .ok_or_else(|| Error::limit("OIHW source byte offset overflows usize"))?;
                for input_channel in block_start..block_end {
                    let source = input
                        .get(source_start..)
                        .and_then(<[u8]>::first_chunk::<WIDTH>)
                        .ok_or_else(|| Error::integrity("OIHW source element is out of bounds"))?;
                    output.push(*source);
                    if input_channel + 1 < block_end {
                        // The complete input byte length and every block start
                        // were checked above, so each stride remains in range.
                        source_start += source_channel_stride;
                    }
                }
                block_start = block_end;
            }
        }
    }
    if output.len() != elements {
        return Err(Error::integrity(
            "OIHW-to-OHWI did not initialize its complete output",
        ));
    }
    cancellation.check()?;
    Ok(output.into_flattened().into_boxed_slice())
}

fn execute_slice_bytes(
    ranges: &[AxisRange],
    input_facts: &TensorFacts,
    input: &ByteView,
    output_facts: &TensorFacts,
    output: &mut [u8],
    cancellation: &CancellationToken,
) -> Result<()> {
    validate_view_len(input, input_facts.byte_len(), "slice input")?;
    if ranges.len() != input_facts.logical_shape().len() {
        return Err(Error::integrity(
            "slice execution range rank differs from input rank",
        ));
    }
    let width = dtype_width(input_facts.representation().dtype())?;
    let output_row_strides = contiguous_strides(output_facts.logical_shape())?;
    for linear in 0..output_facts.element_count() {
        check_copy_cancellation(linear, cancellation)?;
        let mut source_element = 0_u64;
        let mut target_element = 0_u64;
        for (logical_axis, range) in ranges.iter().copied().enumerate() {
            let output_coordinate = logical_coordinate(
                linear,
                logical_axis,
                output_facts.logical_shape(),
                &output_row_strides,
            )?;
            let source_coordinate = range
                .start()
                .checked_add(output_coordinate)
                .ok_or_else(|| Error::limit("slice source coordinate overflows u64"))?;
            source_element = checked_offset_term(
                source_element,
                source_coordinate,
                input_facts.storage_strides()[logical_axis],
            )?;
            target_element = checked_offset_term(
                target_element,
                output_coordinate,
                output_facts.storage_strides()[logical_axis],
            )?;
        }
        copy_element(
            input.as_slice(),
            source_element,
            output,
            target_element,
            width,
        )?;
    }
    cancellation.check()
}

fn append_byte_range_tiled(
    source: &[u8],
    source_offset: u64,
    byte_len: u64,
    target: &mut Vec<u8>,
    cancellation: &CancellationToken,
) -> Result<()> {
    const TILE_BYTES: u64 = 1024 * 1024;
    let mut copied = 0_u64;
    while copied < byte_len {
        cancellation.check()?;
        let tile = TILE_BYTES.min(byte_len - copied);
        let source_start = source_offset
            .checked_add(copied)
            .ok_or_else(|| Error::limit("source byte offset overflows u64"))?;
        let source_range = checked_byte_range(source_start, tile, source.len())?;
        target.extend_from_slice(
            source
                .get(source_range)
                .ok_or_else(|| Error::integrity("source byte tile is out of bounds"))?,
        );
        copied = copied
            .checked_add(tile)
            .ok_or_else(|| Error::limit("copied byte count overflows u64"))?;
    }
    Ok(())
}

fn copy_element(
    source: &[u8],
    source_element: u64,
    target: &mut [u8],
    target_element: u64,
    width: usize,
) -> Result<()> {
    let width_u64 = u64::try_from(width).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "scalar byte width does not fit u64",
            source,
        )
    })?;
    let source_start = source_element
        .checked_mul(width_u64)
        .ok_or_else(|| Error::limit("source element byte offset overflows u64"))?;
    let target_start = target_element
        .checked_mul(width_u64)
        .ok_or_else(|| Error::limit("target element byte offset overflows u64"))?;
    let source_range = checked_byte_range(source_start, width_u64, source.len())?;
    let target_range = checked_byte_range(target_start, width_u64, target.len())?;
    target
        .get_mut(target_range)
        .ok_or_else(|| Error::integrity("target element is out of bounds"))?
        .copy_from_slice(
            source
                .get(source_range)
                .ok_or_else(|| Error::integrity("source element is out of bounds"))?,
        );
    Ok(())
}

fn checked_byte_range(offset: u64, length: u64, bound: usize) -> Result<std::ops::Range<usize>> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::limit("byte range end overflows u64"))?;
    let start = usize::try_from(offset).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "byte range start does not fit usize",
            source,
        )
    })?;
    let end = usize::try_from(end).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "byte range end does not fit usize",
            source,
        )
    })?;
    if end > bound {
        return Err(Error::integrity("byte range lies outside its buffer"));
    }
    Ok(start..end)
}

fn logical_coordinate(linear: u64, axis: usize, shape: &[u64], row_strides: &[u64]) -> Result<u64> {
    let dimension = *shape
        .get(axis)
        .ok_or_else(|| Error::integrity("logical coordinate axis is out of bounds"))?;
    let stride = *row_strides
        .get(axis)
        .ok_or_else(|| Error::integrity("logical coordinate stride is unavailable"))?;
    if dimension == 0 || stride == 0 {
        return Err(Error::integrity(
            "logical coordinate requested for an empty tensor",
        ));
    }
    Ok((linear / stride) % dimension)
}

fn checked_offset_term(offset: u64, coordinate: u64, stride: u64) -> Result<u64> {
    coordinate
        .checked_mul(stride)
        .and_then(|term| offset.checked_add(term))
        .ok_or_else(|| Error::limit("logical element offset overflows u64"))
}

fn check_copy_cancellation(element: u64, cancellation: &CancellationToken) -> Result<()> {
    if element % COPY_TILE_ELEMENTS == 0 {
        cancellation.check()?;
    }
    Ok(())
}

fn dtype_width(dtype: DType) -> Result<usize> {
    let bits = dtype.bits();
    if bits == 0 || bits % 8 != 0 {
        return Err(Error::unsupported(
            "host structural operations require byte-aligned scalar dtypes",
        ));
    }
    Ok(usize::from(bits / 8))
}

fn validate_view_len(view: &ByteView, expected: u64, description: &'static str) -> Result<()> {
    let actual = u64::try_from(view.len()).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            format!("{description} length does not fit u64"),
            source,
        )
    })?;
    if actual != expected {
        return Err(Error::integrity(format!(
            "{description} has {actual} bytes, expected {expected}"
        )));
    }
    Ok(())
}

fn same_view(left: &ByteView, right: &ByteView) -> bool {
    left.len() == right.len() && std::ptr::eq(left.as_slice().as_ptr(), right.as_slice().as_ptr())
}

fn index_as_u32(index: usize, description: &'static str) -> Result<u32> {
    u32::try_from(index)
        .map_err(|source| Error::with_source(ErrorCategory::ResourceLimit, description, source))
}

fn infer_outputs(operation: &Operation, inputs: &[&TensorFacts]) -> Result<Box<[TensorFacts]>> {
    match operation {
        Operation::Concat(concat) => infer_concat(*concat, inputs),
        Operation::Permute(permute) => infer_permute(permute, one_input(inputs)?),
        Operation::Slice(slice) => infer_slice(slice, one_input(inputs)?),
        Operation::Split(split) => infer_split(split, one_input(inputs)?),
        Operation::Reshape(reshape) => infer_reshape(reshape, one_input(inputs)?),
        Operation::Prepare(transform) => infer_prepare(transform, one_input(inputs)?),
    }
}

fn one_input<'a>(inputs: &'a [&TensorFacts]) -> Result<&'a TensorFacts> {
    let [input] = inputs else {
        return Err(Error::binding("unary operation requires exactly one input"));
    };
    Ok(input)
}

fn infer_concat(concat: Concat, inputs: &[&TensorFacts]) -> Result<Box<[TensorFacts]>> {
    if inputs.len() < 2 {
        return Err(Error::binding(
            "concatenation requires at least two ordered inputs",
        ));
    }
    let first = inputs[0];
    let rank = first.logical_shape().len();
    let axis = validate_axis(concat.axis(), rank)?;
    let dtype = first.representation().dtype();
    let mut output_shape = first.logical_shape().to_vec();
    output_shape[axis] = 0;

    for input in inputs {
        if input.logical_shape().len() != rank {
            return Err(Error::binding("concatenation inputs must have equal ranks"));
        }
        if input.representation().dtype() != dtype {
            return Err(Error::binding(
                "concatenation inputs must have equal scalar dtypes",
            ));
        }
        for (index, (&actual, &expected)) in input
            .logical_shape()
            .iter()
            .zip(first.logical_shape())
            .enumerate()
        {
            if index != axis && actual != expected {
                return Err(Error::binding(
                    "concatenation input dimensions disagree outside the selected axis",
                ));
            }
        }
        output_shape[axis] = output_shape[axis]
            .checked_add(input.logical_shape()[axis])
            .ok_or_else(|| Error::limit("concatenated axis length overflows u64"))?;
    }

    Ok(vec![TensorFacts::contiguous(
        output_shape,
        Representation::contiguous(dtype),
    )?]
    .into_boxed_slice())
}

fn infer_permute(permute: &Permute, input: &TensorFacts) -> Result<Box<[TensorFacts]>> {
    validate_permutation(permute.order())?;
    if permute.order().len() != input.logical_shape().len() {
        return Err(Error::binding(
            "permutation rank must equal the input logical rank",
        ));
    }
    let order = permutation_indices(permute.order())?;
    let dtype = input.representation().dtype();
    let output = match permute.mode() {
        PermuteMode::Logical => {
            let shape = order
                .iter()
                .map(|&axis| input.logical_shape()[axis])
                .collect::<Vec<_>>();
            TensorFacts::contiguous(shape, Representation::contiguous(dtype))?
        }
        PermuteMode::Storage { target_layout } => {
            if target_layout.is_contiguous()
                && order
                    .iter()
                    .copied()
                    .enumerate()
                    .any(|(axis, source)| axis != source)
            {
                return Err(Error::binding(
                    "a non-identity storage permutation requires a non-contiguous target layout",
                ));
            }
            let storage_shape = order
                .iter()
                .map(|&axis| input.logical_shape()[axis])
                .collect::<Vec<_>>();
            let physical_axis_strides = contiguous_strides(&storage_shape)?;
            let mut storage_strides = vec![0_u64; order.len()];
            for (physical_axis, &logical_axis) in order.iter().enumerate() {
                storage_strides[logical_axis] = physical_axis_strides[physical_axis];
            }
            let representation = Representation::new(dtype, target_layout.clone());
            let byte_len = dtype.byte_len(&storage_shape)?;
            TensorFacts::new(
                input.logical_shape().to_vec(),
                storage_shape,
                input.logical_strides().to_vec(),
                storage_strides,
                representation,
                byte_len,
            )?
        }
    };
    Ok(vec![output].into_boxed_slice())
}

fn infer_slice(slice: &Slice, input: &TensorFacts) -> Result<Box<[TensorFacts]>> {
    if slice.ranges().len() != input.logical_shape().len() {
        return Err(Error::binding(
            "slice must contain one range per logical input axis",
        ));
    }
    let mut output_shape = Vec::with_capacity(slice.ranges().len());
    for (&range, &dimension) in slice.ranges().iter().zip(input.logical_shape()) {
        validate_range(range, dimension)?;
        output_shape.push(range.len());
    }
    Ok(vec![TensorFacts::contiguous(
        output_shape,
        Representation::contiguous(input.representation().dtype()),
    )?]
    .into_boxed_slice())
}

fn infer_split(split: &Split, input: &TensorFacts) -> Result<Box<[TensorFacts]>> {
    let axis = validate_axis(split.axis(), input.logical_shape().len())?;
    let mut previous_end = None;
    let mut outputs = Vec::with_capacity(split.ranges().len());
    for &range in split.ranges() {
        if range.is_empty() {
            return Err(Error::binding("split ranges must not be empty"));
        }
        validate_range(range, input.logical_shape()[axis])?;
        if previous_end.is_some_and(|end| range.start() < end) {
            return Err(Error::binding(
                "split ranges must be ordered and non-overlapping",
            ));
        }
        let mut shape = input.logical_shape().to_vec();
        shape[axis] = range.len();
        outputs.push(TensorFacts::contiguous(
            shape,
            Representation::contiguous(input.representation().dtype()),
        )?);
        previous_end = Some(range.end());
    }
    if outputs.is_empty() {
        return Err(Error::binding(
            "split must contain at least one output range",
        ));
    }
    Ok(outputs.into_boxed_slice())
}

fn infer_reshape(reshape: &Reshape, input: &TensorFacts) -> Result<Box<[TensorFacts]>> {
    if !input.is_canonical_contiguous() {
        return Err(Error::binding(
            "metadata-only reshape requires canonical contiguous input storage",
        ));
    }
    validate_rank(reshape.shape().len())?;
    let output_elements = checked_element_count(reshape.shape())?;
    if output_elements != input.element_count() {
        return Err(Error::binding(
            "reshape input and output element counts must be equal",
        ));
    }
    Ok(vec![TensorFacts::contiguous(
        reshape.shape().to_vec(),
        input.representation().clone(),
    )?]
    .into_boxed_slice())
}

fn infer_prepare(transform: &PlannedTransform, input: &TensorFacts) -> Result<Box<[TensorFacts]>> {
    if !input.is_canonical_contiguous() {
        return Err(Error::binding(
            "provider preparation requires canonical contiguous logical storage",
        ));
    }
    if transform.transform().source() != input.representation() {
        return Err(Error::binding(
            "provider preparation source representation does not match its input",
        ));
    }
    if !transform.transform().source().layout().is_contiguous()
        || !transform.transform().target().layout().is_contiguous()
    {
        return Err(Error::binding(
            "graph preparation currently requires contiguous source and target layouts",
        ));
    }
    let expected = transform
        .transform()
        .target()
        .dtype()
        .byte_len(input.logical_shape())?;
    if transform.output_size() != expected {
        return Err(Error::binding(format!(
            "provider preparation records {} output bytes, but inferred facts require {expected}",
            transform.output_size()
        )));
    }
    validate_host_len(transform.output_size(), "prepared output byte length")?;
    validate_host_len(transform.scratch_bytes(), "preparation scratch byte length")?;
    Ok(vec![TensorFacts::contiguous(
        input.logical_shape().to_vec(),
        transform.transform().target().clone(),
    )?]
    .into_boxed_slice())
}

fn validate_rank(rank: usize) -> Result<()> {
    u32::try_from(rank).map(|_| ()).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "tensor rank exceeds u32",
            source,
        )
    })
}

fn checked_element_count(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| Error::limit("tensor element count overflows u64"))
    })
}

fn contiguous_strides(shape: &[u64]) -> Result<Box<[u64]>> {
    validate_rank(shape.len())?;
    let mut strides = vec![0_u64; shape.len()];
    let mut stride = 1_u64;
    for (axis, &dimension) in shape.iter().enumerate().rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(dimension)
            .ok_or_else(|| Error::limit("contiguous tensor stride overflows u64"))?;
    }
    Ok(strides.into_boxed_slice())
}

fn validate_dense_strides(shape: &[u64], strides: &[u64], elements: u64) -> Result<()> {
    if elements == 0 {
        return Ok(());
    }
    let mut varying_axes = shape
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|(dimension, _)| *dimension > 1)
        .collect::<Vec<_>>();
    varying_axes.sort_unstable_by_key(|(_, stride)| *stride);
    let mut expected_stride = 1_u64;
    for (dimension, stride) in varying_axes {
        if stride != expected_stride {
            return Err(Error::invalid(
                "logical element strides must describe dense non-overlapping storage",
            ));
        }
        expected_stride = expected_stride
            .checked_mul(dimension)
            .ok_or_else(|| Error::limit("logical element-stride span overflows u64"))?;
    }
    if expected_stride != elements {
        return Err(Error::invalid(
            "logical element strides do not span the complete storage",
        ));
    }
    Ok(())
}

fn validate_storage_shape(
    logical_shape: &[u64],
    storage_shape: &[u64],
    storage_strides: &[u64],
    elements: u64,
) -> Result<()> {
    let mut logical_dimensions = logical_shape.to_vec();
    let mut storage_dimensions = storage_shape.to_vec();
    logical_dimensions.sort_unstable();
    storage_dimensions.sort_unstable();
    if logical_dimensions != storage_dimensions {
        return Err(Error::invalid(
            "storage shape must be an axis permutation of the logical shape",
        ));
    }
    if elements == 0 {
        return Ok(());
    }

    let mut logical_axes = logical_shape
        .iter()
        .copied()
        .zip(storage_strides.iter().copied())
        .filter(|(dimension, _)| *dimension > 1)
        .collect::<Vec<_>>();
    logical_axes.sort_unstable_by_key(|axis| Reverse(axis.1));
    let physical_dimensions = storage_shape
        .iter()
        .copied()
        .filter(|dimension| *dimension > 1);
    if !logical_axes
        .into_iter()
        .map(|(dimension, _)| dimension)
        .eq(physical_dimensions)
    {
        return Err(Error::invalid(
            "storage shape axis order disagrees with physical storage strides",
        ));
    }
    Ok(())
}

fn validate_permutation(order: &[u32]) -> Result<()> {
    validate_rank(order.len())?;
    let mut seen = vec![false; order.len()];
    for &axis in order {
        let index = usize::try_from(axis).map_err(|source| {
            Error::with_source(
                ErrorCategory::ResourceLimit,
                "permutation axis does not fit usize",
                source,
            )
        })?;
        let Some(slot) = seen.get_mut(index) else {
            return Err(Error::invalid(
                "permutation axis lies outside its declared rank",
            ));
        };
        if *slot {
            return Err(Error::invalid("permutation contains a duplicate axis"));
        }
        *slot = true;
    }
    Ok(())
}

fn is_identity_order(order: &[u32]) -> bool {
    order
        .iter()
        .copied()
        .enumerate()
        .all(|(index, axis)| usize::try_from(axis) == Ok(index))
}

fn is_identity_permutation(
    permute: &Permute,
    input_facts: &TensorFacts,
    output_facts: &TensorFacts,
) -> bool {
    is_identity_order(permute.order()) && input_facts == output_facts
}

fn permutation_indices(order: &[u32]) -> Result<Box<[usize]>> {
    order
        .iter()
        .map(|axis| {
            usize::try_from(*axis).map_err(|source| {
                Error::with_source(
                    ErrorCategory::ResourceLimit,
                    "permutation axis does not fit usize",
                    source,
                )
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

fn validate_axis(axis: Axis, rank: usize) -> Result<usize> {
    let axis = axis.as_usize()?;
    if axis >= rank {
        return Err(Error::binding("operation axis lies outside tensor rank"));
    }
    Ok(axis)
}

fn validate_range(range: AxisRange, dimension: u64) -> Result<()> {
    if range.start() > range.end() || range.end() > dimension {
        return Err(Error::binding(
            "operation range lies outside its logical axis",
        ));
    }
    Ok(())
}

fn validate_host_len(byte_len: u64, description: &'static str) -> Result<usize> {
    usize::try_from(byte_len).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            format!("{description} does not fit usize"),
            source,
        )
    })
}

fn dimension_as_usize(dimension: u64) -> Result<usize> {
    usize::try_from(dimension).map_err(|source| {
        Error::with_source(
            ErrorCategory::ResourceLimit,
            "tensor dimension does not fit usize",
            source,
        )
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::prepare::{TransformSpec, builtin_contiguous_implementation};

    fn name(value: &str) -> Result<TensorName> {
        TensorName::parse(value)
    }

    fn contiguous(shape: &[u64], dtype: DType) -> Result<TensorFacts> {
        TensorFacts::contiguous(shape.to_vec(), Representation::contiguous(dtype))
    }

    fn add_operation(
        builder: &mut OperationGraphBuilder,
        operation: Operation,
        inputs: Vec<ValueRef>,
    ) -> Result<Box<[ValueRef]>> {
        let implementation = operation.builtin_implementation()?;
        builder.add_operation(implementation, operation, inputs)
    }

    fn custom_layout(name: &str) -> Result<Layout> {
        Ok(Layout::custom(
            StableName::parse(name)?,
            1,
            Vec::<u8>::new(),
        ))
    }

    #[test]
    fn tensor_facts_distinguish_logical_oihw_from_physical_ohwi_strides() -> Result<()> {
        let facts = TensorFacts::new(
            [2, 3, 2, 2],
            [2, 2, 2, 3],
            [12, 4, 2, 1],
            [12, 1, 6, 3],
            Representation::new(DType::F16, custom_layout("test/ohwi")?),
            48,
        )?;

        assert_eq!(
            (facts.logical_strides(), facts.storage_strides()),
            (&[12, 4, 2, 1][..], &[12, 1, 6, 3][..])
        );
        Ok(())
    }

    #[test]
    fn tensor_facts_reject_overlapping_strides() {
        let error = TensorFacts::new(
            [2, 3],
            [2, 3],
            [3, 1],
            [1, 1],
            Representation::contiguous(DType::U8),
            6,
        )
        .expect_err("overlapping strides must fail");

        assert_eq!(error.category(), ErrorCategory::InvalidFormat);
    }

    #[test]
    fn tensor_facts_reject_incorrect_byte_length() {
        let error = TensorFacts::new(
            [2, 3],
            [2, 3],
            [3, 1],
            [3, 1],
            Representation::contiguous(DType::F16),
            6,
        )
        .expect_err("incorrect byte length must fail");

        assert_eq!(error.category(), ErrorCategory::InvalidFormat);
    }

    #[test]
    fn tensor_facts_reject_storage_shape_order_that_disagrees_with_strides() {
        let error = TensorFacts::new(
            [2, 3],
            [3, 2],
            [3, 1],
            [3, 1],
            Representation::new(
                DType::U8,
                Layout::custom(
                    StableName::parse("test/transposed").expect("static layout name is valid"),
                    1,
                    Vec::<u8>::new(),
                ),
            ),
            6,
        )
        .expect_err("physical shape order must agree with storage strides");

        assert_eq!(error.category(), ErrorCategory::InvalidFormat);
    }

    #[test]
    fn concat_infers_ordered_axis_sum_and_contiguous_output() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let q = builder.add_input(name("q")?, contiguous(&[2, 3], DType::F32)?)?;
        let k = builder.add_input(name("k")?, contiguous(&[4, 3], DType::F32)?)?;
        let v = builder.add_input(name("v")?, contiguous(&[1, 3], DType::F32)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Concat(Concat::new(Axis::from_index(0))),
            vec![q, k, v],
        )?;
        let graph = builder.build(outputs[0])?;

        assert_eq!(graph.output_facts().logical_shape(), [7, 3]);
        Ok(())
    }

    #[test]
    fn concat_rejects_dimension_mismatch_outside_axis() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let first = builder.add_input(name("first")?, contiguous(&[2, 3], DType::F32)?)?;
        let second = builder.add_input(name("second")?, contiguous(&[4, 5], DType::F32)?)?;
        let operation = Operation::Concat(Concat::new(Axis::from_index(0)));
        let implementation = operation.builtin_implementation()?;
        let error = builder
            .add_operation(implementation, operation, vec![first, second])
            .expect_err("mismatched concat dimensions must fail");

        assert_eq!(error.category(), ErrorCategory::Binding);
        Ok(())
    }

    #[test]
    fn storage_permute_preserves_logical_oihw_and_infers_ohwi_strides() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2, 3, 2, 2], DType::F16)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Permute(Permute::storage([0, 2, 3, 1], custom_layout("test/ohwi")?)?),
            vec![input],
        )?;
        let graph = builder.build(outputs[0])?;

        assert_eq!(
            (
                graph.output_facts().logical_shape(),
                graph.output_facts().storage_shape(),
                graph.output_facts().logical_strides(),
                graph.output_facts().storage_strides(),
            ),
            (
                &[2, 3, 2, 2][..],
                &[2, 2, 2, 3][..],
                &[12, 4, 2, 1][..],
                &[12, 1, 6, 3][..],
            )
        );
        Ok(())
    }

    #[test]
    fn logical_permute_changes_shape_and_returns_contiguous_storage() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2, 3, 5], DType::F16)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Permute(Permute::logical([2, 0, 1])?),
            vec![input],
        )?;
        let graph = builder.build(outputs[0])?;

        assert_eq!(graph.output_facts().logical_shape(), [5, 2, 3]);
        Ok(())
    }

    #[test]
    fn permutation_constructor_rejects_duplicate_axis() {
        let error = Permute::logical([0, 0]).expect_err("duplicate permutation axis must fail");

        assert_eq!(error.category(), ErrorCategory::InvalidFormat);
    }

    #[test]
    fn nonidentity_storage_permute_rejects_contiguous_layout() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2, 3], DType::F16)?)?;
        let operation = Operation::Permute(Permute::storage([1, 0], Layout::Contiguous)?);
        let implementation = operation.builtin_implementation()?;
        let error = builder
            .add_operation(implementation, operation, vec![input])
            .expect_err("contiguous layout cannot describe transposed physical storage");

        assert_eq!(error.category(), ErrorCategory::Binding);
        Ok(())
    }

    #[test]
    fn slice_infers_all_half_open_range_lengths() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[4, 6], DType::F32)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Slice(Slice::new([AxisRange::new(1, 4)?, AxisRange::new(2, 5)?])),
            vec![input],
        )?;
        let graph = builder.build(outputs[0])?;

        assert_eq!(graph.output_facts().logical_shape(), [3, 3]);
        Ok(())
    }

    #[test]
    fn slice_rejects_range_outside_input() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[4], DType::F32)?)?;
        let operation = Operation::Slice(Slice::new([AxisRange::new(0, 5)?]));
        let implementation = operation.builtin_implementation()?;
        let error = builder
            .add_operation(implementation, operation, vec![input])
            .expect_err("out-of-bounds slice must fail");

        assert_eq!(error.category(), ErrorCategory::Binding);
        Ok(())
    }

    #[test]
    fn split_infers_ordered_multiple_outputs() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("qkv")?, contiguous(&[9, 2], DType::F32)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Split(Split::new(
                Axis::from_index(0),
                [
                    AxisRange::new(0, 2)?,
                    AxisRange::new(3, 6)?,
                    AxisRange::new(6, 9)?,
                ],
            )?),
            vec![input],
        )?;
        let graph = builder.build(outputs[1])?;

        assert_eq!(graph.nodes()[0].outputs()[2].logical_shape(), [3, 2]);
        Ok(())
    }

    #[test]
    fn split_constructor_rejects_overlap() -> Result<()> {
        let error = Split::new(
            Axis::from_index(0),
            [AxisRange::new(0, 3)?, AxisRange::new(2, 4)?],
        )
        .expect_err("overlapping split ranges must fail");

        assert_eq!(error.category(), ErrorCategory::InvalidFormat);
        Ok(())
    }

    #[test]
    fn reshape_rejects_different_element_count() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2, 3], DType::F32)?)?;
        let operation = Operation::Reshape(Reshape::new([5]));
        let implementation = operation.builtin_implementation()?;
        let error = builder
            .add_operation(implementation, operation, vec![input])
            .expect_err("element-changing reshape must fail");

        assert_eq!(error.category(), ErrorCategory::Binding);
        Ok(())
    }

    #[test]
    fn output_input_alias_traces_only_metadata_reshape_paths() -> Result<()> {
        let mut reshape_builder = OperationGraph::builder();
        let input = reshape_builder.add_input(name("weight")?, contiguous(&[2, 3], DType::F32)?)?;
        let first = add_operation(
            &mut reshape_builder,
            Operation::Reshape(Reshape::new([3, 2])),
            vec![input],
        )?;
        let second = add_operation(
            &mut reshape_builder,
            Operation::Reshape(Reshape::new([6])),
            vec![first[0]],
        )?;
        let reshape_graph = reshape_builder.build(second[0])?;

        assert_eq!(reshape_graph.output_input_alias(), Some(0));

        let mut concat_builder = OperationGraph::builder();
        let left = concat_builder.add_input(name("left")?, contiguous(&[1], DType::U8)?)?;
        let right = concat_builder.add_input(name("right")?, contiguous(&[1], DType::U8)?)?;
        let concatenated = add_operation(
            &mut concat_builder,
            Operation::Concat(Concat::new(Axis::from_index(0))),
            vec![left, right],
        )?;
        let concat_graph = concat_builder.build(concatenated[0])?;

        assert_eq!(concat_graph.output_input_alias(), None);
        Ok(())
    }

    #[test]
    fn graph_rejects_forward_node_reference() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let _ = builder.add_input(name("weight")?, contiguous(&[2, 3], DType::F32)?)?;
        let operation = Operation::Reshape(Reshape::new([6]));
        let implementation = operation.builtin_implementation()?;
        let forward = ValueRef::node_output(NodeId::from_ordinal(0), OutputIndex::from_ordinal(0));
        let error = builder
            .add_operation(implementation, operation, vec![forward])
            .expect_err("forward node reference must fail");

        assert_eq!(error.category(), ErrorCategory::Binding);
        Ok(())
    }

    #[test]
    fn graph_serde_roundtrip_preserves_order_and_reinfers_outputs() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let z = builder.add_input(name("z")?, contiguous(&[1, 2], DType::U8)?)?;
        let a = builder.add_input(name("a")?, contiguous(&[2, 2], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Concat(Concat::new(Axis::from_index(0))),
            vec![z, a],
        )?;
        let graph = builder.build(outputs[0])?;
        let json = serde_json::to_vec(&graph).map_err(|source| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "serialize operation graph test value",
                source,
            )
        })?;
        let decoded: OperationGraph = serde_json::from_slice(&json).map_err(|source| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "deserialize operation graph test value",
                source,
            )
        })?;

        assert_eq!(decoded, graph);
        Ok(())
    }

    #[test]
    fn graph_deserialization_rejects_unknown_schema() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[1], DType::U8)?)?;
        let graph = builder.build(input)?;
        let mut value = serde_json::to_value(&graph).map_err(|source| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "serialize operation graph test value",
                source,
            )
        })?;
        value["schema_version"] = serde_json::json!(2);

        let error = serde_json::from_value::<OperationGraph>(value)
            .expect_err("unknown operation graph schema must fail");
        assert!(error.to_string().contains("schema version"));
        Ok(())
    }

    #[test]
    fn host_concat_preserves_order_and_reports_zero_final_only_scratch() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let left = builder.add_input(name("left")?, contiguous(&[2], DType::U8)?)?;
        let right = builder.add_input(name("right")?, contiguous(&[3], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Concat(Concat::new(Axis::from_index(0))),
            vec![left, right],
        )?;
        let graph = builder.build(outputs[0])?;
        let inputs = [
            ByteView::from_boxed(vec![1, 2].into_boxed_slice()),
            ByteView::from_boxed(vec![3, 4, 5].into_boxed_slice()),
        ];
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&inputs, &engine, &CancellationToken::new())?;

        assert_eq!(
            (
                execution.output().as_slice(),
                execution.peak_scratch_bytes()
            ),
            (&[1, 2, 3, 4, 5][..], 0)
        );
        Ok(())
    }

    #[test]
    fn host_contiguous_concat_interleaves_outer_rows_without_padding() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let left = builder.add_input(name("left")?, contiguous(&[2, 2], DType::U8)?)?;
        let right = builder.add_input(name("right")?, contiguous(&[2, 1], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Concat(Concat::new(Axis::from_index(1))),
            vec![left, right],
        )?;
        let graph = builder.build(outputs[0])?;
        let inputs = [
            ByteView::from_boxed(vec![1, 2, 3, 4].into_boxed_slice()),
            ByteView::from_boxed(vec![5, 6].into_boxed_slice()),
        ];
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&inputs, &engine, &CancellationToken::new())?;

        assert_eq!(execution.output().as_slice(), [1, 2, 5, 3, 4, 6]);
        Ok(())
    }

    #[test]
    fn host_storage_permute_writes_ohwi_bytes() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[1, 2, 2, 2], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Permute(Permute::storage([0, 2, 3, 1], custom_layout("test/ohwi")?)?),
            vec![input],
        )?;
        let graph = builder.build(outputs[0])?;
        let input = ByteView::from_boxed(vec![0, 1, 2, 3, 4, 5, 6, 7].into_boxed_slice());
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&[input], &engine, &CancellationToken::new())?;

        assert_eq!(execution.output().as_slice(), [0, 4, 1, 5, 2, 6, 3, 7]);
        Ok(())
    }

    #[test]
    fn host_storage_permute_matches_oihw_to_ohwi_reference_for_all_widths() -> Result<()> {
        for (dtype, width) in [
            (DType::U8, 1_usize),
            (DType::F16, 2),
            (DType::F32, 4),
            (DType::F64, 8),
        ] {
            let (output_channels, input_channels, spatial) = (2_usize, 3_usize, 9_usize);
            let element_count = output_channels * input_channels * spatial;
            let input_bytes = (0..element_count * width)
                .map(|index| u8::try_from((index * 37 + 11) % 251).expect("value fits u8"))
                .collect::<Vec<_>>();
            let mut expected = Vec::with_capacity(input_bytes.len());
            for output_channel in 0..output_channels {
                for spatial_index in 0..spatial {
                    for input_channel in 0..input_channels {
                        let source_element = (output_channel * input_channels + input_channel)
                            * spatial
                            + spatial_index;
                        let source_start = source_element * width;
                        expected
                            .extend_from_slice(&input_bytes[source_start..source_start + width]);
                    }
                }
            }

            let mut builder = OperationGraph::builder();
            let input = builder.add_input(name("weight")?, contiguous(&[2, 3, 3, 3], dtype)?)?;
            let outputs = add_operation(
                &mut builder,
                Operation::Permute(Permute::storage([0, 2, 3, 1], custom_layout("test/ohwi")?)?),
                vec![input],
            )?;
            let graph = builder.build(outputs[0])?;
            let input = ByteView::from_boxed(input_bytes.into_boxed_slice());
            let engine = PreparationEngine::with_builtins()?;
            let execution = graph.execute_host(&[input], &engine, &CancellationToken::new())?;

            assert_eq!(execution.output().as_slice(), expected, "dtype {dtype:?}");
        }
        Ok(())
    }

    #[test]
    fn host_f16_bf16_3x3_storage_permute_preserves_raw_bits_across_simd_tails() -> Result<()> {
        const RAW_EDGES: [u16; 8] = [
            0x0000, 0x0001, 0x3C00, 0x7BFF, 0x7C00, 0x7E01, 0x8000, 0xFFFF,
        ];
        for dtype in [DType::F16, DType::Bf16] {
            for input_channels in [
                0_usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1_279, 1_280, 1_281,
            ] {
                for pattern in 0..3 {
                    let output_channels = 2_usize;
                    let elements = output_channels * input_channels * 9;
                    let mut input_bytes = Vec::with_capacity(elements * 2);
                    for element in 0..elements {
                        let bits = match pattern {
                            0 => u16::try_from(element & usize::from(u16::MAX))
                                .expect("masked element fits u16"),
                            1 => {
                                let mixed = element
                                    .wrapping_mul(0x9E37)
                                    .wrapping_add(element.rotate_left(7));
                                u16::try_from(mixed & usize::from(u16::MAX))
                                    .expect("masked element fits u16")
                            }
                            _ => RAW_EDGES[element % RAW_EDGES.len()],
                        };
                        input_bytes.extend_from_slice(&bits.to_ne_bytes());
                    }
                    let mut expected = Vec::with_capacity(input_bytes.len());
                    for output_channel in 0..output_channels {
                        for spatial_index in 0..9 {
                            for input_channel in 0..input_channels {
                                let source_element =
                                    (output_channel * input_channels + input_channel) * 9
                                        + spatial_index;
                                let source_start = source_element * 2;
                                expected.extend_from_slice(
                                    &input_bytes[source_start..source_start + 2],
                                );
                            }
                        }
                    }

                    let mut builder = OperationGraph::builder();
                    let input = builder.add_input(
                        name("weight")?,
                        contiguous(
                            &[output_channels as u64, input_channels as u64, 3, 3],
                            dtype,
                        )?,
                    )?;
                    let outputs = add_operation(
                        &mut builder,
                        Operation::Permute(Permute::storage(
                            [0, 2, 3, 1],
                            custom_layout("test/ohwi")?,
                        )?),
                        vec![input],
                    )?;
                    let graph = builder.build(outputs[0])?;
                    let input = ByteView::from_boxed(input_bytes.into_boxed_slice());
                    let engine = PreparationEngine::with_builtins()?;
                    let execution =
                        graph.execute_host(&[input], &engine, &CancellationToken::new())?;

                    assert_eq!(
                        execution.output().as_slice(),
                        expected,
                        "dtype {dtype:?}, channels {input_channels}, pattern {pattern}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn host_logical_permute_writes_transposed_contiguous_bytes() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2, 3], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Permute(Permute::logical([1, 0])?),
            vec![input],
        )?;
        let graph = builder.build(outputs[0])?;
        let input = ByteView::from_boxed(vec![0, 1, 2, 3, 4, 5].into_boxed_slice());
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&[input], &engine, &CancellationToken::new())?;

        assert_eq!(execution.output().as_slice(), [0, 3, 1, 4, 2, 5]);
        Ok(())
    }

    #[test]
    fn host_identity_permutations_retain_the_original_byte_view() -> Result<()> {
        for operation in [
            Operation::Permute(Permute::logical([0, 1])?),
            Operation::Permute(Permute::storage([0, 1], Layout::Contiguous)?),
        ] {
            let mut builder = OperationGraph::builder();
            let input = builder.add_input(name("weight")?, contiguous(&[2, 3], DType::U8)?)?;
            let outputs = add_operation(&mut builder, operation, vec![input])?;
            let graph = builder.build(outputs[0])?;
            let input = ByteView::from_boxed(vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
            let input_pointer = input.as_slice().as_ptr();
            let engine = PreparationEngine::with_builtins()?;
            let execution = graph.execute_host(&[input], &engine, &CancellationToken::new())?;

            assert_eq!(graph.output_input_alias(), Some(0));
            assert_eq!(graph.estimate_host_scratch_bytes()?, 0);
            assert_eq!(execution.peak_scratch_bytes(), 0);
            assert_eq!(execution.output().as_slice().as_ptr(), input_pointer);
        }
        Ok(())
    }

    #[test]
    fn host_nonidentity_square_permutation_does_not_alias_equal_facts() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2, 2], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Permute(Permute::logical([1, 0])?),
            vec![input],
        )?;
        let graph = builder.build(outputs[0])?;
        assert_eq!(graph.inputs()[0].facts(), graph.output_facts());
        assert_eq!(graph.output_input_alias(), None);

        let input = ByteView::from_boxed(vec![0, 1, 2, 3].into_boxed_slice());
        let input_pointer = input.as_slice().as_ptr();
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&[input], &engine, &CancellationToken::new())?;

        assert_ne!(execution.output().as_slice().as_ptr(), input_pointer);
        assert_eq!(execution.output().as_slice(), [0, 2, 1, 3]);
        Ok(())
    }

    #[test]
    fn host_slice_materializes_each_axis_range() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[3, 4], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Slice(Slice::new([AxisRange::new(1, 3)?, AxisRange::new(1, 3)?])),
            vec![input],
        )?;
        let graph = builder.build(outputs[0])?;
        let input = ByteView::from_boxed((0_u8..12).collect::<Vec<_>>().into_boxed_slice());
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&[input], &engine, &CancellationToken::new())?;

        assert_eq!(execution.output().as_slice(), [5, 6, 9, 10]);
        Ok(())
    }

    #[test]
    fn host_split_materializes_selected_range() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[6], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Split(Split::new(
                Axis::from_index(0),
                [AxisRange::new(0, 2)?, AxisRange::new(3, 6)?],
            )?),
            vec![input],
        )?;
        let graph = builder.build(outputs[1])?;
        let input = ByteView::from_boxed(vec![10, 11, 12, 13, 14, 15].into_boxed_slice());
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&[input], &engine, &CancellationToken::new())?;

        assert_eq!(execution.output().as_slice(), [13, 14, 15]);
        Ok(())
    }

    #[test]
    fn host_reshape_is_zero_copy() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2, 3], DType::U8)?)?;
        let outputs = add_operation(
            &mut builder,
            Operation::Reshape(Reshape::new([3, 2])),
            vec![input],
        )?;
        let graph = builder.build(outputs[0])?;
        let input = ByteView::from_boxed(vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
        let input_pointer = input.as_slice().as_ptr();
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&[input], &engine, &CancellationToken::new())?;

        assert_eq!(execution.output().as_slice().as_ptr(), input_pointer);
        Ok(())
    }

    #[test]
    fn host_execution_rejects_nonbuiltin_structural_implementation() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2], DType::U8)?)?;
        let operation = Operation::Reshape(Reshape::new([1, 2]));
        let custom = ImplementationId::new(
            StableName::parse("consumer")?,
            StableName::parse("reshape")?,
            7,
        );
        let outputs = builder.add_operation(custom, operation, vec![input])?;
        let graph = builder.build(outputs[0])?;
        let input = ByteView::from_boxed(vec![1, 2].into_boxed_slice());
        let engine = PreparationEngine::with_builtins()?;
        let error = graph
            .execute_host(&[input], &engine, &CancellationToken::new())
            .expect_err("host execution must reject delegated implementation IDs");

        assert_eq!(error.category(), ErrorCategory::Unsupported);
        Ok(())
    }

    #[test]
    fn host_execution_observes_pre_cancelled_token() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2], DType::U8)?)?;
        let graph = builder.build(input)?;
        let input = ByteView::from_boxed(vec![1, 2].into_boxed_slice());
        let engine = PreparationEngine::with_builtins()?;
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = graph
            .execute_host(&[input], &engine, &cancellation)
            .expect_err("pre-cancelled host execution must stop");

        assert_eq!(error.category(), ErrorCategory::Cancelled);
        Ok(())
    }

    #[test]
    fn liveness_estimate_counts_intermediate_but_excludes_final_output() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let left = builder.add_input(name("left")?, contiguous(&[2], DType::U8)?)?;
        let right = builder.add_input(name("right")?, contiguous(&[2], DType::U8)?)?;
        let concat = add_operation(
            &mut builder,
            Operation::Concat(Concat::new(Axis::from_index(0))),
            vec![left, right],
        )?;
        let slice = add_operation(
            &mut builder,
            Operation::Slice(Slice::new([AxisRange::new(1, 3)?])),
            vec![concat[0]],
        )?;
        let graph = builder.build(slice[0])?;

        assert_eq!(graph.estimate_host_scratch_bytes()?, 4);
        Ok(())
    }

    #[test]
    fn identity_permute_reuses_live_intermediate_allocation() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let left = builder.add_input(name("left")?, contiguous(&[2], DType::U8)?)?;
        let right = builder.add_input(name("right")?, contiguous(&[2], DType::U8)?)?;
        let concat = add_operation(
            &mut builder,
            Operation::Concat(Concat::new(Axis::from_index(0))),
            vec![left, right],
        )?;
        let identity = add_operation(
            &mut builder,
            Operation::Permute(Permute::logical([0])?),
            vec![concat[0]],
        )?;
        let slice = add_operation(
            &mut builder,
            Operation::Slice(Slice::new([AxisRange::new(1, 3)?])),
            vec![identity[0]],
        )?;
        let graph = builder.build(slice[0])?;
        let inputs = [
            ByteView::from_boxed(vec![1, 2].into_boxed_slice()),
            ByteView::from_boxed(vec![3, 4].into_boxed_slice()),
        ];
        let engine = PreparationEngine::with_builtins()?;
        let execution = graph.execute_host(&inputs, &engine, &CancellationToken::new())?;

        assert_eq!(graph.estimate_host_scratch_bytes()?, 4);
        assert_eq!(execution.peak_scratch_bytes(), 4);
        assert_eq!(execution.output().as_slice(), [2, 3]);
        Ok(())
    }

    #[test]
    fn prepare_inference_rejects_wrong_declared_output_bytes() -> Result<()> {
        let mut builder = OperationGraph::builder();
        let input = builder.add_input(name("weight")?, contiguous(&[2], DType::F32)?)?;
        let transform = TransformSpec::new(
            builtin_contiguous_implementation()?,
            Representation::contiguous(DType::F32),
            Representation::contiguous(DType::F16),
        );
        let operation = Operation::Prepare(PlannedTransform::new(transform, 8));
        let implementation = operation.builtin_implementation()?;
        let error = builder
            .add_operation(implementation, operation, vec![input])
            .expect_err("incorrect prepare output bytes must fail");

        assert_eq!(error.category(), ErrorCategory::Binding);
        Ok(())
    }

    proptest! {
        #[test]
        fn logical_permutations_preserve_element_and_byte_counts(
            axes in prop::collection::vec((1_u64..8, any::<u32>()), 1..7),
        ) {
            let dimensions = axes
                .iter()
                .map(|(dimension, _)| *dimension)
                .collect::<Vec<_>>();
            let mut order = (0..dimensions.len()).collect::<Vec<_>>();
            order.sort_by_key(|&axis| (axes[axis].1, axis));
            let order = order
                .into_iter()
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()
                .expect("generated rank fits u32");
            let mut builder = OperationGraph::builder();
            let input = builder
                .add_input(
                    TensorName::parse("input").expect("static name is valid"),
                    TensorFacts::contiguous(
                        dimensions.clone(),
                        Representation::contiguous(DType::F16),
                    ).expect("small generated dimensions fit"),
                )
                .expect("single input is valid");
            let operation = Operation::Permute(
                Permute::logical(order).expect("generated order is a permutation")
            );
            let implementation = operation
                .builtin_implementation()
                .expect("static implementation names are valid");
            let outputs = builder
                .add_operation(implementation, operation, vec![input])
                .expect("logical permutation is valid");
            let graph = builder.build(outputs[0]).expect("output exists");

            prop_assert_eq!(
                graph.output_facts().element_count(),
                graph.inputs()[0].facts().element_count()
            );
            prop_assert_eq!(
                graph.output_facts().byte_len(),
                graph.inputs()[0].facts().byte_len()
            );
        }

        #[test]
        fn concat_axis_length_is_ordered_input_sum(
            first in 0_u64..32,
            second in 0_u64..32,
            third in 0_u64..32,
            axis in 0_usize..3,
        ) {
            let mut first_shape = vec![2_u64, 3, 4];
            let mut second_shape = first_shape.clone();
            let mut third_shape = first_shape.clone();
            first_shape[axis] = first;
            second_shape[axis] = second;
            third_shape[axis] = third;
            let mut builder = OperationGraph::builder();
            let a = builder.add_input(
                TensorName::parse("a").expect("static name is valid"),
                TensorFacts::contiguous(first_shape, Representation::contiguous(DType::U8))
                    .expect("small generated shape fits"),
            ).expect("unique input is valid");
            let b = builder.add_input(
                TensorName::parse("b").expect("static name is valid"),
                TensorFacts::contiguous(second_shape, Representation::contiguous(DType::U8))
                    .expect("small generated shape fits"),
            ).expect("unique input is valid");
            let c = builder.add_input(
                TensorName::parse("c").expect("static name is valid"),
                TensorFacts::contiguous(third_shape, Representation::contiguous(DType::U8))
                    .expect("small generated shape fits"),
            ).expect("unique input is valid");
            let operation = Operation::Concat(Concat::new(
                Axis::try_from(axis).expect("generated axis fits"),
            ));
            let implementation = operation
                .builtin_implementation()
                .expect("static implementation names are valid");
            let outputs = builder
                .add_operation(implementation, operation, vec![a, b, c])
                .expect("generated concat inputs are compatible");
            let graph = builder.build(outputs[0]).expect("output exists");

            prop_assert_eq!(
                graph.output_facts().logical_shape()[axis],
                first + second + third
            );
        }

        #[test]
        fn storage_permutations_preserve_logical_strides_and_infer_physical_strides(
            axes in prop::collection::vec((1_u64..8, any::<u32>()), 1..7),
        ) {
            let dimensions = axes
                .iter()
                .map(|(dimension, _)| *dimension)
                .collect::<Vec<_>>();
            let mut order = (0..dimensions.len()).collect::<Vec<_>>();
            order.sort_by_key(|&axis| (axes[axis].1, axis));
            let serialized_order = order
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()
                .expect("generated rank fits u32");
            let mut builder = OperationGraph::builder();
            let input = builder
                .add_input(
                    TensorName::parse("input").expect("static name is valid"),
                    TensorFacts::contiguous(
                        dimensions.clone(),
                        Representation::contiguous(DType::U8),
                    ).expect("small generated dimensions fit"),
                )
                .expect("single input is valid");
            let operation = Operation::Permute(Permute::storage(
                serialized_order,
                Layout::custom(
                    StableName::parse("test/permuted").expect("static name is valid"),
                    1,
                    Vec::<u8>::new(),
                ),
            ).expect("generated order is a permutation"));
            let implementation = operation
                .builtin_implementation()
                .expect("static implementation names are valid");
            let outputs = builder
                .add_operation(implementation, operation, vec![input])
                .expect("storage permutation is valid");
            let graph = builder.build(outputs[0]).expect("output exists");
            let expected_storage_shape = order
                .iter()
                .map(|&axis| dimensions[axis])
                .collect::<Vec<_>>();
            let physical_strides = contiguous_strides(&expected_storage_shape)
                .expect("small generated dimensions fit");
            let mut expected_storage_strides = vec![0_u64; dimensions.len()];
            for (physical_axis, &logical_axis) in order.iter().enumerate() {
                expected_storage_strides[logical_axis] = physical_strides[physical_axis];
            }

            prop_assert_eq!(
                graph.output_facts().logical_strides(),
                graph.inputs()[0].facts().logical_strides()
            );
            prop_assert_eq!(
                graph.output_facts().storage_strides(),
                expected_storage_strides
            );
        }
    }
}
