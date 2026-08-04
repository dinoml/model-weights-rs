//! Deterministic source selection, target binding, and declarative recipes.
//!
//! Planning is an inert metadata operation. It resolves normalized inventory
//! descriptors against a consumer contract, reports all deterministic binding
//! diagnostics, and pins every byte-affecting transform or quantized route.
//! Materialization and runtime policy remain separate concerns.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};

use crate::identity::{
    BackendId, ContentDigest, ContractId, ImplementationId, ManifestId, PlanId, SelectionId,
    StableName,
};
use crate::operation::{OperationGraph, TensorFacts};
use crate::prepare::{Layout, Representation, TransformSpec};
use crate::quantization::{QuantizedRoute, RouteCapability, Storage};
use crate::{Error, ErrorCategory, Result};

/// The canonical binding-plan schema version.
///
/// Version 2 adds ordered multi-source bindings, typed operation graphs, and
/// explicit logical-versus-physical tensor geometry.
pub const PLAN_SCHEMA_VERSION: u32 = 2;

/// An exact non-empty UTF-8 tensor or target name.
///
/// Tensor names are data, not provider identifiers, so they intentionally do
/// not use [`StableName`]'s portable ASCII grammar or 128-byte limit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct TensorName(Box<str>);

impl TensorName {
    /// Validates a non-empty exact tensor name.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error when the name is empty.
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let value = value.as_ref();
        if value.is_empty() {
            return Err(Error::invalid("tensor name must not be empty"));
        }
        Ok(Self(value.into()))
    }

    /// Validates a tensor name against a caller-selected byte limit.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error for an empty name or a resource-limit
    /// error when the UTF-8 byte length exceeds `max_bytes`.
    pub fn parse_with_max(value: impl AsRef<str>, max_bytes: usize) -> Result<Self> {
        let value = Self::parse(value)?;
        if value.as_str().len() > max_bytes {
            return Err(Error::limit(
                "tensor name exceeds the configured byte limit",
            ));
        }
        Ok(value)
    }

    /// Returns the exact UTF-8 name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for TensorName {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for TensorName {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        Self::parse(value)
    }
}

impl<'de> Deserialize<'de> for TensorName {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Box::<str>::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

/// A deterministic parameter in a provider-defined conversion recipe.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RecipeValue {
    /// A Boolean value.
    Bool(bool),
    /// A signed integer.
    I64(i64),
    /// An unsigned integer.
    U64(u64),
    /// UTF-8 text.
    Text(Box<str>),
    /// Canonical opaque bytes.
    Bytes(Box<[u8]>),
    /// An ordered sequence.
    Sequence(Box<[RecipeValue]>),
    /// A deterministically ordered object.
    Object(BTreeMap<StableName, RecipeValue>),
}

/// Identifies one input edge of a declarative recipe step.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RecipeInput {
    /// A named input supplied by the recipe caller.
    External(TensorName),
    /// The named output of an earlier step.
    Step(StableName),
}

/// One ordered operation in a provider-defined conversion DAG.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecipeStep {
    output: StableName,
    operation: ImplementationId,
    inputs: Box<[RecipeInput]>,
    parameters: BTreeMap<StableName, RecipeValue>,
}

impl RecipeStep {
    /// Creates one declarative operation.
    #[must_use]
    pub fn new(
        output: StableName,
        operation: ImplementationId,
        inputs: impl Into<Box<[RecipeInput]>>,
        parameters: BTreeMap<StableName, RecipeValue>,
    ) -> Self {
        Self {
            output,
            operation,
            inputs: inputs.into(),
            parameters,
        }
    }

    /// Returns the provider-local output name.
    #[must_use]
    pub const fn output(&self) -> &StableName {
        &self.output
    }

    /// Returns the exact provider operation and semantic version.
    #[must_use]
    pub const fn operation(&self) -> &ImplementationId {
        &self.operation
    }

    /// Returns ordered operation inputs.
    #[must_use]
    pub const fn inputs(&self) -> &[RecipeInput] {
        &self.inputs
    }

    /// Returns deterministic provider parameters.
    #[must_use]
    pub const fn parameters(&self) -> &BTreeMap<StableName, RecipeValue> {
        &self.parameters
    }
}

/// A versioned, provider-defined conversion DAG.
///
/// Recipes are deliberately declarative and language-neutral. A Rust adapter,
/// future Python binding, or Diffusers differential validator can interpret the
/// same ordered operations without embedding Python in the core crate. The
/// selected provider executes the recipe as one external source-to-target
/// conversion stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ConversionRecipeWire")]
pub struct ConversionRecipe {
    schema_version: u32,
    implementation: ImplementationId,
    inputs: Box<[TensorName]>,
    steps: Box<[RecipeStep]>,
    outputs: BTreeMap<TensorName, RecipeInput>,
}

impl ConversionRecipe {
    /// Validates and creates a declarative conversion recipe.
    ///
    /// `outputs` maps consumer-visible output names to either a declared input
    /// or a step output. Step inputs may refer only to declared inputs or
    /// earlier steps, so cycles and forward references are impossible.
    ///
    /// # Errors
    ///
    /// Returns a binding error for a zero schema version, duplicate names,
    /// undeclared or forward references, empty outputs, or outputs that do not
    /// name available values.
    pub fn new(
        schema_version: u32,
        implementation: ImplementationId,
        inputs: Vec<TensorName>,
        steps: Vec<RecipeStep>,
        outputs: BTreeMap<TensorName, RecipeInput>,
    ) -> Result<Self> {
        if schema_version == 0 {
            return Err(Error::binding(
                "conversion recipe schema version must be greater than zero",
            ));
        }
        let declared = inputs.iter().cloned().collect::<BTreeSet<_>>();
        if declared.len() != inputs.len() {
            return Err(Error::binding(
                "conversion recipe contains a duplicate declared input",
            ));
        }
        if outputs.is_empty() {
            return Err(Error::binding(
                "conversion recipe must declare at least one output",
            ));
        }

        let mut available_steps = BTreeSet::<StableName>::new();
        for step in &steps {
            if available_steps.contains(step.output()) {
                return Err(Error::binding(
                    "conversion recipe step output conflicts with an existing value",
                ));
            }
            for input in step.inputs() {
                match input {
                    RecipeInput::External(name) if !declared.contains(name) => {
                        return Err(Error::binding(
                            "conversion recipe step references an undeclared external input",
                        ));
                    }
                    RecipeInput::Step(name) if !available_steps.contains(name) => {
                        return Err(Error::binding(
                            "conversion recipe step references a missing or later step output",
                        ));
                    }
                    RecipeInput::External(_) | RecipeInput::Step(_) => {}
                }
            }
            available_steps.insert(step.output().clone());
        }
        for output in outputs.values() {
            let available = match output {
                RecipeInput::External(name) => declared.contains(name),
                RecipeInput::Step(name) => available_steps.contains(name),
            };
            if !available {
                return Err(Error::binding(
                    "conversion recipe output references an unavailable value",
                ));
            }
        }

        Ok(Self {
            schema_version,
            implementation,
            inputs: inputs.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            outputs,
        })
    }

    /// Returns the recipe schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the provider and recipe semantic version.
    #[must_use]
    pub const fn implementation(&self) -> &ImplementationId {
        &self.implementation
    }

    /// Returns declared external inputs in exact semantic order.
    #[must_use]
    pub const fn inputs(&self) -> &[TensorName] {
        &self.inputs
    }

    /// Returns operations in dependency and execution order.
    #[must_use]
    pub const fn steps(&self) -> &[RecipeStep] {
        &self.steps
    }

    /// Returns consumer-visible outputs and their recipe values.
    #[must_use]
    pub const fn outputs(&self) -> &BTreeMap<TensorName, RecipeInput> {
        &self.outputs
    }

    /// Serializes the recipe to deterministic JSON bytes.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if serialization fails.
    pub fn to_canonical_json(&self) -> Result<Box<[u8]>> {
        canonical_json(self, "serialize conversion recipe")
    }

    /// Returns a domain-separated digest of the canonical recipe.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if canonical serialization fails.
    pub fn digest(&self) -> Result<ContentDigest> {
        let bytes = self.to_canonical_json()?;
        Ok(ContentDigest::hash("conversion-recipe-v1", [bytes]))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ConversionRecipeWire {
    schema_version: u32,
    implementation: ImplementationId,
    inputs: Box<[TensorName]>,
    steps: Box<[RecipeStep]>,
    outputs: BTreeMap<TensorName, RecipeInput>,
}

impl TryFrom<ConversionRecipeWire> for ConversionRecipe {
    type Error = Error;

    fn try_from(wire: ConversionRecipeWire) -> Result<Self> {
        Self::new(
            wire.schema_version,
            wire.implementation,
            wire.inputs.into_vec(),
            wire.steps.into_vec(),
            wire.outputs,
        )
    }
}

/// One normalized inventory tensor available to the planner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SourceTensorWire")]
pub struct SourceTensor {
    name: TensorName,
    shape: Box<[u64]>,
    storage: Storage,
}

impl SourceTensor {
    /// Creates and validates an inventory-facing source descriptor.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format or resource-limit error when plain storage
    /// length disagrees with the shape, or quantized metadata describes a
    /// different logical shape.
    pub fn new(name: TensorName, shape: impl Into<Box<[u64]>>, storage: Storage) -> Result<Self> {
        let shape = shape.into();
        match &storage {
            Storage::Plain { dtype, span } => {
                if dtype.byte_len(&shape)? != span.len() {
                    return Err(Error::invalid(
                        "plain source span length does not match its dtype and shape",
                    ));
                }
            }
            Storage::Quantized(quantized) => {
                if quantized.logical_shape() != shape.as_ref() {
                    return Err(Error::invalid(
                        "quantized source logical shape does not match its inventory shape",
                    ));
                }
            }
        }
        Ok(Self {
            name,
            shape,
            storage,
        })
    }

    /// Returns the exact inventory tensor name.
    #[must_use]
    pub const fn name(&self) -> &TensorName {
        &self.name
    }

    /// Returns the logical source shape.
    #[must_use]
    pub const fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the honest plain or quantized storage descriptor.
    #[must_use]
    pub const fn storage(&self) -> &Storage {
        &self.storage
    }
}

#[derive(Debug, Clone, Deserialize)]
struct SourceTensorWire {
    name: TensorName,
    shape: Box<[u64]>,
    storage: Storage,
}

impl TryFrom<SourceTensorWire> for SourceTensor {
    type Error = Error;

    fn try_from(wire: SourceTensorWire) -> Result<Self> {
        Self::new(wire.name, wire.shape, wire.storage)
    }
}

impl TryFrom<&crate::inventory::TensorRecord> for SourceTensor {
    type Error = Error;

    fn try_from(record: &crate::inventory::TensorRecord) -> Result<Self> {
        Self::new(
            TensorName::parse(record.name())?,
            record.shape(),
            record.storage().clone(),
        )
    }
}

/// Whether a consumer target must be bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Requirement {
    /// Absence rejects the plan.
    Required,
    /// Absence is recorded without rejecting the plan.
    Optional,
}

/// One versioned transform and its exact output and workspace lengths.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlannedTransform {
    transform: TransformSpec,
    output_size: u64,
    scratch_bytes: u64,
}

impl PlannedTransform {
    /// Pins a transform and the bytes it must produce without provider scratch.
    ///
    /// Use [`Self::with_scratch_bytes`] when the provider requires
    /// caller-owned workspace.
    #[must_use]
    pub const fn new(transform: TransformSpec, output_size: u64) -> Self {
        Self {
            transform,
            output_size,
            scratch_bytes: 0,
        }
    }

    /// Pins the exact caller-owned workspace required while this transform runs.
    #[must_use]
    pub const fn with_scratch_bytes(mut self, scratch_bytes: u64) -> Self {
        self.scratch_bytes = scratch_bytes;
        self
    }

    /// Returns the source-to-target transform specification.
    #[must_use]
    pub const fn transform(&self) -> &TransformSpec {
        &self.transform
    }

    /// Returns the exact intermediate or final output byte length.
    #[must_use]
    pub const fn output_size(&self) -> u64 {
        self.output_size
    }

    /// Returns the exact caller-owned workspace length required by the provider.
    #[must_use]
    pub const fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }
}

/// One target constant in a consumer-supplied contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "TargetTensorWire")]
pub struct TargetTensor {
    name: TensorName,
    aliases: Box<[TensorName]>,
    requirement: Requirement,
    source_shape: Box<[u64]>,
    shape: Box<[u64]>,
    storage_shape: Box<[u64]>,
    logical_strides: Box<[u64]>,
    storage_strides: Box<[u64]>,
    representation: Representation,
    transforms: Box<[PlannedTransform]>,
    operation_graph: Option<OperationGraph>,
    quantized_route: Option<RouteCapability>,
    conversion_recipe: Option<ConversionRecipe>,
    output_size: u64,
    shared_source_group: Option<StableName>,
}

impl TargetTensor {
    /// Returns a builder with source and target shapes initially equal.
    pub fn builder(
        name: TensorName,
        requirement: Requirement,
        shape: impl Into<Box<[u64]>>,
        representation: Representation,
        output_size: u64,
    ) -> TargetTensorBuilder {
        let shape = shape.into();
        TargetTensorBuilder {
            name,
            aliases: Vec::new(),
            requirement,
            source_shape: shape.clone(),
            shape,
            storage_shape: None,
            logical_strides: None,
            storage_strides: None,
            representation,
            transforms: Vec::new(),
            operation_graph: None,
            quantized_route: None,
            conversion_recipe: None,
            output_size,
            shared_source_group: None,
        }
    }

    /// Returns the target constant name.
    #[must_use]
    pub const fn name(&self) -> &TensorName {
        &self.name
    }

    /// Returns accepted alternate source names in canonical order.
    #[must_use]
    pub const fn aliases(&self) -> &[TensorName] {
        &self.aliases
    }

    /// Returns whether the target is required.
    #[must_use]
    pub const fn requirement(&self) -> Requirement {
        self.requirement
    }

    /// Returns the source shape expected before conversion.
    #[must_use]
    pub const fn source_shape(&self) -> &[u64] {
        &self.source_shape
    }

    /// Returns the consumer-visible target shape.
    #[must_use]
    pub const fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the dense physical storage shape.
    ///
    /// This can differ from [`Self::shape`] for a storage permutation such as
    /// logical OIHW weights stored physically as OHWI/KYXC.
    #[must_use]
    pub const fn storage_shape(&self) -> &[u64] {
        &self.storage_shape
    }

    /// Returns consumer-visible element strides for the logical tensor view.
    ///
    /// These remain independent of a backend storage permutation. See
    /// [`Self::storage_strides`] for byte-addressing geometry.
    #[must_use]
    pub const fn logical_strides(&self) -> &[u64] {
        &self.logical_strides
    }

    /// Returns physical element strides indexed by logical axis.
    ///
    /// For contiguous tensors these equal [`Self::logical_strides`]. A storage
    /// permutation can retain a consumer's logical strides while changing
    /// these strides to describe where each logical coordinate is stored.
    #[must_use]
    pub const fn storage_strides(&self) -> &[u64] {
        &self.storage_strides
    }

    /// Returns the required target representation.
    #[must_use]
    pub const fn representation(&self) -> &Representation {
        &self.representation
    }

    /// Returns ordered in-process byte-affecting transforms.
    ///
    /// This sequence must be empty when an operation graph or external
    /// conversion recipe is present.
    #[must_use]
    pub const fn transforms(&self) -> &[PlannedTransform] {
        &self.transforms
    }

    /// Returns the typed ordered structural operation graph, if present.
    ///
    /// Graphs replace the legacy unary transform path for grouped inputs
    /// and for transforms whose physical shape differs from logical metadata.
    #[must_use]
    pub const fn operation_graph(&self) -> Option<&OperationGraph> {
        self.operation_graph.as_ref()
    }

    /// Returns the pinned quantized capability, if required.
    #[must_use]
    pub const fn quantized_route(&self) -> Option<&RouteCapability> {
        self.quantized_route.as_ref()
    }

    /// Returns the external provider conversion stage, if any.
    ///
    /// The recipe consumes one selected source and produces the declared target
    /// shape and output representation. Planned transforms, operation graphs,
    /// and quantized routes are mutually exclusive with this stage.
    #[must_use]
    pub const fn conversion_recipe(&self) -> Option<&ConversionRecipe> {
        self.conversion_recipe.as_ref()
    }

    /// Returns the exact planned output byte length.
    #[must_use]
    pub const fn output_size(&self) -> u64 {
        self.output_size
    }

    /// Returns the explicit tied-weight group, if any.
    #[must_use]
    pub const fn shared_source_group(&self) -> Option<&StableName> {
        self.shared_source_group.as_ref()
    }
}

/// Builds a validated target constant descriptor.
#[derive(Debug)]
#[must_use]
pub struct TargetTensorBuilder {
    name: TensorName,
    aliases: Vec<TensorName>,
    requirement: Requirement,
    source_shape: Box<[u64]>,
    shape: Box<[u64]>,
    storage_shape: Option<Box<[u64]>>,
    logical_strides: Option<Box<[u64]>>,
    storage_strides: Option<Box<[u64]>>,
    representation: Representation,
    transforms: Vec<PlannedTransform>,
    operation_graph: Option<OperationGraph>,
    quantized_route: Option<RouteCapability>,
    conversion_recipe: Option<ConversionRecipe>,
    output_size: u64,
    shared_source_group: Option<StableName>,
}

impl TargetTensorBuilder {
    /// Adds accepted alternate source names.
    pub fn aliases(mut self, aliases: impl IntoIterator<Item = TensorName>) -> Self {
        self.aliases.extend(aliases);
        self
    }

    /// Sets the expected pre-conversion source shape.
    pub fn source_shape(mut self, shape: impl Into<Box<[u64]>>) -> Self {
        self.source_shape = shape.into();
        self
    }

    /// Sets the dense physical storage shape.
    ///
    /// The default is the consumer-visible logical shape.
    pub fn storage_shape(mut self, shape: impl Into<Box<[u64]>>) -> Self {
        self.storage_shape = Some(shape.into());
        self
    }

    /// Sets consumer-visible element strides for the logical tensor view.
    ///
    /// The default is contiguous row-major logical strides.
    pub fn logical_strides(mut self, strides: impl Into<Box<[u64]>>) -> Self {
        self.logical_strides = Some(strides.into());
        self
    }

    /// Sets physical element strides indexed by logical axis.
    ///
    /// The default is contiguous row-major logical strides.
    pub fn storage_strides(mut self, strides: impl Into<Box<[u64]>>) -> Self {
        self.storage_strides = Some(strides.into());
        self
    }

    /// Adds ordered in-process preparation transforms.
    ///
    /// A non-empty sequence is rejected when an operation graph or conversion
    /// recipe is attached.
    pub fn transforms(mut self, transforms: impl IntoIterator<Item = PlannedTransform>) -> Self {
        self.transforms.extend(transforms);
        self
    }

    /// Attaches a typed, ordered structural operation graph.
    ///
    /// A graph is a complete source-to-target execution path and therefore
    /// cannot be combined with the legacy unary transform chain, an external
    /// conversion recipe, or a quantized route.
    pub fn operation_graph(mut self, graph: OperationGraph) -> Self {
        self.operation_graph = Some(graph);
        self
    }

    /// Pins a quantized capability selected by consumer policy.
    pub fn quantized_route(mut self, capability: RouteCapability) -> Self {
        self.quantized_route = Some(capability);
        self
    }

    /// Attaches an external provider conversion stage.
    ///
    /// Planned transforms, operation graphs, and quantized routes are rejected
    /// when this recipe is attached.
    pub fn conversion_recipe(mut self, recipe: ConversionRecipe) -> Self {
        self.conversion_recipe = Some(recipe);
        self
    }

    /// Allows exact source sharing only with targets carrying the same group.
    pub fn shared_source_group(mut self, group: StableName) -> Self {
        self.shared_source_group = Some(group);
        self
    }

    /// Validates and builds the target descriptor.
    ///
    /// # Errors
    ///
    /// Returns a binding error for duplicate aliases, a name repeated as its
    /// own alias, incompatible execution stages, an invalid transform chain, an
    /// unexplained shape conversion, or zero output bytes for a non-empty
    /// tensor.
    #[expect(
        clippy::too_many_lines,
        reason = "the builder validates one atomic target contract in a single ordered pass"
    )]
    pub fn build(mut self) -> Result<TargetTensor> {
        let storage_shape = self
            .storage_shape
            .take()
            .unwrap_or_else(|| self.shape.clone());
        let logical_strides = match self.logical_strides.take() {
            Some(strides) => strides,
            None => checked_contiguous_strides(&self.shape, "target tensor")?,
        };
        let storage_strides = match self.storage_strides.take() {
            Some(strides) => strides,
            None => checked_contiguous_strides(&self.shape, "target tensor storage")?,
        };
        self.aliases.sort_unstable();
        if self.aliases.windows(2).any(|window| window[0] == window[1]) {
            return Err(Error::binding(
                "target tensor contains a duplicate source alias",
            ));
        }
        if self.aliases.binary_search(&self.name).is_ok() {
            return Err(Error::binding(
                "target tensor repeats its exact name as a source alias",
            ));
        }
        if self.conversion_recipe.is_some() && !self.transforms.is_empty() {
            return Err(Error::binding(
                "target cannot combine a conversion recipe with planned transforms",
            ));
        }
        if self.operation_graph.is_some()
            && (!self.transforms.is_empty()
                || self.conversion_recipe.is_some()
                || self.quantized_route.is_some())
        {
            return Err(Error::binding(
                "target operation graph cannot be combined with another execution route",
            ));
        }
        if self.operation_graph.is_some() && !self.aliases.is_empty() {
            return Err(Error::binding(
                "operation graph inputs replace target source aliases",
            ));
        }
        validate_transform_chain(
            &self.transforms,
            &self.representation,
            &self.shape,
            self.output_size,
        )?;
        if self.source_shape != self.shape
            && self.conversion_recipe.is_none()
            && self.operation_graph.is_none()
            && self.quantized_route.is_none()
        {
            return Err(Error::binding(
                "target source shape differs without a conversion recipe, operation graph, or quantized route",
            ));
        }
        if logical_strides.len() != self.shape.len() {
            return Err(Error::binding(
                "target logical stride rank differs from its logical shape rank",
            ));
        }
        if storage_strides.len() != self.shape.len() {
            return Err(Error::binding(
                "target storage stride rank differs from its logical shape rank",
            ));
        }
        let elements = checked_elements(&self.shape, "target tensor")?;
        if elements > 0 && self.output_size == 0 {
            return Err(Error::binding(
                "non-empty target tensor must declare nonzero output bytes",
            ));
        }
        if let Some(recipe) = &self.conversion_recipe {
            if self.quantized_route.is_some() {
                return Err(Error::binding(
                    "target cannot combine a conversion recipe with a quantized route",
                ));
            }
            if recipe.inputs().len() != 1 {
                return Err(Error::binding(
                    "external conversion recipes currently require exactly one input",
                ));
            }
            if !recipe.outputs().contains_key(&self.name) {
                return Err(Error::binding(
                    "conversion recipe does not declare the target tensor name as an output",
                ));
            }
        }
        if let Some(graph) = &self.operation_graph {
            let target_facts = TensorFacts::new(
                self.shape.clone(),
                storage_shape.clone(),
                logical_strides.clone(),
                storage_strides.clone(),
                self.representation.clone(),
                self.output_size,
            )?;
            if graph.output_facts() != &target_facts {
                return Err(Error::binding(
                    "operation graph output facts differ from the target tensor contract",
                ));
            }
        }
        Ok(TargetTensor {
            name: self.name,
            aliases: self.aliases.into_boxed_slice(),
            requirement: self.requirement,
            source_shape: self.source_shape,
            shape: self.shape,
            storage_shape,
            logical_strides,
            storage_strides,
            representation: self.representation,
            transforms: self.transforms.into_boxed_slice(),
            operation_graph: self.operation_graph,
            quantized_route: self.quantized_route,
            conversion_recipe: self.conversion_recipe,
            output_size: self.output_size,
            shared_source_group: self.shared_source_group,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TargetTensorWire {
    name: TensorName,
    aliases: Box<[TensorName]>,
    requirement: Requirement,
    source_shape: Box<[u64]>,
    shape: Box<[u64]>,
    storage_shape: Box<[u64]>,
    logical_strides: Box<[u64]>,
    storage_strides: Box<[u64]>,
    representation: Representation,
    transforms: Box<[PlannedTransform]>,
    operation_graph: Option<OperationGraph>,
    quantized_route: Option<RouteCapability>,
    conversion_recipe: Option<ConversionRecipe>,
    output_size: u64,
    shared_source_group: Option<StableName>,
}

impl TryFrom<TargetTensorWire> for TargetTensor {
    type Error = Error;

    fn try_from(wire: TargetTensorWire) -> Result<Self> {
        let mut builder = Self::builder(
            wire.name,
            wire.requirement,
            wire.shape,
            wire.representation,
            wire.output_size,
        )
        .aliases(wire.aliases)
        .source_shape(wire.source_shape)
        .storage_shape(wire.storage_shape)
        .logical_strides(wire.logical_strides)
        .storage_strides(wire.storage_strides)
        .transforms(wire.transforms);
        if let Some(graph) = wire.operation_graph {
            builder = builder.operation_graph(graph);
        }
        if let Some(capability) = wire.quantized_route {
            builder = builder.quantized_route(capability);
        }
        if let Some(recipe) = wire.conversion_recipe {
            builder = builder.conversion_recipe(recipe);
        }
        if let Some(group) = wire.shared_source_group {
            builder = builder.shared_source_group(group);
        }
        builder.build()
    }
}

/// Complete immutable identities that influence a binding plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanInputs {
    manifest: ManifestId,
    selection: SelectionId,
    contract: ContractId,
    backend: BackendId,
    source_digests: Box<[ContentDigest]>,
}

impl PlanInputs {
    /// Creates complete plan identity inputs.
    #[must_use]
    pub fn new(
        manifest: ManifestId,
        selection: SelectionId,
        contract: ContractId,
        backend: BackendId,
        source_digests: impl Into<Box<[ContentDigest]>>,
    ) -> Self {
        Self {
            manifest,
            selection,
            contract,
            backend,
            source_digests: source_digests.into(),
        }
    }

    /// Returns the normalized checkpoint/configuration manifest identity.
    #[must_use]
    pub const fn manifest(&self) -> ManifestId {
        self.manifest
    }

    /// Returns normalized selection facts.
    #[must_use]
    pub const fn selection(&self) -> SelectionId {
        self.selection
    }

    /// Returns the consumer target-contract identity.
    #[must_use]
    pub const fn contract(&self) -> ContractId {
        self.contract
    }

    /// Returns the backend and layout ABI identity.
    #[must_use]
    pub const fn backend(&self) -> BackendId {
        self.backend
    }

    /// Returns source digests in file-ordinal order.
    #[must_use]
    pub const fn source_digests(&self) -> &[ContentDigest] {
        &self.source_digests
    }
}

/// Policy for inventory tensors not consumed by the target contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExtraSourcePolicy {
    /// Report extras as errors.
    Reject,
    /// Record extras in the plan without rejecting it.
    Allow,
}

/// Stable severity for a planning diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DiagnosticSeverity {
    /// Informational contract state.
    Info,
    /// A non-fatal condition retained in a valid plan.
    Warning,
    /// A condition that rejects plan construction.
    Error,
}

/// Stable machine-readable class of planning mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DiagnosticCode {
    /// More than one inventory descriptor has the same name.
    DuplicateSource,
    /// More than one contract descriptor has the same name.
    DuplicateTarget,
    /// No source matched a required target.
    MissingRequired,
    /// No source matched an optional target.
    MissingOptional,
    /// More than one source matched an exact name or alias.
    AmbiguousBinding,
    /// A source shape did not satisfy the target contract.
    ShapeMismatch,
    /// A source file ordinal has no corresponding ordered digest.
    MissingSourceDigest,
    /// A source was tied without one common explicit sharing group.
    ConflictingSourceReuse,
    /// Plain storage and requested transforms are incompatible.
    UnsupportedTransform,
    /// Quantized storage and its selected capability are incompatible.
    UnsupportedQuantizedRoute,
    /// An inventory source was not selected.
    ExtraSource,
}

impl DiagnosticCode {
    /// Returns a stable snake-case code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DuplicateSource => "duplicate_source",
            Self::DuplicateTarget => "duplicate_target",
            Self::MissingRequired => "missing_required",
            Self::MissingOptional => "missing_optional",
            Self::AmbiguousBinding => "ambiguous_binding",
            Self::ShapeMismatch => "shape_mismatch",
            Self::MissingSourceDigest => "missing_source_digest",
            Self::ConflictingSourceReuse => "conflicting_source_reuse",
            Self::UnsupportedTransform => "unsupported_transform",
            Self::UnsupportedQuantizedRoute => "unsupported_quantized_route",
            Self::ExtraSource => "extra_source",
        }
    }
}

/// One deterministic, actionable planning diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlanDiagnostic {
    severity: DiagnosticSeverity,
    code: DiagnosticCode,
    subject: TensorName,
    related: Box<[TensorName]>,
    message: Box<str>,
}

impl PlanDiagnostic {
    fn new(
        severity: DiagnosticSeverity,
        code: DiagnosticCode,
        subject: TensorName,
        mut related: Vec<TensorName>,
        message: impl Into<Box<str>>,
    ) -> Self {
        related.sort_unstable();
        related.dedup();
        Self {
            severity,
            code,
            subject,
            related: related.into_boxed_slice(),
            message: message.into(),
        }
    }

    /// Returns diagnostic severity.
    #[must_use]
    pub const fn severity(&self) -> DiagnosticSeverity {
        self.severity
    }

    /// Returns the stable diagnostic code.
    #[must_use]
    pub const fn code(&self) -> DiagnosticCode {
        self.code
    }

    /// Returns the primary target or source name.
    #[must_use]
    pub const fn subject(&self) -> &TensorName {
        &self.subject
    }

    /// Returns other deterministically ordered names involved.
    #[must_use]
    pub const fn related(&self) -> &[TensorName] {
        &self.related
    }

    /// Returns the actionable detail.
    #[must_use]
    pub const fn message(&self) -> &str {
        &self.message
    }
}

impl Display for PlanDiagnostic {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}[{}]: {}",
            self.code.as_str(),
            self.subject,
            self.message
        )
    }
}

/// Deterministic diagnostics produced without allocating or materializing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanAnalysis {
    diagnostics: Box<[PlanDiagnostic]>,
}

impl PlanAnalysis {
    /// Returns diagnostics in canonical order.
    #[must_use]
    pub const fn diagnostics(&self) -> &[PlanDiagnostic] {
        &self.diagnostics
    }

    /// Returns whether no error-severity diagnostic was found.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    }
}

/// One exact ordered source-set-to-target binding in a validated plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "BindingWire")]
pub struct Binding {
    sources: Box<[SourceTensor]>,
    target: TargetTensor,
}

impl Binding {
    fn new(sources: impl Into<Box<[SourceTensor]>>, target: TargetTensor) -> Result<Self> {
        let sources = sources.into();
        if sources.is_empty() {
            return Err(Error::binding(
                "a target binding must contain at least one ordered source",
            ));
        }
        Ok(Self { sources, target })
    }

    /// Returns the first selected source descriptor and exact span.
    ///
    /// Existing one-source consumers can continue to use this accessor.
    /// Group-aware consumers should use [`Self::sources`] so every semantic
    /// input and its order remain visible.
    ///
    /// # Panics
    ///
    /// Panics only if the validated binding invariant is violated and the
    /// ordered source list is empty.
    #[must_use]
    pub fn source(&self) -> &SourceTensor {
        self.sources
            .first()
            .expect("validated bindings always contain at least one source")
    }

    /// Returns all selected source descriptors in semantic graph-input order.
    #[must_use]
    pub const fn sources(&self) -> &[SourceTensor] {
        &self.sources
    }

    /// Returns the complete target representation and recipes.
    #[must_use]
    pub const fn target(&self) -> &TargetTensor {
        &self.target
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BindingWire {
    sources: Box<[SourceTensor]>,
    target: TargetTensor,
}

impl TryFrom<BindingWire> for Binding {
    type Error = Error;

    fn try_from(wire: BindingWire) -> Result<Self> {
        Self::new(wire.sources, wire.target)
    }
}

/// A canonical, content-addressed binding plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BindingPlan {
    schema_version: u32,
    id: PlanId,
    inputs: PlanInputs,
    sources: Box<[SourceTensor]>,
    targets: Box<[TargetTensor]>,
    extra_source_policy: ExtraSourcePolicy,
    bindings: Box<[Binding]>,
    missing_optional: Box<[TensorName]>,
    unused_sources: Box<[TensorName]>,
}

impl BindingPlan {
    /// Returns a builder for immutable plan identity inputs.
    pub const fn builder(inputs: PlanInputs) -> PlanBuilder {
        PlanBuilder {
            inputs,
            sources: Vec::new(),
            targets: Vec::new(),
            extra_source_policy: ExtraSourcePolicy::Reject,
        }
    }

    /// Parses and verifies canonical JSON plan bytes.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format or integrity error for non-canonical JSON, an
    /// unsupported schema version, invalid ordering, semantic mismatches, or a
    /// plan identity that does not match its content.
    pub fn from_canonical_json(bytes: &[u8]) -> Result<Self> {
        let wire: BindingPlanWire = serde_json::from_slice(bytes).map_err(|error| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "parse binding plan JSON",
                error,
            )
        })?;
        let canonical = canonical_json(&wire, "serialize parsed binding plan")?;
        if canonical.as_ref() != bytes {
            return Err(Error::invalid("binding plan JSON is not canonical"));
        }
        if wire.schema_version != PLAN_SCHEMA_VERSION {
            return Err(Error::invalid("unsupported binding plan schema version"));
        }
        validate_sorted_unique(
            wire.sources.iter().map(SourceTensor::name),
            "binding plan planning sources are not sorted and unique",
        )?;
        validate_sorted_unique(
            wire.targets.iter().map(TargetTensor::name),
            "binding plan planning targets are not sorted and unique",
        )?;
        validate_sorted_unique(
            wire.bindings.iter().map(|binding| binding.target.name()),
            "binding plan targets are not sorted and unique",
        )?;
        validate_sorted_unique(
            wire.missing_optional.iter(),
            "binding plan optional misses are not sorted and unique",
        )?;
        validate_sorted_unique(
            wire.unused_sources.iter(),
            "binding plan unused sources are not sorted and unique",
        )?;

        let rebuilt = Self::builder(wire.inputs.clone())
            .sources(wire.sources.iter().cloned())
            .targets(wire.targets.iter().cloned())
            .extra_source_policy(wire.extra_source_policy)
            .build()?;
        if rebuilt.bindings != wire.bindings {
            return Err(Error::binding(
                "serialized bindings differ from deterministic planning",
            ));
        }
        if rebuilt.missing_optional != wire.missing_optional {
            return Err(Error::binding(
                "serialized optional misses differ from deterministic planning",
            ));
        }
        if rebuilt.unused_sources != wire.unused_sources {
            return Err(Error::binding(
                "serialized unused sources differ from deterministic planning",
            ));
        }
        if rebuilt.id != wire.id {
            return Err(Error::integrity(
                "binding plan identity does not match its canonical content",
            ));
        }
        Ok(rebuilt)
    }

    /// Returns the canonical schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the content-addressed plan identity.
    #[must_use]
    pub const fn id(&self) -> PlanId {
        self.id
    }

    /// Returns all immutable cache-key inputs.
    #[must_use]
    pub const fn inputs(&self) -> &PlanInputs {
        &self.inputs
    }

    /// Returns the complete normalized inventory evidence in canonical order.
    #[must_use]
    pub const fn sources(&self) -> &[SourceTensor] {
        &self.sources
    }

    /// Returns the complete consumer target contract in canonical order.
    #[must_use]
    pub const fn targets(&self) -> &[TargetTensor] {
        &self.targets
    }

    /// Returns the policy used to classify unconsumed inventory tensors.
    #[must_use]
    pub const fn extra_source_policy(&self) -> ExtraSourcePolicy {
        self.extra_source_policy
    }

    /// Returns bindings sorted by target name.
    #[must_use]
    pub const fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// Returns unbound optional targets in canonical order.
    #[must_use]
    pub const fn missing_optional(&self) -> &[TensorName] {
        &self.missing_optional
    }

    /// Returns allowed, unused source names in canonical order.
    #[must_use]
    pub const fn unused_sources(&self) -> &[TensorName] {
        &self.unused_sources
    }

    /// Serializes the complete plan to deterministic JSON bytes.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error if serialization fails.
    pub fn to_canonical_json(&self) -> Result<Box<[u8]>> {
        canonical_json(self, "serialize binding plan")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BindingPlanWire {
    schema_version: u32,
    id: PlanId,
    inputs: PlanInputs,
    sources: Box<[SourceTensor]>,
    targets: Box<[TargetTensor]>,
    extra_source_policy: ExtraSourcePolicy,
    bindings: Box<[Binding]>,
    missing_optional: Box<[TensorName]>,
    unused_sources: Box<[TensorName]>,
}

/// Builds and analyzes deterministic binding plans.
#[derive(Debug)]
#[must_use]
pub struct PlanBuilder {
    inputs: PlanInputs,
    sources: Vec<SourceTensor>,
    targets: Vec<TargetTensor>,
    extra_source_policy: ExtraSourcePolicy,
}

impl PlanBuilder {
    /// Adds normalized inventory tensors.
    pub fn sources(mut self, sources: impl IntoIterator<Item = SourceTensor>) -> Self {
        self.sources.extend(sources);
        self
    }

    /// Adds consumer contract targets.
    pub fn targets(mut self, targets: impl IntoIterator<Item = TargetTensor>) -> Self {
        self.targets.extend(targets);
        self
    }

    /// Sets how unused inventory tensors are handled.
    pub const fn extra_source_policy(mut self, policy: ExtraSourcePolicy) -> Self {
        self.extra_source_policy = policy;
        self
    }

    /// Computes every deterministic mismatch without constructing a plan.
    #[must_use]
    pub fn analyze(&self) -> PlanAnalysis {
        let resolution = resolve_bindings(
            &self.inputs,
            &self.sources,
            &self.targets,
            self.extra_source_policy,
        );
        PlanAnalysis {
            diagnostics: resolution.diagnostics.into_boxed_slice(),
        }
    }

    /// Resolves and constructs a canonical binding plan.
    ///
    /// # Errors
    ///
    /// Returns a binding error containing stable, ordered diagnostics when a
    /// required, ambiguous, duplicate, shape, sharing, route, transform,
    /// digest, or extra-source mismatch exists.
    pub fn build(self) -> Result<BindingPlan> {
        let Self {
            inputs,
            mut sources,
            mut targets,
            extra_source_policy,
        } = self;
        sources.sort_unstable_by(|left, right| left.name().cmp(right.name()));
        targets.sort_unstable_by(|left, right| left.name().cmp(right.name()));

        let resolution = resolve_bindings(&inputs, &sources, &targets, extra_source_policy);
        if resolution
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
        {
            let details = resolution
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(Error::binding(format!("binding plan rejected: {details}")));
        }

        let mut bindings = resolution
            .matches
            .into_iter()
            .map(|(target_index, source_indices)| {
                Binding::new(
                    source_indices
                        .iter()
                        .map(|source_index| sources[*source_index].clone())
                        .collect::<Vec<_>>(),
                    targets[target_index].clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        bindings.sort_unstable_by(|left, right| left.target.name().cmp(right.target.name()));

        let mut missing_optional = targets
            .iter()
            .filter(|target| {
                target.requirement == Requirement::Optional
                    && !bindings
                        .iter()
                        .any(|binding| binding.target.name() == target.name())
            })
            .map(|target| target.name.clone())
            .collect::<Vec<_>>();
        missing_optional.sort_unstable();
        missing_optional.dedup();

        let mut unused_sources = resolution
            .unused_sources
            .into_iter()
            .map(|index| sources[index].name.clone())
            .collect::<Vec<_>>();
        unused_sources.sort_unstable();
        unused_sources.dedup();

        let bindings = bindings.into_boxed_slice();
        let missing_optional = missing_optional.into_boxed_slice();
        let unused_sources = unused_sources.into_boxed_slice();
        let sources = sources.into_boxed_slice();
        let targets = targets.into_boxed_slice();
        let id = compute_plan_id(&PlanIdentityPayload {
            schema_version: PLAN_SCHEMA_VERSION,
            inputs: &inputs,
            sources: &sources,
            targets: &targets,
            extra_source_policy,
            bindings: &bindings,
            missing_optional: &missing_optional,
            unused_sources: &unused_sources,
        })?;
        Ok(BindingPlan {
            schema_version: PLAN_SCHEMA_VERSION,
            id,
            inputs,
            sources,
            targets,
            extra_source_policy,
            bindings,
            missing_optional,
            unused_sources,
        })
    }
}

#[derive(Debug)]
struct Resolution {
    matches: BTreeMap<usize, Box<[usize]>>,
    unused_sources: BTreeSet<usize>,
    diagnostics: Vec<PlanDiagnostic>,
}

#[expect(
    clippy::too_many_lines,
    reason = "the single pass keeps diagnostic ordering and source-use accounting together"
)]
fn resolve_bindings(
    inputs: &PlanInputs,
    sources: &[SourceTensor],
    targets: &[TargetTensor],
    extra_source_policy: ExtraSourcePolicy,
) -> Resolution {
    let source_names = group_names(
        sources
            .iter()
            .enumerate()
            .map(|(index, source)| (source.name().clone(), index)),
    );
    let target_names = group_names(
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| (target.name().clone(), index)),
    );
    let mut diagnostics = Vec::new();

    for (name, indices) in &source_names {
        if indices.len() > 1 {
            diagnostics.push(PlanDiagnostic::new(
                DiagnosticSeverity::Error,
                DiagnosticCode::DuplicateSource,
                name.clone(),
                Vec::new(),
                "inventory contains more than one tensor with this name",
            ));
        }
    }
    for (name, indices) in &target_names {
        if indices.len() > 1 {
            diagnostics.push(PlanDiagnostic::new(
                DiagnosticSeverity::Error,
                DiagnosticCode::DuplicateTarget,
                name.clone(),
                Vec::new(),
                "contract contains more than one target with this name",
            ));
        }
    }

    let mut target_order = (0..targets.len()).collect::<Vec<_>>();
    target_order.sort_unstable_by(|left, right| targets[*left].name.cmp(&targets[*right].name));
    let mut matches = BTreeMap::new();
    for target_index in target_order {
        let target = &targets[target_index];
        if target_names
            .get(target.name())
            .is_some_and(|indices| indices.len() > 1)
        {
            continue;
        }
        if target.operation_graph().is_some() {
            resolve_operation_graph_binding(
                inputs,
                sources,
                &source_names,
                target,
                target_index,
                &mut matches,
                &mut diagnostics,
            );
            continue;
        }
        let mut candidates = BTreeSet::<usize>::new();
        if let Some(indices) = source_names.get(target.name()) {
            candidates.extend(indices);
        }
        for alias in target.aliases() {
            if let Some(indices) = source_names.get(alias) {
                candidates.extend(indices);
            }
        }

        match candidates.len() {
            0 => {
                let (severity, code, message) = match target.requirement {
                    Requirement::Required => (
                        DiagnosticSeverity::Error,
                        DiagnosticCode::MissingRequired,
                        "no inventory tensor matched the required target or its aliases",
                    ),
                    Requirement::Optional => (
                        DiagnosticSeverity::Info,
                        DiagnosticCode::MissingOptional,
                        "no inventory tensor matched the optional target or its aliases",
                    ),
                };
                diagnostics.push(PlanDiagnostic::new(
                    severity,
                    code,
                    target.name.clone(),
                    target.aliases.to_vec(),
                    message,
                ));
            }
            1 => {
                if let Some(source_index) = candidates.first().copied() {
                    let source = &sources[source_index];
                    if source.shape() != target.source_shape() {
                        diagnostics.push(PlanDiagnostic::new(
                            DiagnosticSeverity::Error,
                            DiagnosticCode::ShapeMismatch,
                            target.name.clone(),
                            vec![source.name.clone()],
                            format!(
                                "source shape {:?} does not match expected shape {:?}",
                                source.shape(),
                                target.source_shape()
                            ),
                        ));
                    }
                    if first_missing_digest_ordinal(source.storage(), inputs.source_digests.len())
                        .is_some()
                    {
                        diagnostics.push(PlanDiagnostic::new(
                            DiagnosticSeverity::Error,
                            DiagnosticCode::MissingSourceDigest,
                            source.name.clone(),
                            vec![target.name.clone()],
                            "source or quantized companion file ordinal has no ordered content digest",
                        ));
                    }
                    if let Err(message) = validate_binding(inputs, source, target) {
                        let code = match source.storage() {
                            Storage::Plain { .. } => DiagnosticCode::UnsupportedTransform,
                            Storage::Quantized(_) => DiagnosticCode::UnsupportedQuantizedRoute,
                        };
                        diagnostics.push(PlanDiagnostic::new(
                            DiagnosticSeverity::Error,
                            code,
                            target.name.clone(),
                            vec![source.name.clone()],
                            message.message(),
                        ));
                    }
                    matches.insert(target_index, vec![source_index].into_boxed_slice());
                }
            }
            _ => {
                let related = candidates
                    .iter()
                    .map(|source_index| sources[*source_index].name.clone())
                    .collect();
                diagnostics.push(PlanDiagnostic::new(
                    DiagnosticSeverity::Error,
                    DiagnosticCode::AmbiguousBinding,
                    target.name.clone(),
                    related,
                    "more than one inventory tensor matched the target name or aliases",
                ));
            }
        }
    }

    let mut reused = BTreeMap::<usize, Vec<usize>>::new();
    for (target_index, source_indices) in &matches {
        for source_index in source_indices.iter().copied().collect::<BTreeSet<_>>() {
            reused.entry(source_index).or_default().push(*target_index);
        }
    }
    for (source_index, target_indices) in reused {
        if target_indices.len() < 2 {
            continue;
        }
        let group = targets[target_indices[0]].shared_source_group();
        let valid = group.is_some()
            && target_indices
                .iter()
                .all(|index| targets[*index].shared_source_group() == group);
        if !valid {
            let related = target_indices
                .iter()
                .map(|index| targets[*index].name.clone())
                .collect::<Vec<_>>();
            for target_index in target_indices {
                diagnostics.push(PlanDiagnostic::new(
                    DiagnosticSeverity::Error,
                    DiagnosticCode::ConflictingSourceReuse,
                    targets[target_index].name.clone(),
                    related.clone(),
                    format!(
                        "source {} is shared without one explicit tied-weight group",
                        sources[source_index].name()
                    ),
                ));
            }
        }
    }

    let used_sources = matches
        .values()
        .flat_map(|source_indices| source_indices.iter().copied())
        .collect::<BTreeSet<_>>();
    let unused_sources = (0..sources.len())
        .filter(|index| !used_sources.contains(index))
        .collect::<BTreeSet<_>>();
    let severity = match extra_source_policy {
        ExtraSourcePolicy::Reject => DiagnosticSeverity::Error,
        ExtraSourcePolicy::Allow => DiagnosticSeverity::Warning,
    };
    for source_index in &unused_sources {
        diagnostics.push(PlanDiagnostic::new(
            severity,
            DiagnosticCode::ExtraSource,
            sources[*source_index].name.clone(),
            Vec::new(),
            "inventory tensor was not selected by the target contract",
        ));
    }

    diagnostics.sort_unstable();
    diagnostics.dedup();
    Resolution {
        matches,
        unused_sources,
        diagnostics,
    }
}

fn resolve_operation_graph_binding(
    inputs: &PlanInputs,
    sources: &[SourceTensor],
    source_names: &BTreeMap<TensorName, Vec<usize>>,
    target: &TargetTensor,
    target_index: usize,
    matches: &mut BTreeMap<usize, Box<[usize]>>,
    diagnostics: &mut Vec<PlanDiagnostic>,
) {
    let Some(graph) = target.operation_graph() else {
        diagnostics.push(PlanDiagnostic::new(
            DiagnosticSeverity::Error,
            DiagnosticCode::UnsupportedTransform,
            target.name.clone(),
            Vec::new(),
            "operation-graph resolver received a target without a graph",
        ));
        return;
    };
    let mut source_indices = Vec::new();
    if source_indices
        .try_reserve_exact(graph.inputs().len())
        .is_err()
    {
        diagnostics.push(PlanDiagnostic::new(
            DiagnosticSeverity::Error,
            DiagnosticCode::UnsupportedTransform,
            target.name.clone(),
            Vec::new(),
            "could not allocate ordered operation-graph source metadata",
        ));
        return;
    }
    let mut missing = Vec::new();
    let mut ambiguous = Vec::new();
    for input in graph.inputs() {
        match source_names.get(input.name()).map(Vec::as_slice) {
            None | Some([]) => missing.push(input.name().clone()),
            Some([source_index]) => source_indices.push(*source_index),
            Some(indices) => {
                ambiguous.extend(
                    indices
                        .iter()
                        .map(|source_index| sources[*source_index].name.clone()),
                );
            }
        }
    }
    if !ambiguous.is_empty() {
        diagnostics.push(PlanDiagnostic::new(
            DiagnosticSeverity::Error,
            DiagnosticCode::AmbiguousBinding,
            target.name.clone(),
            ambiguous,
            "an operation-graph input matched more than one inventory tensor",
        ));
        return;
    }
    if !missing.is_empty() {
        let (severity, code, message) = match target.requirement {
            Requirement::Required => (
                DiagnosticSeverity::Error,
                DiagnosticCode::MissingRequired,
                "one or more required operation-graph inputs are absent",
            ),
            Requirement::Optional => (
                DiagnosticSeverity::Info,
                DiagnosticCode::MissingOptional,
                "one or more optional operation-graph inputs are absent",
            ),
        };
        diagnostics.push(PlanDiagnostic::new(
            severity,
            code,
            target.name.clone(),
            missing,
            message,
        ));
        return;
    }
    let selected = source_indices
        .iter()
        .map(|source_index| &sources[*source_index])
        .collect::<Vec<_>>();
    if let Err(error) = validate_operation_binding(inputs, &selected, target) {
        diagnostics.push(PlanDiagnostic::new(
            DiagnosticSeverity::Error,
            DiagnosticCode::UnsupportedTransform,
            target.name.clone(),
            selected
                .iter()
                .map(|source| source.name().clone())
                .collect(),
            error.message(),
        ));
    }
    matches.insert(target_index, source_indices.into_boxed_slice());
}

fn validate_operation_binding(
    inputs: &PlanInputs,
    sources: &[&SourceTensor],
    target: &TargetTensor,
) -> Result<()> {
    let graph = target
        .operation_graph()
        .ok_or_else(|| Error::binding("target has no operation graph"))?;
    if graph.inputs().len() != sources.len() {
        return Err(Error::binding(
            "operation graph source count differs from its binding",
        ));
    }
    for (input, source) in graph.inputs().iter().zip(sources) {
        if input.name() != source.name() {
            return Err(Error::binding(
                "operation graph source order differs from its declared inputs",
            ));
        }
        if first_missing_digest_ordinal(source.storage(), inputs.source_digests.len()).is_some() {
            return Err(Error::binding(
                "operation graph source file ordinal has no ordered content digest",
            ));
        }
        let Storage::Plain { dtype, span } = source.storage() else {
            return Err(Error::unsupported(
                "operation graphs do not implicitly decode quantized inputs",
            ));
        };
        let actual = TensorFacts::contiguous(source.shape(), Representation::contiguous(*dtype))?;
        if input.facts() != &actual || input.facts().byte_len() != span.len() {
            return Err(Error::binding(
                "operation graph input facts differ from the inventory source",
            ));
        }
    }
    Ok(())
}

fn validate_binding(
    inputs: &PlanInputs,
    source: &SourceTensor,
    target: &TargetTensor,
) -> Result<()> {
    if source.shape() != target.source_shape() {
        return Err(Error::binding(
            "source shape does not match the target's expected source shape",
        ));
    }
    if first_missing_digest_ordinal(source.storage(), inputs.source_digests.len()).is_some() {
        return Err(Error::binding(
            "source or quantized companion file ordinal has no ordered content digest",
        ));
    }
    validate_conversion_binding(source, target)?;
    match source.storage() {
        Storage::Plain { dtype, span } => {
            if target.quantized_route().is_some() {
                return Err(Error::unsupported(
                    "plain source cannot use a quantized route",
                ));
            }
            if target.conversion_recipe().is_some() {
                validate_contiguous_output(target)?;
                return Ok(());
            }
            let source_representation = Representation::contiguous(*dtype);
            if let Some(first) = target.transforms().first() {
                if first.transform().source() != &source_representation {
                    return Err(Error::unsupported(
                        "first transform does not accept the plain source representation",
                    ));
                }
            } else if target.representation() != &source_representation {
                return Err(Error::unsupported(
                    "plain source requires a transform for the target representation",
                ));
            }
            if target.transforms().is_empty() && target.output_size() != span.len() {
                return Err(Error::binding(
                    "identity binding output size differs from the source span",
                ));
            }
            validate_contiguous_output(target)?;
        }
        Storage::Quantized(storage) => {
            if !target.transforms().is_empty() {
                return Err(Error::unsupported(
                    "quantized bindings use an explicit route instead of host transform specs",
                ));
            }
            let capability = target.quantized_route().ok_or_else(|| {
                Error::unsupported("quantized source is missing a selected capability")
            })?;
            if capability.source_encoding() != storage.encoding() {
                return Err(Error::unsupported(
                    "quantized capability does not accept the source encoding",
                ));
            }
            if capability
                .backend()
                .is_some_and(|backend| backend != inputs.backend)
            {
                return Err(Error::unsupported(
                    "quantized capability backend differs from the plan backend",
                ));
            }
            if !layout_matches(target.representation().layout(), capability.target_layout()) {
                return Err(Error::unsupported(
                    "quantized capability target layout ABI differs from the target representation",
                ));
            }
            match capability.route() {
                QuantizedRoute::HostDequant { target_dtype }
                | QuantizedRoute::DeviceDequantToScratch { target_dtype } => {
                    if *target_dtype != target.representation().dtype() {
                        return Err(Error::unsupported(
                            "quantized dequant dtype differs from the target representation",
                        ));
                    }
                    validate_contiguous_output(target)?;
                }
                QuantizedRoute::PackedDirect => {
                    if target.output_size() != storage.span().len() {
                        return Err(Error::binding(
                            "packed-direct output size differs from the source payload span",
                        ));
                    }
                }
                QuantizedRoute::FusedInTile | QuantizedRoute::Repack { .. } => {}
            }
        }
    }
    Ok(())
}

fn validate_conversion_binding(source: &SourceTensor, target: &TargetTensor) -> Result<()> {
    let Some(recipe) = target.conversion_recipe() else {
        return Ok(());
    };
    if !target.transforms().is_empty() {
        return Err(Error::binding(
            "target cannot combine a conversion recipe with planned transforms",
        ));
    }
    if recipe.inputs().len() != 1 || recipe.inputs().first() != Some(source.name()) {
        return Err(Error::binding(
            "conversion recipe external input does not match the selected source tensor",
        ));
    }
    if !recipe.outputs().contains_key(target.name()) {
        return Err(Error::binding(
            "conversion recipe does not produce the bound target tensor",
        ));
    }
    Ok(())
}

fn validate_transform_chain(
    transforms: &[PlannedTransform],
    target: &Representation,
    shape: &[u64],
    final_output_size: u64,
) -> Result<()> {
    for pair in transforms.windows(2) {
        if pair[0].transform().target() != pair[1].transform().source() {
            return Err(Error::binding(
                "target transform chain contains incompatible adjacent representations",
            ));
        }
    }
    if transforms.last().is_some_and(|planned| {
        planned.transform().target() != target || planned.output_size() != final_output_size
    }) {
        return Err(Error::binding(
            "terminal transform representation or output size differs from the target",
        ));
    }
    let elements = checked_elements(shape, "target transform")?;
    for planned in transforms {
        if elements > 0 && planned.output_size() == 0 {
            return Err(Error::binding(
                "non-empty transform step must declare nonzero output bytes",
            ));
        }
        if planned.transform().target().layout().is_contiguous()
            && planned.transform().target().dtype().byte_len(shape)? != planned.output_size()
        {
            return Err(Error::binding(
                "contiguous transform step output size differs from its dtype and shape",
            ));
        }
    }
    Ok(())
}

fn validate_contiguous_output(target: &TargetTensor) -> Result<()> {
    if target.representation().layout().is_contiguous() {
        let expected_strides = checked_contiguous_strides(target.shape(), "contiguous target")?;
        if target.storage_shape() != target.shape()
            || target.logical_strides() != expected_strides.as_ref()
            || target.storage_strides() != expected_strides.as_ref()
        {
            return Err(Error::binding(
                "contiguous target physical geometry differs from its logical shape",
            ));
        }
        let expected = target.representation().dtype().byte_len(target.shape())?;
        if expected != target.output_size() {
            return Err(Error::binding(
                "contiguous target output size differs from its dtype and shape",
            ));
        }
    }
    Ok(())
}

fn layout_matches(
    layout: &Layout,
    requirement: Option<&crate::quantization::LayoutRequirement>,
) -> bool {
    match (layout.custom_parts(), requirement) {
        (None, None) => true,
        (Some(_), Some(requirement)) => requirement.matches_layout(layout),
        (None, Some(_)) | (Some(_), None) => false,
    }
}

fn checked_elements(shape: &[u64], description: &str) -> Result<u64> {
    shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| Error::limit(format!("{description} element count overflows u64")))
    })
}

fn checked_contiguous_strides(shape: &[u64], description: &str) -> Result<Box<[u64]>> {
    let mut strides = vec![0_u64; shape.len()];
    let mut stride = 1_u64;
    for (dimension, output) in shape.iter().zip(&mut strides).rev() {
        *output = stride;
        stride = stride.checked_mul(*dimension).ok_or_else(|| {
            Error::limit(format!("{description} contiguous stride overflows u64"))
        })?;
    }
    Ok(strides.into_boxed_slice())
}

fn first_missing_digest_ordinal(storage: &Storage, digest_count: usize) -> Option<u32> {
    let primary = storage.span().file().ordinal();
    if usize::try_from(primary).map_or(true, |index| index >= digest_count) {
        return Some(primary);
    }
    let Storage::Quantized(quantized) = storage else {
        return None;
    };
    quantized
        .companions()
        .values()
        .map(|companion| companion.span().file().ordinal())
        .find(|ordinal| usize::try_from(*ordinal).map_or(true, |index| index >= digest_count))
}

fn group_names(
    entries: impl IntoIterator<Item = (TensorName, usize)>,
) -> BTreeMap<TensorName, Vec<usize>> {
    let mut grouped = BTreeMap::<TensorName, Vec<usize>>::new();
    for (name, index) in entries {
        grouped.entry(name).or_default().push(index);
    }
    grouped
}

fn validate_sorted_unique<'a>(
    values: impl IntoIterator<Item = &'a TensorName>,
    message: &'static str,
) -> Result<()> {
    let mut previous: Option<&TensorName> = None;
    for value in values {
        if previous.is_some_and(|previous| previous >= value) {
            return Err(Error::invalid(message));
        }
        previous = Some(value);
    }
    Ok(())
}

#[derive(Serialize)]
struct PlanIdentityPayload<'a> {
    schema_version: u32,
    inputs: &'a PlanInputs,
    sources: &'a [SourceTensor],
    targets: &'a [TargetTensor],
    extra_source_policy: ExtraSourcePolicy,
    bindings: &'a [Binding],
    missing_optional: &'a [TensorName],
    unused_sources: &'a [TensorName],
}

fn compute_plan_id(payload: &PlanIdentityPayload<'_>) -> Result<PlanId> {
    let bytes = canonical_json(payload, "serialize binding plan identity")?;
    Ok(PlanId::from_digest(ContentDigest::hash(
        "binding-plan-v2",
        [bytes],
    )))
}

fn canonical_json(value: &impl Serialize, message: &'static str) -> Result<Box<[u8]>> {
    serde_json::to_vec(value)
        .map(Vec::into_boxed_slice)
        .map_err(|error| Error::with_source(ErrorCategory::InvalidFormat, message, error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepare::builtin_contiguous_implementation;
    use crate::quantization::{Packing, QuantizedStorage};
    use crate::tensor::{DType, FileId, SourceSpan};

    fn stable(value: &str) -> Result<StableName> {
        StableName::parse(value)
    }

    fn tensor(value: &str) -> Result<TensorName> {
        TensorName::parse(value)
    }

    fn digest_id<T>(domain: &str, constructor: impl FnOnce(ContentDigest) -> T) -> T {
        constructor(ContentDigest::hash(domain, [b"test"]))
    }

    fn inputs() -> PlanInputs {
        PlanInputs::new(
            digest_id("manifest", ManifestId::from_digest),
            digest_id("selection", SelectionId::from_digest),
            digest_id("contract", ContractId::from_digest),
            digest_id("backend", BackendId::from_digest),
            [ContentDigest::hash("source", [b"checkpoint"])],
        )
    }

    fn source(source_name: &str) -> Result<SourceTensor> {
        let span = SourceSpan::new(FileId::from_ordinal(0), 0, 8)?;
        SourceTensor::new(
            tensor(source_name)?,
            [2_u64],
            Storage::Plain {
                dtype: DType::F32,
                span,
            },
        )
    }

    fn target(target_name: &str) -> Result<TargetTensor> {
        TargetTensor::builder(
            tensor(target_name)?,
            Requirement::Required,
            [2_u64],
            Representation::contiguous(DType::F32),
            8,
        )
        .build()
    }

    fn refresh_plan_id(plan: &mut BindingPlan) -> Result<()> {
        plan.id = compute_plan_id(&PlanIdentityPayload {
            schema_version: plan.schema_version,
            inputs: &plan.inputs,
            sources: &plan.sources,
            targets: &plan.targets,
            extra_source_policy: plan.extra_source_policy,
            bindings: &plan.bindings,
            missing_optional: &plan.missing_optional,
            unused_sources: &plan.unused_sources,
        })?;
        Ok(())
    }

    #[test]
    fn aliases_are_ambiguous_when_two_candidates_exist() -> Result<()> {
        let target = TargetTensor::builder(
            tensor("target")?,
            Requirement::Required,
            [2_u64],
            Representation::contiguous(DType::F32),
            8,
        )
        .aliases([tensor("alternate")?])
        .build()?;
        let analysis = BindingPlan::builder(inputs())
            .sources([source("target")?, source("alternate")?])
            .targets([target])
            .extra_source_policy(ExtraSourcePolicy::Allow)
            .analyze();

        assert!(
            analysis
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.code() == DiagnosticCode::AmbiguousBinding)
        );
        Ok(())
    }

    #[test]
    fn arbitrary_utf8_tensor_names_round_trip_through_a_plan() -> Result<()> {
        let exact_name = "模型.层 0/权重🔥";
        let plan = BindingPlan::builder(inputs())
            .sources([source(exact_name)?])
            .targets([target(exact_name)?])
            .build()?;
        let decoded = BindingPlan::from_canonical_json(&plan.to_canonical_json()?)?;

        assert_eq!(decoded.bindings()[0].source().name().as_str(), exact_name);
        Ok(())
    }

    #[test]
    fn missing_required_target_has_stable_diagnostic_code() -> Result<()> {
        let analysis = BindingPlan::builder(inputs())
            .targets([target("required")?])
            .analyze();

        assert_eq!(
            analysis.diagnostics()[0].code(),
            DiagnosticCode::MissingRequired
        );
        Ok(())
    }

    #[test]
    fn shared_source_requires_and_accepts_one_tied_weight_group() -> Result<()> {
        let make_tied = |target_name: &str| -> Result<TargetTensor> {
            TargetTensor::builder(
                tensor(target_name)?,
                Requirement::Required,
                [2_u64],
                Representation::contiguous(DType::F32),
                8,
            )
            .aliases([tensor("shared")?])
            .shared_source_group(stable("tied-group")?)
            .build()
        };
        let plan = BindingPlan::builder(inputs())
            .sources([source("shared")?])
            .targets([make_tied("left")?, make_tied("right")?])
            .build()?;

        assert_eq!(plan.bindings().len(), 2);
        Ok(())
    }

    #[test]
    fn contiguous_intermediate_size_is_validated_before_planning() -> Result<()> {
        let source_representation = Representation::contiguous(DType::F32);
        let target_representation = Representation::contiguous(DType::F16);
        let result = TargetTensor::builder(
            tensor("weight")?,
            Requirement::Required,
            [2_u64],
            target_representation.clone(),
            8,
        )
        .transforms([PlannedTransform::new(
            TransformSpec::new(
                builtin_contiguous_implementation()?,
                source_representation,
                target_representation,
            ),
            8,
        )])
        .build();

        let Err(error) = result else {
            return Err(Error::binding(
                "invalid contiguous intermediate size was unexpectedly accepted",
            ));
        };
        assert_eq!(
            error.message(),
            "contiguous transform step output size differs from its dtype and shape"
        );
        Ok(())
    }

    #[test]
    fn input_order_does_not_change_canonical_plan() -> Result<()> {
        let first = source("a")?;
        let second = source("b")?;
        let target_a = target("a")?;
        let target_b = target("b")?;
        let forward = BindingPlan::builder(inputs())
            .sources([first.clone(), second.clone()])
            .targets([target_a.clone(), target_b.clone()])
            .build()?;
        let reverse = BindingPlan::builder(inputs())
            .sources([second, first])
            .targets([target_b, target_a])
            .build()?;

        assert_eq!(forward.to_canonical_json()?, reverse.to_canonical_json()?);
        Ok(())
    }

    #[test]
    fn canonical_plan_round_trips_with_same_identity() -> Result<()> {
        let plan = BindingPlan::builder(inputs())
            .sources([source("weight")?])
            .targets([target("weight")?])
            .build()?;
        let bytes = plan.to_canonical_json()?;
        let decoded = BindingPlan::from_canonical_json(&bytes)?;

        assert_eq!(decoded.id(), plan.id());
        Ok(())
    }

    #[test]
    fn canonical_plan_rejects_arbitrary_optional_misses() -> Result<()> {
        let optional = TargetTensor::builder(
            tensor("optional")?,
            Requirement::Optional,
            [2_u64],
            Representation::contiguous(DType::F32),
            8,
        )
        .build()?;
        let mut plan = BindingPlan::builder(inputs()).targets([optional]).build()?;
        plan.missing_optional = vec![tensor("invented")?].into_boxed_slice();
        refresh_plan_id(&mut plan)?;

        let bytes = plan.to_canonical_json()?;
        let Err(error) = BindingPlan::from_canonical_json(&bytes) else {
            return Err(Error::binding(
                "canonical plan unexpectedly accepted an invented optional miss",
            ));
        };
        assert_eq!(
            error.message(),
            "serialized optional misses differ from deterministic planning"
        );
        Ok(())
    }

    #[test]
    fn canonical_plan_rejects_a_bound_source_listed_as_unused() -> Result<()> {
        let mut plan = BindingPlan::builder(inputs())
            .sources([source("weight")?])
            .targets([target("weight")?])
            .build()?;
        plan.unused_sources = vec![tensor("weight")?].into_boxed_slice();
        refresh_plan_id(&mut plan)?;

        let bytes = plan.to_canonical_json()?;
        let Err(error) = BindingPlan::from_canonical_json(&bytes) else {
            return Err(Error::binding(
                "canonical plan unexpectedly accepted a bound source as unused",
            ));
        };
        assert_eq!(
            error.message(),
            "serialized unused sources differ from deterministic planning"
        );
        Ok(())
    }

    #[test]
    fn canonical_plan_replays_tied_source_invariants() -> Result<()> {
        let make_tied = |target_name: &str| -> Result<TargetTensor> {
            TargetTensor::builder(
                tensor(target_name)?,
                Requirement::Required,
                [2_u64],
                Representation::contiguous(DType::F32),
                8,
            )
            .aliases([tensor("shared")?])
            .shared_source_group(stable("tied-group")?)
            .build()
        };
        let mut plan = BindingPlan::builder(inputs())
            .sources([source("shared")?])
            .targets([make_tied("left")?, make_tied("right")?])
            .build()?;
        plan.targets[0].shared_source_group = None;
        let changed_name = plan.targets[0].name().clone();
        let changed_target = plan.targets[0].clone();
        let binding = plan
            .bindings
            .iter_mut()
            .find(|binding| binding.target.name() == &changed_name)
            .ok_or_else(|| Error::binding("test plan is missing its changed tied target"))?;
        binding.target = changed_target;
        refresh_plan_id(&mut plan)?;

        let bytes = plan.to_canonical_json()?;
        let Err(error) = BindingPlan::from_canonical_json(&bytes) else {
            return Err(Error::binding(
                "canonical plan unexpectedly accepted inconsistent source sharing",
            ));
        };
        assert!(error.message().contains("conflicting_source_reuse"));
        Ok(())
    }

    #[test]
    fn conversion_recipe_rejects_forward_step_reference() -> Result<()> {
        let operation = ImplementationId::new(stable("test-provider")?, stable("rename")?, 1);
        let step = RecipeStep::new(
            stable("first")?,
            operation.clone(),
            [RecipeInput::Step(stable("later")?)],
            BTreeMap::new(),
        );
        let result = ConversionRecipe::new(
            1,
            operation,
            vec![tensor("source")?],
            vec![step],
            BTreeMap::from([(tensor("target")?, RecipeInput::Step(stable("first")?))]),
        );

        let Err(error) = result else {
            return Err(Error::binding(
                "conversion recipe unexpectedly accepted a forward reference",
            ));
        };
        assert_eq!(error.category(), ErrorCategory::Binding);
        Ok(())
    }

    #[test]
    fn conversion_recipe_preserves_semantic_external_input_order() -> Result<()> {
        let operation =
            ImplementationId::new(stable("test-provider")?, stable("ordered-concat")?, 1);
        let q = tensor("q")?;
        let k = tensor("k")?;
        let v = tensor("v")?;
        let output = tensor("qkv")?;
        let step = RecipeStep::new(
            stable("assembled")?,
            operation.clone(),
            [
                RecipeInput::External(q.clone()),
                RecipeInput::External(k.clone()),
                RecipeInput::External(v.clone()),
            ],
            BTreeMap::new(),
        );
        let forward = ConversionRecipe::new(
            1,
            operation.clone(),
            vec![q.clone(), k.clone(), v.clone()],
            vec![step.clone()],
            BTreeMap::from([(output.clone(), RecipeInput::Step(stable("assembled")?))]),
        )?;
        let reordered = ConversionRecipe::new(
            1,
            operation,
            vec![k, q, v],
            vec![step],
            BTreeMap::from([(output, RecipeInput::Step(stable("assembled")?))]),
        )?;

        assert_eq!(
            forward
                .inputs()
                .iter()
                .map(TensorName::as_str)
                .collect::<Vec<_>>(),
            ["q", "k", "v"]
        );
        assert_ne!(forward.digest()?, reordered.digest()?);
        Ok(())
    }

    #[test]
    fn target_rejects_a_recipe_combined_with_planned_transforms() -> Result<()> {
        let source_name = tensor("weight")?;
        let recipe_implementation =
            ImplementationId::new(stable("test-provider")?, stable("identity-recipe")?, 1);
        let recipe = ConversionRecipe::new(
            1,
            recipe_implementation,
            vec![source_name.clone()],
            Vec::new(),
            BTreeMap::from([(
                source_name.clone(),
                RecipeInput::External(source_name.clone()),
            )]),
        )?;
        let source_representation = Representation::contiguous(DType::F32);
        let target_representation = Representation::contiguous(DType::F16);
        let result = TargetTensor::builder(
            source_name,
            Requirement::Required,
            [2_u64],
            target_representation.clone(),
            4,
        )
        .transforms([PlannedTransform::new(
            TransformSpec::new(
                builtin_contiguous_implementation()?,
                source_representation,
                target_representation,
            ),
            4,
        )])
        .conversion_recipe(recipe)
        .build();

        let Err(error) = result else {
            return Err(Error::binding(
                "target unexpectedly combined a recipe and planned transforms",
            ));
        };
        assert_eq!(
            error.message(),
            "target cannot combine a conversion recipe with planned transforms"
        );
        Ok(())
    }

    #[test]
    fn recipe_only_target_allows_provider_shape_and_representation_conversion() -> Result<()> {
        let source_name = tensor("source.weight")?;
        let target_name = tensor("target.weight")?;
        let recipe_implementation =
            ImplementationId::new(stable("test-provider")?, stable("reshape-cast")?, 1);
        let recipe_step = RecipeStep::new(
            stable("converted")?,
            recipe_implementation.clone(),
            [RecipeInput::External(source_name.clone())],
            BTreeMap::new(),
        );
        let recipe = ConversionRecipe::new(
            1,
            recipe_implementation,
            vec![source_name.clone()],
            vec![recipe_step],
            BTreeMap::from([(target_name.clone(), RecipeInput::Step(stable("converted")?))]),
        )?;
        let target_representation = Representation::contiguous(DType::F16);
        let target = TargetTensor::builder(
            target_name,
            Requirement::Required,
            [2_u64],
            target_representation.clone(),
            4,
        )
        .aliases([source_name.clone()])
        .source_shape([4_u64])
        .conversion_recipe(recipe)
        .build()?;
        let source = SourceTensor::new(
            source_name,
            [4_u64],
            Storage::Plain {
                dtype: DType::F32,
                span: SourceSpan::new(FileId::from_ordinal(0), 0, 16)?,
            },
        )?;
        let plan = BindingPlan::builder(inputs())
            .sources([source])
            .targets([target])
            .build()?;
        let binding = plan
            .bindings()
            .first()
            .ok_or_else(|| Error::binding("recipe-only test plan has no binding"))?;

        assert_eq!(
            (
                binding.target().source_shape(),
                binding.target().shape(),
                binding.target().representation(),
                binding.target().output_size(),
            ),
            (
                [4_u64].as_slice(),
                [2_u64].as_slice(),
                &target_representation,
                4,
            )
        );
        Ok(())
    }

    #[test]
    fn target_rejects_a_recipe_combined_with_a_quantized_route() -> Result<()> {
        let recipe_implementation =
            ImplementationId::new(stable("test-provider")?, stable("identity-recipe")?, 1);
        let source_name = tensor("weight")?;
        let recipe = ConversionRecipe::new(
            1,
            recipe_implementation.clone(),
            vec![source_name.clone()],
            Vec::new(),
            BTreeMap::from([(
                source_name.clone(),
                RecipeInput::External(source_name.clone()),
            )]),
        )?;
        let capability = RouteCapability::new(
            recipe_implementation.clone(),
            QuantizedRoute::HostDequant {
                target_dtype: DType::F32,
            },
            recipe_implementation,
            None,
            None,
        )?;
        let result = TargetTensor::builder(
            source_name,
            Requirement::Required,
            [2_u64],
            Representation::contiguous(DType::F32),
            8,
        )
        .conversion_recipe(recipe)
        .quantized_route(capability)
        .build();

        let Err(error) = result else {
            return Err(Error::binding(
                "target unexpectedly combined a recipe and quantized route",
            ));
        };
        assert_eq!(
            error.message(),
            "target cannot combine a conversion recipe with a quantized route"
        );
        Ok(())
    }

    #[test]
    fn packed_direct_bridges_logical_and_packed_target_shapes() -> Result<()> {
        let plan_inputs = inputs();
        let encoding = ImplementationId::new(stable("ggml")?, stable("q8-0")?, 1);
        let storage = QuantizedStorage::new(
            encoding.clone(),
            [2_u64, 32],
            SourceSpan::new(FileId::from_ordinal(0), 0, 68)?,
            Packing::flat_blocks(32, 34)?,
        )?;
        let source =
            SourceTensor::new(tensor("weight")?, [2_u64, 32], Storage::Quantized(storage))?;
        let capability = RouteCapability::new(
            encoding,
            QuantizedRoute::PackedDirect,
            ImplementationId::new(stable("test-provider")?, stable("packed-direct")?, 1),
            Some(plan_inputs.backend()),
            None,
        )?;
        let target = TargetTensor::builder(
            tensor("weight")?,
            Requirement::Required,
            [2_u64, 34],
            Representation::contiguous(DType::U8),
            68,
        )
        .source_shape([2_u64, 32])
        .quantized_route(capability.clone())
        .build()?;

        let plan = BindingPlan::builder(plan_inputs.clone())
            .sources([source.clone()])
            .targets([target])
            .build()?;
        assert_eq!(plan.bindings()[0].target().shape(), &[2, 34]);
        assert_eq!(plan.bindings()[0].target().source_shape(), &[2, 32]);

        let invalid_target = TargetTensor::builder(
            tensor("weight")?,
            Requirement::Required,
            [2_u64, 33],
            Representation::contiguous(DType::U8),
            66,
        )
        .source_shape([2_u64, 32])
        .quantized_route(capability)
        .build()?;
        let error = BindingPlan::builder(plan_inputs)
            .sources([source])
            .targets([invalid_target])
            .build()
            .expect_err("packed-direct target with a different byte size must fail");
        assert!(
            error
                .message()
                .contains("packed-direct output size differs from the source payload span")
        );
        Ok(())
    }

    #[test]
    fn transform_implementation_version_changes_plan_identity() -> Result<()> {
        let representation = Representation::contiguous(DType::F32);
        let make_target = |version| -> Result<TargetTensor> {
            let implementation = if version == 1 {
                builtin_contiguous_implementation()?
            } else {
                ImplementationId::new(
                    stable("model-weights")?,
                    stable("contiguous-cast")?,
                    version,
                )
            };
            TargetTensor::builder(
                tensor("weight")?,
                Requirement::Required,
                [2_u64],
                representation.clone(),
                8,
            )
            .transforms([PlannedTransform::new(
                TransformSpec::new(
                    implementation,
                    representation.clone(),
                    representation.clone(),
                ),
                8,
            )])
            .build()
        };
        let first = BindingPlan::builder(inputs())
            .sources([source("weight")?])
            .targets([make_target(1)?])
            .build()?;
        let second = BindingPlan::builder(inputs())
            .sources([source("weight")?])
            .targets([make_target(2)?])
            .build()?;

        assert_ne!(first.id(), second.id());
        Ok(())
    }

    #[test]
    fn transform_scratch_is_serialized_and_changes_plan_identity() -> Result<()> {
        let representation = Representation::contiguous(DType::F32);
        let make_target = |scratch_bytes| -> Result<TargetTensor> {
            let planned = PlannedTransform::new(
                TransformSpec::new(
                    builtin_contiguous_implementation()?,
                    representation.clone(),
                    representation.clone(),
                ),
                8,
            )
            .with_scratch_bytes(scratch_bytes);
            TargetTensor::builder(
                tensor("weight")?,
                Requirement::Required,
                [2_u64],
                representation.clone(),
                8,
            )
            .transforms([planned])
            .build()
        };
        let zero_scratch = BindingPlan::builder(inputs())
            .sources([source("weight")?])
            .targets([make_target(0)?])
            .build()?;
        let seven_scratch = BindingPlan::builder(inputs())
            .sources([source("weight")?])
            .targets([make_target(7)?])
            .build()?;

        assert_ne!(zero_scratch.id(), seven_scratch.id());
        let canonical = seven_scratch.to_canonical_json()?;
        assert!(
            canonical
                .windows(b"\"scratch_bytes\":7".len())
                .any(|window| window == b"\"scratch_bytes\":7")
        );
        assert_eq!(
            BindingPlan::from_canonical_json(&canonical)?.id(),
            seven_scratch.id()
        );
        assert_eq!(zero_scratch.targets()[0].transforms()[0].scratch_bytes(), 0);
        Ok(())
    }
}
