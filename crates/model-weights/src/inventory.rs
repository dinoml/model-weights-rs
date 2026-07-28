//! Deterministic checkpoint and shard-index inventory.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::identity::ContentDigest;
use crate::limits::ResourceLimits;
use crate::quantization::Storage;
use crate::source::{DigestPolicy, RepoPath, SourceKind};
use crate::tensor::FileId;
use crate::{CancellationToken, Error, ErrorCategory, Result};

/// Records how a file's content digest was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DigestState {
    /// The digest will be computed before content-addressed planning.
    Deferred,
    /// The declared digest will be verified before content-addressed planning.
    Expected(ContentDigest),
    /// An immutable retained snapshot already verified the digest.
    Trusted(ContentDigest),
}

impl From<DigestPolicy> for DigestState {
    fn from(policy: DigestPolicy) -> Self {
        match policy {
            DigestPolicy::ComputeOnDemand => Self::Deferred,
            DigestPolicy::VerifyOnDemand(digest) => Self::Expected(digest),
            DigestPolicy::TrustRetained(digest) => Self::Trusted(digest),
        }
    }
}

/// Provenance and identity facts for one opened checkpoint file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    id: FileId,
    path: RepoPath,
    size: u64,
    kind: SourceKind,
    digest: DigestState,
}

impl FileRecord {
    pub(crate) const fn new(
        id: FileId,
        path: RepoPath,
        size: u64,
        kind: SourceKind,
        digest: DigestState,
    ) -> Self {
        Self {
            id,
            path,
            size,
            kind,
            digest,
        }
    }

    /// Returns the inventory-local file identifier.
    #[must_use]
    pub const fn id(&self) -> FileId {
        self.id
    }

    /// Returns the repository-relative provenance path.
    #[must_use]
    pub const fn path(&self) -> &RepoPath {
        &self.path
    }

    /// Returns the opened file size.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Returns the source lifetime class.
    #[must_use]
    pub const fn kind(&self) -> SourceKind {
        self.kind
    }

    /// Returns the digest verification state at inventory time.
    #[must_use]
    pub const fn digest_state(&self) -> DigestState {
        self.digest
    }
}

/// Metadata and storage provenance for one logical tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRecord {
    name: Box<str>,
    shape: Box<[u64]>,
    storage: Storage,
}

impl TensorRecord {
    pub(crate) fn new(name: Box<str>, shape: Box<[u64]>, storage: Storage) -> Self {
        Self {
            name,
            shape,
            storage,
        }
    }

    /// Returns the exact, case-sensitive source tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the logical tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the plain or explicitly quantized storage descriptor.
    #[must_use]
    pub const fn storage(&self) -> &Storage {
        &self.storage
    }
}

/// A deterministic, immutable checkpoint tensor inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "InventoryWire")]
pub struct Inventory {
    files: Box<[FileRecord]>,
    tensors: Box<[TensorRecord]>,
}

impl Inventory {
    pub(crate) fn new(files: Vec<FileRecord>, mut tensors: Vec<TensorRecord>) -> Result<Self> {
        if files.is_empty() {
            return Err(Error::invalid(
                "checkpoint inventory must contain at least one source file",
            ));
        }
        for (index, file) in files.iter().enumerate() {
            let ordinal = u32::try_from(index)
                .map_err(|_error| Error::limit("inventory file ordinal exceeds u32"))?;
            if file.id != FileId::from_ordinal(ordinal) {
                return Err(Error::invalid(
                    "inventory file identifier differs from its canonical ordinal",
                ));
            }
            if index > 0 && files[index - 1].path >= file.path {
                return Err(Error::invalid(
                    "checkpoint source files are not in unique deterministic order",
                ));
            }
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));

        if tensors.windows(2).any(|pair| pair[0].name == pair[1].name) {
            return Err(Error::invalid(
                "checkpoint contains a duplicate tensor name across source files",
            ));
        }
        for tensor in &tensors {
            if tensor.name.is_empty() {
                return Err(Error::invalid("inventory tensor name must not be empty"));
            }
            match &tensor.storage {
                Storage::Plain { dtype, span } => {
                    if dtype.byte_len(&tensor.shape)? != span.len() {
                        return Err(Error::invalid(
                            "inventory plain storage length disagrees with dtype and shape",
                        ));
                    }
                    validate_inventory_span(*span, &files)?;
                }
                Storage::Quantized(storage) => {
                    if storage.logical_shape() != tensor.shape.as_ref() {
                        return Err(Error::invalid(
                            "inventory quantized shape disagrees with its tensor shape",
                        ));
                    }
                    validate_inventory_span(storage.span(), &files)?;
                    for companion in storage.companions().values() {
                        validate_inventory_span(companion.span(), &files)?;
                    }
                }
            }
        }

        Ok(Self {
            files: files.into_boxed_slice(),
            tensors: tensors.into_boxed_slice(),
        })
    }

    /// Returns source files sorted by repository-relative path.
    #[must_use]
    pub fn files(&self) -> &[FileRecord] {
        &self.files
    }

    /// Returns tensors sorted by exact source name.
    #[must_use]
    pub fn tensors(&self) -> &[TensorRecord] {
        &self.tensors
    }

    /// Finds an exact, case-sensitive tensor name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorRecord> {
        self.tensors
            .binary_search_by(|tensor| tensor.name().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }

    /// Returns the tensor count.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns whether the inventory contains no tensors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterates over tensors in deterministic name order.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &TensorRecord> {
        self.tensors.iter()
    }
}

#[derive(Debug, Deserialize)]
struct InventoryWire {
    files: Vec<FileRecord>,
    tensors: Vec<TensorRecord>,
}

impl TryFrom<InventoryWire> for Inventory {
    type Error = Error;

    fn try_from(wire: InventoryWire) -> Result<Self> {
        Self::new(wire.files, wire.tensors)
    }
}

fn validate_inventory_span(span: crate::tensor::SourceSpan, files: &[FileRecord]) -> Result<()> {
    let file_index = usize::try_from(span.file().ordinal())
        .map_err(|_error| Error::limit("inventory file identifier does not fit usize"))?;
    let file = files
        .get(file_index)
        .ok_or_else(|| Error::invalid("inventory storage refers to an unknown source file"))?;
    if span.end() > file.size {
        return Err(Error::invalid(
            "inventory storage span extends beyond its source file",
        ));
    }
    Ok(())
}

impl<'a> IntoIterator for &'a Inventory {
    type Item = &'a TensorRecord;
    type IntoIter = std::slice::Iter<'a, TensorRecord>;

    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

/// A normalized safetensors index mapping tensor names to safe shard paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShardIndex {
    weight_map: BTreeMap<Box<str>, RepoPath>,
}

impl ShardIndex {
    /// Parses a bounded Hugging Face safetensors index.
    ///
    /// Parsing rejects duplicate top-level keys, duplicate tensor names, unsafe
    /// shard paths, trailing data, and maps exceeding configured limits.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format, invalid-path, or resource-limit error when
    /// the index violates the contract.
    pub fn from_json(bytes: &[u8], limits: &ResourceLimits) -> Result<Self> {
        Self::from_json_with_cancellation(bytes, limits, &CancellationToken::new())
    }

    /// Parses a bounded safetensors index with cooperative cancellation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format, invalid-path, resource-limit, or cancellation
    /// error.
    pub fn from_json_with_cancellation(
        bytes: &[u8],
        limits: &ResourceLimits,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        cancellation.check()?;
        let byte_limit = match usize::try_from(limits.max_header_bytes()) {
            Ok(limit) => limit,
            Err(_) => usize::MAX,
        };
        if bytes.len() > byte_limit {
            return Err(Error::limit(
                "safetensors index exceeds the configured byte limit",
            ));
        }

        let control = ShardIndexParseControl::new(limits, cancellation);
        let raw = deserialize_shard_index_json(bytes, &control)?;
        cancellation.check()?;
        Ok(Self {
            weight_map: raw.weight_map,
        })
    }

    /// Returns mappings in deterministic tensor-name order.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &RepoPath)> {
        self.weight_map
            .iter()
            .map(|(name, path)| (name.as_ref(), path))
    }

    /// Returns the unique shard paths in deterministic order.
    #[must_use]
    pub fn shards(&self) -> Box<[RepoPath]> {
        self.weight_map
            .values()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Finds the shard expected to contain `tensor`.
    #[must_use]
    pub fn shard_for(&self, tensor: &str) -> Option<&RepoPath> {
        self.weight_map.get(tensor)
    }

    /// Returns the mapped tensor count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.weight_map.len()
    }

    /// Returns whether no tensors are mapped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weight_map.is_empty()
    }
}

fn deserialize_shard_index_json(
    bytes: &[u8],
    control: &ShardIndexParseControl<'_>,
) -> Result<RawShardIndex> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let raw = match (ShardIndexSeed { control }).deserialize(&mut deserializer) {
        Ok(raw) => raw,
        Err(source) => {
            return Err(control.classify_json_error(source, "safetensors index JSON is invalid"));
        }
    };
    control.cancellation.check()?;
    if let Err(source) = deserializer.end() {
        return Err(control.classify_json_error(source, "safetensors index contains trailing data"));
    }
    control.cancellation.check()?;
    Ok(raw)
}

#[derive(Debug)]
struct ShardIndexParseControl<'a> {
    limits: &'a ResourceLimits,
    cancellation: &'a CancellationToken,
    failure: RefCell<Option<Error>>,
}

impl<'a> ShardIndexParseControl<'a> {
    fn new(limits: &'a ResourceLimits, cancellation: &'a CancellationToken) -> Self {
        Self {
            limits,
            cancellation,
            failure: RefCell::new(None),
        }
    }

    fn check<E>(&self) -> std::result::Result<(), E>
    where
        E: serde::de::Error,
    {
        if self.cancellation.is_cancelled() {
            Err(self.abort(Error::cancelled()))
        } else {
            Ok(())
        }
    }

    fn limit<E>(&self, message: &'static str) -> E
    where
        E: serde::de::Error,
    {
        self.abort(Error::limit(message))
    }

    fn abort<E>(&self, error: Error) -> E
    where
        E: serde::de::Error,
    {
        let mut failure = self.failure.borrow_mut();
        if failure.is_none() {
            *failure = Some(error);
        }
        E::custom("bounded safetensors index parsing aborted")
    }

    fn classify_json_error(&self, source: serde_json::Error, message: &'static str) -> Error {
        if let Some(error) = self.failure.borrow_mut().take() {
            error
        } else if self.cancellation.is_cancelled() {
            Error::cancelled()
        } else {
            Error::with_source(ErrorCategory::InvalidFormat, message, source)
        }
    }
}

struct ShardIndexSeed<'control, 'context> {
    control: &'control ShardIndexParseControl<'context>,
}

impl<'de> DeserializeSeed<'de> for ShardIndexSeed<'_, '_> {
    type Value = RawShardIndex;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_map(ShardIndexVisitor {
            control: self.control,
        })
    }
}

struct RawShardIndex {
    weight_map: BTreeMap<Box<str>, RepoPath>,
}

struct ShardIndexVisitor<'control, 'context> {
    control: &'control ShardIndexParseControl<'context>,
}

impl<'de> Visitor<'de> for ShardIndexVisitor<'_, '_> {
    type Value = RawShardIndex;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors index object containing weight_map")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = std::collections::BTreeSet::new();
        let mut weight_map = None;

        loop {
            self.control.check()?;
            let Some(key) = map.next_key::<Box<str>>()? else {
                break;
            };
            self.control.check()?;
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format_args!(
                    "duplicate safetensors index key {key:?}"
                )));
            }
            match key.as_ref() {
                "weight_map" => {
                    weight_map = Some(map.next_value_seed(WeightMapSeed {
                        control: self.control,
                    })?);
                }
                _ => {
                    map.next_value_seed(CancellableIgnoredAnySeed {
                        control: self.control,
                    })?;
                }
            }
        }

        let weight_map = weight_map.ok_or_else(|| serde::de::Error::missing_field("weight_map"))?;
        Ok(RawShardIndex { weight_map })
    }
}

struct WeightMapSeed<'control, 'context> {
    control: &'control ShardIndexParseControl<'context>,
}

impl<'de> DeserializeSeed<'de> for WeightMapSeed<'_, '_> {
    type Value = BTreeMap<Box<str>, RepoPath>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_map(WeightMapVisitor {
            control: self.control,
        })
    }
}

struct WeightMapVisitor<'control, 'context> {
    control: &'control ShardIndexParseControl<'context>,
}

impl<'de> Visitor<'de> for WeightMapVisitor<'_, '_> {
    type Value = BTreeMap<Box<str>, RepoPath>;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a tensor-name to shard-path map")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut result = BTreeMap::new();
        let mut shards = std::collections::BTreeSet::new();
        loop {
            self.control.check()?;
            let Some(name) = map.next_key_seed(IndexStringSeed {
                control: self.control,
                maximum: self.control.limits.max_name_bytes(),
                reject_empty: true,
                limit_message: "safetensors index tensor name violates configured limits",
            })?
            else {
                break;
            };
            self.control.check()?;
            if result.contains_key(name.as_ref()) {
                return Err(serde::de::Error::custom(format_args!(
                    "duplicate safetensors tensor name {name:?}"
                )));
            }
            if result.len() >= self.control.limits.max_tensors() {
                return Err(self
                    .control
                    .limit("safetensors index exceeds the configured tensor count"));
            }
            let raw_path = map.next_value_seed(IndexStringSeed {
                control: self.control,
                maximum: self.control.limits.max_name_bytes(),
                reject_empty: false,
                limit_message: "safetensors index shard path violates configured limits",
            })?;
            let path = RepoPath::parse(&raw_path).map_err(|error| self.control.abort(error))?;
            shards.insert(path.clone());
            if shards.len() > self.control.limits.max_shards() {
                return Err(self
                    .control
                    .limit("safetensors index exceeds the configured shard count"));
            }
            result.insert(name, path);
        }
        Ok(result)
    }
}

struct IndexStringSeed<'control, 'context> {
    control: &'control ShardIndexParseControl<'context>,
    maximum: usize,
    reject_empty: bool,
    limit_message: &'static str,
}

impl<'de> DeserializeSeed<'de> for IndexStringSeed<'_, '_> {
    type Value = Box<str>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_str(IndexStringVisitor {
            control: self.control,
            maximum: self.maximum,
            reject_empty: self.reject_empty,
            limit_message: self.limit_message,
        })
    }
}

struct IndexStringVisitor<'control, 'context> {
    control: &'control ShardIndexParseControl<'context>,
    maximum: usize,
    reject_empty: bool,
    limit_message: &'static str,
}

impl IndexStringVisitor<'_, '_> {
    fn validate<E>(&self, value: &str) -> std::result::Result<(), E>
    where
        E: serde::de::Error,
    {
        self.control.check()?;
        if value.len() > self.maximum || (self.reject_empty && value.is_empty()) {
            return Err(self.control.limit(self.limit_message));
        }
        Ok(())
    }
}

impl<'de> Visitor<'de> for IndexStringVisitor<'_, '_> {
    type Value = Box<str>;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded UTF-8 string")
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.validate(value)?;
        Ok(value.into())
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.validate(value)?;
        Ok(value.into())
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.validate(&value)?;
        Ok(value.into_boxed_str())
    }
}

#[derive(Clone, Copy)]
struct CancellableIgnoredAnySeed<'control, 'context> {
    control: &'control ShardIndexParseControl<'context>,
}

impl<'de> DeserializeSeed<'de> for CancellableIgnoredAnySeed<'_, '_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_any(CancellableIgnoredAnyVisitor {
            control: self.control,
        })
    }
}

#[derive(Clone, Copy)]
struct CancellableIgnoredAnyVisitor<'control, 'context> {
    control: &'control ShardIndexParseControl<'context>,
}

impl<'de> Visitor<'de> for CancellableIgnoredAnyVisitor<'_, '_> {
    type Value = ();

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_bool<E>(self, _value: bool) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_i64<E>(self, _value: i64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_u64<E>(self, _value: u64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_f64<E>(self, _value: f64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_borrowed_str<E>(self, _value: &'de str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_str<E>(self, _value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_string<E>(self, _value: String) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.control.check()
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        CancellableIgnoredAnySeed {
            control: self.control,
        }
        .deserialize(deserializer)
    }

    fn visit_newtype_struct<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.visit_some(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        loop {
            self.control.check()?;
            if sequence
                .next_element_seed(CancellableIgnoredAnySeed {
                    control: self.control,
                })?
                .is_none()
            {
                break;
            }
        }
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        loop {
            self.control.check()?;
            if map
                .next_key_seed(CancellableIgnoredAnySeed {
                    control: self.control,
                })?
                .is_none()
            {
                break;
            }
            map.next_value_seed(CancellableIgnoredAnySeed {
                control: self.control,
            })?;
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for RepoPath {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = <&str>::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

impl Serialize for RepoPath {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl Display for RepoPath {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for SourceKind {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(match self {
            Self::Local => "local",
            Self::RetainedSnapshot => "retained_snapshot",
        })
    }
}

impl<'de> Deserialize<'de> for SourceKind {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match <&str>::deserialize(deserializer)? {
            "local" => Ok(Self::Local),
            "retained_snapshot" => Ok(Self::RetainedSnapshot),
            value => Err(serde::de::Error::unknown_variant(
                value,
                &["local", "retained_snapshot"],
            )),
        }
    }
}

#[cfg(test)]
mod bounded_shard_index_parser_tests {
    use super::ShardIndex;
    use crate::limits::ResourceLimits;
    use crate::{ErrorCategory, Result};

    fn limits() -> Result<ResourceLimits> {
        ResourceLimits::builder()
            .max_tensors(1)
            .max_shards(1)
            .max_name_bytes(4)
            .build()
    }

    #[test]
    fn tensor_count_limit_precedes_a_later_invalid_shard_value() -> Result<()> {
        let bytes = br#"{"weight_map":{"a":"a.s","b":false}}"#;
        let error = ShardIndex::from_json(bytes, &limits()?)
            .expect_err("the second mapping must exceed the configured count");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn tensor_name_limit_precedes_its_invalid_shard_value() -> Result<()> {
        let bytes = br#"{"weight_map":{"oversized":false}}"#;
        let error = ShardIndex::from_json(bytes, &limits()?)
            .expect_err("the tensor name must exceed the configured length");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn shard_path_limit_precedes_a_later_duplicate_top_level_key() -> Result<()> {
        let bytes = br#"{"weight_map":{"a":"too-long"},"weight_map":false}"#;
        let error = ShardIndex::from_json(bytes, &limits()?)
            .expect_err("the shard path must exceed the configured length");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn shard_count_limit_precedes_a_later_duplicate_top_level_key() -> Result<()> {
        let limits = ResourceLimits::builder()
            .max_tensors(2)
            .max_shards(1)
            .max_name_bytes(4)
            .build()?;
        let bytes = br#"{"weight_map":{"a":"a.s","b":"b.s"},"weight_map":false}"#;
        let error = ShardIndex::from_json(bytes, &limits)
            .expect_err("the second shard must exceed the configured count");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn nested_index_metadata_is_skipped_without_changing_the_weight_map() -> Result<()> {
        let bytes = br#"{"metadata":{"total_size":1,"nested":[true,null,{"x":"y"}]},"weight_map":{"a":"a.s"}}"#;
        let index = ShardIndex::from_json(bytes, &ResourceLimits::default())?;

        assert_eq!(
            index.shard_for("a").map(crate::source::RepoPath::as_str),
            Some("a.s")
        );
        Ok(())
    }
}
