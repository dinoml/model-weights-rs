//! Safe checkpoint opening, bounded safetensors parsing, and byte access.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Formatter};
use std::fs::{File, Metadata};
use std::io::Read;
#[cfg(not(any(unix, windows)))]
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest as _, Sha256};

use crate::identity::{ContentDigest, ImplementationId, SnapshotId, StableName};
use crate::inventory::{DigestState, FileRecord, Inventory, ShardIndex, TensorRecord};
use crate::limits::ResourceLimits;
use crate::quantization::{Packing, QuantizedStorage, Storage};
use crate::source::{DigestPolicy, RepoPath, SourceDescriptor, SourceKind};
#[cfg(feature = "mmap")]
use crate::tensor::ByteOwner;
use crate::tensor::{ByteView, DType, FileId, SourceSpan};
use crate::{CancellationToken, Error, ErrorCategory, Result};

const SOURCE_READ_BLOCK_BYTES: usize = 1024 * 1024;

/// Controls whether tensor bytes are copied or viewed through a retained map.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AccessMode {
    /// Map retained immutable snapshots and read ordinary files.
    #[default]
    Auto,
    /// Read each requested span into an owned allocation.
    Read,
    /// Require a retained immutable snapshot and return mapped views.
    Mmap,
}

/// Immutable bytes and metadata for one plain tensor.
#[derive(Debug, Clone)]
pub struct PlainTensor {
    name: Box<str>,
    dtype: DType,
    shape: Box<[u64]>,
    bytes: ByteView,
}

impl PlainTensor {
    /// Returns the exact source tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the stored scalar dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the logical shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the immutable stored bytes.
    #[must_use]
    pub const fn bytes(&self) -> &ByteView {
        &self.bytes
    }
}

/// Immutable bytes and metadata for one explicitly encoded tensor.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    name: Box<str>,
    storage: Storage,
    shape: Box<[u64]>,
    bytes: ByteView,
}

impl QuantizedTensor {
    /// Returns the exact source tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the format-neutral packed encoding descriptor.
    #[must_use]
    pub const fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Returns the logical shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the immutable packed bytes.
    #[must_use]
    pub const fn bytes(&self) -> &ByteView {
        &self.bytes
    }
}

/// Tensor bytes loaded without silently decoding an encoded representation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TensorData {
    /// Ordinary scalar storage.
    Plain(PlainTensor),
    /// Packed or otherwise encoded storage.
    Quantized(QuantizedTensor),
}

impl TensorData {
    /// Returns the exact source tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Plain(tensor) => tensor.name(),
            Self::Quantized(tensor) => tensor.name(),
        }
    }

    /// Returns the logical shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        match self {
            Self::Plain(tensor) => tensor.shape(),
            Self::Quantized(tensor) => tensor.shape(),
        }
    }

    /// Returns the immutable stored bytes.
    #[must_use]
    pub const fn bytes(&self) -> &ByteView {
        match self {
            Self::Plain(tensor) => tensor.bytes(),
            Self::Quantized(tensor) => tensor.bytes(),
        }
    }
}

/// An opened, validated checkpoint with cheap clone semantics.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    inner: Arc<CheckpointInner>,
}

#[derive(Debug)]
struct CheckpointInner {
    inventory: Inventory,
    files: Box<[Arc<OpenFile>]>,
    access: AccessMode,
    snapshot: OnceLock<SnapshotId>,
}

impl Checkpoint {
    /// Opens one safetensors file, or a local `*.index.json` shard index.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is unsafe, unavailable, malformed, or
    /// violates default resource limits.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_cancellation(path, &CancellationToken::new())
    }

    /// Opens one safetensors file or local shard index with cancellation.
    ///
    /// # Errors
    ///
    /// Returns a path, I/O, format, integrity, resource-limit, or cancellation
    /// error.
    pub fn open_with_cancellation(
        path: impl AsRef<Path>,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        cancellation.check()?;
        let path = path.as_ref();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".index.json"))
        {
            Self::open_index_with_cancellation(path, cancellation)
        } else {
            Self::open_source_with_cancellation(SourceDescriptor::local(path)?, cancellation)
        }
    }

    /// Opens one source descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error when the source is unavailable, malformed, or violates
    /// default resource limits.
    pub fn open_source(source: SourceDescriptor) -> Result<Self> {
        Self::open_source_with_cancellation(source, &CancellationToken::new())
    }

    /// Opens one source descriptor with cooperative cancellation.
    ///
    /// # Errors
    ///
    /// Returns an I/O, format, integrity, resource-limit, unsupported-mode, or
    /// cancellation error.
    pub fn open_source_with_cancellation(
        source: SourceDescriptor,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        CheckpointBuilder::new(source).open_with_cancellation(cancellation)
    }

    /// Opens every shard referenced by a local safetensors index.
    ///
    /// Paths are resolved beneath the index directory after canonicalization.
    /// Callers must prevent concurrent mutation of an ordinary local directory
    /// while it is opened; retained snapshots provide the stronger lifetime
    /// contract for adversarial or concurrently managed storage.
    ///
    /// # Errors
    ///
    /// Returns an error for an unavailable or malformed index, an unsafe shard
    /// path, a missing shard, or inconsistent shard contents.
    pub fn open_index(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_index_with_cancellation(path, &CancellationToken::new())
    }

    /// Opens every shard referenced by an index with cancellation.
    ///
    /// # Errors
    ///
    /// Returns a path, I/O, format, integrity, resource-limit, or cancellation
    /// error.
    pub fn open_index_with_cancellation(
        path: impl AsRef<Path>,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        CheckpointBuilder::from_local_index_with_limits_and_cancellation(
            path,
            ResourceLimits::default(),
            cancellation,
        )?
        .open_with_cancellation(cancellation)
    }

    /// Starts configuring a checkpoint around one source.
    pub fn builder(source: SourceDescriptor) -> CheckpointBuilder {
        CheckpointBuilder::new(source)
    }

    /// Returns the deterministic source and tensor inventory.
    #[must_use]
    pub fn inventory(&self) -> &Inventory {
        &self.inner.inventory
    }

    /// Returns the configured copied-read or retained-mapping policy.
    #[must_use]
    pub fn access_mode(&self) -> AccessMode {
        self.inner.access
    }

    /// Reads or maps the exact bytes for a named tensor.
    ///
    /// Encoded storage is returned as encoded bytes; this method never
    /// dequantizes implicitly.
    ///
    /// # Errors
    ///
    /// Returns a binding error when the name is absent, or an I/O, integrity,
    /// or resource-limit error when its validated source span cannot be read.
    pub fn tensor(&self, name: &str) -> Result<TensorData> {
        self.tensor_with_cancellation(name, &CancellationToken::new())
    }

    /// Reads or maps a named tensor with cooperative cancellation.
    ///
    /// Encoded storage is returned as encoded bytes; this method never
    /// dequantizes implicitly. Owned reads check the token between fixed-size
    /// blocks.
    ///
    /// # Errors
    ///
    /// Returns a binding, I/O, integrity, unsupported-mode, resource-limit, or
    /// cancellation error.
    pub fn tensor_with_cancellation(
        &self,
        name: &str,
        cancellation: &CancellationToken,
    ) -> Result<TensorData> {
        let record = self
            .inner
            .inventory
            .tensor(name)
            .ok_or_else(|| Error::binding("checkpoint tensor name was not found"))?;
        let bytes = self.read_span_with_cancellation(record.storage().span(), cancellation)?;
        match record.storage() {
            Storage::Plain { dtype, .. } => Ok(TensorData::Plain(PlainTensor {
                name: record.name().into(),
                dtype: *dtype,
                shape: record.shape().into(),
                bytes,
            })),
            Storage::Quantized(_) => Ok(TensorData::Quantized(QuantizedTensor {
                name: record.name().into(),
                storage: record.storage().clone(),
                shape: record.shape().into(),
                bytes,
            })),
        }
    }

    /// Materializes the bytes covered by a previously validated source span.
    ///
    /// This low-level method is intended for plan and preparation providers.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error for an unknown file identifier, or an
    /// I/O, integrity, unsupported-mode, or resource-limit error.
    pub fn read_span(&self, span: SourceSpan) -> Result<ByteView> {
        self.read_span_with_cancellation(span, &CancellationToken::new())
    }

    /// Materializes a validated source span with cooperative cancellation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error for an unknown file identifier, or an
    /// I/O, integrity, unsupported-mode, resource-limit, or cancellation error.
    pub fn read_span_with_cancellation(
        &self,
        span: SourceSpan,
        cancellation: &CancellationToken,
    ) -> Result<ByteView> {
        cancellation.check()?;
        let file_index = usize::try_from(span.file().ordinal())
            .map_err(|_error| Error::limit("source file identifier does not fit usize"))?;
        let file = self
            .inner
            .files
            .get(file_index)
            .ok_or_else(|| Error::invalid("source span refers to an unknown file"))?;
        let view = file.read_span(span, self.inner.access, cancellation)?;
        cancellation.check()?;
        Ok(view)
    }

    /// Establishes and returns the content-addressed checkpoint snapshot.
    ///
    /// Trusted retained digests are reused. Successful ordinary-file hashing
    /// is serialized and cached once per checkpoint, and declared digests are
    /// verified.
    ///
    /// # Errors
    ///
    /// Returns an I/O, integrity, or cancellation error.
    pub fn snapshot_id(&self, cancellation: &CancellationToken) -> Result<SnapshotId> {
        if let Some(snapshot) = self.inner.snapshot.get() {
            return Ok(*snapshot);
        }

        let mut identity = Vec::new();
        identity.extend_from_slice(b"model-weights-snapshot-v1\0");
        for (record, file) in self
            .inner
            .inventory
            .files()
            .iter()
            .zip(self.inner.files.iter())
        {
            cancellation.check()?;
            let digest = file.content_digest(cancellation)?;
            let path = record.path().as_str().as_bytes();
            let path_len = u64::try_from(path.len())
                .map_err(|_error| Error::limit("repository path length does not fit u64"))?;
            let additional = path
                .len()
                .checked_add(8 + 8 + 32)
                .ok_or_else(|| Error::limit("snapshot identity length overflows usize"))?;
            identity
                .try_reserve(additional)
                .map_err(|_error| Error::limit("could not allocate snapshot identity bytes"))?;
            identity.extend_from_slice(&path_len.to_le_bytes());
            identity.extend_from_slice(path);
            identity.extend_from_slice(&record.size().to_le_bytes());
            identity.extend_from_slice(digest.as_bytes());
        }
        let snapshot =
            SnapshotId::from_digest(ContentDigest::hash("snapshot-envelope-v1", [&identity]));
        let _ = self.inner.snapshot.set(snapshot);
        Ok(self.inner.snapshot.get().copied().unwrap_or(snapshot))
    }

    /// Returns verified source digests in inventory file-ordinal order.
    ///
    /// The result can be passed directly to
    /// [`PlanInputs::new`](crate::plan::PlanInputs::new). Trusted retained
    /// digests are reused; successful ordinary-file hashes are cached.
    ///
    /// # Errors
    ///
    /// Returns an I/O, integrity, resource-limit, or cancellation error.
    pub fn source_digests(&self, cancellation: &CancellationToken) -> Result<Box<[ContentDigest]>> {
        self.inner
            .files
            .iter()
            .map(|file| file.content_digest(cancellation))
            .collect::<Result<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    /// Returns source bytes that still require full hashing on this handle.
    ///
    /// Trusted retained digests and successfully cached ordinary-file digests
    /// contribute zero. The value is intended for progress and telemetry, not
    /// as a content identity.
    ///
    /// # Errors
    ///
    /// Returns an integrity or cancellation error while inspecting digest
    /// state, or a resource-limit error if the byte sum overflows.
    pub fn pending_digest_bytes(&self, cancellation: &CancellationToken) -> Result<u64> {
        self.inner.files.iter().try_fold(0_u64, |total, file| {
            if file.needs_hashing(cancellation)? {
                total
                    .checked_add(file.size)
                    .ok_or_else(|| Error::limit("pending digest byte count overflows u64"))
            } else {
                Ok(total)
            }
        })
    }
}

/// Configures resource limits, access mode, sources, and optional shard truth.
#[derive(Debug)]
#[must_use]
pub struct CheckpointBuilder {
    sources: Vec<SourceDescriptor>,
    shard_index: Option<ShardIndex>,
    limits: ResourceLimits,
    access: AccessMode,
}

impl CheckpointBuilder {
    /// Creates a builder containing one source.
    pub fn new(source: SourceDescriptor) -> Self {
        Self {
            sources: vec![source],
            shard_index: None,
            limits: ResourceLimits::default(),
            access: AccessMode::default(),
        }
    }

    /// Creates a builder containing caller-retained sources.
    pub fn from_sources(sources: impl IntoIterator<Item = SourceDescriptor>) -> Self {
        Self {
            sources: sources.into_iter().collect(),
            shard_index: None,
            limits: ResourceLimits::default(),
            access: AccessMode::default(),
        }
    }

    /// Reads a bounded local index and resolves its shards safely.
    ///
    /// # Errors
    ///
    /// Returns an error when the index is unavailable or malformed, or any
    /// canonical shard path escapes the index directory.
    pub fn from_local_index(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_local_index_with_limits(path, ResourceLimits::default())
    }

    /// Reads a local index under explicit limits and resolves its shards.
    ///
    /// # Errors
    ///
    /// Returns an error when the index is unavailable or malformed, exceeds
    /// `limits`, or any canonical shard path escapes the index directory.
    pub fn from_local_index_with_limits(
        path: impl AsRef<Path>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        Self::from_local_index_with_limits_and_cancellation(path, limits, &CancellationToken::new())
    }

    /// Reads and resolves a bounded local index with cancellation.
    ///
    /// # Errors
    ///
    /// Returns a path, I/O, invalid-format, resource-limit, or cancellation
    /// error.
    pub fn from_local_index_with_limits_and_cancellation(
        path: impl AsRef<Path>,
        limits: ResourceLimits,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        cancellation.check()?;
        let path = path.as_ref();
        let bytes = read_bounded_file(path, limits.max_header_bytes(), cancellation)?;
        cancellation.check()?;
        let index = ShardIndex::from_json_with_cancellation(&bytes, &limits, cancellation)?;
        cancellation.check()?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let canonical_parent = std::fs::canonicalize(parent)
            .map_err(|source| Error::io("failed to canonicalize shard-index directory", source))?;
        let mut sources = Vec::with_capacity(index.shards().len());
        for shard in index.shards() {
            cancellation.check()?;
            let joined = canonical_parent.join(repo_path_to_platform(&shard));
            let canonical_shard = std::fs::canonicalize(&joined)
                .map_err(|source| Error::io("failed to canonicalize checkpoint shard", source))?;
            if !canonical_shard.starts_with(&canonical_parent) {
                return Err(Error::new(
                    ErrorCategory::InvalidPath,
                    "checkpoint shard resolves outside the index directory",
                ));
            }
            sources
                .push(SourceDescriptor::local(canonical_shard)?.with_logical_path(shard.as_str())?);
        }
        Ok(Self {
            sources,
            shard_index: Some(index),
            limits,
            access: AccessMode::default(),
        })
    }

    /// Replaces the safetensors shard index used as membership truth.
    pub fn shard_index(mut self, shard_index: ShardIndex) -> Self {
        self.shard_index = Some(shard_index);
        self
    }

    /// Replaces untrusted-format resource limits.
    pub fn limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Selects mapped or copied tensor byte access.
    pub const fn access_mode(mut self, access: AccessMode) -> Self {
        self.access = access;
        self
    }

    /// Opens, inventories, and validates every configured source.
    ///
    /// Header parsing never scans or copies tensor payloads.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or duplicate sources, exceeded limits,
    /// malformed safetensors, or shard-index inconsistencies.
    pub fn open(self) -> Result<Checkpoint> {
        self.open_with_cancellation(&CancellationToken::new())
    }

    /// Opens, inventories, and validates sources with cancellation.
    ///
    /// Cancellation is checked between sources, bounded header-read blocks,
    /// tensor-entry validation steps, and shard-index entries.
    ///
    /// # Errors
    ///
    /// Returns any error described by [`Self::open`] or a cancellation error.
    pub fn open_with_cancellation(
        mut self,
        cancellation: &CancellationToken,
    ) -> Result<Checkpoint> {
        cancellation.check()?;
        if self.sources.is_empty() {
            return Err(Error::invalid(
                "checkpoint must contain at least one source",
            ));
        }
        if self.sources.len() > self.limits.max_shards() {
            return Err(Error::limit(
                "checkpoint exceeds the configured shard count",
            ));
        }
        if self.access == AccessMode::Mmap
            && self
                .sources
                .iter()
                .any(|source| source.kind() != SourceKind::RetainedSnapshot)
        {
            return Err(Error::unsupported(
                "mapped access requires retained immutable snapshot sources",
            ));
        }

        self.sources
            .sort_by(|left, right| left.logical_path().cmp(right.logical_path()));
        if self
            .sources
            .windows(2)
            .any(|pair| pair[0].logical_path() == pair[1].logical_path())
        {
            return Err(Error::invalid(
                "checkpoint contains duplicate logical source paths",
            ));
        }

        let mut open_files = Vec::with_capacity(self.sources.len());
        let mut file_records = Vec::with_capacity(self.sources.len());
        let mut tensor_records = Vec::new();
        for (ordinal, source) in self.sources.into_iter().enumerate() {
            cancellation.check()?;
            let ordinal = u32::try_from(ordinal)
                .map_err(|_error| Error::limit("checkpoint source ordinal exceeds u32"))?;
            let id = FileId::from_ordinal(ordinal);
            let file = Arc::new(OpenFile::open(source, id)?);
            let parsed = file.parse_header(&self.limits, cancellation)?;
            let remaining = self
                .limits
                .max_tensors()
                .checked_sub(tensor_records.len())
                .ok_or_else(|| Error::limit("checkpoint exceeds configured tensor count"))?;
            if parsed.len() > remaining {
                return Err(Error::limit(
                    "checkpoint exceeds the configured tensor count",
                ));
            }
            tensor_records.extend(parsed);
            file_records.push(file.record());
            open_files.push(file);
        }
        cancellation.check()?;
        let inventory = Inventory::new(file_records, tensor_records)?;
        if let Some(index) = &self.shard_index {
            validate_shard_index(index, &inventory, cancellation)?;
        }
        cancellation.check()?;

        Ok(Checkpoint {
            inner: Arc::new(CheckpointInner {
                inventory,
                files: open_files.into_boxed_slice(),
                access: self.access,
                snapshot: OnceLock::new(),
            }),
        })
    }
}

struct OpenFile {
    source: SourceDescriptor,
    id: FileId,
    file: File,
    size: u64,
    stamp: FileStamp,
    digest: Mutex<Option<ContentDigest>>,
    #[cfg(feature = "mmap")]
    mapping: Mutex<Option<Arc<MappedOwner>>>,
}

impl Debug for OpenFile {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenFile")
            .field("source", &self.source)
            .field("id", &self.id)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl OpenFile {
    fn open(source: SourceDescriptor, id: FileId) -> Result<Self> {
        let file = File::open(source.local_path())
            .map_err(|error| Error::io("failed to open checkpoint source", error))?;
        let metadata = file
            .metadata()
            .map_err(|error| Error::io("failed to inspect checkpoint source", error))?;
        if !metadata.is_file() {
            return Err(Error::invalid("checkpoint source is not a regular file"));
        }
        let size = metadata.len();
        if source
            .expected_size()
            .is_some_and(|expected| expected != size)
        {
            return Err(Error::integrity(
                "checkpoint source size differs from its declared size",
            ));
        }
        Ok(Self {
            source,
            id,
            file,
            size,
            stamp: FileStamp::from_metadata(&metadata),
            digest: Mutex::new(None),
            #[cfg(feature = "mmap")]
            mapping: Mutex::new(None),
        })
    }

    fn record(&self) -> FileRecord {
        FileRecord::new(
            self.id,
            self.source.logical_path().clone(),
            self.size,
            self.source.kind(),
            DigestState::from(self.source.digest_policy()),
        )
    }

    fn parse_header(
        &self,
        limits: &ResourceLimits,
        cancellation: &CancellationToken,
    ) -> Result<Vec<TensorRecord>> {
        cancellation.check()?;
        if self.size < 8 {
            return Err(Error::invalid(
                "safetensors source is shorter than its length prefix",
            ));
        }
        let mut prefix = [0_u8; 8];
        read_exact_at(&self.file, 0, &mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > limits.max_header_bytes() {
            return Err(Error::limit(
                "safetensors header length violates configured limits",
            ));
        }
        let data_base = 8_u64
            .checked_add(header_len)
            .ok_or_else(|| Error::limit("safetensors data offset overflows u64"))?;
        if data_base > self.size {
            return Err(Error::invalid(
                "safetensors header extends beyond the source file",
            ));
        }
        let header_size = usize::try_from(header_len)
            .map_err(|_error| Error::limit("safetensors header length does not fit usize"))?;
        let mut header = Vec::new();
        header
            .try_reserve_exact(header_size)
            .map_err(|_error| Error::limit("could not allocate the safetensors header"))?;
        header.resize(header_size, 0);
        for (block_index, block) in header.chunks_mut(SOURCE_READ_BLOCK_BYTES).enumerate() {
            cancellation.check()?;
            let relative = block_index
                .checked_mul(SOURCE_READ_BLOCK_BYTES)
                .ok_or_else(|| Error::limit("header block offset overflows usize"))?;
            let relative = u64::try_from(relative)
                .map_err(|_error| Error::limit("header block offset does not fit u64"))?;
            let offset = 8_u64
                .checked_add(relative)
                .ok_or_else(|| Error::limit("header block offset overflows u64"))?;
            read_exact_at(&self.file, offset, block)?;
        }
        cancellation.check()?;
        let raw = parse_header_json(&header, limits, cancellation)?;
        cancellation.check()?;
        validate_tensor_entries(
            self.id,
            data_base,
            self.size - data_base,
            raw,
            limits,
            cancellation,
        )
    }

    fn read_span(
        &self,
        span: SourceSpan,
        access: AccessMode,
        cancellation: &CancellationToken,
    ) -> Result<ByteView> {
        if span.file() != self.id || span.end() > self.size {
            return Err(Error::invalid(
                "validated source span lies outside its checkpoint file",
            ));
        }
        match access {
            AccessMode::Read => self.read_owned(span, cancellation),
            AccessMode::Auto if self.source.kind() == SourceKind::Local => {
                self.read_owned(span, cancellation)
            }
            AccessMode::Auto => match self.read_mapped(span) {
                Ok(view) => Ok(view),
                Err(error)
                    if matches!(
                        error.category(),
                        ErrorCategory::Io | ErrorCategory::Unsupported
                    ) =>
                {
                    self.read_owned(span, cancellation)
                }
                Err(error) => Err(error),
            },
            AccessMode::Mmap => self.read_mapped(span),
        }
    }

    fn read_owned(&self, span: SourceSpan, cancellation: &CancellationToken) -> Result<ByteView> {
        let length = usize::try_from(span.len())
            .map_err(|_error| Error::limit("tensor byte length does not fit usize"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_error| Error::limit("could not allocate tensor source bytes"))?;
        bytes.resize(length, 0);
        let mut bytes = bytes.into_boxed_slice();
        for (block_index, block) in bytes.chunks_mut(SOURCE_READ_BLOCK_BYTES).enumerate() {
            cancellation.check()?;
            let relative = block_index
                .checked_mul(SOURCE_READ_BLOCK_BYTES)
                .ok_or_else(|| Error::limit("tensor source block offset overflows usize"))?;
            let relative = u64::try_from(relative)
                .map_err(|_error| Error::limit("tensor source block offset does not fit u64"))?;
            let offset = span
                .offset()
                .checked_add(relative)
                .ok_or_else(|| Error::limit("tensor source block offset overflows u64"))?;
            read_exact_at(&self.file, offset, block)?;
        }
        cancellation.check()?;
        Ok(ByteView::from_boxed(bytes))
    }

    #[cfg(feature = "mmap")]
    #[expect(
        unsafe_code,
        reason = "memmap2 requires an unsafe call after the retained immutable lifetime is established"
    )]
    fn read_mapped(&self, span: SourceSpan) -> Result<ByteView> {
        if self.source.kind() != SourceKind::RetainedSnapshot {
            return Err(Error::unsupported(
                "mapped access requires a retained immutable snapshot",
            ));
        }
        let mut slot = self
            .mapping
            .lock()
            .map_err(|_error| Error::integrity("checkpoint mapping lock is poisoned"))?;
        if slot.is_none() {
            // SAFETY: `self.file` remains open for the mapping lifetime, the
            // retained snapshot guard prevents removal or mutation, and views
            // hold the resulting owner independently of the checkpoint.
            let mapping = unsafe { memmap2::MmapOptions::new().map(&self.file) }
                .map_err(|source| Error::io("failed to map checkpoint source", source))?;
            *slot = Some(Arc::new(MappedOwner {
                mapping,
                _retention: self.source.retention().ok_or_else(|| {
                    Error::integrity("retained snapshot source has no lifetime guard")
                })?,
            }));
        }
        let owner = Arc::clone(
            slot.as_ref()
                .ok_or_else(|| Error::integrity("checkpoint mapping was not initialized"))?,
        );
        drop(slot);
        let start = usize::try_from(span.offset())
            .map_err(|_error| Error::limit("tensor byte offset does not fit usize"))?;
        let end = usize::try_from(span.end())
            .map_err(|_error| Error::limit("tensor byte end does not fit usize"))?;
        let erased: Arc<dyn ByteOwner> = owner;
        ByteView::from_owner(erased, start..end)
    }

    #[cfg(not(feature = "mmap"))]
    #[expect(
        clippy::unused_self,
        reason = "the feature-independent dispatch keeps mapped access on the same source instance"
    )]
    fn read_mapped(&self, _span: SourceSpan) -> Result<ByteView> {
        Err(Error::unsupported(
            "mapped access requires the model-weights `mmap` feature",
        ))
    }

    fn needs_hashing(&self, cancellation: &CancellationToken) -> Result<bool> {
        if matches!(self.source.digest_policy(), DigestPolicy::TrustRetained(_)) {
            return Ok(false);
        }
        loop {
            cancellation.check()?;
            match self.digest.try_lock() {
                Ok(cached) => return Ok(cached.is_none()),
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(std::sync::TryLockError::Poisoned(_error)) => {
                    return Err(Error::integrity("checkpoint digest lock is poisoned"));
                }
            }
        }
    }

    fn content_digest(&self, cancellation: &CancellationToken) -> Result<ContentDigest> {
        let mut cached = loop {
            cancellation.check()?;
            match self.digest.try_lock() {
                Ok(cached) => break cached,
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(std::sync::TryLockError::Poisoned(_error)) => {
                    return Err(Error::integrity("checkpoint digest lock is poisoned"));
                }
            }
        };
        if let Some(digest) = *cached {
            return Ok(digest);
        }
        if let DigestPolicy::TrustRetained(digest) = self.source.digest_policy() {
            *cached = Some(digest);
            return Ok(digest);
        }

        let mut hasher = Sha256::new();
        let mut offset = 0_u64;
        let mut buffer = vec![0_u8; 1024 * 1024];
        let buffer_length = u64::try_from(buffer.len())
            .map_err(|_error| Error::limit("digest buffer length does not fit u64"))?;
        while offset < self.size {
            cancellation.check()?;
            let remaining = self.size - offset;
            let amount = usize::try_from(remaining.min(buffer_length))
                .map_err(|_error| Error::limit("digest read length does not fit usize"))?;
            read_exact_at(&self.file, offset, &mut buffer[..amount])?;
            hasher.update(&buffer[..amount]);
            offset = offset
                .checked_add(u64::try_from(amount).map_err(|_error| {
                    Error::limit("digest read length does not fit source offset")
                })?)
                .ok_or_else(|| Error::limit("digest source offset overflows u64"))?;
        }
        cancellation.check()?;
        let metadata = self
            .file
            .metadata()
            .map_err(|source| Error::io("failed to re-inspect checkpoint source", source))?;
        if FileStamp::from_metadata(&metadata) != self.stamp {
            return Err(Error::integrity(
                "ordinary checkpoint source changed while being hashed",
            ));
        }
        let digest = ContentDigest::from_bytes(hasher.finalize().into());
        if let DigestPolicy::VerifyOnDemand(expected) = self.source.digest_policy() {
            if digest != expected {
                return Err(Error::integrity(
                    "checkpoint source digest differs from its declared digest",
                ));
            }
        }
        *cached = Some(digest);
        Ok(digest)
    }
}

#[cfg(feature = "mmap")]
struct MappedOwner {
    mapping: memmap2::Mmap,
    _retention: Arc<dyn std::any::Any + Send + Sync>,
}

#[cfg(feature = "mmap")]
impl Debug for MappedOwner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MappedOwner")
            .field("length", &self.mapping.len())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "mmap")]
impl ByteOwner for MappedOwner {
    fn bytes(&self) -> &[u8] {
        &self.mapping
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    length: u64,
    modified: Option<SystemTime>,
}

impl FileStamp {
    fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            length: metadata.len(),
            modified: metadata.modified().ok(),
        }
    }
}

#[derive(Debug)]
struct RawTensor {
    dtype: Box<str>,
    shape: Box<[u64]>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct RawHeader {
    tensors: BTreeMap<Box<str>, RawTensor>,
}

fn parse_header_json(
    bytes: &[u8],
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<RawHeader> {
    cancellation.check()?;
    if bytes.first() != Some(&b'{') {
        return Err(Error::invalid(
            "safetensors header must begin with a JSON object",
        ));
    }
    let control = HeaderParseControl::new(limits, cancellation);
    deserialize_header_json(bytes, &control)
}

fn deserialize_header_json(bytes: &[u8], control: &HeaderParseControl<'_>) -> Result<RawHeader> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let header = match (HeaderSeed { control }).deserialize(&mut deserializer) {
        Ok(header) => header,
        Err(source) => {
            return Err(control.classify_json_error(source, "safetensors header JSON is invalid"));
        }
    };
    control.cancellation.check()?;
    if let Err(source) = deserializer.end() {
        return Err(control.classify_json_error(
            source,
            "safetensors header contains trailing non-whitespace data",
        ));
    }
    control.cancellation.check()?;
    Ok(header)
}

#[derive(Debug)]
struct HeaderParseControl<'a> {
    limits: &'a ResourceLimits,
    cancellation: &'a CancellationToken,
    failure: RefCell<Option<Error>>,
    #[cfg(test)]
    cancel_after_checks: Cell<Option<usize>>,
}

impl<'a> HeaderParseControl<'a> {
    fn new(limits: &'a ResourceLimits, cancellation: &'a CancellationToken) -> Self {
        Self {
            limits,
            cancellation,
            failure: RefCell::new(None),
            #[cfg(test)]
            cancel_after_checks: Cell::new(None),
        }
    }

    #[cfg(test)]
    fn cancelling_after_checks(
        limits: &'a ResourceLimits,
        cancellation: &'a CancellationToken,
        checks: usize,
    ) -> Self {
        let control = Self::new(limits, cancellation);
        control.cancel_after_checks.set(Some(checks));
        control
    }

    fn check<E>(&self) -> std::result::Result<(), E>
    where
        E: serde::de::Error,
    {
        #[cfg(test)]
        if let Some(remaining) = self.cancel_after_checks.get() {
            if remaining <= 1 {
                self.cancel_after_checks.set(None);
                self.cancellation.cancel();
            } else {
                self.cancel_after_checks.set(Some(remaining - 1));
            }
        }
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
        E::custom("bounded safetensors header parsing aborted")
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

struct HeaderSeed<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
}

impl<'de> DeserializeSeed<'de> for HeaderSeed<'_, '_> {
    type Value = RawHeader;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_map(HeaderVisitor {
            control: self.control,
        })
    }
}

struct HeaderVisitor<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
}

impl<'de> Visitor<'de> for HeaderVisitor<'_, '_> {
    type Value = RawHeader;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors header object")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut tensors = BTreeMap::new();
        let mut seen_metadata = false;
        loop {
            self.control.check()?;
            let Some(name) = map.next_key_seed(HeaderStringSeed {
                control: self.control,
                maximum: self
                    .control
                    .limits
                    .max_name_bytes()
                    .max("__metadata__".len()),
                reject_empty: true,
                limit_message: "safetensors tensor or dtype name violates configured limits",
            })?
            else {
                break;
            };
            self.control.check()?;
            if name.as_ref() == "__metadata__" {
                if seen_metadata {
                    return Err(serde::de::Error::custom(
                        "duplicate safetensors metadata key",
                    ));
                }
                seen_metadata = true;
                map.next_value_seed(MetadataSeed {
                    control: self.control,
                })?;
                continue;
            }
            if name.len() > self.control.limits.max_name_bytes() {
                return Err(self
                    .control
                    .limit("safetensors tensor or dtype name violates configured limits"));
            }
            if tensors.contains_key(name.as_ref()) {
                return Err(serde::de::Error::custom(format_args!(
                    "duplicate safetensors tensor name {name:?}"
                )));
            }
            if tensors.len() >= self.control.limits.max_tensors() {
                return Err(self
                    .control
                    .limit("safetensors header exceeds the configured tensor count"));
            }
            let raw = map.next_value_seed(RawTensorSeed {
                control: self.control,
            })?;
            tensors.insert(name, raw);
        }
        Ok(RawHeader { tensors })
    }
}

#[derive(Debug, Clone, Copy)]
enum RawTensorField {
    Dtype,
    Shape,
    DataOffsets,
}

impl<'de> Deserialize<'de> for RawTensorField {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(RawTensorFieldVisitor)
    }
}

struct RawTensorFieldVisitor;

impl Visitor<'_> for RawTensorFieldVisitor {
    type Value = RawTensorField;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("dtype, shape, or data_offsets")
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        match value {
            "dtype" => Ok(RawTensorField::Dtype),
            "shape" => Ok(RawTensorField::Shape),
            "data_offsets" => Ok(RawTensorField::DataOffsets),
            _ => Err(E::unknown_field(value, &["dtype", "shape", "data_offsets"])),
        }
    }
}

struct RawTensorSeed<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
}

impl<'de> DeserializeSeed<'de> for RawTensorSeed<'_, '_> {
    type Value = RawTensor;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_map(RawTensorVisitor {
            control: self.control,
        })
    }
}

struct RawTensorVisitor<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
}

impl<'de> Visitor<'de> for RawTensorVisitor<'_, '_> {
    type Value = RawTensor;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors tensor descriptor")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut dtype = None;
        let mut shape = None;
        let mut data_offsets = None;
        loop {
            self.control.check()?;
            let Some(field) = map.next_key::<RawTensorField>()? else {
                break;
            };
            self.control.check()?;
            match field {
                RawTensorField::Dtype => {
                    if dtype.is_some() {
                        return Err(serde::de::Error::duplicate_field("dtype"));
                    }
                    dtype = Some(map.next_value_seed(HeaderStringSeed {
                        control: self.control,
                        maximum: self.control.limits.max_name_bytes(),
                        reject_empty: false,
                        limit_message:
                            "safetensors tensor or dtype name violates configured limits",
                    })?);
                }
                RawTensorField::Shape => {
                    if shape.is_some() {
                        return Err(serde::de::Error::duplicate_field("shape"));
                    }
                    shape = Some(map.next_value_seed(ShapeSeed {
                        control: self.control,
                    })?);
                }
                RawTensorField::DataOffsets => {
                    if data_offsets.is_some() {
                        return Err(serde::de::Error::duplicate_field("data_offsets"));
                    }
                    data_offsets = Some(map.next_value::<[u64; 2]>()?);
                }
            }
        }
        Ok(RawTensor {
            dtype: dtype.ok_or_else(|| serde::de::Error::missing_field("dtype"))?,
            shape: shape.ok_or_else(|| serde::de::Error::missing_field("shape"))?,
            data_offsets: data_offsets
                .ok_or_else(|| serde::de::Error::missing_field("data_offsets"))?,
        })
    }
}

struct ShapeSeed<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
}

impl<'de> DeserializeSeed<'de> for ShapeSeed<'_, '_> {
    type Value = Box<[u64]>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_seq(ShapeVisitor {
            control: self.control,
        })
    }
}

struct ShapeVisitor<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
}

impl<'de> Visitor<'de> for ShapeVisitor<'_, '_> {
    type Value = Box<[u64]>;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded tensor shape")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        if sequence
            .size_hint()
            .is_some_and(|length| length > self.control.limits.max_rank())
        {
            return Err(self
                .control
                .limit("safetensors tensor rank exceeds configured limits"));
        }
        let mut shape = Vec::new();
        loop {
            self.control.check()?;
            let Some(dimension) = sequence.next_element::<u64>()? else {
                break;
            };
            if shape.len() >= self.control.limits.max_rank() {
                return Err(self
                    .control
                    .limit("safetensors tensor rank exceeds configured limits"));
            }
            shape.push(dimension);
        }
        Ok(shape.into_boxed_slice())
    }
}

struct HeaderStringSeed<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
    maximum: usize,
    reject_empty: bool,
    limit_message: &'static str,
}

impl<'de> DeserializeSeed<'de> for HeaderStringSeed<'_, '_> {
    type Value = Box<str>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_str(HeaderStringVisitor {
            control: self.control,
            maximum: self.maximum,
            reject_empty: self.reject_empty,
            limit_message: self.limit_message,
        })
    }
}

struct HeaderStringVisitor<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
    maximum: usize,
    reject_empty: bool,
    limit_message: &'static str,
}

impl HeaderStringVisitor<'_, '_> {
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

impl<'de> Visitor<'de> for HeaderStringVisitor<'_, '_> {
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

struct MetadataSeed<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
}

impl<'de> DeserializeSeed<'de> for MetadataSeed<'_, '_> {
    type Value = usize;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        self.control.check()?;
        deserializer.deserialize_map(MetadataVisitor {
            control: self.control,
        })
    }
}

struct MetadataVisitor<'control, 'context> {
    control: &'control HeaderParseControl<'context>,
}

impl<'de> Visitor<'de> for MetadataVisitor<'_, '_> {
    type Value = usize;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors string metadata map")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        let mut bytes = 0_usize;
        loop {
            self.control.check()?;
            let remaining = self
                .control
                .limits
                .max_metadata_bytes()
                .saturating_sub(bytes);
            let Some(key) = map.next_key_seed(HeaderStringSeed {
                control: self.control,
                maximum: remaining,
                reject_empty: false,
                limit_message: "safetensors metadata exceeds configured limits",
            })?
            else {
                break;
            };
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format_args!(
                    "duplicate safetensors metadata key {key:?}"
                )));
            }
            bytes += key.len();
            let remaining = self
                .control
                .limits
                .max_metadata_bytes()
                .saturating_sub(bytes);
            let value = map.next_value_seed(HeaderStringSeed {
                control: self.control,
                maximum: remaining,
                reject_empty: false,
                limit_message: "safetensors metadata exceeds configured limits",
            })?;
            bytes += value.len();
        }
        Ok(bytes)
    }
}

fn validate_tensor_entries(
    file: FileId,
    data_base: u64,
    data_length: u64,
    raw: RawHeader,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<Vec<TensorRecord>> {
    struct Entry {
        name: Box<str>,
        shape: Box<[u64]>,
        dtype: HeaderDtype,
        start: u64,
        end: u64,
    }

    let mut entries = Vec::with_capacity(raw.tensors.len());
    for (name, tensor) in raw.tensors {
        cancellation.check()?;
        if tensor.shape.len() > limits.max_rank() {
            return Err(Error::limit(
                "safetensors tensor rank exceeds configured limits",
            ));
        }
        let dtype = parse_dtype(&tensor.dtype)?;
        let [start, end] = tensor.data_offsets;
        if start > end || end > data_length {
            return Err(Error::invalid(
                "safetensors tensor offset lies outside its data section",
            ));
        }
        let expected = dtype.byte_len(&tensor.shape)?;
        if end - start != expected {
            return Err(Error::invalid(
                "safetensors tensor byte length disagrees with dtype and shape",
            ));
        }
        entries.push(Entry {
            name,
            shape: tensor.shape,
            dtype,
            start,
            end,
        });
    }
    entries.sort_by(|left, right| {
        (left.start, left.end, left.name.as_ref()).cmp(&(
            right.start,
            right.end,
            right.name.as_ref(),
        ))
    });

    let mut cursor = 0_u64;
    for entry in &entries {
        cancellation.check()?;
        if entry.start != cursor {
            return Err(Error::invalid(
                "safetensors data contains an overlap or unclaimed gap",
            ));
        }
        cursor = entry.end;
    }
    if cursor != data_length {
        return Err(Error::invalid(
            "safetensors data section contains unclaimed trailing bytes",
        ));
    }

    let mut records = Vec::with_capacity(entries.len());
    for entry in entries {
        cancellation.check()?;
        let offset = data_base
            .checked_add(entry.start)
            .ok_or_else(|| Error::limit("absolute tensor byte offset overflows u64"))?;
        let span = SourceSpan::new(file, offset, entry.end - entry.start)?;
        let storage = entry.dtype.into_storage(&entry.shape, span)?;
        records.push(TensorRecord::new(entry.name, entry.shape, storage));
    }
    cancellation.check()?;
    Ok(records)
}

#[derive(Debug, Clone, Copy)]
enum HeaderDtype {
    Plain(DType),
    Quantized {
        operation: &'static str,
        values_per_block: u32,
        bytes_per_block: u32,
    },
}

impl HeaderDtype {
    fn byte_len(self, shape: &[u64]) -> Result<u64> {
        match self {
            Self::Plain(dtype) => dtype.byte_len(shape),
            Self::Quantized {
                values_per_block,
                bytes_per_block,
                ..
            } => {
                let elements = shape.iter().try_fold(1_u64, |product, dimension| {
                    product
                        .checked_mul(*dimension)
                        .ok_or_else(|| Error::limit("tensor element count overflows u64"))
                })?;
                if elements == 0 {
                    return Ok(0);
                }
                let values_per_block = u64::from(values_per_block);
                if elements % values_per_block != 0 {
                    return Err(Error::invalid(
                        "sub-byte safetensors dtype is not byte-aligned",
                    ));
                }
                (elements / values_per_block)
                    .checked_mul(u64::from(bytes_per_block))
                    .ok_or_else(|| Error::limit("packed tensor byte length overflows u64"))
            }
        }
    }

    fn into_storage(self, shape: &[u64], span: SourceSpan) -> Result<Storage> {
        match self {
            Self::Plain(dtype) => Ok(Storage::Plain { dtype, span }),
            Self::Quantized {
                operation,
                values_per_block,
                bytes_per_block,
            } => {
                let encoding = ImplementationId::new(
                    StableName::parse("safetensors")?,
                    StableName::parse(operation)?,
                    1,
                );
                Ok(Storage::Quantized(QuantizedStorage::new(
                    encoding,
                    shape,
                    span,
                    Packing::flat_blocks(values_per_block, bytes_per_block)?,
                )?))
            }
        }
    }
}

fn parse_dtype(value: &str) -> Result<HeaderDtype> {
    match value {
        "BOOL" => Ok(HeaderDtype::Plain(DType::Bool)),
        "F4" => Ok(HeaderDtype::Quantized {
            operation: "f4",
            values_per_block: 2,
            bytes_per_block: 1,
        }),
        "F6_E2M3" => Ok(HeaderDtype::Quantized {
            operation: "f6-e2m3",
            values_per_block: 4,
            bytes_per_block: 3,
        }),
        "F6_E3M2" => Ok(HeaderDtype::Quantized {
            operation: "f6-e3m2",
            values_per_block: 4,
            bytes_per_block: 3,
        }),
        "U8" => Ok(HeaderDtype::Plain(DType::U8)),
        "I8" => Ok(HeaderDtype::Plain(DType::I8)),
        "U16" => Ok(HeaderDtype::Plain(DType::U16)),
        "I16" => Ok(HeaderDtype::Plain(DType::I16)),
        "U32" => Ok(HeaderDtype::Plain(DType::U32)),
        "I32" => Ok(HeaderDtype::Plain(DType::I32)),
        "U64" => Ok(HeaderDtype::Plain(DType::U64)),
        "I64" => Ok(HeaderDtype::Plain(DType::I64)),
        "F16" => Ok(HeaderDtype::Plain(DType::F16)),
        "BF16" => Ok(HeaderDtype::Plain(DType::Bf16)),
        "F32" => Ok(HeaderDtype::Plain(DType::F32)),
        "F64" => Ok(HeaderDtype::Plain(DType::F64)),
        "C64" => Ok(HeaderDtype::Plain(DType::C64)),
        "F8_E5M2" => Ok(HeaderDtype::Plain(DType::F8E5M2)),
        "F8_E4M3" => Ok(HeaderDtype::Plain(DType::F8E4M3)),
        "F8_E8M0" => Ok(HeaderDtype::Plain(DType::F8E8M0)),
        "F8_E4M3FNUZ" => Ok(HeaderDtype::Plain(DType::F8E4M3Fnuz)),
        "F8_E5M2FNUZ" => Ok(HeaderDtype::Plain(DType::F8E5M2Fnuz)),
        _ => Err(Error::unsupported(
            "safetensors dtype is not represented by this crate version",
        )),
    }
}

fn validate_shard_index(
    index: &ShardIndex,
    inventory: &Inventory,
    cancellation: &CancellationToken,
) -> Result<()> {
    if index.len() != inventory.len() {
        return Err(Error::integrity(
            "safetensors shard index and checkpoint inventory differ in size",
        ));
    }
    for (name, expected_path) in index.iter() {
        cancellation.check()?;
        let tensor = inventory
            .tensor(name)
            .ok_or_else(|| Error::integrity("shard index names a missing tensor"))?;
        let file_index = usize::try_from(tensor.storage().span().file().ordinal())
            .map_err(|_error| Error::limit("source file identifier does not fit usize"))?;
        let actual_path = inventory
            .files()
            .get(file_index)
            .ok_or_else(|| Error::integrity("tensor refers to an unknown source file"))?
            .path();
        if actual_path != expected_path {
            return Err(Error::integrity(
                "tensor is present in a different shard than its index declares",
            ));
        }
    }
    cancellation.check()?;
    Ok(())
}

fn read_bounded_file(
    path: &Path,
    byte_limit: u64,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>> {
    cancellation.check()?;
    let mut file =
        File::open(path).map_err(|source| Error::io("failed to open metadata file", source))?;
    let size = file
        .metadata()
        .map_err(|source| Error::io("failed to inspect metadata file", source))?
        .len();
    if size > byte_limit {
        return Err(Error::limit(
            "metadata file exceeds the configured byte limit",
        ));
    }
    let capacity = usize::try_from(size)
        .map_err(|_error| Error::limit("metadata file length does not fit usize"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_error| Error::limit("could not allocate metadata file bytes"))?;
    let read_limit = byte_limit.saturating_add(1);
    let mut remaining = read_limit;
    let mut block = Vec::new();
    block
        .try_reserve_exact(64 * 1024)
        .map_err(|_error| Error::limit("could not allocate metadata read block"))?;
    block.resize(64 * 1024, 0);
    let block_len = u64::try_from(block.len())
        .map_err(|_error| Error::limit("metadata read block length does not fit u64"))?;
    while remaining > 0 {
        cancellation.check()?;
        let amount = usize::try_from(remaining.min(block_len))
            .map_err(|_error| Error::limit("metadata read length does not fit usize"))?;
        let read = file
            .read(&mut block[..amount])
            .map_err(|source| Error::io("failed to read metadata file", source))?;
        if read == 0 {
            break;
        }
        bytes
            .try_reserve(read)
            .map_err(|_error| Error::limit("could not grow metadata file bytes"))?;
        bytes.extend_from_slice(&block[..read]);
        remaining = remaining.saturating_sub(
            u64::try_from(read)
                .map_err(|_error| Error::limit("metadata read length does not fit u64"))?,
        );
    }
    cancellation.check()?;
    let actual = u64::try_from(bytes.len())
        .map_err(|_error| Error::limit("metadata file length does not fit u64"))?;
    if actual > byte_limit {
        return Err(Error::limit(
            "metadata file grew beyond the configured byte limit",
        ));
    }
    Ok(bytes)
}

fn repo_path_to_platform(path: &RepoPath) -> PathBuf {
    path.as_str().split('/').collect()
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut offset: u64, mut bytes: &mut [u8]) -> Result<()> {
    use std::os::unix::fs::FileExt as _;

    while !bytes.is_empty() {
        let amount = file
            .read_at(bytes, offset)
            .map_err(|source| Error::io("failed to read checkpoint source", source))?;
        if amount == 0 {
            return Err(Error::integrity(
                "checkpoint source ended before a validated span",
            ));
        }
        offset = offset
            .checked_add(
                u64::try_from(amount)
                    .map_err(|_error| Error::limit("read length does not fit source offset"))?,
            )
            .ok_or_else(|| Error::limit("checkpoint read offset overflows u64"))?;
        bytes = &mut bytes[amount..];
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut offset: u64, mut bytes: &mut [u8]) -> Result<()> {
    use std::os::windows::fs::FileExt as _;

    while !bytes.is_empty() {
        let amount = file
            .seek_read(bytes, offset)
            .map_err(|source| Error::io("failed to read checkpoint source", source))?;
        if amount == 0 {
            return Err(Error::integrity(
                "checkpoint source ended before a validated span",
            ));
        }
        offset = offset
            .checked_add(
                u64::try_from(amount)
                    .map_err(|_error| Error::limit("read length does not fit source offset"))?,
            )
            .ok_or_else(|| Error::limit("checkpoint read offset overflows u64"))?;
        bytes = &mut bytes[amount..];
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, offset: u64, bytes: &mut [u8]) -> Result<()> {
    let mut clone = file
        .try_clone()
        .map_err(|source| Error::io("failed to clone checkpoint source handle", source))?;
    clone
        .seek(SeekFrom::Start(offset))
        .map_err(|source| Error::io("failed to seek checkpoint source", source))?;
    clone
        .read_exact(bytes)
        .map_err(|source| Error::io("failed to read checkpoint source", source))
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;

#[cfg(test)]
mod bounded_header_parser_tests {
    use super::{HeaderParseControl, deserialize_header_json, parse_header_json};
    use crate::limits::ResourceLimits;
    use crate::{CancellationToken, ErrorCategory, Result};

    fn limits() -> Result<ResourceLimits> {
        ResourceLimits::builder()
            .max_tensors(1)
            .max_rank(1)
            .max_name_bytes(4)
            .max_metadata_bytes(4)
            .build()
    }

    #[test]
    fn tensor_count_limit_precedes_a_later_invalid_tensor_value() -> Result<()> {
        let bytes = br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":false}"#;
        let error = parse_header_json(bytes, &limits()?, &CancellationToken::new())
            .expect_err("the second tensor must exceed the configured count");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn tensor_name_limit_precedes_its_invalid_value() -> Result<()> {
        let bytes = br#"{"oversized":false}"#;
        let error = parse_header_json(bytes, &limits()?, &CancellationToken::new())
            .expect_err("the tensor name must exceed the configured length");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn dtype_limit_precedes_a_later_invalid_shape() -> Result<()> {
        let bytes = br#"{"a":{"dtype":"TOO_LONG","shape":false,"data_offsets":[0,0]}}"#;
        let error = parse_header_json(bytes, &limits()?, &CancellationToken::new())
            .expect_err("the dtype must exceed the configured length");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn rank_limit_precedes_later_invalid_offsets() -> Result<()> {
        let bytes = br#"{"a":{"dtype":"U8","shape":[1,2],"data_offsets":false}}"#;
        let error = parse_header_json(bytes, &limits()?, &CancellationToken::new())
            .expect_err("the shape must exceed the configured rank");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn metadata_limit_precedes_a_later_invalid_tensor() -> Result<()> {
        let bytes = br#"{"__metadata__":{"key":"value"},"a":false}"#;
        let error = parse_header_json(bytes, &limits()?, &CancellationToken::new())
            .expect_err("the metadata must exceed the configured byte budget");

        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        Ok(())
    }

    #[test]
    fn metadata_within_the_limit_preserves_valid_tensor_entries() -> Result<()> {
        let bytes = br#"{"__metadata__":{"format":"pt"},"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let header =
            parse_header_json(bytes, &ResourceLimits::default(), &CancellationToken::new())?;

        assert_eq!(header.tensors.len(), 1);
        Ok(())
    }

    #[test]
    fn cancellation_during_nested_header_parsing_preserves_its_category() -> Result<()> {
        let limits = limits()?;
        let cancellation = CancellationToken::new();
        let control = HeaderParseControl::cancelling_after_checks(&limits, &cancellation, 8);
        let bytes = br#"{"a":{"dtype":"U8","shape":false,"data_offsets":[0,0]}}"#;
        let error = deserialize_header_json(bytes, &control)
            .expect_err("the deterministic parser yield point must cancel");

        assert_eq!(error.category(), ErrorCategory::Cancelled);
        Ok(())
    }
}
