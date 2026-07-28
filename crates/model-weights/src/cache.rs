//! Content-addressed plan and prepared-weight caching.
//!
//! Entries are immutable directories published by a same-filesystem rename.
//! By default readers validate the versioned metadata envelope, compatibility
//! contract, payload length, and SHA-256 digest before receiving an open file
//! handle. Trusted cache roots may explicitly select metadata-only validation.

#[cfg(windows)]
use std::error::Error as StdError;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::identity::ContentDigest;
use crate::{CancellationToken, Error, Result};

/// Version of the on-disk cache metadata envelope.
pub const CACHE_ENVELOPE_VERSION: u32 = 1;

const CACHE_DIRECTORY_VERSION: &str = "v1";
const METADATA_FILE: &str = "metadata.json";
const PAYLOAD_FILE: &str = "payload.bin";
const OWNER_FILE: &str = "owner";
const HEARTBEAT_FILE: &str = "heartbeat";
const REPLACEMENT_MARKER: &str = ".previous-";
const MAX_METADATA_BYTES: u64 = 64 * 1024;
const COPY_BUFFER_BYTES: usize = 256 * 1024;
const DEFAULT_LEASE_WAIT: Duration = Duration::from_secs(30);
const DEFAULT_STALE_AFTER: Duration = Duration::from_secs(30 * 60);
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const DEFAULT_MAXIMUM_SCAN_ENTRIES: usize = 1_000_000;
const MAXIMUM_CACHE_SCAN_DEPTH: usize = 16;
const LEASE_RELEASE_RETRIES: usize = 100;
#[cfg(windows)]
const LEASE_RELEASE_RETRY_INTERVAL: Duration = Duration::from_millis(1);

static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A physically separate cache namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheNamespace {
    /// Canonical binding plans and their inert metadata.
    Plan,
    /// Potentially large prepared tensor representations.
    Prepared,
}

impl CacheNamespace {
    const fn directory_name(self) -> &'static str {
        match self {
            Self::Plan => "plans",
            Self::Prepared => "prepared",
        }
    }

    const fn key_tag(self) -> &'static [u8] {
        match self {
            Self::Plan => b"plan",
            Self::Prepared => b"prepared",
        }
    }
}

impl Display for CacheNamespace {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.directory_name())
    }
}

/// Compatibility facts validated independently from payload identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheCompatibility {
    format_version: u32,
    transform: ContentDigest,
    backend_abi: Option<ContentDigest>,
}

impl CacheCompatibility {
    /// Creates a compatibility contract.
    ///
    /// `transform` identifies every byte-affecting implementation choice.
    /// Prepared entries should also provide a consumer backend/layout ABI
    /// digest; plan entries normally use `None`.
    #[must_use]
    pub const fn new(
        format_version: u32,
        transform: ContentDigest,
        backend_abi: Option<ContentDigest>,
    ) -> Self {
        Self {
            format_version,
            transform,
            backend_abi,
        }
    }

    /// Creates a plan compatibility contract.
    #[must_use]
    pub const fn plan(format_version: u32, transform: ContentDigest) -> Self {
        Self::new(format_version, transform, None)
    }

    /// Creates a prepared-representation compatibility contract.
    #[must_use]
    pub const fn prepared(
        format_version: u32,
        transform: ContentDigest,
        backend_abi: ContentDigest,
    ) -> Self {
        Self::new(format_version, transform, Some(backend_abi))
    }

    /// Returns the caller-defined payload format version.
    #[must_use]
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns the byte-affecting transform implementation digest.
    #[must_use]
    pub const fn transform(&self) -> ContentDigest {
        self.transform
    }

    /// Returns the backend/layout ABI digest, when one is required.
    #[must_use]
    pub const fn backend_abi(&self) -> Option<ContentDigest> {
        self.backend_abi
    }
}

/// A deterministic SHA-256 cache address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CacheKey(ContentDigest);

impl CacheKey {
    /// Derives a key from namespace, compatibility, and ordered identities.
    ///
    /// Identity parts normally include immutable source, selection, plan, and
    /// target representation identities. Every part is length-prefixed, so
    /// different part boundaries cannot collide.
    #[must_use]
    pub fn derive(
        namespace: CacheNamespace,
        compatibility: &CacheCompatibility,
        identities: impl IntoIterator<Item = impl AsRef<[u8]>>,
    ) -> Self {
        let mut hasher = Sha256::new();
        update_hash_part(&mut hasher, b"model-weights-rs-cache-key-v1");
        update_hash_part(&mut hasher, namespace.key_tag());
        update_hash_part(&mut hasher, &compatibility.format_version.to_le_bytes());
        update_hash_part(&mut hasher, compatibility.transform.as_bytes());
        match compatibility.backend_abi {
            Some(backend_abi) => {
                update_hash_part(&mut hasher, &[1]);
                update_hash_part(&mut hasher, backend_abi.as_bytes());
            }
            None => update_hash_part(&mut hasher, &[0]),
        }
        for identity in identities {
            update_hash_part(&mut hasher, identity.as_ref());
        }
        Self(ContentDigest::from_bytes(hasher.finalize().into()))
    }

    /// Creates a cache key from an already derived digest.
    #[must_use]
    pub const fn from_digest(digest: ContentDigest) -> Self {
        Self(digest)
    }

    /// Returns the underlying digest.
    #[must_use]
    pub const fn digest(self) -> ContentDigest {
        self.0
    }
}

impl Display for CacheKey {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, formatter)
    }
}

impl FromStr for CacheKey {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        value.parse().map(Self)
    }
}

/// Runtime bounds for cooperative cache leases and maintenance scans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheOptions {
    lease_wait: Duration,
    stale_after: Duration,
    poll_interval: Duration,
    maximum_scan_entries: usize,
}

impl CacheOptions {
    /// Sets how long a publisher waits for another writer of the same key.
    #[must_use]
    pub const fn lease_wait(mut self, lease_wait: Duration) -> Self {
        self.lease_wait = lease_wait;
        self
    }

    /// Sets the minimum age before interrupted staging data may be recovered.
    #[must_use]
    pub const fn stale_after(mut self, stale_after: Duration) -> Self {
        self.stale_after = stale_after;
        self
    }

    /// Sets the interval between cooperative lease attempts.
    #[must_use]
    pub const fn poll_interval(mut self, poll_interval: Duration) -> Self {
        self.poll_interval = poll_interval;
        self
    }

    /// Sets the maximum filesystem entries visited by one maintenance scan.
    ///
    /// This bounds startup recovery, inspection, eviction accounting, and
    /// interrupted-write recovery independently of cache contents.
    #[must_use]
    pub const fn maximum_scan_entries(mut self, maximum_scan_entries: usize) -> Self {
        self.maximum_scan_entries = maximum_scan_entries;
        self
    }

    /// Returns the per-key lease wait limit.
    #[must_use]
    pub const fn lease_wait_duration(&self) -> Duration {
        self.lease_wait
    }

    /// Returns the interrupted-write recovery age.
    #[must_use]
    pub const fn stale_after_duration(&self) -> Duration {
        self.stale_after
    }

    /// Returns the cooperative lease retry interval.
    #[must_use]
    pub const fn poll_interval_duration(&self) -> Duration {
        self.poll_interval
    }

    /// Returns the maximum filesystem entries visited by one maintenance scan.
    #[must_use]
    pub const fn maximum_scan_entries_limit(&self) -> usize {
        self.maximum_scan_entries
    }
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            lease_wait: DEFAULT_LEASE_WAIT,
            stale_after: DEFAULT_STALE_AFTER,
            poll_interval: DEFAULT_POLL_INTERVAL,
            maximum_scan_entries: DEFAULT_MAXIMUM_SCAN_ENTRIES,
        }
    }
}

/// Integrity work performed while opening a cache entry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CacheValidation {
    /// Validate metadata, compatibility, payload length, and payload SHA-256.
    #[default]
    Full,
    /// Trust the recorded digest and validate metadata, compatibility, and length.
    ///
    /// This avoids reading an entire warm payload. Use it only when the cache
    /// root is immutable to untrusted or out-of-band writers. The digest remains
    /// available through [`CacheEntryInfo::payload_digest`] for downstream
    /// provenance and validation.
    TrustedMetadata,
}

/// The deterministic reason a cache lookup could not be reused.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheMissReason {
    /// No published directory exists for the key.
    NotFound,
    /// The published directory lacks a complete metadata/payload pair.
    IncompleteEntry,
    /// The metadata envelope exceeds its defensive size limit.
    MetadataTooLarge {
        /// Actual metadata bytes.
        actual: u64,
        /// Maximum accepted metadata bytes.
        maximum: u64,
    },
    /// The metadata envelope is not valid supported JSON.
    MetadataMalformed,
    /// The metadata envelope version is unsupported.
    EnvelopeVersion {
        /// Version read from the entry.
        found: u32,
        /// Version supported by this crate.
        supported: u32,
    },
    /// The envelope names a different physical namespace.
    NamespaceMismatch {
        /// Namespace read from the entry.
        found: CacheNamespace,
    },
    /// The envelope names a different cache key.
    KeyMismatch {
        /// Key read from the entry.
        found: CacheKey,
    },
    /// Transform, format, or backend ABI compatibility changed.
    CompatibilityMismatch {
        /// Compatibility contract read from the entry.
        found: CacheCompatibility,
    },
    /// The payload length differs from the envelope.
    LengthMismatch {
        /// Length declared by the envelope.
        expected: u64,
        /// Length observed on disk.
        actual: u64,
    },
    /// The payload digest differs from the envelope.
    DigestMismatch {
        /// Digest declared by the envelope.
        expected: ContentDigest,
        /// Digest computed from the payload.
        actual: ContentDigest,
    },
}

impl Display for CacheMissReason {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => formatter.write_str("entry not found"),
            Self::IncompleteEntry => formatter.write_str("entry is incomplete"),
            Self::MetadataTooLarge { actual, maximum } => {
                write!(
                    formatter,
                    "metadata has {actual} bytes, maximum is {maximum}"
                )
            }
            Self::MetadataMalformed => formatter.write_str("metadata is malformed"),
            Self::EnvelopeVersion { found, supported } => {
                write!(
                    formatter,
                    "envelope version {found} is not supported ({supported})"
                )
            }
            Self::NamespaceMismatch { found } => {
                write!(formatter, "entry belongs to the {found} namespace")
            }
            Self::KeyMismatch { found } => write!(formatter, "entry declares key {found}"),
            Self::CompatibilityMismatch { .. } => {
                formatter.write_str("entry compatibility does not match")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(formatter, "payload has {actual} bytes, expected {expected}")
            }
            Self::DigestMismatch { expected, actual } => {
                write!(
                    formatter,
                    "payload digest {actual} does not match {expected}"
                )
            }
        }
    }
}

/// Metadata for one structurally valid published cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntryInfo {
    namespace: CacheNamespace,
    key: CacheKey,
    compatibility: CacheCompatibility,
    payload_len: u64,
    payload_digest: ContentDigest,
    created_unix_millis: u64,
    payload_path: PathBuf,
}

impl CacheEntryInfo {
    /// Returns the physical namespace.
    #[must_use]
    pub const fn namespace(&self) -> CacheNamespace {
        self.namespace
    }

    /// Returns the content-addressed key.
    #[must_use]
    pub const fn key(&self) -> CacheKey {
        self.key
    }

    /// Returns the validated compatibility contract.
    #[must_use]
    pub const fn compatibility(&self) -> &CacheCompatibility {
        &self.compatibility
    }

    /// Returns the declared payload length.
    #[must_use]
    pub const fn payload_len(&self) -> u64 {
        self.payload_len
    }

    /// Returns the declared payload digest.
    #[must_use]
    pub const fn payload_digest(&self) -> ContentDigest {
        self.payload_digest
    }

    /// Returns the publication timestamp as milliseconds after Unix epoch.
    #[must_use]
    pub const fn created_unix_millis(&self) -> u64 {
        self.created_unix_millis
    }

    /// Returns the published payload path.
    #[must_use]
    pub fn payload_path(&self) -> &Path {
        &self.payload_path
    }
}

/// A validated entry retaining an open immutable payload handle.
#[derive(Debug)]
pub struct CacheEntry {
    info: CacheEntryInfo,
    payload: File,
}

impl CacheEntry {
    /// Returns validated entry metadata.
    #[must_use]
    pub const fn info(&self) -> &CacheEntryInfo {
        &self.info
    }

    /// Returns the open payload file positioned at byte zero.
    #[must_use]
    pub const fn payload(&self) -> &File {
        &self.payload
    }

    /// Clones the open payload handle.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the operating system cannot duplicate the
    /// handle.
    pub fn try_clone_payload(&self) -> Result<File> {
        self.payload
            .try_clone()
            .map_err(|error| Error::io("failed to clone cache payload handle", error))
    }

    /// Consumes the entry and returns its open payload file.
    #[must_use]
    pub fn into_payload(self) -> File {
        self.payload
    }
}

/// The result of a validating cache lookup.
#[derive(Debug)]
#[non_exhaustive]
pub enum CacheLookup {
    /// A complete entry accepted by the requested validation policy.
    Hit(CacheEntry),
    /// No reusable entry exists for the deterministic reason.
    Miss(CacheMissReason),
}

impl CacheLookup {
    /// Returns the validated entry, or its miss reason.
    ///
    /// # Errors
    ///
    /// Returns the deterministic miss reason when no entry was reusable.
    pub fn into_result(self) -> std::result::Result<CacheEntry, CacheMissReason> {
        match self {
            Self::Hit(entry) => Ok(entry),
            Self::Miss(reason) => Err(reason),
        }
    }
}

/// The result of publishing a cache payload.
#[derive(Debug)]
#[non_exhaustive]
pub enum CachePublication {
    /// This caller atomically published the returned entry.
    Published(CacheEntry),
    /// A duplicate writer had already published an equivalent valid entry.
    Reused(CacheEntry),
}

impl CachePublication {
    /// Returns the validated published or reused entry.
    #[must_use]
    pub const fn entry(&self) -> &CacheEntry {
        match self {
            Self::Published(entry) | Self::Reused(entry) => entry,
        }
    }

    /// Consumes the outcome and returns the validated entry.
    #[must_use]
    pub fn into_entry(self) -> CacheEntry {
        match self {
            Self::Published(entry) | Self::Reused(entry) => entry,
        }
    }
}

/// A summary of published cache contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheInspection {
    entries: Vec<CacheEntryInfo>,
    invalid_entries: usize,
    payload_bytes: u64,
    total_bytes: u64,
}

impl CacheInspection {
    /// Returns metadata for structurally valid published entries.
    #[must_use]
    pub fn entries(&self) -> &[CacheEntryInfo] {
        &self.entries
    }

    /// Returns the number of malformed or inconsistent published directories.
    #[must_use]
    pub const fn invalid_entries(&self) -> usize {
        self.invalid_entries
    }

    /// Returns actual payload bytes in structurally valid entries.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Returns all on-disk bytes below the versioned cache directory.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

/// Why a published entry was evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EvictionReason {
    /// A caller explicitly invalidated the entry.
    Explicit,
    /// The configured cache capacity required reclamation.
    Capacity,
    /// Integrity validation rejected the entry.
    Corrupt,
    /// A format, transform, or backend ABI changed.
    Incompatible,
    /// The entry no longer has a live source identity.
    SourceChanged,
}

/// One completed cache eviction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionRecord {
    namespace: CacheNamespace,
    key: CacheKey,
    reason: EvictionReason,
    bytes: u64,
}

impl EvictionRecord {
    /// Returns the evicted namespace.
    #[must_use]
    pub const fn namespace(&self) -> CacheNamespace {
        self.namespace
    }

    /// Returns the evicted key.
    #[must_use]
    pub const fn key(&self) -> CacheKey {
        self.key
    }

    /// Returns the caller-supplied eviction reason.
    #[must_use]
    pub const fn reason(&self) -> EvictionReason {
        self.reason
    }

    /// Returns bytes found below the evicted entry directory.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Capacity-eviction results suitable for policy telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionReport {
    before_bytes: u64,
    after_bytes: u64,
    records: Vec<EvictionRecord>,
}

impl EvictionReport {
    /// Returns cache bytes before eviction.
    #[must_use]
    pub const fn before_bytes(&self) -> u64 {
        self.before_bytes
    }

    /// Returns cache bytes after eviction and cleanup.
    #[must_use]
    pub const fn after_bytes(&self) -> u64 {
        self.after_bytes
    }

    /// Returns completed eviction records in policy order.
    #[must_use]
    pub fn records(&self) -> &[EvictionRecord] {
        &self.records
    }
}

/// Results of interrupted staging and abandoned lease recovery.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    removed_directories: usize,
    restored_entries: usize,
    reclaimed_bytes: u64,
}

impl RecoveryReport {
    /// Returns the number of stale internal directories removed.
    #[must_use]
    pub const fn removed_directories(&self) -> usize {
        self.removed_directories
    }

    /// Returns the number of interrupted replacements restored as published entries.
    #[must_use]
    pub const fn restored_entries(&self) -> usize {
        self.restored_entries
    }

    /// Returns bytes found below removed stale directories.
    #[must_use]
    pub const fn reclaimed_bytes(&self) -> u64 {
        self.reclaimed_bytes
    }
}

/// A cloneable content-addressed cache service.
#[derive(Debug, Clone)]
pub struct Cache {
    inner: Arc<CacheInner>,
}

#[derive(Debug)]
struct CacheInner {
    root: PathBuf,
    options: CacheOptions,
}

impl Cache {
    /// Opens or creates a cache using default lease and recovery timing.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured scan limit is invalid or exceeded,
    /// or when namespace directories cannot be created or recovered.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(root, CacheOptions::default())
    }

    /// Opens or creates a cache with explicit lease and recovery timing.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when versioned namespace directories cannot be
    /// created or stale internal directories cannot be recovered.
    pub fn open_with_options(root: impl AsRef<Path>, options: CacheOptions) -> Result<Self> {
        Self::open_with_options_and_cancellation(root, options, &CancellationToken::new())
    }

    /// Opens or creates a cache with bounded, cancellable startup recovery.
    ///
    /// # Errors
    ///
    /// Returns an error when the scan limit is zero, cancellation is requested,
    /// namespace directories cannot be created, or stale internal directories
    /// cannot be recovered.
    pub fn open_with_options_and_cancellation(
        root: impl AsRef<Path>,
        options: CacheOptions,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        cancellation.check()?;
        if options.maximum_scan_entries == 0 {
            return Err(Error::limit(
                "cache maximum scan entries must be greater than zero",
            ));
        }
        let cache = Self {
            inner: Arc::new(CacheInner {
                root: root.as_ref().to_path_buf(),
                options,
            }),
        };
        for namespace in [CacheNamespace::Plan, CacheNamespace::Prepared] {
            cancellation.check()?;
            fs::create_dir_all(cache.namespace_path(namespace)).map_err(|error| {
                Error::io("failed to create a cache namespace directory", error)
            })?;
        }
        let _recovery = cache.recover_stale_with_cancellation(cancellation)?;
        Ok(cache)
    }

    /// Returns the caller-supplied cache root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    /// Returns the immutable cache options.
    #[must_use]
    pub fn options(&self) -> &CacheOptions {
        &self.inner.options
    }

    /// Returns the published directory path for a namespace and key.
    ///
    /// This is intended for diagnostics. Call [`lookup`](Self::lookup) before
    /// opening payload data.
    #[must_use]
    pub fn entry_path(&self, namespace: CacheNamespace, key: CacheKey) -> PathBuf {
        let key_text = key.to_string();
        self.namespace_path(namespace)
            .join(&key_text[..2])
            .join(key_text)
    }

    /// Validates and opens a cache entry.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when an existing entry cannot be inspected or read.
    /// Structural, integrity, and compatibility failures are returned as
    /// [`CacheLookup::Miss`].
    pub fn lookup(
        &self,
        namespace: CacheNamespace,
        key: CacheKey,
        compatibility: &CacheCompatibility,
    ) -> Result<CacheLookup> {
        self.lookup_with_cancellation(namespace, key, compatibility, &CancellationToken::new())
    }

    /// Opens a cache entry with an explicit integrity-validation policy.
    ///
    /// [`CacheValidation::TrustedMetadata`] avoids a full warm-payload read,
    /// but is safe only for a cache root protected from out-of-band mutation.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when an existing entry cannot be inspected or read.
    /// Structural, integrity, and compatibility failures are returned as
    /// [`CacheLookup::Miss`].
    pub fn lookup_with_validation(
        &self,
        namespace: CacheNamespace,
        key: CacheKey,
        compatibility: &CacheCompatibility,
        validation: CacheValidation,
    ) -> Result<CacheLookup> {
        self.lookup_with_validation_and_cancellation(
            namespace,
            key,
            compatibility,
            validation,
            &CancellationToken::new(),
        )
    }

    /// Validates and opens a cache entry with cooperative cancellation.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when an existing entry cannot be inspected or read,
    /// a resource-limit error when replacement discovery exceeds its configured
    /// scan bound, or a cancellation error during discovery or hashing.
    pub fn lookup_with_cancellation(
        &self,
        namespace: CacheNamespace,
        key: CacheKey,
        compatibility: &CacheCompatibility,
        cancellation: &CancellationToken,
    ) -> Result<CacheLookup> {
        self.lookup_with_validation_and_cancellation(
            namespace,
            key,
            compatibility,
            CacheValidation::Full,
            cancellation,
        )
    }

    /// Opens a cache entry with explicit validation and cooperative cancellation.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when an existing entry cannot be inspected or read,
    /// or a cancellation error while hashing a large cached payload.
    pub fn lookup_with_validation_and_cancellation(
        &self,
        namespace: CacheNamespace,
        key: CacheKey,
        compatibility: &CacheCompatibility,
        validation: CacheValidation,
        cancellation: &CancellationToken,
    ) -> Result<CacheLookup> {
        cancellation.check()?;
        let entry_path = self.entry_path(namespace, key);
        let first = lookup_entry_path(
            &entry_path,
            namespace,
            key,
            compatibility,
            validation,
            cancellation,
        )?;
        if matches!(first, CacheLookup::Hit(_)) {
            return Ok(first);
        }

        // Replacement is two atomic renames: old -> previous, then staged ->
        // final. Readers landing between them follow the immutable previous
        // generation, so publication has no logical visibility gap.
        let mut replacement_scan =
            ScanBudget::new(self.inner.options.maximum_scan_entries, cancellation)?;
        let previous = replacement_paths(&entry_path, &mut replacement_scan)?;

        // Prefer a new final generation that appeared while its parent was
        // enumerated.
        let retry = lookup_entry_path(
            &entry_path,
            namespace,
            key,
            compatibility,
            validation,
            cancellation,
        )?;
        if matches!(retry, CacheLookup::Hit(_)) {
            return Ok(retry);
        }

        for previous_path in previous {
            let lookup = match lookup_entry_path(
                &previous_path,
                namespace,
                key,
                compatibility,
                validation,
                cancellation,
            ) {
                Ok(lookup) => lookup,
                Err(error) if replacement_lookup_raced(&error, &previous_path) => continue,
                Err(error) => return Err(error),
            };
            match lookup {
                CacheLookup::Hit(entry) => return Ok(CacheLookup::Hit(entry)),
                CacheLookup::Miss(_) => {}
            }
        }

        // A publisher may have removed its previous generation after the first
        // retry. In that case the final generation is now authoritative.
        lookup_entry_path(
            &entry_path,
            namespace,
            key,
            compatibility,
            validation,
            cancellation,
        )
    }

    /// Atomically publishes an in-memory payload.
    ///
    /// # Errors
    ///
    /// Returns an error when lease acquisition, staging, publication, or final
    /// validation fails.
    pub fn publish_bytes(
        &self,
        namespace: CacheNamespace,
        key: CacheKey,
        compatibility: &CacheCompatibility,
        payload: &[u8],
    ) -> Result<CachePublication> {
        self.publish_reader(
            namespace,
            key,
            compatibility,
            Cursor::new(payload),
            &CancellationToken::new(),
        )
    }

    /// Streams and atomically publishes a payload with cooperative cancellation.
    ///
    /// The staging directory is a sibling of the final entry, which keeps the
    /// publication rename on one filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error when reading, lease acquisition, staging, publication,
    /// final validation, or cancellation fails.
    pub fn publish_reader(
        &self,
        namespace: CacheNamespace,
        key: CacheKey,
        compatibility: &CacheCompatibility,
        mut reader: impl Read,
        cancellation: &CancellationToken,
    ) -> Result<CachePublication> {
        if let CacheLookup::Hit(entry) =
            self.lookup_with_cancellation(namespace, key, compatibility, cancellation)?
        {
            return Ok(CachePublication::Reused(entry));
        }

        let final_path = self.entry_path(namespace, key);
        let parent = final_path
            .parent()
            .ok_or_else(|| Error::cache("cache entry path has no parent directory"))?;
        fs::create_dir_all(parent)
            .map_err(|error| Error::io("failed to create a cache key directory", error))?;
        let mut lease = WriterLease::acquire(parent, key, &self.inner.options, cancellation)?;

        if let CacheLookup::Hit(entry) =
            self.lookup_with_cancellation(namespace, key, compatibility, cancellation)?
        {
            return Ok(CachePublication::Reused(entry));
        }

        let mut staging = StagingEntry::create(parent, key)?;
        let payload_path = staging.path.join(PAYLOAD_FILE);
        let (payload_len, payload_digest) =
            write_payload(&payload_path, &mut reader, &mut lease, cancellation)?;
        let envelope = CacheEnvelope {
            envelope_version: CACHE_ENVELOPE_VERSION,
            namespace,
            key,
            compatibility: compatibility.clone(),
            payload_len,
            payload_digest,
            created_unix_millis: unix_millis(),
        };
        write_envelope(&staging.path, &envelope)?;
        lease.refresh()?;
        cancellation.check()?;
        staging.publish(&final_path)?;

        // Publication already streamed the payload through SHA-256 and synced
        // both files before the rename. Reopen and length-check the published
        // handle without immediately reading the entire payload a second time.
        let entry = open_published_entry(&final_path, envelope)?;
        Ok(CachePublication::Published(entry))
    }

    /// Inspects published entries and accounts for all versioned cache bytes.
    ///
    /// This structural inventory does not rehash payloads; use
    /// [`lookup`](Self::lookup) for integrity validation.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured scan bound is exceeded or cache
    /// directories cannot be enumerated.
    pub fn inspect(&self) -> Result<CacheInspection> {
        self.inspect_with_cancellation(&CancellationToken::new())
    }

    /// Inspects cache storage with cooperative cancellation and bounded scans.
    ///
    /// # Errors
    ///
    /// Returns an error when cancellation is requested, the configured scan
    /// bound is exceeded, or cache directories cannot be enumerated.
    pub fn inspect_with_cancellation(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<CacheInspection> {
        let mut scan = ScanBudget::new(self.inner.options.maximum_scan_entries, cancellation)?;
        let candidates = self.scan_candidates(&mut scan)?;
        let mut entries = Vec::new();
        entries
            .try_reserve(candidates.len())
            .map_err(|_error| Error::limit("cache inspection allocation failed"))?;
        let mut invalid_entries = 0_usize;
        let mut payload_bytes = 0_u64;
        for candidate in candidates {
            scan.check()?;
            match candidate.envelope {
                Some(envelope)
                    if envelope.envelope_version == CACHE_ENVELOPE_VERSION
                        && envelope.namespace == candidate.namespace
                        && envelope.key == candidate.key =>
                {
                    let payload_path = candidate.path.join(PAYLOAD_FILE);
                    let actual_len = match fs::metadata(&payload_path) {
                        Ok(metadata) if metadata.is_file() => metadata.len(),
                        Ok(_) => {
                            invalid_entries = invalid_entries.saturating_add(1);
                            continue;
                        }
                        Err(error) if error.kind() == io::ErrorKind::NotFound => {
                            invalid_entries = invalid_entries.saturating_add(1);
                            continue;
                        }
                        Err(error) => {
                            return Err(Error::io(
                                "failed to inspect an inventoried cache payload",
                                error,
                            ));
                        }
                    };
                    if actual_len != envelope.payload_len {
                        invalid_entries = invalid_entries.saturating_add(1);
                        continue;
                    }
                    payload_bytes = payload_bytes.saturating_add(actual_len);
                    entries.push(info_from_envelope(envelope, payload_path));
                }
                _ => invalid_entries = invalid_entries.saturating_add(1),
            }
        }
        entries.sort_unstable_by_key(|entry| (entry.namespace, entry.key));
        let total_bytes = directory_size(&self.version_path(), &mut scan, 0)?;
        Ok(CacheInspection {
            entries,
            invalid_entries,
            payload_bytes,
            total_bytes,
        })
    }

    /// Atomically removes one published entry under a cooperative writer lease.
    ///
    /// Open readers retain their file handles; new readers cannot discover a
    /// partially removed directory.
    ///
    /// # Errors
    ///
    /// Returns an error when cancellation is requested, the configured scan
    /// bound is exceeded, or lease acquisition, rename, or removal fails.
    pub fn evict(
        &self,
        namespace: CacheNamespace,
        key: CacheKey,
        reason: EvictionReason,
        cancellation: &CancellationToken,
    ) -> Result<Option<EvictionRecord>> {
        let final_path = self.entry_path(namespace, key);
        let parent = final_path
            .parent()
            .ok_or_else(|| Error::cache("cache entry path has no parent directory"))?;
        fs::create_dir_all(parent)
            .map_err(|error| Error::io("failed to create a cache key directory", error))?;
        let _lease = WriterLease::acquire(parent, key, &self.inner.options, cancellation)?;
        cancellation.check()?;
        let mut scan = ScanBudget::new(self.inner.options.maximum_scan_entries, cancellation)?;
        let bytes = match fs::symlink_metadata(&final_path) {
            Ok(metadata) if metadata.is_dir() => directory_size(&final_path, &mut scan, 0)?,
            Ok(_) => 0,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(Error::io("failed to inspect an evicted entry", error));
            }
        };
        let trash_path = parent.join(format!(".{key}.evicted-{}", unique_token()));
        match fs::rename(&final_path, &trash_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(Error::io(
                    "failed to atomically hide an evicted cache entry",
                    error,
                ));
            }
        }
        fs::remove_dir_all(&trash_path)
            .map_err(|error| Error::io("failed to remove an evicted cache entry", error))?;
        Ok(Some(EvictionRecord {
            namespace,
            key,
            reason,
            bytes,
        }))
    }

    /// Evicts oldest published directories toward a total-byte capacity.
    ///
    /// Invalid entries participate in capacity reclamation. Returned records
    /// preserve the supplied policy reason. Active staging or lease bytes are
    /// never selected, so [`EvictionReport::after_bytes`] can remain above the
    /// target while a writer is active.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured scan bound is exceeded, or when
    /// inspection, lease acquisition, rename, removal, or cancellation fails.
    pub fn evict_to(
        &self,
        maximum_bytes: u64,
        reason: EvictionReason,
        cancellation: &CancellationToken,
    ) -> Result<EvictionReport> {
        let mut before_scan =
            ScanBudget::new(self.inner.options.maximum_scan_entries, cancellation)?;
        let before_bytes = directory_size(&self.version_path(), &mut before_scan, 0)?;
        let mut candidate_scan =
            ScanBudget::new(self.inner.options.maximum_scan_entries, cancellation)?;
        let mut candidates = self.scan_candidates(&mut candidate_scan)?;
        candidates.sort_unstable_by_key(|candidate| {
            (candidate.modified, candidate.namespace, candidate.key)
        });
        let mut estimated = before_bytes;
        let mut records = Vec::new();
        for candidate in candidates {
            if estimated <= maximum_bytes {
                break;
            }
            cancellation.check()?;
            if let Some(record) =
                self.evict(candidate.namespace, candidate.key, reason, cancellation)?
            {
                estimated = estimated.saturating_sub(record.bytes);
                records
                    .try_reserve(1)
                    .map_err(|_error| Error::limit("cache eviction report allocation failed"))?;
                records.push(record);
            }
        }
        let mut after_scan =
            ScanBudget::new(self.inner.options.maximum_scan_entries, cancellation)?;
        let after_bytes = directory_size(&self.version_path(), &mut after_scan, 0)?;
        Ok(EvictionReport {
            before_bytes,
            after_bytes,
            records,
        })
    }

    /// Recovers interrupted replacements and removes other stale internal data.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured scan bound is exceeded, or when
    /// internal directories cannot be inspected, claimed, or removed.
    pub fn recover_stale(&self) -> Result<RecoveryReport> {
        self.recover_stale_with_cancellation(&CancellationToken::new())
    }

    /// Recovers interrupted cache writes with bounded filesystem work.
    ///
    /// # Errors
    ///
    /// Returns an error when cancellation is requested, the configured scan
    /// bound is exceeded, or internal directories cannot be recovered.
    pub fn recover_stale_with_cancellation(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<RecoveryReport> {
        let mut scan = ScanBudget::new(self.inner.options.maximum_scan_entries, cancellation)?;
        let mut report = RecoveryReport::default();
        for namespace in [CacheNamespace::Plan, CacheNamespace::Prepared] {
            scan.check()?;
            let namespace_path = self.namespace_path(namespace);
            for shard in read_directories(&namespace_path, &mut scan)? {
                recover_shard(
                    &shard,
                    self.inner.options.stale_after,
                    &mut report,
                    &mut scan,
                )?;
            }
        }
        Ok(report)
    }

    fn version_path(&self) -> PathBuf {
        self.inner.root.join(CACHE_DIRECTORY_VERSION)
    }

    fn namespace_path(&self, namespace: CacheNamespace) -> PathBuf {
        self.version_path().join(namespace.directory_name())
    }

    fn scan_candidates(&self, scan: &mut ScanBudget<'_>) -> Result<Vec<CacheCandidate>> {
        let mut candidates = Vec::new();
        for namespace in [CacheNamespace::Plan, CacheNamespace::Prepared] {
            scan.check()?;
            for shard in read_directories(&self.namespace_path(namespace), scan)? {
                for entry in read_directories(&shard, scan)? {
                    let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
                        continue;
                    };
                    let Ok(key) = name.parse::<CacheKey>() else {
                        continue;
                    };
                    let metadata = fs::metadata(&entry).map_err(|error| {
                        Error::io("failed to inspect a published cache entry", error)
                    })?;
                    let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
                    let envelope = match read_envelope(&entry)? {
                        MetadataRead::Envelope(envelope) => Some(envelope),
                        MetadataRead::Miss(_) => None,
                    };
                    candidates.try_reserve(1).map_err(|_error| {
                        Error::limit("cache candidate inventory allocation failed")
                    })?;
                    candidates.push(CacheCandidate {
                        namespace,
                        key,
                        path: entry,
                        modified,
                        envelope,
                    });
                }
            }
        }
        Ok(candidates)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEnvelope {
    envelope_version: u32,
    namespace: CacheNamespace,
    key: CacheKey,
    compatibility: CacheCompatibility,
    payload_len: u64,
    payload_digest: ContentDigest,
    created_unix_millis: u64,
}

#[derive(Debug)]
enum MetadataRead {
    Envelope(CacheEnvelope),
    Miss(CacheMissReason),
}

#[derive(Debug)]
struct CacheCandidate {
    namespace: CacheNamespace,
    key: CacheKey,
    path: PathBuf,
    modified: SystemTime,
    envelope: Option<CacheEnvelope>,
}

fn lookup_entry_path(
    entry_path: &Path,
    namespace: CacheNamespace,
    key: CacheKey,
    compatibility: &CacheCompatibility,
    validation: CacheValidation,
    cancellation: &CancellationToken,
) -> Result<CacheLookup> {
    cancellation.check()?;
    match fs::metadata(entry_path) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Ok(CacheLookup::Miss(CacheMissReason::IncompleteEntry)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(CacheLookup::Miss(CacheMissReason::NotFound));
        }
        Err(error) => {
            return Err(Error::io(
                "failed to inspect a cache entry directory",
                error,
            ));
        }
    }

    let envelope = match read_envelope(entry_path)? {
        MetadataRead::Envelope(envelope) => envelope,
        MetadataRead::Miss(reason) => return Ok(CacheLookup::Miss(reason)),
    };
    if envelope.envelope_version != CACHE_ENVELOPE_VERSION {
        return Ok(CacheLookup::Miss(CacheMissReason::EnvelopeVersion {
            found: envelope.envelope_version,
            supported: CACHE_ENVELOPE_VERSION,
        }));
    }
    if envelope.namespace != namespace {
        return Ok(CacheLookup::Miss(CacheMissReason::NamespaceMismatch {
            found: envelope.namespace,
        }));
    }
    if envelope.key != key {
        return Ok(CacheLookup::Miss(CacheMissReason::KeyMismatch {
            found: envelope.key,
        }));
    }
    if envelope.compatibility != *compatibility {
        return Ok(CacheLookup::Miss(CacheMissReason::CompatibilityMismatch {
            found: envelope.compatibility,
        }));
    }

    let payload_path = entry_path.join(PAYLOAD_FILE);
    let mut payload = match File::open(&payload_path) {
        Ok(payload) => payload,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(CacheLookup::Miss(CacheMissReason::IncompleteEntry));
        }
        Err(error) => return Err(Error::io("failed to open a cache payload", error)),
    };
    let actual_len = payload
        .metadata()
        .map_err(|error| Error::io("failed to inspect a cache payload", error))?
        .len();
    if actual_len != envelope.payload_len {
        return Ok(CacheLookup::Miss(CacheMissReason::LengthMismatch {
            expected: envelope.payload_len,
            actual: actual_len,
        }));
    }
    if validation == CacheValidation::Full {
        let actual_digest = digest_reader(&mut payload, cancellation)?;
        if actual_digest != envelope.payload_digest {
            return Ok(CacheLookup::Miss(CacheMissReason::DigestMismatch {
                expected: envelope.payload_digest,
                actual: actual_digest,
            }));
        }
    }
    payload
        .seek(SeekFrom::Start(0))
        .map_err(|error| Error::io("failed to rewind a cache payload", error))?;
    let info = info_from_envelope(envelope, payload_path);
    Ok(CacheLookup::Hit(CacheEntry { info, payload }))
}

fn open_published_entry(entry_path: &Path, envelope: CacheEnvelope) -> Result<CacheEntry> {
    let payload_path = entry_path.join(PAYLOAD_FILE);
    let payload = File::open(&payload_path)
        .map_err(|error| Error::io("failed to open the published cache payload", error))?;
    let actual_len = payload
        .metadata()
        .map_err(|error| Error::io("failed to inspect the published cache payload", error))?
        .len();
    if actual_len != envelope.payload_len {
        return Err(Error::cache(
            "published cache payload length changed after staging",
        ));
    }
    let info = info_from_envelope(envelope, payload_path);
    Ok(CacheEntry { info, payload })
}

fn replacement_paths(entry_path: &Path, scan: &mut ScanBudget<'_>) -> Result<Vec<PathBuf>> {
    scan.check()?;
    let parent = entry_path
        .parent()
        .ok_or_else(|| Error::cache("cache entry path has no parent directory"))?;
    let key_name = entry_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::cache("cache entry path has no portable file name"))?;
    let prefix = format!(".{key_name}{REPLACEMENT_MARKER}");
    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(Error::io(
                "failed to enumerate cache replacement generations",
                error,
            ));
        }
    };
    let mut paths = Vec::new();
    for entry in entries {
        scan.visit_entry()?;
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            #[cfg(windows)]
            Err(error) if windows_filesystem_operation_is_transient(&error) => continue,
            Err(error) => {
                return Err(Error::io(
                    "failed to enumerate cache replacement generations",
                    error,
                ));
            }
        };
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !name.starts_with(&prefix) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) if metadata.is_dir() => metadata,
            Ok(_) => continue,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            #[cfg(windows)]
            Err(error) if windows_filesystem_operation_is_transient(&error) => continue,
            Err(error) => {
                return Err(Error::io(
                    "failed to inspect a cache replacement generation",
                    error,
                ));
            }
        };
        paths
            .try_reserve(1)
            .map_err(|_error| Error::limit("cache replacement inventory allocation failed"))?;
        paths.push((metadata.modified().unwrap_or(UNIX_EPOCH), entry.path()));
    }
    paths.sort_unstable_by(|left, right| right.cmp(left));
    Ok(paths.into_iter().map(|(_, path)| path).collect())
}

fn replacement_lookup_raced(error: &Error, path: &Path) -> bool {
    if !path.exists() {
        return true;
    }
    #[cfg(windows)]
    {
        StdError::source(error)
            .and_then(|source| source.downcast_ref::<io::Error>())
            .is_some_and(windows_filesystem_operation_is_transient)
    }
    #[cfg(not(windows))]
    {
        let _error = error;
        false
    }
}

fn update_hash_part(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u128).to_le_bytes());
    hasher.update(bytes);
}

fn read_envelope(entry_path: &Path) -> Result<MetadataRead> {
    let metadata_path = entry_path.join(METADATA_FILE);
    let file = match File::open(&metadata_path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(MetadataRead::Miss(CacheMissReason::IncompleteEntry));
        }
        Err(error) => return Err(Error::io("failed to open cache metadata", error)),
    };
    let metadata_len = file
        .metadata()
        .map_err(|error| Error::io("failed to inspect cache metadata", error))?
        .len();
    if metadata_len > MAX_METADATA_BYTES {
        return Ok(MetadataRead::Miss(CacheMissReason::MetadataTooLarge {
            actual: metadata_len,
            maximum: MAX_METADATA_BYTES,
        }));
    }
    let capacity = usize::try_from(metadata_len)
        .map_err(|_conversion| Error::cache("cache metadata length does not fit in memory"))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(MAX_METADATA_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| Error::io("failed to read cache metadata", error))?;
    let actual_len = u64::try_from(bytes.len())
        .map_err(|_conversion| Error::cache("cache metadata read length does not fit in u64"))?;
    if actual_len > MAX_METADATA_BYTES {
        return Ok(MetadataRead::Miss(CacheMissReason::MetadataTooLarge {
            actual: actual_len,
            maximum: MAX_METADATA_BYTES,
        }));
    }
    match serde_json::from_slice(&bytes) {
        Ok(envelope) => Ok(MetadataRead::Envelope(envelope)),
        Err(_) => Ok(MetadataRead::Miss(CacheMissReason::MetadataMalformed)),
    }
}

fn write_envelope(staging_path: &Path, envelope: &CacheEnvelope) -> Result<()> {
    let bytes = serde_json::to_vec(envelope)
        .map_err(|_serialization| Error::cache("failed to serialize cache metadata"))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_METADATA_BYTES {
        return Err(Error::cache(
            "serialized cache metadata exceeds its defensive size limit",
        ));
    }
    let metadata_path = staging_path.join(METADATA_FILE);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(metadata_path)
        .map_err(|error| Error::io("failed to create staged cache metadata", error))?;
    file.write_all(&bytes)
        .map_err(|error| Error::io("failed to write staged cache metadata", error))?;
    file.sync_all()
        .map_err(|error| Error::io("failed to synchronize staged cache metadata", error))
}

fn info_from_envelope(envelope: CacheEnvelope, payload_path: PathBuf) -> CacheEntryInfo {
    CacheEntryInfo {
        namespace: envelope.namespace,
        key: envelope.key,
        compatibility: envelope.compatibility,
        payload_len: envelope.payload_len,
        payload_digest: envelope.payload_digest,
        created_unix_millis: envelope.created_unix_millis,
        payload_path,
    }
}

fn digest_reader(
    reader: &mut impl Read,
    cancellation: &CancellationToken,
) -> Result<ContentDigest> {
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    loop {
        cancellation.check()?;
        let read = reader
            .read(&mut buffer)
            .map_err(|error| Error::io("failed to read bytes for cache validation", error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(ContentDigest::from_bytes(hasher.finalize().into()))
}

fn write_payload(
    path: &Path,
    reader: &mut impl Read,
    lease: &mut WriterLease,
    cancellation: &CancellationToken,
) -> Result<(u64, ContentDigest)> {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| Error::io("failed to create a staged cache payload", error))?;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    loop {
        cancellation.check()?;
        lease.refresh()?;
        let read = reader
            .read(&mut buffer)
            .map_err(|error| Error::io("failed to read a cache publication payload", error))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|error| Error::io("failed to write a staged cache payload", error))?;
        hasher.update(&buffer[..read]);
        let read = u64::try_from(read)
            .map_err(|_conversion| Error::cache("cache payload length does not fit in u64"))?;
        total = total
            .checked_add(read)
            .ok_or_else(|| Error::cache("cache payload length overflowed u64"))?;
    }
    output
        .sync_all()
        .map_err(|error| Error::io("failed to synchronize a staged cache payload", error))?;
    Ok((total, ContentDigest::from_bytes(hasher.finalize().into())))
}

#[derive(Debug)]
struct StagingEntry {
    path: PathBuf,
    active: bool,
}

impl StagingEntry {
    fn create(parent: &Path, key: CacheKey) -> Result<Self> {
        for _ in 0..100 {
            let path = parent.join(format!(".{key}.tmp-{}", unique_token()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path, active: true }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(Error::io(
                        "failed to create a same-filesystem staging directory",
                        error,
                    ));
                }
            }
        }
        Err(Error::cache(
            "failed to allocate a unique cache staging directory",
        ))
    }

    fn publish(&mut self, final_path: &Path) -> Result<()> {
        let parent = final_path
            .parent()
            .ok_or_else(|| Error::cache("cache entry path has no parent directory"))?;
        let key_name = final_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::cache("cache entry path has no portable file name"))?;
        let previous = parent.join(format!(".{key_name}{REPLACEMENT_MARKER}{}", unique_token()));
        let moved_previous = match rename_directory(final_path, &previous) {
            Ok(()) => true,
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(error) => {
                return Err(Error::io(
                    "failed to move an existing cache entry before publication",
                    error,
                ));
            }
        };
        if let Err(error) = rename_directory(&self.path, final_path) {
            if moved_previous {
                let _restored = rename_directory(&previous, final_path);
            }
            return Err(Error::io(
                "failed to atomically publish a cache entry",
                error,
            ));
        }
        self.active = false;
        if moved_previous {
            let _cleanup = fs::remove_dir_all(previous);
        }
        Ok(())
    }
}

impl Drop for StagingEntry {
    fn drop(&mut self) {
        if self.active {
            let _cleanup = fs::remove_dir_all(&self.path);
        }
    }
}

#[derive(Debug)]
struct WriterLease {
    path: PathBuf,
    owner: String,
    active: bool,
}

impl WriterLease {
    fn acquire(
        parent: &Path,
        key: CacheKey,
        options: &CacheOptions,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        let path = parent.join(format!(".{key}.lease"));
        let started = Instant::now();
        loop {
            cancellation.check()?;
            match fs::create_dir(&path) {
                Ok(()) => {
                    let owner = unique_token();
                    if let Err(error) = fs::write(path.join(OWNER_FILE), owner.as_bytes()) {
                        let _cleanup = reap_directory(&path);
                        return Err(Error::io("failed to initialize a cache lease", error));
                    }
                    if let Err(error) =
                        fs::write(path.join(HEARTBEAT_FILE), unix_millis().to_string())
                    {
                        let _cleanup = reap_directory(&path);
                        return Err(Error::io("failed to initialize a cache lease", error));
                    }
                    return Ok(Self {
                        path,
                        owner,
                        active: true,
                    });
                }
                Err(error) if lease_is_contended(&error) => {
                    if lease_is_stale(&path, options.stale_after, cancellation)? {
                        reap_directory(&path)?;
                        continue;
                    }
                    if started.elapsed() >= options.lease_wait {
                        return Err(Error::cache("timed out waiting for a cache writer lease"));
                    }
                    if options.poll_interval.is_zero() {
                        thread::yield_now();
                    } else {
                        thread::sleep(options.poll_interval);
                    }
                }
                Err(error) => {
                    return Err(Error::io("failed to acquire a cache writer lease", error));
                }
            }
        }
    }

    fn refresh(&mut self) -> Result<()> {
        let owner = fs::read_to_string(self.path.join(OWNER_FILE))
            .map_err(|error| Error::io("failed to verify a cache writer lease", error))?;
        if owner != self.owner {
            self.active = false;
            return Err(Error::cache("cache writer lease ownership changed"));
        }
        let mut heartbeat = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(self.path.join(HEARTBEAT_FILE))
            .map_err(|error| Error::io("failed to refresh a cache writer lease", error))?;
        heartbeat
            .write_all(unix_millis().to_string().as_bytes())
            .map_err(|error| Error::io("failed to refresh a cache writer lease", error))
    }
}

impl Drop for WriterLease {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let still_owned =
            fs::read_to_string(self.path.join(OWNER_FILE)).is_ok_and(|owner| owner == self.owner);
        if still_owned {
            release_writer_lease(&self.path);
        }
    }
}

#[expect(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "`.lease` is an exact internal directory suffix, not a user file extension"
)]
fn recover_shard(
    shard: &Path,
    stale_after: Duration,
    report: &mut RecoveryReport,
    scan: &mut ScanBudget<'_>,
) -> Result<()> {
    for path in read_directories(shard, scan)? {
        scan.check()?;
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let recoverable = name.contains(".tmp-")
            || name.contains(REPLACEMENT_MARKER)
            || name.contains(".evicted-")
            || name.contains(".lease.reap-")
            || name.ends_with(".lease");
        if !recoverable || !path_is_stale(&path, stale_after, scan)? {
            continue;
        }
        if name.contains(".tmp-") || name.contains(REPLACEMENT_MARKER) {
            let key_prefix = if name.contains(".tmp-") {
                name.split(".tmp-").next().unwrap_or_default()
            } else {
                name.split(REPLACEMENT_MARKER).next().unwrap_or_default()
            };
            let lease_path = shard.join(format!("{key_prefix}.lease"));
            if lease_path.exists() && !lease_is_stale(&lease_path, stale_after, scan.cancellation)?
            {
                continue;
            }
        }
        if name.contains(REPLACEMENT_MARKER) {
            restore_replacement(shard, &path, name, report, scan)?;
            continue;
        }
        let bytes = directory_size(&path, scan, 0)?;
        if name.ends_with(".lease") {
            reap_directory(&path)?;
        } else {
            fs::remove_dir_all(&path)
                .map_err(|error| Error::io("failed to recover stale cache staging data", error))?;
        }
        report.removed_directories = report.removed_directories.saturating_add(1);
        report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(bytes);
    }
    Ok(())
}

fn restore_replacement(
    shard: &Path,
    replacement_path: &Path,
    replacement_name: &str,
    report: &mut RecoveryReport,
    scan: &mut ScanBudget<'_>,
) -> Result<()> {
    let Some(key_name) = replacement_name
        .strip_prefix('.')
        .and_then(|name| name.split(REPLACEMENT_MARKER).next())
        .filter(|name| !name.is_empty())
    else {
        return Ok(());
    };
    let final_path = shard.join(key_name);
    if final_path.exists() {
        let bytes = directory_size(replacement_path, scan, 0)?;
        fs::remove_dir_all(replacement_path)
            .map_err(|error| Error::io("failed to remove a superseded cache generation", error))?;
        report.removed_directories = report.removed_directories.saturating_add(1);
        report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(bytes);
        return Ok(());
    }

    match rename_directory(replacement_path, &final_path) {
        Ok(()) => {
            report.restored_entries = report.restored_entries.saturating_add(1);
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(_error) if final_path.is_dir() => {
            let bytes = directory_size(replacement_path, scan, 0)?;
            fs::remove_dir_all(replacement_path).map_err(|cleanup_error| {
                Error::io(
                    "failed to remove a superseded cache generation",
                    cleanup_error,
                )
            })?;
            report.removed_directories = report.removed_directories.saturating_add(1);
            report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(bytes);
            Ok(())
        }
        Err(error) => Err(Error::io(
            "failed to restore an interrupted cache replacement",
            error,
        )),
    }
}

fn reap_directory(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::cache("recoverable cache path has no parent"))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::cache("recoverable cache path has no portable name"))?;
    let claimed = parent.join(format!("{name}.reap-{}", unique_token()));
    match fs::rename(path, &claimed) {
        Ok(()) => fs::remove_dir_all(claimed)
            .map_err(|error| Error::io("failed to remove a recovered cache directory", error)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::io(
            "failed to atomically claim a stale cache directory",
            error,
        )),
    }
}

#[derive(Debug)]
struct ScanBudget<'a> {
    remaining_entries: usize,
    cancellation: &'a CancellationToken,
}

impl<'a> ScanBudget<'a> {
    fn new(maximum_entries: usize, cancellation: &'a CancellationToken) -> Result<Self> {
        cancellation.check()?;
        if maximum_entries == 0 {
            return Err(Error::limit(
                "cache maximum scan entries must be greater than zero",
            ));
        }
        Ok(Self {
            remaining_entries: maximum_entries,
            cancellation,
        })
    }

    fn visit_entry(&mut self) -> Result<()> {
        self.check()?;
        let Some(remaining_entries) = self.remaining_entries.checked_sub(1) else {
            return Err(Error::limit(
                "cache maintenance scan exceeded the configured entry limit",
            ));
        };
        self.remaining_entries = remaining_entries;
        Ok(())
    }

    fn check(&self) -> Result<()> {
        self.cancellation.check()
    }
}

fn read_directories(path: &Path, scan: &mut ScanBudget<'_>) -> Result<Vec<PathBuf>> {
    scan.check()?;
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(Error::io("failed to enumerate a cache directory", error)),
    };
    let mut directories = Vec::new();
    for entry in entries {
        scan.visit_entry()?;
        let entry =
            entry.map_err(|error| Error::io("failed to enumerate a cache directory", error))?;
        let file_type = entry
            .file_type()
            .map_err(|error| Error::io("failed to inspect a cache directory entry", error))?;
        if file_type.is_dir() {
            directories
                .try_reserve(1)
                .map_err(|_error| Error::limit("cache directory inventory allocation failed"))?;
            directories.push(entry.path());
        }
    }
    directories.sort_unstable();
    Ok(directories)
}

fn directory_size(path: &Path, scan: &mut ScanBudget<'_>, depth: usize) -> Result<u64> {
    scan.check()?;
    if depth > MAXIMUM_CACHE_SCAN_DEPTH {
        return Err(Error::limit(
            "cache maintenance scan exceeded the maximum directory depth",
        ));
    }
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(Error::io("failed to inspect cache storage", error)),
    };
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }
    let entries = fs::read_dir(path)
        .map_err(|error| Error::io("failed to enumerate cache storage", error))?;
    let mut total = 0_u64;
    for entry in entries {
        scan.visit_entry()?;
        let entry = entry.map_err(|error| Error::io("failed to enumerate cache storage", error))?;
        total = total.saturating_add(directory_size(
            &entry.path(),
            scan,
            depth.saturating_add(1),
        )?);
    }
    Ok(total)
}

fn path_is_stale(path: &Path, stale_after: Duration, scan: &mut ScanBudget<'_>) -> Result<bool> {
    scan.check()?;
    let mut newest = match fs::metadata(path).and_then(|metadata| metadata.modified()) {
        Ok(modified) => modified,
        // A vanished path is not evidence that a subsequently created lease is
        // stale. Treating it as stale creates an ABA race with the next owner.
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        #[cfg(windows)]
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => return Ok(false),
        Err(error) => {
            return Err(Error::io("failed to inspect cache recovery age", error));
        }
    };
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries {
            scan.visit_entry()?;
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(Error::io("failed to inspect cache recovery age", error));
                }
            };
            let modified = match entry.metadata().and_then(|metadata| metadata.modified()) {
                Ok(modified) => modified,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                #[cfg(windows)]
                Err(error) if error.kind() == io::ErrorKind::PermissionDenied => return Ok(false),
                Err(error) => {
                    return Err(Error::io("failed to inspect cache recovery age", error));
                }
            };
            if modified > newest {
                newest = modified;
            }
        }
    }
    Ok(SystemTime::now().duration_since(newest).unwrap_or_default() >= stale_after)
}

fn lease_is_stale(
    path: &Path,
    stale_after: Duration,
    cancellation: &CancellationToken,
) -> Result<bool> {
    cancellation.check()?;
    let heartbeat = path.join(HEARTBEAT_FILE);
    let modified = match fs::metadata(&heartbeat).and_then(|metadata| metadata.modified()) {
        Ok(modified) => modified,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            match fs::metadata(path).and_then(|metadata| metadata.modified()) {
                Ok(modified) => modified,
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
                #[cfg(windows)]
                Err(error) if error.kind() == io::ErrorKind::PermissionDenied => return Ok(false),
                Err(error) => {
                    return Err(Error::io("failed to inspect cache lease age", error));
                }
            }
        }
        #[cfg(windows)]
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => return Ok(false),
        Err(error) => return Err(Error::io("failed to inspect cache lease age", error)),
    };
    Ok(SystemTime::now()
        .duration_since(modified)
        .unwrap_or_default()
        >= stale_after)
}

fn lease_is_contended(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::AlreadyExists {
        return true;
    }
    #[cfg(windows)]
    {
        // Windows can report ERROR_ACCESS_DENIED while another writer renames
        // or finishes deleting the lease directory.
        if windows_filesystem_operation_is_transient(error) {
            return true;
        }
    }
    false
}

fn release_writer_lease(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    for attempt in 0..LEASE_RELEASE_RETRIES {
        #[cfg(not(windows))]
        let _attempt = attempt;
        let claimed = parent.join(format!("{name}.reap-{}", unique_token()));
        match fs::rename(path, &claimed) {
            Ok(()) => {
                let _cleanup = fs::remove_dir_all(claimed);
                return;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => return,
            #[cfg(windows)]
            Err(error) if windows_filesystem_operation_is_transient(&error) => {
                if attempt + 1 < LEASE_RELEASE_RETRIES {
                    thread::sleep(LEASE_RELEASE_RETRY_INTERVAL);
                }
            }
            Err(_) => return,
        }
    }
}

#[cfg(windows)]
fn windows_filesystem_operation_is_transient(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::PermissionDenied
        || error.kind() == io::ErrorKind::WouldBlock
        || error.raw_os_error() == Some(32)
}

#[cfg(not(windows))]
fn rename_directory(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn rename_directory(source: &Path, destination: &Path) -> io::Result<()> {
    for attempt in 0..LEASE_RELEASE_RETRIES {
        match fs::rename(source, destination) {
            Err(error)
                if windows_filesystem_operation_is_transient(&error)
                    && attempt + 1 < LEASE_RELEASE_RETRIES =>
            {
                thread::sleep(LEASE_RELEASE_RETRY_INTERVAL);
            }
            result => return result,
        }
    }
    Err(io::Error::other(
        "cache directory rename exhausted its retry limit",
    ))
}

fn unique_token() -> String {
    let sequence = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos:x}-{sequence:x}", std::process::id())
}

fn unix_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}
