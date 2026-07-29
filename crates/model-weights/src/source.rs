//! Local and retained-snapshot source descriptors.

use std::any::Any;
use std::fmt::{self, Debug, Formatter};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use crate::identity::ContentDigest;
use crate::{Error, ErrorCategory, Result};

/// A validated portable repository-relative path.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepoPath(Box<str>);

impl RepoPath {
    /// Parses a slash-separated repository path.
    ///
    /// # Errors
    ///
    /// Returns an invalid-path error for absolute, empty, dot, parent, Windows
    /// prefix, backslash, NUL, or empty path components.
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let value = value.as_ref();
        let invalid_text = value.is_empty()
            || value.contains('\\')
            || value.contains(':')
            || value.contains('\0')
            || value.chars().any(char::is_control)
            || value.starts_with('/')
            || value.ends_with('/');
        let invalid_component = value
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."));
        let platform_absolute = Path::new(value).components().any(|component| {
            matches!(
                component,
                Component::Prefix(_) | Component::RootDir | Component::ParentDir
            )
        });
        if invalid_text || invalid_component || platform_absolute {
            return Err(Error::new(
                ErrorCategory::InvalidPath,
                "repository path must be a safe slash-separated relative path",
            ));
        }
        Ok(Self(value.into()))
    }

    /// Returns the canonical slash-separated path.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Describes how a source digest becomes trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DigestPolicy {
    /// Compute SHA-256 only when a content-addressed identity is requested.
    ComputeOnDemand,
    /// Hash on demand and compare with the expected SHA-256.
    VerifyOnDemand(ContentDigest),
    /// Reuse a digest that an upstream source already verified.
    ///
    /// This policy does not make an ordinary local source eligible for mapped
    /// access. The caller remains responsible for preventing concurrent
    /// mutation under the ordinary-local-file contract.
    TrustExternal(ContentDigest),
    /// Reuse an externally verified digest while retaining immutable ownership.
    TrustRetained(ContentDigest),
}

/// Distinguishes ordinary local files from immutable retained snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SourceKind {
    /// An ordinary caller-managed local file.
    Local,
    /// A file protected by a retained immutable snapshot lifetime.
    RetainedSnapshot,
}

#[derive(Clone)]
struct Retention {
    #[cfg_attr(
        not(feature = "mmap"),
        expect(
            dead_code,
            reason = "the guard is retained for ownership even when mapped views are disabled"
        )
    )]
    guard: Arc<dyn Any + Send + Sync>,
}

impl Debug for Retention {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("Retention(..)")
    }
}

/// A live descriptor for one checkpoint file.
#[derive(Clone)]
pub struct SourceDescriptor {
    logical_path: RepoPath,
    local_path: PathBuf,
    expected_size: Option<u64>,
    digest_policy: DigestPolicy,
    kind: SourceKind,
    retention: Option<Retention>,
}

impl SourceDescriptor {
    /// Describes an ordinary local file with digest computation deferred.
    ///
    /// # Errors
    ///
    /// Returns an invalid-path error when `path` has no portable file name.
    pub fn local(path: impl AsRef<Path>) -> Result<Self> {
        let local_path = path.as_ref().to_owned();
        let name = local_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                Error::new(
                    ErrorCategory::InvalidPath,
                    "local checkpoint path must have a UTF-8 file name",
                )
            })?;
        Ok(Self {
            logical_path: RepoPath::parse(name)?,
            local_path,
            expected_size: None,
            digest_policy: DigestPolicy::ComputeOnDemand,
            kind: SourceKind::Local,
            retention: None,
        })
    }

    /// Describes a local file with an expected content digest.
    ///
    /// Verification is deferred until content identity is requested, so header
    /// inventory does not scan tensor payloads.
    ///
    /// # Errors
    ///
    /// Returns an invalid-path error when `path` has no portable file name.
    pub fn local_with_digest(
        path: impl AsRef<Path>,
        expected_size: u64,
        digest: ContentDigest,
    ) -> Result<Self> {
        let mut descriptor = Self::local(path)?;
        descriptor.expected_size = Some(expected_size);
        descriptor.digest_policy = DigestPolicy::VerifyOnDemand(digest);
        Ok(descriptor)
    }

    /// Describes a local file whose digest was already verified upstream.
    ///
    /// Unlike [`Self::retained`], this constructor does not assert that mapped
    /// access is safe. The source remains an ordinary local file and is always
    /// copied when [`crate::AccessMode::Auto`] is used.
    ///
    /// The caller must ensure that `digest` and `expected_size` describe the
    /// selected file and must prevent concurrent mutation while the checkpoint
    /// is in use. An incorrect trusted digest can corrupt content-addressed
    /// identities and cache entries, but cannot enable file-backed mapping.
    ///
    /// # Errors
    ///
    /// Returns an invalid-path error when `path` has no portable file name.
    pub fn local_with_trusted_digest(
        path: impl AsRef<Path>,
        expected_size: u64,
        digest: ContentDigest,
    ) -> Result<Self> {
        let mut descriptor = Self::local(path)?;
        descriptor.expected_size = Some(expected_size);
        descriptor.digest_policy = DigestPolicy::TrustExternal(digest);
        Ok(descriptor)
    }

    /// Describes a retained immutable snapshot file.
    ///
    /// `guard` is held until every checkpoint, mapping, and zero-copy byte view
    /// derived from this descriptor has been dropped.
    ///
    /// # Safety
    ///
    /// Until the guard and every clone derived from this descriptor are
    /// dropped, the caller must guarantee that `local_path` continues to name
    /// the same regular file and that its length and contents cannot be
    /// modified or truncated by this process or another process. `size` and
    /// `digest` must describe those exact immutable bytes. Violating these
    /// requirements can make a file-backed mapping unsound.
    ///
    /// # Errors
    ///
    /// Returns an invalid-path error when `logical_path` is unsafe.
    #[expect(
        unsafe_code,
        reason = "the caller must uphold file-backed mapping invariants that the OS cannot prove"
    )]
    pub unsafe fn retained<T>(
        logical_path: impl AsRef<str>,
        local_path: impl AsRef<Path>,
        size: u64,
        digest: ContentDigest,
        guard: T,
    ) -> Result<Self>
    where
        T: Send + Sync + 'static,
    {
        Ok(Self {
            logical_path: RepoPath::parse(logical_path)?,
            local_path: local_path.as_ref().to_owned(),
            expected_size: Some(size),
            digest_policy: DigestPolicy::TrustRetained(digest),
            kind: SourceKind::RetainedSnapshot,
            retention: Some(Retention {
                guard: Arc::new(guard),
            }),
        })
    }

    /// Replaces the repository-relative path associated with this source.
    ///
    /// # Errors
    ///
    /// Returns an invalid-path error when `logical_path` is unsafe.
    pub fn with_logical_path(mut self, logical_path: impl AsRef<str>) -> Result<Self> {
        self.logical_path = RepoPath::parse(logical_path)?;
        Ok(self)
    }

    /// Returns the repository-relative provenance path.
    #[must_use]
    pub const fn logical_path(&self) -> &RepoPath {
        &self.logical_path
    }

    /// Returns the local path opened by the checkpoint.
    #[must_use]
    pub fn local_path(&self) -> &Path {
        &self.local_path
    }

    /// Returns the declared file size when one was supplied.
    #[must_use]
    pub const fn expected_size(&self) -> Option<u64> {
        self.expected_size
    }

    /// Returns how the content digest becomes trusted.
    #[must_use]
    pub const fn digest_policy(&self) -> DigestPolicy {
        self.digest_policy
    }

    /// Returns the source lifetime class.
    #[must_use]
    pub const fn kind(&self) -> SourceKind {
        self.kind
    }

    #[cfg(feature = "mmap")]
    pub(crate) fn retention(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.retention
            .as_ref()
            .map(|retention| Arc::clone(&retention.guard))
    }
}

impl Debug for SourceDescriptor {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SourceDescriptor")
            .field("logical_path", &self.logical_path)
            .field("local_path", &self.local_path)
            .field("expected_size", &self.expected_size)
            .field("digest_policy", &self.digest_policy)
            .field("kind", &self.kind)
            .field("retained", &self.retention.is_some())
            .finish()
    }
}
