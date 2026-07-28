use std::backtrace::Backtrace;
use std::error::Error as StdError;
use std::fmt::{self, Display, Formatter};

type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// A stable, caller-actionable class of model-weight failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCategory {
    /// An operating-system or filesystem operation failed.
    Io,
    /// A repository-relative or local path was invalid.
    InvalidPath,
    /// A checkpoint, index, plan, or cache envelope was malformed.
    InvalidFormat,
    /// Content did not match its declared identity or length.
    Integrity,
    /// A configured allocation, count, rank, or byte limit was exceeded.
    ResourceLimit,
    /// Weight selection, aliasing, overlay, or target binding failed.
    Binding,
    /// No declared transform, encoding, layout, or backend capability matched.
    Unsupported,
    /// A cache lookup, lease, publication, validation, or eviction failed.
    Cache,
    /// Cooperative cancellation stopped the operation.
    Cancelled,
    /// A consumer delivery callback rejected prepared data.
    Delivery,
}

impl Display for ErrorCategory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Io => "I/O",
            Self::InvalidPath => "invalid path",
            Self::InvalidFormat => "invalid format",
            Self::Integrity => "integrity",
            Self::ResourceLimit => "resource limit",
            Self::Binding => "binding",
            Self::Unsupported => "unsupported capability",
            Self::Cache => "cache",
            Self::Cancelled => "cancelled",
            Self::Delivery => "delivery",
        };
        formatter.write_str(name)
    }
}

/// An error produced by model-weight operations.
#[derive(Debug)]
pub struct Error {
    category: ErrorCategory,
    message: Box<str>,
    source: Option<BoxError>,
    backtrace: Backtrace,
}

impl Error {
    /// Creates an extension error in a stable recovery category.
    ///
    /// This is intended for external preparation providers, cache adapters,
    /// and delivery sinks that must participate in the crate's error contract
    /// without discarding their own context.
    #[must_use]
    pub fn from_category(category: ErrorCategory, message: impl Into<Box<str>>) -> Self {
        Self::new(category, message)
    }

    /// Creates an extension error with a preserved underlying source.
    ///
    /// Consumers should select the category that determines recovery behavior;
    /// the source remains available through [`std::error::Error::source`].
    #[must_use]
    pub fn from_category_with_source(
        category: ErrorCategory,
        message: impl Into<Box<str>>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::with_source(category, message, source)
    }

    pub(crate) fn new(category: ErrorCategory, message: impl Into<Box<str>>) -> Self {
        Self {
            category,
            message: message.into(),
            source: None,
            backtrace: Backtrace::capture(),
        }
    }

    pub(crate) fn with_source(
        category: ErrorCategory,
        message: impl Into<Box<str>>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            category,
            message: message.into(),
            source: Some(Box::new(source)),
            backtrace: Backtrace::capture(),
        }
    }

    pub(crate) fn io(message: impl Into<Box<str>>, source: std::io::Error) -> Self {
        Self::with_source(ErrorCategory::Io, message, source)
    }

    pub(crate) fn invalid(message: impl Into<Box<str>>) -> Self {
        Self::new(ErrorCategory::InvalidFormat, message)
    }

    pub(crate) fn integrity(message: impl Into<Box<str>>) -> Self {
        Self::new(ErrorCategory::Integrity, message)
    }

    pub(crate) fn limit(message: impl Into<Box<str>>) -> Self {
        Self::new(ErrorCategory::ResourceLimit, message)
    }

    pub(crate) fn binding(message: impl Into<Box<str>>) -> Self {
        Self::new(ErrorCategory::Binding, message)
    }

    pub(crate) fn unsupported(message: impl Into<Box<str>>) -> Self {
        Self::new(ErrorCategory::Unsupported, message)
    }

    pub(crate) fn cache(message: impl Into<Box<str>>) -> Self {
        Self::new(ErrorCategory::Cache, message)
    }

    pub(crate) fn delivery(message: impl Into<Box<str>>) -> Self {
        Self::new(ErrorCategory::Delivery, message)
    }

    pub(crate) fn cancelled() -> Self {
        Self::new(
            ErrorCategory::Cancelled,
            "model-weight operation was cancelled",
        )
    }

    /// Returns the stable category for programmatic recovery.
    #[must_use]
    pub const fn category(&self) -> ErrorCategory {
        self.category
    }

    /// Returns the contextual, lower-case diagnostic message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the construction backtrace.
    pub const fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }

    /// Returns whether cooperative cancellation caused this error.
    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        matches!(self.category, ErrorCategory::Cancelled)
    }

    /// Returns whether malformed or inconsistent caller data caused this error.
    #[must_use]
    pub const fn is_invalid(&self) -> bool {
        matches!(
            self.category,
            ErrorCategory::InvalidPath
                | ErrorCategory::InvalidFormat
                | ErrorCategory::Integrity
                | ErrorCategory::Binding
        )
    }

    /// Returns whether the requested operation lacks an implementation.
    #[must_use]
    pub const fn is_unsupported(&self) -> bool {
        matches!(self.category, ErrorCategory::Unsupported)
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.category, self.message)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

/// A result returned by model-weight operations.
pub type Result<T> = std::result::Result<T, Error>;
