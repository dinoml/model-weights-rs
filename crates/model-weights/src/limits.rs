//! Resource limits for untrusted formats and bounded execution.

use crate::{Error, Result};

const DEFAULT_MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const DEFAULT_MAX_TENSORS: usize = 1_000_000;
const DEFAULT_MAX_SHARDS: usize = 16_384;
const DEFAULT_MAX_RANK: usize = 64;
const DEFAULT_MAX_NAME_BYTES: usize = 16 * 1024;
const DEFAULT_MAX_METADATA_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_MAX_WORK_ITEMS: usize = 1_000_000;
const DEFAULT_DELIVERY_QUEUE_DEPTH: usize = 2;

/// Limits applied while parsing checkpoint and index metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::struct_field_names,
    reason = "the max_ prefix distinguishes configured upper bounds from measured values"
)]
pub struct ResourceLimits {
    max_header_bytes: u64,
    max_tensors: usize,
    max_shards: usize,
    max_rank: usize,
    max_name_bytes: usize,
    max_metadata_bytes: usize,
}

impl ResourceLimits {
    /// Returns a builder initialized with conservative defaults.
    pub fn builder() -> ResourceLimitsBuilder {
        ResourceLimitsBuilder::default()
    }

    /// Returns the maximum bytes allocated for one safetensors header.
    #[must_use]
    pub const fn max_header_bytes(&self) -> u64 {
        self.max_header_bytes
    }

    /// Returns the maximum tensors accepted in one inventory.
    #[must_use]
    pub const fn max_tensors(&self) -> usize {
        self.max_tensors
    }

    /// Returns the maximum unique shard files accepted.
    #[must_use]
    pub const fn max_shards(&self) -> usize {
        self.max_shards
    }

    /// Returns the maximum tensor rank accepted.
    #[must_use]
    pub const fn max_rank(&self) -> usize {
        self.max_rank
    }

    /// Returns the maximum UTF-8 bytes accepted for one tensor or path name.
    #[must_use]
    pub const fn max_name_bytes(&self) -> usize {
        self.max_name_bytes
    }

    /// Returns the maximum opaque quantization metadata bytes accepted.
    #[must_use]
    pub const fn max_metadata_bytes(&self) -> usize {
        self.max_metadata_bytes
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
            max_tensors: DEFAULT_MAX_TENSORS,
            max_shards: DEFAULT_MAX_SHARDS,
            max_rank: DEFAULT_MAX_RANK,
            max_name_bytes: DEFAULT_MAX_NAME_BYTES,
            max_metadata_bytes: DEFAULT_MAX_METADATA_BYTES,
        }
    }
}

/// Builds validated checkpoint resource limits.
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct ResourceLimitsBuilder {
    limits: ResourceLimits,
}

impl ResourceLimitsBuilder {
    /// Sets the maximum safetensors header length.
    pub const fn max_header_bytes(mut self, value: u64) -> Self {
        self.limits.max_header_bytes = value;
        self
    }

    /// Sets the maximum inventory tensor count.
    pub const fn max_tensors(mut self, value: usize) -> Self {
        self.limits.max_tensors = value;
        self
    }

    /// Sets the maximum unique shard count.
    pub const fn max_shards(mut self, value: usize) -> Self {
        self.limits.max_shards = value;
        self
    }

    /// Sets the maximum tensor rank.
    pub const fn max_rank(mut self, value: usize) -> Self {
        self.limits.max_rank = value;
        self
    }

    /// Sets the maximum tensor or path name length.
    pub const fn max_name_bytes(mut self, value: usize) -> Self {
        self.limits.max_name_bytes = value;
        self
    }

    /// Sets the maximum opaque quantization metadata length.
    pub const fn max_metadata_bytes(mut self, value: usize) -> Self {
        self.limits.max_metadata_bytes = value;
        self
    }

    /// Validates and builds the limits.
    ///
    /// # Errors
    ///
    /// Returns a resource-limit error when any limit is zero.
    pub fn build(self) -> Result<ResourceLimits> {
        let limits = self.limits;
        let all_nonzero = limits.max_header_bytes > 0
            && limits.max_tensors > 0
            && limits.max_shards > 0
            && limits.max_rank > 0
            && limits.max_name_bytes > 0
            && limits.max_metadata_bytes > 0;
        if !all_nonzero {
            return Err(Error::limit(
                "resource limits must all be greater than zero",
            ));
        }
        Ok(limits)
    }
}

/// Limits for the bounded preparation and delivery pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionLimits {
    /// Maximum host preparation workers.
    ///
    /// [`Self::default`] uses the host's reported available parallelism. A
    /// caller may override this independently of every queue and byte limit.
    pub workers: usize,
    /// Maximum work-item descriptors accepted by one pipeline invocation.
    ///
    /// The pipeline validates this limit while collecting caller metadata,
    /// before sorting or starting worker threads.
    pub max_work_items: usize,
    /// Maximum worker results buffered while the coordinator consumes them.
    pub delivery_queue_depth: usize,
    /// Maximum dispatched work items not yet released after ordered delivery.
    ///
    /// This bounds speculative preparation ahead of a slow earlier ordinal
    /// independently of both the worker count and result-channel capacity.
    pub dispatch_lookahead: usize,
    /// Maximum concurrently owned source-read bytes.
    ///
    /// Automatic access reserves its possible owned-read fallback. Only an
    /// explicit retained mapping can omit these bytes from admission.
    pub source_bytes: u64,
    /// Maximum temporary transform scratch bytes.
    pub scratch_bytes: u64,
    /// Maximum prepared bytes retained before delivery completes.
    pub prepared_bytes: u64,
}

impl ExecutionLimits {
    /// Validates that every execution limit is nonzero.
    ///
    /// # Errors
    ///
    /// Returns a resource-limit error when any limit is zero.
    pub fn validate(&self) -> Result<()> {
        if self.workers == 0
            || self.max_work_items == 0
            || self.delivery_queue_depth == 0
            || self.dispatch_lookahead == 0
            || self.source_bytes == 0
            || self.scratch_bytes == 0
            || self.prepared_bytes == 0
        {
            return Err(Error::limit(
                "execution limits must all be greater than zero",
            ));
        }
        Ok(())
    }
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        let available_parallelism =
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        Self {
            workers: available_parallelism,
            max_work_items: DEFAULT_MAX_WORK_ITEMS,
            delivery_queue_depth: DEFAULT_DELIVERY_QUEUE_DEPTH,
            dispatch_lookahead: available_parallelism,
            source_bytes: 256 * 1024 * 1024,
            scratch_bytes: 64 * 1024 * 1024,
            prepared_bytes: 512 * 1024 * 1024,
        }
    }
}
