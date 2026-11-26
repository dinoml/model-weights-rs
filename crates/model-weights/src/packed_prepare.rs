//! Bounded host preparation of packed weights through consumer-owned providers.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

use serde::Serialize;

use crate::identity::{ImplementationId, StableName};
use crate::plan::TargetTensor;
use crate::quantization::{LayoutRequirement, QuantizedRoute, QuantizedStorage};
use crate::tensor::ByteView;
use crate::{CancellationToken, Error, ErrorCategory, Result};

const ENVELOPE_MAGIC: &[u8; 8] = b"MWPACK01";
const MAX_DESCRIPTOR_BYTES: usize = 64 * 1024;

/// Metadata-only facts supplied to a packed preparation provider.
#[derive(Debug, Clone, Copy)]
pub struct PackedPrepareRequest<'a> {
    /// Canonical source encoding, geometry, payload and companion descriptors.
    pub source: &'a QuantizedStorage,
    /// Exact planned logical target, custom layout, implementation and byte size.
    pub target: &'a TargetTensor,
}

/// Exact caller-owned storage required by one validated host preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedPreparationSizes {
    /// Number of initialized output bytes, excluding the cache envelope.
    pub output_bytes: u64,
    /// Temporary memory that is released before delivery.
    pub scratch_bytes: u64,
}

/// A consumer-owned, deterministic packed-byte transform.
pub trait PackedPreparationProvider: Debug + Send + Sync {
    /// Returns the exact byte-affecting implementation identity.
    fn implementation(&self) -> &ImplementationId;

    /// Validate metadata and return storage requirements without reading payloads.
    ///
    /// # Errors
    /// Reject unsupported formats/layouts, invalid extents, or cancellation.
    fn validate(
        &self,
        request: PackedPrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<PackedPreparationSizes>;

    /// Write the complete output using only caller-owned output and scratch.
    ///
    /// Payload and companions remain immutable. Providers must check cancellation
    /// at bounded block/tile intervals. Host byte slices need not have device
    /// alignment; providers must use byte-safe writes. Device allocation is forbidden.
    ///
    /// # Errors
    /// Return input-validation, provider, or cancellation errors before publication.
    fn prepare_into(
        &self,
        request: PackedPrepareRequest<'_>,
        payload: &ByteView,
        companions: &BTreeMap<StableName, ByteView>,
        output: &mut [u8],
        scratch: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<()>;
}

/// Exact packed-provider identities. The core crate supplies no kernel FFI.
#[derive(Debug, Clone, Default)]
pub struct PackedProviderRegistry {
    providers: BTreeMap<ImplementationId, Arc<dyn PackedPreparationProvider>>,
}

impl PackedProviderRegistry {
    /// Register one exact provider identity.
    ///
    /// # Errors
    /// Reject duplicate implementations instead of replacing existing behavior.
    pub fn register<P: PackedPreparationProvider + 'static>(&mut self, provider: P) -> Result<()> {
        let identity = provider.implementation().clone();
        if self.providers.contains_key(&identity) {
            return Err(Error::invalid(
                "packed provider identity is already registered",
            ));
        }
        self.providers.insert(identity, Arc::new(provider));
        Ok(())
    }

    /// Resolve the exact implementation pinned by a binding.
    ///
    /// # Errors
    /// Return unsupported when that exact implementation was not registered.
    pub fn resolve(&self, identity: &ImplementationId) -> Result<&dyn PackedPreparationProvider> {
        self.providers
            .get(identity)
            .map(Arc::as_ref)
            .ok_or_else(|| {
                Error::unsupported("packed preparation implementation is not registered")
            })
    }

    /// Validate one host-repack request without reading any tensor bytes.
    ///
    /// # Errors
    /// Reject route, identity, output-size or provider-validation mismatches.
    pub fn validate(
        &self,
        request: PackedPrepareRequest<'_>,
        cancellation: &CancellationToken,
    ) -> Result<PackedPreparationSizes> {
        cancellation.check()?;
        let capability = request
            .target
            .quantized_route()
            .ok_or_else(|| Error::binding("packed target has no route"))?;
        if !matches!(capability.route(), QuantizedRoute::HostRepack { .. })
            || capability.source_encoding() != request.source.encoding()
        {
            return Err(Error::unsupported(
                "packed request does not match its source or host route",
            ));
        }
        let sizes = self
            .resolve(capability.implementation())?
            .validate(request, cancellation)?;
        if sizes.output_bytes != request.target.output_size() {
            return Err(Error::binding(
                "packed provider output size differs from its target",
            ));
        }
        sizes
            .output_bytes
            .checked_add(sizes.scratch_bytes)
            .ok_or_else(|| Error::limit("packed preparation allocation overflows"))?;
        cancellation.check()?;
        Ok(sizes)
    }

    /// Execute into already reserved output and scratch storage.
    ///
    /// # Errors
    /// Reject inconsistent input/output sizes and propagate provider errors.
    pub fn prepare_into(
        &self,
        request: PackedPrepareRequest<'_>,
        payload: &ByteView,
        companions: &BTreeMap<StableName, ByteView>,
        output: &mut [u8],
        scratch: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let sizes = self.validate(request, cancellation)?;
        if u64::try_from(output.len()).ok() != Some(sizes.output_bytes)
            || u64::try_from(scratch.len()).ok() != Some(sizes.scratch_bytes)
            || u64::try_from(payload.len()).ok() != Some(request.source.span().len())
            || companions.len() != request.source.companions().len()
        {
            return Err(Error::binding(
                "packed preparation buffer extents differ from metadata",
            ));
        }
        for (role, descriptor) in request.source.companions() {
            cancellation.check()?;
            if companions
                .get(role)
                .and_then(|bytes| u64::try_from(bytes.len()).ok())
                != Some(descriptor.span().len())
            {
                return Err(Error::binding(
                    "packed companion byte extent differs from metadata",
                ));
            }
        }
        let capability = request
            .target
            .quantized_route()
            .ok_or_else(|| Error::binding("packed target has no route"))?;
        self.resolve(capability.implementation())?.prepare_into(
            request,
            payload,
            companions,
            output,
            scratch,
            cancellation,
        )?;
        cancellation.check()
    }
}

/// Typed numerical and physical contract retained with fresh and cached bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackedOutputDescriptor {
    source_encoding: ImplementationId,
    encoding: ImplementationId,
    logical_shape: Box<[u64]>,
    layout: LayoutRequirement,
    bytes: u64,
}

impl PackedOutputDescriptor {
    pub(crate) fn from_target(target: &TargetTensor) -> Result<Self> {
        let capability = target
            .quantized_route()
            .ok_or_else(|| Error::binding("packed target has no capability"))?;
        let QuantizedRoute::HostRepack { target_encoding } = capability.route() else {
            return Err(Error::binding("target is not a host repack"));
        };
        Ok(Self {
            source_encoding: capability.source_encoding().clone(),
            encoding: target_encoding.clone(),
            logical_shape: target.shape().into(),
            layout: capability
                .target_layout()
                .cloned()
                .ok_or_else(|| Error::binding("packed target has no layout"))?,
            bytes: target.output_size(),
        })
    }

    /// Returns the source numerical encoding identity.
    #[must_use]
    pub const fn source_encoding(&self) -> &ImplementationId {
        &self.source_encoding
    }
    /// Returns the prepared encoding identity.
    #[must_use]
    pub const fn encoding(&self) -> &ImplementationId {
        &self.encoding
    }
    /// Returns the logical matrix shape, independently of byte storage size.
    #[must_use]
    pub fn logical_shape(&self) -> &[u64] {
        &self.logical_shape
    }
    /// Returns the complete physical-layout ABI requirement.
    #[must_use]
    pub const fn layout(&self) -> &LayoutRequirement {
        &self.layout
    }
    /// Returns the actual prepared byte extent.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn prefix(&self) -> Result<Vec<u8>> {
        let descriptor = serde_json::to_vec(self).map_err(|error| {
            Error::with_source(
                ErrorCategory::InvalidFormat,
                "serialize packed descriptor",
                error,
            )
        })?;
        if descriptor.len() > MAX_DESCRIPTOR_BYTES {
            return Err(Error::limit(
                "packed descriptor exceeds cache metadata limit",
            ));
        }
        let length = u32::try_from(descriptor.len())
            .map_err(|_error| Error::limit("packed descriptor overflows"))?;
        let mut output = Vec::with_capacity(12 + descriptor.len());
        output.extend_from_slice(ENVELOPE_MAGIC);
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(&descriptor);
        Ok(output)
    }

    pub(crate) fn cached_len(&self) -> Result<u64> {
        let prefix = u64::try_from(self.prefix()?.len())
            .map_err(|_error| Error::limit("packed prefix length overflows"))?;
        self.bytes
            .checked_add(prefix)
            .ok_or_else(|| Error::limit("packed cache extent overflows"))
    }

    pub(crate) fn unpack_cached(&self, bytes: &ByteView) -> Result<ByteView> {
        let prefix = self.prefix()?;
        if u64::try_from(bytes.len()).ok() != Some(self.cached_len()?)
            || !bytes.as_slice().starts_with(&prefix)
        {
            return Err(Error::integrity(
                "cached packed descriptor differs from the target contract",
            ));
        }
        bytes.slice(prefix.len()..bytes.len())
    }
}

pub(crate) fn allocate(length: u64) -> Result<Vec<u8>> {
    let size = usize::try_from(length)
        .map_err(|_error| Error::limit("packed allocation does not fit usize"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_error| Error::limit("packed allocation failed"))?;
    bytes.resize(size, 0);
    Ok(bytes)
}
