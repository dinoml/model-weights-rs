//! Data-only access to named metadata embedded in weight checkpoints.

use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

use serde_json::Value;

use crate::limits::ResourceLimits;
use crate::{CancellationToken, Error, Result};

/// Reads one top-level metadata field from a local `PyTorch` checkpoint.
///
/// Only checkpoint metadata is interpreted. Tensor storage is neither decoded
/// nor materialized. See [`read_pytorch_field_from_reader`] for supported values
/// and limits.
///
/// # Errors
/// Returns an I/O, invalid-format, unsupported-capability, resource-limit, or
/// cancellation error.
pub fn read_pytorch_field(
    path: impl AsRef<Path>,
    field: &str,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<Option<Value>> {
    cancellation.check()?;
    let file = File::open(path)
        .map_err(|error| Error::io("opening PyTorch checkpoint metadata", error))?;
    read_pytorch_field_from_reader(file, field, limits, cancellation)
}

/// Reads one top-level metadata field from a seekable `PyTorch` checkpoint.
///
/// Accepts ZIP/ZIP64, legacy pickle streams, and historical tar archives. The
/// result is a JSON-compatible tree of nulls, booleans, integers within JSON's
/// signed or unsigned 64-bit range, finite floats, UTF-8 strings, lists or
/// tuples, and string-keyed dictionaries. A missing field returns `None`.
///
/// The reader must provide a stable view of the same archive for all reads.
/// ZIP archives without an archive comment require only ZIP headers, the
/// directory, and the pickle member. ZIP comment discovery and buffered legacy reads
/// may read adjacent bytes. Tensor storage is never decoded or materialized.
/// This function changes the reader's position.
///
/// `max_header_bytes` bounds pickle and archive metadata, and
/// `max_metadata_bytes` bounds the expanded result, including repeated
/// references and container overhead. The result has a maximum depth of 64.
///
/// # Errors
/// Returns an error for I/O failure, malformed metadata, duplicate keys, cycles,
/// unsupported values (including tensor references), exceeded limits, or
/// cancellation.
pub fn read_pytorch_field_from_reader<R: Read + Seek>(
    reader: R,
    field: &str,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<Option<Value>> {
    crate::pytorch::metadata::read(reader, field, limits, cancellation)
}
