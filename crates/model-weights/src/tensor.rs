//! Common tensor descriptors and byte views.

use std::fmt::{self, Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize};

use crate::{Error, Result};

/// A non-quantized scalar storage dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DType {
    /// Boolean stored in one byte.
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE binary16.
    F16,
    /// Brain floating point.
    Bf16,
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    /// Complex numbers with two binary32 components.
    C64,
    /// Eight-bit E5M2 floating point.
    F8E5M2,
    /// Eight-bit E4M3 floating point.
    F8E4M3,
    /// Eight-bit E8M0 floating point.
    F8E8M0,
    /// Eight-bit E4M3 FNUZ floating point.
    F8E4M3Fnuz,
    /// Eight-bit E5M2 FNUZ floating point.
    F8E5M2Fnuz,
}

impl DType {
    /// Returns the number of bits occupied by one scalar.
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::Bool
            | Self::U8
            | Self::I8
            | Self::F8E5M2
            | Self::F8E4M3
            | Self::F8E8M0
            | Self::F8E4M3Fnuz
            | Self::F8E5M2Fnuz => 8,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 16,
            Self::U32 | Self::I32 | Self::F32 => 32,
            Self::U64 | Self::I64 | Self::F64 | Self::C64 => 64,
        }
    }

    /// Computes the contiguous byte length for `shape`.
    ///
    /// An empty shape denotes one scalar; any zero dimension denotes an empty
    /// tensor.
    ///
    /// # Errors
    ///
    /// Returns a resource-limit error when element or byte arithmetic
    /// overflows `u64`.
    pub fn byte_len(self, shape: &[u64]) -> Result<u64> {
        let elements = shape.iter().try_fold(1_u64, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| Error::limit("tensor element count overflows u64"))
        })?;
        elements
            .checked_mul(u64::from(self.bits()) / 8)
            .ok_or_else(|| Error::limit("tensor byte length overflows u64"))
    }
}

/// An ordinal identifying a source file within one inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FileId(u32);

impl FileId {
    /// Creates a file identifier from its inventory ordinal.
    #[must_use]
    pub const fn from_ordinal(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Returns the inventory ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self.0
    }
}

/// A validated absolute byte span within an opened source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct SourceSpan {
    file: FileId,
    offset: u64,
    length: u64,
}

impl SourceSpan {
    /// Creates a byte span after validating its end offset.
    ///
    /// # Errors
    ///
    /// Returns a resource-limit error when `offset + length` overflows.
    pub fn new(file: FileId, offset: u64, length: u64) -> Result<Self> {
        offset
            .checked_add(length)
            .ok_or_else(|| Error::limit("source byte span end overflows u64"))?;
        Ok(Self {
            file,
            offset,
            length,
        })
    }

    /// Returns the source file identifier.
    #[must_use]
    pub const fn file(self) -> FileId {
        self.file
    }

    /// Returns the absolute file offset.
    #[must_use]
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Returns the span length.
    #[must_use]
    pub const fn len(self) -> u64 {
        self.length
    }

    /// Returns whether the span is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.length == 0
    }

    /// Returns the exclusive span end.
    ///
    /// Construction guarantees this addition cannot overflow.
    #[must_use]
    pub const fn end(self) -> u64 {
        self.offset + self.length
    }
}

#[derive(Deserialize)]
struct SourceSpanWire {
    file: FileId,
    offset: u64,
    length: u64,
}

impl TryFrom<SourceSpanWire> for SourceSpan {
    type Error = Error;

    fn try_from(wire: SourceSpanWire) -> Result<Self> {
        Self::new(wire.file, wire.offset, wire.length)
    }
}

impl<'de> Deserialize<'de> for SourceSpan {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SourceSpanWire::deserialize(deserializer)?;
        Self::try_from(wire).map_err(serde::de::Error::custom)
    }
}

pub(crate) trait ByteOwner: Debug + Send + Sync {
    fn bytes(&self) -> &[u8];
}

impl ByteOwner for Box<[u8]> {
    fn bytes(&self) -> &[u8] {
        self
    }
}

/// An owned, cheaply cloneable view over immutable bytes.
#[derive(Clone)]
pub struct ByteView {
    owner: Arc<dyn ByteOwner>,
    range: Range<usize>,
}

impl ByteView {
    pub(crate) fn from_boxed(bytes: Box<[u8]>) -> Self {
        let length = bytes.len();
        Self {
            owner: Arc::new(bytes),
            range: 0..length,
        }
    }

    pub(crate) fn from_owner(owner: Arc<dyn ByteOwner>, range: Range<usize>) -> Result<Self> {
        if range.start > range.end || range.end > owner.bytes().len() {
            return Err(Error::invalid("byte view range lies outside its owner"));
        }
        Ok(Self { owner, range })
    }

    /// Returns the immutable byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.owner.bytes()[self.range.clone()]
    }

    /// Returns the number of visible bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns whether the view contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Returns a checked subview without copying bytes.
    ///
    /// # Errors
    ///
    /// Returns an invalid-format error when `range` lies outside this view.
    pub fn slice(&self, range: Range<usize>) -> Result<Self> {
        if range.start > range.end || range.end > self.len() {
            return Err(Error::invalid("byte subview range lies outside its parent"));
        }
        let start = self
            .range
            .start
            .checked_add(range.start)
            .ok_or_else(|| Error::limit("byte subview start overflows usize"))?;
        let end = self
            .range
            .start
            .checked_add(range.end)
            .ok_or_else(|| Error::limit("byte subview end overflows usize"))?;
        Self::from_owner(Arc::clone(&self.owner), start..end)
    }
}

impl Debug for ByteView {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ByteView")
            .field("length", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_span_deserialization_replays_checked_end_validation() -> Result<()> {
        let span = SourceSpan::new(FileId::from_ordinal(7), 11, 13)?;
        let encoded = serde_json::to_vec(&span).map_err(|error| {
            Error::with_source(
                crate::ErrorCategory::InvalidFormat,
                "serialize source span test value",
                error,
            )
        })?;
        let decoded: SourceSpan = serde_json::from_slice(&encoded).map_err(|error| {
            Error::with_source(
                crate::ErrorCategory::InvalidFormat,
                "deserialize source span test value",
                error,
            )
        })?;

        assert_eq!(decoded, span);
        let Err(_) = serde_json::from_str::<SourceSpan>(
            r#"{"file":0,"offset":18446744073709551615,"length":1}"#,
        ) else {
            return Err(Error::invalid(
                "overflowing source span unexpectedly deserialized",
            ));
        };
        Ok(())
    }
}
