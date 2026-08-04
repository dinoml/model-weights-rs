//! Bounded GGUF metadata and tensor-inventory parsing.

use std::collections::BTreeMap;
use std::fs::File;

use crate::identity::{ImplementationId, StableName};
use crate::inventory::TensorRecord;
use crate::limits::ResourceLimits;
use crate::quantization::{BlockPacking, Packing, PackingOrder, QuantizedStorage, Storage};
use crate::tensor::{DType, FileId, SourceSpan};
use crate::{CancellationToken, Error, ErrorCategory, Result};

const MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const MIN_VERSION: u32 = 2;
const MAX_VERSION: u32 = 3;
const ENCODING_VERSION: u32 = 1;
const READER_BUFFER_BYTES: usize = 64 * 1024;
const MAX_METADATA_NESTING_DEPTH: usize = 64;

/// One typed GGUF metadata array.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GgufArray {
    /// Nested typed arrays.
    Array(Box<[GgufArray]>),
    /// Unsigned 8-bit values.
    U8(Box<[u8]>),
    /// Signed 8-bit values.
    I8(Box<[i8]>),
    /// Unsigned 16-bit values.
    U16(Box<[u16]>),
    /// Signed 16-bit values.
    I16(Box<[i16]>),
    /// Unsigned 32-bit values.
    U32(Box<[u32]>),
    /// Signed 32-bit values.
    I32(Box<[i32]>),
    /// IEEE binary32 values.
    F32(Box<[f32]>),
    /// Boolean values.
    Bool(Box<[bool]>),
    /// UTF-8 strings.
    String(Box<[Box<str>]>),
    /// Unsigned 64-bit values.
    U64(Box<[u64]>),
    /// Signed 64-bit values.
    I64(Box<[i64]>),
    /// IEEE binary64 values.
    F64(Box<[f64]>),
}

/// One typed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GgufValue {
    /// An unsigned 8-bit integer.
    U8(u8),
    /// A signed 8-bit integer.
    I8(i8),
    /// An unsigned 16-bit integer.
    U16(u16),
    /// A signed 16-bit integer.
    I16(i16),
    /// An unsigned 32-bit integer.
    U32(u32),
    /// A signed 32-bit integer.
    I32(i32),
    /// An IEEE binary32 value.
    F32(f32),
    /// A Boolean value.
    Bool(bool),
    /// A UTF-8 string.
    String(Box<str>),
    /// A typed array.
    Array(GgufArray),
    /// An unsigned 64-bit integer.
    U64(u64),
    /// A signed 64-bit integer.
    I64(i64),
    /// An IEEE binary64 value.
    F64(f64),
}

impl GgufValue {
    /// Returns this value as unsigned integer metadata when lossless.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(u64::from(*value)),
            Self::U16(value) => Some(u64::from(*value)),
            Self::U32(value) => Some(u64::from(*value)),
            Self::U64(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns this value as UTF-8 metadata.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

/// Parsed metadata from one GGUF container.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufMetadata {
    version: u32,
    alignment: u64,
    entries: BTreeMap<Box<str>, GgufValue>,
}

impl GgufMetadata {
    /// Returns the GGUF container version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the validated tensor-data alignment.
    #[must_use]
    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Returns metadata entries in deterministic key order.
    #[must_use]
    pub const fn entries(&self) -> &BTreeMap<Box<str>, GgufValue> {
        &self.entries
    }

    /// Finds one exact metadata key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.entries.get(key)
    }
}

/// One currently defined GGML tensor storage type accepted by GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum GgmlType {
    /// IEEE binary32 scalar storage.
    F32,
    /// IEEE binary16 scalar storage.
    F16,
    /// GGML `Q4_0` blocks.
    Q4_0,
    /// GGML `Q4_1` blocks.
    Q4_1,
    /// GGML `Q5_0` blocks.
    Q5_0,
    /// GGML `Q5_1` blocks.
    Q5_1,
    /// GGML `Q8_0` blocks.
    Q8_0,
    /// GGML `Q8_1` blocks.
    Q8_1,
    /// GGML `Q2_K` blocks.
    Q2K,
    /// GGML `Q3_K` blocks.
    Q3K,
    /// GGML `Q4_K` blocks.
    Q4K,
    /// GGML `Q5_K` blocks.
    Q5K,
    /// GGML `Q6_K` blocks.
    Q6K,
    /// GGML `Q8_K` blocks.
    Q8K,
    /// GGML `IQ2_XXS` blocks.
    Iq2Xxs,
    /// GGML `IQ2_XS` blocks.
    Iq2Xs,
    /// GGML `IQ3_XXS` blocks.
    Iq3Xxs,
    /// GGML `IQ1_S` blocks.
    Iq1S,
    /// GGML `IQ4_NL` blocks.
    Iq4Nl,
    /// GGML `IQ3_S` blocks.
    Iq3S,
    /// GGML `IQ2_S` blocks.
    Iq2S,
    /// GGML `IQ4_XS` blocks.
    Iq4Xs,
    /// Signed 8-bit scalar storage.
    I8,
    /// Signed 16-bit scalar storage.
    I16,
    /// Signed 32-bit scalar storage.
    I32,
    /// Signed 64-bit scalar storage.
    I64,
    /// IEEE binary64 scalar storage.
    F64,
    /// GGML `IQ1_M` blocks.
    Iq1M,
    /// Brain floating-point scalar storage.
    Bf16,
    /// GGML `TQ1_0` blocks.
    Tq1_0,
    /// GGML `TQ2_0` blocks.
    Tq2_0,
    /// GGML `MXFP4` blocks.
    Mxfp4,
    /// GGML `NVFP4` blocks.
    Nvfp4,
    /// GGML `Q1_0` blocks.
    Q1_0,
    /// GGML `Q2_0` blocks.
    Q2_0,
}

impl GgmlType {
    /// Every currently defined, file-valid GGML storage type in code order.
    pub const ALL: [Self; 35] = [
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q8K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::Iq1M,
        Self::Bf16,
        Self::Tq1_0,
        Self::Tq2_0,
        Self::Mxfp4,
        Self::Nvfp4,
        Self::Q1_0,
        Self::Q2_0,
    ];

    /// Parses one live GGML type code from a GGUF tensor descriptor.
    ///
    /// # Errors
    ///
    /// Returns an unsupported error for removed, reserved, or unknown codes.
    pub fn from_code(code: u32) -> Result<Self> {
        match code {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::Iq2Xxs),
            17 => Ok(Self::Iq2Xs),
            18 => Ok(Self::Iq3Xxs),
            19 => Ok(Self::Iq1S),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::Iq1M),
            30 => Ok(Self::Bf16),
            34 => Ok(Self::Tq1_0),
            35 => Ok(Self::Tq2_0),
            39 => Ok(Self::Mxfp4),
            40 => Ok(Self::Nvfp4),
            41 => Ok(Self::Q1_0),
            42 => Ok(Self::Q2_0),
            _ => Err(Error::unsupported(
                "GGUF tensor uses a removed, reserved, or unknown GGML storage type",
            )),
        }
    }

    /// Returns the stable GGML type code stored in GGUF.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::Iq1M => 29,
            Self::Bf16 => 30,
            Self::Tq1_0 => 34,
            Self::Tq2_0 => 35,
            Self::Mxfp4 => 39,
            Self::Nvfp4 => 40,
            Self::Q1_0 => 41,
            Self::Q2_0 => 42,
        }
    }

    /// Returns the canonical lowercase GGML storage name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Q8K => "q8_k",
            Self::Iq2Xxs => "iq2_xxs",
            Self::Iq2Xs => "iq2_xs",
            Self::Iq3Xxs => "iq3_xxs",
            Self::Iq1S => "iq1_s",
            Self::Iq4Nl => "iq4_nl",
            Self::Iq3S => "iq3_s",
            Self::Iq2S => "iq2_s",
            Self::Iq4Xs => "iq4_xs",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::Iq1M => "iq1_m",
            Self::Bf16 => "bf16",
            Self::Tq1_0 => "tq1_0",
            Self::Tq2_0 => "tq2_0",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Q1_0 => "q1_0",
            Self::Q2_0 => "q2_0",
        }
    }

    /// Returns the logical values represented by one stored block.
    #[must_use]
    pub const fn values_per_block(self) -> u32 {
        match self {
            Self::F32
            | Self::F16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64
            | Self::Bf16 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::Iq4Nl
            | Self::Mxfp4 => 32,
            Self::Nvfp4 | Self::Q2_0 => 64,
            Self::Q1_0 => 128,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M
            | Self::Tq1_0
            | Self::Tq2_0 => 256,
        }
    }

    /// Returns the bytes occupied by one stored block.
    #[must_use]
    pub const fn bytes_per_block(self) -> u32 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::I16 | Self::Bf16 => 2,
            Self::I8 => 1,
            Self::I64 | Self::F64 => 8,
            Self::Q4_0 | Self::Iq4Nl | Self::Q1_0 | Self::Q2_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 | Self::Nvfp4 => 36,
            Self::Q2K => 84,
            Self::Q3K | Self::Iq3S => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::Iq2Xxs | Self::Tq2_0 => 66,
            Self::Iq2Xs => 74,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq2S => 82,
            Self::Iq4Xs => 136,
            Self::Iq1M => 56,
            Self::Tq1_0 => 54,
            Self::Mxfp4 => 17,
        }
    }

    /// Returns the ordinary scalar dtype, if this is not a packed encoding.
    #[must_use]
    pub const fn dtype(self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::I8 => Some(DType::I8),
            Self::I16 => Some(DType::I16),
            Self::I32 => Some(DType::I32),
            Self::I64 => Some(DType::I64),
            Self::F64 => Some(DType::F64),
            Self::Bf16 => Some(DType::Bf16),
            _ => None,
        }
    }

    /// Returns the normalized packed-storage identity, if encoded.
    ///
    /// The version identifies this crate's byte-level GGML ABI descriptor,
    /// not the algorithm recorded in `general.quantization_version`.
    ///
    /// # Panics
    ///
    /// Panics only if a built-in portable identifier is changed to an invalid
    /// [`StableName`], which is a crate programming error.
    #[must_use]
    pub fn encoding(self) -> Option<ImplementationId> {
        self.dtype().is_none().then(|| {
            ImplementationId::new(
                StableName::parse("gguf")
                    .expect("the built-in GGUF provider name must remain valid"),
                StableName::parse(self.name())
                    .expect("the built-in GGML operation name must remain valid"),
                ENCODING_VERSION,
            )
        })
    }
}

#[derive(Debug)]
pub(crate) struct ParsedGguf {
    pub(crate) metadata: GgufMetadata,
    pub(crate) tensors: Vec<TensorRecord>,
}

pub(crate) fn parse(
    file: &File,
    file_id: FileId,
    file_size: u64,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<ParsedGguf> {
    let mut reader = Reader::new(file, file_size, limits.max_header_bytes(), cancellation);
    if reader.read::<4>()? != MAGIC {
        return Err(Error::invalid("GGUF source has invalid magic"));
    }
    let version = reader.u32()?;
    if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
        return Err(Error::unsupported(
            "GGUF container version is not supported",
        ));
    }
    let tensor_count = bounded_count(reader.u64()?, limits.max_tensors(), "GGUF tensor count")?;
    let entries = read_metadata(&mut reader, limits, cancellation)?;
    let alignment = metadata_alignment(&entries)?;
    let raw_tensors = read_tensor_infos(&mut reader, tensor_count, limits, cancellation)?;
    let data_base = align_up(reader.offset(), alignment)?;
    if data_base > file_size {
        return Err(Error::invalid(
            "GGUF tensor data begins beyond the source file",
        ));
    }
    if data_base > limits.max_header_bytes() {
        return Err(Error::limit(
            "GGUF header length violates the configured limit",
        ));
    }
    let tensors = build_tensor_records(
        raw_tensors,
        file_id,
        file_size,
        data_base,
        alignment,
        cancellation,
    )?;

    Ok(ParsedGguf {
        metadata: GgufMetadata {
            version,
            alignment,
            entries,
        },
        tensors,
    })
}

fn read_metadata(
    reader: &mut Reader<'_>,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<Box<str>, GgufValue>> {
    let metadata_limit = usize::try_from(limits.max_header_bytes()).unwrap_or(usize::MAX);
    let minimum_metadata_entry_bytes = 12_usize;
    let metadata_count_limit = metadata_limit / minimum_metadata_entry_bytes;
    let metadata_count = bounded_count(
        reader.u64()?,
        metadata_count_limit,
        "GGUF metadata entry count",
    )?;

    let metadata_start = reader.offset();
    let mut entries = BTreeMap::new();
    for _ in 0..metadata_count {
        cancellation.check()?;
        let key = reader.string(limits.max_name_bytes(), "GGUF metadata key")?;
        let value_type = ValueType::from_raw(reader.u32()?)?;
        let value = read_value(reader, value_type, metadata_limit, 0)?;
        let metadata_bytes = usize::try_from(reader.offset() - metadata_start)
            .map_err(|_error| Error::limit("GGUF metadata length does not fit usize"))?;
        if metadata_bytes > metadata_limit {
            return Err(Error::limit(
                "GGUF metadata exceeds the configured byte limit",
            ));
        }
        if entries.insert(key, value).is_some() {
            return Err(Error::invalid("GGUF contains a duplicate metadata key"));
        }
    }
    Ok(entries)
}

fn metadata_alignment(entries: &BTreeMap<Box<str>, GgufValue>) -> Result<u64> {
    let alignment = entries
        .get("general.alignment")
        .and_then(GgufValue::as_u64)
        .unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::invalid(
            "GGUF general.alignment must be a nonzero power of two",
        ));
    }
    Ok(alignment)
}

fn read_tensor_infos(
    reader: &mut Reader<'_>,
    tensor_count: usize,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<Vec<RawTensor>> {
    let mut raw_tensors = Vec::new();
    raw_tensors
        .try_reserve_exact(tensor_count)
        .map_err(|_error| Error::limit("could not allocate GGUF tensor inventory"))?;
    for _ in 0..tensor_count {
        cancellation.check()?;
        let name = reader.string(limits.max_name_bytes(), "GGUF tensor name")?;
        let rank = bounded_count(reader.u32()?.into(), limits.max_rank(), "GGUF tensor rank")?;
        if rank == 0 {
            return Err(Error::invalid("GGUF tensors must have nonzero rank"));
        }
        let mut dimensions = Vec::new();
        dimensions
            .try_reserve_exact(rank)
            .map_err(|_error| Error::limit("could not allocate GGUF tensor dimensions"))?;
        for _ in 0..rank {
            dimensions.push(reader.u64()?);
        }
        let ggml_type = reader.u32()?;
        let relative_offset = reader.u64()?;
        raw_tensors.push(RawTensor {
            name,
            dimensions,
            ggml_type,
            relative_offset,
        });
    }
    Ok(raw_tensors)
}

fn build_tensor_records(
    raw_tensors: Vec<RawTensor>,
    file_id: FileId,
    file_size: u64,
    data_base: u64,
    alignment: u64,
    cancellation: &CancellationToken,
) -> Result<Vec<TensorRecord>> {
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(raw_tensors.len())
        .map_err(|_error| Error::limit("could not allocate GGUF tensor records"))?;
    let mut occupied = Vec::new();
    occupied
        .try_reserve_exact(raw_tensors.len())
        .map_err(|_error| Error::limit("could not allocate GGUF tensor spans"))?;
    for raw in raw_tensors {
        cancellation.check()?;
        if raw.relative_offset % alignment != 0 {
            return Err(Error::invalid(
                "GGUF tensor offset does not satisfy general.alignment",
            ));
        }
        let absolute_offset = data_base
            .checked_add(raw.relative_offset)
            .ok_or_else(|| Error::limit("GGUF tensor offset overflows u64"))?;
        let mut shape = raw.dimensions;
        shape.reverse();
        let storage = storage(file_id, absolute_offset, &shape, raw.ggml_type)?;
        let span = storage.span();
        if span.end() > file_size {
            return Err(Error::invalid(
                "GGUF tensor payload extends beyond the source file",
            ));
        }
        occupied.push((span.offset(), span.end()));
        tensors.push(TensorRecord::new(
            raw.name,
            shape.into_boxed_slice(),
            storage,
        ));
    }
    occupied.sort_unstable();
    if occupied.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(Error::invalid("GGUF tensor payload spans overlap"));
    }
    Ok(tensors)
}

#[derive(Debug)]
struct RawTensor {
    name: Box<str>,
    dimensions: Vec<u64>,
    ggml_type: u32,
    relative_offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_raw(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(Error::invalid("GGUF metadata has an unknown value type")),
        }
    }
}

fn read_value(
    reader: &mut Reader<'_>,
    value_type: ValueType,
    metadata_limit: usize,
    nesting_depth: usize,
) -> Result<GgufValue> {
    match value_type {
        ValueType::U8 => Ok(GgufValue::U8(reader.u8()?)),
        ValueType::I8 => Ok(GgufValue::I8(reader.i8()?)),
        ValueType::U16 => Ok(GgufValue::U16(reader.u16()?)),
        ValueType::I16 => Ok(GgufValue::I16(reader.i16()?)),
        ValueType::U32 => Ok(GgufValue::U32(reader.u32()?)),
        ValueType::I32 => Ok(GgufValue::I32(reader.i32()?)),
        ValueType::F32 => Ok(GgufValue::F32(reader.f32()?)),
        ValueType::Bool => Ok(GgufValue::Bool(reader.boolean()?)),
        ValueType::String => Ok(GgufValue::String(
            reader.string(metadata_limit, "GGUF metadata string")?,
        )),
        ValueType::Array => Ok(GgufValue::Array(read_array(
            reader,
            metadata_limit,
            nesting_depth,
        )?)),
        ValueType::U64 => Ok(GgufValue::U64(reader.u64()?)),
        ValueType::I64 => Ok(GgufValue::I64(reader.i64()?)),
        ValueType::F64 => Ok(GgufValue::F64(reader.f64()?)),
    }
}

fn read_array(
    reader: &mut Reader<'_>,
    metadata_limit: usize,
    nesting_depth: usize,
) -> Result<GgufArray> {
    if nesting_depth >= MAX_METADATA_NESTING_DEPTH {
        return Err(Error::limit(
            "GGUF metadata array nesting exceeds the configured limit",
        ));
    }
    let element_type = ValueType::from_raw(reader.u32()?)?;
    let element_bytes = match element_type {
        ValueType::U8 | ValueType::I8 | ValueType::Bool => 1,
        ValueType::U16 | ValueType::I16 => 2,
        ValueType::U32 | ValueType::I32 | ValueType::F32 => 4,
        ValueType::String | ValueType::U64 | ValueType::I64 | ValueType::F64 => 8,
        ValueType::Array => 12,
    };
    let max_elements = metadata_limit / element_bytes;
    let count = bounded_count(reader.u64()?, max_elements, "GGUF metadata array length")?;
    match element_type {
        ValueType::U8 => read_elements(reader, count, Reader::u8).map(GgufArray::U8),
        ValueType::I8 => read_elements(reader, count, Reader::i8).map(GgufArray::I8),
        ValueType::U16 => read_elements(reader, count, Reader::u16).map(GgufArray::U16),
        ValueType::I16 => read_elements(reader, count, Reader::i16).map(GgufArray::I16),
        ValueType::U32 => read_elements(reader, count, Reader::u32).map(GgufArray::U32),
        ValueType::I32 => read_elements(reader, count, Reader::i32).map(GgufArray::I32),
        ValueType::F32 => read_elements(reader, count, Reader::f32).map(GgufArray::F32),
        ValueType::Bool => read_elements(reader, count, Reader::boolean).map(GgufArray::Bool),
        ValueType::String => {
            let mut values = reserve_vec(count, "GGUF metadata string array")?;
            let max_string = metadata_limit;
            for _ in 0..count {
                values.push(reader.string(max_string, "GGUF metadata array string")?);
            }
            Ok(GgufArray::String(values.into_boxed_slice()))
        }
        ValueType::U64 => read_elements(reader, count, Reader::u64).map(GgufArray::U64),
        ValueType::I64 => read_elements(reader, count, Reader::i64).map(GgufArray::I64),
        ValueType::F64 => read_elements(reader, count, Reader::f64).map(GgufArray::F64),
        ValueType::Array => {
            let mut values = reserve_vec(count, "GGUF nested metadata array")?;
            for _ in 0..count {
                values.push(read_array(reader, metadata_limit, nesting_depth + 1)?);
            }
            Ok(GgufArray::Array(values.into_boxed_slice()))
        }
    }
}

fn read_elements<'file, T, F>(
    reader: &mut Reader<'file>,
    count: usize,
    mut read: F,
) -> Result<Box<[T]>>
where
    F: FnMut(&mut Reader<'file>) -> Result<T>,
{
    let mut values = reserve_vec(count, "GGUF metadata array")?;
    for _ in 0..count {
        values.push(read(reader)?);
    }
    Ok(values.into_boxed_slice())
}

fn reserve_vec<T>(count: usize, description: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_error| Error::limit(format!("could not allocate {description}")))?;
    Ok(values)
}

fn storage(file_id: FileId, offset: u64, shape: &[u64], ggml_type_code: u32) -> Result<Storage> {
    let ggml_type = GgmlType::from_code(ggml_type_code)?;
    if let Some(dtype) = ggml_type.dtype() {
        let length = dtype.byte_len(shape)?;
        return Ok(Storage::Plain {
            dtype,
            span: SourceSpan::new(file_id, offset, length)?,
        });
    }
    let axis = shape
        .len()
        .checked_sub(1)
        .ok_or_else(|| Error::invalid("packed GGUF tensor has no block axis"))?;
    let values_per_block = ggml_type.values_per_block();
    let bytes_per_block = ggml_type.bytes_per_block();
    if shape[axis] % u64::from(values_per_block) != 0 {
        return Err(Error::invalid(
            "packed GGUF tensor row length is not a whole storage block",
        ));
    }
    let elements = shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| Error::limit("GGUF tensor element count overflows u64"))
    })?;
    let length = elements
        .checked_div(u64::from(values_per_block))
        .and_then(|blocks| blocks.checked_mul(u64::from(bytes_per_block)))
        .ok_or_else(|| Error::limit("packed GGUF tensor byte length overflows u64"))?;
    let span = SourceSpan::new(file_id, offset, length)?;
    let axis = u32::try_from(axis).map_err(|_error| Error::limit("GGUF block axis exceeds u32"))?;
    let packing = Packing::Blocks(BlockPacking::new(
        values_per_block,
        bytes_per_block,
        PackingOrder::Axis(axis),
    )?);
    Ok(Storage::Quantized(QuantizedStorage::new(
        ggml_type
            .encoding()
            .ok_or_else(|| Error::invalid("packed GGUF tensor has no encoding identity"))?,
        shape,
        span,
        packing,
    )?))
}

fn bounded_count(value: u64, maximum: usize, description: &str) -> Result<usize> {
    let value = usize::try_from(value)
        .map_err(|_error| Error::limit(format!("{description} does not fit usize")))?;
    if value > maximum {
        return Err(Error::limit(format!(
            "{description} exceeds the configured limit"
        )));
    }
    Ok(value)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| Error::limit("GGUF aligned data offset overflows u64"))
}

struct Reader<'a> {
    file: &'a File,
    file_size: u64,
    limit: u64,
    offset: u64,
    buffer: Box<[u8]>,
    buffer_start: u64,
    buffer_len: usize,
    cancellation: &'a CancellationToken,
}

impl<'a> Reader<'a> {
    fn new(
        file: &'a File,
        file_size: u64,
        limit: u64,
        cancellation: &'a CancellationToken,
    ) -> Self {
        Self {
            file,
            file_size,
            limit,
            offset: 0,
            buffer: vec![0; READER_BUFFER_BYTES].into_boxed_slice(),
            buffer_start: 0,
            buffer_len: 0,
            cancellation,
        }
    }

    const fn offset(&self) -> u64 {
        self.offset
    }

    fn read<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.cancellation.check()?;
        let length =
            u64::try_from(N).map_err(|_error| Error::limit("GGUF read length does not fit u64"))?;
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| Error::limit("GGUF header offset overflows u64"))?;
        if end > self.limit {
            return Err(Error::limit(
                "GGUF header length violates the configured limit",
            ));
        }
        if end > self.file_size {
            return Err(Error::invalid("GGUF header is truncated"));
        }
        let mut bytes = [0_u8; N];
        self.read_into(&mut bytes)?;
        Ok(bytes)
    }

    fn bytes(&mut self, length: usize) -> Result<Box<[u8]>> {
        let length_u64 = u64::try_from(length)
            .map_err(|_error| Error::limit("GGUF byte string length does not fit u64"))?;
        let end = self
            .offset
            .checked_add(length_u64)
            .ok_or_else(|| Error::limit("GGUF byte string end overflows u64"))?;
        if end > self.limit {
            return Err(Error::limit(
                "GGUF header length violates the configured limit",
            ));
        }
        if end > self.file_size {
            return Err(Error::invalid("GGUF header is truncated"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_error| Error::limit("could not allocate GGUF string bytes"))?;
        bytes.resize(length, 0);
        self.read_into(&mut bytes)?;
        Ok(bytes.into_boxed_slice())
    }

    fn read_into(&mut self, mut output: &mut [u8]) -> Result<()> {
        while !output.is_empty() {
            self.cancellation.check()?;
            let buffer_end =
                self.buffer_start
                    .checked_add(u64::try_from(self.buffer_len).map_err(|_error| {
                        Error::limit("GGUF read buffer length does not fit u64")
                    })?)
                    .ok_or_else(|| Error::limit("GGUF read buffer end overflows u64"))?;
            if self.offset < self.buffer_start || self.offset >= buffer_end {
                self.fill_buffer()?;
            }
            let relative = usize::try_from(self.offset - self.buffer_start)
                .map_err(|_error| Error::limit("GGUF read buffer offset does not fit usize"))?;
            let available = self.buffer_len - relative;
            let amount = available.min(output.len());
            output[..amount].copy_from_slice(&self.buffer[relative..relative + amount]);
            self.offset =
                self.offset
                    .checked_add(u64::try_from(amount).map_err(|_error| {
                        Error::limit("GGUF buffered read length does not fit u64")
                    })?)
                    .ok_or_else(|| Error::limit("GGUF buffered read offset overflows u64"))?;
            output = &mut output[amount..];
        }
        Ok(())
    }

    fn fill_buffer(&mut self) -> Result<()> {
        let remaining = self.file_size.min(self.limit).saturating_sub(self.offset);
        if remaining == 0 {
            return Err(Error::invalid("GGUF header is truncated"));
        }
        let amount = usize::try_from(remaining.min(READER_BUFFER_BYTES as u64))
            .map_err(|_error| Error::limit("GGUF buffered read length does not fit usize"))?;
        super::checkpoint::read_exact_at(self.file, self.offset, &mut self.buffer[..amount])?;
        self.buffer_start = self.offset;
        self.buffer_len = amount;
        Ok(())
    }

    fn string(&mut self, maximum: usize, description: &str) -> Result<Box<str>> {
        let length = bounded_count(self.u64()?, maximum, description)?;
        let bytes = self.bytes(length)?;
        String::from_utf8(bytes.into_vec())
            .map(String::into_boxed_str)
            .map_err(|source| {
                Error::with_source(
                    ErrorCategory::InvalidFormat,
                    format!("{description} is not valid UTF-8"),
                    source,
                )
            })
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.read::<1>()?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(i8::from_le_bytes(self.read()?))
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read()?))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read()?))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read()?))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read()?))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read()?))
    }

    fn boolean(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::invalid(
                "GGUF Boolean metadata is neither zero nor one",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use tempfile::NamedTempFile;

    const TEST_ALIGNMENT: usize = 32;

    use super::*;
    use crate::TensorData;

    #[derive(Clone, Copy)]
    struct CatalogEntry {
        code: u32,
        name: &'static str,
        values: u32,
        bytes: u32,
        dtype: Option<DType>,
    }

    const CATALOG: [CatalogEntry; 35] = [
        CatalogEntry {
            code: 0,
            name: "f32",
            values: 1,
            bytes: 4,
            dtype: Some(DType::F32),
        },
        CatalogEntry {
            code: 1,
            name: "f16",
            values: 1,
            bytes: 2,
            dtype: Some(DType::F16),
        },
        CatalogEntry {
            code: 2,
            name: "q4_0",
            values: 32,
            bytes: 18,
            dtype: None,
        },
        CatalogEntry {
            code: 3,
            name: "q4_1",
            values: 32,
            bytes: 20,
            dtype: None,
        },
        CatalogEntry {
            code: 6,
            name: "q5_0",
            values: 32,
            bytes: 22,
            dtype: None,
        },
        CatalogEntry {
            code: 7,
            name: "q5_1",
            values: 32,
            bytes: 24,
            dtype: None,
        },
        CatalogEntry {
            code: 8,
            name: "q8_0",
            values: 32,
            bytes: 34,
            dtype: None,
        },
        CatalogEntry {
            code: 9,
            name: "q8_1",
            values: 32,
            bytes: 36,
            dtype: None,
        },
        CatalogEntry {
            code: 10,
            name: "q2_k",
            values: 256,
            bytes: 84,
            dtype: None,
        },
        CatalogEntry {
            code: 11,
            name: "q3_k",
            values: 256,
            bytes: 110,
            dtype: None,
        },
        CatalogEntry {
            code: 12,
            name: "q4_k",
            values: 256,
            bytes: 144,
            dtype: None,
        },
        CatalogEntry {
            code: 13,
            name: "q5_k",
            values: 256,
            bytes: 176,
            dtype: None,
        },
        CatalogEntry {
            code: 14,
            name: "q6_k",
            values: 256,
            bytes: 210,
            dtype: None,
        },
        CatalogEntry {
            code: 15,
            name: "q8_k",
            values: 256,
            bytes: 292,
            dtype: None,
        },
        CatalogEntry {
            code: 16,
            name: "iq2_xxs",
            values: 256,
            bytes: 66,
            dtype: None,
        },
        CatalogEntry {
            code: 17,
            name: "iq2_xs",
            values: 256,
            bytes: 74,
            dtype: None,
        },
        CatalogEntry {
            code: 18,
            name: "iq3_xxs",
            values: 256,
            bytes: 98,
            dtype: None,
        },
        CatalogEntry {
            code: 19,
            name: "iq1_s",
            values: 256,
            bytes: 50,
            dtype: None,
        },
        CatalogEntry {
            code: 20,
            name: "iq4_nl",
            values: 32,
            bytes: 18,
            dtype: None,
        },
        CatalogEntry {
            code: 21,
            name: "iq3_s",
            values: 256,
            bytes: 110,
            dtype: None,
        },
        CatalogEntry {
            code: 22,
            name: "iq2_s",
            values: 256,
            bytes: 82,
            dtype: None,
        },
        CatalogEntry {
            code: 23,
            name: "iq4_xs",
            values: 256,
            bytes: 136,
            dtype: None,
        },
        CatalogEntry {
            code: 24,
            name: "i8",
            values: 1,
            bytes: 1,
            dtype: Some(DType::I8),
        },
        CatalogEntry {
            code: 25,
            name: "i16",
            values: 1,
            bytes: 2,
            dtype: Some(DType::I16),
        },
        CatalogEntry {
            code: 26,
            name: "i32",
            values: 1,
            bytes: 4,
            dtype: Some(DType::I32),
        },
        CatalogEntry {
            code: 27,
            name: "i64",
            values: 1,
            bytes: 8,
            dtype: Some(DType::I64),
        },
        CatalogEntry {
            code: 28,
            name: "f64",
            values: 1,
            bytes: 8,
            dtype: Some(DType::F64),
        },
        CatalogEntry {
            code: 29,
            name: "iq1_m",
            values: 256,
            bytes: 56,
            dtype: None,
        },
        CatalogEntry {
            code: 30,
            name: "bf16",
            values: 1,
            bytes: 2,
            dtype: Some(DType::Bf16),
        },
        CatalogEntry {
            code: 34,
            name: "tq1_0",
            values: 256,
            bytes: 54,
            dtype: None,
        },
        CatalogEntry {
            code: 35,
            name: "tq2_0",
            values: 256,
            bytes: 66,
            dtype: None,
        },
        CatalogEntry {
            code: 39,
            name: "mxfp4",
            values: 32,
            bytes: 17,
            dtype: None,
        },
        CatalogEntry {
            code: 40,
            name: "nvfp4",
            values: 64,
            bytes: 36,
            dtype: None,
        },
        CatalogEntry {
            code: 41,
            name: "q1_0",
            values: 128,
            bytes: 18,
            dtype: None,
        },
        CatalogEntry {
            code: 42,
            name: "q2_0",
            values: 64,
            bytes: 18,
            dtype: None,
        },
    ];

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn fixture() -> std::io::Result<NamedTempFile> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&4_u64.to_le_bytes());

        write_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        write_string(&mut bytes, "qwen3");
        write_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
        write_string(&mut bytes, "qwen3.block_count");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&28_u32.to_le_bytes());
        write_string(&mut bytes, "tokenizer.ggml.token_type");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&5_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&1_i32.to_le_bytes());
        bytes.extend_from_slice(&3_i32.to_le_bytes());

        write_string(&mut bytes, "blk.0.attn_q.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());

        write_string(&mut bytes, "blk.0.attn_norm.weight");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&96_u64.to_le_bytes());

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0x5a; 68]);
        bytes.extend_from_slice(&[0; 28]);
        bytes.extend_from_slice(&[0, 0, 0x80, 0x3f, 0, 0, 0, 0x40]);

        let mut file = NamedTempFile::new()?;
        file.write_all(&bytes)?;
        Ok(file)
    }

    fn all_types_fixture() -> Result<NamedTempFile> {
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3_u32.to_le_bytes());
        header.extend_from_slice(&(CATALOG.len() as u64).to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        let mut relative_offset = 0_u64;
        for entry in CATALOG {
            write_string(&mut header, &format!("tensor.{}", entry.code));
            header.extend_from_slice(&2_u32.to_le_bytes());
            header.extend_from_slice(&u64::from(entry.values).to_le_bytes());
            header.extend_from_slice(&2_u64.to_le_bytes());
            header.extend_from_slice(&entry.code.to_le_bytes());
            header.extend_from_slice(&relative_offset.to_le_bytes());
            let length = 2_u64 * u64::from(entry.bytes);
            relative_offset = align_up(relative_offset + length, DEFAULT_ALIGNMENT)?;
        }
        while header.len() % TEST_ALIGNMENT != 0 {
            header.push(0);
        }
        let mut payload = Vec::new();
        for entry in CATALOG {
            while payload.len() % TEST_ALIGNMENT != 0 {
                payload.push(0);
            }
            let marker = u8::try_from(entry.code)
                .map_err(|_error| Error::limit("test GGML code does not fit u8"))?;
            payload.extend(std::iter::repeat_n(marker, 2 * entry.bytes as usize));
        }
        header.extend(payload);
        let mut file = NamedTempFile::new()
            .map_err(|source| Error::io("create all-types GGUF fixture", source))?;
        file.write_all(&header)
            .map_err(|source| Error::io("write all-types GGUF fixture", source))?;
        Ok(file)
    }

    fn nested_array_fixture(depth: usize) -> Result<NamedTempFile> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        write_string(&mut bytes, "test.nested");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        for _ in 0..depth {
            bytes.extend_from_slice(&9_u32.to_le_bytes());
            bytes.extend_from_slice(&1_u64.to_le_bytes());
        }
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&7_u32.to_le_bytes());
        bytes.extend_from_slice(&11_u32.to_le_bytes());
        while bytes.len() % TEST_ALIGNMENT != 0 {
            bytes.push(0);
        }
        let mut file = NamedTempFile::new()
            .map_err(|source| Error::io("create nested-array GGUF fixture", source))?;
        file.write_all(&bytes)
            .map_err(|source| Error::io("write nested-array GGUF fixture", source))?;
        Ok(file)
    }

    #[test]
    fn parses_metadata_plain_tensors_and_q8_blocks() -> Result<()> {
        let fixture = fixture().map_err(|source| Error::io("create GGUF fixture", source))?;
        let checkpoint = crate::Checkpoint::open(fixture.path())?;
        let metadata = checkpoint
            .gguf_metadata()
            .ok_or_else(|| Error::invalid("GGUF metadata is absent"))?;

        assert_eq!(metadata.version(), 3);
        assert_eq!(metadata.alignment(), 32);
        assert_eq!(checkpoint.format(), crate::CheckpointFormat::Gguf);
        assert_eq!(
            metadata
                .get("general.architecture")
                .and_then(GgufValue::as_str),
            Some("qwen3")
        );
        assert_eq!(
            metadata.get("tokenizer.ggml.token_type"),
            Some(&GgufValue::Array(GgufArray::I32(
                vec![1, 3].into_boxed_slice()
            )))
        );
        assert_eq!(
            checkpoint
                .inventory()
                .tensor("blk.0.attn_q.weight")
                .map(crate::inventory::TensorRecord::shape),
            Some([2, 32].as_slice())
        );

        let TensorData::Quantized(q8) = checkpoint.tensor("blk.0.attn_q.weight")? else {
            return Err(Error::invalid("Q8_0 tensor was not preserved as packed"));
        };
        let Storage::Quantized(storage) = q8.storage() else {
            return Err(Error::invalid("Q8_0 storage descriptor is absent"));
        };
        assert_eq!(
            storage.encoding(),
            &GgmlType::Q8_0
                .encoding()
                .ok_or_else(|| { Error::invalid("Q8_0 encoding identity is absent") })?
        );
        let Packing::Blocks(blocks) = storage.packing() else {
            return Err(Error::invalid("Q8_0 block packing is absent"));
        };
        assert_eq!(blocks.values_per_block(), 32);
        assert_eq!(blocks.bytes_per_block(), 34);
        assert_eq!(blocks.order(), PackingOrder::Axis(1));
        assert_eq!(q8.bytes().as_slice(), &[0x5a; 68]);

        let TensorData::Plain(norm) = checkpoint.tensor("blk.0.attn_norm.weight")? else {
            return Err(Error::invalid("F32 tensor was not kept plain"));
        };
        assert_eq!(norm.bytes().as_slice(), &[0, 0, 0x80, 0x3f, 0, 0, 0, 0x40]);
        Ok(())
    }

    #[test]
    fn inventories_every_current_ggml_storage_type() -> Result<()> {
        let fixture = all_types_fixture()?;
        let checkpoint = crate::Checkpoint::open(fixture.path())?;
        assert_eq!(GgmlType::ALL.len(), CATALOG.len());
        assert_eq!(checkpoint.inventory().len(), CATALOG.len());

        for (expected, ggml_type) in CATALOG.into_iter().zip(GgmlType::ALL) {
            assert_eq!(ggml_type, GgmlType::from_code(expected.code)?);
            assert_eq!(ggml_type.code(), expected.code);
            assert_eq!(ggml_type.name(), expected.name);
            assert_eq!(ggml_type.values_per_block(), expected.values);
            assert_eq!(ggml_type.bytes_per_block(), expected.bytes);
            assert_eq!(ggml_type.dtype(), expected.dtype);

            let name = format!("tensor.{}", expected.code);
            let tensor = checkpoint.tensor(&name)?;
            let marker = u8::try_from(expected.code)
                .map_err(|_error| Error::limit("test GGML code does not fit u8"))?;
            assert_eq!(tensor.shape(), [2, u64::from(expected.values)]);
            assert_eq!(tensor.bytes().len(), 2 * expected.bytes as usize);
            assert!(tensor.bytes().as_slice().iter().all(|byte| *byte == marker));
            match (tensor, expected.dtype) {
                (TensorData::Plain(plain), Some(dtype)) => assert_eq!(plain.dtype(), dtype),
                (TensorData::Quantized(packed), None) => {
                    let Storage::Quantized(descriptor) = packed.storage() else {
                        return Err(Error::invalid("packed GGML descriptor is absent"));
                    };
                    let Packing::Blocks(blocks) = descriptor.packing() else {
                        return Err(Error::invalid("GGML block packing is absent"));
                    };
                    assert_eq!(blocks.values_per_block(), expected.values);
                    assert_eq!(blocks.bytes_per_block(), expected.bytes);
                    assert_eq!(
                        descriptor.encoding(),
                        &ggml_type.encoding().ok_or_else(|| {
                            Error::invalid("packed GGML encoding identity is absent")
                        })?
                    );
                }
                _ => return Err(Error::invalid("GGML scalar/packed classification differs")),
            }
        }
        for removed_or_unknown in [4, 5, 31, 32, 33, 36, 37, 38, 43, u32::MAX] {
            assert_eq!(
                GgmlType::from_code(removed_or_unknown)
                    .expect_err("removed or unknown GGML type must fail")
                    .category(),
                ErrorCategory::Unsupported
            );
        }
        Ok(())
    }

    #[test]
    fn parses_nested_metadata_arrays_and_bounds_their_depth() -> Result<()> {
        let fixture = nested_array_fixture(2)?;
        let checkpoint = crate::Checkpoint::open(fixture.path())?;
        assert_eq!(
            checkpoint
                .gguf_metadata()
                .and_then(|metadata| metadata.get("test.nested")),
            Some(&GgufValue::Array(GgufArray::Array(
                vec![GgufArray::Array(
                    vec![GgufArray::U32(vec![7, 11].into_boxed_slice())].into_boxed_slice()
                )]
                .into_boxed_slice()
            )))
        );

        let too_deep = nested_array_fixture(MAX_METADATA_NESTING_DEPTH + 1)?;
        assert_eq!(
            crate::Checkpoint::open(too_deep.path())
                .expect_err("excessively nested GGUF metadata must fail")
                .category(),
            ErrorCategory::ResourceLimit
        );
        Ok(())
    }

    #[test]
    fn retains_metadata_for_multiple_gguf_sources() -> Result<()> {
        let first = nested_array_fixture(1)?;
        let second = nested_array_fixture(2)?;
        let checkpoint = crate::CheckpointBuilder::from_sources([
            crate::source::SourceDescriptor::local(first.path())?,
            crate::source::SourceDescriptor::local(second.path())?,
        ])
        .open()?;

        assert_eq!(checkpoint.format(), crate::CheckpointFormat::Gguf);
        assert_eq!(checkpoint.gguf_metadata_files().len(), 2);
        assert!(checkpoint.gguf_metadata().is_some());
        assert!(checkpoint.inventory().is_empty());
        Ok(())
    }

    #[test]
    fn rejects_truncated_and_mislabeled_gguf_files() -> Result<()> {
        let mut truncated = NamedTempFile::with_suffix(".gguf")
            .map_err(|source| Error::io("create truncated GGUF fixture", source))?;
        truncated
            .write_all(b"GGUF\x03\0\0")
            .map_err(|source| Error::io("write truncated GGUF fixture", source))?;
        assert_eq!(
            crate::Checkpoint::open(truncated.path())
                .expect_err("truncated GGUF must fail")
                .category(),
            ErrorCategory::InvalidFormat
        );

        let mut mislabeled = NamedTempFile::with_suffix(".gguf")
            .map_err(|source| Error::io("create mislabeled GGUF fixture", source))?;
        mislabeled
            .write_all(&[0_u8; 16])
            .map_err(|source| Error::io("write mislabeled GGUF fixture", source))?;
        assert_eq!(
            crate::Checkpoint::open(mislabeled.path())
                .expect_err("a .gguf file without GGUF magic must fail")
                .category(),
            ErrorCategory::InvalidFormat
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MODEL_WEIGHTS_GGUF to name a local GGUF checkpoint"]
    fn opens_local_gguf_and_preserves_one_packed_payload() -> Result<()> {
        use std::io::{Read as _, Seek as _, SeekFrom};

        let path = std::env::var_os("MODEL_WEIGHTS_GGUF")
            .ok_or_else(|| Error::invalid("MODEL_WEIGHTS_GGUF is not set"))?;
        let checkpoint = crate::Checkpoint::open(&path)?;
        let metadata = checkpoint
            .gguf_metadata()
            .ok_or_else(|| Error::invalid("local GGUF metadata is absent"))?;
        assert_eq!(
            metadata
                .get("general.architecture")
                .and_then(GgufValue::as_str),
            Some("qwen3")
        );
        assert!(
            metadata
                .get("qwen3.block_count")
                .and_then(GgufValue::as_u64)
                .is_some_and(|count| count > 0)
        );
        assert!(!checkpoint.inventory().is_empty());

        let record = checkpoint
            .inventory()
            .iter()
            .filter(|record| matches!(record.storage(), Storage::Quantized(_)))
            .min_by_key(|record| record.storage().span().len())
            .ok_or_else(|| Error::invalid("local GGUF has no packed tensor"))?;
        let loaded = checkpoint.tensor(record.name())?;
        let span = record.storage().span();
        let length = usize::try_from(span.len())
            .map_err(|_error| Error::limit("local GGUF tensor length does not fit usize"))?;
        let mut expected = vec![0_u8; length];
        let mut file = File::open(path)
            .map_err(|source| Error::io("open local GGUF for comparison", source))?;
        file.seek(SeekFrom::Start(span.offset()))
            .and_then(|_| file.read_exact(&mut expected))
            .map_err(|source| Error::io("read local GGUF tensor for comparison", source))?;
        assert_eq!(loaded.bytes().as_slice(), expected);
        Ok(())
    }
}
