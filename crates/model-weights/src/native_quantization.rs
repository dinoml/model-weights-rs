//! Bounded normalization of libgguf-native Safetensors carrier metadata.
//!
//! This module describes storage and does not encode, decode, or select a kernel.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::identity::{ImplementationId, StableName};
use crate::inventory::TensorRecord;
use crate::limits::ResourceLimits;
use crate::quantization::{
    BlockPacking, Grouping, MetadataValue, Packing, PackingOrder, QuantizedStorage, Storage,
};
use crate::tensor::DType;
use crate::{CancellationToken, Error, Result};

/// Safetensors metadata key containing the revision-one native descriptor map.
pub const METADATA_KEY: &str = "libgguf.quantization.v1";

/// One architecture-neutral libgguf-native numerical format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeFormat {
    /// Signed four-bit values, one FP16 scale per 32 values.
    S4G32,
    /// Signed four-bit values, one FP16 scale per 64 values.
    S4G64,
    /// Signed eight-bit values, one FP16 scale per 32 values.
    S8G32,
    /// Signed four-bit values with byte subscales in a 256-value superblock.
    S4H256,
}

impl NativeFormat {
    /// Parses a namespaced format spelling.
    ///
    /// # Errors
    /// Returns an unsupported error for an unknown numerical format.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "libgguf.s4g32" => Ok(Self::S4G32),
            "libgguf.s4g64" => Ok(Self::S4G64),
            "libgguf.s8g32" => Ok(Self::S8G32),
            "libgguf.s4h256" => Ok(Self::S4H256),
            _ => Err(Error::unsupported("unknown libgguf-native format")),
        }
    }

    /// Returns the canonical format spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::S4G32 => "libgguf.s4g32",
            Self::S4G64 => "libgguf.s4g64",
            Self::S8G32 => "libgguf.s8g32",
            Self::S4H256 => "libgguf.s4h256",
        }
    }

    /// Returns decoded block values, encoded block bytes, and scale group values.
    #[must_use]
    pub const fn geometry(self) -> (u32, u32, u32) {
        match self {
            Self::S4G32 => (32, 18, 32),
            Self::S4G64 => (64, 34, 64),
            Self::S8G32 => (32, 34, 32),
            Self::S4H256 => (256, 138, 32),
        }
    }

    /// Returns its revision-one normalized canonical-row encoding identity.
    ///
    /// # Errors
    /// Returns an invalid-name error if built-in identity constants are invalid.
    pub fn encoding(self) -> Result<ImplementationId> {
        Ok(ImplementationId::new(
            StableName::parse("libgguf")?,
            StableName::parse(format!("{}.canonical_rows", self.name()))?,
            1,
        ))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DescriptorMap {
    schema_version: u32,
    #[serde(deserialize_with = "unique_tensors")]
    tensors: BTreeMap<Box<str>, Descriptor>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    format: Box<str>,
    format_revision: u32,
    encoding: Box<str>,
    encoding_revision: u32,
    logical_shape: [u64; 2],
    payload: Box<str>,
    // Provenance is opaque: unknown encoders do not change decoding admission.
    // Retain it in normalized source/cache identities without interpreting it.
    #[serde(default, rename = "encoder")]
    encoder: Option<Box<serde_json::value::RawValue>>,
}

fn unique_tensors<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<BTreeMap<Box<str>, Descriptor>, D::Error> {
    struct Unique;
    impl<'de> Visitor<'de> for Unique {
        type Value = BTreeMap<Box<str>, Descriptor>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a native tensor descriptor map without duplicate names")
        }
        fn visit_map<A: MapAccess<'de>>(
            self,
            mut input: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut result = BTreeMap::new();
            while let Some((name, descriptor)) = input.next_entry::<Box<str>, Descriptor>()? {
                if result.insert(name, descriptor).is_some() {
                    return Err(serde::de::Error::custom("duplicate native descriptor name"));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Unique)
}

pub(crate) fn normalize(
    metadata: &str,
    comfy_metadata: Option<&str>,
    mut records: Vec<TensorRecord>,
    limits: &ResourceLimits,
    cancellation: &CancellationToken,
) -> Result<Vec<TensorRecord>> {
    cancellation.check()?;
    if metadata.len() > limits.max_metadata_bytes() {
        return Err(Error::limit(
            "native quantization metadata exceeds configured limits",
        ));
    }
    let descriptors: DescriptorMap = serde_json::from_str(metadata).map_err(|error| {
        Error::invalid(format!("invalid native quantization metadata: {error}"))
    })?;
    if descriptors.schema_version != 1 {
        return Err(Error::unsupported(
            "unknown native quantization metadata revision",
        ));
    }
    if descriptors.tensors.len() > limits.max_tensors() {
        return Err(Error::limit("too many native quantization descriptors"));
    }
    let comfy: Option<serde_json::Value> = comfy_metadata
        .map(serde_json::from_str)
        .transpose()
        .map_err(|_error| Error::invalid("invalid Comfy quantization metadata"))?;
    let mut by_name: BTreeMap<Box<str>, usize> = records
        .iter()
        .enumerate()
        .map(|(index, tensor)| (tensor.name().into(), index))
        .collect();
    for (name, descriptor) in descriptors.tensors {
        cancellation.check()?;
        if name.is_empty() || name.len() > limits.max_name_bytes() || name != descriptor.payload {
            return Err(Error::invalid(
                "native descriptor must reference its own bounded carrier name",
            ));
        }
        let prefix = name.strip_suffix(".weight").unwrap_or(&name);
        if comfy
            .as_ref()
            .and_then(|value| value.get("layers"))
            .and_then(|value| value.get(prefix))
            .is_some()
            || by_name.contains_key(format!("{prefix}.comfy_quant").as_str())
        {
            return Err(Error::invalid(
                "conflicting native and Comfy quantization authorities",
            ));
        }
        if descriptor.format_revision != 1
            || descriptor.encoding_revision != 1
            || descriptor.encoding.as_ref() != "canonical_rows"
        {
            return Err(Error::unsupported(
                "unknown native format or row encoding revision",
            ));
        }
        let kind = NativeFormat::parse(&descriptor.format)?;
        let (values, block_bytes, group) = kind.geometry();
        let [n, k] = descriptor.logical_shape;
        if n == 0 || k == 0 || !k.is_multiple_of(u64::from(values)) {
            return Err(Error::invalid(
                "native logical matrix must contain complete row blocks",
            ));
        }
        let row_bytes = (k / u64::from(values))
            .checked_mul(u64::from(block_bytes))
            .ok_or_else(|| Error::limit("native row byte count overflows"))?;
        let index = by_name
            .remove(name.as_ref())
            .ok_or_else(|| Error::invalid("native carrier is missing from its shard"))?;
        let record = &records[index];
        let Storage::Plain {
            dtype: DType::U8,
            span,
        } = record.storage()
        else {
            return Err(Error::invalid("native carrier must have plain U8 storage"));
        };
        if record.shape() != [n, row_bytes] {
            return Err(Error::invalid(
                "native carrier shape differs from its block geometry",
            ));
        }
        let storage = QuantizedStorage::builder(
            kind.encoding()?,
            vec![n, k],
            *span,
            Packing::Blocks(BlockPacking::new(
                values,
                block_bytes,
                PackingOrder::Axis(1),
            )?),
        )
        .grouping(Grouping::new(1, u64::from(group))?)
        .metadata(encoder_provenance(descriptor.encoder.as_deref())?)
        .build()?;
        records[index] = TensorRecord::new(name, Box::new([n, k]), Storage::Quantized(storage));
    }
    cancellation.check()?;
    Ok(records)
}

fn encoder_provenance(
    encoder: Option<&serde_json::value::RawValue>,
) -> Result<BTreeMap<StableName, MetadataValue>> {
    let mut provenance = BTreeMap::new();
    if let Some(encoder) = encoder {
        provenance.insert(
            StableName::parse("libgguf.encoder")?,
            MetadataValue::Text(encoder.get().into()),
        );
    }
    Ok(provenance)
}
