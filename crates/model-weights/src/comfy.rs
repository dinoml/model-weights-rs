//! Shared Comfy quantization metadata and lossless compressed bundle layout.
//!
//! Bundling copies encoded bytes and their companions. It never decodes weights.

use serde::{Deserialize, Serialize};

use crate::tensor::DType;
use crate::{Checkpoint, Error, Result};

/// Comfy checkpoint formats, independently of the model using them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyFormat {
    /// OCP E4M3 with explicit scales.
    Float8E4m3fn,
    /// OCP E5M2 with explicit scales.
    Float8E5m2,
    /// Per-row INT8, with optional regular-Hadamard rotation.
    Int8Tensorwise,
    /// Older per-row INT8 marker convention.
    Int8W8a8,
    /// E4M3 values with swizzled E8M0 block scales.
    Mxfp8,
    /// Packed E2M1 with swizzled block scales and a global scale.
    Nvfp4,
    /// Groupwise affine INT4, with floating group centers.
    AwqW4a16,
    /// Signed INT4, smoothing and low-rank correction.
    SvdquantW4a4,
}

impl ComfyFormat {
    /// Stable bundle encoding identifier; distinct from GGML types.
    #[must_use]
    pub const fn wire_id(self) -> u64 {
        match self {
            Self::Float8E4m3fn => 1,
            Self::Float8E5m2 => 2,
            Self::Int8Tensorwise | Self::Int8W8a8 => 3,
            Self::Mxfp8 => 4,
            Self::Nvfp4 => 5,
            Self::AwqW4a16 => 6,
            Self::SvdquantW4a4 => 7,
        }
    }
}

/// Execution metadata retained alongside the original encoded tensors.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Fields mirror the checkpoint's independent execution policies"
)]
pub struct ComfyConfig {
    /// Weight format.
    pub format: ComfyFormat,
    /// Optional source packing layout.
    #[serde(default)]
    pub layout: Option<String>,
    /// Whether the weight uses regular-Hadamard rotation.
    #[serde(default)]
    pub convrot: bool,
    /// Rotation width; each block is independent.
    #[serde(default = "default_rotation")]
    pub convrot_groupsize: u64,
    /// Affine or `SVDQuant` scale-group width.
    #[serde(default = "default_group")]
    pub group_size: u64,
    /// Request high-precision activation multiplication for FP8 storage.
    #[serde(default)]
    pub full_precision_matrix_mult: bool,
    /// Whether runtime `SVDQuant` activations use unsigned codes.
    #[serde(default)]
    pub act_unsigned: bool,
    /// Source basis of the low-rank down projection.
    #[serde(default)]
    pub lowrank_branch_input_basis: Option<String>,
    /// Whether smoothing is already folded into the down projection.
    #[serde(default)]
    pub proj_down_smooth_folded: bool,
    /// GGUF exporters explicitly distinguish rotated and legacy raw weights.
    #[serde(default)]
    pub weight_rotated: Option<bool>,
}

const fn default_rotation() -> u64 {
    256
}
const fn default_group() -> u64 {
    64
}

impl ComfyConfig {
    /// Decode a current or legacy per-layer JSON marker.
    /// # Errors
    /// Rejects unknown formats, invalid rotation widths and unsupported layouts.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| Error::invalid(format!("invalid Comfy quantization JSON: {error}")))?;
        Self::from_metadata(value)
    }

    /// Decode an already parsed layer descriptor without serializing it again.
    /// # Errors
    /// Rejects invalid or unsupported quantization metadata.
    pub fn from_metadata(mut value: serde_json::Value) -> Result<Self> {
        if let Some(object) = value.as_object_mut() {
            if let Some(serde_json::Value::Object(params)) = object.remove("params") {
                for (name, parameter) in params {
                    object.entry(name).or_insert(parameter);
                }
            }
        }
        if value.get("format").is_none()
            && value.get("per_row").and_then(serde_json::Value::as_bool) == Some(true)
        {
            value["format"] = "int8_w8a8".into();
        }
        let mut config: Self = serde_json::from_value(value).map_err(|error| {
            Error::unsupported(format!("unsupported Comfy quantization marker: {error}"))
        })?;
        if config.weight_rotated == Some(false) {
            config.convrot = false;
        }
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.convrot
            && (self.weight_rotated == Some(false)
                || self.convrot_groupsize < 4
                || !self.convrot_groupsize.is_power_of_two()
                || self.convrot_groupsize.trailing_zeros() % 2 != 0)
        {
            return Err(Error::invalid("ConvRot group size must be a power of four"));
        }
        if self.group_size == 0 {
            return Err(Error::invalid("quantization group size must be positive"));
        }
        if self
            .layout
            .as_deref()
            .is_some_and(|layout| layout != "kitchen_tile_packed_w4a4")
        {
            return Err(Error::unsupported(
                "unsupported Comfy quantization packing layout",
            ));
        }
        if self
            .lowrank_branch_input_basis
            .as_deref()
            .is_some_and(|basis| !matches!(basis, "raw" | "smoothed"))
        {
            return Err(Error::unsupported(
                "unsupported SVDQuant low-rank input basis",
            ));
        }
        Ok(())
    }
}

/// Fixed component positions in the shared compressed bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum ComfyRole {
    /// Original encoded weights.
    Weight = 0,
    /// Weight scales.
    Scale = 1,
    /// Optional calibrated input scale.
    InputScale = 2,
    /// NVFP4 global scale.
    GlobalScale = 3,
    /// AWQ additive centers.
    Center = 4,
    /// `SVDQuant` smoothing.
    Smooth = 5,
    /// `SVDQuant` down projection.
    Down = 6,
    /// `SVDQuant` up projection.
    Up = 7,
}

/// A lossless packet preparation operation. Tensor dimensions remain source dimensions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ComfyPack {
    /// Format and execution policy.
    pub config: ComfyConfig,
    /// Original logical output channels.
    pub rows: u64,
    /// Logical input channels.
    pub columns: u64,
    /// First selected logical row.
    pub row_start: u64,
    /// Number of selected logical rows.
    pub row_count: u64,
    /// Roles in operation-input order.
    pub roles: Box<[ComfyRole]>,
}

/// Checked source family for one compressed weight.
#[derive(Debug, Clone)]
pub struct ComfyWeight {
    /// Bundle preparation recipe.
    pub pack: ComfyPack,
    /// Checkpoint tensor names in recipe input order.
    pub sources: Box<[String]>,
}

pub(crate) fn merge_layer_metadata(
    layers: &mut std::collections::BTreeMap<Box<str>, serde_json::Value>,
    json: &str,
) -> Result<()> {
    let mut decoded: serde_json::Value = serde_json::from_str(json).map_err(|error| {
        Error::invalid(format!("invalid quantization header metadata: {error}"))
    })?;
    let source = decoded
        .get_mut("layers")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| Error::invalid("quantization header metadata requires a layer map"))?;
    for (name, value) in std::mem::take(source) {
        if layers
            .get(name.as_str())
            .is_some_and(|existing| existing != &value)
        {
            return Err(Error::invalid(
                "shards disagree on quantization layer metadata",
            ));
        }
        layers.insert(name.into_boxed_str(), value);
    }
    Ok(())
}

impl Checkpoint {
    /// Recognize a Comfy Safetensors marker or custom GGUF layer descriptor.
    /// Only the small marker is read; weight and scale payloads remain unopened.
    /// # Errors
    /// Rejects malformed metadata, unsupported encodings and missing companions.
    #[allow(
        clippy::too_many_lines,
        reason = "Keeps format discovery and its companion key table in one place"
    )]
    pub fn comfy_weight(&self, name: &str) -> Result<Option<ComfyWeight>> {
        let prefix = name.strip_suffix("weight").unwrap_or(name);
        let marker = format!("{prefix}comfy_quant");
        let mut config = None;
        for metadata in self.gguf_metadata_files() {
            if let Some(value) = metadata.get(&format!("comfy.gguf.quant.{name}")) {
                let json = value
                    .as_str()
                    .ok_or_else(|| Error::invalid("GGUF Comfy marker must be JSON text"))?;
                config = Some(ComfyConfig::from_json(json.as_bytes())?);
                break;
            }
        }
        if config.is_none() {
            if let Some(record) = self.inventory().tensor(&marker) {
                if record.storage().logical_dtype() != Some(DType::U8)
                    || record.storage().span().len() > 65_536
                {
                    return Err(Error::invalid(
                        "Comfy marker must be a bounded uint8 JSON tensor",
                    ));
                }
                config = Some(ComfyConfig::from_json(
                    self.tensor(&marker)?.bytes().as_slice(),
                )?);
            }
        }
        if config.is_none() {
            if let Some(value) = self.comfy_layer_metadata(prefix.trim_end_matches('.')) {
                config = Some(ComfyConfig::from_metadata(value.clone())?);
            }
        }
        let mut legacy = false;
        if config.is_none()
            && self
                .inventory()
                .tensor(&format!("{prefix}scale_weight"))
                .is_some()
        {
            let legacy_marker = std::iter::once("scaled_fp8".to_owned())
                .chain(
                    name.match_indices('.')
                        .map(|(offset, _)| format!("{}scaled_fp8", &name[..=offset])),
                )
                .filter_map(|key| self.inventory().tensor(&key))
                .next_back();
            if let Some(marker) = legacy_marker {
                let record = self
                    .inventory()
                    .tensor(name)
                    .ok_or_else(|| Error::binding("missing legacy FP8 weight"))?;
                let format = match record.storage().logical_dtype() {
                    Some(DType::F8E4M3) => "float8_e4m3fn",
                    Some(DType::F8E5M2) => "float8_e5m2",
                    _ => {
                        return Err(Error::invalid(
                            "legacy scaled FP8 weight has a different scalar dtype",
                        ));
                    }
                };
                config = Some(ComfyConfig::from_metadata(serde_json::json!({
                    "format": format,
                    "full_precision_matrix_mult": marker.shape().iter().product::<u64>() == 2,
                }))?);
                legacy = true;
            }
        }
        let Some(config) = config else {
            return Ok(None);
        };
        let weight = self
            .inventory()
            .tensor(name)
            .ok_or_else(|| Error::binding("missing Comfy weight"))?;
        let shape = weight.shape();
        let (rows, columns) = match (config.format, shape) {
            (ComfyFormat::SvdquantW4a4, [n, k, 32, 128]) => (n.checked_mul(128), k.checked_mul(64)),
            (ComfyFormat::Nvfp4 | ComfyFormat::AwqW4a16 | ComfyFormat::SvdquantW4a4, [n, k]) => {
                (Some(*n), k.checked_mul(2))
            }
            (_, [n, k]) => (Some(*n), Some(*k)),
            _ => return Err(Error::invalid("invalid Comfy weight rank or tile geometry")),
        };
        let (rows, columns) = rows
            .zip(columns)
            .filter(|(n, k)| *n > 0 && *k > 0)
            .ok_or_else(|| Error::limit("Comfy logical shape is empty or overflows"))?;
        if config.convrot && columns % config.convrot_groupsize != 0 {
            return Err(Error::invalid(
                "ConvRot group size does not divide the input channels",
            ));
        }
        let mut roles = vec![ComfyRole::Weight];
        let mut sources = vec![name.to_owned()];
        for (role, suffix, required) in [
            (ComfyRole::Scale, "weight_scale", true),
            (ComfyRole::InputScale, "input_scale", false),
            (
                ComfyRole::GlobalScale,
                "weight_scale_2",
                config.format == ComfyFormat::Nvfp4,
            ),
            (
                ComfyRole::Center,
                "weight_zero",
                config.format == ComfyFormat::AwqW4a16,
            ),
            (
                ComfyRole::Smooth,
                "smooth_factor",
                config.format == ComfyFormat::SvdquantW4a4,
            ),
            (
                ComfyRole::Down,
                "proj_down",
                config.format == ComfyFormat::SvdquantW4a4,
            ),
            (
                ComfyRole::Up,
                "proj_up",
                config.format == ComfyFormat::SvdquantW4a4,
            ),
        ] {
            let suffix = if legacy && role == ComfyRole::Scale {
                "scale_weight"
            } else if legacy && role == ComfyRole::InputScale {
                "scale_input"
            } else {
                suffix
            };
            let source = format!("{prefix}{suffix}");
            if legacy && role == ComfyRole::InputScale {
                if let Some(record) = self.inventory().tensor(&source) {
                    if record.storage().logical_dtype() == Some(DType::F32)
                        && record.storage().span().len() == 4
                    {
                        let value = self.tensor(&source)?;
                        if value.bytes().as_slice() == 1.0f32.to_le_bytes() {
                            continue;
                        }
                    }
                }
            }
            if self.inventory().tensor(&source).is_some() {
                roles.push(role);
                sources.push(source);
            } else if required {
                return Err(Error::binding(format!("missing Comfy companion {source}")));
            }
        }
        Ok(Some(ComfyWeight {
            pack: ComfyPack {
                config,
                rows,
                columns,
                row_start: 0,
                row_count: rows,
                roles: roles.into_boxed_slice(),
            },
            sources: sources.into_boxed_slice(),
        }))
    }
}
