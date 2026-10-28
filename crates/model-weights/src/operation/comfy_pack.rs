//! Lossless packing of independently typed compressed payloads and companions.
use super::{
    NodeExecution, OperationNode, RuntimeValue, TensorFacts, allocate_tensor, fresh_node_execution,
    only_output,
};
use crate::comfy::{ComfyFormat, ComfyPack, ComfyRole};
use crate::prepare::Representation;
use crate::tensor::DType;
use crate::{CancellationToken, Error, Result};

const HEADER_WORDS: usize = 80;
const HEADER_BYTES: u64 = (HEADER_WORDS * 8) as u64;
const MAGIC: u64 = 0x3154_4e51_4f4d_4f43;

fn selected_fact(pack: &ComfyPack, role: ComfyRole, fact: &TensorFacts) -> Result<TensorFacts> {
    if pack.row_start == 0 && pack.row_count == pack.rows {
        return Ok(fact.clone());
    }
    let shape = fact.logical_shape();
    let selected = match role {
        ComfyRole::Weight => vec![
            pack.row_count,
            if pack.config.format.wire_id() >= 5 {
                pack.columns / 2
            } else {
                pack.columns
            },
        ],
        ComfyRole::Scale
            if matches!(pack.config.format, ComfyFormat::Mxfp8 | ComfyFormat::Nvfp4) =>
        {
            let block = if pack.config.format == ComfyFormat::Mxfp8 {
                32
            } else {
                16
            };
            vec![
                pack.row_count.div_ceil(128) * 128,
                (pack.columns / block).div_ceil(4) * 4,
            ]
        }
        ComfyRole::Scale | ComfyRole::Center if pack.config.format.wire_id() >= 6 => {
            vec![pack.columns / pack.config.group_size, pack.row_count]
        }
        ComfyRole::Scale if fact.element_count() != 1 => vec![pack.row_count],
        ComfyRole::Up => vec![pack.row_count, shape[1]],
        _ => return Ok(fact.clone()),
    };
    TensorFacts::contiguous(selected, fact.representation().clone())
}

fn swizzled_scale(row: u64, group: u64, groups: u64) -> u64 {
    ((row / 128) * groups.div_ceil(4) + group / 4) * 512
        + (row % 32) * 16
        + ((row % 128) / 32) * 4
        + group % 4
}

#[allow(
    clippy::too_many_lines,
    reason = "Exhaustive byte-preserving layout dispatch over companion roles"
)]
fn copy_selected(
    pack: &ComfyPack,
    role: ComfyRole,
    source: &[u8],
    fact: &TensorFacts,
    output: &mut [u8],
    cancellation: &CancellationToken,
) -> Result<()> {
    let element_bytes = usize::try_from(fact.representation().dtype().byte_len(&[1])?)
        .map_err(|_error| Error::limit("component scalar extent overflow"))?;
    let copy = |source_index: u64, target_index: u64, output: &mut [u8]| -> Result<()> {
        let start = usize::try_from(source_index)
            .ok()
            .and_then(|value| value.checked_mul(element_bytes))
            .ok_or_else(|| Error::limit("component index overflow"))?;
        let target = usize::try_from(target_index)
            .ok()
            .and_then(|value| value.checked_mul(element_bytes))
            .ok_or_else(|| Error::limit("component index overflow"))?;
        output[target..target + element_bytes]
            .copy_from_slice(&source[start..start + element_bytes]);
        Ok(())
    };
    match role {
        ComfyRole::Weight => {
            let columns = if pack.config.format.wire_id() >= 5 {
                pack.columns / 2
            } else {
                pack.columns
            };
            if fact.logical_shape().len() == 4 {
                for row in 0..pack.row_count {
                    cancellation.check()?;
                    let original = row + pack.row_start;
                    for column in 0..columns {
                        let index = (((original / 128) * (pack.columns / 64) + column / 32) * 32
                            + (original % 128) / 4)
                            * 128
                            + (original % 4) * 32
                            + column % 32;
                        copy(index, row * columns + column, output)?;
                    }
                }
            } else {
                let begin = usize::try_from(pack.row_start * columns)
                    .map_err(|_error| Error::limit("selected row offset overflow"))?
                    * element_bytes;
                for (input, target) in source[begin..begin + output.len()]
                    .chunks(1024 * 1024)
                    .zip(output.chunks_mut(1024 * 1024))
                {
                    cancellation.check()?;
                    target.copy_from_slice(input);
                }
            }
        }
        ComfyRole::Scale
            if matches!(pack.config.format, ComfyFormat::Mxfp8 | ComfyFormat::Nvfp4) =>
        {
            let block = if pack.config.format == ComfyFormat::Mxfp8 {
                32
            } else {
                16
            };
            let groups = pack.columns / block;
            for row in 0..pack.row_count {
                cancellation.check()?;
                for group in 0..groups {
                    copy(
                        swizzled_scale(row + pack.row_start, group, groups),
                        swizzled_scale(row, group, groups),
                        output,
                    )?;
                }
            }
        }
        ComfyRole::Scale | ComfyRole::Center if pack.config.format.wire_id() >= 6 => {
            let groups = pack.columns / pack.config.group_size;
            for group in 0..groups {
                cancellation.check()?;
                for row in 0..pack.row_count {
                    let original = row + pack.row_start;
                    let index = if fact.logical_shape().len() == 3 {
                        ((original / 128) * groups + group) * 128 + original % 128
                    } else {
                        group * pack.rows + original
                    };
                    copy(index, group * pack.row_count + row, output)?;
                }
            }
        }
        ComfyRole::Scale if fact.element_count() != 1 => {
            let begin = usize::try_from(pack.row_start)
                .map_err(|_error| Error::limit("selected scale offset overflow"))?
                * element_bytes;
            output.copy_from_slice(&source[begin..begin + output.len()]);
        }
        ComfyRole::Up => {
            let rank = fact.logical_shape()[1];
            for row in 0..pack.row_count {
                cancellation.check()?;
                let original = row + pack.row_start;
                for column in 0..rank {
                    let index = if fact.logical_shape().len() == 3 {
                        ((original / 128) * rank + column) * 128 + original % 128
                    } else {
                        original * rank + column
                    };
                    copy(index, row * rank + column, output)?;
                }
            }
        }
        _ => output.copy_from_slice(source),
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "One exhaustive format table validates all companion shapes and dtypes"
)]
fn validate_components(pack: &ComfyPack, facts: &[&TensorFacts]) -> Result<()> {
    let get = |role| {
        pack.roles
            .iter()
            .position(|value| *value == role)
            .map(|index| facts[index])
    };
    let require =
        |role| get(role).ok_or_else(|| Error::binding("missing compressed weight companion"));
    let is_float = |fact: &TensorFacts| {
        matches!(
            fact.representation().dtype(),
            DType::F16 | DType::Bf16 | DType::F32
        )
    };
    let weight = require(ComfyRole::Weight)?;
    let scale = require(ComfyRole::Scale)?;
    let matrix = [pack.rows, pack.columns];
    let packed = [pack.rows, pack.columns / 2];
    let dtype = weight.representation().dtype();
    let valid_weight = match pack.config.format {
        ComfyFormat::Float8E4m3fn | ComfyFormat::Mxfp8 => {
            dtype == DType::F8E4M3 && weight.logical_shape() == matrix
        }
        ComfyFormat::Float8E5m2 => dtype == DType::F8E5M2 && weight.logical_shape() == matrix,
        ComfyFormat::Int8Tensorwise | ComfyFormat::Int8W8a8 => {
            dtype == DType::I8 && weight.logical_shape() == matrix
        }
        ComfyFormat::Nvfp4 | ComfyFormat::AwqW4a16 => {
            matches!(dtype, DType::I8 | DType::U8) && weight.logical_shape() == packed
        }
        ComfyFormat::SvdquantW4a4 => {
            dtype == DType::I8
                && (weight.logical_shape() == packed
                    || (pack.rows % 128 == 0
                        && pack.columns % 64 == 0
                        && weight.logical_shape() == [pack.rows / 128, pack.columns / 64, 32, 128]))
        }
    };
    if !valid_weight {
        return Err(Error::invalid(
            "Comfy weight dtype or physical shape contradicts its format",
        ));
    }
    if let Some(input) = get(ComfyRole::InputScale)
        && (!is_float(input) || input.element_count() != 1)
    {
        return Err(Error::invalid(
            "calibrated input scale must be a floating scalar",
        ));
    }
    match pack.config.format {
        ComfyFormat::Float8E4m3fn
        | ComfyFormat::Float8E5m2
        | ComfyFormat::Int8Tensorwise
        | ComfyFormat::Int8W8a8 => {
            if !is_float(scale) || ![1, pack.rows].contains(&scale.element_count()) {
                return Err(Error::invalid(
                    "FP8/INT8 scale must be scalar or per output row",
                ));
            }
        }
        ComfyFormat::Mxfp8 | ComfyFormat::Nvfp4 => {
            let block = if pack.config.format == ComfyFormat::Mxfp8 {
                32
            } else {
                16
            };
            let rows = pack
                .rows
                .checked_add(127)
                .map(|value| value / 128 * 128)
                .ok_or_else(|| Error::limit("scale padding overflow"))?;
            let groups = (pack.columns / block)
                .checked_add(3)
                .map(|value| value / 4 * 4)
                .ok_or_else(|| Error::limit("scale padding overflow"))?;
            if scale.element_count()
                != rows
                    .checked_mul(groups)
                    .ok_or_else(|| Error::limit("scale extent overflow"))?
                || (pack.config.format == ComfyFormat::Mxfp8
                    && !matches!(scale.representation().dtype(), DType::U8 | DType::F8E8M0))
                || (pack.config.format == ComfyFormat::Nvfp4
                    && !matches!(scale.representation().dtype(), DType::F8E4M3 | DType::U8))
            {
                return Err(Error::invalid(
                    "microscale dtype or swizzled grid is invalid",
                ));
            }
            if pack.config.format == ComfyFormat::Nvfp4 {
                let global = require(ComfyRole::GlobalScale)?;
                if global.representation().dtype() != DType::F32 || global.element_count() != 1 {
                    return Err(Error::invalid("NVFP4 global scale must be an FP32 scalar"));
                }
            }
        }
        ComfyFormat::AwqW4a16 => {
            let expected = [pack.columns / pack.config.group_size, pack.rows];
            let center = require(ComfyRole::Center)?;
            if !is_float(scale)
                || !is_float(center)
                || scale.logical_shape() != expected
                || center.logical_shape() != expected
            {
                return Err(Error::invalid(
                    "AWQ scales and additive centers must be [K/group, N]",
                ));
            }
        }
        ComfyFormat::SvdquantW4a4 => {
            let groups = pack.columns / 64;
            let smooth = require(ComfyRole::Smooth)?;
            let down = require(ComfyRole::Down)?;
            let up = require(ComfyRole::Up)?;
            if pack.config.group_size != 64
                || !is_float(scale)
                || !is_float(smooth)
                || !is_float(down)
                || !is_float(up)
                || smooth.logical_shape() != [pack.columns]
                || down.logical_shape().len() != 2
                || down.logical_shape()[0] != pack.columns
                || !(scale.logical_shape() == [groups, pack.rows]
                    || scale.logical_shape() == [pack.rows / 128, groups, 128])
            {
                return Err(Error::invalid("invalid SVDQuant companion geometry"));
            }
            let rank = down.logical_shape()[1];
            if !(up.logical_shape() == [pack.rows, rank]
                || up.logical_shape() == [pack.rows / 128, rank, 128])
            {
                return Err(Error::invalid(
                    "SVDQuant up projection has an incompatible rank or output width",
                ));
            }
        }
    }
    Ok(())
}

fn dtype_code(dtype: DType) -> Result<u64> {
    match dtype {
        DType::F32 => Ok(1),
        DType::F16 => Ok(2),
        DType::Bf16 => Ok(3),
        DType::F8E4M3 => Ok(4),
        DType::F8E5M2 => Ok(5),
        DType::F8E8M0 => Ok(6),
        DType::I8 => Ok(7),
        DType::U8 => Ok(8),
        _ => Err(Error::unsupported(
            "unsupported Comfy companion scalar encoding",
        )),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "Wire header construction and format invariants remain adjacent"
)]
fn layout(pack: &ComfyPack, facts: &[&TensorFacts]) -> Result<([u64; HEADER_WORDS], u64)> {
    pack.config.validate()?;
    if pack.rows == 0
        || pack.columns == 0
        || pack.row_count == 0
        || pack
            .row_start
            .checked_add(pack.row_count)
            .is_none_or(|end| end > pack.rows)
        || facts.len() != pack.roles.len()
        || facts.is_empty()
        || pack.config.group_size == 0
        || (pack.config.convrot
            && (pack.config.convrot_groupsize == 0
                || pack.columns % pack.config.convrot_groupsize != 0))
        || pack.roles[0] != ComfyRole::Weight
        || pack
            .roles
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != pack.roles.len()
    {
        return Err(Error::invalid(
            "invalid compressed bundle geometry or component roles",
        ));
    }
    validate_components(pack, facts)?;
    let selected = pack.row_start != 0 || pack.row_count != pack.rows;
    let output_facts = pack
        .roles
        .iter()
        .zip(facts)
        .map(|(role, fact)| selected_fact(pack, *role, fact))
        .collect::<Result<Vec<_>>>()?;
    let mut header = [0; HEADER_WORDS];
    header[0] = MAGIC;
    header[1] = 1;
    header[2] = pack.config.format.wire_id();
    header[3] = u64::from(pack.config.convrot)
        | (u64::from(pack.config.full_precision_matrix_mult) << 1)
        | (u64::from(pack.config.act_unsigned) << 2)
        | (u64::from(pack.config.lowrank_branch_input_basis.as_deref() != Some("smoothed")) << 3)
        | (u64::from(pack.config.proj_down_smooth_folded) << 4);
    header[4] = if selected { pack.row_count } else { pack.rows };
    header[5] = pack.columns;
    header[6] = if selected { 0 } else { pack.row_start };
    header[7] = pack.row_count;
    header[8] = if pack.config.convrot {
        pack.config.convrot_groupsize
    } else {
        pack.config.group_size
    };
    header[11] =
        u64::try_from(facts.len()).map_err(|_error| Error::limit("component count overflow"))?;
    let mut offset = HEADER_BYTES;
    for (role, fact) in pack.roles.iter().zip(&output_facts) {
        if !fact.is_canonical_contiguous() || fact.logical_shape().len() > 4 {
            return Err(Error::unsupported(
                "Comfy bundle requires contiguous tensors of rank at most four",
            ));
        }
        offset = offset
            .checked_add(15)
            .map(|v| v & !15)
            .ok_or_else(|| Error::limit("bundle offset overflow"))?;
        let base = 16 + (*role as usize) * 8;
        header[base] = offset;
        header[base + 1] = fact.byte_len();
        // Older exporters store FP8 microscale bit patterns as uint8. This is
        // a reinterpretation of retained bytes, not a numerical conversion.
        header[base + 2] = if *role == ComfyRole::Scale
            && pack.config.format == ComfyFormat::Nvfp4
            && fact.representation().dtype() == DType::U8
        {
            dtype_code(DType::F8E4M3)?
        } else {
            dtype_code(fact.representation().dtype())?
        };
        header[base + 3] = u64::try_from(fact.logical_shape().len())
            .map_err(|_error| Error::limit("rank overflow"))?;
        for (axis, value) in fact.logical_shape().iter().enumerate() {
            header[base + 4 + axis] = *value;
        }
        offset = offset
            .checked_add(fact.byte_len())
            .ok_or_else(|| Error::limit("bundle extent overflow"))?;
    }
    let scale = pack
        .roles
        .iter()
        .position(|role| *role == ComfyRole::Scale)
        .ok_or_else(|| Error::binding("compressed bundle has no weight scale"))?;
    let scale_count = facts[scale].element_count();
    match pack.config.format {
        ComfyFormat::Float8E4m3fn
        | ComfyFormat::Float8E5m2
        | ComfyFormat::Int8Tensorwise
        | ComfyFormat::Int8W8a8
            if scale_count != 1 && scale_count != pack.rows =>
        {
            return Err(Error::invalid(
                "FP8/INT8 scales must be scalar or per output row",
            ));
        }
        ComfyFormat::Mxfp8 if pack.columns % 32 != 0 => {
            return Err(Error::invalid("MXFP8 K must divide into blocks of 32"));
        }
        ComfyFormat::Nvfp4 if pack.columns % 16 != 0 => {
            return Err(Error::invalid("NVFP4 K must divide into blocks of 16"));
        }
        ComfyFormat::AwqW4a16 | ComfyFormat::SvdquantW4a4
            if pack.columns % pack.config.group_size != 0 =>
        {
            return Err(Error::invalid("INT4 K must divide into scale groups"));
        }
        _ => {}
    }
    if let Some(index) = pack.roles.iter().position(|role| *role == ComfyRole::Down) {
        let shape = facts[index].logical_shape();
        if shape.len() != 2 || shape[0] != pack.columns || shape[1] == 0 {
            return Err(Error::invalid("SVDQuant down projection must be [K, rank]"));
        }
        header[9] = shape[1];
    }
    let row_stride = offset
        .checked_add(pack.row_count - 1)
        .map(|v| v / pack.row_count)
        .and_then(|v| v.checked_add(15))
        .map(|v| v & !15)
        .ok_or_else(|| Error::limit("bundle row stride overflow"))?;
    header[10] = row_stride
        .checked_mul(pack.row_count)
        .ok_or_else(|| Error::limit("bundle size overflow"))?;
    Ok((header, row_stride))
}

pub(super) fn infer_comfy_pack(
    pack: &ComfyPack,
    facts: &[&TensorFacts],
) -> Result<Box<[TensorFacts]>> {
    let (_, stride) = layout(pack, facts)?;
    Ok(vec![TensorFacts::contiguous(
        [pack.row_count, stride],
        Representation::contiguous(DType::I8),
    )?]
    .into_boxed_slice())
}

impl ComfyPack {
    /// Describe the compressed output allocation from raw source facts.
    /// # Errors
    /// Rejects invalid component roles, geometry or overflowing extents.
    pub fn output_facts(&self, inputs: &[&TensorFacts]) -> Result<TensorFacts> {
        Ok(infer_comfy_pack(self, inputs)?[0].clone())
    }
}

pub(super) fn execute_comfy_pack(
    pack: &ComfyPack,
    node: &OperationNode,
    inputs: &[RuntimeValue],
    facts: &[&TensorFacts],
    cancellation: &CancellationToken,
) -> Result<NodeExecution> {
    let (header, _) = layout(pack, facts)?;
    let mut output = allocate_tensor(only_output(node)?, cancellation)?;
    for (word, bytes) in header.iter().zip(output.chunks_exact_mut(8)) {
        bytes.copy_from_slice(&word.to_le_bytes());
    }
    for ((role, input), fact) in pack.roles.iter().zip(inputs).zip(facts) {
        let base = 16 + (*role as usize) * 8;
        let start = usize::try_from(header[base])
            .map_err(|_error| Error::limit("bundle offset exceeds usize"))?;
        let size = usize::try_from(header[base + 1])
            .map_err(|_error| Error::limit("bundle component exceeds usize"))?;
        if input.bytes.as_slice().len() as u64 != fact.byte_len() {
            return Err(Error::integrity("bundle component extent changed"));
        }
        if pack.row_start != 0 || pack.row_count != pack.rows {
            copy_selected(
                pack,
                *role,
                input.bytes.as_slice(),
                fact,
                &mut output[start..start + size],
                cancellation,
            )?;
            continue;
        }
        for (source, target) in input
            .bytes
            .as_slice()
            .chunks(1024 * 1024)
            .zip(output[start..start + size].chunks_mut(1024 * 1024))
        {
            cancellation.check()?;
            target.copy_from_slice(source);
        }
    }
    Ok(fresh_node_execution(output))
}
